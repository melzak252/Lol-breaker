"""Causal thirty-played-day player and role-meta features.

``build_meta_features(observations, targets, champion_ids=...)`` accepts mappings.
An observation is one map-player record with ``series_id, played_day,
release_day, competition, patch, player_id, role, champion_id, values12,
observed12``. Release is the corrected all-series release clock supplied by the
caller; effective release is max(played_day, release_day). Targets supply
``row_id, cutoff_day, competition, roster_a, roster_b``; each roster is five
player IDs in caller order, not an assertion of their future map roles. Dates
may be ISO days, date/datetime, or pandas Timestamp (aware clocks use UTC).

Only played days in [cutoff-30d, cutoff) with effective release < cutoff enter
features. Champion vocabulary is supplied in fixed order, excluding bin zero;
no observed or target champion can change tensor widths. None/blank champions
are absent; explicit unknown or out-of-vocabulary champions occupy bin zero.
Unknown historical roles contribute to player summaries and the role-mixture
missing mass, but not to known-role meta pools. Patches are explicit source
strings only: never inferred from calendar dates, series IDs, or target data.

Meta axis 1 is global then exact-competition pooled. Local raw observation
weight is n/(n+100). Means and second moments use each stat's observed count
with 100 global-observed-stat pseudo-observations; champion distributions use
present-champion counts analogously. Coverage fractions use the raw weight;
log count and meta_support retain local raw support, not pseudo-support.
meta_patch retains raw historical distinct patch count and maximum share among
known patches. Empty local context can therefore borrow global distributions
without claiming local support or patch observations.

The result exposes float32 feature arrays (player_valid bool), original target
row order, compact per-row diagnostics, and one normalized source-reference
catalog. Source indices index the original observation sequence, so independent
replay can join them to its pinned raw records. Diagnostics share same-cutoff
membership lists rather than duplicating full source records. Optional target
``patch`` is read only while assembling diagnostics, never for feature inputs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import heapq
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROLES = ("top", "jungle", "mid", "adc", "support")
STATS = ("kills", "deaths", "assists", "csm", "gpm", "dpm", "vspm",
         "gd@15", "csd@15", "xpd@15", "kp%", "dmg%")
SUMMARY_WIDTH = 40
WINDOW_DAYS = 30
POOL_PRIOR = 100.0
_UNKNOWN_LABELS = frozenset(("unknown", "none", "null", "nan", "<na>", "n/a", "?"))


@dataclass(frozen=True)
class MetaFeatureBatch:
    arrays: dict[str, np.ndarray]
    row_ids: tuple[Any, ...]
    diagnostics: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]


def _day(value: Any, name: str) -> date:
    if value is None or pd.isna(value):
        raise ValueError(f"Missing {name}")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid {name}: {value!r}") from error
    raise ValueError(f"Invalid {name}: {value!r}")


def _identity(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _patch(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return None if not text or text.lower() in _UNKNOWN_LABELS else text


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class _Observation:
    index: int
    series: str
    played: date
    release: date
    effective: date
    competition: str | None
    patch: str | None
    player: str | None
    role: int | None
    champion: int | None
    values: np.ndarray
    observed: np.ndarray
    digest: str


def _observations(records: Iterable[Mapping[str, Any]], bins: dict[str, int]):
    normalized = []
    sources = []
    for index, record in enumerate(records):
        series = _identity(record["series_id"])
        if series is None:
            raise ValueError("Missing observation series_id")
        played = _day(record["played_day"], "played_day")
        release = _day(record["release_day"], "release_day")
        effective = max(played, release)
        competition = _identity(record.get("competition"))
        player = _identity(record.get("player_id"))
        role_text = _identity(record.get("role"))
        role_text = role_text.lower() if role_text else None
        role = ROLES.index(role_text) if role_text in ROLES else None
        champion_id = _identity(record.get("champion_id"))
        champion = bins.get(champion_id, 0) if champion_id is not None else None
        patch = _patch(record.get("patch"))
        values = np.asarray(record["values12"], dtype=np.float64)
        observed = np.asarray(record["observed12"], dtype=np.bool_)
        if values.shape != (12,) or observed.shape != (12,):
            raise ValueError("Each observation requires 12 stat values and 12 observed flags")
        if not np.isfinite(values[observed]).all():
            raise ValueError("Observed stat values must be finite")
        values = np.where(observed, values, 0.0)
        ref = dict(source_index=index, series_id=series, played_day=played.isoformat(),
                   release_day=release.isoformat(), effective_release_day=effective.isoformat(),
                   competition=competition, player_id=player, role=ROLES[role] if role is not None else None,
                   source_role=role_text, champion_id=champion_id, champion_bin=champion, patch=patch)
        digest = _digest({key: value for key, value in ref.items() if key != "source_index"} |
                         {"values12": values.tolist(), "observed12": observed.tolist()})
        ref["source_sha256"] = digest
        sources.append(ref)
        normalized.append(_Observation(index, series, played, release, effective, competition,
                                       patch, player, role, champion, values, observed, digest))
    return normalized, tuple(sources)


class _Moments:
    """Removable sufficient statistics for one available-window population."""

    def __init__(self, width: int):
        self.n = 0
        self.count = np.zeros(12, dtype=np.int64)
        self.total = np.zeros(12, dtype=np.float64)
        self.square = np.zeros(12, dtype=np.float64)
        self.champions = np.zeros(width, dtype=np.int64)
        self.roles = np.zeros(5, dtype=np.int64)
        self.patches: Counter[str] = Counter()
        self.members: set[int] = set()

    def update(self, record: _Observation, sign: int):
        self.n += sign
        self.count += sign * record.observed
        self.total += sign * record.values
        self.square += sign * np.square(record.values)
        if record.champion is not None:
            self.champions[record.champion] += sign
        if record.role is not None:
            self.roles[record.role] += sign
        if record.patch is not None:
            self.patches[record.patch] += sign
            if not self.patches[record.patch]:
                del self.patches[record.patch]
        if sign > 0:
            self.members.add(record.index)
        else:
            self.members.remove(record.index)
        # Reset an emptied accumulator to remove cancellation residue.
        if not self.n:
            self.total.fill(0)
            self.square.fill(0)

    def moments(self):
        mean = np.divide(self.total, self.count, out=np.zeros(12), where=self.count > 0)
        second = np.divide(self.square, self.count, out=np.zeros(12), where=self.count > 0)
        return mean, second

    def summary(self):
        result = np.zeros(SUMMARY_WIDTH, dtype=np.float64)
        if self.n:
            mean, second = self.moments()
            result[:12] = mean
            result[12:24] = np.sqrt(np.maximum(0, second - np.square(mean)))
            result[24:36] = self.count / self.n
            result[36:] = (np.log1p(self.n), self.champions.sum() / self.n,
                           self.roles.sum() / self.n, sum(self.patches.values()) / self.n)
        return result

    def champion_distribution(self):
        count = self.champions.sum()
        return self.champions / count if count else np.zeros(len(self.champions))

    def patch_summary(self):
        known = sum(self.patches.values())
        return (len(self.patches), max(self.patches.values()) / known if known else 0.0)


def _pooled(local: _Moments, global_: _Moments):
    if not global_.n:
        return np.zeros(SUMMARY_WIDTH), np.zeros(len(global_.champions)), 0.0
    raw_weight = local.n / (local.n + POOL_PRIOR)
    global_summary = global_.summary()
    local_summary = local.summary()
    result = raw_weight * local_summary + (1 - raw_weight) * global_summary
    local_mean, local_second = local.moments()
    global_mean, global_second = global_.moments()
    stat_weight = local.count / (local.count + POOL_PRIOR)
    mean = stat_weight * local_mean + (1 - stat_weight) * global_mean
    second = stat_weight * local_second + (1 - stat_weight) * global_second
    result[:12] = mean
    result[12:24] = np.sqrt(np.maximum(0, second - np.square(mean)))
    result[36] = np.log1p(local.n)
    present = local.champions.sum()
    champ_weight = present / (present + POOL_PRIOR)
    champions = (champ_weight * local.champion_distribution() +
                 (1 - champ_weight) * global_.champion_distribution())
    return result, champions, raw_weight


class _Window:
    def __init__(self, records: list[_Observation], width: int):
        # Canonical ordering makes accumulation independent of input iteration order.
        self.pending = sorted(records, key=lambda r: (r.effective, r.played, r.series, r.digest))
        self.cursor = 0
        self.width = width
        self.expiry: list[tuple[date, str, int]] = []
        self.active: dict[int, _Observation] = {}
        self.all = _Moments(width)
        self.global_roles = [_Moments(width) for _ in ROLES]
        self.local_roles: dict[tuple[str, int], _Moments] = {}
        self.players: dict[str, _Moments] = {}

    def update(self, record: _Observation, sign: int):
        self.all.update(record, sign)
        if record.player is not None:
            if record.player not in self.players:
                self.players[record.player] = _Moments(self.width)
            self.players[record.player].update(record, sign)
            if not self.players[record.player].n:
                del self.players[record.player]
        if record.role is not None:
            self.global_roles[record.role].update(record, sign)
            if record.competition is not None:
                key = (record.competition, record.role)
                if key not in self.local_roles:
                    self.local_roles[key] = _Moments(self.width)
                self.local_roles[key].update(record, sign)
                if not self.local_roles[key].n:
                    del self.local_roles[key]

    def advance(self, cutoff: date):
        lower = cutoff - timedelta(days=WINDOW_DAYS)
        while self.expiry and self.expiry[0][0] < lower:
            _, _, index = heapq.heappop(self.expiry)
            self.update(self.active.pop(index), -1)
        while self.cursor < len(self.pending) and self.pending[self.cursor].effective < cutoff:
            record = self.pending[self.cursor]
            self.cursor += 1
            # A late release does not rejuvenate the original played day.
            if record.played < lower:
                continue
            self.active[record.index] = record
            heapq.heappush(self.expiry, (record.played, record.digest, record.index))
            self.update(record, 1)

    def provenance(self, cutoff: date):
        indices = sorted(self.active)
        active = [self.active[index] for index in indices]
        known = sum(self.all.patches.values())
        latest = max((r.played for r in active if r.patch is not None), default=None)
        return dict(
            cutoff_day=cutoff.isoformat(), window_start_day=(cutoff - timedelta(days=WINDOW_DAYS)).isoformat(),
            source_indices=indices, source_count=len(indices),
            source_membership_sha256=_digest(sorted(r.digest for r in active)),
            min_played_day=min((r.played.isoformat() for r in active), default=None),
            max_played_day=max((r.played.isoformat() for r in active), default=None),
            max_release_day=max((r.release.isoformat() for r in active), default=None),
            max_effective_release_day=max((r.effective.isoformat() for r in active), default=None),
            global_role_source_indices=[sorted(pool.members) for pool in self.global_roles],
            unknown_role_source_indices=[r.index for r in active if r.role is None],
            patch_coverage=dict(observations=self.all.n, known_observations=known,
                                unknown_observations=self.all.n - known,
                                known_fraction=known / self.all.n if self.all.n else 0.0,
                                labels=dict(sorted(self.all.patches.items()))),
            latest_patch_played_day=latest.isoformat() if latest else None,
            latest_patch_labels=sorted({r.patch for r in active if r.patch is not None and r.played == latest}),
        )


def build_meta_features(
    observations: Iterable[Mapping[str, Any]],
    targets: Iterable[Mapping[str, Any]],
    *,
    champion_ids: Sequence[Any],
) -> MetaFeatureBatch:
    """Build fixed-width causal arrays; do not fit, scale, or acquire any data.

    All target rows are retained, including cold rosters and empty histories.
    Input iterables are consumed once. Targets are replayed by daily cutoff and
    restored to their original order. Duplicate row IDs are allowed: row order,
    not a lossy ID dictionary, aligns features. Feature memberships are reported
    separately for global roles, exact local roles, and each roster player.
    """
    vocabulary = tuple(_identity(value) for value in champion_ids)
    if any(value is None or value == "0" or value.lower() in _UNKNOWN_LABELS for value in vocabulary):
        raise ValueError("champion_ids must exclude absent and unknown champion bins")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("champion_ids must contain unique champion IDs")
    bins = {value: index + 1 for index, value in enumerate(vocabulary)}
    width = len(vocabulary) + 1
    records, sources = _observations(observations, bins)
    rows = list(targets)
    cutoffs = [_day(row["cutoff_day"], "cutoff_day") for row in rows]
    rosters = []
    competitions = []
    for row in rows:
        sides = []
        for side in ("roster_a", "roster_b"):
            roster = tuple(_identity(value) for value in row[side])
            if len(roster) != 5:
                raise ValueError(f"{side} must contain five player IDs")
            sides.append(roster)
        rosters.append(sides)
        competitions.append(_identity(row.get("competition")))
    n = len(rows)
    arrays = {name: np.zeros((n, 2, 5, size), dtype=np.float32) for name, size in (
        ("player_stats", SUMMARY_WIDTH), ("player_roles", 5), ("player_champions", width),
        ("meta_stats", SUMMARY_WIDTH), ("meta_champions", width), ("meta_support", 2), ("meta_patch", 2))}
    arrays["player_valid"] = np.zeros((n, 2, 5), dtype=np.bool_)
    diagnostics: list[dict[str, Any]] = [{} for _ in rows]
    state = _Window(records, width)
    empty = _Moments(width)
    current_day = None
    shared: dict[str, Any] = {}
    context_cache: dict[str | None, tuple[dict[str, np.ndarray], list[list[int]]]] = {}
    player_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = {}
    for index in sorted(range(n), key=lambda position: (cutoffs[position], position)):
        cutoff = cutoffs[index]
        if cutoff != current_day:
            state.advance(cutoff)
            current_day = cutoff
            shared = state.provenance(cutoff)
            context_cache.clear()
            player_cache.clear()
        competition = competitions[index]
        if competition not in context_cache:
            context = {name: np.zeros((2, 5, size), dtype=np.float32) for name, size in (
                ("meta_stats", SUMMARY_WIDTH), ("meta_champions", width), ("meta_support", 2), ("meta_patch", 2))}
            local_members = []
            for role in range(5):
                global_ = state.global_roles[role]
                local = state.local_roles.get((competition, role), empty) if competition is not None else empty
                context["meta_stats"][0, role] = global_.summary()
                context["meta_champions"][0, role] = global_.champion_distribution()
                context["meta_support"][0, role] = (global_.n, float(global_.n > 0))
                context["meta_patch"][0, role] = global_.patch_summary()
                summary, champions, weight = _pooled(local, global_)
                context["meta_stats"][1, role] = summary
                context["meta_champions"][1, role] = champions
                context["meta_support"][1, role] = (local.n, weight)
                context["meta_patch"][1, role] = local.patch_summary()
                local_members.append(sorted(local.members))
            context_cache[competition] = (context, local_members)
        context, local_members = context_cache[competition]
        for name, values in context.items():
            arrays[name][index] = values
        player_members = [[[] for _ in range(5)] for _ in range(2)]
        for side, roster in enumerate(rosters[index]):
            for slot, player in enumerate(roster):
                pool = state.players.get(player) if player is not None else None
                if pool is None:
                    continue
                if player not in player_cache:
                    player_cache[player] = (pool.summary(), pool.roles / pool.n,
                                            pool.champion_distribution(), sorted(pool.members))
                summary, roles, champions, members = player_cache[player]
                arrays["player_stats"][index, side, slot] = summary
                arrays["player_roles"][index, side, slot] = roles
                arrays["player_champions"][index, side, slot] = champions
                arrays["player_valid"][index, side, slot] = True
                player_members[side][slot] = members
        target_patch = _patch(rows[index].get("patch"))
        latest_labels = shared["latest_patch_labels"]
        diagnostics[index] = dict(
            shared, row_id=rows[index]["row_id"], competition=competition,
            local_role_source_indices=local_members, player_source_indices=player_members,
            target_patch=target_patch,
            patch_transition=dict(latest_played_day=shared["latest_patch_played_day"],
                                  latest_labels=latest_labels,
                                  target_patch_seen=(target_patch in shared["patch_coverage"]["labels"]
                                                     if target_patch is not None else None),
                                  target_differs_from_latest=(target_patch not in latest_labels
                                                             if target_patch is not None and latest_labels else None)),
        )
    return MetaFeatureBatch(arrays, tuple(row["row_id"] for row in rows), tuple(diagnostics), sources)
