"""Fixed 180-played-day player features with the original 30-day role meta.

Activity uses every released raw player-map record, including history older than
180 days. Release only determines availability: inactivity is measured from the
latest *played* day, never from the release day. Both windows are left-inclusive
and right-exclusive; effective release must be strictly before the cutoff.

The result mirrors ``MetaFeatureBatch``, with an explicit compact ``sources``
mapping keyed by ORIGINAL observation position, not a dense tuple. Full original
normalized references are retained for records eligible for player/meta moments;
older activity-only evidence references contain source_index, series_id,
player_id, played_day, release_day, and effective_release_day. Every index in the
diagnostics can be looked up as ``batch.sources[index]``. Records irrelevant to
all requested cutoffs are omitted. Old history is streamed into cutoff/player
aggregates without allocating stat arrays, hashes, or a full historical catalog.

``player_activity`` channels are log1p(recent30 count), log1p(long180 count),
log1p(days since latest prior played day), identity_history_known,
recent_supported, long_supported, native_supported, identity_known. Unknown
inactivity is zero accompanied by identity_history_known=0. Native-only roles
are caller-supplied prior-token frequencies; their unknown mass is not imputed.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import date, timedelta
import heapq
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.features.c0_meta_features import (
    MetaFeatureBatch, _Moments, _Observation, _UNKNOWN_LABELS, _day, _identity,
    _observations, build_meta_features,
)

PLAYER_WINDOW_DAYS = 180
ACTIVITY_WIDTH = 8

@dataclass(frozen=True)
class StableMetaFeatureBatch:
    arrays: dict[str, np.ndarray]
    row_ids: tuple[Any, ...]
    diagnostics: tuple[dict[str, Any], ...]
    sources: dict[int, dict[str, Any]]



class _Prior:
    """Constant-size all-prior identity/count and independent clock evidence."""

    def __init__(self):
        self.count = 0
        self.latest: dict[str, Any] | None = None
        self.effective: dict[str, Any] | None = None

    @staticmethod
    def _played_key(ref):
        return (ref["played_day"], ref["effective_release_day"], ref["series_id"])

    @staticmethod
    def _effective_key(ref):
        return (ref["effective_release_day"], ref["played_day"], ref["series_id"])

    def add(self, ref):
        self.count += 1
        if self.latest is None or self._played_key(ref) > self._played_key(self.latest):
            self.latest = ref
        if self.effective is None or self._effective_key(ref) > self._effective_key(self.effective):
            self.effective = ref

    def merge(self, other):
        self.count += other.count
        if self.latest is None or self._played_key(other.latest) > self._played_key(self.latest):
            self.latest = other.latest
        if self.effective is None or self._effective_key(other.effective) > self._effective_key(self.effective):
            self.effective = other.effective

    def diagnostics(self):
        return dict(
            prior_source_count=self.count,
            latest_played_day=self.latest["played_day"] if self.latest else None,
            max_effective_release_day=self.effective["effective_release_day"] if self.effective else None,
            latest_played_source_index=self.latest["source_index"] if self.latest else None,
            max_effective_release_source_index=self.effective["source_index"] if self.effective else None,
        )


class _PlayerWindow:
    """Removable player moments only: no redundant long global/local pools."""

    def __init__(self, records: list[_Observation], width: int, players: set[str]):
        self.pending = sorted((record for record in records if record.player in players),
                              key=lambda record: (record.effective, record.played, record.series, record.digest))
        self.cursor = 0
        self.width = width
        self.expiry: list[tuple[date, str, int, _Observation]] = []
        self.players: dict[str, _Moments] = {}

    def advance(self, cutoff: date):
        lower = cutoff - timedelta(days=PLAYER_WINDOW_DAYS)
        while self.expiry and self.expiry[0][0] < lower:
            _, _, _, record = heapq.heappop(self.expiry)
            pool = self.players[record.player]
            pool.update(record, -1)
            if not pool.n:
                del self.players[record.player]
        while self.cursor < len(self.pending) and self.pending[self.cursor].effective < cutoff:
            record = self.pending[self.cursor]
            self.cursor += 1
            if record.played < lower:
                continue
            pool = self.players.get(record.player)
            if pool is None:
                pool = self.players[record.player] = _Moments(self.width)
            pool.update(record, 1)
            heapq.heappush(self.expiry, (record.played, record.digest, record.index, record))


def _native_inputs(native_valid, native_roles, n):
    if not isinstance(native_valid, np.ndarray) or native_valid.dtype != np.bool_:
        raise TypeError("native_valid must be a boolean numpy array")
    if native_valid.shape != (n, 2, 5):
        raise ValueError("native_valid must have shape [N,2,5]")
    if not isinstance(native_roles, np.ndarray) or native_roles.dtype != np.float32:
        raise TypeError("native_roles must be a float32 numpy array")
    if native_roles.shape != (n, 2, 5, 5):
        raise ValueError("native_roles must have shape [N,2,5,5]")
    if (not np.isfinite(native_roles).all() or (native_roles < 0).any()
            or (native_roles.sum(axis=-1) > 1 + 1e-6).any()):
        raise ValueError("native_roles must contain finite nonnegative frequencies with mass at most one")


def _remap_diagnostics(diagnostics, source_positions):
    # Membership lists are shared between same-cutoff/context rows by the base
    # builder. Remap each list once rather than copying them for every target.
    seen: set[int] = set()

    def remap(members):
        if id(members) not in seen:
            seen.add(id(members))
            members[:] = [source_positions[index] for index in members]

    for diagnostic in diagnostics:
        remap(diagnostic["source_indices"])
        remap(diagnostic["unknown_role_source_indices"])
        for name in ("global_role_source_indices", "local_role_source_indices"):
            for members in diagnostic[name]:
                remap(members)
        for side in diagnostic["player_source_indices"]:
            for members in side:
                remap(members)


def build_stable_meta_features(
    observations: Iterable[Mapping[str, Any]],
    targets: Iterable[Mapping[str, Any]],
    *,
    champion_ids: Sequence[Any],
    native_valid: np.ndarray,
    native_roles: np.ndarray,
) -> StableMetaFeatureBatch:
    """Retain target order; build causal unscaled arrays without changing C0."""
    rows = list(targets)
    n = len(rows)
    _native_inputs(native_valid, native_roles, n)
    vocabulary = tuple(_identity(value) for value in champion_ids)
    if any(value is None or value == "0" or value.lower() in _UNKNOWN_LABELS for value in vocabulary):
        raise ValueError("champion_ids must exclude absent and unknown champion bins")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("champion_ids must contain unique champion IDs")
    bins = {value: index + 1 for index, value in enumerate(vocabulary)}
    cutoffs = [_day(row["cutoff_day"], "cutoff_day") for row in rows]
    days = sorted(set(cutoffs))
    rosters = []
    players = set()
    for row in rows:
        sides = []
        for name in ("roster_a", "roster_b"):
            roster = tuple(_identity(value) for value in row[name])
            if len(roster) != 5:
                raise ValueError(f"{name} must contain five player IDs")
            sides.append(roster)
            players.update(player for player in roster if player is not None)
        rosters.append(sides)

    buckets: list[dict[str, _Prior]] = [{} for _ in days]
    moment_raw = []
    moment_positions = []
    meta_raw = []
    meta_positions = []
    for source_index, raw in enumerate(observations):
        played = _day(raw["played_day"], "played_day")
        release = _day(raw["release_day"], "release_day")
        effective = max(played, release)
        position = bisect_right(days, effective)
        if position == len(days):
            continue
        next_cutoff = days[position]
        player = _identity(raw.get("player_id"))
        meta_eligible = next_cutoff <= played + timedelta(days=30)
        long_eligible = player in players and next_cutoff <= played + timedelta(days=PLAYER_WINDOW_DAYS)
        if meta_eligible:
            meta_raw.append(raw)
            meta_positions.append(source_index)
        if meta_eligible or long_eligible:
            moment_raw.append(raw)
            moment_positions.append(source_index)
        if player in players:
            series = _identity(raw["series_id"])
            if series is None:
                raise ValueError("Missing observation series_id")
            ref = dict(source_index=source_index, series_id=series, player_id=player,
                       played_day=played.isoformat(), release_day=release.isoformat(),
                       effective_release_day=effective.isoformat())
            prior = buckets[position].get(player)
            if prior is None:
                prior = buckets[position][player] = _Prior()
            prior.add(ref)

    # The frozen builder sees exactly the records that can belong to at least
    # one 30-day cutoff. Canonical digests do not include sequence positions, so
    # filtering and remapping preserve both arrays and membership hashes.
    base = build_meta_features(meta_raw, rows, champion_ids=vocabulary)
    _remap_diagnostics(base.diagnostics, meta_positions)
    del meta_raw, meta_positions
    # Discard the base's subset catalog before allocating the final catalog.
    base = MetaFeatureBatch(base.arrays, base.row_ids, base.diagnostics, ())
    records, references = _observations(moment_raw, bins)
    del moment_raw
    sources = {}
    for index, original_position in enumerate(moment_positions):
        records[index] = replace(records[index], index=original_position)
        reference = references[index]
        reference["source_index"] = original_position
        sources[original_position] = reference
    del references, moment_positions

    arrays = base.arrays
    arrays["player_stats"].fill(0)
    arrays["player_roles"].fill(0)
    arrays["player_champions"].fill(0)
    arrays["player_valid"] = np.array(native_valid, copy=True)
    arrays["player_activity"] = np.zeros((n, 2, 5, ACTIVITY_WIDTH), dtype=np.float32)
    arrays["player_activity"][..., 6] = native_valid
    state = _PlayerWindow(records, len(vocabulary) + 1, players)
    del records
    prior_state: dict[str, _Prior] = {}
    bucket_cursor = 0
    current_day = None
    player_cache = {}
    empty_prior = _Prior()
    for index in sorted(range(n), key=lambda position: (cutoffs[position], position)):
        cutoff = cutoffs[index]
        if cutoff != current_day:
            state.advance(cutoff)
            while bucket_cursor < len(days) and days[bucket_cursor] <= cutoff:
                for player, prior in buckets[bucket_cursor].items():
                    if player not in prior_state:
                        prior_state[player] = prior
                    else:
                        prior_state[player].merge(prior)
                buckets[bucket_cursor].clear()
                bucket_cursor += 1
            current_day = cutoff
            player_cache.clear()
        long_members = [[[] for _ in range(5)] for _ in range(2)]
        prior_diagnostics = [[{} for _ in range(5)] for _ in range(2)]
        for side, roster in enumerate(rosters[index]):
            for slot, player in enumerate(roster):
                native_supported = bool(native_valid[index, side, slot])
                recent_count = len(base.diagnostics[index]["player_source_indices"][side][slot])
                pool = state.players.get(player)
                prior = prior_state.get(player, empty_prior)
                prior_diagnostics[side][slot] = prior.diagnostics()
                for ref in (prior.latest, prior.effective):
                    if ref is not None and ref["source_index"] not in sources:
                        sources[ref["source_index"]] = ref
                long_count = pool.n if pool is not None else 0
                activity = arrays["player_activity"][index, side, slot]
                activity[0] = np.log1p(recent_count)
                activity[1] = np.log1p(long_count)
                if prior.latest is not None:
                    activity[2] = np.log1p((cutoff - date.fromisoformat(prior.latest["played_day"])).days)
                    activity[3] = 1
                activity[4] = recent_count > 0
                activity[5] = long_count > 0
                activity[7] = prior.count > 0 or native_supported
                if pool is not None:
                    if player not in player_cache:
                        player_cache[player] = (pool.summary(), pool.roles / pool.n,
                                                pool.champion_distribution(), sorted(pool.members))
                    summary, roles, champions, members = player_cache[player]
                    arrays["player_stats"][index, side, slot] = summary
                    arrays["player_roles"][index, side, slot] = roles
                    arrays["player_champions"][index, side, slot] = champions
                    arrays["player_valid"][index, side, slot] = True
                    long_members[side][slot] = members
                elif native_supported:
                    arrays["player_roles"][index, side, slot] = native_roles[index, side, slot]
        base.diagnostics[index]["long_window_start_day"] = (cutoff - timedelta(days=PLAYER_WINDOW_DAYS)).isoformat()
        base.diagnostics[index]["long_player_source_indices"] = long_members
        base.diagnostics[index]["prior_activity"] = prior_diagnostics
    # The compact mapping preserves direct original-index lookup while avoiding
    # retention of hundreds of thousands of irrelevant source dictionaries.
    return StableMetaFeatureBatch(arrays, base.row_ids, base.diagnostics, sources)
