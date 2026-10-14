"""First-map composition-oracle features with strictly released map history.

Role slots are TOP/JUNGLE/MID/ADC/SUPPORT, never the baseline roster's order.
This module reads no files, fits no model and mutates neither input frame.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src.models.prematch_features import _day, _identity, _integer

_SHARED = ("p_patch_major", "p_patch_minor", "p_patch_missing",
           "p_global_kills_mean", "p_global_duration_mean")
_METRICS = ("champion", "player_logcount", "role_logcount", "role_frequency",
            "kills_mean", "duration_mean")
_TRAIN_END = "2025-07-01"
_CAL_END = "2026-01-01"
_ASSESS_END = "2027-01-01"
_PRIOR_MAPS = 20.


@dataclass(frozen=True, slots=True)
class _Target:
    match_id: str
    day: str
    release: str
    best_of: int
    teams: tuple[str, str]
    rosters: tuple[frozenset[str], frozenset[str]]


@dataclass(slots=True)
class _Map:
    match_id: str
    game_id: str
    index: int
    day: str
    map_day: str
    release: str
    best_of: int
    teams: tuple[str, str]
    slots: tuple[tuple[tuple[str, str], ...], ...] | None
    absence: str | None
    kills: float
    duration: float
    patch: str
    patch_values: tuple[float, float, float]


@dataclass(slots=True)
class _History:
    player_counts: Counter = field(default_factory=Counter)
    role_counts: Counter = field(default_factory=Counter)
    role_totals: Counter = field(default_factory=Counter)
    # Per-metric sums and observed-label counts; composition count is distinct.
    role_kills: dict = field(default_factory=dict)
    role_duration: dict = field(default_factory=dict)
    kills_sum: float = 0.
    kills_count: int = 0
    duration_sum: float = 0.
    duration_count: int = 0
    max_release: str | None = None

    def consume(self, event: _Map) -> None:
        has_kills = math.isfinite(event.kills)
        has_duration = math.isfinite(event.duration)
        if has_kills:
            self.kills_sum += event.kills
            self.kills_count += 1
        if has_duration:
            self.duration_sum += event.duration
            self.duration_count += 1
        if event.slots is not None:
            for team in event.slots:
                for role, (player, champion) in enumerate(team):
                    self.player_counts[player, champion] += 1
                    key = role, champion
                    self.role_counts[key] += 1
                    self.role_totals[role] += 1
                    if has_kills:
                        total, count = self.role_kills.get(key, (0., 0))
                        self.role_kills[key] = total + event.kills, count + 1
                    if has_duration:
                        total, count = self.role_duration.get(key, (0., 0))
                        self.role_duration[key] = total + event.duration, count + 1
        if has_kills or has_duration or event.slots is not None:
            self.max_release = event.release

    @property
    def global_means(self) -> tuple[float, float]:
        return (self.kills_sum / self.kills_count if self.kills_count else math.nan,
                self.duration_sum / self.duration_count if self.duration_count else math.nan)

    @staticmethod
    def shrunk(history: dict, key: tuple[int, str], prior: float) -> float:
        if not math.isfinite(prior):
            return math.nan
        total, count = history.get(key, (0., 0))
        return (total + _PRIOR_MAPS * prior) / (count + _PRIOR_MAPS)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be an explicit boolean")
    return bool(value)


def _label(value: Any, *, kills: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(number):
        return math.nan
    if kills:
        return number if number >= 0 and number.is_integer() else math.nan
    return number if number > 0 else math.nan


def _patch(value: Any) -> tuple[str, tuple[float, float, float]]:
    if value is None or (not isinstance(value, str) and pd.isna(value)) or value == "":
        return "UNKNOWN", (math.nan, math.nan, 1.)
    if not isinstance(value, str):
        raise ValueError("patch must be an observed version string or missing")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", value)
    if match is None:
        raise ValueError(f"invalid observed patch: {value!r}")
    return value, (float(match[1]), float(match[2]), 0.)


def _composition(row: Any, teams: tuple[str, str]) -> tuple[tuple | None, str | None]:
    if not _boolean(row.draft_valid, "draft_valid"):
        reason = getattr(row, "draft_absence", None)
        return None, reason if isinstance(reason, str) and reason else "source_composition_invalid"
    try:
        slots = tuple(tuple((_identity(getattr(row, f"{side}_player_{role}", None), "player ID"),
                             _identity(getattr(row, f"{side}_champion_{role}", None), "champion ID"))
                            for role in range(5)) for side in ("t1", "t2"))
    except ValueError as exc:
        return None, str(exc)
    players = [set(player for player, _ in team) for team in slots]
    if teams[0] == teams[1]:
        return None, "duplicate source team ID"
    if any(len(roster) != 5 for roster in players):
        return None, "duplicate played player"
    if players[0] & players[1]:
        return None, "opposing rosters overlap"
    if len({champion for team in slots for _, champion in team}) != 10:
        return None, "duplicate played champion"
    return slots, None


def _targets(base: pd.DataFrame) -> list[_Target]:
    required = {"match_id", "date", "release_day", "best_of", "team_a", "team_b",
                "roster_a", "roster_b"}
    if not required <= set(base):
        raise ValueError("base frame missing columns: " + ",".join(sorted(required - set(base))))
    targets, identities = [], set()
    for row in base.itertuples(index=False):
        match_id = _identity(row.match_id, "canonical match ID")
        if match_id in identities:
            raise ValueError("duplicate canonical match ID " + match_id)
        identities.add(match_id)
        day = _day(row.date, "canonical date").isoformat()
        release = _day(row.release_day, "canonical release_day").isoformat()
        if release < day:
            raise ValueError("canonical release precedes forecast day for " + match_id)
        best_of = _integer(row.best_of, "canonical best_of", positive=True)
        if best_of not in (1, 3, 5):
            raise ValueError("canonical targets must be BO1/3/5")
        teams = (_identity(row.team_a, "team_a"), _identity(row.team_b, "team_b"))
        rosters = []
        for value in (row.roster_a, row.roster_b):
            if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 5:
                raise ValueError("canonical roster must contain five player IDs")
            roster = frozenset(_identity(player, "canonical player ID") for player in value)
            if len(roster) != 5:
                raise ValueError("canonical roster contains duplicate players")
            rosters.append(roster)
        if teams[0] == teams[1] or rosters[0] & rosters[1]:
            raise ValueError("canonical team/roster identities are invalid")
        targets.append(_Target(match_id, day, release, best_of, teams, tuple(rosters)))
    return targets


def _maps(frame: pd.DataFrame, targets: dict[str, _Target]) -> tuple[list[_Map], dict[str, _Map]]:
    required = {"match_id", "game_id", "map_index", "date", "map_date", "release_day",
                "best_of", "t1_id", "t2_id", "kills", "duration_minutes", "draft_valid"}
    if not required <= set(frame):
        raise ValueError("map frame missing columns: " + ",".join(sorted(required - set(frame))))
    series: dict[str, list[_Map]] = {}
    game_ids = set()
    for row in frame.itertuples(index=False):
        mid, gid = _identity(row.match_id, "source match ID"), _identity(row.game_id, "game ID")
        if gid in game_ids:
            raise ValueError("duplicate source game ID " + gid)
        game_ids.add(gid)
        index = _integer(row.map_index, "map_index", positive=True)
        day = _day(row.date, "source series date").isoformat()
        map_day = _day(row.map_date, "source map_date").isoformat()
        release = _day(row.release_day, "source release_day").isoformat()
        if map_day < day or release < map_day:
            raise ValueError("invalid source release/map chronology for " + gid)
        best_of = _integer(row.best_of, "source best_of", positive=True)
        if best_of not in (1, 2, 3, 5):
            raise ValueError("unsupported source best_of for " + mid)
        teams = (_identity(row.t1_id, "source t1_id"), _identity(row.t2_id, "source t2_id"))
        _boolean(row.draft_valid, "draft_valid")
        slots, absence = _composition(row, teams)
        patch, patch_values = _patch(getattr(row, "patch", None))
        event = _Map(mid, gid, index, day, map_day, release, best_of, teams, slots, absence,
                     _label(row.kills, kills=True), _label(row.duration_minutes, kills=False),
                     patch, patch_values)
        series.setdefault(mid, []).append(event)
    events, first_maps = [], {}
    for mid, games in series.items():
        games.sort(key=lambda event: event.index)
        if [event.index for event in games] != list(range(1, len(games) + 1)):
            raise ValueError("duplicate or noncontiguous source map order for " + mid)
        first = games[0]
        if any(event.day != first.day or event.best_of != first.best_of or
               set(event.teams) != set(first.teams) for event in games):
            raise ValueError("inconsistent source series identity for " + mid)
        if any(left.map_day > right.map_day for left, right in zip(games, games[1:])):
            raise ValueError("source map dates disagree with map order for " + mid)
        target = targets.get(mid)
        if target is not None and (target.day != first.day or target.best_of != first.best_of):
            raise ValueError("source/canonical date or best_of mismatch for " + mid)
        effective = max([event.release for event in games] +
                        ([target.release] if target is not None else []))
        for event in games:
            event.release = effective
        first_maps[mid] = first
        events.extend(games)
    events.sort(key=lambda event: (event.release, event.match_id, event.index))
    return events, first_maps


def _aligned(event: _Map | None, target: _Target) -> tuple[tuple | None, str | None]:
    if event is None:
        return None, "source_first_map_absent"
    if event.slots is None:
        return None, event.absence
    if set(event.teams) != set(target.teams):
        return None, "source_canonical_team_mismatch"
    aligned = tuple(event.slots[event.teams.index(team)] for team in target.teams)
    if any(frozenset(player for player, _ in team) != roster
           for team, roster in zip(aligned, target.rosters, strict=True)):
        return None, "source_canonical_player_set_mismatch"
    return aligned, None


def _partition(target: _Target, effective_release: str) -> str:
    if target.day < _TRAIN_END and effective_release < _TRAIN_END:
        return "TRAIN"
    if _TRAIN_END <= target.day < _CAL_END and effective_release < _CAL_END:
        return "CAL"
    if _CAL_END <= target.day < _ASSESS_END:
        return "ASSESS"
    return "UNASSIGNED"


def prepare_draft_data(map_frame: pd.DataFrame, base_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Return ordered full canonical rows, a frozen feature schema and data audit.

    Invalid/missing target compositions remain rows, with ``draft_valid=False``
    and an explicit absence reason. Structural source identity/order/chronology
    corruption raises instead of silently selecting a source record. Only valid,
    closed TRAIN target compositions define champion categorical codes; unknowns
    use -1. Historical statistics use all maps released strictly before each
    target day, releasing whole series together (including source-only BO2).
    """
    if not base_frame.columns.is_unique or not map_frame.columns.is_unique:
        raise ValueError("input columns must be unique")
    targets = _targets(base_frame)
    by_id = {target.match_id: target for target in targets}
    events, first_maps = _maps(map_frame, by_id)
    compositions, absences, partitions = [], [], []
    for target in targets:
        first = first_maps.get(target.match_id)
        slots, absence = _aligned(first, target)
        compositions.append(slots)
        absences.append(absence)
        partitions.append(_partition(target, first.release if first is not None else target.release))
    train_champions = {champion for partition, slots in zip(partitions, compositions, strict=True)
                       if partition == "TRAIN" and slots is not None
                       for team in slots for _, champion in team}
    vocabulary = {champion: code for code, champion in enumerate(sorted(train_champions))}
    draft_features = [f"d_{metric}_{side}_{role}" for side in ("a", "b")
                      for role in range(5) for metric in _METRICS]
    categorical = [f"d_champion_{side}_{role}" for side in ("a", "b") for role in range(5)]
    side_swap = {f"d_{metric}_{side}_{role}": f"d_{metric}_{other}_{role}"
                 for side, other in (("a", "b"), ("b", "a"))
                 for role in range(5) for metric in _METRICS}
    base_features = [name for name in base_frame.columns if name.startswith("x_")]
    schema = {"base_features": base_features, "shared_features": base_features + list(_SHARED),
              "draft_features": draft_features, "categorical_features": categorical,
              "side_swap": side_swap, "champion_vocabulary": vocabulary}
    schema["oracle_features"] = schema["shared_features"] + draft_features
    metadata_names = {"draft_valid", "draft_absence", "draft_history_max_release_day",
                      "draft_effective_release_day", "draft_player_champ_min_count", "map_index", "draft_patch"}
    if set(base_frame.columns) & (set(_SHARED) | set(draft_features) | metadata_names):
        raise ValueError("base frame already contains draft output columns")
    values = np.full((len(targets), len(_SHARED) + len(draft_features)), np.nan)
    # Missing compositions carry no fabricated counts or identities.
    values[:, len(_SHARED)::len(_METRICS)] = -1
    history = _History()
    position = 0
    max_releases: list[str | None] = [None] * len(targets)
    min_counts = np.full(len(targets), np.nan)
    unknown_slots = Counter()
    unknown_ids: dict[str, set[str]] = {key: set() for key in ("TRAIN", "CAL", "ASSESS", "UNASSIGNED")}
    for index in sorted(range(len(targets)), key=lambda i: (targets[i].day, targets[i].match_id)):
        target = targets[index]
        while position < len(events) and events[position].release < target.day:
            history.consume(events[position])
            position += 1
        global_kills, global_duration = history.global_means
        first = first_maps.get(target.match_id)
        patch_values = first.patch_values if first is not None else (math.nan, math.nan, 1.)
        values[index, :len(_SHARED)] = (*patch_values, global_kills, global_duration)
        max_releases[index] = history.max_release
        slots = compositions[index]
        if slots is None:
            continue
        counts = []
        offset = len(_SHARED)
        for team in slots:
            for role, (player, champion) in enumerate(team):
                key = role, champion
                player_count = history.player_counts.get((player, champion), 0)
                role_count = history.role_counts.get(key, 0)
                total = history.role_totals.get(role, 0)
                code = vocabulary.get(champion, -1)
                if code == -1:
                    unknown_slots[partitions[index]] += 1
                    unknown_ids[partitions[index]].add(champion)
                values[index, offset:offset + len(_METRICS)] = (
                    code, math.log1p(player_count), math.log1p(role_count),
                    role_count / total if total else 0.,
                    history.shrunk(history.role_kills, key, global_kills),
                    history.shrunk(history.role_duration, key, global_duration))
                offset += len(_METRICS)
                counts.append(player_count)
        min_counts[index] = min(counts)
    added = pd.DataFrame(values, index=base_frame.index, columns=list(_SHARED) + draft_features)
    added[categorical] = added[categorical].astype(np.int32)
    added["draft_valid"] = [slots is not None for slots in compositions]
    added["draft_absence"] = absences
    added["draft_history_max_release_day"] = max_releases
    added["draft_effective_release_day"] = [first_maps[t.match_id].release if t.match_id in first_maps
                                          else t.release for t in targets]
    added["draft_player_champ_min_count"] = min_counts
    added["map_index"] = 1
    added["draft_patch"] = [first_maps[t.match_id].patch if t.match_id in first_maps else "UNKNOWN"
                            for t in targets]
    frame = pd.concat((base_frame.copy(deep=True), added), axis=1)
    keys = ("TRAIN", "CAL", "ASSESS")
    partition_counts = Counter(partitions)
    valid_counts = Counter(partition for partition, slots in zip(partitions, compositions, strict=True)
                           if slots is not None)
    absence_counts = Counter(partition for partition, slots in zip(partitions, compositions, strict=True)
                             if slots is None)
    audit = {
        "canonical_rows": len(targets), "valid_target_rows": sum(slots is not None for slots in compositions),
        "invalid_target_rows": sum(slots is None for slots in compositions),
        "target_invalid_reasons": dict(Counter(reason for reason in absences if reason is not None)),
        "source_invalid_reasons": dict(Counter(event.absence for event in events if event.slots is None)),
        "partition_counts": {key: partition_counts[key] for key in keys},
        "partition_valid_counts": {key: valid_counts[key] for key in keys},
        "partition_absence_counts": {key: absence_counts[key] for key in keys},
        "unassigned_rows": partition_counts["UNASSIGNED"],
        "history_source_coverage": {
            "maps": len(events), "series": len(first_maps),
            "valid_composition_maps": sum(event.slots is not None for event in events),
            "invalid_composition_maps": sum(event.slots is None for event in events),
            "valid_kills_maps": sum(math.isfinite(event.kills) for event in events),
            "valid_duration_maps": sum(math.isfinite(event.duration) for event in events),
            "source_only_maps": sum(event.match_id not in by_id for event in events),
            "source_only_series": sum(mid not in by_id for mid in first_maps),
            "bo2_maps": sum(event.best_of == 2 for event in events),
            "maps_consumed_by_last_query": position,
            "effective_release_policy": "whole series max(raw release, corrected canonical release); strictly < target day",
        },
        "unknown_champions": {key: {"slots": unknown_slots[key], "identities": sorted(unknown_ids[key])}
                              for key in (*keys, "UNASSIGNED")},
        "champion_vocabulary_size": len(vocabulary),
        "target_patch_missing_rows": int(added.p_patch_missing.sum()),
    }
    return frame, schema, audit
