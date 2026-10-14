"""Read frozen raw map-player histories without altering either causal clock.

The public loader verifies the archived streaming reader before importing it.
``parse_records`` is the same ingestion path for deterministic in-memory fixtures.
No calendar-to-patch inference, target fields, or output artifacts are involved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.analysis.ev_asof_replay import (
    _as_day, _canonical_competition_context, _normalize_id,
    _player_id_from_payload, _raw_release_day,
)
from src.features.c0_meta_features import STATS, _patch, _UNKNOWN_LABELS

READER_SHA256 = "e1827fabee6f769b6d85e029ec18a62979d4df22e34915d9cea68f395d506c2c"
FLOAT32_MAX = float(np.finfo(np.float32).max)
SERIES_COLUMNS = ("golgg_match_id", "effective_label_release_day", "patch", "patch_source")


@dataclass(frozen=True)
class RawMetaSource:
    observations: list[dict[str, Any]]
    champion_ids: tuple[str, ...]
    series: pd.DataFrame
    audit: dict[str, Any]


def _champion(value: Any) -> str | None:
    identity = _normalize_id(value, "champion ID")
    if identity is None:
        return None
    if identity == "0" or identity.lower() in _UNKNOWN_LABELS:
        return "0"
    return identity


def parse_records(
    records: Iterable[Mapping[str, Any]],
    release_days: Mapping[str, Any],
    *,
    first_cutoff: Any = "2024-01-01",
) -> RawMetaSource:
    """Validate every source row, retaining only potentially usable observations.

    Vocabulary is independent of that retention filter: even an ancient champion
    must be counted if played and all-series release both precede first_cutoff.
    Observation patches use their own explicit map label (match fallback); the
    series catalog uses only the actual first map, never a later map's patch.
    """
    if not isinstance(release_days, Mapping):
        raise ValueError("release_days must be an ID-to-calendar-day mapping")
    cutoff = _as_day(first_cutoff, "first_cutoff")
    floor = cutoff - timedelta(days=30)
    observations: list[dict[str, Any]] = []
    catalog = []
    seen: set[str] = set()
    champions: set[str] = set()
    counts = Counter()
    patch_counts = Counter({key: 0 for key in (
        "explicit_game_map_count", "known_map_count", "unknown_map_count",
        "game_match_conflict_count", "known_match_count", "unknown_match_count",
        "explicit_match_patch_count", "within_series_conflict_count",
    )})
    patch_sources = Counter()
    patch_labels = Counter()
    role_labels = Counter()
    years = defaultdict(Counter)
    stat_present = Counter({stat: 0 for stat in STATS})
    stat_missing = Counter({stat: 0 for stat in STATS})
    stat_invalid = {stat: Counter() for stat in STATS}
    releases = Counter()
    reasons = Counter()
    bounds: dict[str, dict[str, str | None]] = {
        key: {"min": None, "max": None}
        for key in ("source_played_day", "source_release_day", "retained_played_day", "retained_release_day")
    }

    def extend_bounds(key: str, value: str) -> None:
        bound = bounds[key]
        bound["min"] = min(bound["min"], value) if bound["min"] else value
        bound["max"] = max(bound["max"], value) if bound["max"] else value

    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("historical match entries must be objects")
        identity = _normalize_id(raw.get("match_id"), "source match ID")
        if identity is None:
            raise ValueError("historical series has no usable match_id")
        if identity in seen:
            raise ValueError(f"Duplicate source series ID {identity}")
        seen.add(identity)
        release, release_source, release_reason = _raw_release_day(raw, release_days)
        release_text = release.isoformat()
        releases[release_source] += 1
        reasons[release_reason] += 1
        extend_bounds("source_release_day", release_text)
        games = raw["games"]
        competition = _canonical_competition_context(raw.get("tournament_name"))
        counts["missing_competition_series_count"] += int(competition is None)
        counts["source_series_count"] += 1
        series_year = years[str(_as_day(games[0]["date"], "first map date").year)]
        series_year["series_count"] += 1
        match_patch = _patch(raw.get("patch"))
        first_patch = None
        first_patch_source = "unknown"
        distinct_patches = set()
        for map_index, game in enumerate(games, start=1):
            played = _as_day(game["date"], f"historical map {identity}/{map_index} date")
            played_text = played.isoformat()
            year = years[str(played.year)]
            counts["source_map_count"] += 1
            year["map_count"] += 1
            extend_bounds("source_played_day", played_text)
            game_patch = _patch(game.get("patch"))
            patch = game_patch if game_patch is not None else match_patch
            patch_source = "game.patch" if game_patch is not None else "raw.patch" if match_patch is not None else "unknown"
            if map_index == 1:
                first_patch, first_patch_source = patch, patch_source
            patch_sources[patch_source] += 1
            patch_counts["explicit_game_map_count"] += int(game_patch is not None)
            patch_counts["known_map_count"] += int(patch is not None)
            patch_counts["unknown_map_count"] += int(patch is None)
            year["known_patch_map_count"] += int(patch is not None)
            year["unknown_patch_map_count"] += int(patch is None)
            patch_counts["game_match_conflict_count"] += int(
                game_patch is not None and match_patch is not None and game_patch != match_patch)
            if patch is not None:
                distinct_patches.add(patch)
                patch_labels[patch] += 1
            retain = played >= floor
            if retain:
                counts["retained_map_count"] += 1
                extend_bounds("retained_played_day", played_text)
                extend_bounds("retained_release_day", release_text)
            map_players: set[str] = set()
            for side in ("t1", "t2"):
                payload = game.get(f"{side}_players")
                context = f"{identity}/{map_index}/{side}"
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Missing player mapping for {context}")
                if len(payload) > 5:
                    raise ValueError(f"More than five player slots on map {context}")
                counts["missing_roster_slot_count"] += 5 - len(payload)
                for role, player in payload.items():
                    player_id = _player_id_from_payload(player, f"{context}/{role}")
                    if player_id is not None:
                        if player_id in map_players:
                            raise ValueError(f"Duplicate player IDs on map {context}: {player_id}")
                        map_players.add(player_id)
                    counts["missing_player_id_count"] += int(player_id is None)
                    champion = _champion(player.get("champion_id"))
                    counts["missing_champion_id_count"] += int(champion is None)
                    counts["unknown_champion_id_count"] += int(champion == "0")
                    role_labels[str(role)] += 1
                    if champion not in (None, "0") and played < cutoff and release < cutoff:
                        champions.add(champion)
                    stats = player.get("stats")
                    if stats is None:
                        counts["missing_stats_payload_count"] += 1
                        stats = {}
                    if not isinstance(stats, Mapping):
                        raise ValueError(f"Invalid stats payload in {context}/{role}")
                    values = np.zeros(len(STATS), dtype=np.float32) if retain else None
                    observed = np.zeros(len(STATS), dtype=np.bool_) if retain else None
                    for index, stat in enumerate(STATS):
                        value = stats.get(stat)
                        numeric = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                        valid = numeric and math.isfinite(value)
                        # The retained representation must itself remain finite.
                        if valid and abs(value) > FLOAT32_MAX:
                            raise ValueError(f"Stat {stat} exceeds float32 range in {context}/{role}")
                        stat_present[stat] += int(valid)
                        stat_missing[stat] += int(not valid)
                        if not valid:
                            category = "absent_or_null" if value is None else "bool" if isinstance(value, (bool, np.bool_)) else "nonfinite" if numeric else "nonnumeric"
                            stat_invalid[stat][category] += 1
                        if retain and valid:
                            values[index] = value
                            observed[index] = True
                    counts["source_observation_count"] += 1
                    year["observation_count"] += 1
                    if retain:
                        counts["retained_observation_count"] += 1
                        year["retained_observation_count"] += 1
                        observations.append(dict(
                            series_id=identity, played_day=played_text, release_day=release_text,
                            competition=competition, patch=patch, patch_source=patch_source,
                            player_id=player_id, role=role, champion_id=champion,
                            values12=values, observed12=observed, map_index=map_index, side=side,
                        ))
        patch_counts["known_match_count"] += int(first_patch is not None)
        patch_counts["unknown_match_count"] += int(first_patch is None)
        patch_counts["explicit_match_patch_count"] += int(match_patch is not None)
        patch_counts["within_series_conflict_count"] += int(len(distinct_patches) > 1)
        series_year["known_patch_series_count"] += int(first_patch is not None)
        series_year["unknown_patch_series_count"] += int(first_patch is None)
        series_year["patch_conflict_series_count"] += int(len(distinct_patches) > 1)
        catalog.append(dict(golgg_match_id=identity, effective_label_release_day=release_text,
                            patch=first_patch, patch_source=first_patch_source))
    for key in ("source_series_count", "source_map_count", "source_observation_count",
                "retained_map_count", "retained_observation_count", "missing_player_id_count",
                "missing_champion_id_count", "unknown_champion_id_count", "missing_stats_payload_count",
                "missing_roster_slot_count", "missing_competition_series_count"):
        counts.setdefault(key, 0)
    audit = dict(counts)
    audit.update(
        first_cutoff=cutoff.isoformat(), observation_played_floor=floor.isoformat(),
        vocabulary_size=len(champions), vocabulary_rule="played_day < first_cutoff AND all_series_release_day < first_cutoff",
        patch=dict(patch_counts) | {"map_sources": dict(patch_sources), "map_labels": dict(sorted(patch_labels.items()))},
        raw_role_label_counts=dict(sorted(role_labels.items())),
        per_year={key: dict(value) for key, value in sorted(years.items())},
        stats={stat: dict(observed=stat_present[stat], missing=stat_missing[stat],
                          missing_reasons=dict(stat_invalid[stat])) for stat in STATS},
        sample_bounds=bounds, release_sources=dict(releases), release_reasons=dict(reasons),
        corrected_release_entry_count=len(release_days),
        corrected_release_entries_without_source_count=len(set(release_days) - seen),
        provenance={"kind": "iterable_records", "patch_policy": "explicit_map_patch_else_explicit_match_patch", "series_patch_policy": "actual_first_map_only"},
    )
    return RawMetaSource(observations, tuple(sorted(champions)),
                         pd.DataFrame(catalog, columns=SERIES_COLUMNS), audit)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_meta_source(
    raw_path: str | Path,
    release_path: str | Path,
    reader_path: str | Path,
    *,
    first_cutoff: Any = "2024-01-01",
) -> RawMetaSource:
    """Stream frozen files with the exact authorized reader; never write outputs."""
    raw_path, release_path, reader_path = map(Path, (raw_path, release_path, reader_path))
    reader_digest = _sha256(reader_path)
    if reader_digest != READER_SHA256:
        raise ValueError(f"Streaming reader SHA256 mismatch: expected {READER_SHA256}, got {reader_digest}")
    spec = importlib.util.spec_from_file_location("_c0_meta_pinned_history_reader", reader_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import streaming reader {reader_path}")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    release_bytes = release_path.read_bytes()
    release_days = json.loads(release_bytes)
    result = parse_records(reader.iter_matches(raw_path), release_days, first_cutoff=first_cutoff)
    result.audit["provenance"].update(
        kind="frozen_files", raw={"path": str(raw_path), "sha256": _sha256(raw_path)},
        release={"path": str(release_path), "sha256": hashlib.sha256(release_bytes).hexdigest()},
        reader={"path": str(reader_path), "sha256": reader_digest, "expected_sha256": READER_SHA256},
    )
    return result
