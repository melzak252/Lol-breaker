"""Standalone history-only champion research data; target composition is never input.

The parser streams the frozen JSON once. Compact map observations are replayed at
whole-series release, while opponent strength is frozen at the historical series'
pre-series day. Ratings reuse the existing prematch recipe, not a new estimator.
All canonical rows and recorded maps survive independently missing task labels.
"""
from __future__ import annotations

from bisect import insort
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.export_prospective_features import iter_matches
from src.features.c0_meta_features import STATS
from src.features.prop_distribution_data import (
    _excluded_flags, _maps_label, _side_kills, extract_map_labels,
)
from src.models import prematch_features as strength
from src.utils import golgg_schema

VERSION = "champion-history-data-v1"
TRAIN_END = "2025-07-01"
CAL_END = "2026-01-01"
ROLE_NAMES = strength.ROLE_ORDER
RULE_MODES = ("UNKNOWN", "STANDARD", "HARD_FEARLESS", "SOFT_FEARLESS")
HISTORY_FEATURE_NAMES = (*STATS, "kills_for", "kills_against", "total_kills",
                         "duration_minutes", "prior_opponent_strength_mean",
                         "prior_opponent_strength_sd", "played_age_days", "release_age_days")
PLAYER_FEATURE_NAMES = (*strength.PLAYER_FEATURE_NAMES, "known_player_identity",
                        "log_released_map_count", "log_teammate_pair_mean_count",
                        "log_teammate_pair_min_count")
CONTEXT_FEATURE_NAMES = ("best_of_1", "best_of_3", "best_of_5",
                         *("rule_" + mode.lower() for mode in RULE_MODES),
                         "prior_global_total_kills_mean", "prior_global_duration_minutes_mean")
_PROJECT = Path(__file__).resolve().parents[2]
_UNKNOWN = frozenset(("", "0", "unknown", "none", "null", "nan", "<na>", "n/a", "?"))


@dataclass(slots=True)
class _MapObservation:
    game_id: str
    map_index: int
    played: int
    players: tuple[str | None, ...]
    champions: tuple[str | None, ...]
    champion_status: np.ndarray
    roles: np.ndarray
    numeric: np.ndarray  # 10 x 18; age is materialized at each target
    patch: str | None
    competition: str | None
    labels: dict[str, Any]
    history_eligible: bool
    opponent_rosters: tuple[tuple[str | None, ...], tuple[str | None, ...]]


@dataclass(slots=True)
class _SourceSeries:
    match_id: str
    day: date
    header_day: date
    release: date
    best_of: int
    teams: tuple[str, str]
    maps: list[_MapObservation]
    rating_record: Any
    score_index: float
    score_reason: str
    rank: int = 0
    first_rosters: tuple[tuple[str | None, ...], tuple[str | None, ...]] = field(default_factory=tuple)


def _identity(value: Any, context: str, *, missing: bool = False) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        if missing:
            return None
        raise ValueError(f"Missing {context}")
    return strength._identity(value, context)


def _tag(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value.lower() in _UNKNOWN else value


def _champion(value: Any) -> tuple[str | None, int]:
    if value is None:
        return None, 1
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None, 2
    identity = str(value).strip()
    if identity.lower() in _UNKNOWN:
        return None, 2
    return identity, 3


def _player_identity(payload: Any, context: str) -> str | None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid player object {context}")
    ids = {_identity(payload[key], context, missing=True) for key in ("player_id", "id")
           if payload.get(key) is not None}
    ids.discard(None)
    if len(ids) > 1:
        raise ValueError(f"conflicting explicit player identities {context}")
    return next(iter(ids), None)


def _roster(values: Any, context: str) -> tuple[str | None, ...]:
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 5:
        raise ValueError(f"{context} requires five identity slots")
    result = tuple(_identity(v, context, missing=True) for v in values)
    known = [v for v in result if v is not None]
    if len(set(known)) != len(known):
        raise ValueError(f"duplicate known identities in {context}")
    return result


def _paths(research_root: str | Path | None) -> tuple[Path, ...]:
    if research_root is None:
        pointer = _PROJECT / "data/research_root.txt"
        research_root = os.environ.get("ENSEMBLE_RESEARCH_ROOT") or (pointer.read_text().strip() if pointer.is_file() else _PROJECT)
    root = Path(research_root).expanduser().resolve()
    return (_PROJECT, root) if root != _PROJECT else (_PROJECT,)


def _release_overrides(roots: tuple[Path, ...]) -> dict[str, str]:
    # These are explicit historical release corrections, not fitted parameters.
    relative = Path("data/03_primary/ground_up_lol_20260927_run001/release_days.json")
    for root in roots:
        path = root / relative
        if path.is_file():
            values = json.loads(path.read_text())
            return {str(k): strength._day(v, "release override").isoformat() for k, v in values.items()}
    return {}


def _rule_bindings(rules: Any, roots: tuple[Path, ...]) -> dict[str, tuple[str, str]]:
    if rules is None:
        relative = Path("data/05_model_input/draft_prop/draft_prop_20261007_01/rules.json")
        for root in roots:
            if (root / relative).is_file():
                rules = json.loads((root / relative).read_text())
                break
        else:
            rules = {"bindings": []}
    if isinstance(rules, (str, Path)):
        rules = json.loads(Path(rules).read_text())
    if not isinstance(rules, Mapping):
        raise ValueError("rules must be the exact evidence-backed rule matrix")
    result = {}
    for binding in rules.get("bindings", []):
        tag = binding.get("competition_edition_phase")
        mode = binding.get("draft_mode", "UNKNOWN")
        evidence = binding.get("evidence_key")
        if not isinstance(tag, str) or tag in result:
            raise ValueError("missing or duplicate exact competition rule binding")
        if mode not in RULE_MODES:
            mode = "UNKNOWN"
        if mode != "UNKNOWN" and (not isinstance(evidence, str) or not evidence.strip()
                                   or "not verified" in evidence.lower()):
            mode = "UNKNOWN"
        result[tag] = (mode, str(evidence or "no exact-phase evidence"))
    return result


def _terminal_label(raw: Mapping, games: list, best_of: int, team_a: str,
                    team_b: str) -> tuple[float, str]:
    _, reason = _maps_label(raw, games, best_of, team_a)
    if reason != "valid":
        return math.nan, reason
    wins = [0, 0]
    for game in games:
        own = game["t1_win"] if str(game["t1_id"]) == team_a else game["t2_win"]
        wins[0 if own else 1] += 1
    # Explicit header IDs establish the orientation of declared header scores.
    header_a = raw.get("tid_1", raw.get("t1_id"))
    header_b = raw.get("tid_2", raw.get("t2_id"))
    if header_a is not None and header_b is not None:
        for keys, team in ((("score_1", "t1_score"), str(header_a)),
                           (("score_2", "t2_score"), str(header_b))):
            expected = wins[0 if team == team_a else 1]
            try:
                declared = strength._optional_integer_alias(raw, keys, "series score")
            except ValueError:
                return math.nan, "declared_series_score_invalid"
            if declared is not None and declared != expected:
                return math.nan, "declared_series_score_mismatch"
        for key, team in (("t1_win", str(header_a)), ("t2_win", str(header_b))):
            if raw.get(key) is not None:
                try:
                    declared_win = strength._flag(raw[key], "declared series winner")
                except ValueError:
                    return math.nan, "declared_series_winner_invalid"
                if declared_win != (wins[0 if team == team_a else 1] > wins[1 if team == team_a else 0]):
                    return math.nan, "declared_series_winner_mismatch"
    winner = 0 if wins[0] > wins[1] else 1
    return float((0 if winner == 0 else 3) + wins[1 - winner]), "valid"


def _read_source(path: Path, targets: dict, overrides: dict) -> tuple[list[_SourceSeries], dict]:
    records = []
    match_ids, game_ids, player_ids = set(), set(), set()
    counts, formats, identity_provenance = Counter(), Counter(), Counter()
    stats_observed = Counter()
    source_task_reasons = {task: Counter() for task in ("kills", "duration")}
    for source_index, raw in enumerate(iter_matches(path)):
        match_id = _identity(raw.get("match_id"), "source match ID")
        if match_id in match_ids:
            raise ValueError("duplicate source series ID " + match_id)
        match_ids.add(match_id)
        games = raw.get("games")
        if not isinstance(games, list) or not games or any(not isinstance(g, Mapping) for g in games):
            raise ValueError("source series requires recorded maps: " + match_id)
        best_values = {strength._integer(raw[k], "best_of", positive=True)
                       for k in ("BoN", "best_of") if raw.get(k) is not None}
        if len(best_values) != 1:
            raise ValueError("missing or conflicting source best_of: " + match_id)
        best_of = next(iter(best_values))
        formats[str(best_of)] += 1
        header = strength._day(raw.get("date"), "source header day")
        days = [strength._day(g.get("date", header.isoformat()), "map day") for g in games]
        if any(b < a for a, b in zip(days, days[1:])):
            raise ValueError("map dates contradict series chronology: " + match_id)
        forecast = min(header, min(days))
        release = max(header, max(days), strength._day(overrides.get(match_id, header.isoformat()), "release override"))
        row = targets.get(match_id)
        if row is not None:
            base_release = strength._day(row["release_day"], "base release day")
            if base_release < release:
                raise ValueError("canonical release precedes source/corrected release: " + match_id)
            release = base_release
        a, b, provenance = strength._team_ids(raw, games)
        identity_provenance[provenance] += 1
        score, score_reason = _terminal_label(raw, games, best_of, a, b)
        excluded_series = bool(_excluded_flags(raw))
        maps, rating_maps = [], []
        first_rosters = None
        competition = _tag(golgg_schema.match_tournament(raw))
        for map_index, (game, played) in enumerate(zip(games, days), start=1):
            gid = _identity(game.get("game_id"), "source game ID")
            if gid in game_ids:
                raise ValueError("duplicate source game ID " + gid)
            game_ids.add(gid)
            if game.get("match_id") is not None and _identity(game["match_id"], "map match ID") != match_id:
                raise ValueError("source map/series identity mismatch " + gid)
            if game.get("source_best_of") is not None and strength._integer(game["source_best_of"], "source_best_of") != best_of:
                raise ValueError("source map best_of mismatch " + gid)
            sides = tuple(_identity(game.get(side + "_id"), "map team ID") for side in ("t1", "t2"))
            if set(sides) != {a, b}:
                raise ValueError("source map/series team identity mismatch " + gid)
            ordered_sides = ("t1", "t2") if sides[0] == a else ("t2", "t1")
            identities, champs, statuses, roles = [], [], [], []
            numeric = np.full((10, 18), np.nan, dtype=np.float32)
            numeric_row = 0
            own_rosters = []
            for side in ordered_sides:
                payload = game.get(side + "_players")
                if not isinstance(payload, Mapping) or len(payload) > 5:
                    raise ValueError("invalid historical player mapping " + gid + "/" + side)
                side_roster = []
                for raw_role, player in payload.items():
                    pid = _player_identity(player, gid + "/" + side + "/" + str(raw_role))
                    if pid is not None and pid in identities:
                        raise ValueError("duplicate historical map player ID " + gid)
                    identities.append(pid)
                    side_roster.append(pid)
                    if pid is not None:
                        player_ids.add(pid)
                    champion, status = _champion(player.get("champion_id"))
                    champs.append(champion)
                    statuses.append(status)
                    role = strength._canonical_role(raw_role)
                    if role is None:
                        role = strength._canonical_role(player.get("role", player.get("position", "")))
                    roles.append(ROLE_NAMES.index(role) + 1 if role in ROLE_NAMES else 0)
                    stats = player.get("stats")
                    if stats is not None and not isinstance(stats, Mapping):
                        counts["invalid_stats_payload"] += 1
                    stats = stats if isinstance(stats, Mapping) else {}
                    for channel, stat in enumerate(STATS):
                        value = strength._number(stats.get(stat))
                        if math.isfinite(value) and abs(value) <= np.finfo(np.float32).max:
                            numeric[numeric_row, channel] = value
                            stats_observed[stat] += 1
                    counts["source_player_observations"] += 1
                    counts["missing_player_identity"] += int(pid is None)
                    counts["missing_champion_identity"] += int(status == 1)
                    counts["unreliable_champion_identity"] += int(status == 2)
                    counts["unknown_historical_role"] += int(roles[-1] == 0)
                    numeric_row += 1
                missing_slots = 5 - len(side_roster)
                counts["missing_historical_roster_slots"] += missing_slots
                identities.extend([None] * missing_slots)
                champs.extend([None] * missing_slots)
                statuses.extend([1] * missing_slots)
                roles.extend([0] * missing_slots)
                side_roster.extend([None] * missing_slots)
                numeric_row += missing_slots
                own_rosters.append(tuple(side_roster))
            # Repeated identities are ambiguous without a verified mirror-pick
            # rule. Quarantine only those champion channels, not valid player
            # statistics or independent map/series labels.
            champion_counts = Counter(champion for champion in champs if champion is not None)
            ambiguous = {champion for champion, count in champion_counts.items() if count > 1}
            if ambiguous:
                counts["ambiguous_duplicate_champion_maps"] += 1
                for slot, champion in enumerate(champs):
                    if champion in ambiguous:
                        counts["unreliable_champion_identity"] += int(statuses[slot] != 2)
                        champs[slot], statuses[slot] = None, 2
            if first_rosters is None:
                first_rosters = tuple(own_rosters)
            labels = extract_map_labels(game)
            if excluded_series:
                reason = "series_excluded:" + ",".join(_excluded_flags(raw))
                labels.update(kills=math.nan, duration_minutes=math.nan,
                              kills_reason=reason, duration_reason=reason)
            for task in source_task_reasons:
                source_task_reasons[task][labels[task + "_reason"]] += 1
            kills_a, _ = _side_kills(game, ordered_sides[0])
            kills_b, _ = _side_kills(game, ordered_sides[1])
            for side_index, (own, opponent) in enumerate(((kills_a, kills_b), (kills_b, kills_a))):
                numeric[side_index * 5:(side_index + 1) * 5, 12:16] = strength._finite_float32(
                    (own, opponent, labels["kills"], labels["duration_minutes"]))
            history_eligible = not excluded_series and not _excluded_flags(game)
            maps.append(_MapObservation(gid, map_index, played.toordinal(), tuple(identities),
                        tuple(champs), np.asarray(statuses, dtype=np.uint8),
                        np.asarray(roles, dtype=np.int8), numeric,
                        _tag(game.get("patch")) or _tag(raw.get("patch")),
                        _tag(game.get("tournament_name", game.get("tournament"))) or competition,
                        labels, history_eligible, tuple(own_rosters)))
            # Missing role/identity/winner only blocks this rating update, not stats/props.
            if history_eligible:
                try:
                    ra, pa = strength._roster_payload(game, ordered_sides[0])
                    rb, pb = strength._roster_payload(game, ordered_sides[1])
                    wa = strength._flag(game.get(ordered_sides[0] + "_win"), "map winner")
                    wb = strength._flag(game.get(ordered_sides[1] + "_win"), "map winner")
                    if wa == wb or game.get("draw"):
                        raise ValueError("no unique map winner")
                    rating_maps.append(strength._Map(gid, ra, rb, strength._source_stats(pa),
                                                     strength._source_stats(pb), int(wa)))
                except ValueError:
                    counts["rating_ineligible_maps"] += 1
            else:
                counts["excluded_history_maps"] += 1
        if row is not None:
            if row["date"] != forecast.isoformat() or row["best_of"] != best_of:
                raise ValueError("canonical/source chronology or best_of mismatch: " + match_id)
            if {row["team_a"], row["team_b"]} != {a, b}:
                raise ValueError("canonical/source team mismatch: " + match_id)
            for team, roster in ((row["team_a"], row["roster_a"]), (row["team_b"], row["roster_b"])):
                actual = first_rosters[0 if team == a else 1]
                if any(pid is not None and pid not in actual for pid in roster):
                    raise ValueError("canonical/source first-map player alignment mismatch: " + match_id)
        rating = strength._Series(source_index, match_id, forecast, header, release,
                                 release > max(header, max(days)), forecast < header, best_of,
                                 a, b, first_rosters[0], first_rosters[1], competition,
                                 None, "explicit_raw_map_ids", tuple(rating_maps))
        records.append(_SourceSeries(match_id, forecast, header, release, best_of, (a, b),
                                     maps, rating, score, score_reason, first_rosters=first_rosters))
        counts["source_header_date_corrections"] += int(forecast < header)
        counts["whole_series_delayed_release"] += int(release > min(days))
    records.sort(key=lambda r: (r.day, _series_key(r.match_id)))
    for rank, record in enumerate(records):
        record.rank = rank
    audit = dict(counts)
    audit.update(source_series=len(records), source_maps=len(game_ids),
                 source_formats=dict(formats), source_unique_players=len(player_ids),
                 source_identity_provenance=dict(identity_provenance),
                 source_map_task_reasons={k: dict(v) for k, v in source_task_reasons.items()},
                 source_stat_missingness={name: {"observed": stats_observed[name],
                     "missing": counts["source_player_observations"] - stats_observed[name]}
                     for name in STATS}, corrected_release_entries=len(overrides))
    return records, audit


def _series_key(identity: str) -> tuple[int, Any]:
    return (0, int(identity)) if identity.isascii() and identity.isdecimal() else (1, identity)


def _vocabularies(records: list[_SourceSeries]) -> tuple[dict, dict, dict]:
    vocabularies = ({}, {}, {})
    for record in records:
        if record.day.isoformat() >= TRAIN_END or record.release.isoformat() >= TRAIN_END:
            continue
        for event in record.maps:
            if not event.history_eligible:
                continue
            for identity in sorted({identity for identity in event.champions if identity is not None}):
                vocabularies[0].setdefault(identity, len(vocabularies[0]) + 2)
            for vocabulary, identity in zip(vocabularies[1:], (event.patch, event.competition)):
                if identity is not None:
                    vocabulary.setdefault(identity, len(vocabulary) + 2)
    return vocabularies


def _bank_players(rows: list[dict], source_path: Path, roots: tuple[Path, ...]) -> tuple[np.ndarray | None, dict]:
    relative = Path("data/04_feature/prematch_symmetry_20260927_run001")
    source_hash = None
    for root in roots:
        bank = root / relative
        if not all((bank / name).is_file() for name in ("metadata.parquet", "arrays.npz", "audit.json")):
            continue
        receipt = json.loads((bank / "audit.json").read_text())
        if source_hash is None:
            with source_path.open("rb") as stream:
                source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if source_hash != receipt.get("source_sha256"):
            continue
        metadata = pd.read_parquet(bank / "metadata.parquet")
        positions = {str(row.match_id): i for i, row in enumerate(metadata.itertuples(index=False))}
        if any(row["match_id"] not in positions for row in rows):
            continue
        with np.load(bank / "arrays.npz", allow_pickle=False) as data:
            source_players = data["players"]
        result = np.empty((len(rows), 2, 5, len(strength.PLAYER_FEATURE_NAMES)), dtype=np.float32)
        for i, row in enumerate(rows):
            j = positions[row["match_id"]]
            actual = metadata.iloc[j]
            if str(actual.date) != row["date"] or str(actual.release_day) != row["release_day"]:
                raise ValueError("corrected strength bank chronology mismatch " + row["match_id"])
            actual_teams = (str(actual.team_a), str(actual.team_b))
            for side, (team, roster) in enumerate(((row["team_a"], row["roster_a"]),
                                                   (row["team_b"], row["roster_b"]))):
                actual_side = actual_teams.index(team)
                actual_roster = tuple(actual.roster_a if actual_side == 0 else actual.roster_b)
                for slot, player in enumerate(roster):
                    if player is None:
                        result[i, side, slot] = strength._player_team_features(strength.ReplayState(),
                                                       (None,) * 5, date.fromisoformat(row["date"]))[0]
                    else:
                        result[i, side, slot] = source_players[j, actual_side, actual_roster.index(player)]
        return result, {"kind": "corrected_symmetry_bank", "path": str(bank),
                        "source_sha256": source_hash, "loaded_once": True}
    if source_hash is None:
        with source_path.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return None, {"kind": "existing_prematch_replay", "recipe": "prematch_features.ReplayState",
                  "source_sha256": source_hash}


def prepare_history_data(source_path, base_frame, *, history_length=16, rules=None,
                         research_root=None):
    """Return canonical series/maps, prior-only arrays, schema, and complete audit.

    No model preprocessing is fitted here. Numeric absence stays NaN. History
    occupies the rightmost slots in ascending played-map order. Unobserved slots
    have mask=False; unreliable/missing champions on observed maps have code 0,
    while observed identities absent from closed TRAIN have code 1. Optional
    champion_status is audit provenance only and MUST NOT be read by any model.
    """
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length <= 0:
        raise ValueError("history_length must be a positive integer")
    required = ("match_id", "date", "release_day", "best_of", "team_a", "team_b",
                "roster_a", "roster_b", "tournament")
    if not isinstance(base_frame, pd.DataFrame) or not set(required).issubset(base_frame.columns):
        raise ValueError("base_frame is missing canonical pre-series metadata")
    roots = _paths(research_root)
    source_path = Path(source_path)
    if not source_path.is_absolute() and not source_path.is_file():
        source_path = next((root / source_path for root in roots if (root / source_path).is_file()), source_path)
    rows = []
    for row_index, raw in enumerate(base_frame[list(required)].to_dict("records")):
        row = dict(raw, row_index=row_index)
        for key in ("match_id", "team_a", "team_b"):
            row[key] = _identity(row[key], key)
        row["date"] = strength._day(row["date"], "canonical forecast day").isoformat()
        row["release_day"] = strength._day(row["release_day"], "canonical release day").isoformat()
        row["best_of"] = strength._integer(row["best_of"], "canonical best_of", positive=True)
        if row["best_of"] not in (1, 3, 5) or row["release_day"] < row["date"]:
            raise ValueError("unsupported canonical best_of or invalid chronology")
        row["roster_a"] = _roster(row["roster_a"], "roster_a")
        row["roster_b"] = _roster(row["roster_b"], "roster_b")
        if row["team_a"] == row["team_b"] or (set(row["roster_a"]) & set(row["roster_b"])) - {None}:
            raise ValueError("canonical opposing teams/players overlap")
        rows.append(row)
    targets = {row["match_id"]: row for row in rows}
    if len(targets) != len(rows):
        raise ValueError("duplicate canonical match IDs")
    overrides = _release_overrides(roots)
    records, audit = _read_source(source_path, targets, overrides)
    by_id = {record.match_id: record for record in records}
    champion_vocab, patch_vocab, competition_vocab = _vocabularies(records)
    bindings = _rule_bindings(rules, roots)
    n, length = len(rows), history_length
    shape = (n, 2, 5, length)
    arrays = dict(history=np.full((*shape, len(HISTORY_FEATURE_NAMES)), np.nan, dtype=np.float32),
                  champions=np.zeros(shape, dtype=np.int64), roles=np.zeros(shape, dtype=np.int64),
                  patches=np.zeros(shape, dtype=np.int64), competitions=np.zeros(shape, dtype=np.int64),
                  mask=np.zeros(shape, dtype=np.bool_), champion_status=np.zeros(shape, dtype=np.uint8),
                  players=np.full((n, 2, 5, len(PLAYER_FEATURE_NAMES)), np.nan, dtype=np.float32),
                  role_probs=np.zeros((n, 2, 5, 6), dtype=np.float32),
                  context=np.full((n, len(CONTEXT_FEATURE_NAMES)), np.nan, dtype=np.float32),
                  best_of=np.asarray([row["best_of"] for row in rows], dtype=np.int64))
    bank_players, strength_provenance = _bank_players(rows, source_path, roots)
    if bank_players is not None:
        arrays["players"][..., :len(strength.PLAYER_FEATURE_NAMES)] = bank_players
    source_by_day, targets_by_day, releases = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        source_by_day[record.day].append(record)
        releases[record.release].append(record)
    for row in rows:
        targets_by_day[date.fromisoformat(row["date"])].append(row)
    release_days = sorted(releases)
    state = strength.ReplayState()
    history, role_counts, map_counts, pair_counts = defaultdict(list), defaultdict(lambda: np.zeros(6, dtype=np.int64)), Counter(), Counter()
    global_sum, global_count = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.int64)
    release_position = 0
    source_lookup = {record.rank: record for record in records}
    support_rows = [{} for _ in rows]
    for day in sorted(set(source_by_day) | set(targets_by_day)):
        while release_position < len(release_days) and release_days[release_position] < day:
            release = release_days[release_position]
            batch = sorted(releases[release], key=lambda record: _series_key(record.match_id))
            for record in batch:
                for event_index, event in enumerate(record.maps):
                    if not event.history_eligible:
                        continue
                    for slot, player in enumerate(event.players):
                        if player is None:
                            continue
                        entries = history[player]
                        insort(entries, (event.played, record.rank, event_index, slot))
                        if len(entries) > length:
                            entries.pop(0)
                        map_counts[player] += 1
                        role_counts[player][event.roles[slot]] += 1
                    for roster in event.opponent_rosters:
                        known = [player for player in roster if player is not None]
                        for i, player in enumerate(known):
                            for other in known[i + 1:]:
                                pair_counts[tuple(sorted((player, other)))] += 1
                    values = np.asarray((event.labels["kills"], event.labels["duration_minutes"]))
                    valid = np.isfinite(values)
                    global_sum[valid] += values[valid]
                    global_count[valid] += 1
            rating_records = [record.rating_record for record in batch if record.rating_record.maps]
            if rating_records:
                strength._consume_release_day(state, rating_records, release)
            release_position += 1
        # Opponent values use the same strictly-prior state for ALL maps of a series.
        for record in source_by_day[day]:
            for event in record.maps:
                for side in range(2):
                    mean, variance = strength._player_team_strength(state, event.opponent_rosters[1 - side],
                                                                    day, strength._PROCESS_SLOW)
                    event.numeric[side * 5:(side + 1) * 5, 16:18] = mean, math.sqrt(variance)
        for row in targets_by_day[day]:
            i = row["row_index"]
            arrays["context"][i, :7] = 0
            arrays["context"][i, {1: 0, 3: 1, 5: 2}[row["best_of"]]] = 1
            mode, evidence = bindings.get(row["tournament"], ("UNKNOWN", "no exact-phase evidence"))
            arrays["context"][i, 3 + RULE_MODES.index(mode)] = 1
            arrays["context"][i, 7:] = np.divide(global_sum, global_count,
                                                out=np.full(2, np.nan), where=global_count > 0)
            counts, ages, champion_known, champion_unknown, champion_unseen = [], [], 0, 0, 0
            max_release = None
            for side, roster in enumerate((row["roster_a"], row["roster_b"])):
                if bank_players is None:
                    arrays["players"][i, side, :, :len(strength.PLAYER_FEATURE_NAMES)] = strength._player_team_features(state, roster, day)
                for slot, player in enumerate(roster):
                    entries = history.get(player, ()) if player is not None else ()
                    counts.append(len(entries))
                    pairs = [pair_counts.get(tuple(sorted((player, other))), 0)
                             for other in roster if player is not None and other is not None and other != player]
                    arrays["players"][i, side, slot, -4:] = (
                        float(player is not None), math.log1p(map_counts[player]) if player is not None else 0,
                        math.log1p(float(np.mean(pairs))) if pairs else 0,
                        math.log1p(min(pairs)) if pairs else 0)
                    roles = role_counts.get(player) if player is not None else None
                    if roles is None or not roles.sum():
                        arrays["role_probs"][i, side, slot, 0] = 1
                    else:
                        arrays["role_probs"][i, side, slot] = roles / roles.sum()
                    for history_slot, (_, rank, event_index, source_slot) in enumerate(entries, start=length - len(entries)):
                        record = source_lookup[rank]
                        event = record.maps[event_index]
                        values = arrays["history"][i, side, slot, history_slot]
                        values[:18] = event.numeric[source_slot]
                        values[18:] = day.toordinal() - event.played, (day - record.release).days
                        identity = event.champions[source_slot]
                        code = 0 if identity is None else champion_vocab.get(identity, 1)
                        arrays["champions"][i, side, slot, history_slot] = code
                        arrays["roles"][i, side, slot, history_slot] = event.roles[source_slot]
                        arrays["patches"][i, side, slot, history_slot] = 0 if event.patch is None else patch_vocab.get(event.patch, 1)
                        arrays["competitions"][i, side, slot, history_slot] = 0 if event.competition is None else competition_vocab.get(event.competition, 1)
                        arrays["mask"][i, side, slot, history_slot] = True
                        arrays["champion_status"][i, side, slot, history_slot] = event.champion_status[source_slot]
                        champion_known += int(code >= 2)
                        champion_unseen += int(code == 1)
                        champion_unknown += int(code == 0)
                        ages.append((day - record.release).days)
                        max_release = max(max_release or record.release, record.release)
            support_rows[i] = dict(draft_mode=mode, rule_evidence=evidence, rule_unknown=mode == "UNKNOWN",
                history_min=min(counts), history_mean=float(np.mean(counts)), history_max=max(counts),
                cold_player_count=sum(count == 0 for count in counts), unknown_player_count=sum(p is None for p in (*row["roster_a"], *row["roster_b"])),
                history_profile="cold" if not sum(counts) else "partial" if min(counts) < length else "full",
                roster_profile="unknown_identity" if any(p is None for p in (*row["roster_a"], *row["roster_b"])) else "new_roster" if np.min(arrays["players"][i, :, :, -1]) == 0 else "established",
                max_history_release_day=max_release.isoformat() if max_release else None,
                history_max_release_age_days=max(ages) if ages else math.nan,
                known_champion_history=champion_known, missing_champion_history=champion_unknown,
                unseen_champion_history=champion_unseen)
    series_rows, map_rows = [], []
    series_reasons = Counter()
    map_reasons = {task: Counter() for task in ("kills", "duration")}
    for row, support in zip(rows, support_rows):
        record = by_id.get(row["match_id"])
        score = record.score_index if record is not None else math.nan
        reason = record.score_reason if record is not None else "source_series_missing"
        if record is not None and row["team_a"] != record.teams[0] and math.isfinite(score):
            score = score + 3 if score < 3 else score - 3
        first = record.maps[0] if record is not None else None
        split = ("TRAIN" if row["date"] < TRAIN_END and row["release_day"] < TRAIN_END else
                 "CAL" if TRAIN_END <= row["date"] < CAL_END and row["release_day"] < CAL_END else
                 "ASSESS" if CAL_END <= row["date"] < "2027-01-01" else "UNASSIGNED")
        series_reasons[reason] += 1
        series_rows.append(dict(row, **support, effective_release_day=row["release_day"], split=split,
                               score_index=score, winner=float(score < 3) if math.isfinite(score) else math.nan,
                               score_reason=reason, winner_reason=reason,
                               score_absence=None if reason == "valid" else reason,
                               winner_absence=None if reason == "valid" else reason,
                               patch=first.patch if first else None, target_patch_input=False,
                               target_champion_missing_count=int(np.count_nonzero(first.champion_status == 1)) if first else 10,
                               target_champion_unreliable_count=int(np.count_nonzero(first.champion_status == 2)) if first else 0,
                               source_present=record is not None,
                               played_maps=len(record.maps) if record else 0))
        if record is None:
            continue
        for event in record.maps:
            labels = event.labels
            for task in map_reasons:
                map_reasons[task][labels[task + "_reason"]] += 1
            map_rows.append(dict(row_index=row["row_index"], match_id=row["match_id"],
                game_id=event.game_id, map_index=event.map_index, date=row["date"], best_of=row["best_of"],
                tournament=row["tournament"], map_date=date.fromordinal(event.played).isoformat(),
                effective_release_day=row["release_day"], patch=event.patch,
                draft_mode=support["draft_mode"], rule_unknown=support["rule_unknown"],
                history_profile=support["history_profile"], roster_profile=support["roster_profile"],
                missing_champion_count=int(np.count_nonzero(event.champion_status == 1)),
                unreliable_champion_count=int(np.count_nonzero(event.champion_status == 2)),
                missing_player_count=sum(player is None for player in event.players),
                missing_role_count=int(np.count_nonzero(event.roles == 0)),
                missing_stat_count=int(np.isnan(event.numeric[:, :12]).sum()),
                **labels, kills_absence=None if labels["kills_reason"] == "valid" else labels["kills_reason"],
                duration_absence=None if labels["duration_reason"] == "valid" else labels["duration_reason"]))
    series = pd.DataFrame(series_rows)
    map_columns = ("row_index", "match_id", "game_id", "map_index", "date", "best_of", "tournament",
                   "map_date", "effective_release_day", "patch", "draft_mode", "rule_unknown",
                   "history_profile", "roster_profile", "kills", "duration_minutes", "kills_reason",
                   "duration_reason", "kills_absence", "duration_absence", "missing_champion_count",
                   "unreliable_champion_count", "missing_player_count", "missing_role_count", "missing_stat_count")
    maps = pd.DataFrame(map_rows, columns=map_columns)
    schema = dict(version=VERSION, history_length=length,
        history_feature_names=list(HISTORY_FEATURE_NAMES), player_feature_names=list(PLAYER_FEATURE_NAMES),
        context_feature_names=list(CONTEXT_FEATURE_NAMES), role_names=["UNKNOWN", *ROLE_NAMES],
        champion_vocabulary=champion_vocab, patch_vocabulary=patch_vocab,
        competition_vocabulary=competition_vocab, champion_vocabulary_size=len(champion_vocab) + 2,
        patch_vocabulary_size=len(patch_vocab) + 2, competition_vocabulary_size=len(competition_vocab) + 2,
        categorical_codes={"missing_or_unreliable": 0, "out_of_closed_train_vocabulary": 1, "known_minimum": 2},
        model_input_keys=["history", "champions", "roles", "patches", "competitions", "mask", "players", "role_probs", "context", "best_of"],
        audit_only_array_keys=["champion_status"], champion_status_codes={"padding": 0, "missing": 1, "unreliable": 2, "reliable": 3},
        champion_numeric_channels=[], target_patch_input=False, target_composition_input=False,
        vocabulary_rule="source_series_forecast_day < 2025-07-01 AND whole_series_release_day < 2025-07-01",
        history_order="last released played-map observations; chronological right alignment",
        history_release_rule="whole-series effective_release_day < target_series_forecast_day",
        opponent_strength_rule="existing prematch player slow posterior at historical SERIES forecast day, strictly released history only",
        current_roster_scenario="known first-map identity set; supplied slot order is not a realized-role input",
        strength_provenance=strength_provenance,
        array_shapes={key: list(value.shape) for key, value in arrays.items()})
    audit.update(canonical_series=n, canonical_maps=len(maps), matched_canonical_rows=sum(r["match_id"] in by_id for r in rows),
        source_only_series=len(records) - sum(r["match_id"] in by_id for r in rows),
        source_path=str(source_path), source_sha256=strength_provenance["source_sha256"],
        canonical_series_task_reasons={"score": dict(series_reasons), "winner": dict(series_reasons)},
        canonical_map_task_reasons={k: dict(v) for k, v in map_reasons.items()},
        canonical_map_missingness={column: int(maps[column].sum()) for column in
            ("missing_champion_count", "unreliable_champion_count", "missing_player_count",
             "missing_role_count", "missing_stat_count")},
        partition_counts=series.split.value_counts().to_dict() if n else {},
        champion_vocabulary_size=len(champion_vocab), patch_vocabulary_size=len(patch_vocab),
        competition_vocabulary_size=len(competition_vocab), strength_provenance=strength_provenance,
        alignment_audit={"all_canonical_rows_checked": n, "target_roles_used": False,
                         "target_champions_required": False, "whole_series_release": True},
        array_missingness={"history_observed_tokens": int(arrays["mask"].sum()),
            "history_numeric_missing_by_channel": dict(zip(HISTORY_FEATURE_NAMES,
                np.isnan(arrays["history"]).sum(axis=(0, 1, 2, 3)).tolist())),
            "player_numeric_missing_by_channel": dict(zip(PLAYER_FEATURE_NAMES,
                np.isnan(arrays["players"]).sum(axis=(0, 1, 2)).tolist())),
            "canonical_cold_rows": sum(s["history_profile"] == "cold" for s in support_rows)},
        limitations=["Current roster is a known-first-map retrospective identity scenario, not certified live roster availability.",
                     "Raw calendar dates impose strict previous-day release; no intraday order is invented.",
                     "Target patch is descriptive only; exact rule evidence is retrospective applicability, not publication-time certification.",
                     "Unknown historic identities cannot be joined to a player; usable map targets remain independent."])
    return series, maps, arrays, schema, audit
