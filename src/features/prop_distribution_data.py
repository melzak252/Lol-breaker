"""Sports-only prop labels and prior-release pace, independent of serving artifacts.

Labels are never imputed. Duration is converted from source seconds to minutes.
The supplied first-map roster is a scenario, not evidence it was public pre-start.
Pace uses exact player/organization IDs and whole-series releases strictly before
forecast day; BO2 maps remain eligible history, never prediction targets.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd

from scripts.export_prospective_features import iter_matches
from src.models.prematch_features import _day, _identity, _initial_advantage, _integer, _roster_payload, _team_ids
from src.utils import golgg_schema as schema

VERSION = "sports-prop-distribution-data-v1"
PACE_METRICS = ("kills_for", "kills_against", "total_kills", "duration_minutes")
_FLAGS = ("remake", "is_remake", "cancelled", "canceled", "forfeit", "is_forfeit")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _excluded_flags(payload: Mapping[str, Any]) -> list[str]:
    flags = [key for key in _FLAGS if payload.get(key) is True or payload.get(key) == 1]
    status = str(payload.get("status", "")).strip().lower()
    if status in {"remake", "cancelled", "canceled", "forfeit"}:
        flags.append("status=" + status)
    return flags


def _label_number(value: Any) -> tuple[float, str]:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "-", "--", "n/a", "na", "null", "none", "nan", "?"}):
        return math.nan, "missing"
    if isinstance(value, bool):
        return math.nan, "invalid_numeric"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan, "invalid_numeric"
    if not math.isfinite(result):
        return math.nan, "invalid_numeric"
    return result, "valid"


def _side_kills(game: Mapping[str, Any], side: str) -> tuple[float, str]:
    stats = game.get(side + "_stats")
    value, reason = _label_number(stats.get("kills") if isinstance(stats, Mapping) else None)
    if reason == "valid" and (value < 0 or not value.is_integer()):
        value, reason = math.nan, "invalid_count"
    return value, "valid" if reason == "valid" else side + "_kills_" + reason


def extract_map_labels(game: Mapping[str, Any]) -> dict[str, Any]:
    """Extract total kills (zero valid) and strictly positive duration in minutes."""
    if not isinstance(game, Mapping):
        return {"kills": math.nan, "duration_minutes": math.nan,
                "kills_reason": "map_missing", "duration_reason": "map_missing"}
    flags = _excluded_flags(game)
    if flags:
        reason = "map_excluded:" + ",".join(flags)
        return {"kills": math.nan, "duration_minutes": math.nan,
                "kills_reason": reason, "duration_reason": reason}
    a, ra = _side_kills(game, "t1")
    b, rb = _side_kills(game, "t2")
    duration, rd = _label_number(game.get("game_duration"))
    if rd != "valid":
        rd = "duration_" + rd
    elif duration <= 0:
        duration, rd = math.nan, "duration_nonpositive"
    return {"kills": a + b, "duration_minutes": duration / 60.,
            "kills_reason": ";".join(r for r in (ra, rb) if r != "valid") or "valid",
            "duration_reason": rd}


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0.).sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def _summaries(players: np.ndarray, teams: np.ndarray, player_names: list[str], team_names: list[str], prefix: str = "") -> tuple[np.ndarray, list[str]]:
    mean = _finite_mean(players, 2)
    variance = _finite_mean((players - mean[:, :, None, :]) ** 2, 2)
    blocks, names = [], []
    for tag, sides, fields in (("player_mean", mean, player_names),
                               ("player_std", np.sqrt(variance), player_names),
                               ("player_missing", (~np.isfinite(players)).sum(axis=2), player_names),
                               ("team", teams, team_names),
                               ("team_missing", (~np.isfinite(teams)).astype(float), team_names)):
        # Both sides must be observed to define a shared level or a gap. Do not
        # mistake one observed side for the common level of both opponents.
        blocks.extend(((sides[:, 0] + sides[:, 1]) / 2., np.abs(sides[:, 0] - sides[:, 1])))
        names.extend(prefix + tag + "_" + name + "_" + stat for stat in ("level", "gap") for name in fields)
    return np.concatenate(blocks, axis=1), names


def symmetric_features(players: np.ndarray, teams: np.ndarray, context: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Retain common level and gap without current side, draft or identity codes."""
    players, teams, context, baseline = (np.asarray(a, dtype=float) for a in (players, teams, context, baseline))
    n = len(players)
    if players.ndim != 4 or players.shape[1:3] != (2, 5):
        raise ValueError("players must have shape [N,2,5,P]")
    if teams.ndim != 3 or teams.shape[:2] != (n, 2) or context.shape != (n, 3) or baseline.shape != (n, 2):
        raise ValueError("teams/context/baseline shapes must align with players")
    if any(np.isinf(a).any() for a in (players, teams, context, baseline)):
        raise ValueError("features may be missing (NaN), not infinite")
    if np.any(np.isfinite(baseline) & ((baseline < 0) | (baseline > 1))):
        raise ValueError("baseline probabilities must lie in [0,1]")
    x, names = _summaries(players, teams, [str(i) for i in range(players.shape[-1])], [str(i) for i in range(teams.shape[-1])])
    extra = np.concatenate((context, np.abs(baseline - .5), (~np.isfinite(context)).astype(float), (~np.isfinite(baseline)).astype(float)), axis=1)
    names += ["bo1", "bo3", "bo5", "baseline_0_distance_half", "baseline_1_distance_half",
              "bo1_missing", "bo3_missing", "bo5_missing", "baseline_0_missing", "baseline_1_missing"]
    return np.concatenate((x, extra), axis=1), names


def _maps_label(raw: Mapping[str, Any], games: list[dict], best_of: int, team_a: str) -> tuple[float, str]:
    if best_of not in (1, 3, 5):
        return math.nan, "unsupported_best_of"
    if not games:
        return math.nan, "maps_missing"
    if _excluded_flags(raw) or any(_excluded_flags(g) for g in games):
        return math.nan, "series_or_map_excluded_flag"
    team_b = next(str(games[0][side + "_id"]) for side in ("t1", "t2") if str(games[0][side + "_id"]) != team_a)
    try:
        _initial_advantage(raw, team_a, team_b)
    except ValueError:
        return math.nan, "map_advantage"
    threshold = best_of // 2 + 1
    if not threshold <= len(games) <= best_of:
        return math.nan, "illegal_terminal_map_count"
    if raw.get("games_played") is not None:
        try:
            if _integer(raw["games_played"], "games_played") != len(games):
                return math.nan, "declared_map_count_mismatch"
        except ValueError:
            return math.nan, "declared_map_count_invalid"
    wins = [0, 0]
    for game in games:
        a, b = game.get("t1_win"), game.get("t2_win")
        if a not in (True, False, 0, 1) or b not in (True, False, 0, 1) or a == b or game.get("draw"):
            return math.nan, "map_winner_invalid"
        if max(wins) >= threshold:
            return math.nan, "maps_after_deciding_win"
        own_win = a if str(game["t1_id"]) == team_a else b
        wins[0 if own_win else 1] += 1
    if max(wins) != threshold or raw.get("draw"):
        return math.nan, "series_incomplete_or_draw"
    return float(len(games)), "valid"


@dataclass(slots=True)
class _PaceMap:
    release_day: str
    match_id: str
    map_index: int
    teams: tuple[str, str]
    rosters: tuple[tuple[str, ...], tuple[str, ...]]
    values: tuple[tuple[float, ...], tuple[float, ...]]


@dataclass
class PaceState:
    """Serializable bounded last-map history, keyed only by explicit IDs."""
    players: dict[str, deque] = field(default_factory=dict)
    teams: dict[str, deque] = field(default_factory=dict)
    max_release_day: str | None = None

    def consume(self, event: _PaceMap) -> None:
        for team, roster, values in zip(event.teams, event.rosters, event.values, strict=True):
            self.teams.setdefault(team, deque(maxlen=20)).append(values)
            for player in roster:
                self.players.setdefault(player, deque(maxlen=16)).append(values)
        if any(math.isfinite(v) for values in event.values for v in values):
            self.max_release_day = event.release_day

    @staticmethod
    def _row(history: deque | None) -> np.ndarray:
        if not history:
            return np.r_[np.full(4, np.nan), np.zeros(4)]
        values = np.asarray(history, dtype=float)
        return np.r_[_finite_mean(values, 0), np.isfinite(values).sum(axis=0)]

    def features(self, row: Any) -> tuple[np.ndarray, list[str]]:
        players = np.asarray([[self._row(self.players.get(str(player))) for player in roster]
                              for roster in (row.roster_a, row.roster_b)])[None]
        teams = np.asarray([self._row(self.teams.get(str(team))) for team in (row.team_a, row.team_b)])[None]
        fields = list(PACE_METRICS) + [metric + "_count" for metric in PACE_METRICS]
        x, names = _summaries(players, teams, fields, fields, "pace_")
        # Metric-first names keep the window statistic unambiguous to callers.
        for index, name in enumerate(names):
            if name.startswith("pace_player_mean_"):
                metric, statistic = name.removeprefix("pace_player_mean_").rsplit("_", 1)
                names[index] = f"pace_player_{metric}_mean_{statistic}"
        return x[0], names


def _quantiles(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return dict(zip(("min", "p01", "median", "p99", "max"), np.quantile(finite, [0, .01, .5, .99, 1]).tolist())) if len(finite) else {k: None for k in ("min", "p01", "median", "p99", "max")}


def _scan(source_path: Path, metadata: pd.DataFrame | None = None) -> tuple[dict[str, dict], list[_PaceMap], dict]:
    targets = {} if metadata is None else {row.match_id: row for row in metadata.itertuples(index=False)}
    labels, events, match_ids, game_ids = {}, [], set(), set()
    formats, flags, source_reasons = Counter(), Counter(), {task: Counter() for task in ("kills", "duration", "maps")}
    source_values = {task: [] for task in ("kills", "duration", "maps")}
    all_map_reasons = {task: Counter() for task in ("kills", "duration")}
    all_map_values = {task: [] for task in ("kills", "duration")}
    missing_stats = Counter()
    matched = 0
    for raw in iter_matches(source_path):
        match_id = _identity(raw.get("match_id"), "match ID")
        if match_id in match_ids:
            raise ValueError("duplicate source match ID " + match_id)
        match_ids.add(match_id)
        best_values = {_integer(raw[k], "best_of", positive=True) for k in ("BoN", "best_of") if raw.get(k) is not None}
        if len(best_values) != 1:
            raise ValueError("missing or conflicting best_of for " + match_id)
        best_of = next(iter(best_values))
        formats[str(best_of)] += 1
        games = raw.get("games")
        if not isinstance(games, list) or any(not isinstance(g, Mapping) for g in games):
            raise ValueError("invalid source games for " + match_id)
        header_day = _day(raw.get("date"), "series date")
        map_days = [_day(g.get("date", header_day), "map date") for g in games]
        source_release = max([header_day, *map_days]).isoformat()
        forecast_day = min([header_day, *map_days]).isoformat()
        row = targets.get(match_id)
        release = row.release_day if row is not None else source_release
        if release < source_release:
            raise ValueError("bank release precedes raw source release for " + match_id)
        teams = _team_ids(raw, games)[:2] if games else (None, None)
        first_rosters = None
        flags.update("series:" + f for f in _excluded_flags(raw))
        for index, game in enumerate(games):
            gid = _identity(game.get("game_id"), "game ID")
            if gid in game_ids:
                raise ValueError("duplicate source map ID " + gid)
            game_ids.add(gid)
            if game.get("match_id") is not None and _identity(game["match_id"], "map match ID") != match_id:
                raise ValueError("map match identity mismatch for " + gid)
            if game.get("source_best_of") is not None and _integer(game["source_best_of"], "source_best_of", positive=True) != best_of:
                raise ValueError("map best_of identity mismatch for " + gid)
            side_ids = tuple(_identity(game.get(side + "_id"), "map team ID") for side in ("t1", "t2"))
            rosters = tuple(_roster_payload(game, side)[0] for side in ("t1", "t2"))
            if set(rosters[0]) & set(rosters[1]):
                raise ValueError("opposing rosters share explicit player IDs for " + gid)
            if index == 0:
                first_rosters = dict(zip(side_ids, rosters))
            flags.update("map:" + f for f in _excluded_flags(game))
            a, ra = _side_kills(game, "t1")
            b, rb = _side_kills(game, "t2")
            missing_stats.update(r for r in (ra, rb) if r != "valid")
            map_label = extract_map_labels(game)
            duration = map_label["duration_minutes"]
            for task, column, reason in (("kills", "kills", "kills_reason"), ("duration", "duration_minutes", "duration_reason")):
                all_map_values[task].append(map_label[column])
                all_map_reasons[task][map_label[reason]] += 1
            if metadata is not None and not _excluded_flags(raw) and not _excluded_flags(game):
                events.append(_PaceMap(release, match_id, index, side_ids, rosters,
                                      ((a, b, a + b, duration), (b, a, a + b, duration))))
        label = extract_map_labels(games[0] if games else None)
        maps, maps_reason = _maps_label(raw, games, best_of, teams[0])
        if _excluded_flags(raw):
            reason = "series_excluded:" + ",".join(_excluded_flags(raw))
            label.update(kills=math.nan, duration_minutes=math.nan, kills_reason=reason, duration_reason=reason)
        label.update(maps_played=maps, maps_reason=maps_reason,
                     source_present=True, source_header_day=header_day.isoformat(),
                     source_max_day=source_release, label_source="first_recorded_map_and_terminal_series")
        for task, value, reason in (("kills", label["kills"], label["kills_reason"]), ("duration", label["duration_minutes"], label["duration_reason"]), ("maps", maps, maps_reason)):
            source_values[task].append(value)
            source_reasons[task][reason] += 1
        if row is not None:
            labels[match_id] = label
        if row is not None:
            matched += 1
            if row.best_of != best_of or row.date != forecast_day:
                raise ValueError("source/bank date or best_of mismatch for " + match_id)
            if games:
                if {row.team_a, row.team_b} != set(teams):
                    raise ValueError("source/bank team identity mismatch for " + match_id)
                for team, roster in ((row.team_a, row.roster_a), (row.team_b, row.roster_b)):
                    if len(roster) != 5 or len(set(roster)) != 5 or set(roster) != set(first_rosters[team]):
                        raise ValueError("source/bank first-map roster mismatch for " + match_id)
    audit = {"source_series": len(match_ids), "source_maps": len(game_ids), "source_formats": dict(formats),
             "excluded_target_source_formats": {k: v for k, v in formats.items() if int(k) not in (1, 3, 5)},
             "explicit_exclusion_flags": dict(flags), "history_side_kills_absences": dict(missing_stats),
             "source_first_map_task_reasons": {k: dict(v) for k, v in source_reasons.items()},
             "source_first_map_label_quantiles": {k: _quantiles(v) for k, v in source_values.items()},
             "source_all_map_task_reasons": {k: dict(v) for k, v in all_map_reasons.items()},
             "source_all_map_label_quantiles": {k: _quantiles(v) for k, v in all_map_values.items()},
             "matched_canonical_rows": matched, "source_only_series": len(match_ids) - matched if metadata is not None else None}
    return labels, events, audit


def audit_source_labels(source_path: Path) -> dict:
    """Read-only streaming source audit; no outputs, database or model fitting."""
    source_path = Path(source_path)
    digest = _hash(source_path)
    _, _, audit = _scan(source_path)
    return {"version": VERSION, "source_path": str(source_path.resolve()), "source_sha256": digest,
            "label_units": {"kills": "integer total across both teams", "duration_minutes": "source game_duration seconds / 60", "maps_played": "completed recorded maps in legal terminal BO1/3/5 series"}, **audit}


def _validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "date", "release_day", "best_of", "team_a", "team_b", "roster_a", "roster_b", "tournament", "max_history_release_day"}
    if not required <= set(metadata):
        raise ValueError("feature metadata missing columns: " + ",".join(sorted(required - set(metadata))))
    metadata = metadata.copy()
    for col in ("match_id", "team_a", "team_b"):
        metadata[col] = [_identity(v, col) for v in metadata[col]]
    if metadata.match_id.duplicated().any():
        raise ValueError("duplicate canonical match IDs")
    if not metadata.best_of.isin([1, 3, 5]).all():
        raise ValueError("canonical targets must be BO1/3/5")
    for col in ("date", "release_day"):
        metadata[col] = [_day(v, col).isoformat() for v in metadata[col]]
    if (metadata.release_day < metadata.date).any():
        raise ValueError("label release precedes forecast day")
    for col in ("roster_a", "roster_b"):
        metadata[col] = [[_identity(v, col) for v in roster] for roster in metadata[col]]
    for row in metadata.itertuples():
        if row.team_a == row.team_b or any(len(roster) != 5 or len(set(roster)) != 5 for roster in (row.roster_a, row.roster_b)) or set(row.roster_a) & set(row.roster_b):
            raise ValueError("canonical team/roster identities are invalid")
        if pd.notna(row.max_history_release_day) and _day(row.max_history_release_day, "max_history_release_day").isoformat() >= row.date:
            raise ValueError("raw bank history must be strictly before forecast day")
    return metadata


def build_distribution_dataset(source_path: Path, feature_dir: Path, output_dir: Path) -> dict:
    """Write a new versioned run, retaining every canonical row and task absence."""
    source_path, feature_dir, output_dir = map(Path, (source_path, feature_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("distribution output directory already exists: " + str(output_dir))
    source_hash = _hash(source_path)
    feature_audit = json.loads((feature_dir / "audit.json").read_text())
    if source_hash != feature_audit.get("source_sha256"):
        raise ValueError("source hash disagrees with raw feature audit")
    input_hashes = {name: _hash(feature_dir / name) for name in ("metadata.parquet", "arrays.npz", "audit.json")}
    for name in ("metadata.parquet", "arrays.npz"):
        expected = feature_audit.get("output_sha256", {}).get(name)
        if expected is not None and expected != input_hashes[name]:
            raise ValueError("raw feature artifact hash disagrees with audit: " + name)
    frame = _validate_metadata(pd.read_parquet(feature_dir / "metadata.parquet"))
    if feature_audit.get("target_rows", len(frame)) != len(frame):
        raise ValueError("canonical row count disagrees with feature audit")
    with np.load(feature_dir / "arrays.npz", allow_pickle=False) as arrays:
        x, names = symmetric_features(arrays["players"], arrays["teams"], arrays["context"], arrays["baseline_series"])
        if len(x) != len(frame):
            raise ValueError("feature arrays do not align with metadata")
        expected_context = np.column_stack([frame.best_of.eq(b).to_numpy(float) for b in (1, 3, 5)])
        if not np.array_equal(arrays["context"], expected_context):
            raise ValueError("feature BO context does not align with canonical metadata")
    labels, events, audit = _scan(source_path, frame)
    absent = {"kills": math.nan, "duration_minutes": math.nan, "maps_played": math.nan,
              "kills_reason": "source_series_absent", "duration_reason": "source_series_absent", "maps_reason": "source_series_absent",
              "source_present": False, "source_header_day": None, "source_max_day": None, "label_source": "source_series_absent"}
    label_frame = pd.DataFrame([labels.get(match_id, absent) for match_id in frame.match_id])
    state = PaceState()
    events.sort(key=lambda e: (e.release_day, e.match_id, e.map_index))
    position = 0
    pace, max_release, pace_names = [], [], None
    # Replay in forecast order, then restore the bank's exact row ordering.
    pace_by_index = {}
    rows = list(frame.itertuples(index=False))
    for index in sorted(range(len(rows)), key=lambda i: (rows[i].date, i)):
        row = rows[index]
        while position < len(events) and events[position].release_day < row.date:
            state.consume(events[position])
            position += 1
        values, current_names = state.features(row)
        pace_names = current_names
        pace_by_index[index] = (values, state.max_release_day)
    for index in range(len(frame)):
        values, release = pace_by_index[index]
        pace.append(values)
        max_release.append(release)
    if not len(frame):
        raise ValueError("canonical feature bank is empty")
    all_x = np.concatenate((x, np.asarray(pace)), axis=1)
    feature_names = ["x_" + name for name in names + pace_names]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("duplicate distribution feature names")
    frame = pd.concat((frame.reset_index(drop=True), label_frame, pd.DataFrame(all_x, columns=feature_names)), axis=1)
    frame["max_pace_release_day"] = max_release
    while position < len(events):
        state.consume(events[position])
        position += 1
    audit.update({"version": VERSION, "canonical_rows": len(frame), "source_absent_rows": int((~frame.source_present).sum()),
                  "history_maps_including_bo2": len(events), "task_reasons": {task: dict(Counter(frame[reason])) for task, reason in (("kills", "kills_reason"), ("duration", "duration_reason"), ("maps", "maps_reason"))},
                  "label_quantiles": {task: _quantiles(frame[col].tolist()) for task, col in (("kills", "kills"), ("duration", "duration_minutes"), ("maps", "maps_played"))},
                  "feature_missing_counts": {name: int(frame[name].isna().sum()) for name in feature_names},
                  "pace_support_quantiles": {name: _quantiles(frame[name].tolist()) for name in feature_names if name.startswith("x_pace_") and "count" in name},
                  "chronology": {"strict_prior_release": all(d is None or d < day for d, day in zip(max_release, frame.date)),
                                 "release_policy": "matching IDs: bank release; other IDs: max raw header/map date; same day withheld",
                                 "latest_pace_release_day": state.max_release_day},
                  "feature_names": feature_names,
                  "raw_feature_names": feature_audit.get("feature_names", {}),
                  "pace_definition": "exact-ID player last16 maps and organization last20 maps; per-metric observed mean and count; missing metrics skipped, not zero; symmetric side level/gap, player spread and missing counts",
                  "label_units": {"kills": "total integer kills", "duration_minutes": "source game_duration seconds /60", "maps_played": "legal completed series map count"},
                  "roster_availability": "first-map actual roster scenario; not certified pre-start publication",
                  "source_collection_completeness": "frozen local recovery source; source population coverage is not certified complete",
                  "source_path": str(source_path.resolve()), "source_sha256": source_hash,
                  "feature_dir": str(feature_dir.resolve()), "feature_input_sha256": input_hashes,
                  "implementation_sha256": _hash(Path(__file__))})
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path, state_path, audit_path = output_dir / "dataset.parquet", output_dir / "pace_state.joblib", output_dir / "audit.json"
    frame.to_parquet(dataset_path, index=False)
    joblib.dump(state, state_path)
    audit["output_sha256"] = {dataset_path.name: _hash(dataset_path), state_path.name: _hash(state_path)}
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return {"version": VERSION, "dataset_path": str(dataset_path), "audit_path": str(audit_path),
            "pace_state_path": str(state_path), "source_sha256": source_hash,
            "feature_input_sha256": input_hashes, "audit": audit}
