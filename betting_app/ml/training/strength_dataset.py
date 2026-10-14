"""Leakage-safe GOL.GG strength dataset builder for EXP-046.

The builder reads completed series from ``golgg_matches`` and computes every
feature strictly from matches that occurred earlier in chronological order.
It intentionally does not use ``latest-full`` ratings or ``w20-latest`` rows:
those are point-in-time snapshots for live inference and would leak future
information into historical training examples.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from betting_app.core.db import query_df
from betting_app.core.matching import normalize_team_name


@dataclass(frozen=True)
class StrengthDatasetConfig:
    """Configuration for the EXP-046 leakage-safe dataset."""

    min_date: str = "2015-01-01"
    max_date: str | None = None
    min_prior_matches: int = 0
    elo_k: float = 32.0
    initial_elo: float = 1500.0
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    include_draws: bool = False
    limit_rows: int | None = None


@dataclass(frozen=True)
class StrengthDataset:
    """Materialized EXP-046 dataset."""

    frame: pd.DataFrame
    feature_names: list[str]
    metadata: dict[str, Any]


@dataclass
class _TeamState:
    elo: float
    matches: int = 0
    wins: int = 0
    game_wins: int = 0
    game_losses: int = 0
    last_date: pd.Timestamp | None = None
    history: deque[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = deque(maxlen=64)


def load_golgg_match_results(
    *,
    min_date: str = "2015-01-01",
    max_date: str | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """Load completed non-draw GOL.GG series from the active database."""

    clauses = [
        "date IS NOT NULL",
        "team1_name IS NOT NULL",
        "team2_name IS NOT NULL",
        "team1_name <> ''",
        "team2_name <> ''",
        "date >= :min_date",
        "(team1_win = 1 OR team2_win = 1 OR winner_name IS NOT NULL)",
    ]
    params: dict[str, Any] = {"min_date": min_date}
    if max_date:
        clauses.append("date <= :max_date")
        params["max_date"] = max_date

    sql = f"""
        SELECT
            match_id,
            date,
            tournament_name,
            patch,
            team1_name,
            team2_name,
            team1_id,
            team2_id,
            team1_score,
            team2_score,
            team1_win,
            team2_win,
            draw,
            games_played,
            best_of,
            winner_name,
            loser_name
        FROM golgg_matches
        WHERE {' AND '.join(clauses)}
        ORDER BY date ASC, match_id ASC
    """
    if limit_rows:
        sql += "\nLIMIT :limit_rows"
        params["limit_rows"] = int(limit_rows)

    df = query_df(sql, params)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "team1_name", "team2_name"]).copy()
    df["match_id"] = df["match_id"].astype(str)
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)


def build_strength_dataset_from_db(config: StrengthDatasetConfig | None = None) -> StrengthDataset:
    """Load GOL.GG matches from DB and build the EXP-046 dataset."""

    cfg = config or StrengthDatasetConfig()
    raw = load_golgg_match_results(
        min_date=cfg.min_date,
        max_date=cfg.max_date,
        limit_rows=cfg.limit_rows,
    )
    return build_strength_dataset(raw, cfg)


def build_strength_dataset(raw_matches: pd.DataFrame, config: StrengthDatasetConfig | None = None) -> StrengthDataset:
    """Build point-in-time team-strength features from chronological matches."""

    cfg = config or StrengthDatasetConfig()
    teams: dict[str, _TeamState] = defaultdict(lambda: _TeamState(elo=cfg.initial_elo))
    h2h: dict[tuple[str, str], deque[int]] = defaultdict(lambda: deque(maxlen=32))
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)

    if raw_matches.empty:
        return StrengthDataset(frame=pd.DataFrame(), feature_names=[], metadata={"config": asdict(cfg), "rows": 0})

    ordered = raw_matches.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=["date"]).sort_values(["date", "match_id"])

    for record in ordered.to_dict(orient="records"):
        team1 = str(record.get("team1_name") or "").strip()
        team2 = str(record.get("team2_name") or "").strip()
        n1 = normalize_team_name(team1)
        n2 = normalize_team_name(team2)
        if not n1 or not n2 or n1 == n2:
            skipped["bad_team_name"] += 1
            continue

        target = _target_from_record(record)
        if target is None:
            skipped["no_binary_target"] += 1
            continue

        if not cfg.include_draws and _truthy(record.get("draw")):
            skipped["draw"] += 1
            continue

        state1 = teams[n1]
        state2 = teams[n2]
        if min(state1.matches, state2.matches) < cfg.min_prior_matches:
            skipped["min_prior_matches"] += 1
            _update_states(record, target, n1, n2, state1, state2, h2h, cfg)
            continue

        features = _features_before_match(record, n1, n2, state1, state2, h2h, cfg)
        rows.append(
            {
                "match_id": str(record.get("match_id") or ""),
                "date": record["date"].isoformat(),
                "tournament_name": record.get("tournament_name"),
                "patch": record.get("patch"),
                "team1_name": team1,
                "team2_name": team2,
                "team1_key": n1,
                "team2_key": n2,
                "target": int(target),
                **features,
            }
        )
        _update_states(record, target, n1, n2, state1, state2, h2h, cfg)

    frame = pd.DataFrame(rows)
    feature_names = [c for c in frame.columns if c not in _NON_FEATURE_COLUMNS]
    metadata = {
        "experiment_id": "EXP-046",
        "description": "Leakage-safe chronological GOL.GG strength dataset",
        "config": asdict(cfg),
        "raw_rows": int(len(raw_matches)),
        "rows": int(len(frame)),
        "feature_count": int(len(feature_names)),
        "skipped": dict(skipped),
        "target": "team1 series win -> 1, team2 series win -> 0",
        "anti_leakage": "Features are computed before updating each team's state with the current match.",
    }
    if not frame.empty:
        metadata["date_min"] = str(frame["date"].min())
        metadata["date_max"] = str(frame["date"].max())
    return StrengthDataset(frame=frame, feature_names=feature_names, metadata=metadata)


_NON_FEATURE_COLUMNS = {
    "match_id",
    "date",
    "tournament_name",
    "patch",
    "team1_name",
    "team2_name",
    "team1_key",
    "team2_key",
    "target",
}


def _target_from_record(record: dict[str, Any]) -> int | None:
    if _truthy(record.get("team1_win")):
        return 1
    if _truthy(record.get("team2_win")):
        return 0
    winner = str(record.get("winner_name") or "").strip()
    if winner:
        if normalize_team_name(winner) == normalize_team_name(str(record.get("team1_name") or "")):
            return 1
        if normalize_team_name(winner) == normalize_team_name(str(record.get("team2_name") or "")):
            return 0
    score1 = _safe_int(record.get("team1_score"))
    score2 = _safe_int(record.get("team2_score"))
    if score1 is not None and score2 is not None and score1 != score2:
        return int(score1 > score2)
    return None


def _features_before_match(
    record: dict[str, Any],
    n1: str,
    n2: str,
    state1: _TeamState,
    state2: _TeamState,
    h2h: dict[tuple[str, str], deque[int]],
    cfg: StrengthDatasetConfig,
) -> dict[str, float]:
    match_date = pd.Timestamp(record["date"])
    best_of = _safe_int(record.get("best_of")) or _infer_best_of(record) or 1
    patch_major, patch_minor = _parse_patch(record.get("patch"))

    out: dict[str, float] = {
        "team1_elo": state1.elo,
        "team2_elo": state2.elo,
        "elo_diff": state1.elo - state2.elo,
        "elo_expected_team1": _elo_expected(state1.elo, state2.elo),
        "team1_prior_matches": float(state1.matches),
        "team2_prior_matches": float(state2.matches),
        "prior_matches_diff": float(state1.matches - state2.matches),
        "team1_career_win_rate": _safe_ratio(state1.wins, state1.matches),
        "team2_career_win_rate": _safe_ratio(state2.wins, state2.matches),
        "career_win_rate_diff": _safe_ratio(state1.wins, state1.matches) - _safe_ratio(state2.wins, state2.matches),
        "team1_career_game_win_rate": _safe_ratio(state1.game_wins, state1.game_wins + state1.game_losses),
        "team2_career_game_win_rate": _safe_ratio(state2.game_wins, state2.game_wins + state2.game_losses),
        "best_of": float(best_of),
        "is_bo1": float(best_of == 1),
        "is_bo3": float(best_of == 3),
        "is_bo5": float(best_of == 5),
        "patch_major": float(patch_major) if patch_major is not None else np.nan,
        "patch_minor": float(patch_minor) if patch_minor is not None else np.nan,
        "team1_days_since_last": _days_since(state1.last_date, match_date),
        "team2_days_since_last": _days_since(state2.last_date, match_date),
    }
    out["days_since_last_diff"] = out["team1_days_since_last"] - out["team2_days_since_last"]

    for window in cfg.rolling_windows:
        _add_window_features(out, "team1", state1, window)
        _add_window_features(out, "team2", state2, window)
        out[f"win_rate_diff_w{window}"] = out[f"team1_win_rate_w{window}"] - out[f"team2_win_rate_w{window}"]
        out[f"game_win_rate_diff_w{window}"] = out[f"team1_game_win_rate_w{window}"] - out[f"team2_game_win_rate_w{window}"]

    pair_key, team1_is_first = _pair_key(n1, n2)
    prior_h2h = h2h[pair_key]
    if prior_h2h:
        first_win_rate = float(sum(prior_h2h) / len(prior_h2h))
        out["h2h_team1_win_rate"] = first_win_rate if team1_is_first else 1.0 - first_win_rate
    else:
        out["h2h_team1_win_rate"] = np.nan
    out["h2h_matches"] = float(len(prior_h2h))
    return out


def _update_states(
    record: dict[str, Any],
    target: int,
    n1: str,
    n2: str,
    state1: _TeamState,
    state2: _TeamState,
    h2h: dict[tuple[str, str], deque[int]],
    cfg: StrengthDatasetConfig,
) -> None:
    score1 = _safe_int(record.get("team1_score"))
    score2 = _safe_int(record.get("team2_score"))
    if score1 is None or score2 is None:
        score1, score2 = (1, 0) if target == 1 else (0, 1)

    expected1 = _elo_expected(state1.elo, state2.elo)
    margin_factor = max(1.0, float(score1 + score2)) ** 0.25
    delta = cfg.elo_k * margin_factor * (float(target) - expected1)
    state1.elo += delta
    state2.elo -= delta

    match_date = pd.Timestamp(record["date"])
    _append_result(state1, won=bool(target), game_wins=score1, game_losses=score2, match_date=match_date)
    _append_result(state2, won=not bool(target), game_wins=score2, game_losses=score1, match_date=match_date)

    pair_key, team1_is_first = _pair_key(n1, n2)
    h2h[pair_key].append(int(target if team1_is_first else not bool(target)))


def _append_result(state: _TeamState, *, won: bool, game_wins: int, game_losses: int, match_date: pd.Timestamp) -> None:
    state.matches += 1
    state.wins += int(won)
    state.game_wins += int(game_wins)
    state.game_losses += int(game_losses)
    state.last_date = match_date
    assert state.history is not None
    state.history.append({"won": int(won), "game_wins": int(game_wins), "game_losses": int(game_losses)})


def _add_window_features(out: dict[str, float], prefix: str, state: _TeamState, window: int) -> None:
    assert state.history is not None
    hist = list(state.history)[-window:]
    games_for = sum(int(x["game_wins"]) for x in hist)
    games_against = sum(int(x["game_losses"]) for x in hist)
    out[f"{prefix}_matches_w{window}"] = float(len(hist))
    out[f"{prefix}_win_rate_w{window}"] = float(sum(int(x["won"]) for x in hist) / len(hist)) if hist else np.nan
    out[f"{prefix}_game_win_rate_w{window}"] = _safe_ratio(games_for, games_for + games_against)


def _elo_expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def _safe_ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else np.nan


def _days_since(previous: pd.Timestamp | None, current: pd.Timestamp) -> float:
    if previous is None:
        return np.nan
    return float(max(0, (current - previous).days))


def _safe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _infer_best_of(record: dict[str, Any]) -> int | None:
    score1 = _safe_int(record.get("team1_score"))
    score2 = _safe_int(record.get("team2_score"))
    if score1 is None or score2 is None:
        return None
    played = score1 + score2
    if played <= 1:
        return 1
    if max(score1, score2) >= 3:
        return 5
    if max(score1, score2) == 2:
        return 3
    return played


def _parse_patch(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    parts = text.split(".")
    try:
        major = int(parts[0])
    except ValueError:
        return None, None
    minor = None
    if len(parts) > 1:
        digits = "".join(ch for ch in parts[1] if ch.isdigit())
        minor = int(digits) if digits else None
    return major, minor


def _pair_key(left: str, right: str) -> tuple[tuple[str, str], bool]:
    if left <= right:
        return (left, right), True
    return (right, left), False


def iter_feature_rows(dataset: StrengthDataset) -> Iterable[dict[str, Any]]:
    """Yield JSON-serializable rows for artifact snapshots."""

    for row in dataset.frame.to_dict(orient="records"):
        yield {k: _json_scalar(v) for k, v in row.items()}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
