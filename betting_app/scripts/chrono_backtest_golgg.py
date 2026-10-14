#!/usr/bin/env python3
"""Chronological backtest on finished matches using point-in-time ratings + W20 features.

Process:
  1. Restore RatingManager state from 'latest-full' snapshot (cutoff 2026-05-28)
  2. Initialize W20 deque by processing all golgg_matches up to 2026-05-28
  3. Load finished canonical_matches with GOL.GG mappings (May 29 - Jun 8)
  4. For each match in chronological order:
     a. update_before_match() → Glicko time decay
     b. predict_match() → rating features (20 OPTUNA features)
     c. W20 features from deque (20 rolling features)
     d. binomial series features (6 features)
     e. Build 46-feature vector
     f. Run pre-trained pipeline with order symmetry + Platt calibration
     g. Store prediction in canonical_predictions
     h. Update W20 deque with match games
     i. update_after_game() + update_after_match() → ratings

Usage:
    DATABASE_URL=postgresql+psycopg2://betting:betting_local_password@localhost:5432/betting \\
    python betting_app/scripts/chrono_backtest_golgg.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import UTC, date, datetime
from math import comb
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DATABASE_URL"] = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://betting:betting_local_password@localhost:5432/betting",
)

from betting_app.core.db import connect, transaction  # noqa: E402
from betting_app.core.matching import normalize_team_name  # noqa: E402
from betting_app.scripts.rebuild_ratings import rating_from_row  # noqa: E402
from src.ratings.manager import RatingManager  # noqa: E402
from src.models.team_order import swap_orientation, symmetrize_binary_probabilities  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RATINGS_VERSION = "latest-full"
MODEL_VERSION = "exp-039-chrono-backtest"
PIPELINE_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib"

RATING_SYSTEM_PARAMS: dict[str, dict[str, Any]] = {
    "elo": {"k_player": 48, "k_team": 64},
    "ts": {"mu": 25.0, "sigma": 8.333, "beta": 4.16, "tau": 0.25},
    "os": {"mu": 25.0, "sigma": 3.5},
    "pl": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
    "tm": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
}

CONTEXT_WINDOW = 20
EPSILON = 0.001

# ---------------------------------------------------------------------------
# Feature definitions (matching the trained pipeline)
# ---------------------------------------------------------------------------
OPTUNA_BASE_FEATURES = [
    "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
    "player_elo_min1", "player_elo_min2",
    "player_gl_max1", "player_gl_max2",
    "player_gl_rd_avg1", "player_gl_rd_avg2",
    "player_ts_sigma_avg1", "player_ts_sigma_avg2",
    "player_os_sigma_avg1", "player_os_sigma_avg2",
    "player_pl_sigma_avg1", "player_pl_sigma_avg2",
    "player_tm_sigma_avg1", "player_tm_sigma_avg2",
]

ROLLING_FULL_FEATURES = [
    "t1_rolling_win_rate", "t2_rolling_win_rate",
    "t1_rolling_kills", "t2_rolling_kills",
    "t1_rolling_deaths", "t2_rolling_deaths",
    "t1_rolling_gd15", "t2_rolling_gd15",
    "t1_rolling_dpm", "t2_rolling_dpm",
    "t1_rolling_vspm", "t2_rolling_vspm",
    "t1_rolling_towers", "t2_rolling_towers",
    "t1_rolling_nashors", "t2_rolling_nashors",
    "t1_rolling_gold", "t2_rolling_gold",
    "t1_rolling_duration", "t2_rolling_duration",
]

RANK_PROB_FEATURES = [
    "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
]

ALL_FEATURES = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + [f + "_binom_series" for f in RANK_PROB_FEATURES]
# 20 + 20 + 6 = 46

# ---------------------------------------------------------------------------
# Default W20 stats (when no history yet)
# ---------------------------------------------------------------------------
_DEFAULT_TEAM_STATS = {
    "win_rate": 0.5,
    "kills": 12.0,
    "deaths": 12.0,
    "gd15": 0.0,
    "dpm": 1800.0,
    "vspm": 7.0,
    "towers": 5.0,
    "nashors": 0.5,
    "gold": 55000.0,
    "duration": 1800.0,
}


def _safe_stat(player: dict, key: str) -> float:
    return float(player.get("stats", {}).get(key, 0.0) or 0.0)


def _safe_team_stat(game: dict, stats_key: str, key: str) -> float:
    return float((game.get(stats_key, {}) or {}).get(key, 0.0) or 0.0)


def _average_history(history: deque | None) -> dict[str, float]:
    if not history:
        return dict(_DEFAULT_TEAM_STATS)
    rows = list(history)
    return {
        "win_rate": float(np.mean([r["win"] for r in rows])),
        "kills": float(np.mean([r["kills"] for r in rows])),
        "deaths": float(np.mean([r["deaths"] for r in rows])),
        "gd15": float(np.mean([r["gd15"] for r in rows])),
        "dpm": float(np.mean([r["dpm"] for r in rows])),
        "vspm": float(np.mean([r["vspm"] for r in rows])),
        "towers": float(np.mean([r["towers"] for r in rows])),
        "nashors": float(np.mean([r["nashors"] for r in rows])),
        "gold": float(np.mean([r["gold"] for r in rows])),
        "duration": float(np.mean([r["duration"] for r in rows])),
    }


def _update_w20_history(
    team_history: dict[str, deque],
    team_id: str,
    game: dict,
    window_size: int,
) -> None:
    is_team_1 = str(game.get("t1_id")) == str(team_id)
    win = bool(game.get("t1_win")) if is_team_1 else bool(game.get("t2_win"))
    players_key = "t1_players" if is_team_1 else "t2_players"
    stats_key = "t1_stats" if is_team_1 else "t2_stats"
    players = game.get(players_key, {}) or {}
    game_stats = {
        "win": float(win),
        "kills": sum(_safe_stat(p, "kills") for p in players.values()),
        "deaths": sum(_safe_stat(p, "deaths") for p in players.values()),
        "dpm": sum(_safe_stat(p, "dpm") for p in players.values()),
        "vspm": sum(_safe_stat(p, "vspm") for p in players.values()),
        "gd15": sum(_safe_stat(p, "gd@15") for p in players.values()),
        "towers": _safe_team_stat(game, stats_key, "towers"),
        "nashors": _safe_team_stat(game, stats_key, "nashors"),
        "gold": _safe_team_stat(game, stats_key, "gold"),
        "duration": float(game.get("game_duration") or 0.0),
    }
    if team_id not in team_history:
        team_history[team_id] = deque(maxlen=window_size)
    team_history[team_id].append(game_stats)


def _get_rolling_features(
    team_history: dict[str, deque],
    team1_id: str,
    team2_id: str,
) -> dict[str, float]:
    """Get W20 features for team1 vs team2 using current deque state."""
    t1_stats = _average_history(team_history.get(team1_id))
    t2_stats = _average_history(team_history.get(team2_id))
    features = {}
    for stat, val in t1_stats.items():
        features[f"t1_rolling_{stat}"] = val
    for stat, val in t2_stats.items():
        features[f"t2_rolling_{stat}"] = val
    return features


# ---------------------------------------------------------------------------
# Binomial series helpers
# ---------------------------------------------------------------------------
def series_probability(map_probability: np.ndarray, best_of: np.ndarray) -> np.ndarray:
    prob = np.clip(map_probability.astype(float), 0.001, 0.999)
    best_of_int = best_of.astype(int)
    result = prob.copy()
    for n_maps in (3, 5):
        needed = n_maps // 2 + 1
        series_prob = np.zeros_like(prob)
        for wins in range(needed, n_maps + 1):
            series_prob += (
                comb(n_maps, wins)
                * np.power(prob, wins)
                * np.power(1.0 - prob, n_maps - wins)
            )
        result = np.where(best_of_int == n_maps, series_prob, result)
    return np.clip(result, 0.001, 0.999)


def _add_binomial_features(features: dict[str, float]) -> dict[str, float]:
    """Add binomial series-adjusted ranking features to a feature dict."""
    result = features.copy()
    best_of = int(features.get("best_of", 1))
    for feature in RANK_PROB_FEATURES:
        prob = features.get(feature, 0.5)
        series_val = series_probability(
            np.array([prob]),
            np.array([best_of]),
        )[0]
        result[f"{feature}_binom_series"] = float(series_val)
    return result


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def load_games_for_match(connection, match_id: str) -> list[dict[str, Any]]:
    """Load golgg_games for a match, with player and team stats."""
    games_raw = connection.execute(
        """
        SELECT game_id, team1_id, team2_id, team1_name, team2_name,
               team1_win, team2_win, draw, game_duration,
               team1_stats_json, team2_stats_json
        FROM golgg_games
        WHERE match_id = ?
        ORDER BY CAST(game_id AS INTEGER) ASC
        """,
        (match_id,),
    ).fetchall()

    games = []
    for g in games_raw:
        game_id = str(g["game_id"])
        # Load players for each side
        t1_players_raw = connection.execute(
            """
            SELECT player_id, player_name, role, stats_json
            FROM golgg_game_players
            WHERE game_id = ? AND (team_id = ? OR (team_id IS NULL AND team_name = ?))
            """,
            (game_id, str(g["team1_id"]), str(g["team1_name"])),
        ).fetchall()
        t2_players_raw = connection.execute(
            """
            SELECT player_id, player_name, role, stats_json
            FROM golgg_game_players
            WHERE game_id = ? AND (team_id = ? OR (team_id IS NULL AND team_name = ?))
            """,
            (game_id, str(g["team2_id"]), str(g["team2_name"])),
        ).fetchall()

        def _parse_player_stats(raw) -> dict:
            try:
                return json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (json.JSONDecodeError, TypeError):
                return {}

        game_dict = {
            "game_id": game_id,
            "t1_id": str(g["team1_id"]),
            "t2_id": str(g["team2_id"]),
            "t1_win": bool(g["team1_win"]),
            "t2_win": bool(g["team2_win"]),
            "game_duration": g.get("game_duration"),
            "t1_players": {p["player_id"] or p["player_name"]: {"stats": _parse_player_stats(p["stats_json"])} for p in t1_players_raw},
            "t2_players": {p["player_id"] or p["player_name"]: {"stats": _parse_player_stats(p["stats_json"])} for p in t2_players_raw},
            "t1_stats": _parse_player_stats(g.get("team1_stats_json")),
            "t2_stats": _parse_player_stats(g.get("team2_stats_json")),
        }
        games.append(game_dict)
    return games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print(f"Chronological Backtest ({MODEL_VERSION})")
    print("=" * 60)

    # 1. Load pre-trained pipeline + calibrator
    print("\n[1/8] Loading pre-trained pipeline + calibrator...")
    if not PIPELINE_PATH.exists():
        print(f"ERROR: Pipeline not found at {PIPELINE_PATH}")
        print("Train it first with: python -m betting_app.scripts.train_thesis_model")
        sys.exit(1)
    pipeline = joblib.load(PIPELINE_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    print(f"       Pipeline loaded from {PIPELINE_PATH}")
    print(f"       Calibrator loaded from {CALIBRATOR_PATH}")

    # 2. Restore RatingManager from latest-full snapshot (cutoff 2026-05-28)
    print("\n[2/8] Restoring RatingManager from latest-full snapshot...")
    manager = _restore_rating_manager()
    if manager is None:
        print("ERROR: Could not restore rating state. Run rebuild_ratings.py --mode full --ratings-version latest-full --until 2026-05-28 first.")
        sys.exit(1)
    print(f"       RatingManager restored with {len(manager.systems)} systems")

    # 3. Load the 74 finished matches to backtest
    print("\n[3/8] Loading finished matches with GOL.GG mappings...")
    matches = _load_finished_matches()
    print(f"       {len(matches)} matches loaded")

    if not matches:
        print("No matches to backtest.")
        return

    # 4. Identify distinct teams from the matches for W20 history
    print("\n[4/8] Identifying teams for W20 history...")
    all_team_ids: set[str] = set()
    for m in matches:
        all_team_ids.add(m["team1_id"])
        all_team_ids.add(m["team2_id"])
    print(f"       {len(all_team_ids)} distinct teams")

    # 5. Initialize W20 deque from historical games for these teams
    print(f"\n[5/8] Initializing W20 deque from historical games (up to 2026-05-28)...")
    team_history = _init_w20_history_for_teams(all_team_ids)
    print(f"       W20 queue initialized with {len(team_history)} teams")

    # 6. Chronological backtest
    print(f"\n[6/8] Running chronological backtest ({len(matches)} matches)...")
    results = _run_backtest(matches, manager, team_history, pipeline, calibrator)

    # 7. Store predictions
    print(f"\n[7/8] Storing {len(results)} predictions in canonical_predictions...")
    stored = _store_predictions(results)
    print(f"       Stored {stored} predictions with version={MODEL_VERSION}")

    # 8. Evaluate
    print(f"\n[8/8] Evaluation:")
    _evaluate(results)

    print("\nDone!")


def _restore_rating_manager() -> RatingManager | None:
    """Restore a fully-hydrated RatingManager from the entity_ratings snapshot."""

    with connect() as connection:
        run = connection.execute(
            """
            SELECT id, data_cutoff_at
            FROM rating_runs
            WHERE ratings_version = ? AND status = 'completed'
            ORDER BY finished_at DESC NULLS LAST, id DESC
            LIMIT 1
            """,
            (RATINGS_VERSION,),
        ).fetchone()
        if not run:
            return None

        rows = connection.execute(
            """
            SELECT entity_type, entity_name, normalized_entity_name, team_name,
                   rating_system, rating_value, rd, sigma, games_played,
                   last_match_at, state_json
            FROM entity_ratings
            WHERE ratings_version = ?
            """,
            (RATINGS_VERSION,),
        ).fetchall()

    manager = RatingManager(RATING_SYSTEM_PARAMS)
    team_last_match: dict[str, str] = {}
    player_last_match: dict[str, str] = {}

    for row in rows:
        state = json.loads(row.get("state_json") or "{}")
        entity_type = str(row["entity_type"])
        system_name = str(row["rating_system"])
        system = manager.systems.get(system_name)
        if system is None:
            continue

        if entity_type == "team":
            entity_id = str(state.get("team_id") or row["normalized_entity_name"])
            if row.get("last_match_at"):
                team_last_match[entity_id] = str(row["last_match_at"])
        elif entity_type == "player":
            entity_id = str(state.get("player_id") or row["normalized_entity_name"])
            if row.get("last_match_at"):
                player_last_match[entity_id] = str(row["last_match_at"])
        else:
            continue

        rating = rating_from_row(system_name, system, row, state)
        if entity_type == "team":
            system.team_ratings[entity_id] = rating
        else:
            system.player_ratings[entity_id] = rating

    # Restore last_match_date for Glicko time decay
    gl = manager.systems["gl"]
    for team_id, last_at in team_last_match.items():
        gl.team_last_played[team_id] = date.fromisoformat(last_at[:10])
        manager.last_match_date[team_id] = date.fromisoformat(last_at[:10])
    for player_id, last_at in player_last_match.items():
        gl.player_last_played[player_id] = date.fromisoformat(last_at[:10])

    print(f"       Restored from rating_run id={run['id']}, cutoff={run.get('data_cutoff_at')}")
    print(f"       Teams: {len(manager.systems['elo'].team_ratings)}")
    print(f"       Players: {len(manager.systems['elo'].player_ratings)}")

    return manager




def _init_w20_history_for_teams(team_ids: set[str]) -> dict[str, deque]:
    """Initialize W20 deque for specific teams by processing their historical games up to 2026-05-28.

    Args:
        team_ids: Set of GOL.GG team IDs to build W20 history for.

    Returns:
        Dict mapping team_id -> deque of game stats.
    """
    cutoff_date = "2026-05-28"
    team_history: dict[str, deque] = {}

    for team_id in tqdm(team_ids, desc="W20 history init"):
        with connect() as connection:
            match_rows = connection.execute(
                """
                SELECT m.match_id, m.date, m.team1_id, m.team2_id, m.team1_name, m.team2_name
                FROM golgg_matches m
                WHERE m.date IS NOT NULL
                  AND m.date <= ?
                  AND COALESCE(m.draw, 0) = 0
                  AND (m.team1_id = ? OR m.team2_id = ?)
                ORDER BY m.date ASC, CAST(m.match_id AS INTEGER) ASC
                """,
                (cutoff_date, team_id, team_id),
            ).fetchall()

        for row in match_rows:
            match_id = str(row["match_id"])
            t1_id = str(row["team1_id"] or "")
            t2_id = str(row["team2_id"] or "")

            with connect() as conn:
                games_raw = conn.execute(
                    """
                    SELECT game_id, team1_id, team2_id, team1_name, team2_name,
                           team1_win, team2_win, game_duration,
                           team1_stats_json, team2_stats_json
                    FROM golgg_games
                    WHERE match_id = ?
                    ORDER BY CAST(game_id AS INTEGER) ASC
                    """,
                    (match_id,),
                ).fetchall()

            if not games_raw:
                continue

            if not t1_id or t1_id == "None":
                t1_id = str(games_raw[0]["team1_id"] or "")
            if not t2_id or t2_id == "None":
                t2_id = str(games_raw[0]["team2_id"] or "")
            if not t1_id or not t2_id:
                continue

            def _parse_player_stats(raw) -> dict:
                try:
                    return json.loads(raw) if isinstance(raw, str) else (raw or {})
                except (json.JSONDecodeError, TypeError):
                    return {}

            for g in games_raw:
                g_t1_id = str(g["team1_id"] or "")
                g_t2_id = str(g["team2_id"] or "")
                if g_t1_id != team_id and g_t2_id != team_id:
                    continue
                is_team1 = (g_t1_id == team_id)
                game_id = str(g["game_id"])

                with connect() as pconn:
                    players_raw = pconn.execute(
                        """
                        SELECT player_id, player_name, stats_json
                        FROM golgg_game_players
                        WHERE game_id = ? AND (team_id = ? OR (team_id IS NULL AND team_name = ?))
                        """,
                        (game_id,
                         str(g["team1_id"] if is_team1 else g["team2_id"]),
                         str(g["team1_name"] if is_team1 else g["team2_name"])),
                    ).fetchall()

                game = {
                    "t1_id": g_t1_id,
                    "t2_id": g_t2_id,
                    "t1_win": bool(g["team1_win"]),
                    "t2_win": bool(g["team2_win"]),
                    "game_duration": g.get("game_duration"),
                    "t1_players": {str(p["player_id"] or p["player_name"]): {"stats": _parse_player_stats(p["stats_json"])} for p in players_raw} if is_team1 else {},
                    "t2_players": {} if is_team1 else {str(p["player_id"] or p["player_name"]): {"stats": _parse_player_stats(p["stats_json"])} for p in players_raw},
                    "t1_stats": _parse_player_stats(g.get("team1_stats_json")),
                    "t2_stats": _parse_player_stats(g.get("team2_stats_json")),
                }
                _update_w20_history(team_history, team_id, game, CONTEXT_WINDOW)

    print(f"       W20 history built for {len(team_ids)} teams ({len(team_history)} have history)")
    return team_history


def _load_finished_matches() -> list[dict[str, Any]]:
    """Load finished canonical_matches with GOL.GG mappings.

    Returns matches sorted chronologically, with GOL.GG data pre-loaded.
    """
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT cm.id AS canonical_id, cm.normalized_team_a, cm.normalized_team_b,
                   cm.winner_name, cm.winner_normalized, cm.start_time_normalized AS match_date,
                   cm.best_of, cm.team_a_name, cm.team_b_name,
                   gmm.golgg_match_id,
                   gm.team1_id, gm.team2_id, gm.team1_name, gm.team2_name,
                   gm.team1_win, gm.team2_win,
                   gm.date AS golgg_date
            FROM canonical_matches cm
            JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
            JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
            WHERE cm.status = 'finished'
              AND cm.winner_name IS NOT NULL
            ORDER BY gm.date ASC, CAST(gm.match_id AS INTEGER) ASC
            """
        ).fetchall()

    matches = []
    for row in rows:
        norm_a = str(row["normalized_team_a"] or "")
        norm_b = str(row["normalized_team_b"] or "")
        winner_norm = str(row["winner_normalized"] or "")
        winner_is_a = (winner_norm.lower() == norm_a.lower()) if winner_norm else False

        matches.append({
            "canonical_id": int(row["canonical_id"]),
            "golgg_match_id": str(row["golgg_match_id"]),
            "team1_id": str(row["team1_id"]),
            "team2_id": str(row["team2_id"]),
            "team1_name": str(row["team1_name"] or ""),
            "team2_name": str(row["team2_name"] or ""),
            "team1_win": bool(row["team1_win"]),
            "date": row["golgg_date"],
            "best_of": int(row.get("best_of", row.get("best_of")) or 1),
            "winner_name": str(row["winner_name"] or ""),
            "winner_is_a": winner_is_a,
            "normalized_team_a": norm_a,
            "normalized_team_b": norm_b,
        })

    # Load games for all matches (in a fresh connection)
    with connect() as new_conn:
        for m in matches:
            m["games"] = load_games_for_match(new_conn, m["golgg_match_id"])

    return matches


def _run_backtest(
    matches: list[dict[str, Any]],
    manager: RatingManager,
    team_history: dict[str, deque],
    pipeline: Any,
    calibrator: Any,
) -> list[dict[str, Any]]:
    """Run chronological backtest, accumulating results."""
    from src.utils.golgg_schema import team1_id, team2_id, games as golgg_games

    results = []
    feature_log: list[dict[str, float]] = []

    for i, match in enumerate(tqdm(matches, desc="Backtest")):
        canonical_id = match["canonical_id"]
        t1_id = match["team1_id"]
        t2_id = match["team2_id"]
        match_date = match["date"]
        if isinstance(match_date, str):
            match_date = date.fromisoformat(match_date[:10])
        best_of = match["best_of"]
        games = match.get("games", [])

        # Get rosters from first game
        players_1 = list(games[0].get("t1_players", {}).keys()) if games else []
        players_2 = list(games[0].get("t2_players", {}).keys()) if games else []

        if not players_1 or not players_2:
            tqdm.write(f"       Skipping canonical_id={canonical_id}: no players found")
            continue

        # --- a. update_before_match ---
        manager.update_before_match(t1_id, t2_id, players_1, players_2, match_date)

        # --- b. predict_match() → rating features ---
        preds = manager.predict_match(t1_id, t2_id, players_1, players_2)

        # --- c. W20 features ---
        rolling = _get_rolling_features(team_history, t1_id, t2_id)

        # --- d. Combine features ---
        features: dict[str, float] = {}
        for feat in OPTUNA_BASE_FEATURES:
            features[feat] = float(preds.get(feat, 0.5))
        for feat in ROLLING_FULL_FEATURES:
            features[feat] = float(rolling.get(feat, 0.5))
        features["best_of"] = float(best_of)
        features = _add_binomial_features(features)

        # --- e. Run model with order symmetry ---
        feature_vector = np.array([[features[f] for f in ALL_FEATURES]])

        # Original orientation
        p_orig = np.clip(pipeline.predict_proba(feature_vector)[:, 1], EPSILON, 1.0 - EPSILON)

        # Swapped orientation (swap team references)
        swapped_features = features.copy()
        # Swap rolling features (t1 ↔ t2)
        for prefix in ["t1_", "t2_"]:
            opp_prefix = "t2_" if prefix == "t1_" else "t1_"
            for stat in ["win_rate", "kills", "deaths", "gd15", "dpm", "vspm", "towers", "nashors", "gold", "duration"]:
                swapped_features[f"{prefix}rolling_{stat}"] = features[f"{opp_prefix}rolling_{stat}"]
        # Swap main rating probability features (P(team1>team2) → P(team2>team1))
        for feat in RANK_PROB_FEATURES:
            swapped_features[feat] = 1.0 - features[feat]
        # Swap orientation-sensitive rating features
        swapped_features["player_elo_min1"] = features["player_elo_min2"]
        swapped_features["player_elo_min2"] = features["player_elo_min1"]
        swapped_features["player_gl_max1"] = features["player_gl_max2"]
        swapped_features["player_gl_max2"] = features["player_gl_max1"]
        swapped_features["player_gl_rd_avg1"] = features["player_gl_rd_avg2"]
        swapped_features["player_gl_rd_avg2"] = features["player_gl_rd_avg1"]
        for sys_name in ["ts", "os", "pl", "tm"]:
            swapped_features[f"player_{sys_name}_sigma_avg1"] = features[f"player_{sys_name}_sigma_avg2"]
            swapped_features[f"player_{sys_name}_sigma_avg2"] = features[f"player_{sys_name}_sigma_avg1"]
        # Recompute binomial series after team swap
        swapped_features["best_of"] = float(best_of)
        swapped_features = _add_binomial_features(swapped_features)

        swapped_vector = np.array([[swapped_features[f] for f in ALL_FEATURES]])
        p_swapped = np.clip(pipeline.predict_proba(swapped_vector)[:, 1], EPSILON, 1.0 - EPSILON)

        # Symmetrize: p_sym = avg(original, 1-swapped)
        p_sym = (p_orig[0] + (1.0 - p_swapped[0])) / 2.0
        p_sym = np.clip(p_sym, EPSILON, 1.0 - EPSILON)

        # Platt calibration
        from src.models.calibration import logit as logit_fn
        p_cal_logit = logit_fn(np.array([[p_sym]]), epsilon=EPSILON)
        p_calibrated = np.clip(
            calibrator.predict_proba(p_cal_logit)[:, 1],
            EPSILON, 1.0 - EPSILON,
        )[0]

        # y_true: did GOL.GG team1 win? Use golgg_matches data directly.
        team1_win_golgg = match.get("team1_win")
        if team1_win_golgg is not None:
            y_true = 1.0 if team1_win_golgg else 0.0
        else:
            # Fallback: use canonical winner_is_a (for older data without team1_win)
            golgg_team1_is_team_a = (normalize_team_name(match["team1_name"]).lower()
                                     == match["normalized_team_a"].lower())
            y_true = 1.0 if (match["winner_is_a"] == golgg_team1_is_team_a) else 0.0

        # --- f. Store result ---
        results.append({
            "canonical_id": canonical_id,
            "golgg_match_id": match["golgg_match_id"],
            "team1_id": t1_id,
            "team2_id": t2_id,
            "team1_name": match["team1_name"],
            "team2_name": match["team2_name"],
            "match_date": match_date.isoformat() if hasattr(match_date, 'isoformat') else str(match_date),
            "p_prob": float(p_calibrated),
            "y_true": float(y_true),
            "pipeline_prob": float(p_sym),
            "p_orig": float(p_orig[0]),
            "p_swapped": float(p_swapped[0]),
            "features": {k: round(float(v), 6) for k, v in features.items()},
        })

        # --- g. Update W20 deque with match games ---
        for game in games:
            _update_w20_history(team_history, t1_id, game, CONTEXT_WINDOW)
            _update_w20_history(team_history, t2_id, game, CONTEXT_WINDOW)

        # --- h. Update ratings with match results ---
        scores: list[int] = []
        for game in games:
            is_team1_team1 = str(game.get("t1_id")) == t1_id
            score_1 = int(bool(game.get("t1_win"))) if is_team1_team1 else int(bool(game.get("t2_win")))
            score_2 = 1 - score_1
            scores.append(score_1)
            manager.update_after_game(t1_id, t2_id, players_1, players_2, score_1, score_2)
        manager.update_after_match(t1_id, t2_id, players_1, players_2, scores)

    return results


def _store_predictions(results: list[dict[str, Any]]) -> int:
    """Store predictions in canonical_predictions table.

    Schema: canonical_match_id, model_name, model_version, prob_a, prob_b,
            predicted_at, diagnostics_json
    """
    now_utc = datetime.now(UTC).isoformat()
    stored = 0

    with transaction() as connection:
        for r in results:
            diag = {
                "golgg_match_id": r["golgg_match_id"],
                "team1_id": r["team1_id"],
                "team2_id": r["team2_id"],
                "team1_name": r["team1_name"],
                "team2_name": r["team2_name"],
                "pipeline_prob": r["pipeline_prob"],
                "p_orig": r["p_orig"],
                "p_swapped": r["p_swapped"],
                "y_true": r["y_true"],
                "features": r.get("features", {}),
            }
            try:
                connection.execute(
                    """
                    INSERT INTO canonical_predictions
                        (canonical_match_id, model_name, model_version,
                         prob_a, prob_b, predicted_at, diagnostics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["canonical_id"],
                        "Sym-Cal LR-ElasticNet-W20-Binomial",
                        MODEL_VERSION,
                        r["p_prob"],       # prob_a = P(team1 wins)
                        1.0 - r["p_prob"],  # prob_b
                        now_utc,
                        json.dumps(diag),
                    ),
                )
                stored += 1
            except Exception as e:
                tqdm.write(f"       Failed to store canonical_id={r['canonical_id']}: {e}")

    return stored


def _evaluate(results: list[dict[str, Any]]) -> None:
    """Print evaluation metrics."""
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

    y_true = np.array([r["y_true"] for r in results])
    y_prob = np.array([r["p_prob"] for r in results])

    n = len(y_true)
    correct = int(np.sum((y_prob >= 0.5) == y_true))
    accuracy = correct / n

    ll = float(log_loss(y_true, y_prob))
    auc_val = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    print(f"       Matches: {n}")
    print(f"       Accuracy: {accuracy:.4f} ({correct}/{n})")
    print(f"       LogLoss: {ll:.4f}")
    print(f"       AUC: {auc_val:.4f}")
    print(f"       Brier Score: {brier:.4f}")

    # Per-match detail
    print(f"\n       Top 10 most confident predictions (prob near 0 or 1):")
    sorted_by_conf = sorted(results, key=lambda r: abs(r["p_prob"] - 0.5))
    for r in sorted_by_conf[:10]:
        mark = "✓" if (r["p_prob"] >= 0.5) == (r["y_true"] == 1.0) else "✗"
        print(f"         {mark} canonical_id={r['canonical_id']}: P(team1)={r['p_prob']:.4f}, actual={r['y_true']:.0f}")


if __name__ == "__main__":
    main()
