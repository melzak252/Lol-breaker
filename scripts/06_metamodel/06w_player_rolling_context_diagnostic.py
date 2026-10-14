"""Diagnose roster-aggregated rolling player statistics in the metamodel.

This diagnostic keeps the thesis' current team-level W50 context unchanged and
adds leakage-safe individual player histories aggregated to the current roster.
The goal is to test whether player-level rolling form statistics add predictive
value beyond rating systems, uncertainty descriptors, and team-level historical
context.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWO_STAGE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06u_two_stage_ranking_context_metamodel.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "player_rolling_context_diagnostic"
WINDOW_SIZE = 50

PLAYER_STAT_KEYS = {
    "kills": "kills",
    "deaths": "deaths",
    "assists": "assists",
    "dpm": "dpm",
    "gd15": "gd@15",
}
AGGREGATIONS = ["mean", "min", "max", "std"]


def load_two_stage_module() -> object:
    """Load the existing two-stage experiment module.

    Returns:
        Imported module object for reusing walk-forward evaluation utilities.
    """

    spec = importlib.util.spec_from_file_location("two_stage_context", TWO_STAGE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load two-stage module from {TWO_STAGE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert a possibly missing statistic to float.

    Args:
        value: Raw statistic value.
        default: Fallback used for missing or invalid values.

    Returns:
        Numeric statistic value.
    """

    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def player_identifier(player: dict[str, Any]) -> str | None:
    """Return a stable player identifier from a GOL.GG player record.

    Args:
        player: Raw player dictionary from one game.

    Returns:
        Player id/name string or ``None`` when the record is unusable.
    """

    raw_id = player.get("player_id") or player.get("player_name")
    if raw_id is None:
        return None
    return str(raw_id)


def iter_players(game: dict[str, Any], side: str) -> list[dict[str, Any]]:
    """Return player records for one side of a game.

    Args:
        game: Raw game dictionary.
        side: Either ``t1`` or ``t2``.

    Returns:
        List of player dictionaries.
    """

    raw_players = game.get(f"{side}_players", {}) or {}
    if isinstance(raw_players, dict):
        return list(raw_players.values())
    if isinstance(raw_players, list):
        return raw_players
    return []


def default_player_stats() -> dict[str, float]:
    """Return neutral defaults for players without prior professional history.

    Returns:
        Mapping of normalized player-statistic names to default values.
    """

    return {
        "kills": 2.4,
        "deaths": 2.4,
        "assists": 5.0,
        "dpm": 360.0,
        "gd15": 0.0,
    }


def average_player_history(history: deque[dict[str, float]] | None) -> dict[str, float]:
    """Average one player's rolling history.

    Args:
        history: Historical game-level player statistics.

    Returns:
        Averaged player statistics or neutral defaults for unseen players.
    """

    if not history:
        return default_player_stats()
    rows = list(history)
    return {stat: float(np.mean([row[stat] for row in rows])) for stat in PLAYER_STAT_KEYS}


def roster_feature_values(
    players: list[dict[str, Any]],
    player_history: dict[str, deque[dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Compute per-statistic roster aggregates from player histories.

    Args:
        players: Current roster player records.
        player_history: Rolling historical statistics keyed by player id.

    Returns:
        Nested mapping ``stat -> aggregate -> value``.
    """

    averaged_players: list[dict[str, float]] = []
    for player in players:
        player_id = player_identifier(player)
        averaged_players.append(average_player_history(player_history.get(player_id) if player_id else None))

    if not averaged_players:
        averaged_players = [default_player_stats()]

    features: dict[str, dict[str, float]] = {}
    for stat in PLAYER_STAT_KEYS:
        values = np.asarray([row[stat] for row in averaged_players], dtype=float)
        features[stat] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return features


def update_player_history(
    player_history: dict[str, deque[dict[str, float]]],
    players: list[dict[str, Any]],
) -> None:
    """Update rolling histories after a completed game.

    Args:
        player_history: Mutable history dictionary.
        players: Player records from one completed side-game.
    """

    for player in players:
        player_id = player_identifier(player)
        if player_id is None:
            continue
        raw_stats = player.get("stats", {}) or {}
        stats = {
            normalized: safe_float(raw_stats.get(source_key), default_player_stats()[normalized])
            for normalized, source_key in PLAYER_STAT_KEYS.items()
        }
        if player_id not in player_history:
            player_history[player_id] = deque(maxlen=WINDOW_SIZE)
        player_history[player_id].append(stats)


def generate_player_rolling_features(helper: object) -> tuple[pd.DataFrame, list[str]]:
    """Generate leakage-safe roster-aggregated player rolling features.

    Args:
        helper: Best-config helper module exposing GOL.GG schema utilities.

    Returns:
        Feature frame and list of generated feature column names.
    """

    with open(PROJECT_ROOT / "data" / "golgg_matches.json", "r", encoding="utf-8") as file:
        matches = json.load(file)
    matches.sort(key=lambda item: item["date"])

    player_history: dict[str, deque[dict[str, float]]] = {}
    rows: list[dict[str, object]] = []
    feature_names: list[str] = []

    for match in tqdm(matches, desc=f"Player rolling window {WINDOW_SIZE}"):
        match_games = list(helper.games(match))
        first_game = match_games[0] if match_games else {}
        t1_features = roster_feature_values(iter_players(first_game, "t1"), player_history)
        t2_features = roster_feature_values(iter_players(first_game, "t2"), player_history)

        row: dict[str, object] = {
            "golgg_match_id": str(match["match_id"]),
            "player_context_window": WINDOW_SIZE,
        }
        for stat in PLAYER_STAT_KEYS:
            for aggregation in AGGREGATIONS:
                t1_column = f"t1_player_rolling_{stat}_{aggregation}"
                t2_column = f"t2_player_rolling_{stat}_{aggregation}"
                row[t1_column] = t1_features[stat][aggregation]
                row[t2_column] = t2_features[stat][aggregation]
                if t1_column not in feature_names:
                    feature_names.extend([t1_column, t2_column])
            diff_column = f"diff_player_rolling_{stat}_mean"
            row[diff_column] = t1_features[stat]["mean"] - t2_features[stat]["mean"]
            if diff_column not in feature_names:
                feature_names.append(diff_column)
        rows.append(row)

        for game in match_games:
            update_player_history(player_history, iter_players(game, "t1"))
            update_player_history(player_history, iter_players(game, "t2"))

    return pd.DataFrame(rows), feature_names


def run_diagnostic() -> None:
    """Run the player rolling context diagnostic and save artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    two_stage = load_two_stage_module()
    data, helper = two_stage.prepare_data()
    player_features, player_feature_names = generate_player_rolling_features(helper)
    data = data.merge(player_features, on="golgg_match_id", how="inner").sort_values("date").reset_index(drop=True)

    full_features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES
    context_features = helper.ROLLING_FULL_FEATURES
    full_player_features = full_features + player_feature_names
    context_player_features = context_features + player_feature_names
    g2_context_features = two_stage.G2_FEATURES + context_features
    g2_player_features = two_stage.G2_FEATURES + context_player_features

    variants = [
        two_stage.walk_forward_one_stage(data, "LR-W50", full_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "LR-W50 + player rolling", full_player_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "G2 + context", g2_context_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "G2 + context + player rolling", g2_player_features, "l1", 0.30),
        two_stage.walk_forward_two_stage(
            data,
            two_stage.TwoStageVariant(
                "TwoStage",
                two_stage.RANK_PROB_FEATURES + two_stage.RANK_UNCERTAINTY_FEATURES,
                "l1",
                0.30,
                context_features,
                "l1",
                0.30,
            ),
        ),
        two_stage.walk_forward_two_stage(
            data,
            two_stage.TwoStageVariant(
                "TwoStage + player rolling",
                two_stage.RANK_PROB_FEATURES + two_stage.RANK_UNCERTAINTY_FEATURES,
                "l1",
                0.30,
                context_player_features,
                "l1",
                0.30,
            ),
        ),
    ]
    predictions = pd.concat(variants, ignore_index=True)
    metrics = two_stage.evaluate_predictions(predictions)
    bootstrap = two_stage.monthly_block_bootstrap(predictions, baseline="LR-W50")

    player_features.to_csv(OUTPUT_DIR / "player_rolling_features.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "player_rolling_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "player_rolling_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "player_rolling_bootstrap_vs_lr_w50.csv", index=False)

    print("\n=== PLAYER ROLLING CONTEXT DIAGNOSTIC ===")
    print(metrics.to_string(index=False))
    print("\n=== MONTHLY BLOCK BOOTSTRAP VS LR-W50 ===")
    print(bootstrap.to_string(index=False))
    print(f"\nGenerated player rolling features: {len(player_feature_names)}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_diagnostic()
