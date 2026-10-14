#!/usr/bin/env python3
"""CLI script for training, evaluating, and smoke-testing the player statistics generative model."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.models.player_statistics_model import (
    PlayerStatsDistributionModel,
    load_player_statistics_model,
    train_player_statistics_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_player_statistics")


def run_smoke(output_dir: str) -> None:
    """Run smoke test: verify model initialization, sampling, and conservation invariants."""
    logger.info("Executing smoke test...")
    model = load_player_statistics_model(output_dir)
    rng = np.random.default_rng(42)

    roster_a = ["Kiin", "Canyon", "Chovy", "Peyz", "Lehends"]
    roster_b = ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]

    t0 = time.time()
    n_samples = 100
    pentas = 0
    total_duration = 0

    for i in range(n_samples):
        winner = "A" if (i % 2 == 0) else "B"
        res = model.sample_game_stats(
            team_a_roster=roster_a,
            team_b_roster=roster_b,
            winner=winner,
            rng=rng,
        )
        total_duration += res.duration_seconds
        if res.has_pentakill:
            pentas += 1

        # Check conservation laws
        a_kills = sum(s.kills for k, s in res.player_stats.items() if k.startswith("A_"))
        b_kills = sum(s.kills for k, s in res.player_stats.items() if k.startswith("B_"))
        a_deaths = sum(s.deaths for k, s in res.player_stats.items() if k.startswith("A_"))
        b_deaths = sum(s.deaths for k, s in res.player_stats.items() if k.startswith("B_"))

        assert a_kills == res.team_a_kills, f"A kills mismatch: {a_kills} != {res.team_a_kills}"
        assert b_kills == res.team_b_kills, f"B kills mismatch: {b_kills} != {res.team_b_kills}"
        assert a_deaths == res.team_b_kills, f"A deaths mismatch: {a_deaths} != {res.team_b_kills}"
        assert b_deaths == res.team_a_kills, f"B deaths mismatch: {b_deaths} != {res.team_a_kills}"

    elapsed = time.time() - t0
    avg_dur = (total_duration / n_samples) / 60.0
    logger.info(
        "Smoke test passed: %d games sampled in %.3fs (%.1f games/s). Avg duration: %.1fm. Pentas: %d.",
        n_samples,
        elapsed,
        n_samples / max(0.001, elapsed),
        avg_dur,
        pentas,
    )


def run_train(as_of_date: str, output_dir: str, min_games: int) -> None:
    """Train and serialize hierarchical player statistics model."""
    logger.info("Training player statistics model (as-of: %s, min-games: %d)...", as_of_date, min_games)
    t0 = time.time()
    meta = train_player_statistics_model(
        as_of_date=as_of_date,
        output_dir=output_dir,
        min_games=min_games,
    )
    elapsed = time.time() - t0
    logger.info("Training complete in %.2fs. Artifacts saved to %s.", elapsed, output_dir)
    print(json.dumps(meta, indent=2))


def run_evaluate(output_dir: str) -> None:
    """Evaluate serialized model and report empirical parameter distributions."""
    meta_path = os.path.join(output_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        logger.warning("No model metadata found at %s. Running smoke instead.", meta_path)
        run_smoke(output_dir)
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info("--- Model Evaluation Summary ---")
    logger.info("Model: %s (v%s)", meta.get("model_name"), meta.get("version"))
    logger.info("As of date: %s", meta.get("as_of_date"))
    logger.info("Sample games: %d", meta.get("sample_games", 0))
    logger.info("Sample player records: %d", meta.get("sample_player_records", 0))
    logger.info("Player profiles fitted: %d", meta.get("num_player_profiles", 0))
    logger.info("Role kill shares: %s", meta.get("role_kill_shares"))
    logger.info("Role death shares: %s", meta.get("role_death_shares"))
    logger.info("Objective rates: %s", meta.get("objective_rates"))
    logger.info("SHA-256 digest: %s", meta.get("sha256"))
    run_smoke(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate player statistics generative model.")
    parser.add_argument("--train", action="store_true", help="Fit priors and serialize model artifacts.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate fitted priors and run benchmark checks.")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test sampling.")
    parser.add_argument("--as-of-date", type=str, default="2026-01-01", help="Temporal cutoff date (YYYY-MM-DD).")
    parser.add_argument("--output-dir", type=str, default="data/06_models/player_statistics", help="Output artifact directory.")
    parser.add_argument("--min-games", type=int, default=10, help="Minimum games for player adjustment.")

    args = parser.parse_args()

    # Default to smoke if no mode selected
    if not args.train and not args.evaluate and not args.smoke:
        args.smoke = True

    if args.train:
        run_train(args.as_of_date, args.output_dir, args.min_games)
    elif args.evaluate:
        run_evaluate(args.output_dir)
    elif args.smoke:
        run_smoke(args.output_dir)


if __name__ == "__main__":
    main()
