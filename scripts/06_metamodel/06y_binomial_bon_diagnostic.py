"""Diagnose binomial best-of transformations for ranking probabilities.

Instead of adding ``BoN`` as a raw linear feature, this diagnostic transforms
rating-family probabilities as if they were map-win probabilities and computes
the implied probability of winning a Bo1/Bo3/Bo5 series under an independent
Bernoulli map model.
"""

from __future__ import annotations

import importlib.util
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWO_STAGE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06u_two_stage_ranking_context_metamodel.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "binomial_bon_diagnostic"


def load_two_stage_module() -> object:
    """Load the existing two-stage experiment module.

    Returns:
        Imported two-stage module object.
    """

    spec = importlib.util.spec_from_file_location("two_stage_context", TWO_STAGE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load two-stage module from {TWO_STAGE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def series_probability(map_probability: np.ndarray, best_of: np.ndarray) -> np.ndarray:
    """Convert map-win probability to best-of-series win probability.

    Args:
        map_probability: Probability of winning one independent map.
        best_of: Best-of format encoded as 1, 3, or 5.

    Returns:
        Probability of winning the full series.
    """

    p = np.clip(map_probability.astype(float), 0.001, 0.999)
    bon = best_of.astype(int)
    result = p.copy()
    for n in (3, 5):
        needed = n // 2 + 1
        probability = np.zeros_like(p)
        for wins in range(needed, n + 1):
            probability += comb(n, wins) * np.power(p, wins) * np.power(1.0 - p, n - wins)
        result = np.where(bon == n, probability, result)
    return np.clip(result, 0.001, 0.999)


def add_binomial_probability_features(data: pd.DataFrame, probability_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add binomial best-of transformed variants of probability features.

    Args:
        data: Input modeling frame containing ``BoN``.
        probability_features: Probability columns to transform.

    Returns:
        Enriched frame and list of generated columns.
    """

    enriched = data.copy()
    generated: list[str] = []
    bon = enriched["BoN"].fillna(1).astype(int).to_numpy()
    for feature in probability_features:
        column = f"{feature}_binom_series"
        enriched[column] = series_probability(enriched[feature].to_numpy(dtype=float), bon)
        generated.append(column)
    return enriched, generated


def run_diagnostic() -> None:
    """Run the binomial-BoN probability diagnostic and save artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    two_stage = load_two_stage_module()
    data, helper = two_stage.prepare_data()
    data, binomial_features = add_binomial_probability_features(data, two_stage.RANK_PROB_FEATURES)

    rank_uncertainty = two_stage.RANK_UNCERTAINTY_FEATURES
    context_features = helper.ROLLING_FULL_FEATURES
    original_full_features = helper.OPTUNA_BASE_FEATURES + context_features
    binomial_full_features = binomial_features + rank_uncertainty + context_features
    combined_full_features = helper.OPTUNA_BASE_FEATURES + binomial_features + context_features

    g2_binomial_features = ["player_gl_binom_series"] + [
        feature for feature in two_stage.G2_FEATURES if feature != "player_gl"
    ]

    variants = [
        two_stage.walk_forward_one_stage(data, "LR-W50", original_full_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "LR-W50 binomial ranks", binomial_full_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "LR-W50 original+binomial", combined_full_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "G2 + context", two_stage.G2_FEATURES + context_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "G2 binomial + context", g2_binomial_features + context_features, "l1", 0.30),
        two_stage.walk_forward_two_stage(
            data,
            two_stage.TwoStageVariant(
                "TwoStage",
                two_stage.RANK_PROB_FEATURES + rank_uncertainty,
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
                "TwoStage binomial ranks",
                binomial_features + rank_uncertainty,
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
                "TwoStage original+binomial",
                two_stage.RANK_PROB_FEATURES + binomial_features + rank_uncertainty,
                "l1",
                0.30,
                context_features,
                "l1",
                0.30,
            ),
        ),
    ]

    predictions = pd.concat(variants, ignore_index=True)
    metrics = two_stage.evaluate_predictions(predictions)
    bootstrap = two_stage.monthly_block_bootstrap(predictions, baseline="LR-W50")

    predictions.to_csv(OUTPUT_DIR / "binomial_bon_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "binomial_bon_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "binomial_bon_bootstrap_vs_lr_w50.csv", index=False)

    print("\n=== BINOMIAL BON DIAGNOSTIC ===")
    print(metrics.to_string(index=False))
    print("\n=== MONTHLY BLOCK BOOTSTRAP VS LR-W50 ===")
    print(bootstrap.to_string(index=False))
    print(f"\nGenerated binomial features: {binomial_features}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_diagnostic()
