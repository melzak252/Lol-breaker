"""Test sensitivity of the final model to Team 1 / Team 2 ordering.

The final model predicts the probability that the GOL.GG Team 1 wins. This
diagnostic verifies whether the pipeline is stable when match sides are swapped:

1. train and test on the original orientation,
2. train and test on a deterministic 50% random side swap,
3. train on the original orientation, but evaluate each test chunk once in the
   original orientation and once in the swapped orientation converted back to the
   original Team-1 probability,
4. average both probabilities to force order-invariant predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import evaluate_probability_groups

from src.models.team_order import (
    swap_orientation,
    symmetrize_binary_probabilities,
)
from src.utils.module_loading import load_module_from_path


BASE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06ab_w20_binomial_all_models_bootstrap.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "team_order_sensitivity"
TARGET = "y_true"
RANDOM_SEED = 42
UPDATE_INTERVAL = 1000


def load_base_module() -> object:
    """Load the W20-Binomial model-family module."""

    return load_module_from_path(BASE_SCRIPT, "w20_binomial_models")


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all orientation variants."""

    return evaluate_probability_groups(predictions, ["variant"], target_column=TARGET)


def walk_forward_original_and_swapped_test(
    base_module: object,
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Train on original orientation and compare original vs swapped test rows."""

    clean = data.dropna(subset=features + [TARGET]).copy().sort_values("date").reset_index(drop=True)
    train_df = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    parts: list[pd.DataFrame] = []

    for fold, start in enumerate(tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc="test-time swap"), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        model = base_module.build_logistic_regression()
        model.fit(train_df[features], train_df[TARGET].astype(int))

        original_prob = np.clip(model.predict_proba(test_chunk[features])[:, 1], 0.001, 0.999)
        swapped_chunk = swap_orientation(
            test_chunk,
            features,
            base_module.RANK_PROB_FEATURES,
            np.ones(len(test_chunk), dtype=bool),
        )
        swapped_side_prob = np.clip(model.predict_proba(swapped_chunk[features])[:, 1], 0.001, 0.999)
        converted_prob = 1.0 - swapped_side_prob
        symmetrized_prob = symmetrize_binary_probabilities(original_prob, swapped_side_prob)

        for variant, probability in [
            ("Original orientation", original_prob),
            ("Original model + swapped test converted back", converted_prob),
            ("Order-symmetrized prediction", symmetrized_prob),
        ]:
            parts.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "fold": fold,
                        "golgg_match_id": test_chunk["golgg_match_id"].astype(str).to_numpy(),
                        "date": test_chunk["date"].to_numpy(),
                        TARGET: test_chunk[TARGET].astype(int).to_numpy(),
                        "y_prob": probability,
                        "abs_probability_delta": np.abs(original_prob - converted_prob),
                    }
                )
            )
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """Run team-order sensitivity diagnostics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_module = load_base_module()
    data, features = base_module.prepare_data()

    random_mask = np.random.default_rng(RANDOM_SEED).random(len(data)) < 0.5
    randomized_data = swap_orientation(
        data,
        features,
        base_module.RANK_PROB_FEATURES,
        random_mask,
    )
    all_swapped_data = swap_orientation(
        data,
        features,
        base_module.RANK_PROB_FEATURES,
        np.ones(len(data), dtype=bool),
    )

    original_predictions = base_module.walk_forward_model(
        data,
        features,
        "Original retrain",
        base_module.build_logistic_regression,
        mask_rate=0.0,
    )
    randomized_predictions = base_module.walk_forward_model(
        randomized_data,
        features,
        "Random 50% side swap retrain",
        base_module.build_logistic_regression,
        mask_rate=0.0,
    )
    all_swapped_predictions = base_module.walk_forward_model(
        all_swapped_data,
        features,
        "All rows side-swapped retrain",
        base_module.build_logistic_regression,
        mask_rate=0.0,
    )
    test_swap_predictions = walk_forward_original_and_swapped_test(base_module, data, features)

    predictions = pd.concat(
        [original_predictions, randomized_predictions, all_swapped_predictions, test_swap_predictions],
        ignore_index=True,
    )
    metrics = evaluate_predictions(predictions)
    delta_summary = (
        test_swap_predictions
        .groupby("variant")["abs_probability_delta"]
        .agg(["mean", "median", "max"])
        .reset_index()
    )

    predictions.to_csv(OUTPUT_DIR / "team_order_sensitivity_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "team_order_sensitivity_metrics.csv", index=False)
    delta_summary.to_csv(OUTPUT_DIR / "team_order_probability_delta.csv", index=False)

    print("Team-order sensitivity metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTest-time probability deltas:")
    print(delta_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
