"""Diagnose match-format and recency features in the context metamodel.

The diagnostic tests whether structural pre-match variables available in
``golgg_y_predicts.csv`` improve the current W50 metamodel: match format
(``BoN``) and team rest/recency variables (days since the last match for each
team and their difference).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWO_STAGE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06u_two_stage_ranking_context_metamodel.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "match_context_diagnostic"
MATCH_CONTEXT_FEATURES = ["BoN", "days_since_last_1", "days_since_last_2", "days_diff"]


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


def run_diagnostic() -> None:
    """Run the match-context diagnostic and save artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    two_stage = load_two_stage_module()
    data, helper = two_stage.prepare_data()

    full_features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES
    context_features = helper.ROLLING_FULL_FEATURES
    full_match_features = full_features + MATCH_CONTEXT_FEATURES
    context_match_features = context_features + MATCH_CONTEXT_FEATURES

    variants = [
        two_stage.walk_forward_one_stage(data, "LR-W50", full_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(data, "LR-W50 + match context", full_match_features, "l1", 0.30),
        two_stage.walk_forward_one_stage(
            data,
            "G2 + context",
            two_stage.G2_FEATURES + context_features,
            "l1",
            0.30,
        ),
        two_stage.walk_forward_one_stage(
            data,
            "G2 + context + match context",
            two_stage.G2_FEATURES + context_match_features,
            "l1",
            0.30,
        ),
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
                "TwoStage + match context",
                two_stage.RANK_PROB_FEATURES + two_stage.RANK_UNCERTAINTY_FEATURES,
                "l1",
                0.30,
                context_match_features,
                "l1",
                0.30,
            ),
        ),
    ]
    predictions = pd.concat(variants, ignore_index=True)
    metrics = two_stage.evaluate_predictions(predictions)
    bootstrap = two_stage.monthly_block_bootstrap(predictions, baseline="LR-W50")

    predictions.to_csv(OUTPUT_DIR / "match_context_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "match_context_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "match_context_bootstrap_vs_lr_w50.csv", index=False)

    print("\n=== MATCH CONTEXT DIAGNOSTIC ===")
    print(metrics.to_string(index=False))
    print("\n=== MONTHLY BLOCK BOOTSTRAP VS LR-W50 ===")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_diagnostic()
