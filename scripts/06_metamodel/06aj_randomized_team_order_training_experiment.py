"""Evaluate randomized Team1/Team2 ordering during training.

This experiment tests whether the final W20-Binomial logistic-regression model
benefits from randomizing the orientation of teams in the training set. Test rows
remain in the original GOL.GG orientation, so the output probability is directly
comparable with the raw and symmetrized final-model diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import clip_probabilities, evaluate_probability_groups

from src.analysis.bootstrap import bootstrap_long_predictions_against_baseline
from src.models.calibration import expanding_platt_isotonic_calibration
from src.utils.module_loading import load_module_from_path
from src.visualization.thesis_style import DARK_TEXT, PASTEL_BLUE, PASTEL_RED, apply_thesis_style, clean_axis

BASE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06ab_w20_binomial_all_models_bootstrap.py"
SWAP_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06ag_team_order_sensitivity_analysis.py"
CALIBRATION_INPUT = (
    PROJECT_ROOT / "docs" / "assets" / "calibration_symmetry_diagnostic" / "calibration_symmetry_predictions.csv"
)
MARKET_SAMPLE = (
    PROJECT_ROOT / "docs" / "assets" / "final_model_market_comparison" / "final_model_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "randomized_team_order_training"

TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
EPSILON = 0.001
MIN_CALIBRATION_SAMPLES = 1000
N_BOOTSTRAPS = 10000
RANDOM_VARIANT = "Randomized-order training"
BASELINE_VARIANTS = [
    ("Original orientation", "raw", "Original raw"),
    ("Order-symmetrized prediction", "raw", "Symmetrized raw"),
    ("Order-symmetrized prediction", "platt_expanding", "Sym-Cal final"),
]


def evaluate(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Evaluate probability streams.

    Args:
        predictions: Prediction table with labels and probabilities.
        group_columns: Columns defining compared variants.

    Returns:
        Metric table sorted by LogLoss.
    """

    return evaluate_probability_groups(
        predictions,
        group_columns,
        target_column=TARGET,
        epsilon=EPSILON,
    )


def run_randomized_order_training(base_module: object, swap_module: object) -> pd.DataFrame:
    """Train on randomized team order and test on original orientation.

    Args:
        base_module: Module containing W20-Binomial data and model builders.
        swap_module: Module containing orientation-swapping utilities.

    Returns:
        Out-of-time predictions for the randomized-training model.
    """

    data, features = base_module.prepare_data()
    clean = data.dropna(subset=features + [TARGET]).copy().sort_values("date").reset_index(drop=True)
    train_df = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if train_df.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced an empty train or test subset.")

    parts: list[pd.DataFrame] = []
    for fold, start in enumerate(
        tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc="random-order train"),
        start=1,
    ):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        rng = np.random.default_rng(RANDOM_SEED + fold)
        train_swap_mask = rng.random(len(train_df)) < 0.5
        randomized_train = swap_module.swap_orientation(
            train_df,
            features,
            base_module.RANK_PROB_FEATURES,
            train_swap_mask,
        )

        model = base_module.build_logistic_regression()
        model.fit(randomized_train[features], randomized_train[TARGET].astype(int))
        probability = clip_probabilities(model.predict_proba(test_chunk[features])[:, 1], epsilon=EPSILON)

        parts.append(
            pd.DataFrame(
                {
                    "base_variant": RANDOM_VARIANT,
                    "fold": fold,
                    "golgg_match_id": test_chunk["golgg_match_id"].astype(str).to_numpy(),
                    "date": test_chunk["date"].to_numpy(),
                    TARGET: test_chunk[TARGET].astype(int).to_numpy(),
                    "y_prob": probability,
                    "training_swap_rate": float(train_swap_mask.mean()),
                }
            )
        )
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)
    return pd.concat(parts, ignore_index=True)


def expanding_calibrate_randomized(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply leakage-safe Platt and isotonic calibration to randomized training.

    Args:
        predictions: Raw randomized-training predictions.

    Returns:
        Raw and calibrated prediction streams.
    """

    return expanding_platt_isotonic_calibration(
        predictions,
        variant_value=RANDOM_VARIANT,
        variant_column="base_variant",
        target_column=TARGET,
        min_calibration_samples=MIN_CALIBRATION_SAMPLES,
        epsilon=EPSILON,
    )


def load_reference_predictions() -> pd.DataFrame:
    """Load raw and final Sym-Cal reference prediction streams.

    Returns:
        Reference predictions in the same schema as randomized calibration.
    """

    if not CALIBRATION_INPUT.exists():
        raise FileNotFoundError(f"Missing reference calibration file: {CALIBRATION_INPUT}")
    reference = pd.read_csv(CALIBRATION_INPUT, parse_dates=["date"])
    parts: list[pd.DataFrame] = []
    for base_variant, calibration, label in BASELINE_VARIANTS:
        selected = reference[
            (reference["base_variant"] == base_variant) & (reference["calibration"] == calibration)
        ].copy()
        selected["base_variant"] = label
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def add_market_subset_flag(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mark predictions that belong to the strict market-common sample.

    Args:
        predictions: Prediction table.

    Returns:
        Prediction table with ``in_market_common_sample`` flag.
    """

    if not MARKET_SAMPLE.exists():
        raise FileNotFoundError(f"Missing market sample file: {MARKET_SAMPLE}")
    market = pd.read_csv(MARKET_SAMPLE, usecols=["golgg_match_id"])
    market_ids = set(market["golgg_match_id"].astype(str))
    output = predictions.copy()
    output["in_market_common_sample"] = output["golgg_match_id"].astype(str).isin(market_ids)
    return output


def monthly_bootstrap_vs_baseline(predictions: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Compare variants against one baseline using monthly block bootstrap.

    Args:
        predictions: Long prediction table.
        baseline: Baseline variant label. Positive delta means variant is worse.

    Returns:
        Bootstrap comparison table.
    """

    return bootstrap_long_predictions_against_baseline(
        predictions,
        baseline_label=baseline,
        target_column=TARGET,
        n_bootstraps=N_BOOTSTRAPS,
        random_seed=RANDOM_SEED,
    )


def plot_logloss(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot LogLoss values for compared variants."""

    plot_data = metrics.sort_values("logloss", ascending=True).copy()
    apply_thesis_style(context="paper")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = [PASTEL_RED if "Randomized" in label else PASTEL_BLUE for label in plot_data["model_label"]]
    ax.barh(plot_data["model_label"], plot_data["logloss"], color=colors)
    ax.set_xlabel("LogLoss")
    ax.set_title("Wpływ losowania kolejności drużyn w treningu")
    clean_axis(ax, grid_axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_bootstrap(bootstrap: pd.DataFrame, output_path: Path) -> None:
    """Plot monthly-block bootstrap deltas against Sym-Cal final."""

    plot_data = bootstrap.sort_values("observed_delta_logloss_variant_minus_baseline", ascending=True).copy()
    y_pos = np.arange(len(plot_data))
    apply_thesis_style(context="paper")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvline(0.0, color=DARK_TEXT, linewidth=1.1, linestyle="--")
    ax.hlines(
        y=y_pos,
        xmin=plot_data["ci_lower_95"],
        xmax=plot_data["ci_upper_95"],
        color=DARK_TEXT,
        linewidth=1.4,
    )
    ax.scatter(
        plot_data["observed_delta_logloss_variant_minus_baseline"],
        y_pos,
        s=70,
        c=PASTEL_RED,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["variant"])
    ax.set_xlabel("Δ LogLoss względem Sym-Cal final")
    ax.set_title("Bootstrap blokowy: losowanie kolejności drużyn w treningu")
    clean_axis(ax, grid_axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run randomized-order training experiment."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_module = load_module_from_path(BASE_SCRIPT, "w20_binomial_models_random_order")
    swap_module = load_module_from_path(SWAP_SCRIPT, "team_order_swap_utils")

    randomized_raw = run_randomized_order_training(base_module, swap_module)
    randomized_calibrated = expanding_calibrate_randomized(randomized_raw)
    reference = load_reference_predictions()
    combined = pd.concat([reference, randomized_calibrated], ignore_index=True)
    combined = add_market_subset_flag(combined)
    combined["model_label"] = combined["base_variant"] + " + " + combined["calibration"]
    combined.loc[combined["base_variant"] == "Original raw", "model_label"] = "Original raw"
    combined.loc[combined["base_variant"] == "Symmetrized raw", "model_label"] = "Symmetrized raw"
    combined.loc[combined["base_variant"] == "Sym-Cal final", "model_label"] = "Sym-Cal final"

    full_metrics = evaluate(combined, ["model_label"])
    market_metrics = evaluate(combined[combined["in_market_common_sample"]], ["model_label"])
    bootstrap = monthly_bootstrap_vs_baseline(
        combined[combined["in_market_common_sample"]],
        baseline="Sym-Cal final",
    )

    randomized_raw.to_csv(OUTPUT_DIR / "randomized_team_order_training_raw_predictions.csv", index=False)
    randomized_calibrated.to_csv(OUTPUT_DIR / "randomized_team_order_training_calibrated_predictions.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "randomized_team_order_training_comparison_predictions.csv", index=False)
    full_metrics.to_csv(OUTPUT_DIR / "randomized_team_order_training_metrics_full.csv", index=False)
    market_metrics.to_csv(OUTPUT_DIR / "randomized_team_order_training_metrics_market_common.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "randomized_team_order_training_bootstrap_vs_symcal.csv", index=False)
    plot_logloss(market_metrics, OUTPUT_DIR / "randomized_team_order_training_logloss_market_common.png")
    plot_bootstrap(bootstrap, OUTPUT_DIR / "randomized_team_order_training_bootstrap_vs_symcal.png")

    print("Randomized team-order training metrics on market-common sample:")
    print(market_metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBootstrap vs Sym-Cal final on market-common sample:")
    print(bootstrap.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
