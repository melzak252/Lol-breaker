"""Monthly block-bootstrap tests for LR masking and ablation variants.

The script compares every masking/ablation variant from EXP-018 against the
baseline LR-W50 model. Whole calendar months are sampled with replacement to
preserve temporal dependence between matches from the same patch, league phase
or tournament period.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import binary_log_loss_vector as log_loss_vector

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.thesis_style import (
    DARK_TEXT,
    PASTEL_BLUE,
    PASTEL_ORANGE,
    apply_thesis_style,
    clean_axis,
)

INPUT_PATH = PROJECT_ROOT / "docs/assets/logistic_masking_ablation/logistic_masking_ablation_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs/assets/logistic_masking_ablation_bootstrap"
BASELINE_VARIANT = "LR-W50"
RANDOM_SEED = 42
N_BOOTSTRAPS = 10_000
EPSILON = 1e-15


@dataclass(frozen=True)
class BootstrapResult:
    """Summary of one bootstrap comparison against LR-W50."""

    variant: str
    sample_size: int
    n_blocks: int
    baseline_logloss: float
    variant_logloss: float
    observed_delta: float
    ci_low: float
    ci_high: float
    p_variant_worse_one_sided: float
    p_two_sided: float
    significant_05: bool


def prepare_wide_predictions(path: Path) -> pd.DataFrame:
    """Load EXP-018 predictions and pivot variants to columns.

    Args:
        path: Long-format prediction CSV from EXP-018.

    Returns:
        Wide data frame with one row per match and one probability column per variant.
    """

    long_df = pd.read_csv(path)
    long_df["date"] = pd.to_datetime(long_df["date"])
    id_cols = ["golgg_match_id", "date", "year", "BoN", "y_true"]
    wide = long_df.pivot_table(index=id_cols, columns="variant", values="y_prob", aggfunc="first")
    wide = wide.reset_index().sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    wide["month"] = wide["date"].dt.to_period("M").astype(str)
    if BASELINE_VARIANT not in wide.columns:
        raise ValueError(f"Missing baseline variant: {BASELINE_VARIANT}")
    return wide


def monthly_block_bootstrap(
    df: pd.DataFrame,
    variant: str,
    rng: np.random.Generator,
    n_bootstraps: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Bootstrap LogLoss deltas between a variant and LR-W50.

    Args:
        df: Wide prediction table.
        variant: Variant probability column.
        rng: Random number generator.
        n_bootstraps: Number of bootstrap repetitions.

    Returns:
        Bootstrap deltas and monthly observed summary.
    """

    working = df.dropna(subset=[BASELINE_VARIANT, variant, "y_true", "month"]).copy()
    working["baseline_loss"] = log_loss_vector(working["y_true"], working[BASELINE_VARIANT])
    working["variant_loss"] = log_loss_vector(working["y_true"], working[variant])
    working["loss_delta"] = working["variant_loss"] - working["baseline_loss"]

    month_summary = (
        working.groupby("month", as_index=False)
        .agg(
            n_matches=("loss_delta", "size"),
            baseline_loss_sum=("baseline_loss", "sum"),
            variant_loss_sum=("variant_loss", "sum"),
            delta_sum=("loss_delta", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    month_summary["baseline_logloss"] = month_summary["baseline_loss_sum"] / month_summary["n_matches"]
    month_summary["variant_logloss"] = month_summary["variant_loss_sum"] / month_summary["n_matches"]
    month_summary["mean_delta"] = month_summary["delta_sum"] / month_summary["n_matches"]

    n_blocks = len(month_summary)
    block_indices = np.arange(n_blocks)
    n_matches = month_summary["n_matches"].to_numpy(dtype=float)
    delta_sums = month_summary["delta_sum"].to_numpy(dtype=float)
    deltas = np.empty(n_bootstraps, dtype=float)
    for bootstrap_idx in range(n_bootstraps):
        sampled = rng.choice(block_indices, size=n_blocks, replace=True)
        deltas[bootstrap_idx] = delta_sums[sampled].sum() / n_matches[sampled].sum()
    return deltas, month_summary


def summarize_result(
    df: pd.DataFrame,
    variant: str,
    deltas: np.ndarray,
    month_summary: pd.DataFrame,
) -> BootstrapResult:
    """Summarize one variant-vs-baseline bootstrap comparison."""

    working = df.dropna(subset=[BASELINE_VARIANT, variant, "y_true", "month"])
    baseline_loss = log_loss_vector(working["y_true"], working[BASELINE_VARIANT])
    variant_loss = log_loss_vector(working["y_true"], working[variant])
    observed_delta = float(variant_loss.mean() - baseline_loss.mean())
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_worse = float((np.sum(deltas <= 0.0) + 1) / (len(deltas) + 1))
    p_two_sided = float(
        min(
            1.0,
            2.0
            * min(
                (np.sum(deltas <= 0.0) + 1) / (len(deltas) + 1),
                (np.sum(deltas >= 0.0) + 1) / (len(deltas) + 1),
            ),
        )
    )
    return BootstrapResult(
        variant=variant,
        sample_size=len(working),
        n_blocks=len(month_summary),
        baseline_logloss=float(baseline_loss.mean()),
        variant_logloss=float(variant_loss.mean()),
        observed_delta=observed_delta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_variant_worse_one_sided=p_worse,
        p_two_sided=p_two_sided,
        significant_05=bool(ci_low > 0.0),
    )


def plot_confidence_intervals(results: pd.DataFrame, output_path: Path) -> None:
    """Plot bootstrap deltas and confidence intervals."""

    ordered = results.sort_values("observed_delta", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    apply_thesis_style(context="paper")
    colors = [PASTEL_ORANGE if value else PASTEL_BLUE for value in ordered["significant_05"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.errorbar(
        ordered["observed_delta"],
        y_positions,
        xerr=[ordered["observed_delta"] - ordered["ci_low"], ordered["ci_high"] - ordered["observed_delta"]],
        fmt="none",
        ecolor=DARK_TEXT,
        elinewidth=2.0,
        capsize=4,
    )
    ax.scatter(ordered["observed_delta"], y_positions, s=70, color=colors, zorder=3)
    ax.axvline(0.0, color=DARK_TEXT, linestyle="--", linewidth=1.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["variant"], fontsize=10)
    ax.set_xlabel("Δ LogLoss: wariant - LR-W50", fontsize=12)
    ax.set_title("Miesięczny block bootstrap wariantów maskingu", fontsize=15, weight="bold")
    clean_axis(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run block bootstrap for EXP-018 masking variants."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    df = prepare_wide_predictions(INPUT_PATH)
    variants = [column for column in df.columns if column not in {"golgg_match_id", "date", "year", "BoN", "y_true", "month", BASELINE_VARIANT}]

    all_results: list[BootstrapResult] = []
    all_monthly: list[pd.DataFrame] = []
    bootstrap_columns: dict[str, np.ndarray] = {}
    for variant in variants:
        deltas, month_summary = monthly_block_bootstrap(df, variant, rng, N_BOOTSTRAPS)
        result = summarize_result(df, variant, deltas, month_summary)
        all_results.append(result)
        bootstrap_columns[variant] = deltas
        monthly = month_summary.copy()
        monthly["variant"] = variant
        all_monthly.append(monthly)

    results_df = pd.DataFrame([result.__dict__ for result in all_results])
    results_df = results_df.sort_values("observed_delta", ascending=True).reset_index(drop=True)
    monthly_df = pd.concat(all_monthly, ignore_index=True)
    bootstrap_df = pd.DataFrame(bootstrap_columns)

    results_path = OUTPUT_DIR / "masking_monthly_block_bootstrap_results.csv"
    monthly_path = OUTPUT_DIR / "masking_monthly_observed_differences.csv"
    bootstrap_path = OUTPUT_DIR / "masking_monthly_block_bootstrap_samples.csv"
    figure_path = OUTPUT_DIR / "masking_monthly_block_bootstrap_ci.png"

    results_df.to_csv(results_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    plot_confidence_intervals(results_df, figure_path)

    print("Masking monthly block-bootstrap completed.")
    print(f"Rows: {len(df):,}; months: {df['month'].nunique():,}; bootstraps: {N_BOOTSTRAPS:,}")
    print(f"Results: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
