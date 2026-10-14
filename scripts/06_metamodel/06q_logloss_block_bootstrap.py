"""Block-bootstrap significance tests for final LogLoss differences.

The script evaluates whether the final logistic-regression metamodel has a
stable LogLoss advantage over market and model benchmarks. The primary test is
a monthly block bootstrap: whole calendar months are sampled with replacement,
which is more conservative than sampling individual matches because it preserves
within-month dependence caused by leagues, patches, and tournament phases.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import binary_log_loss_vector as log_loss_vector
INPUT_PATH = (
    PROJECT_ROOT
    / "docs/assets/final_logistic_market_comparison/final_logistic_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs/assets/logloss_block_bootstrap"

RANDOM_SEED = 42
N_BOOTSTRAPS = 10_000
EPSILON = 1e-15

MODEL_COLUMN = "prob_logistic_full_context_w50"
COMPARISONS = {
    "vs Mkt Open": "market_open",
    "vs Mkt Close": "market_close",
    "vs Player Glicko-2": "player_glicko2",
    "vs LGBM-W50": "prob_full_context_metamodel",
    "vs LR-Player": "prob_logistic_player_only",
    "vs LR-Core": "prob_logistic_core_context_w50",
}


@dataclass(frozen=True)
class BootstrapResult:
    """Summary of a block-bootstrap comparison.

    Attributes:
        comparison: Name of the benchmark comparison.
        benchmark_column: Probability column used as benchmark.
        sample_size: Number of matches in the comparison.
        n_blocks: Number of calendar-month blocks.
        model_logloss: Mean LogLoss of the logistic model.
        benchmark_logloss: Mean LogLoss of the benchmark.
        observed_difference: Benchmark LogLoss minus model LogLoss.
        ci_low: Lower bound of the 95% bootstrap confidence interval.
        ci_high: Upper bound of the 95% bootstrap confidence interval.
        p_one_sided: Bootstrap p-value for H0: difference <= 0.
        p_two_sided: Two-sided bootstrap p-value around zero.
        significant_05: Whether the 95% CI excludes zero positively.
    """

    comparison: str
    benchmark_column: str
    sample_size: int
    n_blocks: int
    model_logloss: float
    benchmark_logloss: float
    observed_difference: float
    ci_low: float
    ci_high: float
    p_one_sided: float
    p_two_sided: float
    significant_05: bool


def prepare_dataset(path: Path) -> pd.DataFrame:
    """Load the final common sample and add monthly block identifiers.

    Args:
        path: Path to the final logistic-vs-market common-sample CSV.

    Returns:
        Data frame sorted chronologically with a ``month`` column.
    """

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def monthly_block_bootstrap(
    df: pd.DataFrame,
    benchmark_column: str,
    rng: np.random.Generator,
    n_bootstraps: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate LogLoss difference distribution via monthly block bootstrap.

    Args:
        df: Input data with ``month`` and probability columns.
        benchmark_column: Benchmark probability column.
        rng: NumPy random generator.
        n_bootstraps: Number of bootstrap replications.

    Returns:
        Tuple containing bootstrap differences and per-month observed summaries.
    """

    working = df.dropna(subset=[MODEL_COLUMN, benchmark_column, "y_true", "month"]).copy()
    working["model_loss"] = log_loss_vector(working["y_true"], working[MODEL_COLUMN])
    working["benchmark_loss"] = log_loss_vector(working["y_true"], working[benchmark_column])
    working["loss_diff"] = working["benchmark_loss"] - working["model_loss"]

    month_summary = (
        working.groupby("month", as_index=False)
        .agg(
            n_matches=("loss_diff", "size"),
            model_loss_sum=("model_loss", "sum"),
            benchmark_loss_sum=("benchmark_loss", "sum"),
            diff_sum=("loss_diff", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    month_summary["model_logloss"] = (
        month_summary["model_loss_sum"] / month_summary["n_matches"]
    )
    month_summary["benchmark_logloss"] = (
        month_summary["benchmark_loss_sum"] / month_summary["n_matches"]
    )
    month_summary["mean_difference"] = (
        month_summary["diff_sum"] / month_summary["n_matches"]
    )

    n_blocks = len(month_summary)
    diff_values = np.empty(n_bootstraps, dtype=float)
    block_indices = np.arange(n_blocks)
    n_matches = month_summary["n_matches"].to_numpy(dtype=float)
    diff_sums = month_summary["diff_sum"].to_numpy(dtype=float)

    for bootstrap_idx in range(n_bootstraps):
        sampled = rng.choice(block_indices, size=n_blocks, replace=True)
        diff_values[bootstrap_idx] = diff_sums[sampled].sum() / n_matches[sampled].sum()

    return diff_values, month_summary


def summarize_bootstrap(
    df: pd.DataFrame,
    comparison: str,
    benchmark_column: str,
    bootstrap_differences: np.ndarray,
    month_summary: pd.DataFrame,
) -> BootstrapResult:
    """Create a result summary for one bootstrap comparison.

    Args:
        df: Full input data frame.
        comparison: Human-readable comparison name.
        benchmark_column: Benchmark probability column.
        bootstrap_differences: Bootstrap distribution of LogLoss differences.
        month_summary: Per-month loss summary.

    Returns:
        A ``BootstrapResult`` instance.
    """

    working = df.dropna(subset=[MODEL_COLUMN, benchmark_column, "y_true", "month"])
    model_loss = log_loss_vector(working["y_true"], working[MODEL_COLUMN])
    benchmark_loss = log_loss_vector(working["y_true"], working[benchmark_column])
    observed_diff = float(benchmark_loss.mean() - model_loss.mean())
    ci_low, ci_high = np.percentile(bootstrap_differences, [2.5, 97.5])

    # One-sided test for the directional thesis hypothesis: the logistic model
    # has lower LogLoss, i.e. benchmark_loss - model_loss > 0.
    p_one_sided = float((np.sum(bootstrap_differences <= 0.0) + 1) / (len(bootstrap_differences) + 1))
    p_two_sided = float(min(1.0, 2.0 * min(
        (np.sum(bootstrap_differences <= 0.0) + 1) / (len(bootstrap_differences) + 1),
        (np.sum(bootstrap_differences >= 0.0) + 1) / (len(bootstrap_differences) + 1),
    )))

    return BootstrapResult(
        comparison=comparison,
        benchmark_column=benchmark_column,
        sample_size=len(working),
        n_blocks=len(month_summary),
        model_logloss=float(model_loss.mean()),
        benchmark_logloss=float(benchmark_loss.mean()),
        observed_difference=observed_diff,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_one_sided=p_one_sided,
        p_two_sided=p_two_sided,
        significant_05=bool(ci_low > 0.0),
    )


def plot_confidence_intervals(results: pd.DataFrame, output_path: Path) -> None:
    """Plot observed LogLoss differences with bootstrap confidence intervals.

    Args:
        results: Result data frame produced by this script.
        output_path: Target path for the PNG figure.
    """

    ordered = results.sort_values("observed_difference", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(ordered))
    colors = ["#6F8F72" if value else "#A68A64" for value in ordered["significant_05"]]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.errorbar(
        ordered["observed_difference"],
        y_positions,
        xerr=[
            ordered["observed_difference"] - ordered["ci_low"],
            ordered["ci_high"] - ordered["observed_difference"],
        ],
        fmt="none",
        ecolor="#4A5568",
        elinewidth=2.0,
        capsize=4,
    )
    ax.scatter(ordered["observed_difference"], y_positions, s=70, color=colors, zorder=3)
    ax.axvline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["comparison"], fontsize=11)
    ax.set_xlabel("Różnica LogLoss: benchmark - regresja logistyczna W50", fontsize=12)
    ax.set_title("Miesięczny block bootstrap różnic LogLoss", fontsize=15, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run monthly block-bootstrap tests for final model comparisons."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    df = prepare_dataset(INPUT_PATH)

    all_results: list[BootstrapResult] = []
    all_monthly: list[pd.DataFrame] = []
    bootstrap_columns: dict[str, np.ndarray] = {}

    for comparison, benchmark_column in COMPARISONS.items():
        differences, month_summary = monthly_block_bootstrap(
            df=df,
            benchmark_column=benchmark_column,
            rng=rng,
            n_bootstraps=N_BOOTSTRAPS,
        )
        result = summarize_bootstrap(
            df=df,
            comparison=comparison,
            benchmark_column=benchmark_column,
            bootstrap_differences=differences,
            month_summary=month_summary,
        )
        all_results.append(result)
        bootstrap_columns[comparison] = differences
        month_summary = month_summary.copy()
        month_summary["comparison"] = comparison
        month_summary["benchmark_column"] = benchmark_column
        all_monthly.append(month_summary)

    results_df = pd.DataFrame([result.__dict__ for result in all_results])
    results_df = results_df.sort_values("observed_difference", ascending=False).reset_index(drop=True)
    monthly_df = pd.concat(all_monthly, ignore_index=True)
    bootstrap_df = pd.DataFrame(bootstrap_columns)

    results_path = OUTPUT_DIR / "logloss_monthly_block_bootstrap_results.csv"
    monthly_path = OUTPUT_DIR / "logloss_monthly_observed_differences.csv"
    bootstrap_path = OUTPUT_DIR / "logloss_monthly_block_bootstrap_samples.csv"
    figure_path = OUTPUT_DIR / "logloss_monthly_block_bootstrap_ci.png"

    results_df.to_csv(results_path, index=False)
    monthly_df.to_csv(monthly_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    plot_confidence_intervals(results_df, figure_path)

    print("Monthly block-bootstrap LogLoss test completed.")
    print(f"Input sample: {INPUT_PATH}")
    print(f"Rows: {len(df):,}; months: {df['month'].nunique():,}; bootstraps: {N_BOOTSTRAPS:,}")
    print(f"Results: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
