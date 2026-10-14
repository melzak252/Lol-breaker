"""Create a leakage-safe historical GD@15 context plot for the thesis EDA.

The plot compares the pre-match difference in rolling historical GD@15 between
Team 1 and Team 2 with the observed match result. It intentionally does not use
GD@15 from the current match/game, only values computed from earlier games.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.visualization.thesis_style import (
    DARK_TEXT,
    PASTEL_BLUE,
    PASTEL_GREEN,
    PASTEL_ORANGE,
    apply_thesis_style,
    clean_axis,
)

ROLLING_PATH = PROJECT_ROOT / "data" / "golgg_rolling_stats.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "golgg_y_predicts.csv"
ODDS_PATH = PROJECT_ROOT / "data" / "odds.csv"
FINAL_COMMON_SAMPLE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_two_stage_market_comparison"
    / "final_two_stage_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "eda_point4"

RANDOM_SEED = 42
START_DATE = "2024-01-01"


def load_historical_gd15_dataset() -> pd.DataFrame:
    """Load and prepare the model-sample historical GD@15 difference dataset.

    Returns:
        DataFrame with match ID, date, target, Team 1/Team 2 rolling GD@15 and
        their pre-match difference. The returned sample is restricted to the
        odds-mapped, 2024+ period used for the readable EDA visualization.
    """
    rolling = pd.read_csv(ROLLING_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    odds = pd.read_csv(ODDS_PATH, usecols=["golgg_match_id"])
    market = pd.read_csv(
        FINAL_COMMON_SAMPLE_PATH,
        usecols=["golgg_match_id", "market_open", "market_close"],
    )

    rolling["golgg_match_id"] = rolling["golgg_match_id"].astype(str)
    predictions["golgg_match_id"] = predictions["golgg_match_id"].astype(str)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    market["golgg_match_id"] = market["golgg_match_id"].astype(str)

    data = predictions[
        ["golgg_match_id", "date", "team1_name", "team2_name", "y_true"]
    ].merge(rolling, on="golgg_match_id", how="inner")
    data = data[data["golgg_match_id"].isin(set(odds["golgg_match_id"]))].copy()
    data = data.merge(market, on="golgg_match_id", how="inner")
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] >= pd.Timestamp(START_DATE)].copy()

    data["historical_gd15_diff"] = (
        data["t1_rolling_gd15"] - data["t2_rolling_gd15"]
    )
    data = data[data["historical_gd15_diff"] != 0].copy()
    data["historical_gd15_favorite"] = np.where(
        data["historical_gd15_diff"] > 0,
        "Team 1 higher historical GD@15",
        "Team 2 higher historical GD@15",
    )
    return data


def save_outputs(data: pd.DataFrame) -> None:
    """Save CSV summaries and a scatter/binned win-rate plot.

    Args:
        data: Prepared historical GD@15 dataset.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style(context="paper")

    output_csv = OUTPUT_DIR / "historical_gd15_diff_vs_result.csv"
    data.to_csv(output_csv, index=False)

    bins = np.arange(-3000, 3001, 500)
    data["gd15_diff_bin"] = pd.cut(data["historical_gd15_diff"], bins=bins)
    binned = (
        data.groupby("gd15_diff_bin", observed=True)
        .agg(
            matches=("y_true", "size"),
            team1_win_rate=("y_true", "mean"),
            market_open_mean=("market_open", "mean"),
            mean_gd15_diff=("historical_gd15_diff", "mean"),
        )
        .reset_index()
    )
    binned["win_rate_se"] = np.sqrt(
        binned["team1_win_rate"]
        * (1.0 - binned["team1_win_rate"])
        / binned["matches"].clip(lower=1)
    )
    binned["win_rate_ci_low"] = (binned["team1_win_rate"] - 1.96 * binned["win_rate_se"]).clip(
        lower=0.0
    )
    binned["win_rate_ci_high"] = (binned["team1_win_rate"] + 1.96 * binned["win_rate_se"]).clip(
        upper=1.0
    )
    binned.to_csv(OUTPUT_DIR / "historical_gd15_diff_binned_winrate.csv", index=False)

    summary = (
        data.assign(team1_higher_gd15=data["historical_gd15_diff"] > 0)
        .groupby("team1_higher_gd15", observed=True)
        .agg(matches=("y_true", "size"), team1_win_rate=("y_true", "mean"))
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "historical_gd15_diff_summary.csv", index=False)

    metrics = calculate_univariate_metrics(data)
    metrics.to_csv(OUTPUT_DIR / "historical_gd15_univariate_metrics.csv", index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    plot_data = data.copy()
    plot_data["y_jitter"] = plot_data["y_true"] + rng.normal(
        loc=0.0, scale=0.035, size=len(plot_data)
    )
    plot_data["y_jitter"] = plot_data["y_jitter"].clip(-0.08, 1.08)

    colors = np.where(plot_data["y_true"] == 1, PASTEL_BLUE, PASTEL_ORANGE)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.scatter(
        plot_data["historical_gd15_diff"],
        plot_data["y_jitter"],
        c=colors,
        alpha=0.34,
        s=18,
        linewidths=0,
    )
    ax.plot(
        binned["mean_gd15_diff"],
        binned["team1_win_rate"],
        color=DARK_TEXT,
        marker="o",
        linewidth=2.4,
        label="Win rate Team 1 w przedziałach GD@15",
    )
    ax.axhline(0.5, color=DARK_TEXT, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color=DARK_TEXT, linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_title("Historyczna przewaga GD@15 przed meczem a wynik meczu (2024+)", fontsize=15)
    ax.set_xlabel("Różnica średniego historycznego GD@15: Team 1 - Team 2")
    ax.set_ylabel("Wynik Team 1 / win rate w przedziale")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Team 1 przegrał", "50%", "Team 1 wygrał"])
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-0.12, 1.12)
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "historical_gd15_diff_vs_result.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    point_sizes = 35 + 0.55 * binned["matches"]
    scatter = ax.scatter(
        binned["mean_gd15_diff"],
        binned["team1_win_rate"],
        c=binned["team1_win_rate"],
        cmap="YlGnBu",
        vmin=0.25,
        vmax=0.80,
        s=point_sizes,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.9,
        label="Win rate Team 1 w przedziałach GD@15",
        zorder=5,
    )
    ax.errorbar(
        binned["mean_gd15_diff"],
        binned["team1_win_rate"],
        yerr=[
            binned["team1_win_rate"] - binned["win_rate_ci_low"],
            binned["win_rate_ci_high"] - binned["team1_win_rate"],
        ],
        fmt="none",
        ecolor=PASTEL_GREEN,
        elinewidth=1.4,
        capsize=3,
        alpha=0.75,
        label="95% CI win rate",
        zorder=4,
    )
    ax.plot(
        binned["mean_gd15_diff"],
        binned["team1_win_rate"],
        color=DARK_TEXT,
        linewidth=2.0,
        alpha=0.85,
        label="Trend win rate Team 1",
        zorder=4,
    )
    ax.axhline(0.5, color=DARK_TEXT, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color=DARK_TEXT, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title("Historyczne GD@15 a win rate Team 1 (2024+)", fontsize=15)
    ax.set_xlabel("Różnica średniego historycznego GD@15: Team 1 - Team 2")
    ax.set_ylabel("Rzeczywisty win rate Team 1")
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(0.0, 1.0)
    clean_axis(ax)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Win rate Team 1 w przedziale")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "historical_gd15_diff_vs_winrate.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "historical_gd15_diff_vs_market_open.png", dpi=180)
    plt.close(fig)

    positive = data[data["historical_gd15_diff"] > 0]
    non_positive = data[data["historical_gd15_diff"] <= 0]
    print(f"Saved {output_csv}")
    print(f"Sample size: {len(data):,}")
    print(
        "Team 1 higher historical GD@15: "
        f"n={len(positive):,}, win_rate={positive['y_true'].mean():.4f}"
    )
    print(
        "Team 1 lower historical GD@15: "
        f"n={len(non_positive):,}, win_rate={non_positive['y_true'].mean():.4f}"
    )
    print(metrics.to_string(index=False))


def calculate_univariate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple univariate diagnostics for historical GD@15 difference.

    The diagnostics quantify how much information is carried by the pre-match
    historical GD@15 difference alone. They are intentionally descriptive and do
    not replace the final multivariate model.

    Args:
        data: Prepared historical GD@15 dataset.

    Returns:
        One-row DataFrame with correlation and logistic-regression metrics.
    """
    x = data[["historical_gd15_diff"]].to_numpy()
    y = data["y_true"].astype(int).to_numpy()

    pearson = pearsonr(data["historical_gd15_diff"], y)
    spearman = spearmanr(data["historical_gd15_diff"], y)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED),
    )
    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]

    return pd.DataFrame(
        [
            {
                "matches": len(data),
                "positive_rate": float(np.mean(y)),
                "pearson_r": float(pearson.statistic),
                "pearson_p_value": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
                "univariate_auc": float(roc_auc_score(y, probabilities)),
                "univariate_logloss": float(log_loss(y, probabilities)),
                "univariate_brier": float(brier_score_loss(y, probabilities)),
                "univariate_accuracy": float(accuracy_score(y, probabilities >= 0.5)),
            }
        ]
    )


def main() -> None:
    """Run the historical GD@15 EDA figure generation."""
    data = load_historical_gd15_dataset()
    save_outputs(data)


if __name__ == "__main__":
    main()
