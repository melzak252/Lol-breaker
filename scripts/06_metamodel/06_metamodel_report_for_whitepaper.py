"""Generate thesis-ready diagnostics for whitepaper point 6.

This script assumes that ``scripts/06_train_metamodel.py`` has already been
executed and that ``golgg_stacking_results.csv`` is up to date with the current
``golgg_y_predicts.csv``. It creates compact tables and plots under
``docs/assets/metamodel_point6`` for the metamodel chapter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "metamodel_point6"

TIER1_KEYWORDS = (
    "LEC",
    "LCS",
    "LTA",
    "LCK",
    "LPL",
    "European Championship",
    "Championship Series",
    "Champions Korea",
    "Pro League",
    "Mid-Season Invitational",
    "Mid Season Invitational",
    "First Stand",
    "World Championship",
    "Mistrzostwa Świata",
)


def configure_plot_style() -> None:
    """Configure a consistent report plotting style."""

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )



def evaluate_probability(df: pd.DataFrame, column: str, model_name: str) -> dict[str, object]:
    """Evaluate a probability column using classification and calibration metrics.

    Args:
        df: DataFrame containing ``y_true`` and the probability column.
        column: Name of the probability column.
        model_name: Human-readable model name.

    Returns:
        Dictionary with metrics.
    """

    valid = df.dropna(subset=["y_true", column]).copy()
    y_true = valid["y_true"].astype(int)
    y_prob = valid[column].clip(0.001, 0.999)
    return {
        "model": model_name,
        "probability_column": column,
        "sample_size": len(valid),
        "date_min": valid["date"].min().date().isoformat(),
        "date_max": valid["date"].max().date().isoformat(),
        "auc": roc_auc_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": calculate_ece(y_true, y_prob),
        "accuracy_0_5": accuracy_score(y_true, y_prob >= 0.5),
    }


def is_tier1(tournament: object) -> bool:
    """Classify a tournament as Tier 1 using the same keyword logic as EDA."""

    text = str(tournament)
    if "Pro League" in text and "Oceanic" not in text and "Continental" not in text:
        return True
    return any(keyword in text for keyword in TIER1_KEYWORDS)


def load_results() -> pd.DataFrame:
    """Load and enrich metamodel results with market/tournament metadata."""

    stacking = pd.read_csv(PROJECT_ROOT / "data" / "golgg_stacking_results.csv")
    stacking["golgg_match_id"] = stacking["golgg_match_id"].astype(str)
    stacking["date"] = pd.to_datetime(stacking["date"])
    odds = pd.read_csv(PROJECT_ROOT / "data" / "odds.csv", usecols=["golgg_match_id", "tournament"])
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    df = stacking.merge(odds, on="golgg_match_id", how="left")
    df["tier_segment"] = np.where(df["tournament"].apply(is_tier1), "Tier 1", "Regional / ERL")
    df["simple_avg_player_ratings"] = df[
        ["player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm"]
    ].mean(axis=1)
    df["simple_avg_all_ratings"] = df[
        [
            "player_elo",
            "player_gl",
            "player_ts",
            "player_os",
            "player_pl",
            "player_tm",
            "team_elo",
            "team_gl",
            "team_ts",
            "team_os",
            "team_pl",
            "team_tm",
        ]
    ].mean(axis=1)
    return df


def save_metric_bar(metrics: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    """Save a compact bar chart for a selected metric."""

    data = metrics.sort_values(metric, ascending=(metric == "logloss")).copy()
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=data, x="model", y=metric, palette="viridis", hue="model", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=9, padding=3)
    values = data[metric].to_numpy()
    margin = max((values.max() - values.min()) * 0.25, 0.002)
    ax.set_ylim(values.min() - margin, values.max() + margin)
    ax.set_title(title, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.35)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_calibration_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Save reliability curves for the main metamodel variants."""

    columns = {
        "Stage 1 LGBM": "s1_prob",
        "Stage 2 Isotonic": "metamodel_lgbm_isotonic",
        "Stage 2 Platt": "metamodel_lgbm_platt",
        "Simple Avg Player": "simple_avg_player_ratings",
    }
    plt.figure(figsize=(7.5, 7.0))
    for label, column in columns.items():
        valid = df.dropna(subset=[column, "y_true"])
        prob_true, prob_pred = calibration_curve(valid["y_true"], valid[column], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--", label="Idealna kalibracja")
    plt.title("Kalibracja metamodelu")
    plt.xlabel("Średnie przewidywane prawdopodobieństwo")
    plt.ylabel("Rzeczywisty udział zwycięstw")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate Stage 2 isotonic predictions by BoN, tier and year."""

    rows: list[dict[str, object]] = []
    segments = {
        "BoN": df["BoN"].astype(str),
        "tier_segment": df["tier_segment"],
        "year": df["date"].dt.year.astype(str),
    }
    for segment_name, values in segments.items():
        for value in sorted(values.dropna().unique()):
            part = df[values == value]
            if len(part) < 50 or part["y_true"].nunique() < 2:
                continue
            result = evaluate_probability(part, "metamodel_lgbm_isotonic", "Stage 2 Isotonic")
            rows.append(
                {
                    "segment_type": segment_name,
                    "segment_value": value,
                    "sample_size": result["sample_size"],
                    "auc": result["auc"],
                    "logloss": result["logloss"],
                    "brier": result["brier"],
                    "ece": result["ece"],
                }
            )
    segment_metrics = pd.DataFrame(rows)
    segment_metrics.to_csv(ASSETS_DIR / "metamodel_segment_metrics.csv", index=False)
    return segment_metrics


def save_blunder_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Save high-confidence error diagnostics.

    A high-confidence prediction is not expected to be perfect. For example,
    a match predicted at 80% still has a theoretical 20% probability of being
    wrong. Therefore this summary reports both observed and probability-implied
    expected blunders.

    Returns:
        DataFrame with observed and expected high-confidence errors by BoN.
    """

    probs = df["metamodel_lgbm_isotonic"]
    high_confidence = df[(probs > 0.8) | (probs < 0.2)].copy()
    high_confidence["expected_error_prob"] = np.where(
        high_confidence["metamodel_lgbm_isotonic"] > 0.8,
        1.0 - high_confidence["metamodel_lgbm_isotonic"],
        high_confidence["metamodel_lgbm_isotonic"],
    )
    high_confidence["is_blunder"] = np.where(
        high_confidence["metamodel_lgbm_isotonic"] > 0.8,
        high_confidence["y_true"] == 0,
        high_confidence["y_true"] == 1,
    )

    blunders = high_confidence[high_confidence["is_blunder"]].copy()
    blunders["residual"] = (blunders["y_true"] - blunders["metamodel_lgbm_isotonic"]).abs()

    summary = high_confidence.groupby("BoN").agg(
        high_confidence_predictions=("is_blunder", "size"),
        observed_blunders=("is_blunder", "sum"),
        expected_blunders=("expected_error_prob", "sum"),
    ).reset_index()
    summary["observed_blunder_rate_pct"] = (
        100 * summary["observed_blunders"] / summary["high_confidence_predictions"]
    )
    summary["theoretical_blunder_rate_pct"] = (
        100 * summary["expected_blunders"] / summary["high_confidence_predictions"]
    )
    summary["observed_minus_expected"] = summary["observed_blunders"] - summary["expected_blunders"]
    summary["observed_to_expected_ratio"] = (
        summary["observed_blunders"] / summary["expected_blunders"]
    )
    summary["share_pct"] = 100 * summary["observed_blunders"] / summary["observed_blunders"].sum()
    summary.to_csv(ASSETS_DIR / "metamodel_blunders_by_bon.csv", index=False)
    top = blunders.sort_values("residual", ascending=False).head(20)
    top[
        [
            "date",
            "golgg_match_id",
            "BoN",
            "tier_segment",
            "metamodel_lgbm_isotonic",
            "y_true",
            "residual",
        ]
    ].to_csv(ASSETS_DIR / "metamodel_top_blunders.csv", index=False)
    return summary


def main() -> None:
    """Generate all point 6 metamodel tables and plots."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    df = load_results()
    model_columns = {
        "Stage 2 Isotonic": "metamodel_lgbm_isotonic",
        "Stage 2 Platt": "metamodel_lgbm_platt",
        "Stage 1 LGBM ratings": "s1_prob",
        "Simple Avg Player Ratings": "simple_avg_player_ratings",
        "Simple Avg All Ratings": "simple_avg_all_ratings",
        "Player Glicko-2": "player_gl",
    }
    metrics = pd.DataFrame(
        [evaluate_probability(df, column, name) for name, column in model_columns.items()]
    ).sort_values("logloss")
    metrics.to_csv(ASSETS_DIR / "metamodel_variant_metrics.csv", index=False)
    save_metric_bar(metrics, "auc", ASSETS_DIR / "metamodel_variant_auc.png", "Metamodel vs proste baseline'y — AUC")
    save_metric_bar(metrics, "logloss", ASSETS_DIR / "metamodel_variant_logloss.png", "Metamodel vs proste baseline'y — LogLoss")
    save_calibration_plot(df, ASSETS_DIR / "metamodel_calibration_point6.png")
    save_segment_table(df)
    save_blunder_summary(df)
    print(f"Metamodel point 6 artefacts saved to: {ASSETS_DIR}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
