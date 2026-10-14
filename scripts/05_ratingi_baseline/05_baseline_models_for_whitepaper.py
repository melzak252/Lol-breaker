"""Generate baseline model artefacts for whitepaper point 5.

This script evaluates rating-based baselines and market-implied probabilities on
consistent, leakage-aware snapshots. Rating predictions are taken from
``golgg_y_predicts.csv`` and are assumed to be produced before each match update
by ``03_generate_ratings.py``. Market probabilities are derived from average
opening and closing odds in ``odds.csv`` after removing the bookmaker margin.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src.visualization.thesis_style import (
    DARK_TEXT,
    PASTEL_BLUE,
    PASTEL_ORANGE,
    apply_thesis_style,
    clean_axis,
    palette as thesis_palette,
)

ASSET_DIR = ROOT_DIR / "docs" / "assets" / "baseline_point5"
RATINGS_PATH = ROOT_DIR / "data" / "golgg_y_predicts.csv"
ODDS_PATH = ROOT_DIR / "data" / "odds.csv"

RATING_MODELS = {
    "Team Elo": "team_elo",
    "Player Elo": "player_elo",
    "Team Glicko-2": "team_gl",
    "Player Glicko-2": "player_gl",
    "Team TrueSkill": "team_ts",
    "Player TrueSkill": "player_ts",
    "Team OpenSkill": "team_os",
    "Player OpenSkill": "player_os",
    "Team Plackett-Luce": "team_pl",
    "Player Plackett-Luce": "player_pl",
    "Team Thurstone-Mosteller": "team_tm",
    "Player Thurstone-Mosteller": "player_tm",
}

PLAYER_MODELS = [name for name in RATING_MODELS if name.startswith("Player")]
TEAM_MODELS = [name for name in RATING_MODELS if name.startswith("Team")]


def configure_style() -> None:
    """Configure a thesis-friendly visual style for generated plots."""
    apply_thesis_style(context="paper")


def safe_probability(series: pd.Series) -> pd.Series:
    """Clip probabilities to a numerically safe interval.

    Args:
        series: Probability-like values.

    Returns:
        Values converted to numeric and clipped to ``[0.001, 0.999]``.
    """
    return pd.to_numeric(series, errors="coerce").clip(0.001, 0.999)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate expected calibration error for binary probabilities.

    Args:
        y_true: Binary target array.
        y_prob: Predicted probabilities for class 1.
        n_bins: Number of equal-width probability bins.

    Returns:
        Expected calibration error.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        bin_accuracy = np.mean(y_true[in_bin])
        bin_confidence = np.mean(y_prob[in_bin])
        ece += abs(bin_accuracy - bin_confidence) * bin_weight
    return float(ece)


def no_vig_probability(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    """Convert two-way decimal odds into no-vig home/team-1 probability.

    Args:
        home_odds: Decimal odds for team 1.
        away_odds: Decimal odds for team 2.

    Returns:
        Margin-normalized probability for team 1.
    """
    p_home = 1.0 / pd.to_numeric(home_odds, errors="coerce")
    p_away = 1.0 / pd.to_numeric(away_odds, errors="coerce")
    return p_home / (p_home + p_away)


def normalize_team_name(name: object) -> str:
    """Normalize team names for conservative market-side alignment.

    Args:
        name: Raw team name from ratings or market data.

    Returns:
        Lowercase alphanumeric name used for exact comparison after removing
        whitespace and punctuation.
    """

    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def load_ratings() -> pd.DataFrame:
    """Load and normalize rating predictions."""
    ratings = pd.read_csv(RATINGS_PATH)
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")
    ratings["golgg_match_id"] = pd.to_numeric(
        ratings["golgg_match_id"], errors="coerce"
    ).astype("Int64")
    ratings["y_true"] = pd.to_numeric(ratings["y_true"], errors="coerce")
    for column in ["team1_name", "team2_name"]:
        if column not in ratings.columns:
            ratings[column] = pd.NA
    for column in RATING_MODELS.values():
        if column in ratings.columns:
            ratings[column] = safe_probability(ratings[column])
    return ratings


def load_market() -> pd.DataFrame:
    """Load market data and derive no-vig open/close probabilities."""
    odds = pd.read_csv(ODDS_PATH)
    odds["golgg_match_id"] = pd.to_numeric(
        odds["golgg_match_id"], errors="coerce"
    ).astype("Int64")
    odds["market_open"] = no_vig_probability(
        odds["avg_open_home"], odds["avg_open_away"]
    )
    odds["market_close"] = no_vig_probability(
        odds["avg_odds_home"], odds["avg_odds_away"]
    )
    return odds[
        [
            "golgg_match_id",
            "golgg_team1",
            "golgg_team2",
            "market_open",
            "market_close",
        ]
    ]


def orient_market_to_rating_side(data: pd.DataFrame) -> pd.DataFrame:
    """Orient market probabilities to the same team-1 side as rating outputs.

    The stored ``odds.csv`` was mapped before the GOL.GG export changed. For
    some match identifiers, the market-side team order is the reverse of the
    current canonical JSON side. This function uses team names to decide whether
    bookmaker probabilities should be kept or inverted.

    Args:
        data: Merged ratings and market dataset.

    Returns:
        Copy with ``market_open_aligned`` and ``market_close_aligned`` columns.
    """

    output = data.copy()
    rating_t1 = output["team1_name"].map(normalize_team_name)
    rating_t2 = output["team2_name"].map(normalize_team_name)
    market_t1 = output["golgg_team1"].map(normalize_team_name)
    market_t2 = output["golgg_team2"].map(normalize_team_name)

    same_side = (rating_t1 == market_t1) & (rating_t2 == market_t2)
    swapped_side = (rating_t1 == market_t2) & (rating_t2 == market_t1)

    output["market_side_alignment"] = np.select(
        [same_side, swapped_side], ["same", "swapped"], default="unknown"
    )
    output["market_open_aligned"] = np.select(
        [same_side, swapped_side],
        [output["market_open"], 1.0 - output["market_open"]],
        default=np.nan,
    )
    output["market_close_aligned"] = np.select(
        [same_side, swapped_side],
        [output["market_close"], 1.0 - output["market_close"]],
        default=np.nan,
    )
    return output


def add_simple_average(data: pd.DataFrame) -> pd.DataFrame:
    """Add simple average ensemble columns for rating probabilities.

    Args:
        data: Dataset containing rating probability columns.

    Returns:
        Dataset with simple ensemble columns appended.
    """
    output = data.copy()
    available_rating_cols = [col for col in RATING_MODELS.values() if col in output]
    available_player_cols = [RATING_MODELS[name] for name in PLAYER_MODELS]
    available_player_cols = [col for col in available_player_cols if col in output]
    output["simple_avg_all_ratings"] = output[available_rating_cols].mean(axis=1)
    output["simple_avg_player_ratings"] = output[available_player_cols].mean(axis=1)
    return output


def evaluate_model(
    data: pd.DataFrame, model_name: str, probability_column: str
) -> dict[str, float | int | str]:
    """Evaluate a binary probabilistic model.

    Args:
        data: Evaluation data containing ``y_true`` and probability column.
        model_name: Human-readable model name.
        probability_column: Column with class-1 probabilities.

    Returns:
        Dictionary with AUC, LogLoss, Brier, ECE and sample size.
    """
    subset = data[["y_true", probability_column]].dropna().copy()
    subset[probability_column] = safe_probability(subset[probability_column])
    y_true = subset["y_true"].astype(int).to_numpy()
    y_prob = subset[probability_column].to_numpy()
    return {
        "model": model_name,
        "probability_column": probability_column,
        "sample_size": int(len(subset)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def evaluate_models(
    data: pd.DataFrame, model_columns: dict[str, str]
) -> pd.DataFrame:
    """Evaluate a collection of probabilistic models."""
    rows = [evaluate_model(data, name, col) for name, col in model_columns.items()]
    return pd.DataFrame(rows).sort_values(["logloss", "auc"], ascending=[True, False])


def save_roc_plot(
    data: pd.DataFrame,
    model_columns: dict[str, str],
    output_path: Path,
    title: str,
    max_models: int = 8,
) -> None:
    """Save ROC curves for selected models."""
    plt.figure(figsize=(8, 7))
    palette = thesis_palette(min(max_models, len(model_columns)))
    for index, (name, column) in enumerate(list(model_columns.items())[:max_models]):
        subset = data[["y_true", column]].dropna()
        if subset.empty:
            continue
        y_true = subset["y_true"].astype(int).to_numpy()
        y_prob = safe_probability(subset[column]).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} ({auc_value:.3f})", color=palette[index])
    plt.plot([0, 1], [0, 1], linestyle="--", color=DARK_TEXT, alpha=0.6, label="Losowo")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_calibration_plot(
    data: pd.DataFrame,
    model_columns: dict[str, str],
    output_path: Path,
    title: str,
    max_models: int = 8,
) -> None:
    """Save reliability curves for selected models."""
    plt.figure(figsize=(8, 7))
    palette = thesis_palette(min(max_models, len(model_columns)))
    for index, (name, column) in enumerate(list(model_columns.items())[:max_models]):
        subset = data[["y_true", column]].dropna()
        if subset.empty:
            continue
        y_true = subset["y_true"].astype(int).to_numpy()
        y_prob = safe_probability(subset[column]).to_numpy()
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name, color=palette[index])
    plt.plot([0, 1], [0, 1], linestyle="--", color=DARK_TEXT, alpha=0.7, label="Idealna kalibracja")
    plt.title(title)
    plt.xlabel("Średnie przewidywane prawdopodobieństwo")
    plt.ylabel("Empiryczny odsetek zwycięstw")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_metric_bar(
    metrics: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    top_n: int = 10,
) -> None:
    """Save a compact ranking bar chart for a metric."""
    data = metrics.sort_values(metric, ascending=(metric != "auc")).head(top_n)
    min_value = float(data[metric].min())
    max_value = float(data[metric].max())
    value_range = max(max_value - min_value, 0.005)
    y_lower = max(0.0, min_value - value_range * 0.35)
    y_upper = max_value + value_range * 0.45

    plt.figure(figsize=(10.5, 5.5))
    ax = sns.barplot(
        data=data,
        x="model",
        y=metric,
        hue="model",
        palette=thesis_palette(len(data)),
        legend=False,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(y_lower, y_upper)
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    clean_axis(ax)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_player_team_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot matched player-vs-team AUC comparison for rating families."""
    rows = []
    for family in ["Elo", "Glicko-2", "TrueSkill", "OpenSkill", "Plackett-Luce", "Thurstone-Mosteller"]:
        for level in ["Player", "Team"]:
            model = f"{level} {family}"
            match = metrics[metrics["model"] == model]
            if not match.empty:
                rows.append({"family": family, "level": level, "auc": match.iloc[0]["auc"]})
    plot_data = pd.DataFrame(rows)
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=plot_data,
        x="family",
        y="auc",
        hue="level",
        palette={"Player": PASTEL_BLUE, "Team": PASTEL_ORANGE},
    )
    ax.set_title("Player-based vs team-based rating systems — AUC")
    ax.set_xlabel("Rodzina rankingu")
    ax.set_ylabel("AUC")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0.68, max(0.76, plot_data["auc"].max() + 0.01))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    clean_axis(ax)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_probability_distribution(
    data: pd.DataFrame, model_columns: dict[str, str], output_path: Path
) -> None:
    """Save distributions of selected baseline probabilities."""
    selected = {
        "Player Glicko-2": "player_gl",
        "Player Elo": "player_elo",
        "Player TrueSkill": "player_ts",
        "Market Open": "market_open",
        "Simple Avg Player": "simple_avg_player_ratings",
    }
    plt.figure(figsize=(10, 6))
    for name, column in selected.items():
        if column in data.columns:
            sns.kdeplot(
                safe_probability(data[column]).dropna(),
                label=name,
                linewidth=2,
            )
    plt.title("Rozkład predykcji bazowych")
    plt.xlabel("Prawdopodobieństwo zwycięstwa Team 1")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def summarize_uncertainty(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize available uncertainty/RD features from rating systems."""
    columns = [
        "team_gl_rd1",
        "team_gl_rd2",
        "player_gl_rd_avg1",
        "player_gl_rd_avg2",
        "team_ts_sigma1",
        "team_ts_sigma2",
        "player_ts_sigma_avg1",
        "player_ts_sigma_avg2",
        "team_os_sigma1",
        "team_os_sigma2",
        "player_os_sigma_avg1",
        "player_os_sigma_avg2",
    ]
    rows = []
    for column in columns:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        rows.append(
            {
                "feature": column,
                "count": int(values.shape[0]),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p90": float(values.quantile(0.9)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def save_uncertainty_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Save compact uncertainty feature summary plot."""
    if summary.empty:
        return
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(
        data=summary,
        x="feature",
        y="median",
        hue="feature",
        palette=thesis_palette(len(summary)),
        legend=False,
    )
    ax.set_title("Median uncertainty/RD dla systemów rankingowych")
    ax.set_xlabel("")
    ax.set_ylabel("Mediana")
    ax.tick_params(axis="x", rotation=45)
    clean_axis(ax)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def markdown_table(data: pd.DataFrame, columns: Iterable[str]) -> str:
    """Render selected columns as a simple Markdown table without tabulate."""
    selected = data[list(columns)].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.5f}")
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "|" + "|".join(["---"] * len(selected.columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_autogenerated_summary(
    full_metrics: pd.DataFrame,
    common_metrics: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> None:
    """Write a compact autogenerated Markdown summary for point 5."""
    path = ROOT_DIR / "docs" / "whitepaper" / "05_baseline_models_autogenerated.md"
    content = f"""---
type: autogenerated-analysis
tags:
  - whitepaper
  - baselines
  - ratings
  - market
project: inzynierka
date: 2026-04-30
source_script: scripts/05_baseline_models_for_whitepaper.py
---

# 5. Modele bazowe — wyniki autogenerated

> [!abstract]
> Ten plik zawiera automatycznie wygenerowane tabele dla rozdziału 5. Właściwa narracja znajduje się w `05_modele_bazowe_rankingi_i_rynek.md`.

## Rating-only sample, 2020+

{markdown_table(full_metrics.head(12), ["model", "sample_size", "auc", "logloss", "brier", "ece"])}

## Common market sample, 2020+

{markdown_table(common_metrics.head(16), ["model", "sample_size", "auc", "logloss", "brier", "ece"])}

## Uncertainty/RD summary

{markdown_table(uncertainty, ["feature", "count", "mean", "median", "p90", "max"])}
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run baseline evaluation and generate all point-5 artefacts."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    ratings = add_simple_average(load_ratings())
    market = load_market()
    merged = ratings.merge(market, on="golgg_match_id", how="left")
    merged = orient_market_to_rating_side(merged)

    full_sample = merged[(merged["date"] >= "2020-01-01") & merged["y_true"].notna()].copy()
    common_sample = full_sample[full_sample["market_open_aligned"].notna()].copy()

    rating_and_ensemble = {
        **RATING_MODELS,
        "Simple Avg All Ratings": "simple_avg_all_ratings",
        "Simple Avg Player Ratings": "simple_avg_player_ratings",
    }
    common_models = {
        **rating_and_ensemble,
        "Market Open": "market_open_aligned",
        "Market Close": "market_close_aligned",
    }

    full_metrics = evaluate_models(full_sample, rating_and_ensemble)
    common_metrics = evaluate_models(common_sample, common_models)
    uncertainty = summarize_uncertainty(full_sample)

    full_metrics.to_csv(ASSET_DIR / "baseline_rating_metrics_2020.csv", index=False)
    common_metrics.to_csv(ASSET_DIR / "baseline_common_market_metrics_2020.csv", index=False)
    uncertainty.to_csv(ASSET_DIR / "baseline_uncertainty_summary_2020.csv", index=False)

    selected_models = {
        "Player Glicko-2": "player_gl",
        "Player Elo": "player_elo",
        "Player TrueSkill": "player_ts",
        "Simple Avg Player": "simple_avg_player_ratings",
        "Market Open": "market_open_aligned",
        "Market Close": "market_close_aligned",
    }
    save_roc_plot(
        common_sample,
        selected_models,
        ASSET_DIR / "baseline_roc_curves_common_2020.png",
        "ROC — rankingi bazowe vs rynek, common sample 2020+",
    )
    save_calibration_plot(
        common_sample,
        selected_models,
        ASSET_DIR / "baseline_calibration_common_2020.png",
        "Reliability diagram — rankingi bazowe vs rynek, common sample 2020+",
    )
    save_metric_bar(
        common_metrics,
        ASSET_DIR / "baseline_auc_ranking_common_2020.png",
        "auc",
        "Ranking modeli bazowych według AUC, common sample 2020+",
    )
    save_metric_bar(
        common_metrics,
        ASSET_DIR / "baseline_logloss_ranking_common_2020.png",
        "logloss",
        "Ranking modeli bazowych według LogLoss, common sample 2020+",
    )
    save_player_team_comparison(
        full_metrics, ASSET_DIR / "baseline_player_vs_team_auc_2020.png"
    )
    save_probability_distribution(
        common_sample, common_models, ASSET_DIR / "baseline_probability_distributions.png"
    )
    save_uncertainty_plot(uncertainty, ASSET_DIR / "baseline_uncertainty_summary.png")
    write_autogenerated_summary(full_metrics, common_metrics, uncertainty)

    print("Baseline artefacts saved to:", ASSET_DIR)
    print("Common sample size:", len(common_sample))
    print("Market side alignment:")
    print(full_sample["market_side_alignment"].value_counts(dropna=False).to_string())
    print(common_metrics.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
