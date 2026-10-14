"""Analyze correlation and VIF among rating signals used by the metamodel.

The analysis is restricted to the odds-mapped evaluation period to keep the
diagnostics aligned with the thesis model-vs-market sample. It focuses on
rating probability signals, because these features are the most naturally
redundant: several rating systems attempt to estimate the same underlying team
strength from similar match histories.
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
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.visualization.thesis_style import PASTEL_BLUE, apply_thesis_style, clean_axis

ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "metamodel_correlation_vif"

RATING_COLUMNS = [
    "team_elo",
    "team_gl",
    "team_ts",
    "team_os",
    "team_pl",
    "team_tm",
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]

LABELS = {
    "team_elo": "Team Elo",
    "team_gl": "Team Glicko-2",
    "team_ts": "Team TrueSkill",
    "team_os": "Team OpenSkill",
    "team_pl": "Team PL",
    "team_tm": "Team TM",
    "player_elo": "Player Elo",
    "player_gl": "Player Glicko-2",
    "player_ts": "Player TrueSkill",
    "player_os": "Player OpenSkill",
    "player_pl": "Player PL",
    "player_tm": "Player TM",
}


def load_odds_sample() -> pd.DataFrame:
    """Load rating predictions restricted to the odds-mapped period.

    Returns:
        Chronologically filtered DataFrame with rating probability columns.
    """

    predictions = pd.read_csv(PROJECT_ROOT / "data" / "golgg_y_predicts.csv")
    odds = pd.read_csv(PROJECT_ROOT / "data" / "odds.csv", usecols=["golgg_match_id"])
    predictions["golgg_match_id"] = predictions["golgg_match_id"].astype(str)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    data = predictions.merge(odds.drop_duplicates(), on="golgg_match_id", how="inner")
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] >= pd.Timestamp("2021-01-01")].copy()
    return data.dropna(subset=RATING_COLUMNS).sort_values("date").reset_index(drop=True)


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate variance inflation factors for numeric features.

    Args:
        frame: Feature matrix without missing values.

    Returns:
        DataFrame sorted by descending VIF.
    """

    standardized = (frame - frame.mean()) / frame.std(ddof=0)
    standardized = standardized.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    values = standardized.to_numpy()
    rows = []
    for index, column in enumerate(standardized.columns):
        rows.append(
            {
                "feature": column,
                "label": LABELS.get(column, column),
                "vif": float(variance_inflation_factor(values, index)),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def calculate_top_correlations(correlation: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Extract strongest absolute off-diagonal correlations.

    Args:
        correlation: Square correlation matrix.
        top_n: Number of pairs to return.

    Returns:
        DataFrame with the strongest correlated pairs.
    """

    rows = []
    columns = list(correlation.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(correlation.loc[left, right])
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "label_1": LABELS.get(left, left),
                    "label_2": LABELS.get(right, right),
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).head(top_n)


def save_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    """Save a heatmap of rating-signal correlations.

    Args:
        correlation: Correlation matrix.
        output_path: Destination image path.
    """

    labelled = correlation.rename(index=LABELS, columns=LABELS)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        labelled,
        vmin=-1.0,
        vmax=1.0,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Korelacja Pearsona"},
        annot_kws={"fontsize": 7},
    )
    plt.title("Macierz korelacji sygnałów rankingowych", pad=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_vif_plot(vif: pd.DataFrame, output_path: Path) -> None:
    """Save a compact plot of the highest VIF values.

    Args:
        vif: VIF table sorted descending.
        output_path: Destination image path.
    """

    data = vif.head(12).sort_values("vif", ascending=True)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=data, x="vif", y="label", color=PASTEL_BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("VIF, skala logarytmiczna")
    ax.set_ylabel("")
    ax.set_title("Najwyższe współczynniki VIF dla sygnałów rankingowych", pad=12)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=3)
    clean_axis(ax, grid_axis="x")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def main() -> None:
    """Run correlation and VIF diagnostics."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style(context="paper")
    data = load_odds_sample()
    feature_frame = data[RATING_COLUMNS].copy()
    correlation = feature_frame.corr(method="pearson")
    vif = calculate_vif(feature_frame)
    top_correlations = calculate_top_correlations(correlation)

    correlation.to_csv(ASSETS_DIR / "rating_signal_correlation_matrix.csv")
    vif.to_csv(ASSETS_DIR / "rating_signal_vif.csv", index=False)
    top_correlations.to_csv(ASSETS_DIR / "rating_signal_top_correlations.csv", index=False)
    save_correlation_heatmap(correlation, ASSETS_DIR / "rating_signal_correlation_heatmap.png")
    save_vif_plot(vif, ASSETS_DIR / "rating_signal_vif_top.png")

    print("\n=== RATING SIGNAL CORRELATION / VIF ===")
    print("Sample size:", len(data))
    print("Date range:", data["date"].min().date(), "to", data["date"].max().date())
    print("\nTop VIF:")
    print(vif.head(12).to_string(index=False))
    print("\nTop correlations:")
    print(top_correlations.head(12).to_string(index=False))
    print("\nSaved artefacts to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
