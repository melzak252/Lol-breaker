"""Create standalone model figures for the final whitepaper.

This script generates polished, presentation-oriented figures for chapter 04.
It reads already generated experiment artifacts and converts selected results into
clearer visuals for the standalone narrative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT_DIR / "docs" / "assets"
OUTPUT_DIR = ASSETS_DIR / "whitepaper_final"
ABLATION_RESULTS_PATH = ASSETS_DIR / "metamodel_experiments_point6" / "metamodel_ablation_results.csv"


def load_player_team_ablation() -> pd.DataFrame:
    """Load player-vs-team ablation results.

    Returns:
        DataFrame containing only player-base and player+team variants.
    """
    data = pd.read_csv(ABLATION_RESULTS_PATH)
    subset = data.loc[data["experiment_group"] == "player_vs_team_features"].copy()
    order = ["Player-base", "Player-base + Team-base"]
    subset["variant"] = pd.Categorical(subset["variant"], categories=order, ordered=True)
    return subset.sort_values("variant")


def save_player_vs_team_readable(data: pd.DataFrame) -> Path:
    """Save a readable player-vs-team ablation chart.

    Args:
        data: DataFrame with variants and model metrics.

    Returns:
        Path to the generated figure.
    """
    output_path = OUTPUT_DIR / "metamodel_player_vs_team_features_readable.png"
    colors = ["#2E7D32", "#90A4AE"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1]},
    )

    variants = data["variant"].astype(str).tolist()
    y_positions = range(len(variants))

    logloss_axis = axes[0]
    logloss_axis.barh(y_positions, data["logloss"], color=colors, height=0.5)
    logloss_axis.set_yticks(list(y_positions), variants)
    logloss_axis.invert_yaxis()
    logloss_axis.set_xlabel("LogLoss (niżej = lepiej)")
    logloss_axis.set_title("Jakość prawdopodobieństw")
    logloss_axis.set_xlim(data["logloss"].min() - 0.0005, data["logloss"].max() + 0.0007)
    logloss_axis.grid(axis="x", alpha=0.25)
    for index, value in enumerate(data["logloss"]):
        logloss_axis.text(value + 0.00005, index, f"{value:.4f}", va="center", fontweight="bold")

    auc_axis = axes[1]
    auc_axis.barh(y_positions, data["auc"], color=colors, height=0.5)
    auc_axis.set_yticks(list(y_positions), ["", ""])
    auc_axis.invert_yaxis()
    auc_axis.set_xlabel("AUC (wyżej = lepiej)")
    auc_axis.set_title("Jakość rankingu")
    auc_axis.set_xlim(data["auc"].min() - 0.001, data["auc"].max() + 0.0015)
    auc_axis.grid(axis="x", alpha=0.25)
    for index, value in enumerate(data["auc"]):
        auc_axis.text(value + 0.00012, index, f"{value:.4f}", va="center", fontweight="bold")

    player = data.loc[data["variant"].astype(str) == "Player-base"].iloc[0]
    team = data.loc[data["variant"].astype(str) == "Player-base + Team-base"].iloc[0]
    delta_logloss = team["logloss"] - player["logloss"]
    delta_auc = team["auc"] - player["auc"]

    fig.suptitle("Ablation: cechy graczowe vs dodanie cech drużynowych", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"Dodanie team-base: ΔLogLoss = {delta_logloss:+.4f}, ΔAUC = {delta_auc:+.4f}. "
        "W tej próbie prostszy wariant player-base pozostaje korzystniejszy.",
        ha="center",
        fontsize=10,
        color="#37474F",
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate all chapter 04 standalone figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_player_team_ablation()
    print(save_player_vs_team_readable(data))


if __name__ == "__main__":
    main()
