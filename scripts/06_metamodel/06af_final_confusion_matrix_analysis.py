"""Confusion-matrix diagnostics for the final model and market benchmarks.

The thesis primarily evaluates probabilistic quality with LogLoss, Brier score,
ECE and AUC. This script adds a threshold-based diagnostic view: how the final
model and bookmaker benchmarks behave if converted into hard Team-1 win/loss
predictions at threshold 0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.thesis_style import apply_thesis_style


INPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_w20_binomial_market_comparison"
    / "final_w20_binomial_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "final_confusion_matrix_analysis"
THRESHOLD = 0.5

MODELS = {
    "LR-ElasticNet-W20-Binomial": "prob_lr_elasticnet_w20_binomial",
    "Player Glicko-2": "player_glicko2",
    "Market Close": "market_close",
    "Market Open": "market_open",
}


def calculate_confusion_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    threshold: float = THRESHOLD,
) -> dict[str, float | int | str]:
    """Calculate threshold-based confusion-matrix metrics.

    Args:
        y_true: Binary target where 1 means Team 1 win.
        y_prob: Probability assigned to Team 1 win.
        model_name: Human-readable model name.
        threshold: Decision threshold for Team 1 win.

    Returns:
        Dictionary with confusion counts and classification metrics.
    """

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
    predicted_team1_rate = y_pred.mean()

    return {
        "model": model_name,
        "threshold": threshold,
        "n_matches": len(y_true),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_team1": precision_score(y_true, y_pred, zero_division=0),
        "recall_team1": recall_score(y_true, y_pred, zero_division=0),
        "specificity_team2": specificity,
        "f1_team1": f1_score(y_true, y_pred, zero_division=0),
        "negative_precision_team2": negative_precision,
        "predicted_team1_rate": predicted_team1_rate,
        "actual_team1_rate": y_true.mean(),
    }


def save_confusion_heatmaps(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    threshold: float = THRESHOLD,
) -> None:
    """Save a grid of normalized confusion matrices."""

    apply_thesis_style(context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    axes_flat = axes.ravel()
    y_true = data["y_true"].to_numpy(dtype=int)

    for ax, (model_name, column) in zip(axes_flat, MODELS.items()):
        y_pred = (data[column].to_numpy(dtype=float) >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        labels = np.array(
            [
                [f"TN\n{matrix[0, 0]}\n{normalized[0, 0]:.1%}", f"FP\n{matrix[0, 1]}\n{normalized[0, 1]:.1%}"],
                [f"FN\n{matrix[1, 0]}\n{normalized[1, 0]:.1%}", f"TP\n{matrix[1, 1]}\n{normalized[1, 1]:.1%}"],
            ]
        )
        sns.heatmap(
            normalized,
            annot=labels,
            fmt="",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=0.6,
            linecolor="white",
            ax=ax,
        )
        accuracy = metrics.loc[metrics["model"] == model_name, "accuracy"].iloc[0]
        ax.set_title(f"{model_name}\nAccuracy={accuracy:.3f}")
        ax.set_xlabel("Predykcja")
        ax.set_ylabel("Wynik rzeczywisty")
        ax.set_xticklabels(["Team 2", "Team 1"])
        ax.set_yticklabels(["Team 2", "Team 1"], rotation=0)

    fig.suptitle("Macierze pomyłek przy progu 0.5", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_confusion_matrices_threshold_05.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run confusion-matrix diagnostics."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    required = ["y_true", *MODELS.values()]
    data = data.dropna(subset=required).copy()
    y_true = data["y_true"].to_numpy(dtype=int)

    rows = []
    for model_name, column in MODELS.items():
        rows.append(
            calculate_confusion_metrics(
                y_true=y_true,
                y_prob=data[column].to_numpy(dtype=float),
                model_name=model_name,
            )
        )

    metrics = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "final_confusion_matrix_metrics.csv", index=False)
    save_confusion_heatmaps(data, metrics)

    print("Confusion-matrix metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
