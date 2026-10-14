"""Analyze final W20-Binomial model advantage by series format.

This diagnostic uses the final common GOL.GG/OddsPortal sample and compares the
LR-ElasticNet-W20-Binomial model with market benchmarks separately for Bo1, Bo3
and Bo5 series. The goal is to identify where the final model gains the most in
probabilistic quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.visualization.thesis_style import (
    DARK_TEXT,
    MODEL_PALETTE,
    PASTEL_BLUE,
    PASTEL_GREEN,
    PASTEL_ORANGE,
    apply_thesis_style,
    clean_axis,
)

GOLGG_MATCHES_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"
FINAL_SAMPLE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_w20_binomial_market_comparison"
    / "final_w20_binomial_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "final_w20_binomial_market_comparison"

MODEL_COLUMNS = {
    "LR-ElasticNet-W20-Binomial": "prob_lr_elasticnet_w20_binomial",
    "Mkt Open": "market_open",
    "Mkt Close": "market_close",
    "Player Glicko-2": "player_glicko2",
}


def load_series_metadata() -> pd.DataFrame:
    """Load match identifiers and best-of format from GOL.GG metadata.

    Returns:
        Data frame with ``golgg_match_id`` and ``BoN`` columns.
    """

    with GOLGG_MATCHES_PATH.open(encoding="utf-8") as file:
        matches = json.load(file)

    return pd.DataFrame(
        [
            {
                "golgg_match_id": str(match.get("match_id")),
                "BoN": int(match.get("best_of") or 1),
            }
            for match in matches
        ]
    )


def prepare_data() -> pd.DataFrame:
    """Load final sample and attach BoN format metadata.

    Returns:
        Final common sample with a ``BoN`` column.
    """

    sample = pd.read_csv(FINAL_SAMPLE_PATH, dtype={"golgg_match_id": str})
    metadata = load_series_metadata()
    data = sample.merge(metadata, on="golgg_match_id", how="left")
    if data["BoN"].isna().any():
        missing = int(data["BoN"].isna().sum())
        raise ValueError(f"Missing BoN metadata for {missing} matches.")
    return data


def summarize_by_bon(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize model and benchmark quality by series format.

    Args:
        data: Final common sample with model probabilities and BoN metadata.

    Returns:
        Long-form metric summary by format and model.
    """

    rows: list[dict[str, float | int | str]] = []
    for best_of, group in data.groupby("BoN"):
        for model_name, column in MODEL_COLUMNS.items():
            rows.append(
                {
                    "BoN": int(best_of),
                    "format": f"Bo{int(best_of)}",
                    "model": model_name,
                    "matches": int(len(group)),
                    "auc": roc_auc_score(group["y_true"], group[column]),
                    "logloss": log_loss(group["y_true"], group[column], labels=[0, 1]),
                    "brier": brier_score_loss(group["y_true"], group[column]),
                    "ece": calculate_ece(group["y_true"], group[column]),
                }
            )
    return pd.DataFrame(rows).sort_values(["BoN", "model"])


def summarize_advantage(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute LogLoss advantage of final model versus benchmarks by BoN.

    Args:
        summary: Metric summary returned by :func:`summarize_by_bon`.

    Returns:
        Wide-form advantage table where positive values favor the final model.
    """

    pivot = summary.pivot(index="format", columns="model", values="logloss")
    matches = summary.drop_duplicates("format").set_index("format")["matches"]
    final_loss = pivot["LR-ElasticNet-W20-Binomial"]

    rows = []
    for benchmark in ["Mkt Open", "Mkt Close", "Player Glicko-2"]:
        for series_format in pivot.index:
            rows.append(
                {
                    "format": series_format,
                    "benchmark": benchmark,
                    "matches": int(matches.loc[series_format]),
                    "final_logloss": final_loss.loc[series_format],
                    "benchmark_logloss": pivot.loc[series_format, benchmark],
                    "delta_benchmark_minus_final": pivot.loc[series_format, benchmark]
                    - final_loss.loc[series_format],
                }
            )
    return pd.DataFrame(rows)


def plot_model_advantage(advantage: pd.DataFrame, output_path: Path) -> None:
    """Plot final-model LogLoss advantage by BoN and benchmark.

    Args:
        advantage: Output of :func:`summarize_advantage`.
        output_path: Path for the generated PNG figure.
    """

    formats = ["Bo1", "Bo3", "Bo5"]
    benchmarks = ["Mkt Open", "Mkt Close", "Player Glicko-2"]
    apply_thesis_style(context="paper")
    colors = {
        "Mkt Open": PASTEL_BLUE,
        "Mkt Close": PASTEL_ORANGE,
        "Player Glicko-2": MODEL_PALETTE.get("Player Glicko-2", PASTEL_GREEN),
    }
    pivot = (
        advantage.pivot(index="format", columns="benchmark", values="delta_benchmark_minus_final")
        .reindex(formats)
    )
    counts = advantage.drop_duplicates("format").set_index("format")["matches"].reindex(formats)

    x_positions = np.arange(len(formats))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    for offset, benchmark in zip(offsets, benchmarks, strict=False):
        values = pivot[benchmark].to_numpy()
        bars = ax.bar(
            x_positions + offset,
            values,
            width=width,
            label=benchmark,
            color=colors[benchmark],
            edgecolor="white",
        )
        for bar, value in zip(bars, values, strict=False):
            va = "bottom" if value >= 0 else "top"
            y_offset = 0.0006 if value >= 0 else -0.0006
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + y_offset,
                f"{value:+.3f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    labels = [f"{fmt}\n(n={int(counts.loc[fmt]):,})".replace(",", " ") for fmt in formats]
    ax.axhline(0.0, color=DARK_TEXT, linewidth=0.9)
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Różnica LogLoss benchmark - model")
    ax.set_title("Przewaga LR-ElasticNet-W20-Binomial według formatu serii")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate BoN-segment analysis artifacts for the final model."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    summary = summarize_by_bon(data)
    advantage = summarize_advantage(summary)

    summary.to_csv(OUTPUT_DIR / "final_model_bon_segment_metrics.csv", index=False)
    advantage.to_csv(OUTPUT_DIR / "final_model_bon_segment_advantage.csv", index=False)
    plot_model_advantage(
        advantage,
        OUTPUT_DIR / "final_model_bon_segment_advantage.png",
    )

    print(summary.to_string(index=False))
    print(advantage.to_string(index=False))
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
