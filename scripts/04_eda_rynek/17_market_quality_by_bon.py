"""Analyze bookmaker benchmark quality by match series format.

The final thesis compares model probabilities with OddsPortal-derived market
probabilities on the common GOL.GG/OddsPortal sample. This script adds an EDA
view that checks whether market predictive quality differs between Bo1, Bo3 and
Bo5 series.
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

from src.visualization.thesis_style import MARKET_PALETTE, apply_thesis_style, clean_axis

GOLGG_MATCHES_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"
FINAL_SAMPLE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_w20_binomial_market_comparison"
    / "final_w20_binomial_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "eda_point4"
MARKET_COLUMNS = ("market_open", "market_close")


def load_series_metadata() -> pd.DataFrame:
    """Load GOL.GG match identifiers and best-of format metadata.

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


def prepare_market_sample() -> pd.DataFrame:
    """Merge final market-comparison sample with series-format metadata.

    Returns:
        Common sample with ``BoN`` attached.
    """

    sample = pd.read_csv(FINAL_SAMPLE_PATH, dtype={"golgg_match_id": str})
    metadata = load_series_metadata()
    merged = sample.merge(metadata, on="golgg_match_id", how="left")
    if merged["BoN"].isna().any():
        missing = int(merged["BoN"].isna().sum())
        raise ValueError(f"Missing BoN metadata for {missing} matches.")
    return merged


def summarize_market_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Compute bookmaker benchmark metrics by series format.

    Args:
        data: Common market sample with ``BoN``, target and market columns.

    Returns:
        Long-form metric summary for market open and market close.
    """

    rows: list[dict[str, float | int | str]] = []

    for best_of, group in data.groupby("BoN"):
        for column in MARKET_COLUMNS:
            label = "Market Open" if column == "market_open" else "Market Close"
            rows.append(
                {
                    "BoN": int(best_of),
                    "format": f"Bo{int(best_of)}",
                    "benchmark": label,
                    "matches": int(len(group)),
                    "auc": roc_auc_score(group["y_true"], group[column]),
                    "logloss": log_loss(group["y_true"], group[column], labels=[0, 1]),
                    "brier": brier_score_loss(group["y_true"], group[column]),
                    "ece": calculate_ece(group["y_true"], group[column]),
                }
            )

    return pd.DataFrame(rows).sort_values(["BoN", "benchmark"])


def plot_market_logloss_by_format(summary: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart of market LogLoss by BoN format.

    Args:
        summary: Long-form summary returned by :func:`summarize_market_quality`.
        output_path: Path for the generated PNG figure.
    """

    formats = ["Bo1", "Bo3", "Bo5"]
    benchmarks = ["Market Open", "Market Close"]
    apply_thesis_style(context="paper")
    colors = {"Market Open": MARKET_PALETTE["Market Open"], "Market Close": MARKET_PALETTE["Market Close"]}

    pivot = summary.pivot(index="format", columns="benchmark", values="logloss").reindex(formats)
    counts = summary.drop_duplicates("format").set_index("format")["matches"].reindex(formats)

    x_positions = np.arange(len(formats))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for offset, benchmark in zip([-width / 2, width / 2], benchmarks, strict=False):
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
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    xtick_labels = [f"{fmt}\n(n={int(counts.loc[fmt]):,})".replace(",", " ") for fmt in formats]
    ax.set_xticks(x_positions, xtick_labels)
    ax.set_ylabel("LogLoss")
    ax.set_title("Jakość predykcji bukmacherów według formatu serii")
    ax.legend(frameon=False)
    clean_axis(ax)
    min_value = float(pivot.min().min())
    max_value = float(pivot.max().max())
    margin = max(0.01, 0.18 * (max_value - min_value))
    ax.set_ylim(min_value - margin, max_value + margin)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate market-quality-by-BoN EDA artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_market_sample()
    summary = summarize_market_quality(data)

    summary.to_csv(OUTPUT_DIR / "market_quality_by_bon.csv", index=False)
    plot_market_logloss_by_format(summary, OUTPUT_DIR / "market_quality_by_bon_logloss.png")

    print(summary.to_string(index=False))
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
