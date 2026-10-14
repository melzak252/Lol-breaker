"""Create alpha-temperature heatmaps for financial hybrid diagnostics.

The script reads the expanded financial validation grid and visualizes how the
model-market hybrid weight (`alpha`) and probability temperature (`T`) affect
ROI, yield, CLV and maximum drawdown for a selected staking policy.

The composite score is deliberately diagnostic: it rewards high ROI, high yield
and positive CLV, while penalizing high MaxDD. It is not an investment rule.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.thesis_style import apply_thesis_style


INPUT_PATH = PROJECT_ROOT / "docs" / "assets" / "financial_point8" / "financial_validation_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "financial_point8"
DEFAULT_SCOPE = "2024+"
DEFAULT_POLICY = "Kelly 0.05 min2 max100"
POLICIES_TO_PLOT = [
    "Kelly 0.05 min2 max100",
    "Kelly 0.25 min2 max100",
]


def parse_alpha_temperature(candidate: str) -> tuple[float | None, float | None]:
    """Extract alpha and temperature from a hybrid candidate label.

    Args:
        candidate: Candidate name, e.g. ``Hybrid a=0.50 T=0.80``.

    Returns:
        Tuple ``(alpha, temperature)``. Non-hybrid labels return ``(None, None)``.
    """

    alpha_match = re.search(r"a=([0-9.]+)", candidate)
    if alpha_match is None:
        return None, None
    temp_match = re.search(r"T=([0-9.]+)", candidate)
    alpha = float(alpha_match.group(1))
    temperature = float(temp_match.group(1)) if temp_match else 1.0
    return alpha, temperature


def robust_minmax(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normalize a metric to [0, 1] using percentile clipping.

    Args:
        values: Metric values to normalize.
        higher_is_better: Whether larger raw values should map to larger scores.

    Returns:
        Normalized score in the range [0, 1].
    """

    lower = float(values.quantile(0.05))
    upper = float(values.quantile(0.95))
    if np.isclose(lower, upper):
        normalized = pd.Series(0.5, index=values.index)
    else:
        normalized = ((values.clip(lower, upper) - lower) / (upper - lower)).clip(0, 1)
    return normalized if higher_is_better else 1.0 - normalized


def prepare_grid(
    summary: pd.DataFrame,
    scope: str = DEFAULT_SCOPE,
    policy: str = DEFAULT_POLICY,
) -> pd.DataFrame:
    """Prepare alpha-temperature grid and composite scores.

    Args:
        summary: Financial validation summary table.
        scope: Time scope, usually ``2024+``.
        policy: Staking policy to isolate when comparing alpha and temperature.

    Returns:
        DataFrame with parsed alpha/T values and normalized component scores.
    """

    data = summary[
        (summary["scope"] == scope)
        & (summary["staking_policy"] == policy)
        & summary["candidate"].str.startswith("Hybrid")
    ].copy()
    parsed = data["candidate"].apply(parse_alpha_temperature)
    data["alpha"] = parsed.apply(lambda item: item[0])
    data["temperature"] = parsed.apply(lambda item: item[1])
    data = data.dropna(subset=["alpha", "temperature"]).copy()

    data["roi_score"] = robust_minmax(np.log1p(data["roi_pct"].clip(lower=0)))
    data["yield_score"] = robust_minmax(data["yield_pct"])
    data["clv_score"] = robust_minmax(data["avg_clv_pct"])
    data["drawdown_score"] = robust_minmax(data["max_drawdown_pct"], higher_is_better=False)
    data["composite_score"] = (
        0.25 * data["roi_score"]
        + 0.25 * data["yield_score"]
        + 0.25 * data["clv_score"]
        + 0.25 * data["drawdown_score"]
    )
    data["risk_adjusted_score"] = data["composite_score"] * (data["bets"] >= 100).astype(float)
    return data.sort_values(["temperature", "alpha"]).reset_index(drop=True)


def pivot_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot alpha-temperature grid for one metric."""

    return data.pivot(index="temperature", columns="alpha", values=metric).sort_index(ascending=False)


def plot_heatmaps(data: pd.DataFrame, policy_slug: str) -> None:
    """Save raw metric and composite alpha-temperature heatmaps."""

    apply_thesis_style(context="paper")
    metrics = [
        ("roi_pct", "ROI [%]", "YlGnBu", ".0f"),
        ("yield_pct", "Yield [%]", "YlGnBu", ".0f"),
        ("avg_clv_pct", "Średni CLV [%]", "YlGnBu", ".1f"),
        ("max_drawdown_pct", "MaxDD [%]", "YlOrRd", ".1f"),
        ("composite_score", "Composite score", "YlGnBu", ".2f"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    axes_flat = axes.ravel()
    best = data.sort_values("composite_score", ascending=False).iloc[0]

    for ax, (metric, title, cmap, fmt) in zip(axes_flat, metrics):
        pivot = pivot_metric(data, metric)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar=True,
            ax=ax,
        )
        ax.scatter(
            [list(pivot.columns).index(best["alpha"]) + 0.5],
            [list(pivot.index).index(best["temperature"]) + 0.5],
            marker="*",
            s=220,
            color="#2F3640",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("alpha — waga modelu")
        ax.set_ylabel("T")

    axes_flat[-1].axis("off")
    fig.suptitle(
        "Alpha × temperatura: zysk, CLV, MaxDD i wynik złożony",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"financial_alpha_temperature_heatmaps_{policy_slug}.png", bbox_inches="tight")
    plt.close(fig)


def plot_composite(data: pd.DataFrame, policy_slug: str) -> None:
    """Save a focused composite score heatmap."""

    apply_thesis_style(context="paper")
    pivot = pivot_metric(data, "composite_score")
    best = data.sort_values("composite_score", ascending=False).iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Composite score"},
        ax=ax,
    )
    ax.scatter(
        [list(pivot.columns).index(best["alpha"]) + 0.5],
        [list(pivot.index).index(best["temperature"]) + 0.5],
        marker="*",
        s=260,
        color="#2F3640",
        edgecolor="white",
        linewidth=0.9,
    )
    ax.set_title("Najlepszy obszar alpha × T według composite score")
    ax.set_xlabel("alpha — waga modelu")
    ax.set_ylabel("T")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"financial_alpha_temperature_composite_{policy_slug}.png", bbox_inches="tight")
    plt.close(fig)


def policy_slug(policy: str) -> str:
    """Convert a staking policy label into a filesystem-safe slug."""

    return policy.lower().replace(" ", "_").replace("%", "pct")


def generate_for_policy(summary: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Generate all alpha-temperature artefacts for one staking policy."""

    slug = policy_slug(policy)
    grid = prepare_grid(summary, policy=policy)
    grid.to_csv(OUTPUT_DIR / f"financial_alpha_temperature_grid_{slug}.csv", index=False)
    plot_heatmaps(grid, slug)
    plot_composite(grid, slug)

    columns = [
        "candidate",
        "alpha",
        "temperature",
        "roi_pct",
        "yield_pct",
        "avg_clv_pct",
        "max_drawdown_pct",
        "bets",
        "logloss",
        "composite_score",
    ]
    best = grid.sort_values("composite_score", ascending=False)[columns].head(12)
    best.to_csv(OUTPUT_DIR / f"financial_alpha_temperature_best_{slug}.csv", index=False)
    print(f"\nBest alpha-temperature configurations for {policy}:")
    print(best.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return best


def main() -> None:
    """Generate heatmaps and recommendation tables."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(INPUT_PATH)
    for policy in POLICIES_TO_PLOT:
        generate_for_policy(summary, policy)


if __name__ == "__main__":
    main()
