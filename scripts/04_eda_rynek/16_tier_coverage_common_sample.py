"""Analyze Tier 1 / Tier 2 coverage in the bookmaker comparison sample.

The raw GOL.GG match export does not contain an explicit official tier field,
therefore this script uses a transparent tournament-name heuristic. The goal is
not to establish an official taxonomy of all League of Legends leagues, but to
check whether the final OddsPortal-aligned evaluation sample is dominated by a
single low-level segment.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.thesis_style import PASTEL_BLUE, PASTEL_ORANGE, apply_thesis_style, clean_axis

GOLGG_MATCHES_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"
FINAL_SAMPLE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_w20_binomial_market_comparison"
    / "final_w20_binomial_market_common_sample.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "eda_point4"

MAJOR_REGION_PATTERNS = (
    r"^LPL\b",
    r"^LCK\b",
    r"^LEC\b",
    r"^LCS\b",
    r"^NA LCS\b",
    r"^EU LCS\b",
)
INTERNATIONAL_PATTERNS = (
    r"World Championship",
    r"\bMSI\b",
    r"Mid-Season Invitational",
    r"First Stand",
    r"Esports World Cup",
    r"\bEWC\b",
)
LOWER_TIER_MARKERS = (
    "academy",
    "challenger",
    "challengers",
    "nacl",
    " lck cl",
    "lck cl",
    "cl spring",
    "cl summer",
    "ldl",
)


def classify_tournament_tier(tournament_name: str | None) -> str:
    """Classify a tournament into an operational Tier 1 / Tier 2 bucket.

    Tier 1 contains major-region top leagues and international events. Tier 2
    contains regional leagues, ERLs, academy/challenger competitions and all
    tournaments not matched by the major-region rule.

    Args:
        tournament_name: Tournament name from the GOL.GG match export.

    Returns:
        Operational tier label used in the EDA section.
    """

    if tournament_name is None or pd.isna(tournament_name):
        return "Tier 2 / regionalne"

    name = str(tournament_name).strip()
    lowered = f" {name.lower()} "

    if any(marker in lowered for marker in LOWER_TIER_MARKERS):
        return "Tier 2 / regionalne"

    if any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in INTERNATIONAL_PATTERNS):
        return "Tier 1 / major"

    if any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in MAJOR_REGION_PATTERNS):
        return "Tier 1 / major"

    return "Tier 2 / regionalne"


def load_match_metadata() -> pd.DataFrame:
    """Load GOL.GG match identifiers, dates and tournament names.

    Returns:
        Data frame with one row per match and an operational tier label.
    """

    with GOLGG_MATCHES_PATH.open(encoding="utf-8") as file:
        matches = json.load(file)

    rows = [
        {
            "golgg_match_id": str(match.get("match_id")),
            "date": match.get("date"),
            "tournament_name": match.get("tournament_name"),
        }
        for match in matches
    ]
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["tier"] = data["tournament_name"].map(classify_tournament_tier)
    return data


def summarize_tiers(data: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """Summarize tier counts and shares for a data frame.

    Args:
        data: Data frame containing a ``tier`` column.
        sample_name: Human-readable sample name.

    Returns:
        Tier summary with counts and percentages.
    """

    summary = (
        data["tier"]
        .value_counts()
        .rename_axis("tier")
        .reset_index(name="matches")
    )
    summary["sample"] = sample_name
    summary["share_pct"] = 100.0 * summary["matches"] / summary["matches"].sum()
    return summary[["sample", "tier", "matches", "share_pct"]]


def plot_tier_distribution(summary: pd.DataFrame, output_path: Path) -> None:
    """Create a stacked bar chart of Tier 1 / Tier 2 sample composition.

    Args:
        summary: Summary returned by :func:`summarize_tiers` for all samples.
        output_path: Path for the PNG figure.
    """

    order = ["GOL.GG 2020+", "Próba z kursami"]
    tiers = ["Tier 1 / major", "Tier 2 / regionalne"]
    apply_thesis_style(context="paper")
    colors = {"Tier 1 / major": PASTEL_BLUE, "Tier 2 / regionalne": PASTEL_ORANGE}

    pivot = (
        summary.pivot(index="sample", columns="tier", values="share_pct")
        .reindex(order)
        .fillna(0.0)
    )
    counts = (
        summary.pivot(index="sample", columns="tier", values="matches")
        .reindex(order)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    left = [0.0] * len(pivot)
    y_positions = range(len(pivot))

    for tier in tiers:
        values = pivot[tier].to_numpy()
        bars = ax.barh(
            y_positions,
            values,
            left=left,
            label=tier,
            color=colors[tier],
            edgecolor="white",
        )
        for bar, value, count in zip(bars, values, counts[tier].to_numpy(), strict=False):
            if value >= 8.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%\n(n={int(count):,})".replace(",", " "),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        left = [current + value for current, value in zip(left, values, strict=False)]

    ax.set_yticks(list(y_positions), pivot.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Udział meczów [%]")
    ax.set_title("Struktura poziomu rozgrywek w pełnej i bukmacherskiej próbie")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    clean_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate tier coverage tables and figures for the EDA chapter."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_match_metadata()
    final_sample = pd.read_csv(FINAL_SAMPLE_PATH, dtype={"golgg_match_id": str})
    final_sample = final_sample.merge(
        matches[["golgg_match_id", "date", "tournament_name", "tier"]],
        on="golgg_match_id",
        how="left",
        suffixes=("", "_golgg"),
    )

    model_period = matches[matches["date"] >= pd.Timestamp("2020-01-01")].copy()
    summaries = pd.concat(
        [
            summarize_tiers(model_period, "GOL.GG 2020+"),
            summarize_tiers(final_sample, "Próba z kursami"),
        ],
        ignore_index=True,
    )

    tournament_summary = (
        final_sample.groupby(["tier", "tournament_name"], dropna=False)
        .size()
        .reset_index(name="matches")
        .sort_values(["tier", "matches"], ascending=[True, False])
    )

    summaries.to_csv(OUTPUT_DIR / "tier_coverage_summary.csv", index=False)
    tournament_summary.to_csv(OUTPUT_DIR / "tier_coverage_tournaments.csv", index=False)
    plot_tier_distribution(summaries, OUTPUT_DIR / "tier_coverage_common_sample.png")

    print(summaries.to_string(index=False))
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
