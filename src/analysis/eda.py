from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_temporal_distribution(
    df: pd.DataFrame,
    date_col: str = "date",
    title: str = "Matches per Year",
    save_path: str = "docs/assets/temporal_distribution.png",
) -> pd.Series:
    """Analyze and plot temporal match distribution.

    Args:
        df: Input dataframe containing a date column.
        date_col: Name of the date column.
        title: Plot title.
        save_path: Target path for the generated plot.

    Returns:
        Number of matches per year.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year

    matches_per_year = df.groupby("year").size()

    plt.figure(figsize=(12, 6))
    matches_per_year.plot(kind="bar", color="#2ecc71")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved temporal distribution plot to {save_path}")

    return matches_per_year


def analyze_categorical_counts(
    df: pd.DataFrame,
    col: str,
    top_n: int = 10,
) -> pd.Series:
    """Return top category counts.

    Args:
        df: Input dataframe.
        col: Categorical column name.
        top_n: Number of categories to return.

    Returns:
        Series with top category counts.
    """
    return df[col].value_counts().head(top_n)


def calculate_margin(o1: float, o2: float) -> float:
    """Calculate two-way bookmaker overround.

    Args:
        o1: Decimal odds for side one.
        o2: Decimal odds for side two.

    Returns:
        Bookmaker margin. Invalid odds return `np.nan`.
    """
    if pd.isna(o1) or pd.isna(o2) or o1 <= 0 or o2 <= 0:
        return np.nan
    return (1 / o1 + 1 / o2) - 1


def analyze_bookmaker_margins(
    df: pd.DataFrame,
    bookies: List[str],
    odds1_suffix: str = "_close",
    odds2_suffix: str = "_close",
    prefix: str = "odds1_",
) -> pd.DataFrame:
    """Analyze bookmaker margins for multiple providers.

    Args:
        df: Dataframe with bookmaker odds columns.
        bookies: Bookmaker identifiers used in column names.
        odds1_suffix: Column suffix for side-one odds.
        odds2_suffix: Column suffix for side-two odds.
        prefix: Column prefix for side-one odds.

    Returns:
        Dataframe with average, median, standard deviation, and valid count.
    """
    margin_results = []

    for bookie in bookies:
        col1 = (
            f"{prefix}{bookie}{odds1_suffix}"
            if prefix
            else f"{bookie}{odds1_suffix}"
        )
        col2_prefix = prefix.replace("1", "2")
        col2 = (
            f"{col2_prefix}{bookie}{odds2_suffix}"
            if prefix
            else f"{bookie}{odds2_suffix}"
        )

        if col1 in df.columns and col2 in df.columns:
            margins = df.apply(
                lambda row: calculate_margin(row[col1], row[col2]),
                axis=1,
            )
            margin_results.append(
                {
                    "bookie": bookie,
                    "avg_margin": margins.mean(),
                    "median_margin": margins.median(),
                    "std_margin": margins.std(),
                    "count": margins.count(),
                }
            )

    return pd.DataFrame(margin_results)


def determine_match_format(score1: int, score2: int) -> int:
    """Determine match format (BoN) from final score.

    Args:
        score1: Final score of team one.
        score2: Final score of team two.

    Returns:
        Match format as `1`, `3`, or `5`.
    """
    max_score = max(score1, score2)
    format_bon = max_score * 2 - 1
    if format_bon not in [1, 3, 5]:
        if max_score == 1:
            return 1
        if max_score == 2:
            return 3
        if max_score == 3:
            return 5
        return 1
    return format_bon
