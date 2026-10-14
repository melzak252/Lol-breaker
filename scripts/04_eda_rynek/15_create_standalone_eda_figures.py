"""Create standalone EDA insight figures for the final whitepaper.

The figures focus on interpretable observations that can later be translated
into Canva slides: format evolution, favorite probability distribution, class
balance, tier mix, and market difficulty by match format.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[2]
ODDS_PATH = ROOT_DIR / "data" / "odds.csv"
GOLGG_PATH = ROOT_DIR / "data" / "golgg_matches.json"
OUTPUT_DIR = ROOT_DIR / "docs" / "assets" / "whitepaper_final"
BOOKMAKERS = ["betclic", "betfan", "efortuna", "lv_bet", "sts", "superbet"]
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


def setup_style() -> None:
    """Configure a clean chart style for standalone whitepaper figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
        }
    )


def load_odds() -> pd.DataFrame:
    """Load and prepare the odds dataset.

    Returns:
        DataFrame with typed dates, scores, inferred BoN format, opening no-vig
        probabilities, favorite probabilities, and broad tier labels.
    """
    data = pd.read_csv(ODDS_PATH)
    data["golgg_date"] = pd.to_datetime(data["golgg_date"], errors="coerce")
    data["year"] = data["golgg_date"].dt.year
    data["t1_score"] = pd.to_numeric(data["t1_score"], errors="coerce")
    data["t2_score"] = pd.to_numeric(data["t2_score"], errors="coerce")
    data["avg_open_home"] = pd.to_numeric(data["avg_open_home"], errors="coerce")
    data["avg_open_away"] = pd.to_numeric(data["avg_open_away"], errors="coerce")
    data["avg_odds_home"] = pd.to_numeric(data["avg_odds_home"], errors="coerce")
    data["avg_odds_away"] = pd.to_numeric(data["avg_odds_away"], errors="coerce")
    data["t1_win"] = data["t1_win"].astype(str).str.lower().eq("true")
    data["bon"] = data.apply(infer_bon, axis=1)
    data["tier_segment"] = data["tournament"].fillna("").map(classify_tier)

    data["p_home_open_novig"] = no_vig_home_probability(
        data["avg_open_home"], data["avg_open_away"]
    )
    data["p_home_close_novig"] = no_vig_home_probability(
        data["avg_odds_home"], data["avg_odds_away"]
    )
    data["p_away_open_novig"] = 1.0 - data["p_home_open_novig"]
    data["p_favorite_open"] = data[["p_home_open_novig", "p_away_open_novig"]].max(axis=1)
    return data


def load_golgg_matches() -> pd.DataFrame:
    """Load GOL.GG matches for sport-level format evolution.

    Returns:
        DataFrame with match date, year, and standardized BoN label.
    """
    with GOLGG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = []
    for match in payload:
        score_1 = int(match.get("score_1", 0) or 0)
        score_2 = int(match.get("score_2", 0) or 0)
        games = match.get("games") or []
        raw_bon = match.get("BoN")
        bon = standardize_golgg_bon(raw_bon, score_1, score_2, len(games))
        rows.append({"date": match.get("date"), "bon": bon})

    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])
    data["year"] = data["date"].dt.year
    return data


def standardize_golgg_bon(
    raw_bon: object,
    score_1: int,
    score_2: int,
    games_count: int,
) -> str:
    """Standardize GOL.GG match format into Bo1, Bo3, or Bo5.

    Args:
        raw_bon: Raw BoN field from GOL.GG.
        score_1: Team 1 match score.
        score_2: Team 2 match score.
        games_count: Number of maps in the match payload.

    Returns:
        Standardized format label.
    """
    try:
        bon = int(raw_bon)
    except (TypeError, ValueError):
        winner_score = max(score_1, score_2)
        if winner_score <= 1 and games_count <= 1:
            bon = 1
        elif winner_score <= 2 and games_count <= 3:
            bon = 3
        else:
            bon = 5

    if bon <= 1:
        return "Bo1"
    if bon <= 3:
        return "Bo3"
    return "Bo5"


def no_vig_home_probability(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    """Convert two-way decimal odds to normalized home/team-1 probability.

    Args:
        home_odds: Decimal odds for team 1.
        away_odds: Decimal odds for team 2.

    Returns:
        No-vig implied probability for team 1.
    """
    valid = (home_odds > 1.0) & (away_odds > 1.0)
    implied_home = 1.0 / home_odds.where(valid)
    implied_away = 1.0 / away_odds.where(valid)
    return implied_home / (implied_home + implied_away)


def infer_bon(row: pd.Series) -> str:
    """Infer match format from the winner score.

    Args:
        row: Odds dataset row with numeric team scores.

    Returns:
        One of Bo1, Bo3, Bo5, or Other.
    """
    winner_score = max(row.get("t1_score", 0), row.get("t2_score", 0))
    if winner_score == 1:
        return "Bo1"
    if winner_score == 2:
        return "Bo3"
    if winner_score == 3:
        return "Bo5"
    return "Other"


def classify_tier(tournament: str) -> str:
    """Classify tournament into broad Tier 1 vs Regional/ERL segments.

    Args:
        tournament: Tournament name from the odds dataset.

    Returns:
        Broad segment label.
    """
    value = str(tournament)
    if "Pro League" in value and "Oceanic" not in value and "Continental" not in value:
        return "Tier 1"
    if any(keyword in value for keyword in TIER1_KEYWORDS):
        return "Tier 1"
    return "Regional / ERL"


def save_format_evolution(data: pd.DataFrame) -> Path:
    """Save stacked format evolution by year for the GOL.GG dataset.

    Args:
        data: Prepared GOL.GG match DataFrame.

    Returns:
        Path to the saved figure.
    """
    counts = (
        data.dropna(subset=["year"])
        .query("bon != 'Other'")
        .groupby(["year", "bon"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts = counts[[column for column in ["Bo1", "Bo3", "Bo5"] if column in counts.columns]]
    fig, axis = plt.subplots(figsize=(14, 6.5))
    colors = ["#3F5C99", "#24958D", "#58C65D"]
    counts.plot(kind="bar", stacked=True, ax=axis, color=colors, width=0.72)
    totals = counts.sum(axis=1)
    for index, total in enumerate(totals):
        axis.text(index, total + totals.max() * 0.015, f"{int(total)}", ha="center", fontweight="bold", fontsize=10)

    axis.set_title("Ewolucja formatów meczów w GOL.GG")
    axis.set_ylabel("Liczba meczów")
    axis.set_xlabel("Rok")
    axis.set_ylim(0, totals.max() * 1.12)
    axis.legend(title="Format", loc="upper left", frameon=True)
    axis.grid(axis="y", alpha=0.30, linestyle="--")
    axis.tick_params(axis="x", rotation=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    output_path = OUTPUT_DIR / "eda_format_evolution_golgg.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_favorite_probability_distribution(data: pd.DataFrame) -> Path:
    """Save the opening favorite probability distribution.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        Path to the saved figure.
    """
    favorite_prob = data["p_favorite_open"].dropna()
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(favorite_prob, bins=30, color="#7E57C2", edgecolor="white", alpha=0.9)
    axis.axvline(favorite_prob.median(), color="#D84315", linewidth=2, label=f"Mediana: {favorite_prob.median():.2f}")
    axis.set_title("Dystrybucja prawdopodobieństwa faworyta na opening odds")
    axis.set_xlabel("No-vig prawdopodobieństwo faworyta")
    axis.set_ylabel("Liczba meczów")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    output_path = OUTPUT_DIR / "eda_favorite_open_probability_distribution.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_team1_balance(data: pd.DataFrame) -> Path:
    """Save Team 1 vs Team 2 win balance as a donut chart.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        Path to the saved figure.
    """
    counts = pd.Series(
        {
            "Team 1 win": int(data["t1_win"].sum()),
            "Team 2 win": int((~data["t1_win"].astype(bool)).sum()),
        }
    )
    total = counts.sum()
    colors = ["#42A5F5", "#EF5350"]
    fig, axis = plt.subplots(figsize=(8, 6))
    wedges, _ = axis.pie(
        counts.values,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 3},
    )
    axis.text(0, 0.08, f"{int(total):,}".replace(",", " "), ha="center", fontsize=18, fontweight="bold")
    axis.text(0, -0.10, "meczów", ha="center", fontsize=11, color="#455A64")
    legend_labels = [
        f"{label}: {int(value):,} ({value / total * 100:.1f}%)".replace(",", " ")
        for label, value in counts.items()
    ]
    axis.legend(wedges, legend_labels, title="Wynik", loc="center left", bbox_to_anchor=(0.95, 0.5))
    axis.set_title("Balans klas w odds.csv: Team 1 vs Team 2")
    axis.set_aspect("equal")
    output_path = OUTPUT_DIR / "eda_team1_class_balance_odds.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_tier_distribution(data: pd.DataFrame) -> Path:
    """Save Tier 1 vs Regional/ERL distribution for odds.csv.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        Path to the saved figure.
    """
    counts = data["tier_segment"].value_counts().reindex(["Tier 1", "Regional / ERL"]).fillna(0)
    colors = ["#2E7D32", "#F9A825"]
    fig, axis = plt.subplots(figsize=(8, 6))
    total = counts.sum()
    wedges, _ = axis.pie(
        counts.values,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 3},
    )
    axis.text(0, 0.08, f"{int(total):,}".replace(",", " "), ha="center", fontsize=18, fontweight="bold")
    axis.text(0, -0.10, "meczów", ha="center", fontsize=11, color="#455A64")
    legend_labels = [
        f"{label}: {int(value):,} ({value / total * 100:.1f}%)".replace(",", " ")
        for label, value in counts.items()
    ]
    axis.legend(wedges, legend_labels, title="Segment", loc="center left", bbox_to_anchor=(0.95, 0.5))
    axis.set_title("Struktura zbioru odds.csv: Tier 1 vs Regional / ERL")
    axis.set_aspect("equal")
    output_path = OUTPUT_DIR / "eda_tier_distribution_odds.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_tier_market_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Compute opening and closing market metrics by tier segment.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        DataFrame with sample size, AUC, and LogLoss by tier.
    """
    rows = []
    for tier, group in data.groupby("tier_segment"):
        row = {
            "tier_segment": tier,
            "matches_total": int(len(group)),
            "share_pct": float(len(group) / len(data) * 100.0),
        }
        for label, column in [
            ("open", "p_home_open_novig"),
            ("close", "p_home_close_novig"),
        ]:
            valid = group.dropna(subset=[column, "t1_win"])
            y_true = valid["t1_win"].astype(int)
            y_prob = valid[column]
            row[f"{label}_n"] = int(len(valid))
            row[f"{label}_auc"] = float(roc_auc_score(y_true, y_prob))
            row[f"{label}_logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        rows.append(row)
    result = pd.DataFrame(rows)
    order = pd.CategoricalDtype(["Tier 1", "Regional / ERL"], ordered=True)
    result["tier_segment"] = result["tier_segment"].astype(order)
    return result.sort_values("tier_segment")


def save_tier_market_performance(data: pd.DataFrame) -> Path:
    """Save a paired AUC chart for market performance by tier.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        Path to the saved figure.
    """
    metrics = compute_tier_market_performance(data)
    metrics.to_csv(OUTPUT_DIR / "eda_tier_market_performance.csv", index=False)

    labels = metrics["tier_segment"].tolist()
    x_positions = np.arange(len(labels))
    width = 0.34
    fig, axis = plt.subplots(figsize=(9, 5.5))
    open_bars = axis.bar(
        x_positions - width / 2,
        metrics["open_auc"],
        width,
        label="Opening odds",
        color="#42A5F5",
    )
    close_bars = axis.bar(
        x_positions + width / 2,
        metrics["close_auc"],
        width,
        label="Closing odds",
        color="#5E35B1",
    )
    axis.set_title("Performance rynku według tieru rozgrywek")
    axis.set_ylabel("AUC")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.68, 0.77)
    axis.grid(axis="y", alpha=0.30, linestyle="--")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for bars in [open_bars, close_bars]:
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    output_path = OUTPUT_DIR / "eda_tier_market_performance.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_bookmaker_coverage_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Compute bookmaker-level opening odds coverage and predictive AUC.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        DataFrame with opening coverage, matches, AUC, and LogLoss per bookmaker.
    """
    rows = []
    for bookmaker in BOOKMAKERS:
        home_column = f"odds1_{bookmaker}_open"
        away_column = f"odds2_{bookmaker}_open"
        if home_column not in data.columns or away_column not in data.columns:
            continue

        probability = no_vig_home_probability(
            pd.to_numeric(data[home_column], errors="coerce"),
            pd.to_numeric(data[away_column], errors="coerce"),
        )
        valid = data.assign(p_bookmaker_open=probability).dropna(
            subset=["p_bookmaker_open", "t1_win"]
        )
        y_true = valid["t1_win"].astype(int)
        y_prob = valid["p_bookmaker_open"]
        rows.append(
            {
                "bookmaker": bookmaker,
                "open_matches": int(len(valid)),
                "open_coverage_pct": float(len(valid) / len(data) * 100.0),
                "open_auc": float(roc_auc_score(y_true, y_prob)),
                "open_logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
            }
        )

    return pd.DataFrame(rows).sort_values("open_auc", ascending=False)


def save_bookmaker_coverage_performance(data: pd.DataFrame) -> Path:
    """Save chart combining bookmaker AUC with opening odds coverage.

    Args:
        data: Prepared odds DataFrame.

    Returns:
        Path to the saved figure.
    """
    metrics = compute_bookmaker_coverage_performance(data)
    metrics.to_csv(OUTPUT_DIR / "eda_bookmaker_coverage_performance.csv", index=False)

    labels = metrics["bookmaker"].str.replace("_", " ").str.title()
    x_positions = np.arange(len(metrics))
    fig, auc_axis = plt.subplots(figsize=(11, 5.8))
    bars = auc_axis.bar(
        x_positions,
        metrics["open_auc"],
        color="#42A5F5",
        alpha=0.88,
        label="AUC opening",
    )
    auc_axis.set_title("AUC bukmachera trzeba czytać razem z coverage")
    auc_axis.set_ylabel("AUC opening odds")
    auc_axis.set_ylim(0.70, 0.755)
    auc_axis.set_xticks(x_positions)
    auc_axis.set_xticklabels(labels, rotation=0)
    auc_axis.grid(axis="y", alpha=0.30, linestyle="--")
    auc_axis.spines["top"].set_visible(False)

    coverage_axis = auc_axis.twinx()
    coverage_axis.plot(
        x_positions,
        metrics["open_coverage_pct"],
        color="#D84315",
        marker="o",
        linewidth=2.5,
        label="Coverage opening [%]",
    )
    coverage_axis.set_ylabel("Coverage opening [%]")
    coverage_axis.set_ylim(0, 100)
    coverage_axis.spines["top"].set_visible(False)

    for bar, coverage, matches in zip(
        bars,
        metrics["open_coverage_pct"],
        metrics["open_matches"],
    ):
        auc_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0015,
            f"AUC {bar.get_height():.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        coverage_axis.text(
            bar.get_x() + bar.get_width() / 2,
            coverage + 3,
            f"{coverage:.0f}%\nN={matches:,}".replace(",", " "),
            ha="center",
            fontsize=8,
            color="#D84315",
            fontweight="bold",
        )

    lines, labels_left = auc_axis.get_legend_handles_labels()
    lines_right, labels_right = coverage_axis.get_legend_handles_labels()
    auc_axis.legend(lines + lines_right, labels_left + labels_right, loc="lower right")
    output_path = OUTPUT_DIR / "eda_bookmaker_coverage_performance.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_market_format_difficulty() -> Path:
    """Save market performance by BoN format from generated EDA metrics.

    Returns:
        Path to the saved figure.
    """
    metrics_path = ROOT_DIR / "docs" / "assets" / "eda_point4" / "market_auc_logloss.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["segment"].isin(["Bo1", "Bo3", "Bo5"])]
    metrics = metrics.set_index("segment").loc[["Bo1", "Bo3", "Bo5"]].reset_index()

    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(metrics["segment"], metrics["open_auc"], color=["#90CAF9", "#A5D6A7", "#FFCC80"])
    axis.set_title("Market performance po formacie meczu")
    axis.set_ylabel("AUC opening odds")
    axis.set_ylim(0.62, 0.80)
    axis.grid(axis="y", alpha=0.25)
    for bar, matches in zip(bars, metrics["matches"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"AUC {bar.get_height():.3f}\nN={matches:,}".replace(",", " "),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    output_path = OUTPUT_DIR / "eda_market_format_difficulty.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate all standalone EDA figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    odds = load_odds()
    golgg_matches = load_golgg_matches()
    paths = [
        save_format_evolution(golgg_matches),
        save_favorite_probability_distribution(odds),
        save_team1_balance(odds),
        save_tier_distribution(odds),
        save_tier_market_performance(odds),
        save_bookmaker_coverage_performance(odds),
        save_market_format_difficulty(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
