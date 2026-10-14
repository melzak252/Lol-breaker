"""Generate EDA artefacts for whitepaper point 4.

The script uses only the source-of-truth datasets identified for the thesis:
``golgg_matches.json`` and ``odds.csv``. It creates compact tables and figures
for the chapter about sport/game EDA and bookmaker-market EDA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score

from src.utils.golgg_schema import (
    games,
    players1,
    players2,
    score1,
    score2,
    team1_name,
    team2_name,
    match_tournament,
)
from src.visualization.thesis_style import (
    DARK_TEXT,
    PASTEL_BLUE,
    PASTEL_ORANGE,
    PASTEL_PALETTE,
    apply_thesis_style,
    palette as thesis_palette,
)


ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "eda_point4"

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


def configure_plot_style() -> None:
    """Configure a consistent thesis-friendly plotting style."""
    apply_thesis_style(context="paper")


def load_json_matches(path: Path) -> list[dict[str, Any]]:
    """Load GOL.GG JSON matches.

    Args:
        path: Path to ``golgg_matches.json``.

    Returns:
        Parsed list of match dictionaries.
    """

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Expected golgg_matches.json to contain a list.")
    return [item for item in payload if isinstance(item, dict)]


def infer_best_of(score_1: int, score_2: int, games_count: int) -> int:
    """Infer standard Best-of-N format from score and games count.

    Args:
        score_1: Team 1 match score.
        score_2: Team 2 match score.
        games_count: Number of game payloads in the match.

    Returns:
        Standardized BoN value: 1, 3, or 5.
    """

    winner_score = max(int(score_1), int(score_2))
    if winner_score <= 1 and games_count <= 1:
        return 1
    if winner_score <= 2 and games_count <= 3:
        return 3
    return 5


def build_match_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a compact match-level GOL.GG DataFrame.

    Args:
        matches: Parsed GOL.GG match dictionaries.

    Returns:
        Match-level DataFrame with date, format and outcome fields.
    """

    rows: list[dict[str, Any]] = []
    for match in matches:
        match_score_1 = score1(match)
        match_score_2 = score2(match)
        game_payload = games(match)
        players_1 = players1(match)
        players_2 = players2(match)
        rows.append(
            {
                "match_id": str(match.get("match_id")),
                "date": match.get("date"),
                "tournament": match_tournament(match),
                "team_1": team1_name(match),
                "team_2": team2_name(match),
                "score_1": match_score_1,
                "score_2": match_score_2,
                "t1_win": match_score_1 > match_score_2,
                "games_count": len(game_payload),
                "players_1_count": len(players_1) if isinstance(players_1, list) else 0,
                "players_2_count": len(players_2) if isinstance(players_2, list) else 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame["year"] = frame["date"].dt.year
    frame["bon"] = frame.apply(
        lambda row: infer_best_of(row["score_1"], row["score_2"], row["games_count"]),
        axis=1,
    )
    frame["tier_segment"] = np.where(
        frame["tournament"].apply(is_tier1),
        "Tier 1",
        "Regional / ERL",
    )
    frame["has_full_rosters"] = (frame["players_1_count"] == 5) & (
        frame["players_2_count"] == 5
    )
    return frame


def build_game_frame(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Build game-level rows for side-bias and GD15 analyses.

    Args:
        matches: Parsed GOL.GG match dictionaries.

    Returns:
        Game-level DataFrame.
    """

    rows: list[dict[str, Any]] = []
    for match in matches:
        for game in games(match):
            if not isinstance(game, dict):
                continue
            rows.append(
                {
                    "match_id": str(match.get("match_id")),
                    "date": match.get("date"),
                    "t1_win": bool(game.get("t1_win")),
                    "team_gd15": sum_team_gd15(game.get("t1_players") or {}),
                    "has_gd15": has_any_gd15(game.get("t1_players") or {}),
                }
            )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["year"] = frame["date"].dt.year
    return frame


def sum_team_gd15(players: dict[str, Any]) -> float:
    """Sum player-level GD@15 values for a team.

    Args:
        players: Role-to-player dictionary from a game payload.

    Returns:
        Sum of non-missing GD@15 values.
    """

    total = 0.0
    for player_data in players.values():
        stats = player_data.get("stats", {}) if isinstance(player_data, dict) else {}
        value = stats.get("gd@15")
        if value is not None and pd.notna(value):
            total += float(value)
    return total


def has_any_gd15(players: dict[str, Any]) -> bool:
    """Check whether a team payload contains any GD@15 value.

    Args:
        players: Role-to-player dictionary from a game payload.

    Returns:
        True if at least one player has GD@15.
    """

    for player_data in players.values():
        stats = player_data.get("stats", {}) if isinstance(player_data, dict) else {}
        if stats.get("gd@15") is not None and pd.notna(stats.get("gd@15")):
            return True
    return False


def is_tier1(tournament: object) -> bool:
    """Classify a tournament as Tier 1 using transparent keyword rules.

    Args:
        tournament: Tournament name.

    Returns:
        True for likely major-region/international events.
    """

    text = str(tournament)
    if "Pro League" in text and "Oceanic" not in text and "Continental" not in text:
        return True
    return any(keyword in text for keyword in TIER1_KEYWORDS)


def load_odds(path: Path) -> pd.DataFrame:
    """Load and enrich the mapped odds dataset.

    Args:
        path: Path to ``odds.csv``.

    Returns:
        Enriched odds DataFrame.
    """

    frame = pd.read_csv(path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["odds_date"], errors="coerce")
    frame["year"] = frame["date"].dt.year
    frame["bon"] = frame.apply(
        lambda row: infer_best_of(row["t1_score"], row["t2_score"], int(row["t1_score"] + row["t2_score"])),
        axis=1,
    )
    frame["is_tier1"] = frame["tournament"].apply(is_tier1)
    frame["prob_open_t1"] = no_vig_home_probability(
        frame["avg_open_home"], frame["avg_open_away"]
    )
    frame["prob_close_t1"] = no_vig_home_probability(
        frame["avg_odds_home"], frame["avg_odds_away"]
    )
    frame["prob_shift_close_minus_open"] = frame["prob_close_t1"] - frame["prob_open_t1"]
    return frame


def no_vig_home_probability(home_odds: pd.Series, away_odds: pd.Series) -> pd.Series:
    """Convert two-way decimal odds to normalized home probability.

    Args:
        home_odds: Decimal odds for team 1/home.
        away_odds: Decimal odds for team 2/away.

    Returns:
        No-vig implied probability for team 1.
    """

    valid = (home_odds > 1.0) & (away_odds > 1.0)
    implied_home = 1.0 / home_odds.where(valid)
    implied_away = 1.0 / away_odds.where(valid)
    return implied_home / (implied_home + implied_away)


def safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Calculate AUC safely.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.

    Returns:
        AUC or NaN when impossible.
    """

    data = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if data["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(data["y"], data["p"]))


def safe_logloss(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Calculate binary log loss safely.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities.

    Returns:
        LogLoss or NaN when impossible.
    """

    data = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    if data["y"].nunique() < 2:
        return float("nan")
    return float(log_loss(data["y"], data["p"].clip(0.001, 0.999)))


def annotate_bars(ax: plt.Axes, suffix: str = "", decimals: int = 0) -> None:
    """Annotate bar charts with value labels.

    Args:
        ax: Matplotlib axes containing bar patches.
        suffix: Text suffix appended to the value.
        decimals: Number of decimals to show.
    """

    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        label = f"{height:.{decimals}f}{suffix}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )


def save_bar(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    path: Path,
    palette: str | list[str] = PASTEL_PALETTE,
    value_suffix: str = "",
    value_decimals: int = 0,
) -> None:
    """Save a styled bar chart.

    Args:
        data: Plot data.
        x_col: X-axis column.
        y_col: Y-axis column.
        title: Chart title.
        path: Output PNG path.
        palette: Seaborn palette name or explicit color list.
        value_suffix: Text suffix for bar labels.
        value_decimals: Number of decimals for bar labels.
    """

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_data = data.copy()
    plot_data[x_col] = plot_data[x_col].astype(str)
    sns.barplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=x_col,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    annotate_bars(ax, suffix=value_suffix, decimals=value_decimals)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_auc_bar(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    path: Path,
    palette: str | list[str] = PASTEL_PALETTE,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Save a compact single-line AUC bar chart.

    Args:
        data: Plot data.
        x_col: X-axis column.
        y_col: AUC column to plot.
        title: Chart title.
        path: Output PNG path.
        palette: Seaborn palette name or explicit color list.
        ylim: Optional y-axis limits.
    """

    fig_width = max(7.5, min(11.0, 1.35 * len(data) + 4.5))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    plot_data = data.copy()
    plot_data[x_col] = plot_data[x_col].astype(str)
    sns.barplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=x_col,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("AUC — kursy otwarcia")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax, decimals=3)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout(pad=0.9)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_market_open_close_comparison(
    market_metrics: pd.DataFrame,
    path: Path,
) -> None:
    """Save an open-vs-close market LogLoss comparison figure.

    Args:
        market_metrics: Market quality metrics with the aggregate ``all`` row.
        path: Output PNG path.
    """

    overall = market_metrics.loc[market_metrics["segment"] == "all"].iloc[0]
    plot_data = pd.DataFrame(
        {
            "benchmark": ["Market Open", "Market Close"],
            "logloss": [overall["open_logloss"], overall["close_logloss"]],
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    palette = [PASTEL_BLUE, PASTEL_ORANGE]

    sns.barplot(
        data=plot_data,
        x="benchmark",
        y="logloss",
        hue="benchmark",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title("LogLoss kursów otwarcia i zamknięcia")
    ax.set_xlabel("")
    ax.set_ylabel("LogLoss (niżej = lepiej)")
    loss_min = float(plot_data["logloss"].min())
    loss_max = float(plot_data["logloss"].max())
    loss_margin = max(0.003, 0.35 * (loss_max - loss_min))
    ax.set_ylim(loss_min - loss_margin, loss_max + loss_margin)
    annotate_bars(ax, decimals=3)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax, left=True, bottom=True)

    fig.tight_layout(pad=0.9)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_grouped_bar(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    path: Path,
    palette: str | list[str] = PASTEL_PALETTE,
    value_suffix: str = "",
    value_decimals: int = 3,
) -> None:
    """Save a styled grouped bar chart.

    Args:
        data: Plot data.
        x_col: X-axis column.
        y_col: Y-axis column.
        hue_col: Grouping column.
        title: Chart title.
        path: Output PNG path.
        palette: Seaborn palette name or explicit color list.
        value_suffix: Text suffix for bar labels.
        value_decimals: Number of decimals for bar labels.
    """

    fig, ax = plt.subplots(figsize=(10, 5.2))
    plot_data = data.copy()
    plot_data[x_col] = plot_data[x_col].astype(str)
    sns.barplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=30)
    annotate_bars(ax, suffix=value_suffix, decimals=value_decimals)
    ax.legend(title=hue_col, frameon=True)
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_side_pie(data: pd.DataFrame, path: Path) -> None:
    """Save a pie chart for team-side proxy win distribution.

    Args:
        data: Side win-rate table.
        path: Output PNG path.
    """

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    labels = ["Team 1 / blue proxy", "Team 2 / red proxy"]
    colors = [PASTEL_BLUE, PASTEL_ORANGE]
    _, _, autotexts = ax.pie(
        data["win_rate_pct"],
        labels=labels,
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        explode=(0.035, 0.0),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11, "fontweight": "bold"},
        radius=1.08,
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    ax.set_title("Rozkład zwycięstw według strony / pozycji w payloadzie")
    ax.axis("equal")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def generate_sport_eda(match_df: pd.DataFrame, game_df: pd.DataFrame) -> None:
    """Generate sport-side EDA tables and figures.

    Args:
        match_df: Match-level GOL.GG data.
        game_df: Game-level GOL.GG data.
    """

    matches_per_year = match_df.groupby("year").size().reset_index(name="matches")
    games_per_year = game_df.groupby("year").size().reset_index(name="games")
    bon_distribution = match_df["bon"].value_counts().sort_index().reset_index()
    bon_distribution.columns = ["bon", "matches"]
    tier_distribution = match_df["tier_segment"].value_counts().reset_index()
    tier_distribution.columns = ["tier_segment", "matches"]
    tier_distribution["share_pct"] = (
        tier_distribution["matches"] / tier_distribution["matches"].sum() * 100.0
    ).round(2)
    side_win = pd.DataFrame(
        {
            "side_proxy": ["team_1_blue_proxy", "team_2_red_proxy"],
            "win_rate": [
                float(game_df["t1_win"].mean()),
                1.0 - float(game_df["t1_win"].mean()),
            ],
            "win_rate_pct": [
                float(game_df["t1_win"].mean()) * 100.0,
                (1.0 - float(game_df["t1_win"].mean())) * 100.0,
            ],
            "games": [int(game_df.shape[0]), int(game_df.shape[0])],
        }
    )
    gd15 = game_df[game_df["has_gd15"]].copy()
    gd15["gd15_bin"] = pd.cut(gd15["team_gd15"], bins=np.arange(-10000, 10001, 1000))
    gd15_bins = gd15.groupby("gd15_bin", observed=True).agg(
        win_rate=("t1_win", "mean"), games=("t1_win", "size")
    )
    gd15_bins = gd15_bins[gd15_bins["games"] >= 50].reset_index()
    gd15_bins["gd15_mid"] = gd15_bins["gd15_bin"].map(lambda value: value.mid)

    matches_per_year.to_csv(ASSETS_DIR / "sport_matches_per_year.csv", index=False)
    games_per_year.to_csv(ASSETS_DIR / "sport_games_per_year.csv", index=False)
    bon_distribution.to_csv(ASSETS_DIR / "sport_bon_distribution.csv", index=False)
    tier_distribution.to_csv(ASSETS_DIR / "sport_tier_distribution.csv", index=False)
    side_win.to_csv(ASSETS_DIR / "sport_side_win_rate.csv", index=False)
    gd15_bins.to_csv(ASSETS_DIR / "sport_gd15_winrate_bins.csv", index=False)

    save_bar(
        matches_per_year,
        "year",
        "matches",
        "Liczba meczów w zbiorze GOL.GG",
        ASSETS_DIR / "sport_matches_per_year.png",
        palette=thesis_palette(len(matches_per_year)),
    )
    save_bar(
        games_per_year,
        "year",
        "games",
        "Liczba pojedynczych gier w zbiorze GOL.GG",
        ASSETS_DIR / "sport_games_per_year.png",
        palette=thesis_palette(len(games_per_year)),
    )
    save_bar(
        bon_distribution,
        "bon",
        "matches",
        "Dystrybucja formatów Best-of-N",
        ASSETS_DIR / "sport_bon_distribution.png",
        palette=thesis_palette(len(bon_distribution)),
    )
    save_bar(
        tier_distribution,
        "tier_segment",
        "matches",
        "Segmentacja meczów: Tier 1 vs ligi regionalne",
        ASSETS_DIR / "sport_tier_distribution.png",
        palette=thesis_palette(len(tier_distribution)),
    )
    save_side_pie(side_win, ASSETS_DIR / "sport_side_win_rate.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=gd15_bins,
        x=gd15_bins["gd15_mid"].astype(float),
        y="win_rate",
        size="games",
        sizes=(60, 420),
        color=PASTEL_BLUE,
        alpha=0.72,
        ax=ax,
        legend=False,
    )
    sns.lineplot(
        data=gd15_bins,
        x=gd15_bins["gd15_mid"].astype(float),
        y="win_rate",
        color=PASTEL_ORANGE,
        linewidth=2.5,
        ax=ax,
    )
    ax.axhline(0.5, color=DARK_TEXT, linestyle="--", linewidth=1)
    ax.axvline(0, color=DARK_TEXT, linestyle="--", linewidth=1)
    ax.set_title("Zależność Win Rate od różnicy złota w 15. minucie")
    ax.set_xlabel("Suma GD@15 drużyny 1 — środek przedziału")
    ax.set_ylabel("Win Rate drużyny 1")
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "sport_gd15_vs_winrate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_market_eda(odds: pd.DataFrame) -> None:
    """Generate bookmaker-market EDA tables and figures.

    Args:
        odds: Enriched mapped odds DataFrame.
    """

    margins = summarize_margins(odds)
    market_metrics = summarize_market_metrics(odds)
    bookmaker_metrics = summarize_bookmaker_market_metrics(odds)
    bookmaker_open_summary = summarize_bookmaker_open_quality(
        margins,
        bookmaker_metrics,
        market_metrics,
        len(odds),
    )
    arbitrage = summarize_arbitrage(odds)
    movement = odds["prob_shift_close_minus_open"].dropna().describe().reset_index()
    movement.columns = ["statistic", "value"]

    margins.to_csv(ASSETS_DIR / "market_bookmaker_margins.csv", index=False)
    market_metrics.to_csv(ASSETS_DIR / "market_auc_logloss.csv", index=False)
    bookmaker_metrics.to_csv(
        ASSETS_DIR / "market_bookmaker_auc_logloss.csv",
        index=False,
    )
    bookmaker_open_summary.to_csv(
        ASSETS_DIR / "market_bookmaker_open_quality_summary.csv",
        index=False,
    )
    arbitrage.to_csv(ASSETS_DIR / "market_arbitrage_by_year.csv", index=False)
    movement.to_csv(ASSETS_DIR / "market_open_close_movement_summary.csv", index=False)

    save_bar(
        margins,
        "bookmaker",
        "open_margin_pct",
        "Średnia marża opening u bukmacherów",
        ASSETS_DIR / "market_bookmaker_margins.png",
        palette=thesis_palette(len(margins)),
        value_suffix="%",
        value_decimals=2,
    )
    save_market_open_close_comparison(
        market_metrics,
        ASSETS_DIR / "market_auc_overall.png",
    )
    save_auc_bar(
        market_metrics[market_metrics["segment"].isin(["Bo1", "Bo3", "Bo5"])],
        "segment",
        "open_auc",
        "Siła rynku według formatu meczu — AUC opening",
        ASSETS_DIR / "market_auc_by_bon.png",
        palette=thesis_palette(3),
        ylim=(0.64, 0.79),
    )
    save_auc_bar(
        market_metrics[market_metrics["segment"].isin(["Tier1", "Regional_ERL"])],
        "segment",
        "open_auc",
        "Siła rynku według tieru rozgrywek — AUC opening",
        ASSETS_DIR / "market_auc_by_tier.png",
        palette=thesis_palette(2),
        ylim=(0.70, 0.75),
    )
    save_auc_bar(
        bookmaker_metrics,
        "bookmaker",
        "open_auc",
        "Siła predykcyjna indywidualnych bukmacherów — AUC opening",
        ASSETS_DIR / "market_auc_by_bookmaker.png",
        palette=thesis_palette(len(bookmaker_metrics)),
        ylim=(0.70, 0.755),
    )
    save_bar(
        arbitrage,
        "year",
        "raw_arbitrage_count",
        "Okazje arbitrażowe brutto według roku",
        ASSETS_DIR / "market_raw_arbitrage_by_year.png",
        palette=thesis_palette(len(arbitrage)),
    )
    save_bar(
        arbitrage,
        "year",
        "tax_arbitrage_count",
        "Okazje arbitrażowe po podatku 12% według roku",
        ASSETS_DIR / "market_tax_arbitrage_by_year.png",
        palette=thesis_palette(len(arbitrage)),
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(
        odds["prob_shift_close_minus_open"].dropna(),
        bins=100,
        kde=True,
        color=PASTEL_BLUE,
        ax=ax,
    )
    ax.axvline(0, color=PASTEL_ORANGE, linestyle="--", linewidth=2)
    ax.set_title("Korekta rynku: prawdopodobieństwo closing minus opening")
    ax.set_xlabel("Zmiana no-vig probability dla drużyny 1")
    ax.set_ylabel("Liczba meczów")
    sns.despine(ax=ax, left=True, bottom=True)
    fig.tight_layout()
    fig.savefig(
        ASSETS_DIR / "market_open_close_probability_shift.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def summarize_margins(odds: pd.DataFrame) -> pd.DataFrame:
    """Summarize opening/closing bookmaker overround.

    Args:
        odds: Enriched mapped odds DataFrame.

    Returns:
        Margin table by bookmaker.
    """

    rows = []
    for bookmaker in BOOKMAKERS:
        open_home = odds[f"odds1_{bookmaker}_open"]
        open_away = odds[f"odds2_{bookmaker}_open"]
        close_home = odds[f"odds1_{bookmaker}_close"]
        close_away = odds[f"odds2_{bookmaker}_close"]
        open_valid = (open_home > 1.0) & (open_away > 1.0)
        close_valid = (close_home > 1.0) & (close_away > 1.0)
        rows.append(
            {
                "bookmaker": bookmaker,
                "open_matches": int(open_valid.sum()),
                "open_coverage": round(float(open_valid.mean()), 4),
                "open_margin_pct": round(float(((1 / open_home[open_valid]) + (1 / open_away[open_valid]) - 1).mean() * 100), 4),
                "close_margin_pct": round(float(((1 / close_home[close_valid]) + (1 / close_away[close_valid]) - 1).mean() * 100), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("open_margin_pct")


def summarize_market_metrics(odds: pd.DataFrame) -> pd.DataFrame:
    """Summarize AUC/LogLoss for opening and closing market averages.

    Args:
        odds: Enriched mapped odds DataFrame.

    Returns:
        Segment-level market metric table.
    """

    segments = [("all", odds)]
    segments.extend((f"Bo{bon}", odds[odds["bon"] == bon]) for bon in [1, 3, 5])
    segments.extend(
        [
            ("Tier1", odds[odds["is_tier1"]]),
            ("Regional_ERL", odds[~odds["is_tier1"]]),
        ]
    )
    rows = []
    for segment_name, segment in segments:
        rows.append(
            {
                "segment": segment_name,
                "matches": int(segment.dropna(subset=["prob_open_t1"]).shape[0]),
                "open_auc": round(safe_auc(segment["t1_win"], segment["prob_open_t1"]), 5),
                "close_auc": round(safe_auc(segment["t1_win"], segment["prob_close_t1"]), 5),
                "open_logloss": round(safe_logloss(segment["t1_win"], segment["prob_open_t1"]), 5),
                "close_logloss": round(safe_logloss(segment["t1_win"], segment["prob_close_t1"]), 5),
            }
        )
    return pd.DataFrame(rows)


def summarize_bookmaker_market_metrics(odds: pd.DataFrame) -> pd.DataFrame:
    """Summarize predictive metrics for individual bookmakers.

    Args:
        odds: Enriched mapped odds DataFrame.

    Returns:
        Bookmaker-level AUC/LogLoss table.
    """

    rows = []
    for bookmaker in BOOKMAKERS:
        open_prob = no_vig_home_probability(
            odds[f"odds1_{bookmaker}_open"],
            odds[f"odds2_{bookmaker}_open"],
        )
        close_prob = no_vig_home_probability(
            odds[f"odds1_{bookmaker}_close"],
            odds[f"odds2_{bookmaker}_close"],
        )
        rows.append(
            {
                "bookmaker": bookmaker,
                "open_matches": int(open_prob.notna().sum()),
                "close_matches": int(close_prob.notna().sum()),
                "open_coverage_pct": round(float(open_prob.notna().mean() * 100.0), 2),
                "close_coverage_pct": round(float(close_prob.notna().mean() * 100.0), 2),
                "open_auc": round(safe_auc(odds["t1_win"], open_prob), 5),
                "close_auc": round(safe_auc(odds["t1_win"], close_prob), 5),
                "open_logloss": round(safe_logloss(odds["t1_win"], open_prob), 5),
                "close_logloss": round(safe_logloss(odds["t1_win"], close_prob), 5),
            }
        )
    return pd.DataFrame(rows).sort_values("open_auc", ascending=False)


def summarize_bookmaker_open_quality(
    margins: pd.DataFrame,
    bookmaker_metrics: pd.DataFrame,
    market_metrics: pd.DataFrame,
    total_matches: int,
) -> pd.DataFrame:
    """Combine bookmaker coverage, margin and predictive quality.

    Args:
        margins: Bookmaker-level overround summary.
        bookmaker_metrics: Bookmaker-level AUC and LogLoss summary.
        market_metrics: Segment-level market average metrics.
        total_matches: Number of mapped odds rows used as coverage denominator.

    Returns:
        Combined opening-odds table with one additional market-average row.
    """

    summary = margins.merge(
        bookmaker_metrics[["bookmaker", "open_auc", "open_logloss"]],
        on="bookmaker",
        how="left",
    )[
        [
            "bookmaker",
            "open_matches",
            "open_coverage",
            "open_margin_pct",
            "open_auc",
            "open_logloss",
        ]
    ]
    summary["open_coverage_pct"] = (summary["open_coverage"] * 100.0).round(2)

    all_market = market_metrics.loc[market_metrics["segment"] == "all"].iloc[0]
    weighted_margin = float(
        np.average(summary["open_margin_pct"], weights=summary["open_matches"])
    )
    market_row = pd.DataFrame(
        [
            {
                "bookmaker": "market_average",
                "open_matches": int(all_market["matches"]),
                "open_coverage": round(float(all_market["matches"] / total_matches), 4),
                "open_margin_pct": round(weighted_margin, 4),
                "open_auc": float(all_market["open_auc"]),
                "open_logloss": float(all_market["open_logloss"]),
                "open_coverage_pct": round(float(all_market["matches"] / total_matches * 100.0), 2),
            }
        ]
    )

    summary = summary.drop(columns=["open_coverage"])
    market_row = market_row.drop(columns=["open_coverage"])
    return pd.concat([summary, market_row], ignore_index=True)


def melt_market_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert segment-level metrics to long format for plotting.

    Args:
        metrics: Wide segment metric table.

    Returns:
        Long table with one AUC row per market line.
    """

    return pd.concat(
        [
            metrics[["segment", "open_auc"]].rename(columns={"open_auc": "auc"}).assign(
                market_line="Opening"
            ),
            metrics[["segment", "close_auc"]].rename(columns={"close_auc": "auc"}).assign(
                market_line="Closing"
            ),
        ],
        ignore_index=True,
    )


def melt_bookmaker_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert bookmaker metrics to long format for plotting.

    Args:
        metrics: Wide bookmaker metric table.

    Returns:
        Long table with one AUC row per bookmaker and market line.
    """

    return pd.concat(
        [
            metrics[["bookmaker", "open_auc"]].rename(columns={"open_auc": "auc"}).assign(
                market_line="Opening"
            ),
            metrics[["bookmaker", "close_auc"]].rename(columns={"close_auc": "auc"}).assign(
                market_line="Closing"
            ),
        ],
        ignore_index=True,
    )


def summarize_arbitrage(odds: pd.DataFrame) -> pd.DataFrame:
    """Count raw and Polish-tax-adjusted arbitrage opportunities.

    Args:
        odds: Enriched mapped odds DataFrame.

    Returns:
        Year-level arbitrage counts.
    """

    rows = []
    for year, group in odds.groupby("year"):
        raw_count = 0
        tax_count = 0
        for _, row in group.iterrows():
            home = max_valid_odds(row, "odds1", "open")
            away = max_valid_odds(row, "odds2", "open")
            if home <= 1.0 or away <= 1.0:
                continue
            if (1.0 / home) + (1.0 / away) < 1.0:
                raw_count += 1
            if (1.0 / (home * 0.88)) + (1.0 / (away * 0.88)) < 1.0:
                tax_count += 1
        rows.append(
            {
                "year": int(year),
                "matches": int(group.shape[0]),
                "raw_arbitrage_count": raw_count,
                "tax_arbitrage_count": tax_count,
            }
        )
    return pd.DataFrame(rows)


def max_valid_odds(row: pd.Series, side_prefix: str, timing: str) -> float:
    """Return the maximum valid bookmaker odds in a row.

    Args:
        row: Odds row.
        side_prefix: ``odds1`` or ``odds2``.
        timing: ``open`` or ``close``.

    Returns:
        Maximum odds or 0.0 when unavailable.
    """

    values = [row.get(f"{side_prefix}_{bookmaker}_{timing}") for bookmaker in BOOKMAKERS]
    valid_values = [float(value) for value in values if pd.notna(value) and float(value) > 1.0]
    return max(valid_values) if valid_values else 0.0


def main() -> None:
    """Run the point-4 EDA generation workflow."""

    configure_plot_style()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_json_matches(PROJECT_ROOT / "data" / "golgg_matches.json")
    match_df = build_match_frame(matches)
    game_df = build_game_frame(matches)
    odds = load_odds(PROJECT_ROOT / "data" / "odds.csv")

    match_df.to_csv(ASSETS_DIR / "sport_match_level_profile.csv", index=False)
    game_df.to_csv(ASSETS_DIR / "sport_game_level_profile.csv", index=False)
    generate_sport_eda(match_df, game_df)
    generate_market_eda(odds)

    print(f"Generated point-4 EDA artefacts in: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
