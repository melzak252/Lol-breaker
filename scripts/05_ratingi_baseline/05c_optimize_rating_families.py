"""Optimize hyperparameters for rating-family baselines.

This experiment evaluates whether the apparent dominance of Player Glicko-2 is
partly caused by non-optimized hyperparameters in the remaining rating systems.
All configurations are evaluated chronologically on the odds-mapped thesis
sample. The ratings are updated using all available historical GOL.GG matches,
while metrics are computed only for matches that also have bookmaker odds.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glicko2 import Player
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ratings.elo import EloRating
from src.ratings.openskill_rating import OpenSkillRating
from src.ratings.plackett_luce import PlackettLuceRating
from src.ratings.thurstone import ThurstoneRating
from src.ratings.trueskill_rating import TrueSkillRating
from src.utils.golgg_schema import (
    game_score_for_match_team1,
    games,
    players1,
    players2,
    team1_id,
    team2_id,
)

MATCHES_PATH = ROOT_DIR / "data" / "golgg_matches.json"
ODDS_PATH = ROOT_DIR / "data" / "odds.csv"
OUTPUT_DIR = ROOT_DIR / "docs" / "assets" / "rating_family_optimization"
RANDOM_SEED = 42


@dataclass(frozen=True)
class MatchRecord:
    """Compact chronological match record for rating optimization."""

    match_id: int
    match_date: date
    team_1: str
    team_2: str
    players_1: tuple[str, ...]
    players_2: tuple[str, ...]
    y_true: int
    scores: tuple[int, ...]


class RatingSystemProtocol(Protocol):
    """Protocol for rating systems used in the optimization loop."""

    def predict_player_win_prob(
        self, players_1: list[str], players_2: list[str]
    ) -> float:
        """Predict probability that team 1 wins from player ratings."""

    def update_team(
        self, t1: str, t2: str, score_1: int, score_2: int
    ) -> None:
        """Update team ratings after one game."""

    def update_player(
        self,
        players_1: list[str],
        players_2: list[str],
        score_1: int,
        score_2: int,
    ) -> None:
        """Update player ratings after one game."""


class TunableGlickoRating:
    """Minimal Glicko-2 player/team rating system with tunable parameters."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_vol: float = 0.06,
        period_days: int = 7,
    ) -> None:
        """Initialize Glicko-2 ratings.

        Args:
            initial_rating: Initial rating assigned to unseen entities.
            initial_rd: Initial rating deviation assigned to unseen entities.
            initial_vol: Initial volatility assigned to unseen entities.
            period_days: Number of inactivity days corresponding to one Glicko
                ``did_not_compete`` period.
        """
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol
        self.period_days = period_days
        self.team_ratings: dict[str, Player] = {}
        self.player_ratings: dict[str, Player] = {}
        self.team_last_played: dict[str, date] = {}
        self.player_last_played: dict[str, date] = {}

    def _new_player(self) -> Player:
        """Create a new Glicko-2 player object."""
        return Player(self.initial_rating, self.initial_rd, self.initial_vol)

    def get_team_rating(self, team_id: str) -> Player:
        """Return current team rating, creating it if needed."""
        if team_id not in self.team_ratings:
            self.team_ratings[team_id] = self._new_player()
        return self.team_ratings[team_id]

    def get_player_rating(self, player_id: str) -> Player:
        """Return current player rating, creating it if needed."""
        if player_id not in self.player_ratings:
            self.player_ratings[player_id] = self._new_player()
        return self.player_ratings[player_id]

    @staticmethod
    def _g(rd: float) -> float:
        """Calculate Glicko-2 uncertainty scaling factor."""
        q = math.log(10) / 400
        return 1 / math.sqrt(1 + 3 * (q**2) * (rd**2) / (math.pi**2))

    def _expected_score(self, player_i: Player, player_j: Player) -> float:
        """Calculate expected score for one Glicko-2 entity vs another."""
        combined_rd = math.sqrt(player_i.rd**2 + player_j.rd**2)
        g_factor = self._g(combined_rd)
        exponent = -g_factor * (player_i.rating - player_j.rating) / 400
        return float(1 / (1 + 10**exponent))

    def _aggregate_players(self, players: list[Player]) -> tuple[float, float]:
        """Aggregate player ratings to team-level mean rating and RD."""
        if not players:
            return self.initial_rating, self.initial_rd
        ratings = [player.rating for player in players]
        rds = [player.rd**2 for player in players]
        return float(np.mean(ratings)), float(math.sqrt(np.mean(rds)))

    def update_before_match(
        self,
        t1: str,
        t2: str,
        players_1: list[str],
        players_2: list[str],
        match_date: date,
    ) -> None:
        """Apply inactivity RD decay before a match."""
        for player_id in players_1 + players_2:
            last_date = self.player_last_played.get(player_id)
            self._apply_time_decay(self.get_player_rating(player_id), last_date, match_date)
            self.player_last_played[player_id] = match_date
        for team_id in [t1, t2]:
            last_date = self.team_last_played.get(team_id)
            self._apply_time_decay(self.get_team_rating(team_id), last_date, match_date)
            self.team_last_played[team_id] = match_date

    def _apply_time_decay(
        self, player: Player, last_played_date: date | None, current_date: date
    ) -> None:
        """Increase RD for inactive entities."""
        if last_played_date is None:
            return
        days = (current_date - last_played_date).days
        if days <= 0:
            return
        for _ in range(days // self.period_days):
            player.did_not_compete()

    def predict_player_win_prob(
        self, players_1: list[str], players_2: list[str]
    ) -> float:
        """Predict team-1 win probability from player Glicko-2 ratings."""
        p1 = [self.get_player_rating(player_id) for player_id in players_1]
        p2 = [self.get_player_rating(player_id) for player_id in players_2]
        rating_1, rd_1 = self._aggregate_players(p1)
        rating_2, rd_2 = self._aggregate_players(p2)
        return self._expected_score(
            Player(int(rating_1), int(rd_1), self.initial_vol),
            Player(int(rating_2), int(rd_2), self.initial_vol),
        )

    def update_team(
        self, t1: str, t2: str, score_1: int, score_2: int
    ) -> None:
        """Update team Glicko-2 ratings after one game."""
        player_1 = self.get_team_rating(t1)
        player_2 = self.get_team_rating(t2)
        player_1.update_player([player_2.rating], [player_2.rd], [score_1])
        player_2.update_player([player_1.rating], [player_1.rd], [score_2])
        self.team_ratings[t1] = player_1
        self.team_ratings[t2] = player_2

    def update_player(
        self,
        players_1: list[str],
        players_2: list[str],
        score_1: int,
        score_2: int,
    ) -> None:
        """Update player Glicko-2 ratings after one game."""
        ratings_1 = [self.get_player_rating(player_id) for player_id in players_1]
        ratings_2 = [self.get_player_rating(player_id) for player_id in players_2]
        team_rating_1, team_rd_1 = self._aggregate_players(ratings_1)
        team_rating_2, team_rd_2 = self._aggregate_players(ratings_2)
        for rating, player_id in zip(ratings_1, players_1):
            rating.update_player([team_rating_2], [team_rd_2], [score_1])
            self.player_ratings[player_id] = rating
        for rating, player_id in zip(ratings_2, players_2):
            rating.update_player([team_rating_1], [team_rd_1], [score_2])
            self.player_ratings[player_id] = rating


def configure_style() -> None:
    """Configure plotting style."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"savefig.dpi": 300, "axes.titleweight": "bold"})


def load_common_market_ids() -> set[int]:
    """Load match identifiers with valid opening odds."""
    odds = pd.read_csv(ODDS_PATH, usecols=["golgg_match_id", "avg_open_home", "avg_open_away"])
    odds = odds.dropna(subset=["avg_open_home", "avg_open_away"])
    return set(pd.to_numeric(odds["golgg_match_id"], errors="coerce").dropna().astype(int))


def load_matches() -> list[MatchRecord]:
    """Load GOL.GG matches as chronological compact records."""
    with MATCHES_PATH.open("r", encoding="utf-8") as handle:
        raw_matches: list[dict[str, Any]] = json.load(handle)

    records: list[MatchRecord] = []
    for match in raw_matches:
        if bool(match.get("draw")):
            continue
        p1 = tuple(players1(match))
        p2 = tuple(players2(match))
        if not p1 or not p2:
            continue
        scores = tuple(game_score_for_match_team1(match, game) for game in games(match))
        if not scores:
            continue
        records.append(
            MatchRecord(
                match_id=int(match["match_id"]),
                match_date=date.fromisoformat(match["date"]),
                team_1=team1_id(match),
                team_2=team2_id(match),
                players_1=p1,
                players_2=p2,
                y_true=int(sum(scores) > len(scores) / 2),
                scores=scores,
            )
        )
    return sorted(records, key=lambda item: (item.match_date, item.match_id))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate expected calibration error for binary probabilities."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if not np.any(in_bin):
            continue
        ece += abs(np.mean(y_true[in_bin]) - np.mean(y_prob[in_bin])) * np.mean(in_bin)
    return float(ece)


def calculate_metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float | int]:
    """Calculate standard probabilistic classification metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.clip(np.asarray(y_prob, dtype=float), 0.001, 0.999)
    return {
        "sample_size": int(len(y_true_array)),
        "auc": float(roc_auc_score(y_true_array, y_prob_array)),
        "logloss": float(log_loss(y_true_array, y_prob_array)),
        "brier": float(brier_score_loss(y_true_array, y_prob_array)),
        "ece": expected_calibration_error(y_true_array, y_prob_array),
        "accuracy_0_5": float(np.mean((y_prob_array >= 0.5) == y_true_array)),
    }


def build_system(system_name: str, params: dict[str, Any]) -> RatingSystemProtocol:
    """Instantiate a rating system for a named parameter configuration."""
    if system_name == "elo":
        return EloRating(**params)
    if system_name == "glicko2":
        return TunableGlickoRating(**params)
    if system_name == "trueskill":
        return TrueSkillRating(**params)
    if system_name == "openskill":
        return OpenSkillRating(**params)
    if system_name == "plackett_luce":
        return PlackettLuceRating(**params)
    if system_name == "thurstone_mosteller":
        return ThurstoneRating(**params)
    raise ValueError(f"Unsupported rating system: {system_name}")


def build_parameter_grid() -> list[tuple[str, dict[str, Any]]]:
    """Build a compact grid for all rating families under study."""
    configs: list[tuple[str, dict[str, Any]]] = []

    for k_player in [16.0, 24.0, 32.0, 48.0, 64.0]:
        configs.append(("elo", {"initial_rating": 1500.0, "k_team": 64.0, "k_player": k_player}))

    for initial_rd, initial_vol, period_days in product(
        [150.0, 250.0, 350.0], [0.06], [7, 14, 30]
    ):
        configs.append(
            (
                "glicko2",
                {
                    "initial_rating": 1500.0,
                    "initial_rd": initial_rd,
                    "initial_vol": initial_vol,
                    "period_days": period_days,
                },
            )
        )

    for beta, tau in product([4.16, 6.25, 8.33, 12.50], [0.05, 0.25]):
        configs.append(
            (
                "trueskill",
                {"mu": 25.0, "sigma": 8.333, "beta": beta, "tau": tau, "draw_probability": 0.0},
            )
        )

    for sigma in [3.5, 5.0, 6.25, 8.333, 10.0]:
        configs.append(("openskill", {"mu": 25.0, "sigma": sigma}))

    for family in ["plackett_luce", "thurstone_mosteller"]:
        for beta, tau in product([8.33, 12.50, 18.75], [0.05, 0.166]):
            configs.append((family, {"mu": 25.0, "sigma": 8.333, "beta": beta, "tau": tau}))

    return configs


def update_after_match(system: RatingSystemProtocol, match: MatchRecord) -> None:
    """Update a rating system after all games from one match."""
    for score_1 in match.scores:
        score_2 = 1 - score_1
        system.update_team(match.team_1, match.team_2, score_1, score_2)
        system.update_player(list(match.players_1), list(match.players_2), score_1, score_2)


def evaluate_config(
    system_name: str,
    params: dict[str, Any],
    matches: Iterable[MatchRecord],
    common_ids: set[int],
) -> dict[str, Any]:
    """Evaluate one rating configuration on chronological common-market matches."""
    system = build_system(system_name, params)
    tune_true: list[int] = []
    tune_prob: list[float] = []
    test_true: list[int] = []
    test_prob: list[float] = []
    all_true: list[int] = []
    all_prob: list[float] = []

    for match in matches:
        if isinstance(system, TunableGlickoRating):
            system.update_before_match(
                match.team_1,
                match.team_2,
                list(match.players_1),
                list(match.players_2),
                match.match_date,
            )
        probability = float(system.predict_player_win_prob(list(match.players_1), list(match.players_2)))
        if match.match_id in common_ids and match.match_date >= date(2021, 1, 1):
            all_true.append(match.y_true)
            all_prob.append(probability)
            if match.match_date < date(2024, 1, 1):
                tune_true.append(match.y_true)
                tune_prob.append(probability)
            else:
                test_true.append(match.y_true)
                test_prob.append(probability)
        update_after_match(system, match)

    row: dict[str, Any] = {"system": system_name, **params}
    for prefix, y_true, y_prob in [
        ("tune_2021_2023", tune_true, tune_prob),
        ("test_2024_plus", test_true, test_prob),
        ("all_2021_plus", all_true, all_prob),
    ]:
        metrics = calculate_metrics(y_true, y_prob)
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return row


def make_config_label(row: pd.Series) -> str:
    """Create a compact label for a rating parameter configuration."""
    system = row["system"]
    if system == "elo":
        return f"Elo K={row['k_player']:.0f}"
    if system == "glicko2":
        return f"G2 RD={row['initial_rd']:.0f} vol={row['initial_vol']:.2f} P={row['period_days']:.0f}"
    if system == "trueskill":
        return f"TS β={row['beta']:.2f} τ={row['tau']:.2f}"
    if system == "openskill":
        return f"OS σ={row['sigma']:.2f}"
    if system == "plackett_luce":
        return f"PL β={row['beta']:.2f} τ={row['tau']:.2f}"
    if system == "thurstone_mosteller":
        return f"TM β={row['beta']:.2f} τ={row['tau']:.2f}"
    return str(system)


def save_top_plot(results: pd.DataFrame) -> None:
    """Show test scores for configurations selected by validation LogLoss."""
    plot_data = results.sort_values("tune_2021_2023_logloss", kind="stable").head(18).copy()
    plot_data["label"] = plot_data.apply(make_config_label, axis=1)
    min_value = plot_data["test_2024_plus_logloss"].min()
    max_value = plot_data["test_2024_plus_logloss"].max()
    margin = max(max_value - min_value, 0.003)

    plt.figure(figsize=(13, 6.5))
    ax = sns.barplot(
        data=plot_data,
        x="label",
        y="test_2024_plus_logloss",
        hue="system",
        dodge=False,
        palette="muted",
    )
    ax.set_title("Najlepsze konfiguracje ratingów — LogLoss 2024+")
    ax.set_xlabel("")
    ax.set_ylabel("LogLoss, 2024+")
    ax.set_ylim(min_value - margin * 0.4, max_value + margin * 0.5)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.legend(title="Rodzina", loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rating_family_optimization_top_logloss.png")
    plt.close()


def save_family_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Select each family using validation only; preserve each complete row."""
    best = (
        results.sort_values("tune_2021_2023_logloss", kind="stable")
        .drop_duplicates("system", keep="first")
        .sort_values("tune_2021_2023_logloss", kind="stable")
        .copy()
    )
    best["label"] = best.apply(make_config_label, axis=1)
    best.to_csv(OUTPUT_DIR / "rating_family_best_by_system.csv", index=False)
    return best


def main() -> None:
    """Run rating-family hyperparameter optimization."""
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    print("Loading matches and common market IDs...", flush=True)
    matches = load_matches()
    common_ids = load_common_market_ids()
    configs = build_parameter_grid()
    print(
        f"Matches: {len(matches):,}; common IDs: {len(common_ids):,}; configs: {len(configs)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for index, (system_name, params) in enumerate(configs, start=1):
        print(f"[{index:03d}/{len(configs):03d}] {system_name}: {params}", flush=True)
        rows.append(evaluate_config(system_name, params, matches, common_ids))
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "rating_family_optimization_results_partial.csv",
            index=False,
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "rating_family_optimization_results.csv", index=False)
    best = save_family_summary(results)
    save_top_plot(results)
    print("\nBest configurations by family:")
    print(
        best[
            [
                "system",
                "label",
                "test_2024_plus_auc",
                "test_2024_plus_logloss",
                "test_2024_plus_brier",
                "test_2024_plus_ece",
                "all_2021_plus_logloss",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
