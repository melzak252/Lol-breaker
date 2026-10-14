"""Worlds Crystal Ball Tournament Simulator.

Simulates the entire Worlds championship (Play-In -> Swiss Stage -> Knockout)
under Hard Fearless Draft rules with hierarchical in-game player statistics generation.
Computes calibrated probability distributions, percentiles, and fair odds for all official
Riot Worlds Crystal Ball Pick'Em categories.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import logging
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.champion_pool_entropy import (
    DEFAULT_GAMMA,
    ROLES,
    ensure_player_pools_loaded,
)
from src.models.fearless_simulator import (
    FearlessSeriesState,
    c0_predict_fearless_map,
    sample_map_draft,
)
from src.models.player_statistics_model import (
    GameStatsResult,
    PlayerStatsDistributionModel,
    load_player_statistics_model,
)

logger = logging.getLogger(__name__)

# Standard competitive ratings for Worlds 2026 participants
DEFAULT_WORLDS_RATINGS: dict[str, float] = {
    # Pool 1
    "Gen.G": 1740.0,
    "Bilibili Gaming": 1720.0,
    "LYON": 1510.0,
    "G2 Esports": 1580.0,
    # Pool 2
    "Team Secret Whales": 1480.0,
    "Los Grandes": 1420.0,
    "Hanwha Life Esports": 1680.0,
    "Anyone's Legend": 1620.0,
    # Pool 3
    "Team Liquid": 1520.0,
    "Karmine Corp": 1500.0,
    "T1": 1690.0,
    "Top Esports": 1660.0,
    # Pool 4
    "CTBC Flying Oyster": 1460.0,
    "Dplus KIA": 1610.0,
    "LGD Gaming": 1560.0,
    # Play-In
    "Cloud9": 1490.0,
    "GIANTX": 1470.0,
    "MVK Esports": 1400.0,
    "LOUD": 1390.0,
}

# Standard starting rosters (TOP, JUNGLE, MID, ADC, SUPPORT)
DEFAULT_WORLDS_ROSTERS: dict[str, list[str]] = {
    "Gen.G": ["Kiin", "Canyon", "Chovy", "Ruler", "Duro"],
    "Bilibili Gaming": ["Bin", "XUN", "Knight", "Viper", "ON"],
    "T1": ["Doran", "Oner", "Faker", "Peyz", "Keria"],
    "Hanwha Life Esports": ["Zeus", "Kanavi", "Zeka", "Gumayusi", "Delight"],
    "Top Esports": ["ZUIAN", "Tian", "Creme", "JackeyLove", "Zhuo"],
    "Anyone's Legend": ["Breathe", "Tarzan", "Shanks", "Hope", "Kael"],
    "Dplus KIA": ["Siwoo", "Lucid", "ShowMaker", "Smash", "Career"],
    "G2 Esports": ["BrokenBlade", "SkewMond", "Caps", "Hans Sama", "Labrov"],
    "Team Liquid": ["Morgan", "Josedeodo", "Quid", "Yeon", "CoreJJ"],
    "LYON": ["Dhokla", "Inspired", "Saint", "Berserker", "Isles"],
    "Karmine Corp": ["Canna", "Yike", "Kyeahoo", "Caliste", "Busio"],
    "Invictus Gaming": ["TheShy", "Wei", "RooKie", "JiaQi", "meiko"],
    "Cloud9": ["Thanatos", "Blaber", "Loki", "Tactical", "Vulcan"],
    "Movistar KOI": ["Myrwn", "Elyoya", "jojopyun", "Supa", "Alvaro"],
    "Team Secret Whales": ["Pun", "Hizto", "Dire", "Eddie", "Bie"],
    "CTBC Flying Oyster": ["Rest", "Shad0w", "Pout", "Doggo", "Kino"],
    "Los Grandes": ["Zest", "Curse", "Feisty", "Duduhh", "Ackerman"],
    "MVK Esports": ["Hasmed", "Yushin", "Kuro", "Slayder", "Palette"],
    "FURIA": ["Guigo", "Tatu", "Tutsz", "Ayu", "Jojo"],
}

# 15 Direct Swiss + 4 Play-In participants (Official Liquipedia Pools)
DEFAULT_DIRECT_TEAMS: list[dict[str, Any]] = [
    {"team": "Gen.G", "region": "LCK", "pool": 1},
    {"team": "Anyone's Legend", "region": "LPL", "pool": 1},
    {"team": "Team Liquid", "region": "LCS", "pool": 1},
    {"team": "G2 Esports", "region": "LEC", "pool": 1},
    {"team": "Team Secret Whales", "region": "LCP", "pool": 2},
    {"team": "Los Grandes", "region": "CBLOL", "pool": 2},
    {"team": "Hanwha Life Esports", "region": "LCK", "pool": 2},
    {"team": "Bilibili Gaming", "region": "LPL", "pool": 2},
    {"team": "LYON", "region": "LCS", "pool": 3},
    {"team": "Movistar KOI", "region": "LEC", "pool": 3},
    {"team": "T1", "region": "LCK", "pool": 3},
    {"team": "Top Esports", "region": "LPL", "pool": 3},
    {"team": "CTBC Flying Oyster", "region": "LCP", "pool": 4},
    {"team": "Dplus KIA", "region": "LCK", "pool": 4},
    {"team": "Invictus Gaming", "region": "LPL", "pool": 4},
]

DEFAULT_PLAY_IN_TEAMS: list[dict[str, Any]] = [
    {"team": "Cloud9", "region": "LCS"},
    {"team": "Karmine Corp", "region": "LEC"},
    {"team": "FURIA", "region": "CBLOL"},
    {"team": "MVK Esports", "region": "LCP"},
]
# Champion ID to standard display name mapping
CHAMPION_ID_TO_NAME: dict[int, str] = {
    1: "Annie", 2: "Olaf", 3: "Galio", 4: "Twisted Fate", 7: "LeBlanc", 12: "Alistar",
    21: "Miss Fortune", 22: "Ashe", 24: "Jax", 25: "Morgana", 38: "Karma", 40: "Janna",
    41: "Gangplank", 51: "Caitlyn", 53: "Blitzcrank", 58: "Renekton", 59: "Jarvan IV",
    60: "Elise", 61: "Orianna", 64: "Lee Sin", 76: "Nidalee", 81: "Ezreal", 84: "Akali",
    85: "Kennen", 89: "Leona", 98: "Shen", 103: "Ahri", 104: "Graves", 111: "Nautilus",
    113: "Sejuani", 114: "Fiora", 119: "Draven", 120: "Hecarim", 121: "Kha'Zix",
    126: "Jayce", 127: "Kindred", 134: "Syndra", 145: "Kai'Sa", 147: "Aphelios",
    150: "Gnar", 201: "Braum", 202: "Jhin", 222: "Jinx", 236: "Lucian", 238: "Zed",
    245: "Ekko", 254: "Vi", 266: "Aatrox", 412: "Thresh", 421: "Rek'Sai", 432: "Bard",
    497: "Rakan", 498: "Xayah", 516: "Ornn", 875: "Sett",
}


def _ensure_champion_names_loaded() -> None:
    if len(CHAMPION_ID_TO_NAME) > 150:
        return
    import os
    if os.getenv("PYTEST_CURRENT_TEST") or "betting_test" in os.getenv("DATABASE_URL", ""):
        return
    try:
        from betting_app.core.db import get_session
        from sqlalchemy import text
        with get_session() as session:
            rows = session.execute(text("SELECT DISTINCT champion_id::int as cid, champion_name FROM golgg_game_players WHERE champion_id IS NOT NULL AND champion_name IS NOT NULL")).fetchall()
            for r in rows:
                if r[0] and r[1]:
                    CHAMPION_ID_TO_NAME[int(r[0])] = str(r[1])
    except Exception:
        pass


def get_champion_name(champion_id: int) -> str:
    """Return champion name from ID with fallback."""
    _ensure_champion_names_loaded()
    return CHAMPION_ID_TO_NAME.get(champion_id, f"Champion {champion_id}")


@dataclass(frozen=True)
class CrystalBallOutcome:
    """Category ranking distribution with probability mass summing to 1.0."""

    category_name: str
    ranking: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_name": self.category_name,
            "ranking": self.ranking,
        }


@dataclass(frozen=True)
class WorldsCrystalBallResults:
    """Complete simulated Worlds Crystal Ball Pick'Em predictions."""

    n_simulations: int
    player_most_kills_single_game: CrystalBallOutcome
    player_highest_tournament_kda: CrystalBallOutcome
    player_most_unique_champions: CrystalBallOutcome
    player_highest_kill_participation: CrystalBallOutcome
    team_most_unique_champions: CrystalBallOutcome
    team_most_kills_single_game: CrystalBallOutcome
    team_highest_kda: CrystalBallOutcome
    tournament_shortest_game_seconds: dict[str, float]
    tournament_longest_game_seconds: dict[str, float]
    tournament_total_pentakills: dict[int, float]
    tournament_total_barons: dict[str, float]
    champion_most_picked: CrystalBallOutcome
    champion_most_banned: CrystalBallOutcome
    champion_highest_winrate: CrystalBallOutcome
    champion_most_deaths: CrystalBallOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "player_most_kills_single_game": self.player_most_kills_single_game.to_dict(),
            "player_highest_tournament_kda": self.player_highest_tournament_kda.to_dict(),
            "player_most_unique_champions": self.player_most_unique_champions.to_dict(),
            "player_highest_kill_participation": self.player_highest_kill_participation.to_dict(),
            "team_most_unique_champions": self.team_most_unique_champions.to_dict(),
            "team_most_kills_single_game": self.team_most_kills_single_game.to_dict(),
            "team_highest_kda": self.team_highest_kda.to_dict(),
            "tournament_shortest_game_seconds": self.tournament_shortest_game_seconds,
            "tournament_longest_game_seconds": self.tournament_longest_game_seconds,
            "tournament_total_pentakills": {str(k): v for k, v in self.tournament_total_pentakills.items()},
            "tournament_total_barons": self.tournament_total_barons,
            "champion_most_picked": self.champion_most_picked.to_dict(),
            "champion_most_banned": self.champion_most_banned.to_dict(),
            "champion_highest_winrate": self.champion_highest_winrate.to_dict(),
            "champion_most_deaths": self.champion_most_deaths.to_dict(),
        }


class _PathTracker:
    """Accumulates in-game player, team, and champion events for one tournament path."""

    __slots__ = (
        "player_kills", "player_deaths", "player_assists", "player_games",
        "player_unique_champs", "player_max_kills", "player_pentakills",
        "team_kills", "team_deaths", "team_assists", "team_games",
        "team_unique_champs", "team_max_kills",
        "champion_picks", "champion_bans", "champion_wins", "champion_deaths",
        "shortest_game", "longest_game", "total_pentakills", "total_barons",
    )

    def __init__(self) -> None:
        self.player_kills: Counter[str] = Counter()
        self.player_deaths: Counter[str] = Counter()
        self.player_assists: Counter[str] = Counter()
        self.player_games: Counter[str] = Counter()
        self.player_unique_champs: defaultdict[str, set[int]] = defaultdict(set)
        self.player_max_kills: Counter[str] = Counter()
        self.player_pentakills: Counter[str] = Counter()

        self.team_kills: Counter[str] = Counter()
        self.team_deaths: Counter[str] = Counter()
        self.team_assists: Counter[str] = Counter()
        self.team_games: Counter[str] = Counter()
        self.team_unique_champs: defaultdict[str, set[int]] = defaultdict(set)
        self.team_max_kills: Counter[str] = Counter()

        self.champion_picks: Counter[str] = Counter()
        self.champion_bans: Counter[str] = Counter()
        self.champion_wins: Counter[str] = Counter()
        self.champion_deaths: Counter[str] = Counter()

        self.shortest_game: int = 99999
        self.longest_game: int = 0
        self.total_pentakills: int = 0
        self.total_barons: int = 0

    def record_game(
        self,
        team_a: str,
        team_b: str,
        winner: str,
        game_res: GameStatsResult,
        burned_champs_this_map: Sequence[int],
    ) -> None:
        """Record game statistics into the tournament path tracker."""
        dur = game_res.duration_seconds
        if dur < self.shortest_game:
            self.shortest_game = dur
        if dur > self.longest_game:
            self.longest_game = dur

        if game_res.has_pentakill:
            self.total_pentakills += 1
            if game_res.pentakill_player:
                self.player_pentakills[game_res.pentakill_player] += 1
        tot_barons = game_res.team_a_barons + game_res.team_b_barons
        self.total_barons += tot_barons

        # Team stats
        self.team_kills[team_a] += game_res.team_a_kills
        self.team_deaths[team_a] += game_res.team_a_deaths
        self.team_games[team_a] += 1
        if game_res.team_a_kills > self.team_max_kills[team_a]:
            self.team_max_kills[team_a] = game_res.team_a_kills

        self.team_kills[team_b] += game_res.team_b_kills
        self.team_deaths[team_b] += game_res.team_b_deaths
        self.team_games[team_b] += 1
        if game_res.team_b_kills > self.team_max_kills[team_b]:
            self.team_max_kills[team_b] = game_res.team_b_kills

        # Fearless bans / lockouts
        for c in burned_champs_this_map:
            c_name = get_champion_name(c)
            self.champion_bans[c_name] += 1

        # Player stats
        for key, s in game_res.player_stats.items():
            p_name = s.player_id
            c_name = get_champion_name(s.champion_id)
            is_team_a = key.startswith("A_")
            t_name = team_a if is_team_a else team_b
            is_win = (winner == "A") if is_team_a else (winner == "B")

            self.player_kills[p_name] += s.kills
            self.player_deaths[p_name] += s.deaths
            self.player_assists[p_name] += s.assists
            self.player_games[p_name] += 1
            self.player_unique_champs[p_name].add(s.champion_id)
            if s.kills > self.player_max_kills[p_name]:
                self.player_max_kills[p_name] = s.kills

            # Team assists and unique champions
            self.team_assists[t_name] += s.assists
            self.team_unique_champs[t_name].add(s.champion_id)

            # Champion picks, wins, deaths
            self.champion_picks[c_name] += 1
            if is_win:
                self.champion_wins[c_name] += 1
            self.champion_deaths[c_name] += s.deaths


def _simulate_fearless_series(
    team_a: str,
    team_b: str,
    best_of: int,
    rosters: Mapping[str, list[str]],
    ratings: Mapping[str, float],
    stats_model: PlayerStatsDistributionModel,
    tracker: _PathTracker,
    rng: np.random.Generator,
) -> str:
    """Simulate one Bo1/Bo3/Bo5 series under Hard Fearless rules, recording map stats."""
    target_wins = (best_of + 1) // 2
    score_a = 0
    score_b = 0
    burned_champions: set[int] = set()
    current_side_a = "blue"

    roster_a = rosters.get(team_a) or [f"{team_a}_{r}" for r in ROLES]
    roster_b = rosters.get(team_b) or [f"{team_b}_{r}" for r in ROLES]

    r_a = ratings.get(team_a, 1500.0)
    r_b = ratings.get(team_b, 1500.0)
    base_prob_a = float(1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0)))

    while score_a < target_wins and score_b < target_wins:
        # 1. Map win probability with Fearless champion pool depletion and side advantage
        p_map_a = c0_predict_fearless_map(
            features_a=roster_a,
            features_b=roster_b,
            burned_champions=burned_champions,
            side_a=current_side_a,
            base_map_prob=base_prob_a,
            gamma=DEFAULT_GAMMA,
        )

        map_winner = "A" if (rng.random() < p_map_a) else "B"

        # 2. Sample drafted champions under Fearless constraints
        drafted_10 = sample_map_draft(
            team_a_roster=roster_a,
            team_b_roster=roster_b,
            burned_champions=burned_champions,
            rng=rng,
        )

        # 3. Sample complete in-game stats for all 10 players
        game_res = stats_model.sample_game_stats(
            team_a_roster=roster_a,
            team_b_roster=roster_b,
            winner=map_winner,
            drafted_champions=drafted_10,
            rng=rng,
        )

        # 4. Record to path accumulator
        tracker.record_game(
            team_a=team_a,
            team_b=team_b,
            winner=map_winner,
            game_res=game_res,
            burned_champs_this_map=drafted_10,
        )

        # Burn drafted champions for subsequent games in this series
        burned_champions.update(drafted_10)

        # 5. Advance series state (loser gets blue side next map)
        if map_winner == "A":
            score_a += 1
            current_side_a = "red"
        else:
            score_b += 1
            current_side_a = "blue"

    return team_a if score_a == target_wins else team_b


def _sample_swiss_round_pairs(
    teams: list[str],
    played_pairs: set[frozenset[str]],
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """Sample legal Swiss round pairings with no rematches."""
    n = len(teams)
    if n % 2 != 0:
        raise ValueError("Swiss buckets must contain an even number of teams.")
    if n == 0:
        return []

    shuffled = list(teams)
    rng.shuffle(shuffled)

    # Fast backtracking to find a legal matching
    def find_matching(remaining: list[str]) -> list[tuple[str, str]] | None:
        if not remaining:
            return []
        first = remaining[0]
        candidates = [t for t in remaining[1:] if frozenset((first, t)) not in played_pairs]
        rng.shuffle(candidates)
        for cand in candidates:
            sub = [t for t in remaining[1:] if t != cand]
            res = find_matching(sub)
            if res is not None:
                return [(first, cand)] + res
        return None

    pairs = find_matching(shuffled)
    if pairs is None:
        # Fallback if no perfect non-rematch matching exists
        pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, n, 2)]

    return pairs


def _simulate_single_tournament_path(
    direct_teams: list[dict[str, Any]],
    play_in_teams: list[dict[str, Any]],
    play_in_winner_pool: int,
    rosters: Mapping[str, list[str]],
    ratings: Mapping[str, float],
    stats_model: PlayerStatsDistributionModel,
    rng: np.random.Generator,
) -> _PathTracker:
    """Simulate one complete Worlds tournament path and return accumulated records."""
    tracker = _PathTracker()

    def run_series(t1: str, t2: str, bo: int) -> str:
        return _simulate_fearless_series(
            team_a=t1,
            team_b=t2,
            best_of=bo,
            rosters=rosters,
            ratings=ratings,
            stats_model=stats_model,
            tracker=tracker,
            rng=rng,
        )

    # 1. Play-In Tournament (4 teams, Bo5 double elimination)
    pi_bracket = [t["team"] for t in play_in_teams]
    rng.shuffle(pi_bracket)

    upper_w1 = run_series(pi_bracket[0], pi_bracket[1], 5)
    upper_l1 = pi_bracket[1] if upper_w1 == pi_bracket[0] else pi_bracket[0]

    upper_w2 = run_series(pi_bracket[2], pi_bracket[3], 5)
    upper_l2 = pi_bracket[3] if upper_w2 == pi_bracket[2] else pi_bracket[2]

    lower_rd_w = run_series(upper_l1, upper_l2, 5)
    upper_fin_w = run_series(upper_w1, upper_w2, 5)
    upper_fin_l = upper_w2 if upper_fin_w == upper_w1 else upper_w1

    lower_fin_w = run_series(lower_rd_w, upper_fin_l, 5)
    play_in_qualifier = run_series(upper_fin_w, lower_fin_w, 5)

    # 2. Swiss Stage (16 participants: 15 direct + 1 Play-In qualifier)
    swiss_participants: list[dict[str, Any]] = [
        *direct_teams,
        {"team": play_in_qualifier, "region": "Play-In", "pool": play_in_winner_pool},
    ]

    records: dict[str, list[int]] = {t["team"]: [0, 0] for t in swiss_participants}
    played_pairs: set[frozenset[str]] = set()
    advanced_teams: list[str] = []

    # Round 1: Pool 1 vs Pool 4, Pool 2 vs Pool 3 (all Bo1)
    pools: dict[int, list[str]] = defaultdict(list)
    for t in swiss_participants:
        pools[int(t.get("pool", 4))].append(t["team"])

    next_buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for high_p, low_p in ((1, 4), (2, 3)):
        h_teams = list(pools[high_p])
        l_teams = list(pools[low_p])
        rng.shuffle(h_teams)
        rng.shuffle(l_teams)
        for t1, t2 in zip(h_teams, l_teams):
            winner = run_series(t1, t2, 1)
            loser = t2 if winner == t1 else t1
            played_pairs.add(frozenset((t1, t2)))
            records[winner][0] += 1
            records[loser][1] += 1
            next_buckets[(1, 0)].append(winner)
            next_buckets[(0, 1)].append(loser)

    buckets = next_buckets

    # Rounds 2 to 5
    for _round_num in range(2, 6):
        next_buckets = defaultdict(list)
        for (w, l), b_teams in list(buckets.items()):
            best_of = 3 if (w == 2 or l == 2) else 1
            pairs = _sample_swiss_round_pairs(b_teams, played_pairs, rng)
            for t1, t2 in pairs:
                winner = run_series(t1, t2, best_of)
                loser = t2 if winner == t1 else t1
                played_pairs.add(frozenset((t1, t2)))
                records[winner][0] += 1
                records[loser][1] += 1

                new_w, new_l = records[winner]
                if new_w == 3:
                    advanced_teams.append(winner)
                else:
                    next_buckets[(new_w, new_l)].append(winner)

                new_lw, new_ll = records[loser]
                if new_ll < 3:
                    next_buckets[(new_lw, new_ll)].append(loser)

        buckets = next_buckets
        if len(advanced_teams) == 8:
            break

    # 3. Knockout Stage (8 qualifiers, single-elimination Bo5)
    rng.shuffle(advanced_teams)
    qf_winners = [
        run_series(advanced_teams[i], advanced_teams[i + 1], 5)
        for i in range(0, 8, 2)
    ]
    sf_winners = [
        run_series(qf_winners[i], qf_winners[i + 1], 5)
        for i in range(0, 4, 2)
    ]
    run_series(sf_winners[0], sf_winners[1], 5)  # Finals

    return tracker


def _compute_ranking(
    counts: Mapping[str, float],
    n_simulations: int,
    category_name: str,
    top_k: int | None = None,
) -> CrystalBallOutcome:
    """Normalize win counts into exact PMF ranking with fair odds."""
    ranking = []
    # Sort descending by win count
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for cand, wins in sorted_items:
        if wins <= 0:
            continue
        prob = float(wins / n_simulations)
        fair_odds = round(1.0 / prob, 2) if prob > 0 else 999.0
        ranking.append({
            "candidate": str(cand),
            "wins": round(float(wins), 2),
            "probability": round(prob, 5),
            "fair_odds": fair_odds,
        })

    # Guard: if no candidate won, provide default 50/50 or empty
    if not ranking:
        ranking = [{"candidate": "Unknown", "wins": n_simulations, "probability": 1.0, "fair_odds": 1.0}]

    # Exact probability mass conservation adjustment
    total_p = sum(r["probability"] for r in ranking)
    if total_p > 0 and not math.isclose(total_p, 1.0, abs_tol=1e-5):
        for r in ranking:
            r["probability"] = round(r["probability"] / total_p, 5)

    final_ranking = ranking if top_k is None else ranking[:top_k]
    # Ensure probability mass conservation on final ranking
    final_p = sum(r["probability"] for r in final_ranking)
    if final_p > 0 and not math.isclose(final_p, 1.0, abs_tol=1e-5):
        for r in final_ranking:
            r["probability"] = round(r["probability"] / final_p, 5)

    return CrystalBallOutcome(
        category_name=category_name,
        ranking=final_ranking,
    )


def simulate_worlds_crystal_ball(
    direct_teams: Sequence[Any] | None = None,
    play_in_teams: Sequence[Any] | None = None,
    n_simulations: int = 1000,
    seed: int = 42,
    stats_model: PlayerStatsDistributionModel | None = None,
) -> WorldsCrystalBallResults:
    """Execute complete Worlds tournament simulations under Fearless rules.

    Parameters
    ----------
    direct_teams : Sequence[Any], optional
        15 direct Swiss participants. If None, uses default 2026 participants.
    play_in_teams : Sequence[Any], optional
        4 Play-In participants. If None, uses default 2026 participants.
    n_simulations : int
        Number of complete tournament Monte Carlo rollouts (default: 1000).
    seed : int
        RNG seed for exact reproducibility.
    stats_model : PlayerStatsDistributionModel, optional
        Fitted player stats distribution model. If None, loads from serialized artifact or defaults.

    Returns
    -------
    WorldsCrystalBallResults
        Calibrated outcome distributions and fair odds for all official Crystal Ball categories.
    """
    n_sims = max(10, int(n_simulations))
    rng = np.random.default_rng(seed)
    ensure_player_pools_loaded()

    if stats_model is None:
        stats_model = load_player_statistics_model()

    # Normalize teams
    def parse_team_input(teams_in: Sequence[Any] | None, default_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not teams_in:
            return [dict(t) for t in default_list]
        parsed = []
        for t in teams_in:
            if isinstance(t, Mapping):
                parsed.append({"team": str(t.get("team", t.get("name", ""))), "region": str(t.get("region", "")), "pool": t.get("pool")})
            elif hasattr(t, "name"):
                parsed.append({"team": str(t.name), "region": str(getattr(t, "region", "")), "pool": getattr(t, "pool", None)})
            else:
                parsed.append({"team": str(t), "region": "Unknown", "pool": None})
        return parsed

    parsed_direct = parse_team_input(direct_teams, DEFAULT_DIRECT_TEAMS)
    parsed_play_in = parse_team_input(play_in_teams, DEFAULT_PLAY_IN_TEAMS)

    # Accumulators across all N simulations
    cat_player_max_kills: Counter[str] = Counter()
    cat_player_kda: Counter[str] = Counter()
    cat_player_unique_champs: Counter[str] = Counter()
    cat_player_kp: Counter[str] = Counter()

    cat_team_unique_champs: Counter[str] = Counter()
    cat_team_max_kills: Counter[str] = Counter()
    cat_team_kda: Counter[str] = Counter()

    cat_champ_picked: Counter[str] = Counter()
    cat_champ_banned: Counter[str] = Counter()
    cat_champ_winrate: Counter[str] = Counter()
    cat_champ_deaths: Counter[str] = Counter()

    shortest_durations: list[int] = []
    longest_durations: list[int] = []
    pentakill_counts: list[int] = []
    baron_counts: list[int] = []

    # Run N tournament paths
    for _ in range(n_sims):
        tracker = _simulate_single_tournament_path(
            direct_teams=parsed_direct,
            play_in_teams=parsed_play_in,
            play_in_winner_pool=4,
            rosters=DEFAULT_WORLDS_ROSTERS,
            ratings=DEFAULT_WORLDS_RATINGS,
            stats_model=stats_model,
            rng=rng,
        )

        shortest_durations.append(tracker.shortest_game)
        longest_durations.append(tracker.longest_game)
        pentakill_counts.append(tracker.total_pentakills)
        baron_counts.append(tracker.total_barons)

        # 1. Player Most Kills in a Single Game
        if tracker.player_max_kills:
            top_kill_val = max(tracker.player_max_kills.values())
            tied_killers = [p for p, k in tracker.player_max_kills.items() if k == top_kill_val]
            split_win = 1.0 / len(tied_killers)
            for p in tied_killers:
                cat_player_max_kills[p] += split_win

        # 2. Player Highest Tournament KDA (min 5 games)
        qual_players_kda = {
            p: (tracker.player_kills[p] + tracker.player_assists[p]) / max(1, tracker.player_deaths[p])
            for p, g in tracker.player_games.items() if g >= 5
        }
        if qual_players_kda:
            max_kda_val = max(qual_players_kda.values())
            tied_kda = [p for p, val in qual_players_kda.items() if math.isclose(val, max_kda_val, rel_tol=1e-4)]
            split_win = 1.0 / len(tied_kda)
            for p in tied_kda:
                cat_player_kda[p] += split_win

        # 3. Player Most Unique Champions
        player_champ_counts = {p: len(champs) for p, champs in tracker.player_unique_champs.items()}
        if player_champ_counts:
            max_unique = max(player_champ_counts.values())
            tied_champs = [p for p, c in player_champ_counts.items() if c == max_unique]
            split_win = 1.0 / len(tied_champs)
            for p in tied_champs:
                cat_player_unique_champs[p] += split_win

        # 4. Player Highest Kill Participation (min 5 games)
        player_kp_rates: dict[str, float] = {}
        # Find player's team to get team kills
        for p, g in tracker.player_games.items():
            if g < 5:
                continue
            # Associate player with team
            p_team = None
            for t_name, r_list in DEFAULT_WORLDS_ROSTERS.items():
                if p in r_list:
                    p_team = t_name
                    break
            t_kills = tracker.team_kills.get(p_team, 1) if p_team else 1
            kp = (tracker.player_kills[p] + tracker.player_assists[p]) / max(1, t_kills)
            player_kp_rates[p] = kp

        if player_kp_rates:
            max_kp = max(player_kp_rates.values())
            tied_kp = [p for p, val in player_kp_rates.items() if math.isclose(val, max_kp, rel_tol=1e-4)]
            split_win = 1.0 / len(tied_kp)
            for p in tied_kp:
                cat_player_kp[p] += split_win

        # 5. Team Most Unique Champions
        team_champ_counts = {t: len(champs) for t, champs in tracker.team_unique_champs.items()}
        if team_champ_counts:
            max_t_unique = max(team_champ_counts.values())
            tied_t_unique = [t for t, c in team_champ_counts.items() if c == max_t_unique]
            split_win = 1.0 / len(tied_t_unique)
            for t in tied_t_unique:
                cat_team_unique_champs[t] += split_win

        # 6. Team Most Kills in a Single Game
        if tracker.team_max_kills:
            max_t_kills = max(tracker.team_max_kills.values())
            tied_t_kills = [t for t, k in tracker.team_max_kills.items() if k == max_t_kills]
            split_win = 1.0 / len(tied_t_kills)
            for t in tied_t_kills:
                cat_team_max_kills[t] += split_win

        # 7. Team Highest KDA (min 5 games)
        qual_team_kda = {
            t: (tracker.team_kills[t] + tracker.team_assists[t]) / max(1, tracker.team_deaths[t])
            for t, g in tracker.team_games.items() if g >= 5
        }
        if qual_team_kda:
            max_tkda = max(qual_team_kda.values())
            tied_tkda = [t for t, val in qual_team_kda.items() if math.isclose(val, max_tkda, rel_tol=1e-4)]
            split_win = 1.0 / len(tied_tkda)
            for t in tied_tkda:
                cat_team_kda[t] += split_win

        # 8. Champion Most Picked
        if tracker.champion_picks:
            max_picks = max(tracker.champion_picks.values())
            tied_picks = [c for c, p in tracker.champion_picks.items() if p == max_picks]
            split_win = 1.0 / len(tied_picks)
            for c in tied_picks:
                cat_champ_picked[c] += split_win

        # 9. Champion Most Banned
        if tracker.champion_bans:
            max_bans = max(tracker.champion_bans.values())
            tied_bans = [c for c, b in tracker.champion_bans.items() if b == max_bans]
            split_win = 1.0 / len(tied_bans)
            for c in tied_bans:
                cat_champ_banned[c] += split_win

        # 10. Champion Highest Winrate (min 5 picks, or all picks if none reached 5)
        qual_champ_wr = {
            c: tracker.champion_wins[c] / p
            for c, p in tracker.champion_picks.items() if p >= 5
        }
        if not qual_champ_wr and tracker.champion_picks:
            qual_champ_wr = {c: tracker.champion_wins[c] / p for c, p in tracker.champion_picks.items()}
        if qual_champ_wr:
            max_wr = max(qual_champ_wr.values())
            tied_wr = [c for c, val in qual_champ_wr.items() if math.isclose(val, max_wr, rel_tol=1e-4)]
            split_win = 1.0 / len(tied_wr)
            for c in tied_wr:
                cat_champ_winrate[c] += split_win

        # 11. Champion Most Deaths
        if tracker.champion_deaths:
            max_deaths = max(tracker.champion_deaths.values())
            tied_deaths = [c for c, d in tracker.champion_deaths.items() if d == max_deaths]
            split_win = 1.0 / len(tied_deaths)
            for c in tied_deaths:
                cat_champ_deaths[c] += split_win

    # Aggregate percentiles and statistics
    def calc_percentiles(vals: list[int | float]) -> dict[str, float]:
        arr = np.array(vals, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p25": round(float(np.percentile(arr, 25)), 1),
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "min": round(float(np.min(arr)), 1),
            "max": round(float(np.max(arr)), 1),
        }

    # Pentakill PMF
    penta_counter = Counter(pentakill_counts)
    penta_pmf = {
        k: round(penta_counter.get(k, 0) / n_sims, 4)
        for k in range(max(4, max(pentakill_counts or [0]) + 1))
    }
    # Ensure exact PMF sum
    p_sum = sum(penta_pmf.values())
    if p_sum > 0:
        penta_pmf = {k: round(v / p_sum, 4) for k, v in penta_pmf.items()}

    return WorldsCrystalBallResults(
        n_simulations=n_sims,
        player_most_kills_single_game=_compute_ranking(cat_player_max_kills, n_sims, "Najwięcej zabójstw w 1 grze (Gracz)"),
        player_highest_tournament_kda=_compute_ranking(cat_player_kda, n_sims, "Najwyższe KDA turnieju (Gracz)"),
        player_most_unique_champions=_compute_ranking(cat_player_unique_champs, n_sims, "Najwięcej unikalnych bohaterów (Gracz)"),
        player_highest_kill_participation=_compute_ranking(cat_player_kp, n_sims, "Najwyższy udział w zabójstwach KP% (Gracz)"),
        team_most_unique_champions=_compute_ranking(cat_team_unique_champs, n_sims, "Najwięcej unikalnych bohaterów (Drużyna)"),
        team_most_kills_single_game=_compute_ranking(cat_team_max_kills, n_sims, "Najwięcej zabójstw w 1 grze (Drużyna)"),
        team_highest_kda=_compute_ranking(cat_team_kda, n_sims, "Najwyższe KDA turnieju (Drużyna)"),
        tournament_shortest_game_seconds=calc_percentiles(shortest_durations),
        tournament_longest_game_seconds=calc_percentiles(longest_durations),
        tournament_total_pentakills=penta_pmf,
        tournament_total_barons=calc_percentiles(baron_counts),
        champion_most_picked=_compute_ranking(cat_champ_picked, n_sims, "Najczęściej wybierany bohater"),
        champion_most_banned=_compute_ranking(cat_champ_banned, n_sims, "Najczęściej banowany bohater (Fearless)"),
        champion_highest_winrate=_compute_ranking(cat_champ_winrate, n_sims, "Najwyższy winrate bohatera"),
        champion_most_deaths=_compute_ranking(cat_champ_deaths, n_sims, "Najwięcej zgonów (Bohater)"),
    )
