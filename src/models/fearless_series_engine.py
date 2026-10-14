"""Sequential map-by-map Hard Fearless Draft series simulation engine.

Implements sequential Markov map-by-map simulation under official Hard Fearless Draft
rules (MSI 2025+, Worlds 2025+, LCK Cup, etc.):
1. Side choice dynamics:
   - Game 1: Higher seed (or team_a) gets Blue side choice.
   - Game 2+: Loser of previous map chooses side (default Blue).
   - Blue side advantage: Delta_side = +-0.1242 in logit space (+25.0 rating points).
2. Hard Fearless Draft champion pool depletion:
   - Exactly 10 unique champions burned per map (|B| = 10 * (map - 1)).
   - Comfort pool depletion penalty:
     Delta R_team = -gamma * sum_{p=1}^5 (sum_{c in C_p^* cap B} P(c)) * exp(-D_p / 10.0)
     using player pool metrics from `src.features.champion_pool_entropy`.
3. Scoreline progression and terminal series resolution:
   - Series concludes when a team achieves floor(best_of / 2) + 1 map wins.
   - Returns full map history, exact scoreline string (e.g., "3-0", "3-1", "3-2"),
     burned champions list, and final team depletion values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

from src.features.champion_pool_entropy import (
    DEFAULT_COMFORT_THRESHOLD,
    DEFAULT_GAMMA,
    DEFAULT_ROLE_META,
    ROLES,
    compute_player_pool_metrics,
)
from src.ratings.glicko2_core import series_win_probability

# Blue Side offset in logit space and rating points
BLUE_SIDE_LOGIT_DELTA: float = 0.1242
BLUE_SIDE_RATING_OFFSET: float = 25.0
RATING_TO_LOGIT_SCALE: float = BLUE_SIDE_LOGIT_DELTA / BLUE_SIDE_RATING_OFFSET  # ~0.004968


@dataclass
class MapResult:
    """Individual map outcome in a Fearless series.

    Attributes
    ----------
    map_number : int
        1-indexed map counter (1, 2, 3, ...).
    winner : str
        Canonical name of the team that won this map.
    loser : str
        Canonical name of the team that lost this map.
    blue_team : str
        Canonical name of the team playing on Blue side.
    red_team : str
        Canonical name of the team playing on Red side.
    score_a : int
        Cumulative wins by Team A after this map completes.
    score_b : int
        Cumulative wins by Team B after this map completes.
    burned_count : int
        Number of burned champions in effect for this map (|B| = 10 * (map_number - 1)).
    """
    map_number: int
    winner: str
    loser: str
    blue_team: str
    red_team: str
    score_a: int
    score_b: int
    burned_count: int

    @property
    def cumulative_burned_count(self) -> int:
        """Total cumulative burned champions in the series following this map's draft."""
        return self.burned_count + 10

    def to_dict(self) -> dict[str, Any]:
        """Serialize map result to dictionary."""
        return {
            "map_number": self.map_number,
            "winner": self.winner,
            "loser": self.loser,
            "blue_team": self.blue_team,
            "red_team": self.red_team,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "burned_count": self.burned_count,
        }


@dataclass
class FearlessSeriesResult:
    """Complete outcome and progression history of a simulated Fearless Draft series.

    Attributes
    ----------
    winner : str
        Series winner.
    loser : str
        Series loser.
    score_winner : int
        Wins accumulated by the winning team.
    score_loser : int
        Wins accumulated by the losing team.
    maps_played : int
        Total number of maps played in the series.
    scoreline : str
        Exact series scoreline in '{score_a}-{score_b}' format (e.g. '3-1', '0-3').
    map_history : list[MapResult]
        Sequential map outcomes in chronological order.
    burned_champions : list[int]
        Cumulative champion IDs drafted and burned across all maps in the series.
    final_depletion_a : float
        Cumulative comfort pool depletion penalty Delta R_team for Team A (rating points <= 0).
    final_depletion_b : float
        Cumulative comfort pool depletion penalty Delta R_team for Team B (rating points <= 0).
    """
    winner: str
    loser: str
    score_winner: int
    score_loser: int
    maps_played: int
    scoreline: str
    map_history: list[MapResult]
    burned_champions: list[int]
    final_depletion_a: float
    final_depletion_b: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize series result to dictionary."""
        return {
            "winner": self.winner,
            "loser": self.loser,
            "score_winner": self.score_winner,
            "score_loser": self.score_loser,
            "maps_played": self.maps_played,
            "scoreline": self.scoreline,
            "map_history": [m.to_dict() for m in self.map_history],
            "burned_champions": list(self.burned_champions),
            "final_depletion_a": round(self.final_depletion_a, 4),
            "final_depletion_b": round(self.final_depletion_b, 4),
        }


def series_prob_to_map_prob(series_prob: float, best_of: int = 5) -> float:
    """Invert series win probability to implied single-map win probability.

    Parameters
    ----------
    series_prob : float
        Base series win probability in [0, 1].
    best_of : int
        Series length format (1, 3, 5, etc.).

    Returns
    -------
    float
        Implied single-map win probability p in (0, 1).
    """
    p_clipped = float(np.clip(series_prob, 1e-6, 1.0 - 1e-6))
    if best_of <= 1:
        return p_clipped
    if abs(p_clipped - 0.5) < 1e-7:
        return 0.5

    try:
        return float(brentq(lambda p: series_win_probability(p, best_of) - p_clipped, 1e-6, 1.0 - 1e-6))
    except Exception:
        # Monotonic bisection fallback
        low, high = 1e-6, 1.0 - 1e-6
        for _ in range(60):
            mid = 0.5 * (low + high)
            if series_win_probability(mid, best_of) < p_clipped:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)


def compute_team_depletion(
    players: Sequence[str],
    burned_champions: set[int] | frozenset[int] | Sequence[int],
    metrics_cache: dict[str, dict[str, Any]] | None = None,
    db: Any = None,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Compute comfort pool depletion penalty Delta R_team for a 5-player lineup.

    Formula:
        Delta R_team = -gamma * sum_{p=1}^5 (sum_{c in C_p^* cap B} P(c)) * exp(-D_p / 10.0)

    Parameters
    ----------
    players : Sequence[str]
        Ordered 5-player roster identifiers [TOP, JUG, MID, ADC, SUP].
    burned_champions : set[int] | frozenset[int] | Sequence[int]
        Cumulative champions burned so far.
    metrics_cache : dict[str, dict[str, Any]], optional
        Precomputed player pool metrics dictionary to avoid database re-queries.
    db : Any, optional
        Database session for computing player metrics if not cached.
    gamma : float
        Depletion scale factor in rating points (default 15.0).

    Returns
    -------
    float
        Depletion penalty Delta R_team <= 0.0 in rating points.
    """
    burned_set = {int(c) for c in burned_champions}
    if not burned_set:
        return 0.0

    total_penalty = 0.0
    for p in players:
        if metrics_cache is not None and p in metrics_cache:
            m = metrics_cache[p]
        else:
            m = compute_player_pool_metrics(p, db=db)
            if metrics_cache is not None:
                metrics_cache[p] = m

        freqs = m.get("champion_frequencies", {})
        comfort_champs = m.get("comfort_champions", [])
        if not comfort_champs and freqs:
            comfort_champs = [
                cid for cid, freq in freqs.items() if freq >= DEFAULT_COMFORT_THRESHOLD
            ]

        comfort_set = {int(c) for c in comfort_champs}
        burned_comfort = comfort_set.intersection(burned_set)
        if not burned_comfort:
            continue

        p_sum = sum(float(freqs.get(c, freqs.get(str(c), 0.0))) for c in burned_comfort)
        depth = float(m.get("effective_pool_depth", 0.0))
        depth_decay = math.exp(-depth / 10.0)
        total_penalty += p_sum * depth_decay

    return -float(gamma) * total_penalty


def sample_fearless_map_draft(
    players_a: Sequence[str],
    players_b: Sequence[str],
    burned_set: set[int],
    metrics_cache: dict[str, dict[str, Any]],
    rng: random.Random,
) -> list[int]:
    """Sample exactly 10 unique champions for a single map draft under Hard Fearless rules.

    Drafts 5 champions for Team A and 5 champions for Team B across roles:
    - Prioritizes player's historical champion distribution excluding burned champions.
    - Falls back to standard role meta picks excluding burned champions.
    - Guarantees exactly 10 unique champion IDs disjoint from burned_set.

    Parameters
    ----------
    players_a : Sequence[str]
        Team A 5-player lineup.
    players_b : Sequence[str]
        Team B 5-player lineup.
    burned_set : set[int]
        Set of champion IDs burned in prior maps.
    metrics_cache : dict[str, dict[str, Any]]
        Pre-cached player pool metrics.
    rng : random.Random
        Random generator instance.

    Returns
    -------
    list[int]
        List of exactly 10 newly drafted champion IDs.
    """
    drafted: list[int] = []
    drafted_set: set[int] = set()

    def draw_champion(p_id: str, role: str) -> int:
        role_upper = role.upper()
        metrics = metrics_cache.get(p_id, {})
        freqs = metrics.get("champion_frequencies", {})
        candidates = [
            (int(cid), weight)
            for cid, weight in freqs.items()
            if int(cid) not in burned_set and int(cid) not in drafted_set and weight > 0
        ]
        if candidates:
            cids, weights = zip(*candidates)
            total_w = sum(weights)
            norm_w = [w / total_w for w in weights]
            return int(rng.choices(cids, weights=norm_w, k=1)[0])

        meta_candidates = [
            cid for cid in DEFAULT_ROLE_META.get(role_upper, [])
            if cid not in burned_set and cid not in drafted_set
        ]
        if meta_candidates:
            return int(rng.choice(meta_candidates))

        fallback_id = 1000 + len(drafted_set) + 1
        while fallback_id in burned_set or fallback_id in drafted_set:
            fallback_id += 1
        return fallback_id

    for i, role in enumerate(ROLES):
        pick_a = draw_champion(players_a[i], role)
        drafted.append(pick_a)
        drafted_set.add(pick_a)

        pick_b = draw_champion(players_b[i], role)
        drafted.append(pick_b)
        drafted_set.add(pick_b)

    return drafted


def simulate_fearless_series_maps(
    team_a: str,
    team_b: str,
    roster_a: list[str] | None = None,
    roster_b: list[str] | None = None,
    best_of: int = 5,
    higher_seed: str | None = None,
    base_series_prob: float | None = None,
    base_rating_a: float | None = None,
    base_rating_b: float | None = None,
    db: Any = None,
    rng: random.Random = None,
) -> FearlessSeriesResult:
    """Simulate a sequential Best-of-N series under official Hard Fearless Draft rules.

    Parameters
    ----------
    team_a : str
        Canonical identifier/name of Team A.
    team_b : str
        Canonical identifier/name of Team B.
    roster_a : list[str] | None, optional
        List of 5 player IDs for Team A in [TOP, JUG, MID, ADC, SUP] order.
    roster_b : list[str] | None, optional
        List of 5 player IDs for Team B in [TOP, JUG, MID, ADC, SUP] order.
    best_of : int
        Series length format (typically 3 or 5; must be odd positive integer).
    higher_seed : str | None, optional
        Name of the higher-seeded team receiving Game 1 side choice (defaults to team_a).
    base_series_prob : float | None, optional
        Unadjusted series win probability P(Team A wins series).
    base_rating_a : float | None, optional
        Base rating points for Team A (e.g. Elo/Glicko).
    base_rating_b : float | None, optional
        Base rating points for Team B.
    db : Any, optional
        Database session for historical player champion queries.
    rng : random.Random, optional
        Deterministic random generator instance.

    Returns
    -------
    FearlessSeriesResult
        Terminal series result with exact scoreline and chronological map history.
    """
    if best_of not in (1, 3, 5, 7):
        raise ValueError(f"best_of must be an odd integer (1, 3, 5, 7), got {best_of}")

    if rng is None:
        rng = random.Random()
    elif isinstance(rng, int):
        rng = random.Random(rng)

    target_wins = (best_of // 2) + 1

    def resolve_roster(team: str, roster: list[str] | None) -> list[str]:
        if not roster:
            return [f"{team}_{r}" for r in ROLES]
        players = list(roster)
        if len(players) < 5:
            players.extend(f"{team}_{ROLES[i]}" for i in range(len(players), 5))
        return players[:5]

    players_a = resolve_roster(team_a, roster_a)
    players_b = resolve_roster(team_b, roster_b)

    metrics_cache: dict[str, dict[str, Any]] = {}
    for p in players_a + players_b:
        metrics_cache[p] = compute_player_pool_metrics(p, db=db)

    # Derive base neutral logit
    if base_series_prob is not None:
        p_map = series_prob_to_map_prob(base_series_prob, best_of)
        p_clipped = float(np.clip(p_map, 1e-6, 1.0 - 1e-6))
        base_logit = math.log(p_clipped / (1.0 - p_clipped))
    elif base_rating_a is not None or base_rating_b is not None:
        ra = 1500.0 if base_rating_a is None else float(base_rating_a)
        rb = 1500.0 if base_rating_b is None else float(base_rating_b)
        base_logit = (ra - rb) * RATING_TO_LOGIT_SCALE
    else:
        base_logit = 0.0

    score_a = 0
    score_b = 0
    burned_champions: list[int] = []
    burned_set: set[int] = set()
    map_history: list[MapResult] = []
    current_map = 1

    while score_a < target_wins and score_b < target_wins:
        # Side choice sequence:
        # Game 1: Higher seed (or team_a) gets Blue side choice.
        # Game 2+: Loser of previous map chooses side for the next map (default Blue).
        if current_map == 1:
            if higher_seed is not None and higher_seed == team_b:
                blue_team = team_b
                red_team = team_a
            else:
                blue_team = team_a
                red_team = team_b
        else:
            prev_map = map_history[-1]
            blue_team = prev_map.loser
            red_team = prev_map.winner

        # Blue side rating advantage: Delta_side = +-0.1242 in logit space (+25 rating points)
        is_a_blue = (blue_team == team_a)
        side_logit = BLUE_SIDE_LOGIT_DELTA if is_a_blue else -BLUE_SIDE_LOGIT_DELTA

        # Fearless champion pool depletion penalty:
        # |B| = 10 * (current_map - 1) champions burned prior to this map
        burned_count_prior = len(burned_set)
        depletion_a = compute_team_depletion(players_a, burned_set, metrics_cache=metrics_cache)
        depletion_b = compute_team_depletion(players_b, burned_set, metrics_cache=metrics_cache)
        net_depletion_rating = depletion_a - depletion_b
        depletion_logit = net_depletion_rating * RATING_TO_LOGIT_SCALE

        total_logit = base_logit + side_logit + depletion_logit
        p_map_a = 1.0 / (1.0 + math.exp(-total_logit))
        p_map_a = float(np.clip(p_map_a, 1e-6, 1.0 - 1e-6))

        draw = rng.random()
        if draw < p_map_a:
            map_winner = team_a
            map_loser = team_b
            score_a += 1
        else:
            map_winner = team_b
            map_loser = team_a
            score_b += 1

        # Draft exactly 10 champions for this map under Hard Fearless rules
        drafted_10 = sample_fearless_map_draft(
            players_a=players_a,
            players_b=players_b,
            burned_set=burned_set,
            metrics_cache=metrics_cache,
            rng=rng,
        )
        burned_champions.extend(drafted_10)
        burned_set.update(drafted_10)

        map_res = MapResult(
            map_number=current_map,
            winner=map_winner,
            loser=map_loser,
            blue_team=blue_team,
            red_team=red_team,
            score_a=score_a,
            score_b=score_b,
            burned_count=burned_count_prior,
        )
        map_history.append(map_res)
        current_map += 1

    winner = team_a if score_a == target_wins else team_b
    loser = team_b if winner == team_a else team_a
    score_winner = max(score_a, score_b)
    score_loser = min(score_a, score_b)
    maps_played = len(map_history)
    scoreline = f"{score_a}-{score_b}"

    # Calculate final series depletion values over all burned champions
    final_depletion_a = compute_team_depletion(players_a, burned_set, metrics_cache=metrics_cache)
    final_depletion_b = compute_team_depletion(players_b, burned_set, metrics_cache=metrics_cache)

    return FearlessSeriesResult(
        winner=winner,
        loser=loser,
        score_winner=score_winner,
        score_loser=score_loser,
        maps_played=maps_played,
        scoreline=scoreline,
        map_history=map_history,
        burned_champions=burned_champions,
        final_depletion_a=final_depletion_a,
        final_depletion_b=final_depletion_b,
    )


__all__ = [
    "BLUE_SIDE_LOGIT_DELTA",
    "BLUE_SIDE_RATING_OFFSET",
    "FearlessSeriesResult",
    "MapResult",
    "RATING_TO_LOGIT_SCALE",
    "compute_team_depletion",
    "sample_fearless_map_draft",
    "series_prob_to_map_prob",
    "simulate_fearless_series_maps",
]
