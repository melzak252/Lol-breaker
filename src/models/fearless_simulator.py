"""Fearless Draft Map-Level Simulator & Series Scoreline Engine.

Implements sequential Markov map-level simulation under official Hard Fearless Draft rules
(e.g., MSI 2025+, Worlds 2025+, LCK Cup).
Couples native C0 attention token masking, parametric champion pool entropy depletion,
and loser side-selection dynamics to produce exact scoreline PMFs and map-by-map trajectories.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.special import expit, logit

from src.features.champion_pool_entropy import (
    DEFAULT_COMFORT_THRESHOLD,
    DEFAULT_GAMMA,
    DEFAULT_ROLE_META,
    ROLES,
    calculate_depletion_penalty,
    compute_player_pool_metrics,
)

# Blue Side offset: +25.0 rating points gives ~+3.1% win rate at 50%
BLUE_SIDE_RATING_OFFSET: float = 25.0
# Scale factor translating rating points to logit space: logit(0.531) / 25.0
RATING_TO_LOGIT_SCALE: float = math.log(0.531 / (1.0 - 0.531)) / 25.0  # ~0.0049683

Side = Literal["blue", "red"]
TeamKey = Literal["A", "B"]


@dataclass(frozen=True)
class FearlessSeriesState:
    """Immutable sequential state for a Fearless Draft series.

    Attributes
    ----------
    score_a : int
        Wins accumulated by Team A.
    score_b : int
        Wins accumulated by Team B.
    burned_champions : frozenset[int]
        Cumulative champion IDs drafted across prior maps in the series.
    current_map : int
        1-indexed map counter (starts at 1).
    current_side_a : Literal['blue', 'red']
        Side assignment for Team A on the current map.
        Map 1 is assigned to higher seed / coin-flip winner.
        Subsequent maps are assigned to the previous map's loser (loser chooses Blue).
    best_of : int
        Best-of series length (typically 3 or 5).
    """
    score_a: int = 0
    score_b: int = 0
    burned_champions: frozenset[int] = field(default_factory=frozenset)
    current_map: int = 1
    current_side_a: Side = "blue"
    best_of: int = 5

    def __post_init__(self) -> None:
        if self.best_of not in (1, 3, 5, 7):
            raise ValueError(f"best_of must be an odd integer (1, 3, 5, 7), got {self.best_of}")
        if self.score_a < 0 or self.score_b < 0:
            raise ValueError(f"Scores must be non-negative, got ({self.score_a}, {self.score_b})")
        if self.current_map < 1:
            raise ValueError(f"current_map must be >= 1, got {self.current_map}")
        if self.current_side_a not in ("blue", "red"):
            raise ValueError(f"current_side_a must be 'blue' or 'red', got {self.current_side_a!r}")
        
        # Hard Fearless Invariant: exactly 10 champions burned per prior map completed
        expected_burned = 10 * (self.current_map - 1)
        if len(self.burned_champions) != expected_burned:
            raise ValueError(
                f"Fearless Invariant violated: current_map={self.current_map} expects "
                f"{expected_burned} burned champions, got {len(self.burned_champions)}"
            )

    @property
    def target_wins(self) -> int:
        """Wins required to win the series: (best_of + 1) // 2."""
        return (self.best_of + 1) // 2

    @property
    def is_terminal(self) -> bool:
        """Check if the series has concluded."""
        return self.score_a >= self.target_wins or self.score_b >= self.target_wins

    @property
    def winner(self) -> TeamKey | None:
        """Winner of the series if terminal, else None."""
        if self.score_a >= self.target_wins:
            return "A"
        if self.score_b >= self.target_wins:
            return "B"
        return None

    @property
    def current_side_b(self) -> Side:
        """Side assignment for Team B."""
        return "red" if self.current_side_a == "blue" else "blue"

    def advance(
        self,
        winner: TeamKey,
        drafted_champions: Sequence[int] | set[int] | frozenset[int],
        loser_side_choice: Side = "blue",
    ) -> FearlessSeriesState:
        """Advance the series state by one completed map.

        Parameters
        ----------
        winner : Literal['A', 'B']
            Winner of the current map.
        drafted_champions : Sequence[int] | set[int]
            The 10 unique champions drafted in the map just played.
        loser_side_choice : Literal['blue', 'red']
            Side chosen by the previous map's loser for the next map (default 'blue').

        Returns
        -------
        FearlessSeriesState
            The new immutable state for the next map.
        """
        if self.is_terminal:
            raise ValueError("Cannot advance a terminal series state")

        drafted_set = frozenset(int(c) for c in drafted_champions)
        if len(drafted_set) != 10:
            raise ValueError(f"Drafted champions must contain exactly 10 unique IDs, got {len(drafted_set)}")
        
        # Ensure no overlap with previously burned champions
        overlap = drafted_set.intersection(self.burned_champions)
        if overlap:
            raise ValueError(f"Drafted champions contain already burned champions: {sorted(overlap)}")

        new_score_a = self.score_a + (1 if winner == "A" else 0)
        new_score_b = self.score_b + (1 if winner == "B" else 0)
        new_burned = self.burned_champions | drafted_set
        next_map = self.current_map + 1

        # Side selection dynamics:
        # The loser of the previous map selects side for the next map.
        # If A won, B is the loser: B chooses `loser_side_choice`.
        #   If B chooses 'blue', then A is 'red'.
        #   If B chooses 'red', then A is 'blue'.
        # If B won, A is the loser: A chooses `loser_side_choice`.
        #   If A chooses 'blue', then A is 'blue'.
        #   If A chooses 'red', then A is 'red'.
        if winner == "A":
            next_side_a: Side = "red" if loser_side_choice == "blue" else "blue"
        else:
            next_side_a = loser_side_choice

        return FearlessSeriesState(
            score_a=new_score_a,
            score_b=new_score_b,
            burned_champions=new_burned,
            current_map=next_map,
            current_side_a=next_side_a,
            best_of=self.best_of,
        )


@dataclass
class FearlessSeriesResult:
    """Complete distribution and progression result of a Fearless Draft series simulation."""
    series_prob_a: float
    series_prob_b: float
    scorelines: dict[str, float]
    scoreline_pmf: dict[tuple[int, int], float]
    map_win_probabilities: dict[int, float]
    map_reach_probabilities: dict[int, float]
    mean_unique_champions: float
    champions_drafted_distribution: dict[int, float]
    n_simulations: int
    best_of: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a clean dictionary."""
        return {
            "best_of": self.best_of,
            "n_simulations": self.n_simulations,
            "series_prob_a": round(self.series_prob_a, 6),
            "series_prob_b": round(self.series_prob_b, 6),
            "scorelines": {k: round(v, 6) for k, v in self.scorelines.items()},
            "scoreline_pmf": {f"{k[0]}-{k[1]}": round(v, 6) for k, v in self.scoreline_pmf.items()},
            "map_win_probabilities": {int(k): round(v, 6) for k, v in self.map_win_probabilities.items()},
            "map_reach_probabilities": {int(k): round(v, 6) for k, v in self.map_reach_probabilities.items()},
            "mean_unique_champions": round(self.mean_unique_champions, 2),
            "champions_drafted_distribution": {int(k): round(v, 6) for k, v in self.champions_drafted_distribution.items()},
        }


def mask_c0_tokens_for_burned_champions(
    hist_arrays: Mapping[str, np.ndarray],
    burned_champions: set[int] | frozenset[int] | Sequence[int],
) -> dict[str, np.ndarray]:
    """Zero-mask C0 historical player map tokens whose champion was burned.

    Parameters
    ----------
    hist_arrays : Mapping[str, np.ndarray]
        Dictionary containing:
        - 'history': shape (..., 2, 5, 16, 33)
        - 'mask': shape (..., 2, 5, 16)
        - 'champions': shape (..., 2, 5, 16)
    burned_champions : set[int] | frozenset[int] | Sequence[int]
        Set of champion IDs burned in prior maps.

    Returns
    -------
    dict[str, np.ndarray]
        Modified copies of the history and mask arrays with burned champion tokens zeroed out.
    """
    burned_set = {int(c) for c in burned_champions}
    out_hist = {k: v.copy() for k, v in hist_arrays.items()}
    if not burned_set or "champions" not in out_hist:
        return out_hist

    champions = out_hist["champions"]
    mask = out_hist.get("mask")
    history = out_hist.get("history")

    # Boolean mask of elements to zero out
    is_burned = np.isin(champions, list(burned_set))

    if mask is not None:
        mask[is_burned] = 0
    if history is not None:
        history[is_burned] = 0.0

    return out_hist


def c0_predict_fearless_map(
    features_a: Any,
    features_b: Any,
    burned_champions: set[int] | frozenset[int] | Sequence[int],
    side_a: Side = "blue",
    c0_adapter: Any = None,
    gamma: float = DEFAULT_GAMMA,
    base_map_prob: float | None = None,
) -> float:
    """Predict single map win probability P(Team A wins) under Fearless Draft rules.

    Reuses native C0 representation/checkpoint, incorporates Blue Side offset
    (+25.0 rating points), zero-masks burned champion tokens, and applies
    the aggregate team champion depletion penalties.

    Parameters
    ----------
    features_a : Any
        Features or roster representation for Team A. Can be a dict with 'roster'
        or 'player_ids', or full C0 snapshot.
    features_b : Any
        Features or roster representation for Team B.
    burned_champions : set[int]
        Cumulative champions burned so far.
    side_a : Literal['blue', 'red']
        Side assignment for Team A ('blue' or 'red').
    c0_adapter : Any, optional
        Native C0 adapter instance.
    gamma : float
        Parametric depletion scale factor (default 15.0 rating points).
    base_map_prob : float, optional
        Explicit base neutral 50/50 or model-derived probability if adapter is omitted.

    Returns
    -------
    float
        Calibrated win probability P(Team A wins map).
    """
    burned_set = {int(c) for c in burned_champions}

    # Extract rosters
    roster_a: list[str] = []
    roster_b: list[str] = []
    if isinstance(features_a, Mapping):
        roster_a = list(features_a.get("roster", features_a.get("player_ids", [])))
    elif isinstance(features_a, (list, tuple)):
        roster_a = [str(p) for p in features_a]

    if isinstance(features_b, Mapping):
        roster_b = list(features_b.get("roster", features_b.get("player_ids", [])))
    elif isinstance(features_b, (list, tuple)):
        roster_b = [str(p) for p in features_b]

    # Compute base logit from C0 adapter if available and full snapshot is provided
    base_logit = 0.0
    if c0_adapter is not None and isinstance(features_a, Mapping) and "snapshot" in features_a:
        snapshot = features_a["snapshot"]
        raw = snapshot.get("raw", {})
        hist = snapshot.get("hist", {})
        # Mask burned champions
        masked_hist = mask_c0_tokens_for_burned_champions(hist, burned_set)
        # Execute C0 prediction
        pred = c0_adapter._predict_arrays(
            features_a.get("request", {"best_of": 1, "mode": "full", "decision_at": None}),
            snapshot.get("target", {}),
            raw,
            masked_hist,
            snapshot.get("provenance", {}),
            canonical_match_id=None,
        )
        p_c0 = float(pred.get("prob_a", 0.5))
        base_logit = float(logit(np.clip(p_c0, 1e-6, 1.0 - 1e-6)))
    elif base_map_prob is not None:
        p0 = float(np.clip(base_map_prob, 1e-6, 1.0 - 1e-6))
        base_logit = float(logit(p0))
    elif isinstance(features_a, Mapping) and "rating" in features_a and "rating" in features_b:
        r_a = float(features_a["rating"])
        r_b = float(features_b["rating"])
        base_logit = (r_a - r_b) * RATING_TO_LOGIT_SCALE
    elif isinstance(features_a, Mapping) and "p_base" in features_a:
        p0 = float(np.clip(features_a["p_base"], 1e-6, 1.0 - 1e-6))
        base_logit = float(logit(p0))

    # Calculate side advantage in rating points:
    # Blue side receives +25.0 rating points; Red side receives -25.0 rating points
    side_rating_offset = BLUE_SIDE_RATING_OFFSET if side_a == "blue" else -BLUE_SIDE_RATING_OFFSET
    side_logit = side_rating_offset * RATING_TO_LOGIT_SCALE

    # Calculate depletion penalties for both teams
    depletion_a = sum(
        calculate_depletion_penalty(p, burned_set, gamma=gamma)
        for p in roster_a
    )
    depletion_b = sum(
        calculate_depletion_penalty(p, burned_set, gamma=gamma)
        for p in roster_b
    )

    # Net depletion effect: delta_r_a - delta_r_b (where delta_r <= 0)
    # If Team A suffers -30 and Team B suffers 0, net is -30 (hurts Team A)
    net_depletion_rating = depletion_a - depletion_b
    depletion_logit = net_depletion_rating * RATING_TO_LOGIT_SCALE

    # Combined map logit
    total_logit = base_logit + side_logit + depletion_logit
    p_map_a = float(expit(total_logit))

    return float(np.clip(p_map_a, 1e-6, 1.0 - 1e-6))


def sample_map_draft(
    team_a_roster: Sequence[str] | Mapping[str, Any],
    team_b_roster: Sequence[str] | Mapping[str, Any],
    burned_champions: set[int] | frozenset[int] | Sequence[int],
    patch: str | None = None,
    rng: np.random.Generator | random.Random | None = None,
) -> list[int]:
    """Sample 10 unique champions for a map draft adhering to Hard Fearless rules.

    For each role (TOP, JUG, MID, ADC, SUP) for both teams:
    - Samples from the player's historical champion distribution conditioned on
      champion_id not in burned_champions and not already drafted in this map.
    - If the player's pool is exhausted, samples from the role's general meta pool.

    Parameters
    ----------
    team_a_roster : Sequence[str] | Mapping[str, Any]
        Team A players in order [TOP, JUG, MID, ADC, SUP] or dict mapping role to player.
    team_b_roster : Sequence[str] | Mapping[str, Any]
        Team B players in order [TOP, JUG, MID, ADC, SUP] or dict mapping role to player.
    burned_champions : set[int]
        Champions burned across previous maps.
    patch : str, optional
        Game patch (optional filter).
    rng : np.random.Generator | random.Random, optional
        Random number generator for reproducibility.

    Returns
    -------
    list[int]
        List of exactly 10 newly drafted champion IDs (5 for Team A, 5 for Team B).
    """
    burned_set = {int(c) for c in burned_champions}
    drafted: list[int] = []
    drafted_set: set[int] = set()

    # Extract role-to-player mappings
    def get_role_map(roster: Sequence[str] | Mapping[str, Any]) -> dict[str, str]:
        if isinstance(roster, Mapping):
            return {role.upper(): str(roster.get(role.lower(), roster.get(role.upper(), ""))) for role in ROLES}
        return {ROLES[i]: str(roster[i]) if i < len(roster) else "" for i in range(len(ROLES))}

    map_a = get_role_map(team_a_roster)
    map_b = get_role_map(team_b_roster)

    def draw_champion(player_id: str, role: str) -> int:
        role_upper = role.upper()
        # 1. Try to sample from player's historical pool
        if player_id:
            metrics = compute_player_pool_metrics(player_id)
            freqs = metrics.get("champion_frequencies", {})
            # Filter candidates: not in burned_champions and not in current map's drafted_set
            candidates = [
                (int(cid), weight)
                for cid, weight in freqs.items()
                if int(cid) not in burned_set and int(cid) not in drafted_set and weight > 0
            ]
            if candidates:
                cids, weights = zip(*candidates)
                total_w = sum(weights)
                norm_w = [w / total_w for w in weights]
                if isinstance(rng, np.random.Generator):
                    pick = int(rng.choice(cids, p=norm_w))
                elif isinstance(rng, random.Random):
                    pick = int(rng.choices(cids, weights=norm_w, k=1)[0])
                else:
                    pick = int(np.random.choice(cids, p=norm_w))
                return pick

        # 2. Fall back to general role meta
        meta_candidates = [
            cid for cid in DEFAULT_ROLE_META.get(role_upper, [])
            if cid not in burned_set and cid not in drafted_set
        ]
        if meta_candidates:
            if isinstance(rng, np.random.Generator):
                pick = int(rng.choice(meta_candidates))
            elif isinstance(rng, random.Random):
                pick = int(rng.choice(meta_candidates))
            else:
                pick = int(np.random.choice(meta_candidates))
            return pick

        # 3. Last-resort fallback: any unused valid champion ID from 1000 upwards
        fallback_id = 1000 + len(drafted_set) + 1
        while fallback_id in burned_set or fallback_id in drafted_set:
            fallback_id += 1
        return fallback_id

    # Alternating draft sequence for 5 roles
    for role in ROLES:
        # Team A pick
        pick_a = draw_champion(map_a[role], role)
        drafted.append(pick_a)
        drafted_set.add(pick_a)

        # Team B pick
        pick_b = draw_champion(map_b[role], role)
        drafted.append(pick_b)
        drafted_set.add(pick_b)

    assert len(drafted) == 10, f"Expected 10 drafted champions, got {len(drafted)}"
    assert len(drafted_set) == 10, f"Expected 10 unique champions, got {len(drafted_set)}"
    assert drafted_set.isdisjoint(burned_set), "Drafted champions overlap with burned champions"

    return drafted


def simulate_fearless_series(
    team_a: Any,
    team_b: Any,
    best_of: int = 5,
    n_simulations: int = 10_000,
    seed: int = 42,
    initial_side_a: Side = "blue",
    c0_adapter: Any = None,
    base_map_prob: float | None = None,
    gamma: float = DEFAULT_GAMMA,
) -> FearlessSeriesResult:
    """Execute sequential Monte Carlo simulation of a Fearless Draft series.

    Parameters
    ----------
    team_a : Any
        Representation of Team A (roster dict, list of players, or features).
    team_b : Any
        Representation of Team B.
    best_of : int
        Best-of series length (3 or 5).
    n_simulations : int
        Number of full sequential series paths to simulate (default 10,000).
    seed : int
        RNG seed for reproducibility.
    initial_side_a : Literal['blue', 'red']
        Map 1 side for Team A (default 'blue').
    c0_adapter : Any, optional
        Native C0 adapter for map inference.
    base_map_prob : float, optional
        Base unadjusted map probability.
    gamma : float
        Depletion decay scale.

    Returns
    -------
    FearlessSeriesResult
        Full result containing exact scoreline PMFs, series win probability,
        map-by-map conditional win rates, and champion usage distributions.
    """
    if best_of not in (1, 3, 5, 7):
        raise ValueError(f"best_of must be 1, 3, 5, or 7, got {best_of}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}")

    rng = np.random.default_rng(seed)
    target_wins = (best_of + 1) // 2

    # Storage for statistics
    wins_a = 0
    scoreline_counts: Counter[tuple[int, int]] = Counter()
    map_played_counts: Counter[int] = Counter()
    map_win_a_counts: Counter[int] = Counter()
    unique_champions_counts: Counter[int] = Counter()

    # Pre-generate legal scorelines to guarantee exact PMF completeness
    legal_scorelines: list[tuple[int, int]] = []
    for lost in range(target_wins):
        legal_scorelines.append((target_wins, lost))
        legal_scorelines.append((lost, target_wins))
    legal_scorelines = sorted(legal_scorelines, key=lambda s: (s[0] + s[1], -s[0]))

    for _ in range(n_simulations):
        state = FearlessSeriesState(
            score_a=0,
            score_b=0,
            burned_champions=frozenset(),
            current_map=1,
            current_side_a=initial_side_a,
            best_of=best_of,
        )

        while not state.is_terminal:
            m = state.current_map
            map_played_counts[m] += 1

            # Predict map win probability
            p_map_a = c0_predict_fearless_map(
                features_a=team_a,
                features_b=team_b,
                burned_champions=state.burned_champions,
                side_a=state.current_side_a,
                c0_adapter=c0_adapter,
                gamma=gamma,
                base_map_prob=base_map_prob,
            )

            # Sample map winner
            map_winner: TeamKey = "A" if rng.random() < p_map_a else "B"
            if map_winner == "A":
                map_win_a_counts[m] += 1

            # Sample 10 drafted champions for this map
            drafted_10 = sample_map_draft(
                team_a_roster=team_a,
                team_b_roster=team_b,
                burned_champions=state.burned_champions,
                rng=rng,
            )

            # Advance state (loser chooses Blue side)
            state = state.advance(
                winner=map_winner,
                drafted_champions=drafted_10,
                loser_side_choice="blue",
            )

        # Record terminal series statistics
        if state.winner == "A":
            wins_a += 1
        scoreline = (state.score_a, state.score_b)
        scoreline_counts[scoreline] += 1
        unique_champions_counts[len(state.burned_champions)] += 1

    # Probability mass aggregation
    series_prob_a = float(wins_a / n_simulations)
    series_prob_b = 1.0 - series_prob_a

    scoreline_pmf: dict[tuple[int, int], float] = {
        s: float(scoreline_counts[s] / n_simulations) for s in legal_scorelines
    }
    
    # Ensure exact conservation of probability mass
    pmf_sum = sum(scoreline_pmf.values())
    if pmf_sum > 0:
        scoreline_pmf = {s: p / pmf_sum for s, p in scoreline_pmf.items()}

    scorelines_str: dict[str, float] = {
        f"{s[0]}-{s[1]}": p for s, p in scoreline_pmf.items()
    }

    # Conditional map win probabilities: P(A wins map t | map t is played)
    map_win_probs: dict[int, float] = {}
    map_reach_probs: dict[int, float] = {}
    for m in range(1, best_of + 1):
        played = map_played_counts[m]
        map_reach_probs[m] = float(played / n_simulations)
        if played > 0:
            map_win_probs[m] = float(map_win_a_counts[m] / played)
        else:
            map_win_probs[m] = 0.0

    # Cumulative unique champions drafted distribution
    total_unique_sum = sum(c * cnt for c, cnt in unique_champions_counts.items())
    mean_unique = float(total_unique_sum / n_simulations)
    champions_dist = {
        c: float(cnt / n_simulations) for c, cnt in sorted(unique_champions_counts.items())
    }

    return FearlessSeriesResult(
        series_prob_a=series_prob_a,
        series_prob_b=series_prob_b,
        scorelines=scorelines_str,
        scoreline_pmf=scoreline_pmf,
        map_win_probabilities=map_win_probs,
        map_reach_probabilities=map_reach_probs,
        mean_unique_champions=mean_unique,
        champions_drafted_distribution=champions_dist,
        n_simulations=n_simulations,
        best_of=best_of,
    )
