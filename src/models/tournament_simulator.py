"""Corrected Tournament Predictor and Series Simulator for LoL.

Addresses two core limitations of the naive sports model for tournament simulation:
1. Inter-region rating desynchronization (hierarchical regional offsets).
2. Binomial series independence violation (Markov momentum series dynamics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


DEFAULT_REGIONAL_OFFSETS: dict[str, float] = {
    "LCK": 0.463,         # +80 Elo equivalent vs LEC anchor
    "LPL": 0.355,         # +62 Elo equivalent vs LEC anchor
    "LCS": 0.013,         # +2 Elo equivalent vs LEC anchor
    "LEC": 0.000,         # reference anchor
    "Minor Tier 1": -0.456,# -79 Elo equivalent vs LEC anchor
    "Other": 0.000,
}


@dataclass
class CorrectedTournamentPredictor:
    """Predicts match probabilities and simulates tournament series without market odds."""

    regional_gamma: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_REGIONAL_OFFSETS))
    beta_momentum: float = 0.35
    eps: float = 1e-6

    def adjust_single_game_prob(
        self,
        base_p: float,
        region_a: str | None,
        region_b: str | None,
    ) -> float:
        """Adjust base single-game probability by regional strength offset."""
        if not region_a or not region_b or region_a == region_b:
            return float(np.clip(base_p, self.eps, 1.0 - self.eps))

        p_clipped = float(np.clip(base_p, self.eps, 1.0 - self.eps))
        logit0 = math.log(p_clipped / (1.0 - p_clipped))
        ga = self.regional_gamma.get(region_a, 0.0)
        gb = self.regional_gamma.get(region_b, 0.0)
        adj_logit = logit0 + (ga - gb)
        return float(1.0 / (1.0 + math.exp(-adj_logit)))

    def series_probabilities(
        self,
        p_game: float,
        best_of: int,
    ) -> tuple[float, dict[tuple[int, int], float]]:
        """Return (win_prob_team_a, exact_score_distribution)."""
        if best_of <= 1:
            p = float(np.clip(p_game, self.eps, 1.0 - self.eps))
            return p, {(1, 0): p, (0, 1): 1.0 - p}

        wins_needed = best_of // 2 + 1
        p_clipped = float(np.clip(p_game, self.eps, 1.0 - self.eps))
        logit0 = math.log(p_clipped / (1.0 - p_clipped))

        scores: dict[tuple[int, int], float] = {}

        def _traverse(w1: int, w2: int, path_prob: float) -> None:
            if w1 == wins_needed or w2 == wins_needed:
                scores[(w1, w2)] = scores.get((w1, w2), 0.0) + path_prob
                return
            diff = w1 - w2
            logit_k = logit0 + self.beta_momentum * diff
            pk = 1.0 / (1.0 + math.exp(-logit_k))
            _traverse(w1 + 1, w2, path_prob * pk)
            _traverse(w1, w2 + 1, path_prob * (1.0 - pk))

        _traverse(0, 0, 1.0)
        win_prob_a = sum(prob for (w1, w2), prob in scores.items() if w1 == wins_needed)
        return win_prob_a, scores

    def simulate_series(
        self,
        p_game: float,
        best_of: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int, int]:
        """Simulate one series outcome returning (winner_index_0_or_1, score_a, score_b)."""
        gen = rng or np.random.default_rng()
        if best_of <= 1:
            win = 1 if gen.random() < p_game else 0
            return (win, win, 1 - win)

        wins_needed = best_of // 2 + 1
        w1, w2 = 0, 0
        p_clipped = float(np.clip(p_game, self.eps, 1.0 - self.eps))
        logit0 = math.log(p_clipped / (1.0 - p_clipped))

        while w1 < wins_needed and w2 < wins_needed:
            diff = w1 - w2
            logit_k = logit0 + self.beta_momentum * diff
            pk = 1.0 / (1.0 + math.exp(-logit_k))
            if gen.random() < pk:
                w1 += 1
            else:
                w2 += 1

        return (1 if w1 > w2 else 0, w1, w2)

TOURNAMENT_PAIRWISE_CAP: float = 0.88
TOURNAMENT_TEMPERATURE: float = 1.12
TOURNAMENT_ENTROPY_DAMPENING: float = 0.08


def calibrate_pairwise_table(
    raw_series_table: dict[tuple[str, str, int], float],
    cap: float = TOURNAMENT_PAIRWISE_CAP,
    temperature: float = TOURNAMENT_TEMPERATURE,
    dampening: float = TOURNAMENT_ENTROPY_DAMPENING,
) -> dict[tuple[str, str, int], float]:
    """Calibrate pairwise series probabilities for realistic tournament simulation.

    Prevents multi-round probability compounding over long bracket paths.
    """
    calibrated: dict[tuple[str, str, int], float] = {}
    for (team_a, team_b, best_of), p_raw in raw_series_table.items():
        p_clipped = min(0.9999, max(0.0001, float(p_raw)))
        z = math.log(p_clipped / (1.0 - p_clipped))

        # 1. Temperature scaling on logits
        z_scaled = z / temperature
        p_scaled = 1.0 / (1.0 + math.exp(-z_scaled))

        # 2. Entropy dampening (unmodeled fatigue/draft variance)
        p_dampened = (1.0 - dampening) * p_scaled + dampening * 0.50

        # 3. Variance ceiling truncation
        p_final = min(cap, max(1.0 - cap, p_dampened))
        calibrated[team_a, team_b, best_of] = float(p_final)

    return calibrated
