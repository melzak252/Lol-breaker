"""In-path tournament Bayesian momentum filter for sequential rounds.

Tracks team momentum, confidence, and meta-adaptation drift across sequential
tournament rounds using a state-space Bayesian filter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import random
from typing import Any, Mapping

import numpy as np


@dataclass
class TeamMomentumState:
    """State variables tracking in-tournament team momentum, confidence, and meta-adaptation drift.

    Attributes:
        current_logit_offset: Drift applied to team's match logit, clamped to [-max_abs, +max_abs].
        series_played: Number of series played in tournament so far.
        series_won: Number of series won.
        series_lost: Number of series lost.
        maps_won: Total individual maps won across all series.
        maps_lost: Total individual maps lost across all series.
        win_streak: Current consecutive series win streak.
        loss_streak: Current consecutive series loss streak.
        team: Identifier of the team.
    """

    current_logit_offset: float = 0.0
    series_played: int = 0
    series_won: int = 0
    series_lost: int = 0
    maps_won: int = 0
    maps_lost: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    team: str = ""

    @property
    def map_differential(self) -> int:
        """Net map differential (maps_won - maps_lost)."""
        return self.maps_won - self.maps_lost

    @property
    def series_differential(self) -> int:
        """Net series differential (series_won - series_lost)."""
        return self.series_won - self.series_lost

    @property
    def win_rate(self) -> float:
        """Series win rate (series_won / series_played) or 0.0 if unplayed."""
        return self.series_won / self.series_played if self.series_played > 0 else 0.0

    @property
    def map_win_rate(self) -> float:
        """Map win rate or 0.0 if no maps played."""
        total = self.maps_won + self.maps_lost
        return self.maps_won / total if total > 0 else 0.0


def _safe_logit(p: float, eps: float = 1e-12) -> float:
    """Numerically stable logit with epsilon clamping."""
    if p <= 0.0:
        return math.log(eps / (1.0 - eps))
    if p >= 1.0:
        return math.log((1.0 - eps) / eps)
    p_clamped = min(max(float(p), eps), 1.0 - eps)
    return math.log(p_clamped / (1.0 - p_clamped))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class InPathTournamentBayesianFilter:
    """State-space Bayesian filter tracking in-tournament team momentum.

    Tracks team momentum, confidence, and meta-adaptation drift across sequential rounds.
    The update preserves the martingale property E[Δθ | p] = 0, remaining unbiased on
    average while capturing path-dependent variance and underdog runs.
    """

    def __init__(
        self,
        learning_rate: float = 0.12,
        adaptation_volatility: float = 0.04,
        max_abs_logit_offset: float = 0.85,
        rng: random.Random | None = None,
        map_margin_weight: float = 0.03,
    ) -> None:
        """Initialize the in-path Bayesian momentum filter.

        Args:
            learning_rate: Innovation weight (eta), default 0.12.
            adaptation_volatility: Meta adaptation shock standard deviation (sigma), default 0.04.
            max_abs_logit_offset: Maximum absolute logit offset clamping bound, default 0.85.
            rng: Optional random.Random instance or seed for reproducible meta shocks.
            map_margin_weight: Bonus per map differential, default 0.03.
        """
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError(f"learning_rate must be non-negative and finite (got {learning_rate})")
        if not math.isfinite(adaptation_volatility) or adaptation_volatility < 0.0:
            raise ValueError(
                f"adaptation_volatility must be non-negative and finite (got {adaptation_volatility})"
            )
        if not math.isfinite(max_abs_logit_offset) or max_abs_logit_offset < 0.0:
            raise ValueError(
                f"max_abs_logit_offset must be non-negative and finite (got {max_abs_logit_offset})"
            )
        if not math.isfinite(map_margin_weight) or map_margin_weight < 0.0:
            raise ValueError(
                f"map_margin_weight must be non-negative and finite (got {map_margin_weight})"
            )

        self.learning_rate = float(learning_rate)
        self.adaptation_volatility = float(adaptation_volatility)
        self.max_abs_logit_offset = float(max_abs_logit_offset)
        self.map_margin_weight = float(map_margin_weight)

        if rng is None:
            self.rng = random.Random()
        elif isinstance(rng, int):
            self.rng = random.Random(rng)
        else:
            self.rng = rng

        self._states: dict[str, TeamMomentumState] = {}

    def get_state(self, team: str) -> TeamMomentumState:
        """Retrieve the momentum state for a given team, initializing if unseen.

        Args:
            team: Team identifier string.

        Returns:
            TeamMomentumState for the specified team.
        """
        team_key = str(team).strip()
        if team_key not in self._states:
            self._states[team_key] = TeamMomentumState(team=team_key)
        return self._states[team_key]

    def get_adjusted_probability(
        self,
        base_probability: float,
        team_a: str,
        team_b: str,
    ) -> float:
        """Compute momentum-adjusted series or match probability for Team A vs Team B.

        Formula:
            logit(p_adj) = logit(p_base) + theta_A - theta_B
            p_adj = sigmoid(logit(p_adj))

        Args:
            base_probability: Unadjusted base probability of Team A winning.
            team_a: Team A identifier.
            team_b: Team B identifier.

        Returns:
            Adjusted probability in [0.0, 1.0].
        """
        if base_probability <= 0.0:
            return 0.0
        if base_probability >= 1.0:
            return 1.0

        theta_a = self.get_state(team_a).current_logit_offset
        theta_b = self.get_state(team_b).current_logit_offset

        base_logit = _safe_logit(base_probability)
        adj_logit = base_logit + theta_a - theta_b
        adj_prob = _sigmoid(adj_logit)
        return float(min(max(adj_prob, 0.0), 1.0))

    def update_after_series(
        self,
        team_a: str,
        team_b: str,
        winner: str,
        maps_a: int,
        maps_b: int,
        pre_match_prob_a: float,
    ) -> None:
        """Update momentum states for both teams after a completed series.

        Update formula:
            Delta theta = eta * (1_won - p_pre_match) + map_margin_bonus + eps_meta
            where:
                eta = self.learning_rate (default 0.12)
                map_margin_bonus = self.map_margin_weight * (maps_won - maps_lost) (default 0.03 * delta)
                eps_meta ~ Normal(0, sigma^2) (default sigma 0.04)

        Args:
            team_a: Team A identifier.
            team_b: Team B identifier.
            winner: Identifier of the winning team (must match team_a or team_b).
            maps_a: Maps won by Team A in this series.
            maps_b: Maps won by Team B in this series.
            pre_match_prob_a: Pre-match win probability for Team A.
        """
        team_a_clean = str(team_a).strip()
        team_b_clean = str(team_b).strip()
        winner_clean = str(winner).strip()

        if team_a_clean == team_b_clean:
            raise ValueError(f"team_a and team_b must be distinct (got '{team_a}')")

        if winner_clean == team_a_clean or winner_clean.lower() == team_a_clean.lower():
            won_a = 1.0
            won_b = 0.0
        elif winner_clean == team_b_clean or winner_clean.lower() == team_b_clean.lower():
            won_a = 0.0
            won_b = 1.0
        else:
            raise ValueError(
                f"Winner '{winner}' does not match team_a ('{team_a}') or team_b ('{team_b}')"
            )

        if maps_a < 0 or maps_b < 0:
            raise ValueError(f"Map scores must be non-negative (got {maps_a}, {maps_b})")

        p_a = float(min(max(pre_match_prob_a, 0.0), 1.0))
        p_b = 1.0 - p_a

        margin_a = maps_a - maps_b
        margin_b = maps_b - maps_a

        bonus_a = self.map_margin_weight * margin_a
        bonus_b = self.map_margin_weight * margin_b

        eps_a = (
            self.rng.gauss(0.0, self.adaptation_volatility)
            if self.adaptation_volatility > 0.0
            else 0.0
        )
        eps_b = (
            self.rng.gauss(0.0, self.adaptation_volatility)
            if self.adaptation_volatility > 0.0
            else 0.0
        )

        delta_theta_a = self.learning_rate * (won_a - p_a) + bonus_a + eps_a
        delta_theta_b = self.learning_rate * (won_b - p_b) + bonus_b + eps_b

        state_a = self.get_state(team_a_clean)
        state_b = self.get_state(team_b_clean)

        new_offset_a = state_a.current_logit_offset + delta_theta_a
        new_offset_b = state_b.current_logit_offset + delta_theta_b

        state_a.current_logit_offset = float(
            min(max(new_offset_a, -self.max_abs_logit_offset), self.max_abs_logit_offset)
        )
        state_b.current_logit_offset = float(
            min(max(new_offset_b, -self.max_abs_logit_offset), self.max_abs_logit_offset)
        )

        state_a.series_played += 1
        if won_a > 0.5:
            state_a.series_won += 1
            state_a.win_streak += 1
            state_a.loss_streak = 0
        else:
            state_a.series_lost += 1
            state_a.loss_streak += 1
            state_a.win_streak = 0
        state_a.maps_won += maps_a
        state_a.maps_lost += maps_b

        state_b.series_played += 1
        if won_b > 0.5:
            state_b.series_won += 1
            state_b.win_streak += 1
            state_b.loss_streak = 0
        else:
            state_b.series_lost += 1
            state_b.loss_streak += 1
            state_b.win_streak = 0
        state_b.maps_won += maps_b
        state_b.maps_lost += maps_a

    def reset(self) -> None:
        """Reset all team momentum states, clearing filter back to round 0."""
        self._states.clear()

    def copy(self) -> InPathTournamentBayesianFilter:
        """Create an independent copy of this filter."""
        new_filter = InPathTournamentBayesianFilter(
            learning_rate=self.learning_rate,
            adaptation_volatility=self.adaptation_volatility,
            max_abs_logit_offset=self.max_abs_logit_offset,
            rng=copy.deepcopy(self.rng),
            map_margin_weight=self.map_margin_weight,
        )
        new_filter._states = {
            team: copy.copy(state) for team, state in self._states.items()
        }
        return new_filter

    def get_all_states(self) -> dict[str, TeamMomentumState]:
        """Return a mapping of team names to current TeamMomentumState."""
        return dict(self._states)

    @property
    def team_offsets(self) -> dict[str, float]:
        """Return current logit offsets for all tracked teams."""
        return {team: s.current_logit_offset for team, s in self._states.items()}

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, team: str) -> bool:
        return str(team).strip() in self._states

    def __repr__(self) -> str:
        return (
            f"InPathTournamentBayesianFilter("
            f"learning_rate={self.learning_rate}, "
            f"volatility={self.adaptation_volatility}, "
            f"max_abs_offset={self.max_abs_logit_offset}, "
            f"tracked_teams={len(self._states)})"
        )


__all__ = [
    "TeamMomentumState",
    "InPathTournamentBayesianFilter",
]
