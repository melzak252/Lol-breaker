"""Shared competition-strength adjustment for rating probabilities.

The family-calibrated Glicko successor learns competition location differences in
Glicko rating points.  Other rating systems can consume the same posterior in a
scale-independent way by converting it to log-odds.  Missing affiliations must
be represented by the neutral adjustment ``mean=variance=0``; absence of bridge
evidence is not evidence that the underlying rating probability is 50%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


RATING_POINTS_TO_LOG_ODDS = math.log(10.0) / 400.0
PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class CompetitionAdjustment:
    """Gaussian side-A minus side-B competition strength in rating points."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("competition adjustment mean must be finite")
        if not math.isfinite(self.variance) or self.variance < 0.0:
            raise ValueError("competition adjustment variance must be finite and non-negative")

    def reversed(self) -> CompetitionAdjustment:
        """Return the same posterior from the opposite side orientation."""

        return CompetitionAdjustment(mean=-self.mean, variance=self.variance)


NEUTRAL_COMPETITION_ADJUSTMENT = CompetitionAdjustment(mean=0.0, variance=0.0)


def adjust_probability(
    probability: float,
    adjustment: CompetitionAdjustment,
    *,
    epsilon: float = PROBABILITY_EPSILON,
) -> float:
    """Marginalize a shared competition-strength posterior into a probability.

    ``probability`` is the base side-A win probability from any rating family.
    The Gaussian location difference is converted from Glicko rating points to
    log-odds.  The logistic-normal approximation attenuates the resulting logit
    when the bridge posterior is uncertain.  Zero adjustment is an exact
    identity operation.
    """

    probability = float(probability)
    epsilon = float(epsilon)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("base probability must be finite and in [0, 1]")
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5)")
    if adjustment == NEUTRAL_COMPETITION_ADJUSTMENT:
        return probability

    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    base_logit = math.log(clipped / (1.0 - clipped))
    mean_logit = adjustment.mean * RATING_POINTS_TO_LOG_ODDS
    variance_logit = adjustment.variance * RATING_POINTS_TO_LOG_ODDS**2
    attenuation = math.sqrt(1.0 + math.pi * variance_logit / 8.0)
    adjusted_logit = (base_logit + mean_logit) / attenuation
    if adjusted_logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-adjusted_logit))
    exp_logit = math.exp(adjusted_logit)
    return exp_logit / (1.0 + exp_logit)
