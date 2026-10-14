"""Closing line value helpers."""

from __future__ import annotations

from betting_app.core.ev import implied_probability


def clv_odds_pct(taken_odds: float, comparison_odds: float) -> float:
    """Return CLV as relative odds movement.

    Positive value means the taken odds were better than the comparison/closing
    odds for the same side.
    """

    if comparison_odds <= 0:
        return 0.0
    return (taken_odds / comparison_odds - 1.0) * 100.0


def clv_probability_points(taken_odds: float, comparison_odds: float) -> float:
    """Return CLV in implied-probability percentage points."""

    return (implied_probability(comparison_odds) - implied_probability(taken_odds)) * 100.0
