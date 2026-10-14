"""Settlement helpers for simulated bets."""

from __future__ import annotations

from betting_app.ml.backtesting.types import Side


def settle_profit(stake: float, decimal_odds: float, side: Side, winner_side: Side, tax_rate: float) -> tuple[str, float]:
    """Return result label and taxed profit for one simulated bet."""

    if side == winner_side:
        return "won", stake * (decimal_odds * (1.0 - tax_rate) - 1.0)
    return "lost", -stake
