"""Stake sizing policies used by historical backtests."""

from __future__ import annotations

from betting_app.core.staking import fixed_stake, fractional_kelly_stake, percent_stake
from betting_app.ml.config import StakingConfig


def stake_for_bet(
    bankroll: float,
    model_prob: float,
    decimal_odds: float,
    config: StakingConfig,
    tax_rate: float,
) -> float:
    """Return stake for one simulated bet using the configured strategy."""

    if bankroll <= 0:
        return 0.0
    strategy = config.strategy.lower()
    if strategy in {"fixed", "flat"}:
        return fixed_stake(config.fixed_stake, bankroll)
    if strategy in {"percent", "bankroll_percent"}:
        return percent_stake(
            bankroll=bankroll,
            fraction=config.bankroll_fraction,
            min_stake=config.min_stake,
            max_stake=config.max_stake or bankroll,
        )
    if strategy in {"kelly", "fractional_kelly"}:
        return fractional_kelly_stake(
            bankroll=bankroll,
            probability=model_prob,
            decimal_odds=decimal_odds,
            fraction=config.kelly_fraction,
            tax_rate=tax_rate,
            min_stake=config.min_stake,
            max_stake=config.max_stake or bankroll,
        )
    raise ValueError(f"Unsupported staking strategy: {config.strategy}")
