"""Typed configuration objects for production ML workflows."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StakingConfig:
    """Stake sizing parameters used by backtests and diagnostics."""

    strategy: str = "fractional_kelly"
    fixed_stake: float = 10.0
    bankroll_fraction: float = 0.01
    kelly_fraction: float = 0.25
    min_stake: float = 0.0
    max_stake: float | None = 100.0
    max_bankroll_fraction: float | None = None


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for deterministic historical bookmaker backtests."""

    bankroll_start: float = 1_000.0
    min_ev: float = 0.0
    tax_rate: float = 0.12
    staking: StakingConfig = field(default_factory=StakingConfig)
    max_bets_per_match: int = 1
    odds_policy: str = "latest_pre_match"
    min_minutes_before_start: int = 0
    require_prediction_before_odds: bool = True
    strict_temporal_eligibility: bool = True
