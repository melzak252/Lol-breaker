"""Unit tests for portfolio-aware fractional Kelly staking."""

from __future__ import annotations

import pytest

from betting_app.core.staking import (
    fixed_stake,
    fractional_kelly_stake,
    percent_stake,
)


def test_fractional_kelly_standard():
    # Bankroll = 1000 PLN, Win Prob = 0.60, Decimal Odds = 2.00, Tax = 12%
    # Net Decimal = 2.0 * 0.88 = 1.76, b = 0.76
    # Full Kelly = (0.76 * 0.60 - 0.40) / 0.76 = (0.456 - 0.40) / 0.76 = 0.056 / 0.76 ~= 0.07368 (7.37%)
    # 5% fractional Kelly = 0.07368 * 0.05 = 0.003684 (0.37% of 1000 = 3.68 PLN)
    stake = fractional_kelly_stake(1000.0, 0.60, 2.00, fraction=0.05, tax_rate=0.12)
    assert 3.5 <= stake <= 4.0


def test_quarter_kelly_default():
    # Default fraction is now 0.25 (1/4 Kelly)
    # 1000 PLN * 0.07368 * 0.25 = 18.42 PLN
    stake = fractional_kelly_stake(1000.0, 0.60, 2.00, tax_rate=0.12)
    assert 18.0 <= stake <= 19.0

def test_fractional_kelly_reserved_bankroll():
    # Total bankroll = 1000 PLN, but 800 PLN is reserved in currently open bets -> Available = 200 PLN
    stake_full = fractional_kelly_stake(1000.0, 0.60, 2.00, fraction=0.05, tax_rate=0.12, reserved_bankroll=0.0)
    stake_reserved = fractional_kelly_stake(1000.0, 0.60, 2.00, fraction=0.05, tax_rate=0.12, reserved_bankroll=800.0)

    # Stake should scale proportionally to the available bankroll (200 / 1000 = 1/5)
    # But subject to min_stake (2.0)
    assert stake_reserved <= stake_full
    assert stake_reserved <= 200.0


def test_fractional_kelly_exhausted_bankroll():
    # When reserved bankroll equals or exceeds total bankroll, stake must be 0.0
    stake = fractional_kelly_stake(500.0, 0.65, 2.20, reserved_bankroll=500.0)
    assert stake == 0.0

    stake_over = fractional_kelly_stake(500.0, 0.65, 2.20, reserved_bankroll=600.0)
    assert stake_over == 0.0


def test_fractional_kelly_portfolio_exposure_cap():
    # Even if Kelly fraction is large, single bet must not exceed max_portfolio_exposure (e.g. 25% of available)
    # Available = 100 PLN, max_exposure = 0.20 -> cap = 20.0 PLN
    stake = fractional_kelly_stake(
        100.0,
        0.90,
        2.50,
        fraction=1.0,  # Full Kelly
        reserved_bankroll=0.0,
        max_portfolio_exposure=0.20,
    )
    assert stake <= 20.0


def test_fractional_kelly_negative_ev():
    # Odds = 1.50, Prob = 0.50 -> Net = 1.32, b = 0.32 -> EV is negative
    stake = fractional_kelly_stake(1000.0, 0.50, 1.50)
    assert stake == 0.0


def test_fixed_and_percent_stake():
    assert fixed_stake(50.0, 100.0) == 50.0
    assert fixed_stake(150.0, 100.0) == 100.0
    assert percent_stake(100.0, 0.05) == 5.0
    assert percent_stake(10.0, 0.01) == 2.0  # clamped to min_stake 2.0
    assert percent_stake(1.0, 0.01) == 0.0  # cannot meet min_stake within 1.0 bankroll
