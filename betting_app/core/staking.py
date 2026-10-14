"""Staking helpers for diagnostic bet sizing."""

from __future__ import annotations


def fixed_stake(amount: float, bankroll: float) -> float:
    """Return a fixed stake capped by bankroll."""

    return max(0.0, min(float(amount), float(bankroll)))


def percent_stake(bankroll: float, fraction: float, min_stake: float = 2.0, max_stake: float = 100.0) -> float:
    """Return a bankroll-percent stake with min/max limits."""

    if bankroll <= 0:
        return 0.0
    stake = bankroll * fraction
    stake = min(max(stake, min_stake), max_stake)
    return stake if stake <= bankroll else 0.0


def fractional_kelly_stake(
    bankroll: float,
    probability: float,
    decimal_odds: float,
    fraction: float = 0.25,
    tax_rate: float = 0.12,
    min_stake: float = 2.0,
    max_stake: float = 150.0,
    reserved_bankroll: float = 0.0,
    max_portfolio_exposure: float = 0.25,
) -> float:
    """Calculate fractional Kelly stake for taxed decimal odds with bankroll reservation.

    Parameters
    ----------
    bankroll : float
        Total nominal bankroll.
    probability : float
        Estimated win probability for the selected side.
    decimal_odds : float
        Gross decimal odds offered by the bookmaker.
    fraction : float, default 0.25
        Kelly fraction multiplier (default 25% / quarter-Kelly).
    tax_rate : float, default 0.12
        Polish turnover betting tax (12%).
    min_stake : float, default 2.0
        Minimum allowable stake in currency units.
    max_stake : float, default 150.0
        Absolute maximum single bet stake ceiling.
    reserved_bankroll : float, default 0.0
        Capital currently committed in open/unsettled bets.
    max_portfolio_exposure : float, default 0.25
        Maximum fraction of the available bankroll that a single bet can consume.
    """
    effective_bankroll = max(0.0, float(bankroll) - max(0.0, float(reserved_bankroll)))
    if effective_bankroll <= 0 or decimal_odds <= 1.0 or not 0 < probability < 1:
        return 0.0
    net_decimal = decimal_odds * (1.0 - tax_rate)
    b_value = net_decimal - 1.0
    if b_value <= 0:
        return 0.0
    full_kelly = ((b_value * probability) - (1.0 - probability)) / b_value
    if full_kelly <= 0:
        return 0.0
    stake = effective_bankroll * full_kelly * fraction
    portfolio_cap = effective_bankroll * max_portfolio_exposure
    effective_max = min(max_stake, portfolio_cap)
    if effective_max < min_stake:
        return 0.0
    stake = min(max(stake, min_stake), effective_max)
    return stake if stake <= effective_bankroll else 0.0


def step_compounded_flat_stake(
    bankroll: float,
    unit_fraction: float = 0.03,
    min_stake: float = 2.0,
    max_stake: float = 150.0,
    reserved_bankroll: float = 0.0,
    max_portfolio_exposure: float = 0.25,
    integer_round: bool = True,
) -> float:
    """Calculate step-compounded flat unit stake rounded to whole integer PLN.

    Grows unit size smoothly with bankroll (e.g. 3 PLN at 100 PLN -> 30 PLN at 1,000 PLN
    -> capped at 150 PLN at 5,000+ PLN) while giving every qualifying bet the same
    equal-weighted unit at a given bankroll level and avoiding non-integer bot flags.
    """
    effective_bankroll = max(0.0, float(bankroll) - max(0.0, float(reserved_bankroll)))
    if effective_bankroll < min_stake or unit_fraction <= 0:
        return 0.0
    raw_unit = effective_bankroll * float(unit_fraction)
    if integer_round:
        raw_unit = float(round(raw_unit))
    portfolio_cap = effective_bankroll * max_portfolio_exposure
    effective_max = min(float(max_stake), portfolio_cap)
    if integer_round:
        effective_max = float(int(effective_max))
    if effective_max < min_stake:
        return 0.0
    stake = min(max(raw_unit, float(min_stake)), effective_max)
    return stake if stake <= effective_bankroll else 0.0


def capped_wasserstein_kelly_stake(
    bankroll: float,
    probability: float,
    decimal_odds: float,
    wasserstein_epsilon: float = 0.025,
    fraction: float = 0.25,
    tax_rate: float = 0.12,
    min_stake: float = 2.0,
    max_stake: float = 150.0,
    reserved_bankroll: float = 0.0,
    max_portfolio_exposure: float = 0.25,
    integer_round: bool = False,
) -> float:
    """Calculate capped fractional Wasserstein-Kelly stake using p_DRO = max(0, p - epsilon)."""
    prob_dro = max(0.0, float(probability) - max(0.0, float(wasserstein_epsilon)))
    if prob_dro <= 0.0:
        return 0.0
    stake = fractional_kelly_stake(
        bankroll=bankroll,
        probability=prob_dro,
        decimal_odds=decimal_odds,
        fraction=fraction,
        tax_rate=tax_rate,
        min_stake=min_stake,
        max_stake=max_stake,
        reserved_bankroll=reserved_bankroll,
        max_portfolio_exposure=max_portfolio_exposure,
    )
    if integer_round and stake > 0.0:
        effective_bankroll = max(0.0, float(bankroll) - max(0.0, float(reserved_bankroll)))
        stake = float(min(int(max_stake), max(int(min_stake), round(stake))))
        if stake > effective_bankroll:
            return 0.0
    return stake
