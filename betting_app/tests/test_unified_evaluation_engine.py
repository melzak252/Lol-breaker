"""Observable qualification, historical EV heuristic and simulation contracts."""

import math
import pytest
from src.analysis.unified_evaluation_engine import (
    EvaluatedBet,
    UnifiedBettingEngine,
    calculate_quarter_kelly_stake,
    compute_p_low,
    compute_shrunk_p_low,
)


def test_compute_p_low_invariants() -> None:
    """p_low must always be strictly between 0 and p, monotonically increasing."""
    p_vals = [0.10, 0.25, 0.50, 0.75, 0.90]
    for p in p_vals:
        p_low = compute_p_low(p, kappa=0.75, sigma_z=0.15)
        assert 0.0 < p_low < p
        p_low_high_sigma = compute_p_low(p, kappa=0.75, sigma_z=0.30)
        assert p_low_high_sigma < p_low


def test_compute_shrunk_p_low_invariants() -> None:
    """shrunk_p_low must penalize model-market disagreement and high odds heteroskedasticity."""
    p_shrunk = 0.60
    p_mod = 0.65
    p_mkt = 0.55

    p_low_base = compute_shrunk_p_low(
        prob_shrunk=p_shrunk, prob_model=p_mod, prob_market=p_mkt, odds=1.50
    )
    assert 0.0 < p_low_base < p_shrunk

    # Higher odds must produce a lower (more conservative) lower bound
    p_low_high_odds = compute_shrunk_p_low(
        prob_shrunk=p_shrunk, prob_model=p_mod, prob_market=p_mkt, odds=3.00
    )
    assert p_low_high_odds < p_low_base

    # Higher model-market disagreement must produce a lower bound
    p_low_high_disagreement = compute_shrunk_p_low(
        prob_shrunk=p_shrunk, prob_model=0.75, prob_market=0.45, odds=1.50
    )
    assert p_low_high_disagreement < p_low_base

def test_quarter_kelly_stake_calculation() -> None:
    """1/4 Kelly stake must respect cap, tax, and nonnegativity."""
    bankroll = 1000.0
    stk_neg = calculate_quarter_kelly_stake(
        bankroll=bankroll, prob_for_ev=0.40, odds=2.00, tax_rate=0.12
    )
    assert stk_neg == 0.0

    stk_pos = calculate_quarter_kelly_stake(
        bankroll=bankroll, prob_for_ev=0.65, odds=2.00, tax_rate=0.12, cap_fraction=0.025
    )
    assert 0.0 < stk_pos <= 25.0


def test_historical_ev_heuristic_keeps_existing_coefficients() -> None:
    """The historical adjustment remains arithmetic, not a calibration guarantee."""
    engine = UnifiedBettingEngine(tax_rate=0.0, min_ev_net=0.03)

    # 1. Favorite (p=0.75): low entropy H=0.811 -> minimal haircut
    _, bet_fav = engine.qualify_quote(
        match_id="m1", side="team_a", odds=1.50, prob_model=0.75, prob_conservative=0.74, won=True
    )
    # raw EV = 0.74 * 1.50 - 1 = +11.0%
    h_fav = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    expected_fav = (bet_fav.ev_net * 0.73 / 1.075) * (1.0 - 0.15 * h_fav)
    assert bet_fav.ev_calibrated == pytest.approx(expected_fav, abs=1e-3)

    # 2. Coin-flip Underdog with same raw EV (+11.0%): p_cons=0.444 on odds=2.50
    _, bet_dog = engine.qualify_quote(
        match_id="m2", side="team_a", odds=2.50, prob_model=0.50, prob_conservative=0.444, won=False
    )
    h_dog = 1.00
    expected_dog = (bet_dog.ev_net * 0.73 / (1.0 + 0.15 * 1.5)) * (1.0 - 0.15 * h_dog)
    assert bet_dog.ev_calibrated < bet_fav.ev_calibrated
    assert bet_dog.ev_calibrated == pytest.approx(expected_dog, abs=1e-3)
def test_unified_engine_simulation_calibration() -> None:
    """Simulation must compute exact Expected EV per bet vs Realized return per bet."""
    engine = UnifiedBettingEngine(tax_rate=0.12)

    bets = [
        EvaluatedBet(
            match_id="m1",
            side="team_a",
            bookmaker="STS",
            odds=2.50,
            odds_close=2.30,
            prob_model=0.52,
            prob_conservative=0.50,
            ev_net=0.10,
            ev_calibrated=0.08,
            prob_for_ev=0.50,
            uncertainty_basis="measured_bounds",
            uncertainty_reason="provided_lower_bound",
            won=True,
            stake=100.0,
            pnl=120.0,
            clv_pct=8.7,
        ),
        EvaluatedBet(
            match_id="m2",
            side="team_b",
            bookmaker="STS",
            odds=2.50,
            odds_close=2.40,
            prob_model=0.52,
            prob_conservative=0.50,
            ev_net=0.10,
            ev_calibrated=0.08,
            prob_for_ev=0.50,
            uncertainty_basis="measured_bounds",
            uncertainty_reason="provided_lower_bound",
            won=False,
            stake=100.0,
            pnl=-100.0,
            clv_pct=4.17,
        ),
    ]

    summary = engine.run_simulation(bets, strategy="flat", initial_bankroll=1000.0, flat_stake=100.0)
    assert summary.bets_count == 2
    assert summary.wins_count == 1
    assert summary.win_rate_pct == 50.0
    assert summary.total_staked == 200.0
    assert summary.total_pnl == 20.0
    assert summary.final_bankroll == 1020.0
    assert summary.roi_pct == 10.0
    assert summary.result_status == "computed"


@pytest.mark.parametrize("with_standalone", [False, True])
def test_engine_missing_required_bounds_never_auto_generates_sigma(with_standalone):
    engine = UnifiedBettingEngine()
    eligible, bet = engine.qualify_quote(
        match_id="missing", side="team_a", odds=2.0, prob_model=0.65, won=True,
        prob_standalone_model=0.7 if with_standalone else None,
        prob_market_novig=0.5 if with_standalone else None,
    )

    assert eligible is False
    assert bet.prob_conservative is None
    assert bet.prob_for_ev is None
    assert bet.ev_net is None
    assert bet.ev_calibrated is None
    assert bet.uncertainty_basis == "missing_bounds"
    assert bet.quarantine_reason == "missing_uncertainty"


@pytest.mark.parametrize("with_standalone", [False, True])
def test_engine_auto_haircut_requires_explicit_research_identity(with_standalone):
    engine = UnifiedBettingEngine()
    eligible, bet = engine.qualify_quote(
        match_id="assumed", side="team_a", odds=2.0, prob_model=0.65, won=True,
        prob_standalone_model=0.7 if with_standalone else None,
        prob_market_novig=0.5 if with_standalone else None,
        uncertainty_basis="assumed_research_haircut",
    )
    expected = (
        compute_shrunk_p_low(0.65, 0.7, 0.5, 2.0)
        if with_standalone else compute_p_low(0.65, sigma_z=0.15)
    )

    assert bet.prob_model == 0.65
    assert bet.prob_conservative == pytest.approx(expected)
    assert bet.prob_for_ev == pytest.approx(expected)
    assert bet.uncertainty_basis == "assumed_research_haircut"
    assert bet.uncertainty_reason == "explicit_research_assumption"
    assert bet.ev_net == pytest.approx(round(expected * 2.0 * 0.88 - 1, 4))
    assert eligible is not with_standalone


@pytest.mark.parametrize("basis", ["measured_bounds", "assumed_research_haircut"])
def test_engine_preserves_explicit_bound_without_reapplying_haircut(basis):
    engine = UnifiedBettingEngine()
    eligible, bet = engine.qualify_quote(
        match_id="provided", side="team_a", odds=2.0, prob_model=0.65, won=True,
        prob_conservative=0.62, prob_standalone_model=0.7, prob_market_novig=0.5,
        uncertainty_basis=basis, uncertainty_reason="persisted_side_evidence",
    )

    assert eligible is True
    assert bet.prob_conservative == 0.62
    assert bet.prob_for_ev == 0.62
    assert bet.ev_net == pytest.approx(0.0912)
    assert bet.uncertainty_basis == basis
    assert bet.uncertainty_reason == "persisted_side_evidence"


def test_mean_quote_keeps_stored_probability_without_research_haircut():
    eligible, bet = UnifiedBettingEngine(uncertainty_required=False).qualify_quote(
        match_id="stored_hybrid", side="team_a", odds=2.0,
        prob_model=0.65, won=True, prob_standalone_model=0.7,
        prob_market_novig=0.5,
    )

    assert eligible is True
    assert bet.prob_model == bet.prob_for_ev == 0.65
    assert bet.prob_conservative is None
    assert bet.uncertainty_basis == "mean"
    assert bet.uncertainty_reason == "uncertainty_not_requested"
    assert bet.ev_net == pytest.approx(0.144)


@pytest.mark.parametrize("lower", [float("nan"), -0.1, 0.7])
def test_invalid_bound_rejection_has_no_fabricated_ev(lower):
    eligible, bet = UnifiedBettingEngine().qualify_quote(
        match_id="invalid", side="team_a", odds=2.0,
        prob_model=0.65, won=True, prob_conservative=lower,
    )

    assert eligible is False
    assert bet.quarantine_reason == "invalid_uncertainty"
    assert bet.uncertainty_reason == "invalid_uncertainty"
    assert bet.prob_for_ev is None
    assert bet.ev_net is None
    assert bet.ev_calibrated is None


@pytest.mark.parametrize("strategy", ["flat", "quarter_kelly", "fixed_fraction"])
def test_empty_portfolio_keeps_cash_but_has_no_conditional_performance(strategy):
    summary = UnifiedBettingEngine().run_simulation(
        [], strategy=strategy, initial_bankroll=1234.5,
    )

    assert summary.final_bankroll == summary.initial_bankroll == 1234.5
    assert summary.total_staked == summary.total_pnl == 0.0
    assert summary.max_drawdown_pct == summary.max_drawdown_pln == 0.0
    assert summary.bets_count == summary.wins_count == 0
    for field in (
        "roi_pct", "win_rate_pct", "avg_odds", "avg_stake", "expected_ev_pct",
        "raw_ev_pct", "expected_profit_per_bet", "realized_ev_pct",
        "realized_profit_per_bet", "calibration_gap_pln",
    ):
        assert summary.to_dict()[field] is None
    assert summary.result_status == "no_eligible_bets"
    assert summary.bins == summary.by_odds_bracket == summary.by_bookmaker == []

def test_shin_devigging_favorite_longshot_bias() -> None:
    """Shin (1992) devigging must strip favorite-longshot bias and preserve probability mass."""
    from betting_app.core.ev import fair_market_probabilities
    from src.analysis.shin_devig import shin_implied_probabilities

    # Symmetric match: 1.90 vs 1.90
    pa, pb = fair_market_probabilities(1.90, 1.90)
    assert pa == pytest.approx(0.50, abs=1e-4)
    assert pb == pytest.approx(0.50, abs=1e-4)
    assert (pa + pb) == pytest.approx(1.0, abs=1e-6)

    # Skewed match: 1.30 favorite vs 3.50 underdog
    pa_mult, pb_mult = fair_market_probabilities(1.30, 3.50, method="multiplicative")
    assert (pa_mult + pb_mult) == pytest.approx(1.0, abs=1e-6)
    assert pa_mult == pytest.approx((1.0 / 1.30) / (1.0 / 1.30 + 1.0 / 3.50), abs=1e-6)

    # For a two-outcome overround, Shin equals the additive-margin solution.
    pa_shin, pb_shin = shin_implied_probabilities(1.30, 3.50)
    assert (pa_shin + pb_shin) == pytest.approx(1.0, abs=1e-6)
    assert pa_shin > pa_mult
    assert pa_shin == pytest.approx((1.0 + 1.0 / 1.30 - 1.0 / 3.50) / 2.0)
    # Invalid odds must raise ValueError
    with pytest.raises(ValueError):
        fair_market_probabilities(0.95, 2.10)


