"""Contracts for retrospective base-A0 calibration on production prices."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import bootstrap, production_market_benchmark as production


def quote(match_id="m1", horizon="OPEN", date="2026-05-01", **overrides):
    row = dict(
        golgg_match_id=match_id,
        canonical_match_id=match_id,
        horizon=horizon,
        date=date,
        p_a0=0.2,
        p_market=0.25,
        y=0,
        best_of=3,
        is_common=False,
        time_compatible_strict_prior_day=True,
        best_odds_team1=5.0,
        best_odds_team2=1.8,
        best_bookmaker_team1="book_a",
        best_bookmaker_team2="book_b",
        best_quote_id_team1="quote_a",
        best_quote_id_team2="quote_b",
        best_quote_time_team1=pd.Timestamp(date, tz="UTC"),
        best_quote_time_team2=pd.Timestamp(date, tz="UTC"),
        cutoff=pd.Timestamp(date, tz="UTC"),
        start_dt=pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=12),
    )
    row.update(overrides)
    return row


def test_ratio_bootstrap_preserves_zero_selection_months_without_conditional_ci():
    result = bootstrap.block_ratio_interval(
        numerator=[0.0, 0.1, 0.2], denominator=[0.0, 1.0, 1.0],
        blocks=["2026-01", "2026-02", "2026-03"], draws=5000, seed=20260927,
    )
    assert result["blocks"] == 3
    assert result["active_blocks"] == 2
    assert result["estimate"] == pytest.approx(0.15)
    assert 0 < result["empty_replicates"] < 5000
    assert result["valid_replicates"] + result["empty_replicates"] == 5000
    assert result["ci_lower"] is None and result["ci_upper"] is None
    assert result["status"] == "UNDEFINED_REPLICATES"


def test_ratio_bootstrap_constant_error_is_exact_and_one_month_is_not_a_ci():
    result = bootstrap.block_ratio_interval(
        numerator=[0.02, 0.04, 0.06], denominator=[1, 2, 3],
        blocks=["2026-01", "2026-02", "2026-03"], draws=5000, seed=20260927,
    )
    assert result["estimate"] == pytest.approx(0.02)
    assert result["ci_lower"] == pytest.approx(0.02)
    assert result["ci_upper"] == pytest.approx(0.02)
    assert result["valid_replicates"] == 5000
    single = bootstrap.block_ratio_interval([0.02], [1], ["2026-01"])
    assert single["status"] == "INSUFFICIENT_BLOCKS"
    assert single["ci_lower"] is None and single["ci_upper"] is None


def test_ratio_bootstrap_rejects_nonfinite_or_negative_exposure():
    with pytest.raises(ValueError):
        bootstrap.block_ratio_interval([np.nan, 0], [1, 1], ["a", "b"])
    with pytest.raises(ValueError):
        bootstrap.block_ratio_interval([0, 0], [-1, 1], ["a", "b"])


def test_base_a0_selects_correct_b_side_without_hidden_uncertainty_or_ev_cap():
    bets = production.select_a0_ev_bets(pd.DataFrame([quote()]), tax_rate=0.12, min_ev=0.05)
    assert len(bets) == 1
    bet = bets.iloc[0]
    assert bet.side == "B"
    assert bet.probability == pytest.approx(0.8)
    assert bet.market_probability == pytest.approx(0.75)
    assert bet.expected_ev == pytest.approx(0.2672)
    assert bet.unit_return == pytest.approx(0.584)
    assert bet.bookmaker == "book_b" and bet.quote_id == "quote_b"
    assert bool(bet.won)


def test_ev_threshold_is_strict_and_duplicate_horizon_fixtures_are_rejected():
    frame = pd.DataFrame([quote(p_a0=0.625, best_odds_team1=2.0, best_odds_team2=2.0)])
    assert production.select_a0_ev_bets(frame, tax_rate=0, min_ev=0.25).empty
    with pytest.raises(ValueError, match="duplicate"):
        production.select_a0_ev_bets(pd.concat([frame, frame], ignore_index=True))


def test_future_quote_is_rejected_instead_of_silently_scored():
    frame = pd.DataFrame([quote(best_quote_time_team1=pd.Timestamp("2026-05-02", tz="UTC"))])
    with pytest.raises(ValueError, match="quote"):
        production.select_a0_ev_bets(frame)


def test_common_sample_is_four_separate_horizon_portfolios():
    frame = pd.DataFrame([quote(horizon=h, is_common=True) for h in production.HORIZONS])
    summaries, bets, profiles = production.evaluate_a0_calibration(frame, draws=100)
    common = [row for row in summaries if row["cohort"] == "common"]
    assert {row["horizon"] for row in common} == set(production.HORIZONS)
    assert all(row["n_matches"] == 1 and row["bets"] == 1 for row in common)
    assert not bets.duplicated(["cohort", "horizon", "golgg_match_id"]).any()
    assert len(bets[bets.cohort == "common"]) == 4


def test_calibration_compares_yield_and_win_rate_with_actual_base_probabilities():
    frame = pd.DataFrame([
        quote("a", date="2026-05-01", p_a0=0.6, y=1, best_odds_team1=2.0, best_odds_team2=2.0),
        quote("b", date="2026-06-01", p_a0=0.6, y=0, best_odds_team1=2.0, best_odds_team2=2.0),
    ])
    summaries, bets, profiles = production.evaluate_a0_calibration(frame, tax_rate=0, draws=100)
    summary = next(row for row in summaries if row["cohort"] == "all" and row["horizon"] == "OPEN")
    assert summary["bets"] == 2 and summary["wins"] == 1
    assert summary["mean_probability"] == pytest.approx(0.6)
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["mean_ev"] == pytest.approx(0.2)
    assert summary["yield"] == pytest.approx(0)
    assert summary["yield_gap"] == pytest.approx(-0.2)
    assert summary["win_rate_gap"] == pytest.approx(-0.1)
    assert summary["yield_gap_interval"]["valid_replicates"] == 100
    assert bets[bets.cohort == "all"].pnl.sum() == pytest.approx(0)


def test_filtered_bets_retain_their_original_month_not_their_new_row_number():
    frame = pd.DataFrame([
        quote("skip_may", date="2026-05-01", p_a0=0.5, best_odds_team1=1.9, best_odds_team2=1.9),
        quote("win_may", date="2026-05-02", p_a0=0.6, y=1, best_odds_team1=2.0, best_odds_team2=2.0),
        quote("skip_june", date="2026-06-01", p_a0=0.5, best_odds_team1=1.9, best_odds_team2=1.9),
        quote("lose_june", date="2026-06-02", p_a0=0.6, y=0, best_odds_team1=2.0, best_odds_team2=2.0),
    ])
    summaries, _, _ = production.evaluate_a0_calibration(frame, tax_rate=0, draws=5000)
    summary = next(row for row in summaries if row["cohort"] == "all" and row["horizon"] == "OPEN")
    interval = summary["yield_gap_interval"]
    assert interval["active_blocks"] == 2
    assert interval["estimate"] == pytest.approx(summary["yield_gap"])
    assert interval["ci_lower"] == pytest.approx(-1.2)
    assert interval["ci_upper"] == pytest.approx(0.8)


def test_swapping_teams_preserves_selected_outcome_and_profit():
    original = quote()
    swapped = quote(p_a0=0.8, p_market=0.75, y=1)
    for field in ("odds", "bookmaker", "quote_id", "quote_time"):
        swapped[f"best_{field}_team1"] = original[f"best_{field}_team2"]
        swapped[f"best_{field}_team2"] = original[f"best_{field}_team1"]
    first = production.select_a0_ev_bets(pd.DataFrame([original])).iloc[0]
    second = production.select_a0_ev_bets(pd.DataFrame([swapped])).iloc[0]
    assert first.side == "B" and second.side == "A"
    assert first.probability == pytest.approx(second.probability)
    assert first.pnl == pytest.approx(second.pnl)
    assert first.bookmaker == second.bookmaker and first.quote_id == second.quote_id
