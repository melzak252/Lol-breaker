"""Unit tests for standalone financial and betting benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.financial_benchmark import (
    calculate_quarter_kelly,
    compute_financial_slices,
    run_financial_benchmark,
)


def test_calculate_quarter_kelly_with_tax() -> None:
    # Odds = 2.00, Tax = 12% -> Net decimal = 2.00 * 0.88 = 1.76 -> b = 0.76
    # Probability = 0.65 -> EV = 0.65 * 1.76 - 1.0 = 0.144 (14.4%)
    # Full Kelly: f* = 0.144 / 0.76 = 0.18947
    # Quarter Kelly: 0.25 * 0.18947 = 0.047368
    # With cap = 0.02, stake fraction should be clipped to 0.02
    stake_capped = calculate_quarter_kelly(
        probability=0.65,
        odds=2.00,
        tax_rate=0.12,
        bankroll=10_000.0,
        fraction=0.25,
        max_cap_pct=0.02,
    )
    assert stake_capped == pytest.approx(200.0)

    # Without cap (cap = 0.10), stake should be 10000 * 0.047368 = 473.68
    stake_uncapped = calculate_quarter_kelly(
        probability=0.65,
        odds=2.00,
        tax_rate=0.12,
        bankroll=10_000.0,
        fraction=0.25,
        max_cap_pct=0.10,
    )
    assert stake_uncapped == pytest.approx(473.6842, abs=0.01)

    # Negative EV should yield 0 stake
    stake_neg_ev = calculate_quarter_kelly(
        probability=0.50,
        odds=2.00,
        tax_rate=0.12,
        bankroll=10_000.0,
    )
    assert stake_neg_ev == 0.0


def test_run_financial_benchmark_simulation() -> None:
    # 10 synthetic matches
    # Odds: team A = 2.00, team B = 2.00
    # Effective payout = 2.00 * 0.88 = 1.76
    # If prob_a = 0.70 -> EV_a = 0.70 * 1.76 - 1 = +0.232 (qualifies)
    # Win rate = 7 out of 10 wins
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "prob_a": [0.70] * 10,
            "y_true": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 7 wins, 3 losses
            "odds_a": [2.00] * 10,
            "odds_b": [2.00] * 10,
        }
    )

    report = run_financial_benchmark(
        data=df,
        candidate_prob_col="prob_a",
        target_col="y_true",
        odds_a_col="odds_a",
        odds_b_col="odds_b",
        tax_rate=0.12,
        min_ev=0.03,
        initial_bankroll=10_000.0,
        staking_strategy="flat_100",
        model_name="Test-Model",
        model_version="v1.0",
    )

    s = report.summary
    assert s.total_matches_evaluated == 10
    assert s.bets_count == 10
    assert s.wins_count == 7
    assert s.losses_count == 3
    assert s.win_rate_pct == 70.0
    assert s.total_staked == 1000.0

    # PnL:
    # 7 wins: 7 * (100 * (1.76 - 1)) = 7 * 76.0 = 532.0 PLN
    # 3 losses: 3 * (-100) = -300.0 PLN
    # Total profit: 532.0 - 300.0 = 232.0 PLN
    assert s.total_profit == pytest.approx(232.0)
    assert s.yield_pct == pytest.approx(23.2)  # 232 / 1000 * 100%
    assert s.roi_pct == pytest.approx(2.32)    # 232 / 10000 * 100%
    assert s.expected_yield_pct == pytest.approx(23.2)  # EV = +23.2%
    assert s.final_bankroll == pytest.approx(10_232.0)

    # Markdown format check
    md = report.format_markdown()
    assert "Raport Finansowy i Benchmark Bukmacherski" in md
    assert "23.20%" in md
    assert "10,232.00 PLN" in md


def test_run_financial_benchmark_ev_filter() -> None:
    # Matches with EV below min_ev should NOT be bet
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "prob_a": [0.52, 0.50],  # With odds 2.00 and tax 12%, EV = 0.52 * 1.76 - 1 = -0.0848
            "y_true": [1, 0],
            "odds_a": [2.00, 2.00],
            "odds_b": [2.00, 2.00],
        }
    )

    report = run_financial_benchmark(
        data=df,
        candidate_prob_col="prob_a",
        target_col="y_true",
        odds_a_col="odds_a",
        odds_b_col="odds_b",
        tax_rate=0.12,
        min_ev=0.03,
    )

    assert report.summary.bets_count == 0
    assert report.summary.total_staked == 0.0
    assert report.summary.total_profit == 0.0
    assert report.summary.yield_pct == 0.0


def test_run_financial_benchmark_quarter_kelly() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "prob_a": [0.70] * 5,
            "y_true": [1, 1, 1, 1, 1],
            "odds_a": [2.00] * 5,
            "odds_b": [2.00] * 5,
        }
    )
    report = run_financial_benchmark(
        data=df,
        candidate_prob_col="prob_a",
        target_col="y_true",
        odds_a_col="odds_a",
        odds_b_col="odds_b",
        staking_strategy="quarter_kelly",
        initial_bankroll=10_000.0,
    )
    s = report.summary
    assert s.bets_count == 5
    assert s.wins_count == 5
    assert s.staking_strategy == "quarter_kelly"
    assert s.avg_stake > 0.0
    assert s.total_profit > 0.0
    assert s.final_bankroll > 10_000.0


def test_run_financial_benchmark_clv() -> None:
    # Placed odds = 2.20, Closing odds = 2.00 -> CLV = (2.20 - 2.00) / 2.00 = +10%
    # Second bet: Placed odds = 1.90, Closing odds = 2.00 -> CLV = (1.90 - 2.00) / 2.00 = -5%
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "prob_a": [0.70, 0.70],
            "y_true": [1, 1],
            "open_odds_a": [2.20, 1.90],
            "open_odds_b": [1.60, 1.80],
            "close_odds_a": [2.00, 2.00],
            "close_odds_b": [1.80, 1.80],
        }
    )
    report = run_financial_benchmark(
        data=df,
        candidate_prob_col="prob_a",
        target_col="y_true",
        odds_a_col="open_odds_a",
        odds_b_col="open_odds_b",
        close_odds_a_col="close_odds_a",
        close_odds_b_col="close_odds_b",
        time_horizon="opening (>12h)",
        min_ev=0.03,
    )
    s = report.summary
    assert s.bets_count == 2
    assert s.avg_clv_pct is not None
    # (10.0% + (-5.0%)) / 2 = 2.5%
    assert s.avg_clv_pct == pytest.approx(2.5, abs=0.01)
    # 1 out of 2 beat closing line -> 50%
    assert s.beat_closing_rate_pct == pytest.approx(50.0)
    assert s.time_horizon == "opening (>12h)"

    md = report.format_markdown()
    assert "Closing Line Value" in md
    assert "+2.50%" in md
    assert "50.0%" in md
    assert "OSTRY MODEL (CLV > 0)" in md


def test_run_financial_benchmark_odds_calibration() -> None:
    # 4 matches: 2 favorites (odds 1.50) and 2 underdogs (odds 2.50)
    # Favorites: model prob = 0.80, wins = 2 (100%), EV = 0.80 * 1.50 * 0.88 - 1 = +0.056 (>= 0.05)
    # Implied prob = 1/1.50 = 66.7%, Win rate = 100%, Model prob = 80% -> gap = 80% - 100% = -20%
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "prob_a": [0.80, 0.80, 0.50, 0.50],
            "y_true": [1, 1, 0, 1],
            "odds_a": [1.50, 1.50, 2.50, 2.50],
            "odds_b": [2.60, 2.60, 1.50, 1.50],
        }
    )
    report = run_financial_benchmark(
        data=df,
        candidate_prob_col="prob_a",
        target_col="y_true",
        odds_a_col="odds_a",
        odds_b_col="odds_b",
        # Default min_ev is 0.05 -> underdogs (EV = 0.50 * 2.50 * 0.88 - 1 = +0.10) also qualify
    )
    assert report.summary.min_ev == 0.05
    assert report.summary.bets_count == 4

    odds_slices = report.slices.get("Przedział Kursowy", [])
    assert len(odds_slices) >= 2
    fav_slice = next((s for s in odds_slices if "1.40 - 1.80" in s.bucket), None)
    assert fav_slice is not None
    assert fav_slice.bets_count == 2
    assert fav_slice.avg_prob_pct == pytest.approx(80.0)
    assert fav_slice.win_rate_pct == pytest.approx(100.0)
    assert fav_slice.calibration_gap_pct == pytest.approx(-20.0)
    assert fav_slice.implied_prob_pct == pytest.approx(100.0 / 1.50, abs=0.1)
    assert fav_slice.brier_score is not None

    md = report.format_markdown()
    assert "Rentowność i Kalibracja Szans" in md
    assert "Gap Kalibracji" in md
    assert "Implikowane (1/kurs)" in md
