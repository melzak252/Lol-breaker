"""Tests for the Odds Bracket Calibration and Betting Benchmark module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.odds_bracket_benchmark import (
    DEFAULT_ODDS_BRACKETS,
    BracketCalibrationStats,
    OddsBenchmarkReport,
    evaluate_odds_bracket_benchmark,
)


def test_evaluate_odds_bracket_benchmark_synthetic() -> None:
    # 100 observations across different odds brackets
    np.random.seed(42)
    odds = np.array([1.10] * 20 + [1.35] * 20 + [1.65] * 20 + [2.00] * 20 + [4.00] * 20)
    # Calibrated probabilities matching 1/odds approximately
    p_true = 1.0 / odds
    y_true = (np.random.uniform(0, 1, size=len(odds)) < p_true).astype(int)
    p_model = p_true.copy()
    p_novig = p_true.copy()

    df = pd.DataFrame({
        "odds": odds,
        "p_model": p_model,
        "y_true": y_true,
        "p_novig": p_novig,
    })

    report = evaluate_odds_bracket_benchmark(
        df,
        odds_col="odds",
        model_prob_col="p_model",
        target_col="y_true",
        market_prob_col="p_novig",
        dataset_name="Synthetic Test",
    )

    assert isinstance(report, OddsBenchmarkReport)
    assert report.total_lines == 100
    assert 0.0 <= report.overall_brier <= 0.30
    assert 0.0 <= report.overall_log_loss <= 0.80
    assert report.bracket_weighted_ece >= 0.0

    # Verify brackets populated
    bracket_names = [b.bracket_name for b in report.brackets]
    assert "Mega Favorite" in bracket_names
    assert "Solid Favorite" in bracket_names
    assert "Moderate Favorite" in bracket_names
    assert "Coin-Flip / Tight" in bracket_names
    assert "Big Underdog" in bracket_names

    # Check markdown rendering
    md = report.to_markdown()
    assert "# Odds Bracket Calibration & Betting Benchmark: Synthetic Test" in md
    assert "## 1. Calibration and LogLoss by Bookmaker Odds Tier" in md
    assert "## 2. Simulated Value Betting Performance (+EV Bets)" in md
    assert "## 3. Key Calibration Findings & Diagnostics" in md


def test_empty_dataframe_raises_error() -> None:
    empty_df = pd.DataFrame({"odds": [], "p_model": [], "y_true": []})
    with pytest.raises(ValueError, match="Cannot evaluate odds benchmark on empty DataFrame"):
        evaluate_odds_bracket_benchmark(empty_df)


def test_tax_and_roi_calculation() -> None:
    # 2 bets at odds 2.00, both won -> 100% win rate
    # Gross return: (2.00 - 1.0) = +100%
    # Net return (12% tax): (2.00 * 0.88 - 1.0) = +76%
    df = pd.DataFrame({
        "odds": [2.00, 2.00],
        "p_model": [0.70, 0.70],  # p_model * odds = 1.4 > 1.0 (+EV)
        "y_true": [1, 1],
        "p_novig": [0.50, 0.50],
    })

    report = evaluate_odds_bracket_benchmark(df, tax_rate=0.12)
    coin_flip = [b for b in report.brackets if b.bracket_name == "Coin-Flip / Tight"][0]
    assert coin_flip.n_positive_ev_bets == 2
    assert coin_flip.roi_gross_pct == pytest.approx(100.0, abs=0.1)
    assert coin_flip.roi_net_pct == pytest.approx(76.0, abs=0.1)
