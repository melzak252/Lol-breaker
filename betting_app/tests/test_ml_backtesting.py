from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from betting_app.ml.backtesting import HistoricalPrediction, MatchLabel, OddsQuote, compare_predictions_to_market, run_backtest
from betting_app.ml.backtesting.odds_selection import select_quotes_for_match
from betting_app.ml.config import BacktestConfig, StakingConfig
from betting_app.ml.metrics import binary_auc, brier_score, expected_calibration_error, max_drawdown, roi


UTC = timezone.utc


def dt(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, tzinfo=UTC)


def test_latest_pre_match_selects_latest_quote_per_bookmaker() -> None:
    label = MatchLabel(canonical_match_id=1, winner_side="a", start_time=dt(18))
    quotes = [
        OddsQuote(1, 10, "book-a", 2.10, 1.70, dt(10), 101),
        OddsQuote(1, 10, "book-a", 2.20, 1.65, dt(12), 102),
        OddsQuote(1, 11, "book-b", 2.05, 1.75, dt(11), 201),
        OddsQuote(1, 12, "invalid", 1.00, 14.0, dt(11), 301),
        OddsQuote(1, 10, "too-late", 2.50, 1.50, dt(19), 103),
    ]

    selected = select_quotes_for_match(quotes, label, BacktestConfig())

    assert {q.odds_snapshot_id for q in selected} == {102, 201}


def test_open_and_mid_pre_match_selects_expected_snapshots() -> None:
    label = MatchLabel(canonical_match_id=1, winner_side="a", start_time=dt(18))
    quotes = [
        OddsQuote(1, 10, "book-a", 2.10, 1.70, dt(10), 101),
        OddsQuote(1, 10, "book-a", 2.15, 1.68, dt(12), 102),
        OddsQuote(1, 10, "book-a", 2.20, 1.65, dt(14), 103),
        OddsQuote(1, 11, "book-b", 2.05, 1.75, dt(11), 201),
        OddsQuote(1, 11, "book-b", 2.00, 1.80, dt(15), 202),
    ]

    open_selected = select_quotes_for_match(
        quotes, label, BacktestConfig(odds_policy="open_pre_match")
    )
    assert {q.odds_snapshot_id for q in open_selected} == {101, 201}

    mid_selected = select_quotes_for_match(
        quotes, label, BacktestConfig(odds_policy="mid_pre_match")
    )
    # For book-a (3 snapshots, idx (3-1)//2 = 1): 102; for book-b (2 snapshots, idx (2-1)//2 = 0): 201
    assert {q.odds_snapshot_id for q in mid_selected} == {102, 201}

def test_backtest_places_highest_ev_bet_and_settles_with_tax() -> None:
    result = run_backtest(
        predictions=[
            HistoricalPrediction(
                canonical_match_id=1,
                model_name="test",
                model_version="v1",
                prob_a=0.60,
                prob_b=0.40,
                predicted_at=dt(10),
                data_cutoff_at=dt(9),
            )
        ],
        labels=[MatchLabel(canonical_match_id=1, winner_side="a", start_time=dt(18))],
        odds_quotes=[OddsQuote(1, 10, "book-a", 2.20, 1.70, dt(12), 102)],
        config=BacktestConfig(
            bankroll_start=1_000.0,
            min_ev=0.0,
            staking=StakingConfig(strategy="fixed", fixed_stake=10.0),
        ),
    )

    assert len(result.bets) == 1
    bet = result.bets[0]
    assert bet.side == "a"
    assert bet.result == "won"
    assert bet.stake == 10.0
    assert bet.profit == pytest.approx(10.0 * (2.20 * 0.88 - 1.0))
    assert result.bankroll_end == pytest.approx(1_000.0 + bet.profit)
    assert result.roi == pytest.approx(bet.profit / bet.stake)


def test_backtest_can_require_odds_after_prediction_time() -> None:
    result = run_backtest(
        predictions=[HistoricalPrediction(1, "test", "v1", prob_a=0.60, prob_b=0.40, predicted_at=dt(10), data_cutoff_at=dt(9))],
        labels=[MatchLabel(1, "a", start_time=dt(18))],
        odds_quotes=[
            OddsQuote(1, 10, "book-a", 2.50, 1.50, dt(9), 101),
            OddsQuote(1, 10, "book-a", 2.10, 1.70, dt(11), 102),
        ],
        config=BacktestConfig(
            bankroll_start=1_000.0,
            min_ev=0.0,
            require_prediction_before_odds=True,
            staking=StakingConfig(strategy="fixed", fixed_stake=10.0),
        ),
    )

    assert len(result.bets) == 1
    assert result.bets[0].odds_snapshot_id == 102

def test_metrics_helpers() -> None:
    assert brier_score([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)
    assert binary_auc([0, 1, 1], [0.2, 0.7, 0.8]) == pytest.approx(1.0)
    assert expected_calibration_error([1, 0], [0.75, 0.25]) == pytest.approx(0.25)
    assert roi(12.0, 100.0) == pytest.approx(0.12)
    assert max_drawdown([100.0, 120.0, 90.0, 130.0]) == pytest.approx(30.0)


def test_compare_predictions_to_market_returns_probability_metrics() -> None:
    comparison = compare_predictions_to_market(
        predictions=[HistoricalPrediction(1, "test", "v1", prob_a=0.70, prob_b=0.30, predicted_at=dt(10), data_cutoff_at=dt(9))],
        labels=[MatchLabel(1, "a", start_time=dt(18))],
        odds_quotes=[OddsQuote(1, 10, "book-a", 2.20, 1.70, dt(12), 102)],
        config=BacktestConfig(),
    )

    assert comparison.observations == 1
    assert comparison.model_brier == pytest.approx(0.09)
    assert comparison.model_ece == pytest.approx(0.3)
    assert comparison.model_log_loss < comparison.market_log_loss


def test_market_comparison_excludes_post_start_predictions_and_pre_prediction_quotes() -> None:
    comparison = compare_predictions_to_market(
        predictions=[
            HistoricalPrediction(1, "test", "v1", 0.70, 0.30, predicted_at=dt(10), data_cutoff_at=dt(9)),
            HistoricalPrediction(1, "test", "v1", 0.80, 0.20, predicted_at=dt(19), data_cutoff_at=dt(9)),
        ],
        labels=[MatchLabel(1, "a", start_time=dt(18))],
        odds_quotes=[
            OddsQuote(1, 10, "book-a", 2.20, 1.70, dt(9), 101),
            OddsQuote(1, 10, "book-a", 2.10, 1.75, dt(12), 102),
        ],
    )

    assert comparison.observations == 1
    assert comparison.eligible_predictions == 1
    assert comparison.prediction_exclusions == {"prediction_not_before_match_start": 1}
    assert comparison.quote_exclusions == {"quote_before_prediction": 1}
