"""Unit tests for GameDurationModel, Gamma parameters, and duration Over/Under pricing (IDEA-018)."""

from __future__ import annotations

import math
import pytest

from betting_app.ml.props.duration_model import GameDurationModel
from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker


def test_duration_symmetry_and_swapping() -> None:
    model = GameDurationModel()

    # Team A vs Team B with +150 Elo
    pred_ab = model.predict_duration(
        team_a="T1",
        team_b="Gen.G",
        expected_pace_duration=32.0,
        elo_diff=150.0,
    )

    # Swapped perspective: Team B vs Team A with -150 Elo
    pred_ba = model.predict_duration(
        team_a="Gen.G",
        team_b="T1",
        expected_pace_duration=32.0,
        elo_diff=-150.0,
    )

    # Invariant: Duration is a symmetric property of the match
    assert math.isclose(pred_ab.expected_duration_minutes, pred_ba.expected_duration_minutes, abs_tol=1e-3)
    assert math.isclose(pred_ab.params.shape_k, pred_ba.params.shape_k, abs_tol=1e-3)
    assert math.isclose(pred_ab.params.scale_theta, pred_ba.params.scale_theta, abs_tol=1e-3)

    for line in [28.5, 30.5, 32.5, 34.5]:
        p_ab = pred_ab.get_line(line)
        p_ba = pred_ba.get_line(line)
        assert p_ab is not None and p_ba is not None
        assert math.isclose(p_ab.prob_over, p_ba.prob_over, abs_tol=1e-4)
        assert math.isclose(p_ab.prob_under, p_ba.prob_under, abs_tol=1e-4)


def test_duration_monotonicity_and_probabilities() -> None:
    model = GameDurationModel()
    pred = model.predict_duration(
        team_a="Team A",
        team_b="Team B",
        expected_pace_duration=33.0,
        elo_diff=50.0,
        target_lines=(26.5, 28.5, 30.5, 32.5, 34.5, 36.5),
    )

    sorted_lines = sorted(pred.lines.keys())
    prev_prob_over = 2.0
    for line in sorted_lines:
        line_pred = pred.lines[line]
        # Monotonicity: P(Duration > L) strictly decreases as line L increases
        assert line_pred.prob_over < prev_prob_over
        prev_prob_over = line_pred.prob_over

        # Probabilities on half-line must sum to 1.0
        assert math.isclose(line_pred.prob_over + line_pred.prob_under, 1.0, abs_tol=1e-4)

        # Fair odds must be positive and > 1.0
        assert line_pred.fair_odds_over > 1.0
        assert line_pred.fair_odds_under > 1.0


def test_elo_disparity_shortens_games() -> None:
    model = GameDurationModel(spread_coef=0.010)

    # Equal match: Elo diff 0
    pred_even = model.predict_duration(
        team_a="Even A",
        team_b="Even B",
        expected_pace_duration=32.0,
        elo_diff=0.0,
    )

    # Severe mismatch: Elo diff 300
    pred_stomp = model.predict_duration(
        team_a="Heavy Fav",
        team_b="Heavy Und",
        expected_pace_duration=32.0,
        elo_diff=300.0,
    )

    # Heavy favorite stomps shorten match
    assert pred_stomp.expected_duration_minutes < pred_even.expected_duration_minutes
    # Under probability on 30.5 should be higher for the stomp
    assert pred_stomp.get_line(30.5).prob_under > pred_even.get_line(30.5).prob_under
    # Over probability on 32.5 should be lower for the stomp
    assert pred_stomp.get_line(32.5).prob_over < pred_even.get_line(32.5).prob_over


def test_duration_quantiles_ordering() -> None:
    model = GameDurationModel()
    params = model.estimate_params(expected_pace_duration=33.0, elo_diff=80.0)

    # Quantiles must be strictly increasing
    assert params.p10 < params.p25 < params.median < params.p75 < params.p90

    # Median and mean should be close in modest skewness
    assert math.isclose(params.mean, params.median, abs_tol=1.5)
    assert params.std > 0.0
    assert params.shape_k > 0.0
    assert params.scale_theta > 0.0


def test_duration_market_odds_and_ev() -> None:
    model = GameDurationModel()
    market_lines = {
        31.5: (1.95, 1.80), # Over 31.5 at 1.95, Under 31.5 at 1.80
    }
    pred = model.predict_duration(
        team_a="Team A",
        team_b="Team B",
        expected_pace_duration=32.5,
        elo_diff=20.0,
        target_lines=(31.5,),
        market_lines=market_lines,
        tax_rate=0.12,
    )

    line = pred.get_line(31.5)
    assert line is not None
    assert line.market_odds_over == 1.95
    assert line.market_odds_under == 1.80

    # EV checks
    exp_gross_over = (line.prob_over * 1.95) - 1.0
    exp_net_over = (line.prob_over * 1.95 * 0.88) - 1.0
    assert math.isclose(line.ev_over_gross, exp_gross_over, abs_tol=1e-4)
    assert math.isclose(line.ev_over_net_tax12, exp_net_over, abs_tol=1e-4)
    assert line.ev_over_net_tax12 < line.ev_over_gross


def test_predict_from_tracker_integration() -> None:
    tracker = ChronologicalPaceTracker()
    for _ in range(3):
        tracker.update_game("Slow Macro A", "Opponent 1", kills_1=12, kills_2=10, duration_minutes=37.0, winner_team_1=True)
        tracker.update_game("Slow Macro B", "Opponent 2", kills_1=14, kills_2=11, duration_minutes=36.0, winner_team_1=True)

    model = GameDurationModel()
    pred = model.predict_from_tracker(tracker, "Slow Macro A", "Slow Macro B", league_name="LCK")

    # Both teams play slow macro games (36-37 min), so expected duration should be high (> 33.5 min, shrunk by LCK prior)
    assert pred.expected_duration_minutes > 33.5
    # Over 32.5 should have high probability
    assert pred.get_line(32.5) is not None
    assert pred.get_line(32.5).prob_over > 0.60
