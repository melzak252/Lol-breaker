"""Unit tests for TeamKillsSpreadModel, discrete convolution, and handicap pricing (IDEA-018)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.ml.props.schemas import MatchupSpreadFeatures
from betting_app.ml.props.team_spread_model import TeamKillsSpreadModel
from src.ratings.elo import EloRating


def test_spread_symmetry_and_swapping() -> None:
    model = TeamKillsSpreadModel()

    # Features for Team A vs Team B
    features_ab = MatchupSpreadFeatures(
        team_a="T1",
        team_b="Gen.G",
        rating_a=1650.0,
        rating_b=1550.0,
        elo_diff=100.0,
        abs_elo_diff=100.0,
        prob_win_a=0.6401,
        prob_win_b=0.3599,
        exp_pace_kills_a=14.0,
        exp_pace_kills_b=13.0,
        expected_total_kills=27.0,
        expected_duration_minutes=32.0,
    )

    # Features for Team B vs Team A (swapped perspective)
    features_ba = MatchupSpreadFeatures(
        team_a="Gen.G",
        team_b="T1",
        rating_a=1550.0,
        rating_b=1650.0,
        elo_diff=-100.0,
        abs_elo_diff=100.0,
        prob_win_a=0.3599,
        prob_win_b=0.6401,
        exp_pace_kills_a=13.0,
        exp_pace_kills_b=14.0,
        expected_total_kills=27.0,
        expected_duration_minutes=32.0,
    )

    pred_ab = model.predict_matchup(features_ab)
    pred_ba = model.predict_matchup(features_ba)

    # Invariant: mu_A in (A, B) must equal mu_B in (B, A)
    assert math.isclose(pred_ab.mu_team_a, pred_ba.mu_team_b, abs_tol=1e-3)
    assert math.isclose(pred_ab.mu_team_b, pred_ba.mu_team_a, abs_tol=1e-3)
    assert math.isclose(pred_ab.mu_total, pred_ba.mu_total, abs_tol=1e-3)

    # Handicap symmetry:
    # Team A covering -5.5 in AB should equal Team B covering -5.5 in BA
    h_ab = pred_ab.get_handicap_line(-5.5)
    h_ba = pred_ba.get_handicap_line(5.5)
    assert h_ab is not None and h_ba is not None
    # P(K_A - K_B > -5.5) == P(K_B - K_A < 5.5) == 1 - P(K_B - K_A > 5.5)
    assert math.isclose(h_ab.prob_cover_a, 1.0 - h_ba.prob_cover_a, abs_tol=1e-3)


def test_convolution_mass_and_normalization() -> None:
    model = TeamKillsSpreadModel(alpha=0.10)
    diffs, pmf = model.compute_difference_pmf(mu_a=16.5, mu_b=12.5, max_kills=60)

    # Exact range check
    assert len(diffs) == 121
    assert diffs[0] == -60
    assert diffs[-1] == 60

    # Total probability mass across [-60, 60] should be >= 0.9999
    total_mass = float(np.sum(pmf))
    assert math.isclose(total_mass, 1.0, abs_tol=1e-3)


def test_elo_spread_amplification() -> None:
    model = TeamKillsSpreadModel(spread_coef=0.0005)

    # Case 1: Equal teams
    mu_even_a, mu_even_b = model.estimate_team_mus(exp_pace_a=14.0, exp_pace_b=14.0, elo_diff=0.0)
    assert math.isclose(mu_even_a, mu_even_b, abs_tol=1e-3)

    # Case 2: Favorite with +200 Elo advantage
    mu_fav_a, mu_fav_b = model.estimate_team_mus(exp_pace_a=14.0, exp_pace_b=14.0, elo_diff=200.0)
    assert mu_fav_a > mu_fav_b
    assert mu_fav_a > mu_even_a
    assert mu_fav_b < mu_even_b

    # With +200 Elo, favorite expected kills should increase by ~10% (exp(0.0005 * 200) = exp(0.10) ~ 1.105)
    ratio = mu_fav_a / mu_even_a
    assert math.isclose(ratio, math.exp(0.10), rel_tol=1e-2)


def test_tracker_integrated_flow() -> None:
    tracker = ChronologicalPaceTracker()
    # Simulate a dominant team winning 5 games with high kills
    for _ in range(5):
        tracker.update_game(
            team_1="Dominant Team",
            team_2="Struggling Team",
            kills_1=22,
            kills_2=8,
            duration_minutes=28.0,
            winner_team_1=True,
        )

    # Check that Elo updated
    features = tracker.get_matchup_spread_features("Dominant Team", "Struggling Team", league_name="LCK")
    assert features.rating_a > 1500.0
    assert features.rating_b < 1500.0
    assert features.elo_diff > 0.0
    assert features.prob_win_a > 0.50

    model = TeamKillsSpreadModel()
    pred = model.predict_from_tracker(tracker, "Dominant Team", "Struggling Team", league_name="LCK")

    assert pred.mu_team_a > pred.mu_team_b
    # Handicap -5.5 for Dominant Team should have high cover probability
    h_minus_5 = pred.get_handicap_line(-5.5)
    assert h_minus_5 is not None
    assert h_minus_5.prob_cover_a > 0.60

def test_handicap_strict_monotonicity_and_complementarity() -> None:
    model = TeamKillsSpreadModel()
    features = MatchupSpreadFeatures(
        team_a="Favorite",
        team_b="Underdog",
        rating_a=1700.0,
        rating_b=1400.0,
        elo_diff=300.0,
        abs_elo_diff=300.0,
        prob_win_a=0.849,
        prob_win_b=0.151,
        exp_pace_kills_a=15.0,
        exp_pace_kills_b=11.0,
        expected_total_kills=26.0,
        expected_duration_minutes=30.0,
    )
    pred = model.predict_matchup(features, target_handicaps=(-10.5, -8.5, -6.5, -4.5, -2.5, 2.5, 4.5, 6.5, 8.5, 10.5))

    sorted_lines = sorted(pred.handicap_lines.keys())
    prev_prob = 2.0
    for line in sorted_lines:
        hp = pred.handicap_lines[line]
        assert hp is not None
        # Monotonicity: P(K_A - K_B > H) decreases as H increases
        assert hp.prob_cover_a < prev_prob
        prev_prob = hp.prob_cover_a
        assert math.isclose(hp.prob_cover_a + hp.prob_cover_b, 1.0, abs_tol=1e-5)


def test_handicap_ev_and_tax_calculation() -> None:
    model = TeamKillsSpreadModel()
    diffs, pmf = model.compute_difference_pmf(mu_a=16.0, mu_b=12.0)

    # Test line with market odds
    # Say Team A -4.5 has market odds of 2.10 on A and 1.75 on B
    market_lines = {-4.5: (2.10, 1.75)}
    hp = model.compute_handicap_line(
        diff_values=diffs,
        diff_pmf=pmf,
        handicap_line=-4.5,
        market_odds_a=2.10,
        market_odds_b=1.75,
        tax_rate=0.12,
    )


    assert hp.market_odds_a == 2.10
    assert hp.market_odds_b == 1.75
    assert hp.fair_odds_a > 1.0
    assert hp.fair_odds_b > 1.0

    # Expected value formulas
    expected_gross_a = (hp.prob_cover_a * 2.10) - 1.0
    expected_net_a = (hp.prob_cover_a * 2.10 * 0.88) - 1.0
    assert math.isclose(hp.ev_a_gross, expected_gross_a, abs_tol=1e-4)
    assert math.isclose(hp.ev_a_net_tax12, expected_net_a, abs_tol=1e-4)
    # Net EV with 12% tax must be strictly less than Gross EV
    assert hp.ev_a_net_tax12 < hp.ev_a_gross


def test_team_total_kills_lines_and_probabilities() -> None:
    model = TeamKillsSpreadModel()
    lines = [8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5]
    predictions = {
        line: model.compute_team_total_line(mu=15.0, line=line, tax_rate=0.12)
        for line in lines
    }
    prev_prob_over = 2.0
    for line in lines:
        p = predictions[line]
        # Monotonicity: P(Over L) decreases as L increases
        assert p.prob_over < prev_prob_over
        prev_prob_over = p.prob_over
        # Total probability on half-line must equal 1.0
        assert math.isclose(p.prob_over + p.prob_under, 1.0, abs_tol=1e-5)
        # Fair odds > 1.0
        assert p.fair_odds_over > 1.0
        assert p.fair_odds_under > 1.0
