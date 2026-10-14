"""Unit tests for in-game prop KillDistributionModel and ChronologicalPaceTracker (IDEA-018)."""

from __future__ import annotations

import math
import pytest

from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker, normalize_league_key
from betting_app.ml.props.kill_distribution_model import KillDistributionModel
from betting_app.ml.props.schemas import KillDistributionPrediction


def test_normalize_league_key() -> None:
    assert normalize_league_key("LCK Spring 2026") == "LCK"
    assert normalize_league_key("LCK Challengers League") == "LCK CL"
    assert normalize_league_key("LPL Summer 2025") == "LPL"
    assert normalize_league_key("LEC Winter 2026") == "LEC"
    assert normalize_league_key("VCS Dawn 2025") == "VCS"
    assert normalize_league_key("Unknown Polish Cup") == "DEFAULT"
    assert normalize_league_key(None) == "DEFAULT"


def test_distribution_moments_and_parameters() -> None:
    model = KillDistributionModel(intercept=2.5, pace_coef=0.03, alpha=0.08)
    params = model.estimate_params(expected_pace_kills=30.0)

    # Check that mu matches exponential link
    expected_mu = math.exp(2.5 + 0.03 * 30.0)
    assert math.isclose(params.mu, expected_mu, rel_tol=1e-2)

    # Check that variance = mu + alpha * mu^2
    expected_var = expected_mu + 0.08 * (expected_mu**2)
    assert math.isclose(params.variance, expected_var, rel_tol=1e-2)
    assert math.isclose(params.std, math.sqrt(expected_var), rel_tol=1e-2)

    # Check Negative Binomial overdispersion
    assert params.variance > params.mu


def test_half_line_complementarity_and_monotonicity() -> None:
    model = KillDistributionModel()
    params = model.estimate_params(expected_pace_kills=28.0)

    lines = [20.5, 24.5, 28.5, 32.5, 36.5]
    prev_prob_over = 1.0

    for line in lines:
        pred = model.compute_line_probabilities(params, line)

        # Complementarity: P(Over) + P(Under) == 1.0
        assert math.isclose(pred.prob_over + pred.prob_under, 1.0, abs_tol=1e-4)

        # Fair odds == 1 / prob
        assert math.isclose(pred.fair_odds_over, 1.0 / pred.prob_over, rel_tol=1e-3)
        assert math.isclose(pred.fair_odds_under, 1.0 / pred.prob_under, rel_tol=1e-3)

        # Monotonicity: P(Over L) strictly decreases as line L increases
        assert pred.prob_over < prev_prob_over
        prev_prob_over = pred.prob_over


def test_density_curve_mass_and_coverage() -> None:
    model = KillDistributionModel()
    params = model.estimate_params(expected_pace_kills=29.0)
    curve = model.generate_density_curve(params, min_kills=5, max_kills=70)

    # Check non-empty curve with valid probabilities
    assert len(curve) == 66
    total_mass = sum(pt["prob"] for pt in curve)

    # For mean ~29 and alpha ~0.08, the range [5, 70] must contain > 99% of total probability mass
    assert total_mass > 0.99
    assert total_mass <= 1.0001


def test_team_order_symmetry() -> None:
    tracker = ChronologicalPaceTracker()
    # Seed teams with asymmetric stats
    for _ in range(5):
        tracker.update_post_game("Team Aggressive", kills=22, deaths=18, duration_minutes=30.0)
        tracker.update_post_game("Team Passive", kills=8, deaths=10, duration_minutes=35.0)

    model = KillDistributionModel()

    # Prediction A vs B
    pred_ab = model.predict_from_tracker(tracker, "Team Aggressive", "Team Passive", league="LCK")
    # Prediction B vs A
    pred_ba = model.predict_from_tracker(tracker, "Team Passive", "Team Aggressive", league="LCK")

    # Invariant: Total match kills distribution must be strictly symmetric
    assert math.isclose(pred_ab.params.mu, pred_ba.params.mu, abs_tol=1e-4)
    assert math.isclose(pred_ab.params.variance, pred_ba.params.variance, abs_tol=1e-4)

    for line in (24.5, 26.5, 28.5, 30.5):
        line_ab = pred_ab.get_line(line)
        line_ba = pred_ba.get_line(line)
        assert line_ab is not None and line_ba is not None
        assert math.isclose(line_ab.prob_over, line_ba.prob_over, abs_tol=1e-4)
        assert math.isclose(line_ab.prob_under, line_ba.prob_under, abs_tol=1e-4)


def test_bayesian_shrinkage_on_small_sample() -> None:
    tracker = ChronologicalPaceTracker(prior_weight=5.0)

    # 0 games: exact league prior for LCK (mean_kills = 25.73 -> 12.865 per team)
    pace_zero = tracker.get_team_pace("New Team", league_name="LCK")
    assert pace_zero.sample_games == 0
    assert math.isclose(pace_zero.avg_kills, 12.86, abs_tol=1e-2)

    # 1 game with an extreme outlier (40 kills)
    tracker.update_post_game("New Team", kills=40, deaths=10, duration_minutes=30.0)
    pace_one = tracker.get_team_pace("New Team", league_name="LCK")
    assert pace_one.sample_games == 1

    # Shrinkage: (1 * 40 + 5 * 12.865) / 6 = (40 + 64.325) / 6 = 17.3875
    # The average should be shrunk significantly below 40 towards ~17.4!
    assert pace_one.avg_kills < 20.0
    assert pace_one.avg_kills > 15.0


def test_ev_and_tax_calculation() -> None:
    model = KillDistributionModel()
    params = model.estimate_params(expected_pace_kills=30.0)

    # Suppose line is 26.5, market offers 2.00 on Over and 1.80 on Under
    pred = model.compute_line_probabilities(
        params=params,
        line=26.5,
        market_odds_over=2.00,
        market_odds_under=1.80,
        tax_rate=0.12,
    )

    # Model prob_over should be > 0.5 because expected mean is ~30 > 26.5
    assert pred.prob_over > 0.5
    assert pred.ev_over_gross is not None
    assert pred.ev_over_net_tax12 is not None

    # Net EV with 12% tax: prob * (2.00 * 0.88) - 1.0 = prob * 1.76 - 1.0
    expected_net_ev = (pred.prob_over * (2.00 * 0.88)) - 1.0
    assert math.isclose(pred.ev_over_net_tax12, expected_net_ev, abs_tol=1e-3)


def test_recent_form_tracking_and_rolling_stats() -> None:
    tracker = ChronologicalPaceTracker()

    # Update 3 games for T1
    tracker.update_game(
        team_1="T1",
        team_2="Gen.G",
        kills_1=18,
        kills_2=12,
        duration_minutes=32.5,
        winner_team_1=True,
        date="2026-02-10",
    )
    tracker.update_game(
        team_1="T1",
        team_2="KT Rolster",
        kills_1=25,
        kills_2=15,
        duration_minutes=35.0,
        winner_team_1=True,
        date="2026-02-14",
    )
    tracker.update_game(
        team_1="T1",
        team_2="Hanwha Life",
        kills_1=10,
        kills_2=20,
        duration_minutes=28.0,
        winner_team_1=False,
        date="2026-02-18",
    )

    form = tracker.get_team_recent_form("T1", n=5)
    assert form.team_name == "T1"
    assert form.sample_size == 3
    assert len(form.recent_games) == 3

    # Order: most recent first
    assert form.recent_games[0].opponent == "Hanwha Life"
    assert form.recent_games[0].won is False
    assert form.recent_games[0].kills_for == 10
    assert form.recent_games[0].kills_against == 20
    assert form.recent_games[0].total_kills == 30
    assert form.recent_games[0].date == "2026-02-18"

    assert form.recent_games[1].opponent == "KT Rolster"
    assert form.recent_games[1].won is True

    # Rolling averages: kills = (18 + 25 + 10) / 3 = 17.67 -> 17.7
    assert math.isclose(form.avg_kills, 17.7, abs_tol=0.1)
    # deaths = (12 + 15 + 20) / 3 = 15.67 -> 15.7
    assert math.isclose(form.avg_deaths, 15.7, abs_tol=0.1)
    # total kills = (30 + 40 + 30) / 3 = 33.3
    assert math.isclose(form.avg_total_kills, 33.3, abs_tol=0.1)
    # Empirical distribution checks
    assert form.min_kills == 10
    assert form.max_kills == 25
    assert form.std_kills > 0
    assert form.kill_brackets["10-14"] == 1
    assert form.kill_brackets["15-19"] == 1
    assert form.kill_brackets["25+"] == 1
    assert form.kill_brackets["<10"] == 0


def test_league_pace_context_benchmarks() -> None:
    tracker = ChronologicalPaceTracker()

    lck_ctx = tracker.get_league_context("LCK Spring 2026")
    assert lck_ctx.league_key == "LCK"
    assert math.isclose(lck_ctx.avg_kills, 25.73, abs_tol=1e-2)
    assert "wolne makro" in lck_ctx.pace_category
    assert lck_ctx.p25_kills < lck_ctx.median_kills < lck_ctx.p75_kills

    vcs_ctx = tracker.get_league_context("VCS Dawn 2025")
    assert vcs_ctx.league_key == "VCS"
    assert math.isclose(vcs_ctx.avg_kills, 30.89, abs_tol=1e-2)
    assert "fiesta" in vcs_ctx.pace_category


def test_match_prop_context_in_prediction() -> None:
    tracker = ChronologicalPaceTracker()
    tracker.update_game("JDG", "BLG", kills_1=25, kills_2=20, duration_minutes=33.0, winner_team_1=True)
    tracker.update_game("JDG", "TES", kills_1=22, kills_2=18, duration_minutes=30.0, winner_team_1=True)
    tracker.update_game("BLG", "WBG", kills_1=24, kills_2=16, duration_minutes=31.0, winner_team_1=True)

    model = KillDistributionModel()
    pred = model.predict_from_tracker(tracker, "JDG", "BLG", league="LPL")

    assert pred.context is not None
    assert pred.context.team_a_form.team_name == "JDG"
    assert pred.context.team_b_form.team_name == "BLG"
    assert pred.context.league_context.league_key == "LPL"
    assert pred.context.expected_pace_kills > 0
    assert isinstance(pred.context.delta_vs_league_avg, float)
    assert "LPL" in pred.context.summary_narrative
    assert len(pred.context.summary_narrative) > 20
