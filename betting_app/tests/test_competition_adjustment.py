from __future__ import annotations

import json
import math

import pytest

from src.ratings.competition_adjustment import (
    CompetitionAdjustment,
    NEUTRAL_COMPETITION_ADJUSTMENT,
    adjust_probability,
)
from betting_app.services.upcoming_inference_service import (
    player_rating_probabilities,
    rating_probabilities,
)


def test_neutral_adjustment_is_exact_identity_including_boundaries() -> None:
    for probability in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert adjust_probability(probability, NEUTRAL_COMPETITION_ADJUSTMENT) == probability


def test_positive_location_difference_increases_side_a_probability() -> None:
    adjusted = adjust_probability(
        0.5,
        CompetitionAdjustment(mean=100.0, variance=0.0),
    )

    assert adjusted == pytest.approx(1.0 / (1.0 + 10.0 ** -0.25))
    assert adjusted > 0.5


def test_orientation_reversal_preserves_probability_symmetry() -> None:
    adjustment = CompetitionAdjustment(mean=73.0, variance=2_500.0)

    forward = adjust_probability(0.63, adjustment)
    reversed_probability = adjust_probability(0.37, adjustment.reversed())

    assert forward + reversed_probability == pytest.approx(1.0)


def test_uncertain_bridge_attenuates_probability_toward_half() -> None:
    certain = adjust_probability(
        0.7,
        CompetitionAdjustment(mean=80.0, variance=0.0),
    )
    uncertain = adjust_probability(
        0.7,
        CompetitionAdjustment(mean=80.0, variance=40_000.0),
    )

    assert 0.5 < uncertain < certain


@pytest.mark.parametrize(
    "adjustment",
    [
        CompetitionAdjustment(mean=0.0, variance=0.0),
        CompetitionAdjustment(mean=-20.0, variance=10.0),
    ],
)
def test_adjusted_probability_is_bounded(adjustment: CompetitionAdjustment) -> None:
    for probability in (0.0, 1e-15, 0.5, 1.0 - 1e-15, 1.0):
        result = adjust_probability(probability, adjustment)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0


@pytest.mark.parametrize(
    ("mean", "variance"),
    [(math.nan, 0.0), (0.0, math.nan), (0.0, -1.0), (math.inf, 0.0)],
)
def test_invalid_adjustments_fail_closed(mean: float, variance: float) -> None:
    with pytest.raises(ValueError):
        CompetitionAdjustment(mean=mean, variance=variance)


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_base_probability_fails_closed(probability: float) -> None:
    with pytest.raises(ValueError):
        adjust_probability(probability, NEUTRAL_COMPETITION_ADJUSTMENT)


def test_regional_posterior_projects_once_onto_raw_systems_not_glicko() -> None:
    systems = ("elo", "gl", "ts", "os", "pl", "tm")
    team_a = {system: {"rating_value": 100.0} for system in systems}
    team_b = {system: {"rating_value": 100.0} for system in systems}
    player_a = {system: {"avg_rating_value": 100.0} for system in systems}
    player_b = {system: {"avg_rating_value": 100.0} for system in systems}
    adjustment = CompetitionAdjustment(mean=100.0, variance=0.0)

    team_probabilities = rating_probabilities(team_a, team_b, adjustment)
    player_probabilities = player_rating_probabilities(player_a, player_b, adjustment)

    for probabilities in (team_probabilities, player_probabilities):
        assert probabilities["gl"] == 0.5
        assert all(probabilities[system] > 0.5 for system in systems if system != "gl")
        assert probabilities["consensus"] > 0.5


def test_resolve_team_affiliation_prefers_state_json() -> None:
    from betting_app.services.upcoming_inference_service import resolve_team_affiliation

    rating = {"state_json": json.dumps({"family": "LEC", "tier": "major"})}
    assert resolve_team_affiliation(rating, "LPL Summer 2024") == ("LEC", "major")


def test_resolve_team_affiliation_falls_back_to_tournament() -> None:
    from betting_app.services.upcoming_inference_service import resolve_team_affiliation

    assert resolve_team_affiliation(None, "LCK Spring 2024") == ("LCK", "major")
    assert resolve_team_affiliation({}, "CBLOL Split 1 2024") == ("CBLOL", "minor_top_level")
    assert resolve_team_affiliation({"state_json": None}, "LFL Spring 2024") == ("LFL", "regional")


def test_resolve_team_affiliation_ignores_cross_league_or_unknown() -> None:
    from betting_app.services.upcoming_inference_service import resolve_team_affiliation

    assert resolve_team_affiliation(None, "World Championship 2024") is None
    assert resolve_team_affiliation(None, "Mid-Season Invitational 2024") is None
    assert resolve_team_affiliation(None, "Some Random Tournament 2024") is None


def test_load_regional_adjustment_graceful_on_missing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    from betting_app.services import upcoming_inference_service as inference

    monkeypatch.setattr(inference, "query_df", lambda *_: pd.DataFrame())
    inference._REGIONAL_ENGINE_CACHE.clear()

    ratings_a = {"gl": {"state_json": json.dumps({"family": "LCK", "tier": "major"})}}
    ratings_b = {"gl": {"state_json": json.dumps({"family": "LEC", "tier": "major"})}}
    adj = inference.load_regional_adjustment(ratings_a, ratings_b, "ratings-v2")
    assert adj == NEUTRAL_COMPETITION_ADJUSTMENT


def test_load_regional_adjustment_resolves_via_tournament_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    from betting_app.services import upcoming_inference_service as inference
    from src.ratings.family_calibrated_glicko2 import FamilyCalibratedGlicko2, GaussianOffsetState

    engine = FamilyCalibratedGlicko2()
    engine._tier_states["major"] = GaussianOffsetState(mean=100.0, variance=50.0)
    engine._tier_states["minor_top_level"] = GaussianOffsetState(mean=-50.0, variance=50.0)
    engine._family_states["LCK"] = GaussianOffsetState(mean=150.0, variance=50.0)
    engine._family_tiers["LCK"] = "major"
    engine._family_states["CBLOL"] = GaussianOffsetState(mean=-50.0, variance=50.0)
    engine._family_tiers["CBLOL"] = "minor_top_level"
    mock_systems_json = json.dumps({
        "gl": {
            "engine": "family-calibrated-glicko2-v1",
            "state": engine.to_state(),
        }
    })
    df_run = pd.DataFrame([{"systems_json": mock_systems_json}])
    monkeypatch.setattr(inference, "query_df", lambda *_: df_run)
    inference._REGIONAL_ENGINE_CACHE.clear()

    ratings_a = {"gl": {"rating_value": 1700.0}}
    ratings_b = {"gl": {"rating_value": 1600.0}}
    adj = inference.load_regional_adjustment(
        ratings_a,
        ratings_b,
        "ratings-v2",
        "LCK Spring 2024",
        "CBLOL Split 1 2024",
    )
    assert adj != NEUTRAL_COMPETITION_ADJUSTMENT
    assert adj.mean > 0.0


def test_roster_player_ratings_applies_rookie_tier_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    from betting_app.services import upcoming_inference_service as inference
    from src.models.competition_tiers import CompetitionTier

    roster = {
        "team_name": "Test Team",
        "players": [
            {"player_id": "101", "player_name": "RookieStar", "role": "mid"},
            {"player_id": "102", "player_name": "VeteranFaker", "role": "bot"},
        ],
    }

    df_ratings = pd.DataFrame([
        # RookieStar (has huge uncalibrated ERL ratings)
        {"rating_system": "elo", "entity_name": "RookieStar", "normalized_entity_name": "101", "rating_value": 2300.0, "rd": None, "sigma": None, "games_played": 100, "last_match_at": "2026-08-01"},
        {"rating_system": "ts", "entity_name": "RookieStar", "normalized_entity_name": "101", "rating_value": 35.0, "rd": None, "sigma": 2.0, "games_played": 100, "last_match_at": "2026-08-01"},
        {"rating_system": "gl", "entity_name": "RookieStar", "normalized_entity_name": "101", "rating_value": 1950.0, "rd": 60.0, "sigma": None, "games_played": 100, "last_match_at": "2026-08-01"},
        # VeteranFaker (genuine major veteran)
        {"rating_system": "elo", "entity_name": "VeteranFaker", "normalized_entity_name": "102", "rating_value": 2400.0, "rd": None, "sigma": None, "games_played": 1000, "last_match_at": "2026-08-01"},
        {"rating_system": "ts", "entity_name": "VeteranFaker", "normalized_entity_name": "102", "rating_value": 34.0, "rd": None, "sigma": 2.1, "games_played": 1000, "last_match_at": "2026-08-01"},
        {"rating_system": "gl", "entity_name": "VeteranFaker", "normalized_entity_name": "102", "rating_value": 1980.0, "rd": 65.0, "sigma": None, "games_played": 1000, "last_match_at": "2026-08-01"},
    ])

    # Mock query_df to return ratings
    monkeypatch.setattr(inference, "query_df", lambda sql, *_: df_ratings)
    # Mock get_player_major_game_counts
    monkeypatch.setattr(inference, "get_player_major_game_counts", lambda ids: {"101": 5, "102": 500})

    # 1. In a Major Tier match (e.g. LEC)
    major_result = inference.load_roster_player_ratings(roster, "ratings-v2", competition_tier=CompetitionTier.MAJOR)
    rookie_elo = next(p for p in major_result["elo"]["players"] if p["player_id"] == "101")
    veteran_elo = next(p for p in major_result["elo"]["players"] if p["player_id"] == "102")
    rookie_ts = next(p for p in major_result["ts"]["players"] if p["player_id"] == "101")
    rookie_gl = next(p for p in major_result["gl"]["players"] if p["player_id"] == "101")

    assert rookie_elo["rating_value"] == 1750.0  # Capped from 2300.0
    assert veteran_elo["rating_value"] == 2400.0  # Untouched for veteran
    assert rookie_ts["rating_value"] == 28.0  # Capped from 35.0
    assert rookie_ts["sigma"] == 3.83  # Uncertainty floor applied
    assert rookie_gl["rd"] == 150.0  # RD floor applied

    # 2. In a Regional Tier match (e.g. SuperLiga)
    regional_result = inference.load_roster_player_ratings(roster, "ratings-v2", competition_tier=CompetitionTier.REGIONAL)
    rookie_regional_elo = next(p for p in regional_result["elo"]["players"] if p["player_id"] == "101")
    assert rookie_regional_elo["rating_value"] == 2300.0  # Untouched in regional
