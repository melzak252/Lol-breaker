"""Family evidence must preserve symmetry and cannot learn from a shared location."""
from datetime import date, timedelta
import math

import pandas as pd
import pytest

from src.models.competition_tiers import CompetitionIdentity, CompetitionScope, CompetitionTier
from src.models.family_reliability import (
    ReleasedAffiliations,
    ReplayMap,
    ReplaySeries,
    _replay_forecasts,
    apply_temperature,
)
from src.ratings.family_calibrated_glicko2 import FamilyMapBridge

def observation(a, b, outcome, p=.5, weight=1.):
    return {"player_probability": p, "family_weights_a": a, "family_weights_b": b,
            "outcome": outcome, "weight": weight}


def test_family_update_moves_cross_family_forecasts_without_breaking_side_symmetry():
    model = FamilyMapBridge(initial_deviation=150., process_deviation=1.)
    a, b = {"LCK": 1.}, {"LPL": 1.}
    model.update_period(date(2024, 1, 1), [observation(a, b, 1)])
    p = model.predict_map_probability(.5, a, b, date(2024, 1, 2))
    reverse = model.predict_map_probability(.5, b, a, date(2024, 1, 2))
    assert .5 < p < 1.
    assert p + reverse == pytest.approx(1., abs=1e-14)
    assert model.predict_map_probability(.5, a, a, date(2024, 1, 2)) == .5


def test_unknown_family_is_neutral_and_hypothetical_query_does_not_mutate_state():
    model = FamilyMapBridge(initial_deviation=150., process_deviation=1.)
    model.update_period(date(2024, 1, 1), [observation({"LCK": 1.}, {"LPL": 1.}, 1)])
    before = model.to_state()
    assert model.predict_map_probability(.7, {}, {}, date(2024, 8, 1)) == pytest.approx(.7)
    model.predict_map_probability(.7, {"LCK": .4}, {"LPL": .6}, date(2024, 8, 1))
    assert model.to_state() == before
    restored = FamilyMapBridge.from_state(before)
    assert restored.predict_map_probability(.7, {"LCK": .4}, {"LPL": .6}, date(2024, 8, 1)) == model.predict_map_probability(.7, {"LCK": .4}, {"LPL": .6}, date(2024, 8, 1))


def test_extreme_family_probabilities_are_not_clipped_before_scoring():
    model = FamilyMapBridge()
    assert model.predict_map_probability(1e-20, {}, {}, date(2024, 1, 1)) == pytest.approx(1e-20, rel=1e-12, abs=0)
    logit = math.log(1e-8) - math.log1p(-1e-8)
    expected = math.exp(logit / .25) / (1 + math.exp(logit / .25))
    assert apply_temperature(1e-8, .25) == pytest.approx(expected, rel=1e-12, abs=0)


def test_unrepresentable_calibrated_endpoint_stops_instead_of_capping_loss():
    with pytest.raises(ArithmeticError, match="interior"):
        apply_temperature(1-1e-10, .25)


def test_diagonal_posterior_matches_frozen_prior_bernoulli_score_and_information():
    model = FamilyMapBridge(initial_deviation=150., process_deviation=0.)
    coefficient = math.log(10.) / 400.
    prior_variance = 150. ** 2
    information = coefficient ** 2 * .25
    gradient = coefficient * .5
    expected_variance = 1. / (1. / prior_variance + information)
    expected_mean = expected_variance * gradient
    model.update_period(date(2024, 1, 1), [observation({"LCK": 1.}, {"LPL": 1.}, 1)])
    assert model.get_family_state("LCK").mean == pytest.approx(expected_mean)
    assert model.get_family_state("LCK").variance == pytest.approx(expected_variance)
    assert model.get_family_state("LPL").mean == pytest.approx(-expected_mean)


def test_family_batch_is_order_invariant_and_invalid_batch_is_atomic():
    first = FamilyMapBridge()
    second = FamilyMapBridge()
    observations = [
        observation({"LCK": 1.}, {"LPL": 1.}, 1, p=.63, weight=.5),
        observation({"LEC": .6, "LCK": .4}, {"LPL": 1.}, 0, p=.41, weight=.5),
    ]
    first.update_period(date(2024, 1, 1), observations)
    second.update_period(date(2024, 1, 1), list(reversed(observations)))
    assert first.to_state() == second.to_state()
    before = first.to_state()
    with pytest.raises(ValueError):
        first.update_period(date(2024, 1, 2), [observations[0], {"outcome": 2}])
    assert first.to_state() == before


def test_released_affiliation_transfer_conflict_ttl_and_mixed_roster_profiles():
    state = ReleasedAffiliations()
    lck = _history("lck", date(2024, 1, 1), date(2024, 1, 1), "LCK", ("p", "a", "b", "c", "d"))
    state.consume_period(date(2024, 1, 1), [lck])
    assert state.lookup("p", date(2024, 1, 1)) == (
        None, "not_prior_to_query_day", 0,
    )
    affiliation, reason, age = state.lookup("p", date(2024, 1, 2))
    assert affiliation is not None and reason is None and age == 1

    lpl = _history("lpl", date(2024, 1, 2), date(2024, 1, 2), "LPL", ("p", "a", "b", "c", "d"))
    state.consume_period(date(2024, 1, 2), [lpl])
    affiliation, reason, _ = state.lookup("p", date(2024, 1, 3))
    assert affiliation is not None and reason is None
    assert affiliation.family == "LPL"
    assert affiliation.transfer_day == date(2024, 1, 2)
    assert affiliation.transferred_from == "LCK"

    profile = state.profile(("p", "a", "b", "c", "unknown"), date(2024, 1, 3))
    assert profile["known_players"] == 4
    assert profile["unknown_players"] == 1
    assert profile["mixed"] is False
    assert profile["weights"] == {"LPL": .8}
    age_365 = state.lookup("p", date(2024, 1, 2) + timedelta(days=365))
    assert age_365[0] is not None and age_365[2] == 365
    mixed_state = ReleasedAffiliations()
    mixed_state.consume_period(date(2024, 3, 1), [
        _history("mixed-lck", date(2024, 3, 1), date(2024, 3, 1), "LCK", ("a1", "a2", "z1", "z2", "z3")),
        _history("mixed-lpl", date(2024, 3, 1), date(2024, 3, 1), "LPL", ("b1", "b2", "z4", "z5", "z6")),
    ])
    mixed_profile = mixed_state.profile(
        ("a1", "a2", "b1", "unknown1", "unknown2"), date(2024, 3, 2)
    )
    assert mixed_profile["known_players"] == 3
    assert mixed_profile["unknown_players"] == 2
    assert mixed_profile["mixed"] is True
    assert mixed_profile["weights"] == {"LCK": .4, "LPL": .2}

    cross_state = ReleasedAffiliations()
    cross_state.consume_period(date(2024, 4, 1), [
        _history("cross-prior-lck", date(2024, 4, 1), date(2024, 4, 1), "LCK", ("p", "a", "b", "c", "d")),
    ])
    cross_state.consume_period(date(2024, 4, 2), [
        _history(
            "cross-overwrite", date(2024, 4, 2), date(2024, 4, 2), "International",
            ("p", "u", "v", "w", "x"), scope=CompetitionScope.CROSS_LEAGUE,
        ),
    ])
    cross_lookup = cross_state.lookup("p", date(2024, 4, 3))
    assert cross_lookup[0] is not None and cross_lookup[0].family == "LCK"

    expiry = state.lookup("p", date(2024, 1, 2) + timedelta(days=366))
    assert expiry[0] is None and expiry[1] == "expired_over_365_days"

    conflict_state = ReleasedAffiliations()
    conflict_state.consume_period(date(2024, 2, 1), [
        _history("same-lck", date(2024, 2, 1), date(2024, 2, 1), "LCK", ("shared", "a", "b", "c", "d")),
        _history("same-lpl", date(2024, 2, 1), date(2024, 2, 1), "LPL", ("shared", "u", "v", "w", "x")),
        _history(
            "cross", date(2024, 2, 1), date(2024, 2, 1), "Cross League",
            ("shared", "p2", "p3", "p4", "p5"), scope=CompetitionScope.CROSS_LEAGUE,
        ),
    ])
    assert conflict_state.lookup("shared", date(2024, 2, 2)) == (
        None, "same_release_conflict", None,
    )
    conflict_state.consume_period(date(2024, 2, 2), [
        _history("resolve", date(2024, 2, 2), date(2024, 2, 2), "LCK", ("shared", "a", "b", "c", "d")),
    ])
    resolved = conflict_state.lookup("shared", date(2024, 2, 3))
    assert resolved[0] is not None and resolved[0].family == "LCK"


def test_release_replay_uses_actual_map_substitutes_and_ignores_future_outcome():
    history, targets = _mini_replay_inputs()
    changed_history = [
        _history(
            item.match_id, item.day, item.release_day, item.competition.family,
            item.maps[0].roster_a, roster_b=item.maps[0].roster_b,
            maps=tuple(
                ReplayMap(game.game_id, game.day, game.roster_a, game.roster_b, 1 - game.outcome)
                if item.match_id == "late-lck" else game
                for game in item.maps
            ),
            best_of=item.best_of,
            scope=item.competition.scope,
        )
        for item in history
    ]
    before, before_receipts, _, _, _ = _replay_forecasts(history, targets)
    perturbed, perturbed_receipts, _, engine, _ = _replay_forecasts(changed_history, targets)
    assert before["target-before"]["p_player_raw"] == perturbed["target-before"]["p_player_raw"]
    assert before["target-before"]["state_chain_sha256"] == perturbed["target-before"]["state_chain_sha256"]
    assert before_receipts["target-before"]["state_receipt_sha256"] == perturbed_receipts["target-before"]["state_receipt_sha256"]
    after = before["target-after"]
    assert after["side_a"]["weights"] == {"LCK": 1.}
    assert after["side_b"]["weights"] == {"LPL": 1.}
    assert after["player_support_a"]["min_games"] == 1
    assert engine.get_player_games("substitute") == 1
    assert before_receipts["target-after"]["update_policy"] == "released_daily_frozen_map_periods"


def test_known_lineup_replay_rejects_rosters_on_wrong_sides():
    source = _history("target", date(2024, 1, 1), date(2024, 1, 1), "LCK",
                      ("a1", "a2", "a3", "a4", "a5"))
    row = {"golgg_match_id": source.match_id, "date": source.day,
           "result_day": source.release_day, "team1_id": source.team_a,
           "team2_id": source.team_b, "best_of": 1, "y": 1,
           "feature_source_max_day": None, "roster_a": source.maps[0].roster_b,
           "roster_b": source.maps[0].roster_a, "tournament_name": "LCK"}
    with pytest.raises(ValueError, match="roster"):
        _replay_forecasts([source], pd.DataFrame([row]))


def test_known_lineup_replay_aligns_reversed_canonical_source_orientation():
    roster = ("a1", "a2", "a3", "a4", "a5")
    source = _history("target", date(2024, 1, 2), date(2024, 1, 2), "LCK", roster)
    prior = _history("prior", date(2024, 1, 1), date(2024, 1, 1), "LCK", roster,
                     roster_b=source.maps[0].roster_b)
    row = {"golgg_match_id": source.match_id, "date": source.day,
           "result_day": source.release_day, "team1_id": source.team_a,
           "team2_id": source.team_b, "best_of": 1, "y": 1,
           "feature_source_max_day": prior.release_day, "roster_a": source.maps[0].roster_a,
           "roster_b": source.maps[0].roster_b, "tournament_name": "LCK"}
    direct, _, _, _, _ = _replay_forecasts([prior, source], pd.DataFrame([row]))
    row["team1_id"], row["team2_id"] = row["team2_id"], row["team1_id"]
    row["roster_a"], row["roster_b"] = row["roster_b"], row["roster_a"]
    row["y"] = 0
    reverse, _, _, _, _ = _replay_forecasts([prior, source], pd.DataFrame([row]))
    for column in ("p_player_raw", "p_family_raw"):
        assert direct["target"][column] > .5
        assert direct["target"][column] + reverse["target"][column] == pytest.approx(1., abs=1e-14)


def test_family_history_classification_uses_reconciled_event_date(tmp_path):
    import json
    from betting_app.tests.test_siamese_research_dataset import match
    from src.models.family_reliability import _load_history

    source = match(1, "2023-01-01")
    source["tournament_name"] = "TCL"
    source["games"][0]["date"] = "2022-12-31"
    for side in ("t1", "t2"):
        players = source["games"][0][f"{side}_players"]
        source["games"][0][f"{side}_players"] = dict(zip(("top", "jungle", "mid", "adc", "support"), players.values(), strict=True))
    path = tmp_path / "history.json"
    path.write_text(json.dumps([source]))
    history, _ = _load_history(path, {})
    assert history[0].day == date(2022, 12, 31)
    assert history[0].competition.tier is CompetitionTier.MINOR_TOP_LEVEL


def test_same_family_tier_transition_is_neither_transfer_nor_conflict():
    from dataclasses import replace
    from src.models.competition_tiers import classify_competition

    roster = ("p", "a", "b", "c", "d")
    old = _history("old", date(2022, 12, 30), date(2022, 12, 30), "TCL", roster)
    old = replace(old, competition=classify_competition("TCL", old.day))
    new = _history("new", date(2023, 1, 2), date(2023, 1, 2), "TCL", roster)
    new = replace(new, competition=classify_competition("TCL", new.day))
    state = ReleasedAffiliations()
    state.consume_period(old.release_day, [old])
    state.consume_period(new.release_day, [new])
    affiliation, _, _ = state.lookup("p", date(2023, 1, 3))
    assert affiliation.family == "TCL"
    assert affiliation.transfer_day is None
    simultaneous = date(2023, 1, 3)
    state.consume_period(simultaneous, [replace(old, release_day=simultaneous),
                                       replace(new, release_day=simultaneous)])
    affiliation, reason, _ = state.lookup("p", date(2023, 1, 4))
    assert reason is None
    assert affiliation.family == "TCL" and affiliation.transfer_day is None


def _history(
    match_id, day, release_day, family, roster_a, *, roster_b=None, maps=None,
    best_of=1, scope=CompetitionScope.DOMESTIC,
):
    if roster_b is None:
        roster_b = tuple(f"{match_id}-b{index}" for index in range(5))
    if maps is None:
        maps = (ReplayMap(f"{match_id}-map", day, tuple(roster_a), tuple(roster_b), 1),)
    tier = CompetitionTier.MAJOR if scope is CompetitionScope.DOMESTIC else CompetitionTier.INTERNATIONAL
    identity = CompetitionIdentity(family, tier, scope, f"test.{family.lower()}")
    return ReplaySeries(
        match_id=match_id,
        day=day,
        source_header_day=day,
        release_day=release_day,
        team_a=f"{match_id}-A",
        team_b=f"{match_id}-B",
        best_of=best_of,
        tournament=family,
        competition=identity,
        maps=tuple(maps),
    )


def _mini_replay_inputs():
    side_a = ("l1", "l2", "l3", "l4", "l5")
    side_b = ("q1", "q2", "q3", "q4", "q5")
    lck_opponents = ("h1", "h2", "h3", "h4", "h5")
    lck_maps = (
        ReplayMap("late-1", date(2024, 1, 1), side_a, lck_opponents, 1),
        ReplayMap("late-2", date(2024, 1, 2), ("l1", "substitute", "l3", "l4", "l5"), lck_opponents, 1),
    )
    history = [
        _history(
            "early-lpl", date(2024, 1, 2), date(2024, 1, 4), "LPL",
            side_b, maps=(ReplayMap("early-map", date(2024, 1, 2), side_b, tuple(f"u{i}" for i in range(5)), 1),),
        ),
        _history(
            "late-lck", date(2024, 1, 1), date(2024, 1, 10), "LCK",
            side_a, roster_b=lck_opponents, maps=lck_maps, best_of=3,
        ),
    ]
    roster_after = ("l1", "substitute", "l3", "l4", "l5")
    targets = pd.DataFrame([
        {
            "golgg_match_id": "target-before", "team1_id": "target-A", "team2_id": "target-B",
            "date": date(2024, 1, 5), "result_day": date(2024, 1, 12),
            "feature_source_max_day": date(2024, 1, 4), "best_of": 3, "y": 1,
            "roster_a": side_a, "roster_b": side_b, "tournament_name": "LCK Spring",
        },
        {
            "golgg_match_id": "target-after", "team1_id": "target-A", "team2_id": "target-B",
            "date": date(2024, 1, 11), "result_day": date(2024, 1, 20),
            "feature_source_max_day": date(2024, 1, 10), "best_of": 3, "y": 1,
            "roster_a": roster_after, "roster_b": side_b, "tournament_name": "LCK Spring",
        },
    ])
    return history, targets
