"""Counterfactual LPL routes use simulated qualifiers, never observed entrants."""
from itertools import combinations

import pytest

from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_formats import simulate_tournament


def _run(profile_id, count, upsets=()):
    teams = [f"seed-{index + 1}" for index in range(count)]
    spec = instantiate_profile(profile_id, teams)
    probabilities = {(a, b, bo): 1.0 for a, b in combinations(teams, 2) for bo in (1, 3, 5)}
    for first, second in upsets:
        for bo in (1, 3, 5):
            probabilities[teams[first - 1], teams[second - 1], bo] = 0.0
    return simulate_tournament(spec, probabilities, simulations=1, seed=81,
                               score_distributions={3: [1, 0], 5: [1, 0, 0]})


def _matches(result, stage):
    return [match for match in result["trace"] if match["stage"] == stage]


def _pair(match):
    return {match["team_a"], match["team_b"]}


def test_staggered_2020_bracket_preserves_byes_and_plays_bronze():
    result = _run("lpl-2020-staggered-playoffs", 8)
    matches = {match["round"]: match for match in _matches(result, "playoffs")}
    assert _pair(matches["r1a"]) == {"seed-5", "seed-8"}
    assert _pair(matches["r1b"]) == {"seed-6", "seed-7"}
    assert _pair(matches["sfa"]) == {"seed-1", "seed-4"}
    assert _pair(matches["bronze"]) == {"seed-3", "seed-4"}
    assert result["stage_rank_prob"]["playoffs"]["all"]["seed-3"][2] == 1
    assert result["expected_series"] == 8


def test_2024_placements_send_three_from_large_group_and_two_from_others():
    result = _run("lpl-2024-summer-placements", 17)
    qualified = {team for team, chance in result["advance_prob"]["ascend"].items() if chance == 1}
    assert qualified == {f"seed-{n}" for n in (1, 2, 5, 6, 9, 10, 13, 14, 15)}
    assert result["advance_prob"]["nirvana"]["seed-16"] == 1
    assert result["expected_series"] == 56


def test_2024_play_in_upset_reseeds_qualifiers_before_staggered_entry():
    result = _run("lpl-2024-summer-playoffs", 13, upsets=[(8, 13)])
    matches = _matches(result, "ladder_a")
    assert _pair(matches[0]) == {"seed-9", "seed-10"}
    assert _pair(_matches(result, "ladder_b")[0]) == {"seed-7", "seed-13"}
    assert result["advance_prob"]["play_in"]["seed-8"] == 0
    assert result["advance_prob"]["play_in"]["seed-13"] == 1
    assert result["expected_series"] == 15


def test_2025_placement_lcq_routes_thirds_and_changes_series_length():
    result = _run("lpl-2025-split-2-placements", 16)
    matches = _matches(result, "last_chance")
    assert {frozenset(_pair(match)) for match in matches[:2]} == {
        frozenset(("seed-3", "seed-15")), frozenset(("seed-7", "seed-11"))
    }
    assert [match["best_of"] for match in matches] == [1, 1, 3, 3, 3]
    ascend = result["advance_prob"]["ascend"]
    assert ascend["seed-3"] == ascend["seed-7"] == 1
    assert ascend["seed-11"] == ascend["seed-15"] == 0
    assert sum(ascend.values()) == 10
    assert result["expected_series"] == 53


def test_partial_entry_runners_up_have_one_life_but_group_winners_have_two():
    result = _run("lpl-2025-split-1-playoffs", 8)
    matches = _matches(result, "playoffs")
    losses = {f"seed-{n}": 0 for n in range(1, 9)}
    for match in matches:
        losses[match["loser"]] += 1
    assert losses["seed-3"] == losses["seed-4"] == 2
    assert all(losses[f"seed-{n}"] == 1 for n in range(5, 9))
    assert result["expected_series"] == 10


@pytest.mark.parametrize("profile_id,ascend_series,nirvana_series", [
    ("lpl-2025-split-3-groups", 56, 30),
    ("lpl-2026-split-2-groups", 56, 15),
])
def test_group_cycle_variants_do_not_reuse_a_superficially_similar_schedule(
        profile_id, ascend_series, nirvana_series):
    result = _run(profile_id, 14)
    assert len(_matches(result, "ascend_regular")) == ascend_series
    assert len(_matches(result, "nirvana_regular")) == nirvana_series
    assert result["advance_prob"]["direct_playoffs"]["seed-4"] == 1
    assert result["advance_prob"]["play_in_slots"]["seed-5"] == 1
    assert result["advance_prob"]["play_in_slots"]["seed-12"] == 1
    assert result["advance_prob"]["play_in_slots"]["seed-13"] == 0


def test_knights_reseeds_surviving_losers_and_lower_round_winners():
    result = _run("lpl-2026-split-1-playoffs", 12, upsets=[(5, 8)])
    rounds = {match["round"]: match for match in _matches(result, "knights_round_3")}
    assert _pair(rounds["r3a"]) == {"seed-5", "seed-10"}
    assert _pair(rounds["r3b"]) == {"seed-7", "seed-9"}
    qualified = result["advance_prob"]["qualifiers"]
    assert {team for team, chance in qualified.items() if chance == 1} == {
        "seed-5", "seed-6", "seed-7", "seed-8"
    }
    assert result["expected_series"] == 20


def test_declared_opponent_policy_chooses_from_actual_play_in_survivors():
    result = _run("lpl-2025-grand-finals", 12, upsets=[(5, 12)])
    opening = {match["round"]: match for match in _matches(result, "playoffs")}
    assert _pair(opening["u1"]) == {"seed-1", "seed-12"}
    assert _pair(opening["u2"]) == {"seed-4", "seed-6"}
    assert _pair(opening["u3"]) == {"seed-2", "seed-8"}
    assert _pair(opening["u4"]) == {"seed-3", "seed-7"}
    assert result["expected_series"] == 18


def test_first_stand_qualifies_both_finalists_not_only_the_champion():
    result = _run("lpl-2026-split-1-playoffs", 12, upsets=[(1, 2)])
    assert result["champion_prob"]["seed-2"] == 1
    assert {team for team, chance in result["advance_prob"]["playoffs"].items() if chance == 1} == {
        "seed-1", "seed-2"
    }


def test_2020_spring_and_summer_have_different_actual_qualification_cutoffs():
    spring = _run("lpl-2020-spring-mid-season-cup", 8)
    summer = _run("lpl-2020-staggered-playoffs", 8)
    assert {team for team, chance in spring["advance_prob"]["playoffs"].items() if chance == 1} == {
        "seed-1", "seed-2", "seed-3", "seed-4"
    }
    assert {team for team, chance in summer["advance_prob"]["playoffs"].items() if chance == 1} == {"seed-1"}
