"""Qualification and seed advantages for completed LCK/LEC scenario recipes."""
from itertools import combinations

import pytest

from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_formats import simulate_tournament


def _forecast(profile_id, count, *, upsets=(), reverse=False, formats=(1, 3, 5)):
    teams = [f"seed-{i + 1}" for i in range(count)]
    spec = instantiate_profile(profile_id, teams)
    probabilities = {
        (a, b, bo): float(not reverse and (a, b) not in upsets)
        for a, b in combinations(teams, 2)
        for bo in formats
    }
    return simulate_tournament(
        spec, probabilities, simulations=1, seed=81,
        score_distributions={3: [1.0, 0.0], 5: [1.0, 0.0, 0.0]},
    )


def _pairs(result, stage):
    return {
        row["round"]: {row["team_a"], row["team_b"]}
        for row in result["trace"] if row["stage"] == stage
    }


@pytest.mark.parametrize("profile_id", [
    "lck-2021-2022-six-team-semifinal-byes",
    "lck-2026-cup-three-qualifier-reselection",
])
def test_top_seed_reselects_lowest_surviving_seed_after_quarterfinal_upset(profile_id):
    result = _forecast(profile_id, 6, upsets={("seed-3", "seed-6")})
    pairs = _pairs(result, "playoffs" if "semifinal" in profile_id else "play_in")
    assert pairs["r1a"] == {"seed-3", "seed-6"}
    assert pairs["r1b"] == {"seed-4", "seed-5"}
    assert pairs["r2a"] == {"seed-1", "seed-6"}
    assert pairs["r2b"] == {"seed-2", "seed-4"}
    first_two = result["trace"][:2]
    assert all(not ({row["team_a"], row["team_b"]} & {"seed-1", "seed-2"}) for row in first_two)


def test_cup_play_in_bye_seed_can_lose_once_and_still_take_third_qualification():
    result = _forecast("lck-2025-cup-three-qualifier", 6, upsets={("seed-1", "seed-4")})
    assert _pairs(result, "play_in")["decider"] == {"seed-1", "seed-3"}
    assert {team for team, probability in result["advance_prob"]["play_in"].items() if probability} == {
        "seed-1", "seed-2", "seed-4",
    }
    assert [row["best_of"] for row in result["trace"]] == [3, 3, 3, 3, 5]
    assert result["expected_series"] == 5
    assert result["champion_prob"] is None


def test_season_play_in_preserves_group_pairings_and_two_qualification_paths():
    result = _forecast("lck-2025-season-play-in-two-qualifier", 4)
    pairs = _pairs(result, "play_in")
    assert pairs["opening-1"] == {"seed-1", "seed-4"}
    assert pairs["opening-2"] == {"seed-2", "seed-3"}
    assert pairs["decider"] == {"seed-2", "seed-3"}
    assert result["advance_prob"]["play_in"]["seed-1"] == 1
    assert result["advance_prob"]["play_in"]["seed-2"] == 1
    assert result["expected_series"] == 5
    assert sum(row["winner"] == "seed-1" for row in result["trace"]) == 2
    assert sum(row["winner"] == "seed-2" for row in result["trace"]) == 2


@pytest.mark.parametrize("upsets,early_loser,late_loser", [
    (set(), "seed-4", "seed-3"),
    ({("seed-1", "seed-4")}, "seed-3", "seed-1"),
])
def test_lec_upper_losers_enter_different_lower_rounds_by_seed_not_upper_match(upsets, early_loser, late_loser):
    result = _forecast("lec-2020-2022-spring-seeded-lower-playoffs", 6, upsets=upsets)
    pairs = _pairs(result, "playoffs")
    assert pairs["l1"] == {"seed-5", "seed-6"}
    assert pairs["l2"] == {"seed-5", early_loser}
    assert pairs["l3"] == {early_loser, late_loser}
    assert all("seed-6" not in pair for match, pair in pairs.items() if match != "l1")
    assert result["expected_series"] == 8


@pytest.mark.parametrize("profile_id,qualifiers", [
    ("lec-2020-2022-spring-seeded-lower-playoffs", 1),
    ("lec-2020-2022-summer-four-qualifier-playoffs", 4),
    ("lec-2021-summer-three-qualifier-playoffs", 3),
])
def test_lec_season_specific_qualification_cutoff(profile_id, qualifiers):
    result = _forecast(profile_id, 6)
    assert {team for team, probability in result["advance_prob"]["playoffs"].items() if probability} == {
        f"seed-{rank}" for rank in range(1, qualifiers + 1)
    }


def test_lec_single_round_robin_advances_eight_without_replaying_return_matches():
    result = _forecast("lec-2024-2025-single-round-robin-bo1", 10, reverse=True)
    assert result["expected_series"] == 45
    assert {team for team, probability in result["advance_prob"]["regular"].items() if probability} == {
        f"seed-{rank}" for rank in range(3, 11)
    }


def test_lec_summer_groups_dynamically_route_top_two_and_cross_play_seeds():
    result = _forecast("lec-2025-summer-group-qualification", 10, reverse=True, formats=(3,))
    assert result["expected_series"] == 20
    assert all((int(row["team_a"].split("-")[1]) <= 5) == (int(row["team_b"].split("-")[1]) <= 5)
               for row in result["trace"])
    assert {team for team, probability in result["advance_prob"]["upper_entrants"].items() if probability} == {
        "seed-5", "seed-4", "seed-10", "seed-9",
    }
    assert {team for team, probability in result["advance_prob"]["cross_play_entrants"].items() if probability} == {
        "seed-3", "seed-2", "seed-8", "seed-7",
    }


def test_lec_summer_cross_play_winners_feed_distinct_lower_matches():
    result = _forecast("lec-2025-summer-cross-play-playoffs", 8)
    pairs = _pairs(result, "playoffs")
    # Input slots: A1,A2,A3,A4,B1,B2,B3,B4.
    assert pairs["cross_a"] == {"seed-3", "seed-8"}
    assert pairs["cross_b"] == {"seed-7", "seed-4"}
    assert pairs["u1"] == {"seed-1", "seed-6"}
    assert pairs["u2"] == {"seed-5", "seed-2"}
    assert pairs["l1"] == {"seed-6", "seed-3"}
    assert pairs["l2"] == {"seed-5", "seed-4"}
    assert pairs["ls"] == {"seed-3", "seed-4"}
    assert result["advance_prob"]["playoffs"]["seed-3"] == 1
    assert result["expected_series"] == 10
    assert all("seed-7" not in pair and "seed-8" not in pair
               for match, pair in pairs.items() if not match.startswith("cross_"))
