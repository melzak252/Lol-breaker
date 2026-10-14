"""Qualification recipes consume simulated ranks, never archived finishers."""
from itertools import combinations

import pytest

from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_formats import simulate_tournament


def _run(profile, count, probability):
    teams = [f"team-{i:02d}" for i in range(count)]
    spec = instantiate_profile(profile, teams)
    probabilities = {
        (a, b, best_of): probability
        for a, b in combinations(teams, 2)
        for best_of in (1, 3, 5)
    }
    return simulate_tournament(spec, probabilities, simulations=1, seed=81)


@pytest.mark.parametrize("probability,offset", [(1.0, 0), (0.0, 3)])
def test_masters_group_ranks_cross_into_four_independent_qualifiers(probability, offset):
    result = _run("emea-2020-2024-play-in-cross-qualification", 16, probability)
    runner_offset = 1 if offset == 0 else 2
    expected = [
        (f"team-{4 * a + offset:02d}", f"team-{4 * b + runner_offset:02d}")
        for a, b in ((0, 1), (1, 0), (2, 3), (3, 2))
    ]
    qualifiers = [match for match in result["trace"] if match["stage"].startswith("qualify_")]
    assert [(match["team_a"], match["team_b"]) for match in qualifiers] == expected
    assert all(match["best_of"] == 3 for match in qualifiers)
    assert sum(result["advance_prob"]["groups"].values()) == 8
    assert sum(sum(result["advance_prob"][f"qualify_{group}"].values()) for group in "ABCD") == 4
    assert result["expected_series"] == 52
    assert result["champion_prob"] is None


def test_worlds_upper_winners_bypass_opposite_lower_bracket_qualifiers():
    result = _run("worlds-2024-cross-bracket-play-in", 8, 1.0)
    matches = {(match["stage"], match["round"]): match for match in result["trace"]}
    for group, other in (("A", "B"), ("B", "A")):
        upper = matches[(f"group_{group}", "upper")]
        opposite_lower = matches[(f"group_{other}", "lower")]
        decider = next(match for match in result["trace"] if match["stage"] == f"qualify_{group}")
        assert (decider["team_a"], decider["team_b"]) == (upper["loser"], opposite_lower["winner"])
        assert result["advance_prob"][f"group_{group}"][upper["winner"]] == 1
        assert upper["winner"] not in (decider["team_a"], decider["team_b"])
    qualified = {
        team for stage in ("group_A", "group_B", "qualify_A", "qualify_B")
        for team, probability in result["advance_prob"][stage].items() if probability == 1
    }
    assert qualified == {"team-00", "team-01", "team-02", "team-04"}
    assert result["expected_series"] == 10
    assert result["champion_prob"] is None


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_winter_group_finals_supply_eight_winners_while_champions_receive_byes(probability):
    result = _run("emea-2026-winter-champions-groups-playoffs", 36, probability)
    matches = {(match["stage"], match["round"]): match for match in result["trace"]}
    champions = [next(match for match in result["trace"] if match["stage"] == f"champions_{i}")
                 for i in range(4)]
    group_winners = set()
    for group in "ABCDEFGH":
        stage = f"group_{group}"
        upper, lower, final = [matches[(stage, node)] for node in ("upper", "decider", "final")]
        assert (final["team_a"], final["team_b"]) == (upper["winner"], lower["winner"])
        assert result["advance_prob"][stage][final["winner"]] == 1
        assert sum(result["advance_prob"][stage].values()) == 1
        group_winners.add(final["winner"])
    group_entrants = {team for match in result["trace"] if match["stage"].startswith("group_")
                      for team in (match["team_a"], match["team_b"])}
    assert len(group_entrants) == 32
    assert {match["loser"] for match in champions} <= group_entrants
    assert not ({match["winner"] for match in champions} & group_entrants)
    upper_openers = [matches[("playoffs", f"u1_{i}")] for i in range(4)]
    assert {team for match in upper_openers for team in (match["team_a"], match["team_b"])} == group_winners
    for i, champion in enumerate(champions):
        assert matches[("playoffs", f"u2_{i}")]["team_a"] == champion["winner"]
    assert sum(result["advance_prob"]["playoffs"].values()) == 2  # Finalists, not EWC eligibility.
    assert result["expected_series"] == 74
    assert sum(result["champion_prob"].values()) == 1


def test_winter_lower_rounds_preserve_cross_drops_and_round_six_series_lengths():
    result = _run("emea-2026-winter-champions-groups-playoffs", 36, 1.0)
    matches = {match["round"]: match for match in result["trace"] if match["stage"] == "playoffs"}
    for i, other in enumerate((3, 2, 1, 0)):
        assert (matches[f"l2_{i}"]["team_a"], matches[f"l2_{i}"]["team_b"]) == (
            matches[f"u2_{other}"]["loser"], matches[f"u1_{i}"]["loser"])
    assert {name for name, match in matches.items() if match["best_of"] == 5} == {
        "upper_final", "lower_final", "final"}
    assert (matches["lower_final"]["team_a"], matches["lower_final"]["team_b"]) == (
        matches["upper_final"]["loser"], matches["l5"]["winner"])
