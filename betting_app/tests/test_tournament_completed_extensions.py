"""Dynamic mechanics needed by completed league editions, without historic results."""
from collections import Counter
from copy import deepcopy
from itertools import combinations

import pytest

from src.models.tournament_formats import simulate_tournament


def _probabilities(teams, bo=1):
    return {(a, b, bo): 1.0 for a, b in combinations(teams, 2)}


def test_fixed_round_swiss_keeps_every_team_playing_and_orders_terminal_records():
    teams = list("ABCDEFGH")
    spec = {"version": 1, "id": "fixed-swiss", "teams": teams, "stages": [{
        "id": "swiss", "kind": "swiss", "entrants": teams, "best_of": 1,
        "rounds": 3, "no_rematches": True, "tiebreakers": ["wins", "seed"], "advance": 4,
    }]}
    result = simulate_tournament(spec, _probabilities(teams), simulations=1, seed=81)
    assert result["expected_series"] == 12
    assert Counter((r["wins"], r["losses"]) for r in result["stage_records"]["swiss"].values()) == {
        (3, 0): 1, (2, 1): 3, (1, 2): 3, (0, 3): 1,
    }
    for number in (1, 2, 3):
        matches = [m for m in result["trace"] if m["round"] == number]
        assert Counter(t for m in matches for t in (m["team_a"], m["team_b"])) == Counter(teams)
    assert len({frozenset((m["team_a"], m["team_b"])) for m in result["trace"]}) == 12
    for team, record in result["stage_records"]["swiss"].items():
        assert result["advance_prob"]["swiss"][team] == int(record["wins"] >= 2)
    spec["stages"][0]["wins_to_advance"] = 2
    with pytest.raises(ValueError, match="rounds|threshold"):
        simulate_tournament(spec, _probabilities(teams), simulations=1)


def _seeded_graph():
    pool = [{"loser": "u1"}, {"loser": "u2"}]
    return {"version": 1, "id": "seeded-supply", "teams": list("ABCDEF"), "stages": [{
        "id": "bracket", "kind": "graph", "entrants": list("FEDCBA"), "best_of": 1,
        "seed_order": "tournament",
        "matches": [
            {"id": "u1", "a": "A", "b": "D"},
            {"id": "u2", "a": "B", "b": "C"},
            {"id": "l1", "a": {"seeded": pool, "rank": 2}, "b": "E"},
            {"id": "l2", "a": {"winner": "l1"}, "b": "F"},
            {"id": "l3", "a": {"seeded": list(reversed(pool)), "rank": 1}, "b": {"winner": "l2"}},
            {"id": "uf", "a": {"winner": "u1"}, "b": {"winner": "u2"}},
            {"id": "final", "a": {"winner": "uf"}, "b": {"winner": "l3"}},
        ],
        "ranking": [{"winner": "final"}, {"loser": "final"}, {"loser": "uf"},
                    {"loser": "l3"}, {"loser": "l2"}, {"loser": "l1"}],
    }], "champion": {"stage": "bracket", "group": "all", "rank": 1}}


@pytest.mark.parametrize("seed_order,first_lower_entrant", [("tournament", "C"), ("stage", "A")])
def test_upper_losers_enter_lower_rounds_by_selected_seed_order(seed_order, first_lower_entrant):
    spec = _seeded_graph()
    spec["stages"][0]["seed_order"] = seed_order
    probabilities = _probabilities(spec["teams"])
    probabilities["A", "D", 1] = 0.0
    result = simulate_tournament(spec, probabilities, simulations=1)
    lower = next(m for m in result["trace"] if m["round"] == "l1")
    assert (lower["team_a"], lower["team_b"]) == (first_lower_entrant, "E")
    assert result["expected_series"] == 7
    assert sum(result["champion_prob"].values()) == 1


@pytest.mark.parametrize("duplicate_source", [True, False])
def test_seeded_supply_cannot_be_consumed_again_as_raw_source_or_rank(duplicate_source):
    spec = _seeded_graph()
    spec["stages"][0]["matches"][4]["a"] = (
        {"loser": "u1"} if duplicate_source else deepcopy(spec["stages"][0]["matches"][2]["a"])
    )
    with pytest.raises(ValueError, match="consum|suppl|reus"):
        simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)


def test_pick_and_play_uses_completed_round_records_not_future_results_or_round_robin():
    teams = list("ABCDEFGH")
    spec = {"version": 1, "id": "pick-play", "teams": teams, "stages": [{
        "id": "weeks", "kind": "pick_and_play", "entrants": teams, "best_of": 1,
        "rounds": 3, "initial_pairings": [{"a": a, "b": b} for a, b in zip(teams[:4], reversed(teams[4:]))],
        "choice_policy": "nearest_better_record", "no_rematches": False,
        "tiebreakers": ["wins", "seed"], "advance": 8,
    }]}
    probabilities = _probabilities(teams)
    probabilities["A", "H", 1] = 0.0
    result = simulate_tournament(spec, probabilities, simulations=1)
    second_round = [m for m in result["trace"] if m["round"] == 2]
    assert [(m["team_a"], m["team_b"]) for m in second_round] == [("G", "H"), ("F", "D"), ("E", "C"), ("A", "B")]
    assert result["expected_series"] == 12
    assert all(r["wins"] + r["losses"] == 3 for r in result["stage_records"]["weeks"].values())
    assert all(p == 1 for p in result["advance_prob"]["weeks"].values())
    spec["stages"][0]["initial_pairings"][1]["a"] = "A"
    with pytest.raises(ValueError, match="pair|once|duplicate"):
        simulate_tournament(spec, probabilities, simulations=1)
