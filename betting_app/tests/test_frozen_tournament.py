"""Behavioral contracts for pre-start, fixed-graph SERIES simulations."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest

from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from src.models.frozen_tournament import simulate_frozen_bracket, validate_bracket


def _node(node_id: str, **kwargs) -> BracketMatchNode:
    return BracketMatchNode(
        id=node_id, name=node_id, round_name=node_id, bracket_section="upper", **kwargs
    )


def _single() -> TournamentBracket:
    return TournamentBracket(
        id="research", name="Research", region="test", format="single_elimination",
        teams=["A", "A Academy", "C", "D"],
        matches={
            "title": _node("title", best_of=5),
            "semi1": _node("semi1", best_of=3, team1="A", team2="A Academy",
                           next_match_winner_id="title", next_match_winner_slot=1),
            "semi2": _node("semi2", best_of=1, team1="C", team2="D",
                           next_match_winner_id="title", next_match_winner_slot=2),
        },
    )


def _double() -> TournamentBracket:
    bracket = _single()
    bracket.format = "double_elimination"
    for node in bracket.matches.values():
        node.best_of = 3
    bracket.matches["semi1"].next_match_winner_id = "upper"
    bracket.matches["semi2"].next_match_winner_id = "upper"
    bracket.matches["semi1"].next_match_loser_id = "lower1"
    bracket.matches["semi1"].next_match_loser_slot = 1
    bracket.matches["semi2"].next_match_loser_id = "lower1"
    bracket.matches["semi2"].next_match_loser_slot = 2
    bracket.matches["upper"] = _node(
        "upper", best_of=3, next_match_winner_id="title", next_match_winner_slot=2,
        next_match_loser_id="lower2", next_match_loser_slot=1,
    )
    bracket.matches["lower1"] = _node(
        "lower1", best_of=3, next_match_winner_id="lower2", next_match_winner_slot=2,
    )
    bracket.matches["lower2"] = _node(
        "lower2", best_of=3, next_match_winner_id="title", next_match_winner_slot=1,
    )
    return bracket


def _probabilities(bracket: TournamentBracket, probability: float = 0.5) -> dict:
    return {
        (first, second, best_of): probability
        for first, second in combinations(bracket.teams, 2)
        for best_of in {node.best_of for node in bracket.matches.values()}
    }


def test_fair_single_elimination_has_analytic_reach_and_champion_probabilities() -> None:
    bracket = _single()
    order = validate_bracket(bracket)
    assert order.index("semi1") < order.index("title")
    assert order.index("semi2") < order.index("title")
    result = simulate_frozen_bracket(bracket, _probabilities(bracket), simulations=80000)
    for team in bracket.teams:
        assert result["champion_prob"][team] == pytest.approx(0.25, abs=0.008)
        assert result["final_prob"][team] == pytest.approx(0.5, abs=0.008)
    assert result["node_reach_prob"]["semi1"] == {"A": 1, "A Academy": 1, "C": 0, "D": 0}
    assert result["node_reach_prob"]["semi2"] == {"A": 0, "A Academy": 0, "C": 1, "D": 1}
    assert sum(result["champion_prob"].values()) == pytest.approx(1)
    for node_reach in result["node_reach_prob"].values():
        assert sum(node_reach.values()) == pytest.approx(2)
    assert result["final_prob"] == result["node_reach_prob"]["title"]
    assert result["championship_node_id"] == "title"
    assert result["probability_unit"] == "series_win"


@pytest.mark.parametrize("probability,champion", [(0.0, "D"), (1.0, "A")])
def test_extreme_series_probabilities_route_winners_exactly(probability: float, champion: str) -> None:
    bracket = _single()
    result = simulate_frozen_bracket(bracket, _probabilities(bracket, probability), simulations=31)
    assert result["champion_prob"] == {team: float(team == champion) for team in bracket.teams}


def test_double_elimination_rematch_and_loser_routing_are_preserved() -> None:
    bracket = _double()
    result = simulate_frozen_bracket(bracket, _probabilities(bracket, 1.0), simulations=31)
    assert result["champion_prob"] == {"A": 1, "A Academy": 0, "C": 0, "D": 0}
    assert result["node_reach_prob"]["lower1"] == {"A": 0, "A Academy": 1, "C": 0, "D": 1}
    assert result["node_reach_prob"]["lower2"] == {"A": 0, "A Academy": 1, "C": 1, "D": 0}
    assert result["node_reach_prob"]["title"] == result["node_reach_prob"]["semi1"]
    assert sum(result["champion_prob"].values()) == 1


@pytest.mark.parametrize("seed_slot", [1, 2])
def test_explicit_seed_bye_advances_without_inventing_an_opponent(seed_slot: int) -> None:
    bracket = _single()
    bracket.teams.remove("A Academy")
    bracket.matches["semi1"].team1 = "A" if seed_slot == 1 else None
    bracket.matches["semi1"].team2 = "A" if seed_slot == 2 else None
    result = simulate_frozen_bracket(bracket, _probabilities(bracket, 1.0), simulations=19)
    assert result["node_reach_prob"]["semi1"] == {"A": 1, "C": 0, "D": 0}
    assert result["final_prob"]["A"] == 1
    assert result["champion_prob"]["A"] == 1


def test_series_probability_is_used_directly_and_reverse_is_complementary() -> None:
    bracket = TournamentBracket(
        id="two", name="Two", region="test", format="single_elimination", teams=["A", "B"],
        matches={"decider": _node("decider", best_of=5, team1="B", team2="A")},
    )
    result = simulate_frozen_bracket(bracket, {("A", "B", 5): 0.25}, simulations=80000)
    assert result["champion_prob"]["A"] == pytest.approx(0.25, abs=0.008)
    assert result == simulate_frozen_bracket(bracket, {("B", "A", 5): 0.75}, simulations=80000)


def test_simulation_is_repeatable_without_mutating_inputs_or_global_rng() -> None:
    bracket = _double()
    probabilities = _probabilities(bracket, 0.375)
    original = deepcopy((bracket, probabilities))
    rng_before = np.random.get_state()
    result = simulate_frozen_bracket(bracket, probabilities, simulations=101, seed=7)
    assert result == simulate_frozen_bracket(bracket, probabilities, simulations=101, seed=7)
    assert (bracket, probabilities) == original
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    assert result["simulations"] == 101
    assert result["seed"] == 7


@pytest.mark.parametrize("fault", [
    "duplicate_team", "empty_team", "duplicate_seed", "unknown_seed", "missing_seed",
    "wrong_id", "invalid_best_of", "invalid_slot", "dangling", "self_link", "cycle",
    "slot_collision", "seed_overwrite", "empty_match", "missing_incoming_slot",
    "winner", "score1", "score2", "two_champions", "terminal_loser", "swiss", "reset_format",
    "single_loser", "bye_loser", "reset_graph",
])
def test_unsafe_graphs_are_rejected(fault: str) -> None:
    bracket = _single()
    first, second, title = (bracket.matches[key] for key in ("semi1", "semi2", "title"))
    if fault == "duplicate_team":
        bracket.teams.append("A")
    elif fault == "empty_team":
        bracket.teams.append("  ")
    elif fault == "duplicate_seed":
        second.team1 = "A"
    elif fault == "unknown_seed":
        second.team1 = "unknown"
    elif fault == "missing_seed":
        bracket.teams.append("unseeded")
    elif fault == "wrong_id":
        first.id = "other"
    elif fault == "invalid_best_of":
        first.best_of = 2
    elif fault == "invalid_slot":
        first.next_match_winner_slot = 3
    elif fault == "dangling":
        first.next_match_winner_id = "missing"
    elif fault == "self_link":
        first.next_match_winner_id = "semi1"
    elif fault == "cycle":
        first.team1 = None
        title.next_match_winner_id = "semi1"
        title.next_match_winner_slot = 1
    elif fault == "slot_collision":
        second.next_match_winner_slot = 1
    elif fault == "seed_overwrite":
        title.team1 = "A"
    elif fault == "empty_match":
        bracket.matches["empty"] = _node("empty")
    elif fault == "missing_incoming_slot":
        first.next_match_winner_id = None
    elif fault in {"winner", "score1", "score2"}:
        setattr(first, fault, "A" if fault == "winner" else 0)
    elif fault == "two_champions":
        first.next_match_winner_id = None
    elif fault == "terminal_loser":
        title.next_match_loser_id = "semi1"
    elif fault == "swiss":
        bracket.format = "swiss"
    elif fault == "reset_format":
        bracket.format = "double_elimination_reset"
    elif fault == "single_loser":
        first.next_match_loser_id = "semi2"
    elif fault == "bye_loser":
        bracket = _double()
        bracket.matches["semi1"].team2 = None
        bracket.teams.remove("A Academy")
    elif fault == "reset_graph":
        bracket = _double()
        title = bracket.matches["title"]
        title.next_match_winner_id = "reset"
        title.next_match_winner_slot = 1
        title.next_match_loser_id = "reset"
        title.next_match_loser_slot = 2
        bracket.matches["reset"] = _node("reset")
    with pytest.raises(ValueError):
        validate_bracket(bracket)


def test_unseeded_one_sided_match_is_not_a_bye() -> None:
    bracket = _single()
    bracket.matches["extra"] = _node("extra", next_match_winner_id="title")
    bracket.matches["semi1"].next_match_winner_id = "extra"
    with pytest.raises(ValueError):
        validate_bracket(bracket)


@pytest.mark.parametrize("fault", ["missing", "nan", "infinity", "negative", "above_one", "reverse", "unknown", "self", "format"])
def test_invalid_or_incomplete_probability_tables_are_rejected(fault: str) -> None:
    bracket = _single()
    probabilities = _probabilities(bracket)
    key = ("A", "A Academy", 3)
    if fault == "missing":
        del probabilities[key]
    elif fault in {"nan", "infinity", "negative", "above_one"}:
        probabilities[key] = {"nan": float("nan"), "infinity": float("inf"), "negative": -0.1, "above_one": 1.1}[fault]
    elif fault == "reverse":
        probabilities[("A Academy", "A", 3)] = 0.5
    elif fault == "unknown":
        probabilities[("A", "unknown", 3)] = 0.5
    elif fault == "self":
        probabilities[("A", "A", 3)] = 0.5
    elif fault == "format":
        probabilities[("A", "A Academy", 7)] = 0.5
    with pytest.raises(ValueError):
        simulate_frozen_bracket(bracket, probabilities, simulations=10)


@pytest.mark.parametrize("simulations", [0, -1, 1.5, True])
def test_invalid_simulation_count_is_rejected(simulations) -> None:
    bracket = _single()
    with pytest.raises(ValueError):
        simulate_frozen_bracket(bracket, _probabilities(bracket), simulations=simulations)


def test_zero_strength_draws_preserve_exact_fixed_probability_results():
    bracket = _double()
    probabilities = _probabilities(bracket, 0.37)
    draws = {team: np.zeros(101) for team in bracket.teams}
    assert simulate_frozen_bracket(bracket, probabilities, 101, 7) == simulate_frozen_bracket(
        bracket, probabilities, 101, 7, team_strength_draws=draws,
    )


def test_strength_is_shared_across_rounds_not_redrawn_per_match():
    bracket = _single()
    simulations = 1000
    draws = {team: np.zeros(simulations) for team in bracket.teams}
    # A is effectively unbeatable in half of the tournament worlds and cannot
    # win its opener in the other half. Independent round noise would yield 1/4.
    draws["A"][:500] = 100
    draws["A"][500:] = -100
    before = deepcopy(draws)
    result = simulate_frozen_bracket(
        bracket, _probabilities(bracket), simulations, 82, team_strength_draws=draws,
    )
    assert result["champion_prob"]["A"] == 0.5
    assert result["final_prob"]["A"] == 0.5
    assert sum(result["champion_prob"].values()) == pytest.approx(1)
    for reach in result["node_reach_prob"].values():
        assert sum(reach.values()) == pytest.approx(2)
    for team in draws:
        np.testing.assert_array_equal(draws[team], before[team])


def test_uncertain_series_logits_are_not_binomially_expanded():
    bracket = TournamentBracket(
        id="two", name="Two", region="test", format="single_elimination", teams=["A", "B"],
        matches={"decider": _node("decider", best_of=5, team1="A", team2="B")},
    )
    draws = {"A": np.full(80000, np.log(2)), "B": np.zeros(80000)}
    result = simulate_frozen_bracket(bracket, {("A", "B", 5): 0.25}, 80000, 82, team_strength_draws=draws)
    assert result["champion_prob"]["A"] == pytest.approx(0.4, abs=0.008)
    assert result == simulate_frozen_bracket(bracket, {("B", "A", 5): 0.75}, 80000, 82, team_strength_draws=draws)


def test_common_strength_shift_and_endpoints_preserve_outcomes_and_global_rng():
    bracket = _double()
    draws = {team: np.full(101, index - 1.5) for index, team in enumerate(bracket.teams)}
    shifted = {team: values + 17 for team, values in draws.items()}
    before = np.random.get_state()
    for probability in (0, 0.375, 1):
        table = _probabilities(bracket, probability)
        result = simulate_frozen_bracket(bracket, table, 101, 7, team_strength_draws=draws)
        assert result == simulate_frozen_bracket(bracket, table, 101, 7, team_strength_draws=shifted)
        if probability in (0, 1):
            assert result == simulate_frozen_bracket(bracket, table, 101, 7)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]


@pytest.mark.parametrize("fault", ["missing", "extra", "length", "matrix", "nan", "infinite", "string", "boolean"])
def test_strength_draw_boundaries_are_validated(fault):
    bracket = _single()
    draws = {team: np.zeros(10) for team in bracket.teams}
    if fault == "missing": del draws["A"]
    elif fault == "extra": draws["unknown"] = np.zeros(10)
    elif fault == "length": draws["A"] = np.zeros(9)
    elif fault == "matrix": draws["A"] = np.zeros((10, 1))
    elif fault == "nan": draws["A"][0] = np.nan
    elif fault == "infinite": draws["A"][0] = np.inf
    elif fault == "string": draws["A"] = ["0"] * 10
    elif fault == "boolean": draws["A"] = [False] * 10
    with pytest.raises(ValueError):
        simulate_frozen_bracket(bracket, _probabilities(bracket), 10, team_strength_draws=draws)
