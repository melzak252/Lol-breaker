"""Counterfactual qualification and complete-series probability contracts."""
from itertools import combinations

import pytest

from src.models.tournament_formats import simulate_tournament


def test_simulated_standings_supply_finalists_without_observed_results():
    teams = list("ABCD")
    spec = {
        "version": 1, "id": "league-final", "teams": teams,
        "stages": [
            {"id": "regular", "kind": "round_robin", "entrants": teams,
             "cycles": 2, "best_of": 1, "advance": 2,
             "tiebreakers": ["wins", "head_to_head", "seed"]},
            {"id": "final", "kind": "single_elimination", "best_of": 3,
             "entrants": [{"stage": "regular", "group": "all", "rank": n} for n in (1, 2)]},
        ],
        "champion": {"stage": "final", "group": "all", "rank": 1},
    }
    probabilities = {(a, b, bo): 1.0 for a, b in combinations(teams, 2) for bo in (1, 3)}
    result = simulate_tournament(spec, probabilities, simulations=10, seed=81)
    assert result["champion_prob"] == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    assert result["stage_rank_prob"]["regular"]["all"]["B"] == [0.0, 1.0, 0.0, 0.0]
    assert result["advance_prob"]["regular"] == {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}
    assert result["expected_series"] == 13.0


def test_bo5_probability_is_not_expanded_as_a_map_probability():
    spec = {
        "version": 1, "id": "final", "teams": ["A", "B"],
        "stages": [{"id": "final", "kind": "single_elimination", "entrants": ["A", "B"], "best_of": 5}],
        "champion": {"stage": "final", "group": "all", "rank": 1},
    }
    result = simulate_tournament(spec, {("A", "B", 5): 0.7}, simulations=20000, seed=81)
    assert result["champion_prob"]["A"] == pytest.approx(0.7, abs=0.015)


def test_future_map_distribution_uses_score_pmf_not_only_first_rollout():
    spec = {
        "version": 1, "id": "final", "teams": ["A", "B"],
        "stages": [{"id": "final", "kind": "single_elimination",
                    "entrants": ["A", "B"], "best_of": 3}],
        "champion": {"stage": "final", "group": "all", "rank": 1},
    }
    result = simulate_tournament(
        spec, {("A", "B", 3): 0.7}, simulations=10000, seed=81,
        score_distributions={3: [0.25, 0.75]},
        observables=[
            {"id": "future-maps", "kind": "future_map_count", "categories": ["2", "3"]},
            {"id": "future-series", "kind": "future_series_count", "categories": ["1"]},
        ],
    )
    assert result["observable_prob"]["future-series"] == {"1": 1.0}
    assert result["observable_prob"]["future-maps"] == pytest.approx({"2": 0.25, "3": 0.75}, abs=0.02)
    assert sum(result["hits"]["observable"]["future-maps"].values()) == 10000
    assert result["champion_prob"]["A"] == pytest.approx(0.7, abs=0.02)


def _spec(kind, teams="ABCD", **rules):
    return {
        "version": 1, "id": kind, "teams": list(teams),
        "stages": [{"id": "event", "kind": kind, "entrants": list(teams),
                    "best_of": 1, **rules}],
        "champion": {"stage": "event", "group": "all", "rank": 1},
    }


def _probabilities(teams, formats=(1, 3, 5), probability=1.0):
    return {(a, b, bo): probability for a, b in combinations(teams, 2) for bo in formats}


@pytest.mark.parametrize("kind,rules,series", [
    ("single_elimination", {}, 3),
    ("double_elimination", {"final_reset": "if_lower_wins"}, 6),
    ("gauntlet", {}, 3),
    ("gsl", {}, 5),
    ("round_robin", {"tiebreakers": ["wins", "seed"]}, 6),
])
def test_formats_conserve_entrants_and_champion(kind, rules, series):
    spec = _spec(kind, **rules)
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=2)
    assert result["champion_prob"]["A"] == 1
    ranks = result["stage_rank_prob"]["event"]["all"]
    assert all(sum(vector) == 1 for vector in ranks.values())
    assert all(sum(vector[i] for vector in ranks.values()) == 1 for i in range(4))
    assert result["expected_series"] == series


def test_swiss_records_stop_at_threshold_and_never_repeat_pair():
    spec = _spec("swiss", teams="ABCDEFGH", wins_to_advance=2,
                 losses_to_eliminate=2, tiebreakers=["wins", "seed"], advance=4)
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    records = result["stage_records"]["event"]
    assert all(record["wins"] == 2 or record["losses"] == 2 for record in records.values())
    pairs = [frozenset((m["team_a"], m["team_b"])) for m in result["trace"]]
    assert len(pairs) == len(set(pairs))
    assert sum(result["advance_prob"]["event"].values()) == 4
    assert all(m["best_of"] == 3 for m in result["trace"] if m["round"] > 1)


def test_top_seed_receives_bye_without_fake_series():
    spec = _spec("single_elimination", teams="ABC")
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    assert result["expected_series"] == 2
    assert {result["trace"][0]["team_a"], result["trace"][0]["team_b"]} == {"B", "C"}
    assert result["trace"][1]["winner"] == "A"


def test_graph_routes_each_loser_once_and_rejects_duplicate_token():
    spec = _spec("graph", matches=[
        {"id": "s1", "a": "A", "b": "D"},
        {"id": "s2", "a": "B", "b": "C"},
        {"id": "bronze", "a": {"loser": "s1"}, "b": {"loser": "s2"}},
        {"id": "gold", "a": {"winner": "s1"}, "b": {"winner": "s2"}},
    ], ranking=[{"winner": "gold"}, {"loser": "gold"},
                {"winner": "bronze"}, {"loser": "bronze"}])
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    assert result["stage_rank_prob"]["event"]["all"]["C"] == [0, 0, 1, 0]
    spec["stages"][0]["matches"][2]["a"] = {"winner": "s1"}
    with pytest.raises(ValueError, match="consum|reuse|suppl"):
        simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)


@pytest.mark.parametrize("policy,expected", [("all", 3), ("qualified_only", 2)])
def test_carryover_counts_only_selected_history_without_replaying(policy, expected):
    spec = _spec("round_robin", tiebreakers=["wins", "seed"])
    spec["stages"].append({
        "id": "second", "kind": "round_robin", "best_of": 1,
        "entrants": [{"stage": "event", "group": "all", "rank": rank} for rank in (2, 3)],
        "carry_from": "event", "carry_policy": policy, "tiebreakers": ["wins", "seed"],
    })
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    assert result["stage_records"]["second"]["B"]["wins"] == expected
    assert result["expected_series"] == 7


def test_map_tiebreaks_require_separate_score_model():
    spec = _spec("round_robin", teams="ABC", tiebreakers=["wins", "map_diff", "seed"])
    spec["stages"][0]["best_of"] = 3
    p = {("A", "B", 3): 1, ("A", "C", 3): 0, ("B", "C", 3): 1}
    with pytest.raises(ValueError, match="score"):
        simulate_tournament(spec, p, simulations=1)
    result = simulate_tournament(spec, p, simulations=1, score_distributions={3: [1, 0]})
    assert result["champion_prob"]["A"] == 1
    assert all(sorted((m["score_a"], m["score_b"])) == [0, 2] for m in result["trace"])


def test_uniform_draw_finds_completion_and_fails_impossible_constraints():
    spec = _spec("single_elimination", draw={
        "policy": "uniform", "avoid_same": ["region"],
        "labels": {"A": {"region": "x"}, "B": {"region": "x"},
                   "C": {"region": "y"}, "D": {"region": "z"}},
    })
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1, seed=11)
    first = [m for m in result["trace"] if m["round"] == 1]
    assert len(first) == 2
    assert all({m["team_a"], m["team_b"]} != {"A", "B"} for m in first)
    spec["stages"][0]["draw"]["labels"]["C"]["region"] = "x"
    with pytest.raises(ValueError, match="feasible|impossible"):
        simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)


@pytest.mark.parametrize("mutation", ["unknown", "observed", "forward", "tie"])
def test_invalid_specs_rejected_before_simulation(mutation):
    spec = _spec("round_robin", tiebreakers=["wins", "seed"])
    if mutation == "unknown":
        spec["stages"][0]["mystery_rule"] = True
    elif mutation == "observed":
        spec["stages"][0]["winner"] = "A"
    elif mutation == "forward":
        spec["stages"][0]["entrants"][0] = {"stage": "later", "group": "all", "rank": 1}
    else:
        spec["stages"][0]["tiebreakers"] = ["wins"]
    with pytest.raises(ValueError):
        simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)


def test_seed_reproducibility_and_input_is_not_mutated():
    from copy import deepcopy

    spec = _spec("single_elimination", draw={"policy": "uniform"})
    snapshot = deepcopy(spec)
    probabilities = _probabilities(spec["teams"], probability=0.5)
    first = simulate_tournament(spec, probabilities, simulations=25, seed=18)
    assert first == simulate_tournament(spec, probabilities, simulations=25, seed=18)
    assert spec == snapshot


def test_weighted_group_total_dynamically_selects_winning_group_qualifier():
    spec = {
        "version": 1, "id": "weighted-cup", "teams": list("ABCD"),
        "stages": [
            {"id": "cup", "kind": "round_robin", "best_of": 1,
             "groups": {"one": ["A", "B"], "two": ["C", "D"]},
             "tiebreakers": ["wins", "seed"], "group_tiebreakers": ["wins", "seed"],
             "schedule": [{"a": "A", "b": "C", "weight": 1},
                          {"a": "B", "b": "D", "best_of": 3, "weight": 2}]},
            {"id": "qualified", "kind": "single_elimination", "best_of": 1,
             "entrants": [{"stage": "cup", "group_rank": 1, "rank": 1}]}],
        "champion": None,
    }
    probabilities = _probabilities(spec["teams"])
    probabilities["B", "D", 3] = 0
    result = simulate_tournament(spec, probabilities, simulations=1)
    assert result["champion_prob"] is None
    assert result["stage_rank_prob"]["qualified"]["all"]["D"] == [1]
    assert result["stage_records"]["cup"]["D"]["wins"] == 2
    assert result["expected_series"] == 2


def test_conditional_reset_gives_upper_finalist_second_life_only_when_required():
    spec = _spec("double_elimination", teams="AB", final_reset="if_lower_wins")
    p = {("A", "B", 1): 0.5}
    result = simulate_tournament(spec, p, simulations=1, seed=1)
    assert [match["winner"] for match in result["trace"]] == ["A", "B", "B"]
    assert result["champion_prob"]["B"] == 1
    spec["stages"][0]["final_reset"] = "single_final"
    result = simulate_tournament(spec, p, simulations=1, seed=1)
    assert [match["winner"] for match in result["trace"]] == ["A", "B"]


def test_swiss_terminal_record_priority_precedes_random_tie_resolution():
    spec = _spec("swiss", teams="ABCDEFGHIJKLMNOP", wins_to_advance=3,
                 losses_to_eliminate=3, tiebreakers=["random"], advance=8)
    result = simulate_tournament(spec, _probabilities(spec["teams"], probability=0.5),
                                 simulations=1, seed=81)
    ranked = sorted(spec["teams"], key=lambda t: result["stage_rank_prob"]["event"]["all"][t].index(1))
    records = result["stage_records"]["event"]
    assert [(records[t]["wins"], records[t]["losses"]) for t in ranked[:8]] == (
        [(3, 0)] * 2 + [(3, 1)] * 3 + [(3, 2)] * 3)
    assert [(records[t]["wins"], records[t]["losses"]) for t in ranked[8:]] == (
        [(2, 3)] * 3 + [(1, 3)] * 3 + [(0, 3)] * 2)


def test_eliminated_graph_token_cannot_reenter_or_lose_bye():
    spec = _spec("graph", teams="ABC", matches=[
        {"id": "bye", "a": "A", "b": None},
        {"id": "semi", "a": "B", "b": "C"},
        {"id": "final", "a": {"winner": "bye"}, "b": {"winner": "semi"}},
    ], ranking=[{"winner": "final"}, {"loser": "final"}, {"loser": "semi"}])
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    assert result["expected_series"] == 2
    spec["stages"][0]["ranking"][-1] = {"loser": "bye"}
    with pytest.raises(ValueError, match="suppl"):
        simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)


def test_aliasing_literal_and_dynamic_entrant_is_rejected_before_play():
    spec = _spec("round_robin", teams="AB", tiebreakers=["wins", "seed"])
    spec["stages"].append({
        "id": "final", "kind": "single_elimination", "best_of": 1,
        "entrants": ["A", {"stage": "event", "group": "all", "rank": 1}]})
    with pytest.raises(ValueError, match="overlap|duplicate"):
        simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)


def test_head_to_head_resolves_equal_wins_before_seed():
    spec = _spec("round_robin", tiebreakers=["wins", "head_to_head", "seed"])
    p = _probabilities(spec["teams"])
    p["A", "B", 1] = 0
    p["B", "D", 1] = 0
    result = simulate_tournament(spec, p, simulations=1)
    assert result["stage_records"]["event"]["A"]["wins"] == 2
    assert result["stage_records"]["event"]["B"]["wins"] == 2
    assert result["champion_prob"]["B"] == 1


def test_independent_map_scores_change_tied_series_standings():
    spec = _spec("round_robin", teams="ABC", tiebreakers=["wins", "map_diff", "seed"],
                 schedule=[{"a": "A", "b": "B"},
                           {"a": "B", "b": "C", "best_of": 3},
                           {"a": "C", "b": "A", "best_of": 3}])
    p = _probabilities(spec["teams"])
    p["A", "C", 3] = 0
    result = simulate_tournament(spec, p, simulations=1, score_distributions={3: [1, 0], 5: [1, 0, 0]})
    assert result["champion_prob"]["B"] == 1


def test_opponent_choice_is_explicit_not_a_hidden_random_draw():
    spec = _spec("single_elimination", opponent_choice="highest_seed")
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    assert [(m["team_a"], m["team_b"]) for m in result["trace"][:2]] == [("A", "B"), ("C", "D")]
    spec["stages"][0]["opponent_choice"] = "lowest_seed"
    result = simulate_tournament(spec, _probabilities(spec["teams"]), simulations=1)
    assert [(m["team_a"], m["team_b"]) for m in result["trace"][:2]] == [("A", "D"), ("B", "C")]


def test_thirty_two_team_swiss_completes_with_terminal_records_and_no_rematches():
    teams = [f"Team{i:02d}" for i in range(32)]
    spec = _spec("swiss", teams=teams, wins_to_advance=3, losses_to_eliminate=3,
                 tiebreakers=["seed"], advance=16)
    result = simulate_tournament(spec, _probabilities(teams, probability=0.5),
                                 simulations=1, seed=81)
    assert result["expected_series"] == 66
    assert sum(result["advance_prob"]["event"].values()) == 16
    records = result["stage_records"]["event"]
    assert all(record["wins"] == 3 or record["losses"] == 3 for record in records.values())
    assert sum(record["wins"] == 3 for record in records.values()) == 16
    pairs = [frozenset((match["team_a"], match["team_b"])) for match in result["trace"]]
    assert len(set(pairs)) == 66
    ranked = sorted(teams, key=lambda team: result["stage_rank_prob"]["event"]["all"][team].index(1))
    assert [(records[team]["wins"], records[team]["losses"]) for team in ranked[:16]] == (
        [(3, 0)] * 4 + [(3, 1)] * 6 + [(3, 2)] * 6)


def test_named_final_joint_oracle_and_unsampled_support_are_not_marginal_products():
    event = _spec("single_elimination")
    pairs = ['["A","B"]', '["A","C"]', '["B","D"]', '["C","D"]']
    observable = {"id": "final-pair", "kind": "node_matchup", "categories": pairs,
                  "node": {"stage": "event", "round": 2}}
    result = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                                 simulations=12000, observables=[observable])
    assert result["observable_prob"]["final-pair"] == pytest.approx(dict.fromkeys(pairs, 0.25), abs=0.02)
    assert result["observable_prob"]["final-pair"] == result["joint_final_prob"]
    for team in event["teams"]:
        joint_marginal = sum(p for pair, p in result["joint_final_prob"].items() if f'"{team}"' in pair)
        assert joint_marginal == pytest.approx(result["final_prob"][team], abs=1e-12)
    tiny = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                               simulations=1, observables=[observable])
    assert sum(value == 0 for value in tiny["hits"]["observable"]["final-pair"].values()) == 3
    assert tiny["structural_zero_categories"]["observable"]["final-pair"] == []


def test_oriented_node_scores_preserve_direct_winner_probability_when_sides_reverse():
    event = _spec("single_elimination", teams="BA")
    event["stages"][0]["best_of"] = 3
    result = simulate_tournament(
        event, {("A", "B", 3): 0.7}, simulations=16000, score_distributions={3: [0.25, 0.75]},
        observables=[{"id": "score", "kind": "node_score", "participants": ["A", "B"],
                      "node": {"stage": "event", "round": 1},
                      "categories": ["2-0", "2-1", "0-2", "1-2"]}])
    scores = result["observable_prob"]["score"]
    assert scores == pytest.approx({"2-0": 0.175, "2-1": 0.525, "0-2": 0.075, "1-2": 0.225}, abs=0.02)
    assert scores["2-0"] + scores["2-1"] == pytest.approx(result["champion_prob"]["A"])


def test_double_elimination_reset_has_legal_absence_and_exact_count_distribution():
    event = _spec("double_elimination", teams="AB", final_reset="if_lower_wins")
    result = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                                 simulations=12000, observables=[
        {"id": "series", "kind": "future_series_count", "categories": ["2", "3"]},
        {"id": "repeats", "kind": "repeat_encounter_count", "categories": ["1", "2"]},
        {"id": "reset", "kind": "node_matchup", "categories": ['["A","B"]', "absent"],
         "absent_category": "absent", "node": {"stage": "event", "round": "grand-final-reset"}},
    ])
    assert result["observable_prob"]["series"] == pytest.approx({"2": 0.5, "3": 0.5}, abs=0.02)
    assert result["observable_prob"]["repeats"] == {
        "1": result["observable_prob"]["series"]["2"], "2": result["observable_prob"]["series"]["3"]}
    assert result["observable_prob"]["reset"] == {
        '["A","B"]': result["observable_prob"]["series"]["3"],
        "absent": result["observable_prob"]["series"]["2"]}


def test_small_round_robin_playoff_oracle_counts_each_real_series_once():
    event = _spec("round_robin", teams="ABC", tiebreakers=["wins", "seed"], advance=2)
    event["stages"].append({"id": "final", "kind": "single_elimination", "best_of": 1,
                            "entrants": [{"stage": "event", "group": "all", "rank": rank} for rank in (1, 2)]})
    event["champion"] = {"stage": "final", "group": "all", "rank": 1}
    result = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                                 simulations=16000, observables=[
        {"id": "series", "kind": "future_series_count", "categories": ["4"]},
        {"id": "repeat", "kind": "repeat_encounter_count", "categories": ["1"]},
        {"id": "final", "kind": "node_matchup", "node": {"stage": "final", "round": 1},
         "categories": ['["A","B"]', '["A","C"]', '["B","C"]']},
    ])
    # Six transitive orientations contribute two each; the two cycles use seed ties.
    assert result["observable_prob"]["final"] == pytest.approx(
        {'["A","B"]': 0.5, '["A","C"]': 0.25, '["B","C"]': 0.25}, abs=0.02)
    assert result["observable_prob"]["series"] == {"4": 1}
    assert result["observable_prob"]["repeat"] == {"1": 1}
    assert result["joint_advance_prob"]["event"] == result["observable_prob"]["final"]


def test_constrained_swiss_has_exact_series_and_zero_repeat_counts():
    event = _spec("swiss", teams="ABCD", rounds=2, no_rematches=True,
                  tiebreakers=["wins", "seed"], advance=2,
                  draw={"policy": "uniform", "avoid_same": ["pool"],
                        "labels": {"A": {"pool": "x"}, "B": {"pool": "x"},
                                   "C": {"pool": "y"}, "D": {"pool": "y"}}})
    result = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                                 simulations=100, observables=[
        {"id": "series", "kind": "future_series_count", "categories": ["4"]},
        {"id": "repeat", "kind": "repeat_encounter_count", "categories": ["0"]},
    ])
    assert result["observable_prob"] == {"series": {"4": 1}, "repeat": {"0": 1}}
    assert sum(result["advance_prob"]["event"].values()) == pytest.approx(2)


def test_fixed_reference_upsets_do_not_use_candidate_probability_or_tie_as_upset():
    event = _spec("single_elimination", teams="AB")
    observable = {"id": "upsets", "kind": "upset_count", "categories": ["0", "1"],
                  "reference_probabilities": {("B", "A", 1): 0.8}}
    result = simulate_tournament(event, {("A", "B", 1): 1}, simulations=3, observables=[observable])
    assert result["observable_prob"]["upsets"] == {"0": 0, "1": 1}
    observable["reference_probabilities"] = {("A", "B", 1): 0.5}
    result = simulate_tournament(event, {("A", "B", 1): 1}, simulations=3, observables=[observable])
    assert result["observable_prob"]["upsets"] == {"0": 1, "1": 0}


@pytest.mark.parametrize("observable", [
    {"id": "x", "kind": "future_map_count", "categories": ["2", "3"]},
    {"id": "x", "kind": "node_score", "categories": ["2-0", "2-1", "0-2", "1-2"],
     "participants": ["A", "B"], "node": {"stage": "event", "round": 1}},
])
def test_claimed_score_forecasts_require_separate_score_pmf(observable):
    event = _spec("single_elimination", teams="AB")
    event["stages"][0]["best_of"] = 3
    with pytest.raises(ValueError, match="score"):
        simulate_tournament(event, {("A", "B", 3): 0.7}, simulations=1, observables=[observable])


@pytest.mark.parametrize("observables", [
    [{"id": "x", "kind": "future_series_count", "categories": ["1", "1"]}],
    [{"id": "x", "kind": "future_series_count", "categories": ["01"]}],
    [{"id": "x", "kind": "unknown", "categories": ["1"]}],
    [{"id": "x", "kind": "upset_count", "categories": ["0", "1"]}],
    [{"id": "x", "kind": "future_series_count", "categories": ["1"], "scope": {"stage": "missing"}}],
    [{"id": "x", "kind": "node_matchup", "categories": ['["A","B"]'],
      "node": {"stage": "event", "round": 99}}],
    [{"id": "x", "kind": "future_series_count", "categories": ["1"]}] * 2,
])
def test_invalid_observable_schema_fails_closed(observables):
    with pytest.raises(ValueError, match="observable|Observable|reference"):
        simulate_tournament(_spec("single_elimination", teams="AB"), {("A", "B", 1): 0.5},
                            simulations=1, observables=observables)


def test_source_declared_champion_path_oracle_distinguishes_lower_bracket_route():
    event = _spec("double_elimination", teams="AB", final_reset="if_lower_wins")
    upper = {"stage": "event", "round": "upper-1"}
    paths = {"upper": [{"node": upper, "outcome": "winner"}],
             "lower": [{"node": upper, "outcome": "loser"}]}
    observable = {"id": "path", "kind": "champion_path", "categories": ["upper", "lower"],
                  "source": "synthetic://two-team-double-elimination/explicit-upper-node", "paths": paths}
    result = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                                 simulations=12000, observables=[observable])
    assert result["observable_prob"]["path"] == pytest.approx({"upper": 0.75, "lower": 0.25}, abs=0.02)
    del observable["source"]
    with pytest.raises(ValueError, match="source"):
        simulate_tournament(event, _probabilities(event["teams"]), simulations=1, observables=[observable])


def test_overlapping_source_declared_paths_are_rejected_instead_of_double_counted():
    event = _spec("double_elimination", teams="AB", final_reset="single_final")
    predicate = {"node": {"stage": "event", "round": "grand-final"}, "outcome": "winner"}
    with pytest.raises(ValueError, match="partition"):
        simulate_tournament(event, _probabilities(event["teams"]), simulations=1, observables=[
            {"id": "path", "kind": "champion_path", "categories": ["upper", "lower"],
             "source": "synthetic://invalid-overlapping-paths",
             "paths": {"upper": [predicate], "lower": [predicate]}}])


def test_graph_bye_node_is_absent_not_an_invented_played_match():
    event = _spec("graph", teams="ABC", matches=[
        {"id": "bye", "a": "A", "b": None},
        {"id": "semi", "a": "B", "b": "C"},
        {"id": "final", "a": {"winner": "bye"}, "b": {"winner": "semi"}},
    ], ranking=[{"winner": "final"}, {"loser": "final"}, {"loser": "semi"}])
    result = simulate_tournament(event, _probabilities(event["teams"]), simulations=3, observables=[
        {"id": "bye", "kind": "node_matchup", "node": {"stage": "event", "round": "bye"},
         "categories": ["absent"], "absent_category": "absent"},
        {"id": "series", "kind": "future_series_count", "categories": ["2"]},
    ])
    assert result["observable_prob"] == {"bye": {"absent": 1}, "series": {"2": 1}}
    assert result["champion_prob"] == {"A": 1, "B": 0, "C": 0}


def test_requested_score_matchup_retains_nonreach_as_explicit_absence():
    event = _spec("single_elimination")
    result = simulate_tournament(event, _probabilities(event["teams"], probability=0.5),
                                 simulations=12000, observables=[
        {"id": "score", "kind": "node_score", "participants": ["A", "B"],
         "node": {"stage": "event", "round": 2}, "categories": ["1-0", "0-1", "absent"],
         "absent_category": "absent"}])
    assert result["observable_prob"]["score"] == pytest.approx(
        {"1-0": 0.125, "0-1": 0.125, "absent": 0.75}, abs=0.02)


def test_online_observables_do_not_change_existing_probability_stream():
    event = _spec("double_elimination", final_reset="if_lower_wins")
    p = _probabilities(event["teams"], probability=0.5)
    original = simulate_tournament(event, p, simulations=20, seed=71)
    observed = simulate_tournament(event, p, simulations=20, seed=71, observables=[
        {"id": "series", "kind": "future_series_count", "categories": ["6", "7"]}])
    for key in original:
        if key not in {"hits", "structural_zero_categories"}:
            assert observed[key] == original[key]
    assert {k: v for k, v in observed["hits"].items() if k != "observable"} == original["hits"]
    assert {k: v for k, v in observed["structural_zero_categories"].items() if k != "observable"} == (
        original["structural_zero_categories"])


def test_conditional_node_requires_absent_category_before_sampling_even_if_reset_occurs():
    event = _spec("double_elimination", teams="AB", final_reset="if_lower_wins")
    with pytest.raises(ValueError, match="absent_category"):
        simulate_tournament(event, {("A", "B", 1): 0.5}, simulations=1, seed=1, observables=[
            {"id": "reset", "kind": "node_matchup", "categories": ['["A","B"]'],
             "node": {"stage": "event", "round": "grand-final-reset"}}])


def test_observable_category_omission_fails_instead_of_dropping_probability_mass():
    with pytest.raises(ValueError, match="undeclared category"):
        simulate_tournament(_spec("single_elimination", teams="AB"), {("A", "B", 1): 1},
                            simulations=1, observables=[
            {"id": "series", "kind": "future_series_count", "categories": ["0"]}])
