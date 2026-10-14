"""Checkpoint facts must not become random outcomes or hindsight draws."""
from copy import deepcopy
import hashlib
import json
from itertools import combinations

import pytest

from src.models.tournament_formats import simulate_tournament
from src.models.tournament_research_cohort import build_research_event_spec


CUTOFF = "2026-01-02T00:00:00Z"
PAST = "2026-01-01T12:00:00Z"


def spec(kind="single_elimination", teams="ABCD", **rules):
    return {"version": 1, "id": "cup", "teams": list(teams), "stages": [
        {"id": "event", "kind": kind, "entrants": list(teams), "best_of": 1, **rules}],
        "champion": {"stage": "event", "group": "all", "rank": 1}}


def probabilities(teams, p=0.5):
    return {(a, b, bo): p for a, b in combinations(teams, 2) for bo in (1, 3, 5)}


def checkpoint(records, **extra):
    return {"version": 1, "cutoff": CUTOFF, "completed": [
        {**record, "completed_at": PAST, "available_at": PAST} for record in records], **extra}


def publication(round_number, pairs):
    return {"stage": "event", "group": "all", "round": round_number,
            "pairs": [list(pair) for pair in pairs], "published_at": PAST, "available_at": PAST}


def test_empty_state_is_exactly_start_even_with_random_draws():
    event = spec(draw={"policy": "uniform"})
    kwargs = {"simulations": 40, "seed": 72}
    assert simulate_tournament(event, probabilities(event["teams"]), **kwargs) == simulate_tournament(
        event, probabilities(event["teams"]), state=checkpoint([]), **kwargs)


def test_completed_semifinals_are_facts_and_only_final_is_sampled():
    event = spec()
    event["placement_bands"] = [{"label": label, "placements": [
        {"stage": "event", "group": "all", "rank": rank} for rank in ranks]}
        for label, ranks in [("1", [1]), ("2", [2]), ("3-4", [3, 4])]]
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:2]
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=10000,
                                 state=checkpoint(history))
    assert result["champion_prob"]["A"] == pytest.approx(0.5, abs=0.02)
    assert result["champion_prob"]["B"] == pytest.approx(0.5, abs=0.02)
    assert result["champion_prob"]["C"] == result["champion_prob"]["D"] == 0
    assert result["final_prob"] == {"A": 1, "B": 1, "C": 0, "D": 0}
    assert result["placement_prob"]["C"] == {"1": 0, "2": 0, "3-4": 1}
    assert result["future_expected_series"] == 1
    assert result["expected_series"] == 3
    assert result["trace"][:2] == history
    assert result["target_resolved"]["final"]
    assert result["trace_scope"] == "first_simulation_only"


def test_completed_lower_bracket_loss_cannot_regain_champion_mass():
    event = spec("double_elimination", final_reset="if_lower_wins")
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:3]
    eliminated = history[-1]["loser"]
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=100,
                                 state=checkpoint(history))
    assert result["champion_prob"][eliminated] == 0
    assert result["trace"][:3] == history
    assert sum(result["champion_prob"].values()) == pytest.approx(1)
    assert sum(result["final_prob"].values()) == pytest.approx(2)


@pytest.mark.parametrize("rules", [{"rounds": 2}, {"wins_to_advance": 2, "losses_to_eliminate": 2}])
def test_swiss_preserves_published_historical_round_and_records(rules):
    event = spec("swiss", teams="ABCDEFGH", tiebreakers=["wins", "seed"], **rules)
    initial = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)
    history = [row for row in initial["trace"] if row["round"] == 1]
    pairs = [(row["team_a"], row["team_b"]) for row in history]
    state = checkpoint(history, pairings=[publication(1, pairs)])
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=30, state=state)
    assert result["trace"][:4] == history
    assert result["future_expected_series"] == result["expected_series"] - 4
    assert len({frozenset((r["team_a"], r["team_b"])) for r in result["trace"]}) == len(result["trace"])
    with pytest.raises(ValueError, match="published|pairing|draw"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=checkpoint(history))


def test_unknown_future_draw_is_sampled_but_published_future_pairing_is_frozen():
    event = spec(draw={"policy": "uniform"})
    state = checkpoint([], pairings=[publication(1, [("A", "B"), ("C", "D")])])
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=200, state=state)
    assert result["joint_final_prob"].get('["A","B"]', 0) == 0
    assert result["joint_final_prob"].get('["C","D"]', 0) == 0
    assert sum(result["joint_final_prob"].values()) == pytest.approx(1)


@pytest.mark.parametrize("field,value", [("available_at", "2026-01-03T00:00:00Z"),
                                         ("completed_at", "2026-01-03T00:00:00Z"),
                                         ("completed_at", "2026-01-01")])
def test_future_or_day_only_result_timestamps_are_rejected(field, value):
    event = spec()
    history = simulate_tournament(event, probabilities(event["teams"]), simulations=1)["trace"][:1]
    state = checkpoint(history)
    state["completed"][0][field] = value
    with pytest.raises(ValueError, match="timestamp|cutoff|available"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


def test_impossible_route_and_missing_ancestor_are_rejected():
    event = spec()
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    invalid = deepcopy(history)
    invalid[-1].update(team_b="C", loser="C")
    for records in (invalid, history[-1:]):
        with pytest.raises(ValueError, match="route|history|ancestor|completed"):
            simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=checkpoint(records))


def test_round_robin_non_traversal_prefix_preserves_independent_scores():
    event = spec("round_robin", tiebreakers=["wins", "seed"])
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=20,
                                 state=checkpoint([history[-1], history[0]]))
    by_pair = {(row["team_a"], row["team_b"]): row for row in result["trace"]}
    assert by_pair["C", "D"] == history[-1]
    assert by_pair["A", "B"] == history[0]
    assert result["future_expected_series"] == 4


def test_resolved_final_uses_actual_winner_even_when_future_model_disagrees():
    event = spec(teams="AB")
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=10, state=checkpoint(history))
    assert result["champion_prob"] == {"A": 1, "B": 0}
    assert result["future_expected_series"] == 0
    assert result["target_resolved"]["champion"]
    assert result["known_pairings"][0]["completed"]
    assert result["known_pairings"][0]["winner"] == "A"


@pytest.mark.parametrize("kind,rules", [
    ("gsl", {}),
    ("gauntlet", {}),
    ("pick_and_play", {"rounds": 2, "tiebreakers": ["wins", "seed"],
                      "choice_policy": "nearest_better_record", "no_rematches": False,
                      "initial_pairings": [{"a": "A", "b": "D"}, {"a": "B", "b": "C"}]}),
])
def test_completed_structural_openings_survive_changed_future_strength(kind, rules):
    event = spec(kind, **rules)
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:2]
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=10, state=checkpoint(history))
    assert result["trace"][:2] == history
    assert result["expected_series"] - result["future_expected_series"] == 2


def test_graph_independent_later_node_can_complete_before_earlier_list_node():
    event = spec("graph", matches=[
        {"id": "s1", "a": "A", "b": "D"}, {"id": "s2", "a": "B", "b": "C"},
        {"id": "final", "a": {"winner": "s1"}, "b": {"winner": "s2"}},
    ], ranking=[{"winner": "final"}, {"loser": "final"}, {"loser": "s1"}, {"loser": "s2"}])
    event["final"] = {"stage": "event", "group": "all", "round": "final"}
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=100,
                                 state=checkpoint([history[1]]))
    assert result["trace"][1] == history[1]
    assert result["final_prob"]["B"] == 1
    assert result["final_prob"]["C"] == 0
    assert result["target_resolved"]["final_by_team"]["B"]
    assert result["structural_zero_categories"]["champion"] == ["C"]
    assert result["future_expected_series"] == 2


def test_recorded_map_scores_do_not_require_a_model_when_no_future_scores_remain():
    event = spec("round_robin", teams="AB", tiebreakers=["wins", "map_diff", "seed"])
    event["stages"][0]["best_of"] = 3
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                                  score_distributions={3: [0, 1]})["trace"]
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=1,
                                 state=checkpoint(history))
    assert result["champion_prob"] == {"A": 1, "B": 0}
    assert result["trace"][0]["score_a"] == 2
    assert result["trace"][0]["score_b"] == 1
    assert result["future_expected_series"] == 0


def test_prior_day_reconstruction_never_claims_result_availability():
    event = spec(teams="AB")
    record = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][0]
    state = {"version": 1, "cutoff": CUTOFF, "temporal_policy": "prior_day_reconstruction",
             "completed": [{**record, "completed_date": "2026-01-01"}]}
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=1, state=state)
    assert result["champion_prob"] == {"A": 1, "B": 0}
    assert result["state_qualification"] == "reconstructed_not_availability_certified"
    state["completed"][0]["completed_date"] = "2026-01-02"
    with pytest.raises(ValueError, match="precede|cutoff"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)
    state["completed"][0]["completed_date"] = "2026-01-01"
    state["completed"][0]["available_at"] = PAST
    with pytest.raises(ValueError, match="fabricated|completed_date"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


def test_daily_rollforward_includes_only_prior_source_days_without_availability_claim():
    event = spec(teams="AB")
    record = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][0]
    state = {"version": 1, "cutoff": CUTOFF, "temporal_policy": "daily_rollforward",
             "completed": [{**record, "source_date": "2026-01-01"}]}
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=1, state=state)
    assert result["champion_prob"] == {"A": 1, "B": 0}
    assert result["state_qualification"] == "daily_reconstructed_not_availability_certified"


def test_daily_rollforward_requires_a_midnight_cutoff_and_prior_strict_source_date():
    event = spec(teams="AB")
    record = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][0]
    state = {"version": 1, "cutoff": CUTOFF, "temporal_policy": "daily_rollforward",
             "completed": [{**record, "source_date": "2026-01-01"}]}
    state["cutoff"] = "2026-01-02T12:00:00Z"
    with pytest.raises(ValueError, match="midnight"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)
    state["cutoff"] = CUTOFF
    for source_date in ("2026-01-02", "2026/01/01", None):
        if source_date is None:
            state["completed"][0].pop("source_date", None)
        else:
            state["completed"][0]["source_date"] = source_date
        with pytest.raises(ValueError, match="source_date|precede|calendar"):
            simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


@pytest.mark.parametrize("field,value", [
    ("completed_date", "2026-01-01"),
    ("completed_at", PAST),
    ("available_at", PAST),
])
def test_daily_rollforward_rejects_result_timestamps(field, value):
    event = spec(teams="AB")
    record = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][0]
    state = {"version": 1, "cutoff": CUTOFF, "temporal_policy": "daily_rollforward",
             "completed": [{**record, "source_date": "2026-01-01", field: value}]}
    with pytest.raises(ValueError, match="source_date|timestamp|daily"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


def test_daily_rollforward_accepts_prior_day_known_pairings_without_availability_claim():
    event = spec(draw={"policy": "uniform"})
    pairing = {"stage": "event", "group": "all", "round": 1,
               "pairs": [["A", "B"], ["C", "D"]], "source_date": "2026-01-01"}
    state = {"version": 1, "cutoff": CUTOFF, "temporal_policy": "daily_rollforward",
             "completed": [], "pairings": [pairing]}
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=200, state=state)
    assert {frozenset((row["team_a"], row["team_b"])) for row in result["known_pairings"]} == {
        frozenset(("A", "B")), frozenset(("C", "D"))}
    assert result["joint_final_prob"].get('["A","B"]', 0) == 0
    pairing["source_date"] = "2026-01-02"
    with pytest.raises(ValueError, match="source_date|precede|cutoff"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


def test_published_pairing_cannot_appear_after_the_forecast_origin():
    event = spec(draw={"policy": "uniform"})
    pairing = publication(1, [("A", "B"), ("C", "D")])
    pairing["available_at"] = "2026-01-03T00:00:00Z"
    with pytest.raises(ValueError, match="cutoff|timestamp"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1,
                            state=checkpoint([], pairings=[pairing]))


def test_completed_pick_round_requires_recorded_choices_not_scenario_hindsight():
    event = spec("pick_and_play", rounds=2, tiebreakers=["wins", "seed"],
                 choice_policy="nearest_better_record", no_rematches=False,
                 initial_pairings=[{"a": "A", "b": "D"}, {"a": "B", "b": "C"}])
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    with pytest.raises(ValueError, match="published|pairing"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=checkpoint(history))
    pairs = [(row["team_a"], row["team_b"]) for row in history if row["round"] == 2]
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=1,
                                 state=checkpoint(history, pairings=[publication(2, pairs)]))
    assert result["future_expected_series"] == 0
    assert result["trace"] == history


def test_recorded_random_group_order_routes_the_actual_group_winner():
    event = {
        "version": 1, "id": "groups", "teams": list("ABCD"),
        "stages": [
            {"id": "groups", "kind": "round_robin", "groups": {"x": ["A", "B"], "y": ["C", "D"]},
             "best_of": 1, "tiebreakers": ["wins", "seed"], "group_tiebreakers": ["wins", "random"]},
            {"id": "event", "kind": "gauntlet", "best_of": 1, "entrants": [
                {"stage": "groups", "group_rank": 1, "rank": 1},
                {"stage": "groups", "group_rank": 2, "rank": 1}]},
        ], "champion": {"stage": "event", "group": "all", "rank": 1},
    }
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    # Both groups finish tied at one total win; the order is a separate source fact.
    group_order = {"stage": "groups", "groups": ["y", "x"], "published_at": PAST, "available_at": PAST}
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                                 state=checkpoint(history[:2], group_orders=[group_order]))
    assert (result["trace"][-1]["team_a"], result["trace"][-1]["team_b"]) == ("C", "A")


def test_reseeding_does_not_infer_missing_other_branch_results_even_at_probability_one():
    event = spec(teams="ABCDEFGH", reseed=True)
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    incomplete_round = history[:2] + [history[4]]
    with pytest.raises(ValueError, match="earlier round|ancestor|completed"):
        simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                            state=checkpoint(incomplete_round))


def test_threshold_swiss_masks_early_qualified_and_eliminated_without_masking_live_teams():
    event = spec("swiss", teams="ABCDEFGHIJKLMNOP", wins_to_advance=3, losses_to_eliminate=3,
                 advance=8, tiebreakers=["wins", "seed"])
    initial = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                                  score_distributions={3: [1, 0]})
    history = [record for record in initial["trace"] if record["round"] <= 3]
    published = [publication(number, [(record["team_a"], record["team_b"]) for record in history
                                      if record["round"] == number]) for number in (1, 2, 3)]
    qualified = {team for team in event["teams"] if sum(r["winner"] == team for r in history) == 3}
    eliminated = {team for team in event["teams"] if sum(r["loser"] == team for r in history) == 3}
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=5,
                                 state=checkpoint(history, pairings=published))
    assert result["future_expected_series"] > 0
    for team in event["teams"]:
        assert result["target_resolved"]["advance_by_team"]["event"][team] == (team in qualified | eliminated)
    assert all(result["advance_prob"]["event"][team] == 1 for team in qualified)
    assert all(result["advance_prob"]["event"][team] == 0 for team in eliminated)
    assert any(set(band["teams"]) == qualified and band["ranks"] == list(range(1, 9))
               for band in result["certified_placement_bands"])
    assert any(set(band["teams"]) == eliminated and band["ranks"] == list(range(9, 17))
               for band in result["certified_placement_bands"])
    # Deterministic future probabilities do not turn unplayed facts into observations.
    assert not any(result["stage_rank_resolved"]["event"]["all"].values())


@pytest.mark.parametrize("final_advance", [1, 2])
def test_completed_group_eliminations_propagate_to_later_champion_and_placement_support(final_advance):
    event = {
        "version": 1, "id": "cup", "teams": list("ABCD"),
        "stages": [
            {"id": "groups", "kind": "round_robin", "entrants": list("ABCD"), "best_of": 1,
             "tiebreakers": ["wins", "seed"], "advance": 2},
            {"id": "final", "kind": "single_elimination", "best_of": 1, "advance": final_advance, "entrants": [
                {"stage": "groups", "group": "all", "rank": 1},
                {"stage": "groups", "group": "all", "rank": 2}]},
        ], "champion": {"stage": "final", "group": "all", "rank": 1},
        "placement_bands": [
            {"label": "1", "placements": [{"stage": "final", "group": "all", "rank": 1}]},
            {"label": "2", "placements": [{"stage": "final", "group": "all", "rank": 2}]},
            {"label": "3-4", "placements": [
                {"stage": "groups", "group": "all", "rank": 3},
                {"stage": "groups", "group": "all", "rank": 4}]},
        ],
    }
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:6]
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1, state=checkpoint(history))
    assert result["structural_zero_categories"]["champion"] == ["C", "D"]
    assert result["possible_placement_categories"] == {"A": ["1", "2"], "B": ["1", "2"],
                                                       "C": ["3-4"], "D": ["3-4"]}
    assert result["stage_rank_possible"]["final"]["all"]["A"] == [1, 2]
    assert result["stage_rank_possible"]["final"]["all"]["C"] == []
    assert result["target_resolved"]["advance_by_team"]["final"] == {
        "A": final_advance == 2, "B": final_advance == 2, "C": True, "D": True}
    assert result["target_resolved"]["advance"]["final"] == (final_advance == 2)


def test_standings_source_availability_cannot_move_backward_with_match_traversal():
    event = {
        "version": 1, "id": "cup", "teams": list("ABCD"), "stages": [
            {"id": "groups", "kind": "round_robin", "entrants": list("ABC"), "best_of": 1,
             "tiebreakers": ["wins", "seed"]},
            {"id": "final", "kind": "single_elimination", "best_of": 1, "entrants": [
                {"stage": "groups", "group": "all", "rank": 1}, "D"]},
        ], "champion": {"stage": "final", "group": "all", "rank": 1},
    }
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"]
    state = checkpoint(history)
    for row, time in zip(state["completed"], ["12:00:00", "09:00:00", "09:30:00", "10:00:00"]):
        row["completed_at"] = row["available_at"] = f"2026-01-01T{time}Z"
    with pytest.raises(ValueError, match="ancestor|availability|timestamp"):
        simulate_tournament(event, probabilities(event["teams"], 1), simulations=1, state=state)


def test_known_series_proof_excludes_random_draws_and_future_winners_at_probability_one():
    event = spec()
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)
    assert {(row["round"], frozenset((row["team_a"], row["team_b"]))) for row in result["known_pairings"]} == {
        (1, frozenset("AD")), (1, frozenset("BC"))}
    event["stages"][0]["draw"] = {"policy": "uniform"}
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)
    assert result["known_pairings"] == []
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                                 state=checkpoint([], pairings=[publication(1, [("A", "B"), ("C", "D")])]))
    assert {frozenset((row["team_a"], row["team_b"])) for row in result["known_pairings"]} == {
        frozenset("AB"), frozenset("CD")}
    assert not any(row["completed"] for row in result["known_pairings"])


def test_known_round_robin_schedule_does_not_depend_on_sampled_earlier_scores():
    event = spec("round_robin", tiebreakers=["wins", "seed"])
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=1)
    assert {frozenset((row["team_a"], row["team_b"])) for row in result["known_pairings"]} == {
        frozenset(pair) for pair in combinations(event["teams"], 2)}


def test_gauntlet_top_seed_is_known_finalist_without_observing_any_future_winner():
    event = spec("gauntlet")
    event["placement_bands"] = [{"label": str(rank), "placements": [
        {"stage": "event", "group": "all", "rank": rank}]} for rank in range(1, 5)]
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)
    assert result["final_prob"]["A"] == 1
    assert result["possible_placement_categories"]["A"] == ["1", "2"]
    assert result["target_resolved"]["final_by_team"]["A"]
    assert not result["target_resolved"]["champion"]


def test_observable_future_counts_exclude_completed_prefix_but_rematches_include_history():
    event = spec("double_elimination", teams="AB", final_reset="single_final")
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:1]
    result = simulate_tournament(event, probabilities(event["teams"], 1), simulations=5,
                                 state=checkpoint(history), observables=[
        {"id": "series", "kind": "future_series_count", "categories": ["1"]},
        {"id": "maps", "kind": "future_map_count", "categories": ["1"]},
        {"id": "repeat", "kind": "repeat_encounter_count", "categories": ["1"]},
        {"id": "upsets", "kind": "upset_count", "categories": ["1"],
         "reference_probabilities": {("A", "B", 1): 0.2}},
    ])
    assert result["observable_prob"] == {name: {"1": 1} for name in ("series", "maps", "repeat", "upsets")}
    assert result["champion_prob"] == {"A": 1, "B": 0}


def test_completed_oriented_score_and_zero_future_maps_need_no_future_score_model():
    event = spec(teams="AB")
    event["stages"][0]["best_of"] = 3
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                                  score_distributions={3: [0, 1]})["trace"]
    result = simulate_tournament(event, probabilities(event["teams"], 0), simulations=4,
                                 state=checkpoint(history), observables=[
        {"id": "maps", "kind": "future_map_count", "categories": ["0"]},
        {"id": "score", "kind": "node_score", "participants": ["B", "A"],
         "node": {"stage": "event", "round": 1}, "categories": ["0-2", "1-2", "2-0", "2-1"]},
    ])
    assert result["observable_prob"]["maps"] == {"0": 1}
    assert result["observable_prob"]["score"] == {"0-2": 0, "1-2": 1, "2-0": 0, "2-1": 0}


@pytest.mark.parametrize("after_first", [False, True])
def test_lck_qualifying_graph_joint_distribution_matches_independent_enumeration(after_first):
    from scripts.verify_tournament_distribution_oracles import fixtures
    name = "lck-prefix" if after_first else "lck-start"
    _, event, table, kwargs, expected = next(fixture for fixture in fixtures() if fixture[0] == name)
    table = {key: value for key, value in table.items() if key[2] == 5}
    result = simulate_tournament(event, table, simulations=16000, seed=202, **kwargs)
    for source, exact in expected.items():
        observed = result
        for component in source.split("/"):
            observed = observed[component]
        assert {category: observed.get(category, 0) for category in exact} == pytest.approx(exact, abs=.018)
    assert sum(result["advance_prob"]["qualifier"].values()) == pytest.approx(2)
    joint = result["joint_advance_prob"]["qualifier"]
    for team in event["teams"]:
        marginal = sum(value for category, value in joint.items() if f'"{team}"' in category)
        assert marginal == pytest.approx(result["advance_prob"]["qualifier"][team])


@pytest.mark.parametrize("seed", [1, 12])
def test_joint_final_support_uses_fixed_branches_not_simulation_hits(seed):
    from scripts.tournament_evaluation import _condition_target
    event = spec(seeding="ordered")
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=1, seed=seed)
    target = _condition_target({"source": "joint_final_prob"}, result)
    assert set(target["structural_zero_categories"]) == {'["A","B"]', '["C","D"]'}
    assert len(result["joint_final_prob"]) == 1


def test_unpublished_draw_does_not_certify_sampled_final_branches():
    from scripts.tournament_evaluation import _condition_target
    event = spec(draw={"policy": "uniform"})
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=1)
    assert _condition_target({"source": "joint_final_prob"}, result)["structural_zero_categories"] == []


def test_published_draw_and_completed_prefix_prove_joint_support():
    from scripts.tournament_evaluation import _condition_target
    event = spec(draw={"policy": "uniform"})
    state = checkpoint([], pairings=[publication(1, [("A", "B"), ("C", "D")])])
    start = simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)
    assert set(_condition_target({"source": "joint_final_prob"}, start)["structural_zero_categories"]) == {
        '["A","B"]', '["C","D"]'}
    prefix = checkpoint(start["trace"][:1], pairings=state["pairings"])
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=prefix)
    loser = start["trace"][0]["loser"]
    zeros = _condition_target({"source": "joint_final_prob"}, result)["structural_zero_categories"]
    assert len(zeros) == 4
    assert sum(loser in pair for pair in zeros) == 3


def waiver_swiss_spec(teams="ABCDEFGH", **rules):
    return spec("swiss", teams=teams, tiebreakers=["wins", "seed"],
                no_rematches=True, rematch_fallback="waive_only_if_no_feasible_draw",
                initial_pairings=[list(pair) for pair in zip(teams[::2], teams[1::2])], **rules)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_waiver_swiss_completed_round_one_preserves_published_round_two(seed):
    event = waiver_swiss_spec(rounds=2)
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:4]
    pairs = [("A", "C"), ("E", "G"), ("B", "D"), ("F", "H")]
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=4, seed=seed,
                                 state=checkpoint(history, pairings=[publication(2, pairs)]))
    assert result["trace"][:4] == history
    assert [(row["team_a"], row["team_b"]) for row in result["trace"][4:]] == pairs
    assert {frozenset((row["team_a"], row["team_b"])) for row in result["known_pairings"]
            if row["round"] == 2} == {frozenset(pair) for pair in pairs}


@pytest.mark.parametrize("pairs", [
    [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")],
    [("A", "D"), ("C", "F"), ("E", "H"), ("G", "B")],
])
def test_waiver_swiss_rejects_published_rematches_or_cross_record_pairs(pairs):
    event = waiver_swiss_spec(rounds=2)
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:4]
    with pytest.raises(ValueError, match="pairing|draw|record|rematch"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1,
                            state=checkpoint(history, pairings=[publication(2, pairs)]))


def completed_swiss_quarterfinal_checkpoint():
    event = waiver_swiss_spec(teams="ABCDEFGHIJKLMNOP", wins_to_advance=3,
                             losses_to_eliminate=3, advance=8)
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1,
                                  seed=31, score_distributions={3: [1, 0]})["trace"]
    published = [publication(number, [(row["team_a"], row["team_b"])
                                     for row in history if row["round"] == number])
                 for number in range(1, 6)]
    records = {team: (sum(row["winner"] == team for row in history),
                      sum(row["loser"] == team for row in history)) for team in event["teams"]}
    unbeaten = [team for team in event["teams"] if records[team] == (3, 0)]
    three_two = [team for team in event["teams"] if records[team] == (3, 2)]
    remaining = [team for team in event["teams"]
                 if records[team][0] == 3 and team not in unbeaten + three_two[:2]]
    pairs = [(unbeaten[0], three_two[0]), tuple(remaining[:2]),
             (unbeaten[1], three_two[1]), tuple(remaining[2:])]
    event["stages"].append({
        "id": "playoffs", "kind": "single_elimination", "best_of": 1,
        "swiss_quarterfinal_from": "event",
        "entrants": [{"stage": "event", "group": "all", "rank": rank} for rank in range(1, 9)]})
    event["champion"] = {"stage": "playoffs", "group": "all", "rank": 1}
    published.append({**publication(1, pairs), "stage": "playoffs"})
    return event, checkpoint(history, pairings=published), pairs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_completed_swiss_preserves_legal_published_quarterfinal_slots(seed):
    event, state, pairs = completed_swiss_quarterfinal_checkpoint()
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=4,
                                 seed=seed, state=state)
    quarterfinals = [row for row in result["trace"] if row["stage"] == "playoffs" and row["round"] == 1]
    assert [(row["team_a"], row["team_b"]) for row in quarterfinals] == pairs
    assert result["future_expected_series"] == 7
    assert {frozenset((row["team_a"], row["team_b"])) for row in result["known_pairings"]
            if row["stage"] == "playoffs" and row["round"] == 1} == {frozenset(pair) for pair in pairs}


@pytest.mark.parametrize("violation", ["same_half", "wrong_record"])
def test_swiss_quarterfinal_rejects_illegal_published_slots(violation):
    event, state, pairs = completed_swiss_quarterfinal_checkpoint()
    if violation == "same_half":
        pairs[1], pairs[2] = pairs[2], pairs[1]
    else:
        pairs[0], pairs[1] = (pairs[0][0], pairs[1][0]), (pairs[0][1], pairs[1][1])
    state["pairings"][-1]["pairs"] = [list(pair) for pair in pairs]
    with pytest.raises(ValueError, match="pairing|draw|quarterfinal"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


def test_waiver_swiss_unknown_round_two_is_sampled():
    event = waiver_swiss_spec(rounds=2)
    history = simulate_tournament(event, probabilities(event["teams"], 1), simulations=1)["trace"][:4]
    draws = set()
    for seed in range(4):
        result = simulate_tournament(event, probabilities(event["teams"]), simulations=1,
                                     seed=seed, state=checkpoint(history))
        draws.add(frozenset(frozenset((row["team_a"], row["team_b"]))
                            for row in result["trace"] if row["round"] == 2))
    assert len(draws) > 1


def test_completed_swiss_unknown_quarterfinal_is_sampled():
    event, state, _ = completed_swiss_quarterfinal_checkpoint()
    state["pairings"].pop()
    draws = set()
    for seed in range(4):
        result = simulate_tournament(event, probabilities(event["teams"]), simulations=1,
                                     seed=seed, state=state)
        draws.add(tuple(frozenset((row["team_a"], row["team_b"])) for row in result["trace"]
                        if row["stage"] == "playoffs" and row["round"] == 1))
    assert len(draws) > 1


def completed_worlds_swiss_checkpoint(tmp_path, edition):
    """Use actual edition mechanics; never invent orders for equal Swiss records."""
    year = int(edition.rsplit("-", 1)[1])
    contract = {
        "edition_id": edition, "authenticated_prestart": False,
        "draw_basis": "retrospective_known_opening_draw_scenario",
        "opening_seed_basis": "prior_source",
        "swiss": {"no_rematches": year >= 2024, "wins_to_advance": 3, "losses_to_eliminate": 3},
        "knockout": {"three_zero_opposite_halves": True, "three_zero_vs_three_two": True},
    }
    if year >= 2024:
        contract["swiss"]["rematch_fallback"] = "waive_only_if_no_feasible_draw"
    evidence = tmp_path / f"{edition}-rules.json"
    payload = json.dumps(contract, sort_keys=True).encode()
    evidence.write_bytes(payload)
    contract["source_path"] = str(evidence)
    contract["source_sha256"] = {"rules": hashlib.sha256(payload).hexdigest()}
    teams = list("ABCDEFGHIJKLMNOP")
    event = build_research_event_spec(
        edition, teams, initial_pairings=list(zip(teams[::2], teams[1::2])),
        source_contract=contract)
    initial = simulate_tournament(
        event, probabilities(teams, 1), simulations=1, seed=31,
        score_distributions={3: [1, 0], 5: [1, 0, 0]})
    history = [row for row in initial["trace"] if row["stage"] == "swiss"]
    quarterfinals = [row for row in initial["trace"]
                    if row["stage"] == "knockout" and row["round"] == 1]
    published = [{**publication(number, [(row["team_a"], row["team_b"])
                                        for row in history if row["round"] == number]),
                  "stage": "swiss"} for number in range(1, 6)]
    published.append({**publication(1, [(row["team_a"], row["team_b"]) for row in quarterfinals]),
                      "stage": "knockout"})
    return event, checkpoint(history, pairings=published), quarterfinals


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
@pytest.mark.parametrize("completed_quarterfinals", [False, True])
def test_worlds_completed_swiss_accepts_published_quarterfinal_without_tie_orders(
        tmp_path, edition, completed_quarterfinals):
    event, state, quarterfinals = completed_worlds_swiss_checkpoint(tmp_path, edition)
    if completed_quarterfinals:
        state["completed"].extend(checkpoint(quarterfinals)["completed"])
    qualifiers = {team for row in quarterfinals for team in (row["team_a"], row["team_b"])}
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=4, seed=7, state=state)
    actual = [row for row in result["trace"] if row["stage"] == "knockout" and row["round"] == 1]
    assert [(row["team_a"], row["team_b"]) for row in actual] == [
        (row["team_a"], row["team_b"]) for row in quarterfinals]
    if completed_quarterfinals:
        assert actual == quarterfinals
    assert result["advance_prob"]["swiss"] == {
        team: float(team in qualifiers) for team in event["teams"]}
    assert result["target_resolved"]["advance"]["swiss"]
    assert not any(result["stage_rank_resolved"]["swiss"]["all"].values())
    assert result["future_expected_series"] == (3 if completed_quarterfinals else 7)
    assert {(row["team_a"], row["team_b"]) for row in result["known_pairings"]
            if row["stage"] == "knockout" and row["round"] == 1} == {
        (row["team_a"], row["team_b"]) for row in quarterfinals}


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
@pytest.mark.parametrize("violation", ["same_half", "wrong_record"])
def test_worlds_completed_swiss_rejects_illegal_quarterfinal_without_tie_orders(
        tmp_path, edition, violation):
    event, state, _ = completed_worlds_swiss_checkpoint(tmp_path, edition)
    wins = {team: sum(row["winner"] == team for row in state["completed"]) for team in event["teams"]}
    losses = {team: sum(row["loser"] == team for row in state["completed"]) for team in event["teams"]}
    pairs = state["pairings"][-1]["pairs"]
    unbeaten_slots = [i for i, pair in enumerate(pairs) if any(losses[team] == 0 for team in pair)]
    if violation == "same_half":
        other = next(i for i in range(2) if i != unbeaten_slots[0])
        pairs[other], pairs[unbeaten_slots[1]] = pairs[unbeaten_slots[1]], pairs[other]
    else:
        first = unbeaten_slots[0]
        other = next(i for i, pair in enumerate(pairs)
                     if not any(losses[team] == 0 for team in pair)
                     and any(wins[team] == 3 and losses[team] == 1 for team in pair))
        opponent = next(i for i, team in enumerate(pairs[first]) if losses[team] == 2)
        replacement = next(i for i, team in enumerate(pairs[other]) if losses[team] == 1)
        pairs[first][opponent], pairs[other][replacement] = pairs[other][replacement], pairs[first][opponent]
    with pytest.raises(ValueError, match="pairing|draw|quarterfinal"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
def test_worlds_incomplete_swiss_rejects_future_published_quarterfinal_qualifiers(tmp_path, edition):
    event, state, _ = completed_worlds_swiss_checkpoint(tmp_path, edition)
    state["completed"] = [row for row in state["completed"] if row["round"] < 5]
    state["pairings"] = [row for row in state["pairings"] if row["stage"] == "knockout" or row["round"] < 5]
    with pytest.raises(ValueError, match="unresolved"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
def test_worlds_incomplete_swiss_does_not_certify_sampled_quarterfinal_qualifiers(tmp_path, edition):
    event, state, _ = completed_worlds_swiss_checkpoint(tmp_path, edition)
    state["completed"] = [row for row in state["completed"] if row["round"] < 5]
    state["pairings"] = [row for row in state["pairings"] if row["stage"] == "swiss" and row["round"] < 5]
    result = simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)
    live = {team for team in event["teams"]
            if sum(row["winner"] == team for row in state["completed"]) == 2
            and sum(row["loser"] == team for row in state["completed"]) == 2}
    assert not result["target_resolved"]["advance"]["swiss"]
    assert all(not result["target_resolved"]["advance_by_team"]["swiss"][team] for team in live)
    assert not any(row["stage"] == "knockout" for row in result["known_pairings"])


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
def test_worlds_completed_swiss_does_not_resolve_individual_rank_routes(tmp_path, edition):
    event, state, _ = completed_worlds_swiss_checkpoint(tmp_path, edition)
    del event["stages"][1]["swiss_quarterfinal_from"]
    with pytest.raises(ValueError, match="unresolved"):
        simulate_tournament(event, probabilities(event["teams"]), simulations=1, state=state)


@pytest.mark.parametrize("edition", ["worlds-2023", "worlds-2024"])
def test_worlds_completed_swiss_unknown_quarterfinal_draw_remains_sampled(tmp_path, edition):
    event, state, _ = completed_worlds_swiss_checkpoint(tmp_path, edition)
    state["pairings"].pop()
    draws = set()
    for seed in range(4):
        result = simulate_tournament(event, probabilities(event["teams"]), simulations=1, seed=seed, state=state)
        draws.add(tuple(frozenset((row["team_a"], row["team_b"])) for row in result["trace"]
                        if row["stage"] == "knockout" and row["round"] == 1))
        assert not any(result["stage_rank_resolved"]["swiss"]["all"].values())
        assert not any(row["stage"] == "knockout" for row in result["known_pairings"])
    assert len(draws) > 1
