"""Numeric and chronology contracts for native-C0 player residuals.

These fixtures are synthetic, disjoint five-player rosters; they do not load
historical data or claim calibrated real-world posterior uncertainty.
"""
from copy import deepcopy
from itertools import combinations, permutations
import json

import numpy as np
import pytest
from scipy.special import expit, logit

from src.models import tournament_uncertainty as residual
from src.models.tournament_formats import simulate_tournament


ROSTERS = {team: [f"{team}-player-{i}" for i in range(5)] for team in "ABCDEF"}
DAY = "2025-01-01"


def make_filter(**kwargs):
    return residual.PlayerResidualFilter(**kwargs)


def queue(state, a="A", b="B", p=.73, y=1, event_day=DAY,
          result_day="2025-01-02", match_id="first", model_year=2025):
    state.observe(p, y, ROSTERS[a], ROSTERS[b], event_day, result_day,
                  match_id=match_id, model_year=model_year)


def predict(state, a="A", b="B", p=.5, day="2025-01-03", year=2025):
    return state.predict(p, ROSTERS[a], ROSTERS[b], day, model_year=year)


def one_label_moments(prior, a, b, p, y):
    """Independent scalar-normal conditional moments, not sequential updates.

    Cov(player, match logit) = signed player variance / 5. Conditioning each
    player on the SAME scalar posterior therefore gives the ten diagonal ADF
    updates directly, without reading implementation state or private helpers.
    """
    players = ROSTERS[a] + ROSTERS[b]
    coefficients = np.array([.2] * 5 + [-.2] * 5)
    means = np.array([prior[player][0] for player in players])
    variances = np.array([prior[player][1] for player in players])
    location = logit(p) + coefficients @ means
    variance = coefficients ** 2 @ variances
    nodes, weights = np.polynomial.hermite.hermgauss(32)
    z = location + np.sqrt(2 * variance) * nodes
    likelihood = expit(z if y else -z)
    normalized = weights * likelihood
    normalized /= normalized.sum()
    posterior_z_mean = normalized @ z
    posterior_z_variance = normalized @ ((z - posterior_z_mean) ** 2)
    gain = coefficients * variances / variance
    updated_means = means + gain * (posterior_z_mean - location)
    updated_variances = variances + gain ** 2 * (posterior_z_variance - variance)
    return dict(zip(players, zip(updated_means, updated_variances)))


def assert_prediction_moments(actual, prior, a, b, p):
    mean_a = np.mean([prior[player][0] for player in ROSTERS[a]])
    mean_b = np.mean([prior[player][0] for player in ROSTERS[b]])
    variance_a = sum(prior[player][1] for player in ROSTERS[a]) / 25
    variance_b = sum(prior[player][1] for player in ROSTERS[b]) / 25
    nodes, weights = np.polynomial.hermite.hermgauss(32)
    location = logit(p) + mean_a - mean_b
    integrated = expit(location + np.sqrt(2 * (variance_a + variance_b)) * nodes)
    expected = {
        "mean_a": mean_a, "mean_b": mean_b,
        "variance_a": variance_a, "variance_b": variance_b,
        "mean_probability": expit(location),
        "integrated_probability": integrated @ weights / np.sqrt(np.pi),
    }
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, abs=1e-11)


def test_same_day_label_is_withheld_and_next_day_snapshot_is_pure():
    state = make_filter()
    queue(state)
    pending = deepcopy(state.to_dict())
    withheld = predict(state, day="2025-01-02")
    assert withheld["mean_probability"] == pytest.approx(.5, abs=1e-14)
    assert withheld["integrated_probability"] == pytest.approx(.5, abs=1e-14)
    assert withheld["mean_a"] == withheld["mean_b"] == 0
    assert state.to_dict() == pending
    state.advance("2025-01-02", model_year=2025)
    same_day = deepcopy(state.to_dict())
    released = predict(state)
    assert released["mean_a"] > 0 > released["mean_b"]
    assert released["mean_probability"] > .5
    assert state.to_dict() == same_day
    state.advance("2025-01-03", model_year=2025)
    assert predict(state) == pytest.approx(released, abs=1e-13)
    # The observed outcome must really have changed a prediction, not merely
    # survived a serialization or been present in a pending-label container.
    assert released["integrated_probability"] > withheld["integrated_probability"]


@pytest.mark.parametrize("y", [0, 1])
def test_all_ten_players_condition_on_one_prior_with_paired_mean_directions(y):
    state = make_filter(prior_sd=.35, process_variance_per_day=0)
    queue(state, p=.73, y=y)
    state.advance("2025-01-03", model_year=2025)
    prior = {player: (0., .35 ** 2) for roster in ROSTERS.values() for player in roster}
    posterior = prior | one_label_moments(prior, "A", "B", .73, y)
    actual = predict(state, p=.61)
    assert_prediction_moments(actual, posterior, "A", "B", .61)
    assert actual["mean_a"] == pytest.approx(-actual["mean_b"], abs=1e-12)
    assert actual["variance_a"] == pytest.approx(actual["variance_b"], abs=1e-12)
    direction = 1 if y else -1
    assert direction * actual["mean_a"] > 0
    assert direction * actual["mean_b"] < 0
    # A's learned offset follows its players to a previously unobserved rival.
    assert_prediction_moments(predict(state, b="C", p=.61), posterior, "A", "C", .61)


@pytest.mark.parametrize("p", [0., .17, .5, .83, 1.])
def test_cold_rosters_are_symmetric_and_certainties_are_not_floored(p):
    state = make_filter(prior_sd=.35)
    forward = predict(state, a="E", b="F", p=p)
    reverse = predict(state, a="F", b="E", p=1 - p)
    assert forward["mean_a"] == forward["mean_b"] == 0
    assert forward["variance_a"] == pytest.approx(.35 ** 2 / 5)
    assert forward["variance_b"] == pytest.approx(.35 ** 2 / 5)
    assert forward["mean_probability"] == pytest.approx(p, abs=1e-14)
    assert reverse["integrated_probability"] == pytest.approx(
        1 - forward["integrated_probability"], abs=1e-14)
    if p in (0., .5, 1.):
        assert forward["integrated_probability"] == pytest.approx(p, abs=1e-14)
    elif p < .5:
        assert p < forward["integrated_probability"] < .5
    else:
        assert .5 < forward["integrated_probability"] < p


def test_reversing_every_observation_side_preserves_player_state_and_complements_prediction():
    original, reversed_state = make_filter(), make_filter()
    observations = [
        ("A", "B", .72, 1, "2025-01-01", "2025-01-02"),
        ("A", "C", .39, 0, "2025-01-03", "2025-01-04"),
        ("D", "B", .63, 1, "2025-01-04", "2025-01-04"),
    ]
    for i, (a, b, p, y, day, release) in enumerate(observations):
        queue(original, a, b, p, y, day, release, f"match-{i}")
        queue(reversed_state, b, a, 1 - p, 1 - y, day, release, f"match-{i}")
    for state in (original, reversed_state):
        state.advance("2025-01-05", model_year=2025)
    for a, b in combinations("ABCD", 2):
        forward = predict(original, a, b, .67, "2025-01-05")
        same = predict(reversed_state, a, b, .67, "2025-01-05")
        reverse = predict(reversed_state, b, a, .33, "2025-01-05")
        assert same == pytest.approx(forward, abs=1e-12)
        for name in ("mean_probability", "integrated_probability"):
            assert reverse[name] == pytest.approx(1 - forward[name], abs=1e-12)
        assert reverse["mean_a"] == pytest.approx(forward["mean_b"], abs=1e-12)
        assert reverse["variance_b"] == pytest.approx(forward["variance_a"], abs=1e-12)


def test_json_roundtrip_preserves_numeric_state_process_growth_and_pending_predictions():
    state = make_filter(prior_sd=.42, process_variance_per_day=.001)
    queue(state)
    state.advance("2025-01-03", model_year=2025)
    queue(state, "C", "A", .29, 1, "2025-01-04", "2025-01-08", "pending")
    restored = residual.PlayerResidualFilter.from_dict(json.loads(json.dumps(state.to_dict())))
    at_origin = predict(state)
    before_release = predict(restored, day="2025-01-08")
    # Five independent player variances, each diffusing for five days.
    assert before_release["variance_a"] - at_origin["variance_a"] == pytest.approx(.001)
    assert before_release["variance_b"] - at_origin["variance_b"] == pytest.approx(.001)
    for day in ("2025-01-03", "2025-01-08", "2025-01-09"):
        for a, b in (("A", "B"), ("C", "A"), ("E", "F")):
            assert predict(restored, a, b, .64, day) == pytest.approx(
                predict(state, a, b, .64, day), abs=1e-13)
    for item in (state, restored):
        item.advance("2025-01-09", model_year=2025)
    assert predict(restored, "C", "A", .64, "2025-01-09") == pytest.approx(
        predict(state, "C", "A", .64, "2025-01-09"), abs=1e-13)
    assert predict(restored, "C", "A", .64, "2025-01-09")["mean_a"] > 0


def test_future_labels_and_hypothetical_predictions_cannot_mutate_current_forecast():
    states = [make_filter(process_variance_per_day=.001) for _ in range(2)]
    for state in states:
        queue(state)
        state.advance("2025-01-03", model_year=2025)
    queue(states[0], "A", "C", .2, 1, "2025-01-06", "2025-01-10", "future")
    queue(states[1], "A", "C", .9, 0, "2025-01-06", "2025-01-10", "future")
    current = predict(states[0], day="2025-01-08")
    assert predict(states[1], day="2025-01-08") == pytest.approx(current, abs=1e-13)
    before = [deepcopy(state.to_dict()) for state in states]
    future = [predict(state, b="C", day="2025-01-20") for state in states]
    assert future[0]["mean_probability"] > future[1]["mean_probability"]
    for state, snapshot in zip(states, before):
        predict(state, "E", "F", .72, "2025-02-01")
        assert state.to_dict() == snapshot
        assert predict(state, day="2025-01-08") == pytest.approx(current, abs=1e-13)


def test_same_release_day_permutations_equal_one_prior_natural_parameter_batch():
    prior = {player: (0., .35 ** 2) for roster in ROSTERS.values() for player in roster}
    prior |= one_label_moments(prior, "A", "B", .73, 1)
    batch = [("A", "B", .42, 0), ("A", "C", .65, 1), ("D", "A", .28, 1)]
    precision = {player: 1 / variance for player, (_, variance) in prior.items()}
    information = {player: mean / variance for player, (mean, variance) in prior.items()}
    for a, b, p, y in batch:
        for player, (mean, variance) in one_label_moments(prior, a, b, p, y).items():
            old_mean, old_variance = prior[player]
            precision[player] += 1 / variance - 1 / old_variance
            information[player] += mean / variance - old_mean / old_variance
    expected = {player: (information[player] / precision[player], 1 / precision[player])
                for player in prior}
    for ordering in permutations(range(len(batch))):
        state = make_filter()
        queue(state)
        state.advance("2025-01-03", model_year=2025)
        for i in ordering:
            a, b, p, y = batch[i]
            queue(state, a, b, p, y, "2025-01-04", "2025-01-05", f"batch-{i}")
        state.advance("2025-01-06", model_year=2025)
        for a, b in (("A", "B"), ("A", "C"), ("D", "A")):
            assert_prediction_moments(predict(state, a, b, .59, "2025-01-06"),
                                      expected, a, b, .59)


def test_changed_annual_anchor_resets_bias_and_discards_old_year_delayed_labels():
    state = make_filter(prior_sd=.42)
    queue(state, event_day="2024-12-20", result_day="2024-12-21", model_year=2024)
    state.advance("2024-12-22", model_year=2024)
    assert predict(state, day="2024-12-22", year=2024)["mean_a"] > 0
    queue(state, p=.1, y=1, event_day="2024-12-30", result_day="2025-01-05",
          match_id="old-delayed", model_year=2024)
    state.advance("2025-01-02", model_year=2025)
    reset = predict(state, day="2025-01-02")
    assert reset["mean_a"] == reset["mean_b"] == 0
    assert reset["variance_a"] == pytest.approx(.42 ** 2 / 5)
    fresh = make_filter(prior_sd=.42)
    for item in (state, fresh):
        queue(item, p=.8, y=0, event_day="2025-01-03", result_day="2025-01-03",
              match_id="new-anchor")
        item.advance("2025-01-06", model_year=2025)
    assert predict(state, day="2025-01-06") == pytest.approx(
        predict(fresh, day="2025-01-06"), abs=1e-13)
    assert predict(state, day="2025-01-06")["mean_a"] < 0


def event_and_table(probabilities=(.61, .36, .79, .48, .67, .22)):
    teams = list("ABCD")
    spec = {"version": 1, "id": "residual-cup", "teams": teams, "stages": [
        {"id": "cup", "kind": "single_elimination", "entrants": teams,
         "seeding": "ordered", "best_of": 1}],
        "champion": {"stage": "cup", "group": "all", "rank": 1}}
    event = {"id": "residual-cup", "teams": teams,
             "rosters": {team: ROSTERS[team] for team in teams},
             "cutoff": "2025-01-03", "year": 2025, "spec": spec}
    table = {"probability_unit": "series_win", "rows": [
        {"team_a": a, "team_b": b, "best_of": 1, "p": p}
        for (a, b), p in zip(combinations(teams, 2), probabilities)],
        "provenance": {"model_kind": "c0", "cutoff": event["cutoff"]}}
    return event, table


def arm_probabilities(arm):
    return {(row["team_a"], row["team_b"], row["best_of"]): row["p"]
            for row in arm["table"]["rows"]}


def test_arms_match_each_actual_input_pair_using_exact_shared_finite_draw_vectors():
    state = make_filter()
    queue(state)
    event, table = event_and_table()
    original_state, original_table = deepcopy(state.to_dict()), deepcopy(table)
    arms = residual.build_residual_tournament_arms(table, event, state,
                                                   simulations=4096, seed=47)
    coherent = arms["residual_coherent"]
    independent = arms["residual_independent_matched_marginals"]
    draws = coherent["team_strength_draws"]
    mean_p = arm_probabilities(arms["residual_posterior_mean"])
    matched_p = arm_probabilities(independent)
    base_p = arm_probabilities(arms["native_c0_unchanged"])
    assert independent["team_strength_draws"] is None
    assert arms["residual_posterior_mean"]["team_strength_draws"] is None
    for row in table["rows"]:
        a, b, bo, p = row["team_a"], row["team_b"], row["best_of"], row["p"]
        shifted = expit(logit(p) + np.asarray(draws[a]) - np.asarray(draws[b]))
        assert matched_p[a, b, bo] == pytest.approx(shifted.mean(), abs=1e-13)
        assert base_p[a, b, bo] == p
        assert arm_probabilities(coherent)[a, b, bo] == p
        snapshot = predict(state, a, b, p)
        assert mean_p[a, b, bo] == pytest.approx(snapshot["mean_probability"], abs=1e-13)
    assert mean_p["A", "B", 1] > base_p["A", "B", 1]
    assert state.to_dict() == original_state
    assert table == original_table


def test_arm_draw_integration_preserves_absorbing_native_probabilities():
    event, table = event_and_table((0., 1., .5, 1., 0., .5))
    arms = residual.build_residual_tournament_arms(table, event, make_filter(),
                                                   simulations=128, seed=19)
    for arm in arms.values():
        actual = arm_probabilities(arm)
        for row in table["rows"]:
            if row["p"] in (0., 1.):
                assert actual[row["team_a"], row["team_b"], 1] == row["p"]


def test_coherent_rounds_change_two_win_joint_probability_despite_matched_pair_marginals():
    # A is cold; balanced earlier labels make B/C/D less uncertain, without
    # introducing a winner-direction mean advantage. Heterogeneous uncertainty
    # exposes across-round selection instead of cancelling by four-team symmetry.
    state = make_filter(prior_sd=6., process_variance_per_day=0)
    for a, b in combinations("BCD", 2):
        for repetition in range(12):
            for y in (0, 1):
                queue(state, a, b, .5, y, match_id=f"{a}-{b}-{repetition}-{y}")
    event, table = event_and_table((.5,) * 6)
    n = 8000
    arms = residual.build_residual_tournament_arms(table, event, state,
                                                   simulations=n, seed=137)
    coherent = arms["residual_coherent"]
    independent = arms["residual_independent_matched_marginals"]
    draws = coherent["team_strength_draws"]
    conditional = {}
    for a, b in combinations("ABCD", 2):
        conditional[a, b] = expit(np.asarray(draws[a]) - np.asarray(draws[b]))
        assert arm_probabilities(independent)[a, b, 1] == pytest.approx(
            conditional[a, b].mean(), abs=1e-13)
    ab, ac, ad, cd = (conditional[pair] for pair in (("A", "B"), ("A", "C"),
                                                    ("A", "D"), ("C", "D")))
    # Ordered semifinals are A-B and C-D. Reuse the SAME per-rollout A residual
    # in both wins: E[p_AB * (p_CD*p_AC + (1-p_CD)*p_AD)], not products of E[p].
    coherent_two_wins = np.mean(ab * (cd * ac + (1 - cd) * ad))
    independent_two_wins = ab.mean() * (cd.mean() * ac.mean() + (1 - cd.mean()) * ad.mean())
    assert coherent_two_wins - independent_two_wins > .04
    outputs = {}
    for name, arm in (("coherent", coherent), ("independent", independent)):
        outputs[name] = simulate_tournament(event["spec"], arm_probabilities(arm),
            simulations=n, seed=73, team_strength_draws=arm["team_strength_draws"])
    assert outputs["coherent"]["final_prob"]["A"] == pytest.approx(ab.mean(), abs=.02)
    assert outputs["independent"]["final_prob"]["A"] == pytest.approx(ab.mean(), abs=.02)
    assert outputs["coherent"]["champion_prob"]["A"] == pytest.approx(coherent_two_wins, abs=.025)
    assert outputs["independent"]["champion_prob"]["A"] == pytest.approx(independent_two_wins, abs=.025)
    assert outputs["coherent"]["champion_prob"]["A"] - outputs["independent"]["champion_prob"]["A"] > .025


def test_restored_player_origin_must_precede_persisted_decision_day():
    state = make_filter()
    queue(state)
    state.advance("2025-01-03", model_year=2025)
    payload = state.to_dict()
    payload["players"][ROSTERS["A"][0]]["day"] = payload["cutoff_day"]
    with pytest.raises(ValueError):
        residual.PlayerResidualFilter.from_dict(payload)
