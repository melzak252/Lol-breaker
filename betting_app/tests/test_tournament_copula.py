"""Synthetic marginal and complete-route contracts; no historical fitting."""
from copy import deepcopy
from itertools import combinations
import json
import math

import numpy as np
import pytest
from scipy.special import ndtr, ndtri, roots_hermitenorm

from src.models.tournament_copula import (
    GaussianTournamentCopula, copula_win_probability,
    fit_tournament_copula, tournament_copula_path_likelihood,
)


ROSTERS = {team: [f"{team}-{i}" for i in range(5)] for team in "ABC"}


def event(event_id="toy", p=.5, wins=(True, True)):
    pairs = [("C", "A"), ("B", "A")]
    return {
        "event_id": event_id, "cutoff": "2023-01-01T00:00:00Z",
        "rosters": deepcopy(ROSTERS),
        "probabilities": {(a, b, 1): p for a, b in combinations("ABC", 2)},
        "spec": {"version": 1, "id": "toy", "teams": list("BCA"),
                 "stages": [{"id": "event", "kind": "gauntlet", "entrants": list("BCA"), "best_of": 1}],
                 "champion": {"stage": "event", "group": "all", "rank": 1}},
        "matches": [{"match_id": str(i), "date": f"2023-01-0{i + 2}",
                     "stage": "event", "round": i + 1,
                     "team_a": a, "team_b": b, "best_of": 1,
                     "wins_a": int(not win), "wins_b": int(win)}
                    for i, ((a, b), win) in enumerate(zip(pairs, wins))],
    }


@pytest.mark.parametrize("partition", ["CAL", "ASSESS"])
def test_fit_rejects_explicit_nontrain_events_even_without_a_date_boundary(partition):
    path = event()
    path["partition"] = partition
    with pytest.raises(ValueError):
        fit_tournament_copula([path], sobol_power=3, replicates=2)


@pytest.mark.parametrize("p", [0., 1e-12, 1e-6, .03, .5, .91, 1 - 1e-12, 1.])
@pytest.mark.parametrize("rho", [0., .17, .5])
def test_independent_gauss_hermite_integrates_to_original_marginal(p, rho):
    nodes, weights = roots_hermitenorm(160)
    conditional = np.array([copula_win_probability(p, x, rho) for x in nodes])
    assert weights @ conditional / math.sqrt(2 * math.pi) == pytest.approx(p, abs=3e-15)
    if 0 < p < .5:
        assert weights @ conditional / math.sqrt(2 * math.pi) == pytest.approx(
            p, rel=1e-10, abs=1e-25)
    if 0 < p < 1:
        expected = ndtr((ndtri(p) + math.sqrt(rho) * .7) / math.sqrt(1 - rho))
        assert copula_win_probability(p, .7, rho) == pytest.approx(expected, abs=2e-15)
    for x in (-1e300, -2., 0., 2., 1e300):
        assert copula_win_probability(1 - p, -x, rho) == pytest.approx(
            1 - copula_win_probability(p, x, rho), abs=2e-15)
        if rho == 0 or p in (0, 1):
            assert copula_win_probability(p, x, rho) == p


def test_longdouble_finite_contrast_saturates_without_float_overflow():
    huge = np.finfo(np.longdouble).max / 4
    assert copula_win_probability(.2, huge, .5) == 1.
    assert copula_win_probability(.2, -huge, .5) == 0.
    assert copula_win_probability(.2, huge, 0.) == .2


@pytest.mark.parametrize("rho", [-.001, .50001, float("nan"), float("inf"), True])
def test_invalid_dependence_rejected(rho):
    with pytest.raises(ValueError):
        GaussianTournamentCopula(rho)
    with pytest.raises(ValueError):
        copula_win_probability(.5, 0., rho)


@pytest.mark.parametrize("p,contrast", [(float("nan"), 0), (-.1, 0), (1.1, 0), (.5, float("inf")), (.5, True)])
def test_invalid_scalar_probability_or_contrast_rejected(p, contrast):
    with pytest.raises(ValueError):
        copula_win_probability(p, contrast, .2)


def test_player_streams_ignore_roster_order_team_labels_and_unrelated_entrants():
    model = GaussianTournamentCopula(.3)
    original = deepcopy(ROSTERS)
    draws = model.sample(ROSTERS, 128, 71)
    renamed = {"renamed": list(reversed(ROSTERS["A"])), "other": ROSTERS["B"]}
    replay = model.sample(renamed, 128, 71)
    assert np.array_equal(draws["A"], replay["renamed"])
    assert np.array_equal(draws["B"], replay["other"])
    # The same player identity persists across independent lineup scenarios.
    swapped = deepcopy(ROSTERS)
    swapped["A"][0], swapped["B"][0] = swapped["B"][0], swapped["A"][0]
    changed = model.sample(swapped, 128, 71)
    assert changed["A"] + changed["B"] == pytest.approx(draws["A"] + draws["B"], abs=2e-15)
    assert not np.array_equal(changed["A"], draws["A"])
    assert ROSTERS == original


def test_sampler_rejects_non_disjoint_or_non_five_player_rosters():
    for invalid in ({"A": ROSTERS["A"][:4]}, {"A": ["x"] * 5},
                    {"A": ROSTERS["A"], "B": ROSTERS["A"]}):
        with pytest.raises(ValueError):
            GaussianTournamentCopula(.2).sample(invalid, 4, 1)
    for simulations, seed in ((True, 1), (0, 1), (1, -1), (1, True)):
        with pytest.raises(ValueError):
            GaussianTournamentCopula(.2).sample(ROSTERS, simulations, seed)


def test_strict_serialization_roundtrip_and_nested_provenance_are_isolated():
    provenance = {"numbers": [1, .2], "scope": "synthetic"}
    model = GaussianTournamentCopula(.2, provenance)
    provenance["numbers"][0] = 9
    payload = model.to_dict()
    restored = GaussianTournamentCopula.from_dict(json.loads(json.dumps(payload, allow_nan=False)))
    assert restored.sample(ROSTERS, 12, 2)["A"] == pytest.approx(model.sample(ROSTERS, 12, 2)["A"], abs=0)
    payload["provenance"]["numbers"][0] = 8
    assert model.to_dict()["provenance"]["numbers"][0] == 1
    for invalid in ({**model.to_dict(), "extra": 1}, {**model.to_dict(), "rho": float("nan")}):
        with pytest.raises(ValueError):
            GaussianTournamentCopula.from_dict(invalid)
    with pytest.raises(ValueError):
        GaussianTournamentCopula(.2, {"nonfinite": float("inf")})


def likelihood(data, rho, power=12):
    return tournament_copula_path_likelihood(data, rho, sobol_power=power, replicates=4, seed=9)


def test_shared_team_two_response_path_matches_analytic_orthant_probability():
    data = event()
    untouched = deepcopy(data)
    for rho in (0., .2, .5):
        result = likelihood(data, rho)
        analytic = .25 + math.asin(rho / 2) / (2 * math.pi)
        assert math.exp(result["log_probability"]) == pytest.approx(analytic, abs=5e-5)
        assert result["normalized_nll"] == pytest.approx(-math.log(analytic) / 2, abs=1e-4)
        assert result["normalized_nll_standard_error"] >= 0
        json.dumps(result, allow_nan=False)
    assert data == untouched


def test_identity_path_uses_original_probabilities_and_side_reversal_is_equivalent():
    data = event(p=.3)
    result = likelihood(data, 0.)
    assert result["log_probability"] == pytest.approx(math.log(.3 ** 2), abs=1e-14)
    reversed_data = deepcopy(data)
    reversed_data["probabilities"] = {(b, a, bo): 1 - p for (a, b, bo), p in data["probabilities"].items()}
    for row in reversed_data["matches"]:
        row["team_a"], row["team_b"] = row["team_b"], row["team_a"]
        row["wins_a"], row["wins_b"] = row["wins_b"], row["wins_a"]
    assert likelihood(reversed_data, .4)["log_probability"] == pytest.approx(likelihood(data, .4)["log_probability"], abs=1e-13)


def test_impossible_certainty_path_stays_impossible_with_strict_json_diagnostic():
    result = likelihood(event(p=0.), .5)
    assert result["log_probability"] is None  # strict JSON encoding of -infinity
    assert result["normalized_nll"] is None
    assert result["reason"] == "structurally_impossible_certain_response"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("first_date,origin,admitted", [
    ("2023-01-01", "2023-01-01T00:00:00Z", True),
    ("2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z", False),
    ("2023-01-01", "2023-01-01T01:00:00Z", False),
    ("2022-12-31", "2023-01-01T00:00:00Z", False),
])
def test_calendar_day_scenario_does_not_invent_an_exact_start_time(first_date, origin, admitted):
    data = event()
    data["cutoff"] = origin
    data["matches"][0]["date"] = first_date
    if admitted:
        result = likelihood(data, 0.)
        assert result["log_probability"] == pytest.approx(math.log(.25), abs=1e-14)
        assert result["target_clock_audit"]["same_origin_calendar_day_scenarios"] == 1
        assert result["target_clock_audit"]["authenticated_target_start_claim"] is False
    else:
        with pytest.raises(ValueError):
            likelihood(data, 0.)


def test_admission_rejects_incomplete_illegal_daily_predictions_and_future_train_labels():
    cases = []
    incomplete = event(); incomplete["matches"].pop(); cases.append(incomplete)
    illegal = event(); illegal["matches"][0]["wins_b"] = 2; cases.append(illegal)
    daily = event(); daily["probability_cutoff"] = "2023-01-02T00:00:00Z"; cases.append(daily)
    row_probability = event(); row_probability["matches"][0]["p"] = .5; cases.append(row_probability)
    for data in cases:
        with pytest.raises(ValueError):
            likelihood(data, .2)
        with pytest.raises(ValueError):
            fit_tournament_copula([data], sobol_power=5, replicates=2, seed=2)
    with pytest.raises(ValueError):
        fit_tournament_copula([event()], training_cutoff="2023-01-03T00:00:00Z",
                              sobol_power=5, replicates=2, seed=2)
    admitted = fit_tournament_copula([event()], training_cutoff="2024-01-01T00:00:00Z",
                                    sobol_power=5, replicates=2, seed=2)
    assert 0 <= admitted.rho <= .5


def test_global_fit_includes_identity_and_uses_equal_edition_normalized_objective():
    # A wins once then loses: positive dependence reduces this path likelihood.
    data = event()
    data["matches"][1]["wins_a"], data["matches"][1]["wins_b"] = 1, 0
    original = deepcopy(data)
    model = fit_tournament_copula([data], sobol_power=8, replicates=2, seed=7)
    assert model.rho == 0.
    objective = model.provenance["fit"]["normalized_nll"]
    assert objective == pytest.approx(math.log(2), abs=1e-14)
    assert data == original
    correlated = fit_tournament_copula([event()], sobol_power=8, replicates=2, seed=7)
    assert correlated.rho == .5
    assert correlated.provenance["fit"]["normalized_nll"] < math.log(2)
    json.dumps(correlated.to_dict(), allow_nan=False)


def test_fit_weights_editions_equally_not_series_count():
    longer = event("two-series", p=.3)
    shorter = event("one-series", p=.8)
    shorter["rosters"].pop("C")
    shorter["spec"]["teams"] = ["A", "B"]
    shorter["spec"]["stages"][0]["kind"] = "single_elimination"
    shorter["spec"]["stages"][0]["entrants"] = ["A", "B"]
    shorter["matches"] = [{
        "match_id": "final", "date": "2023-01-02", "stage": "event", "round": 1,
        "team_a": "A", "team_b": "B", "best_of": 1, "wins_a": 1, "wins_b": 0,
    }]
    shorter["probabilities"] = {("A", "B", 1): .8}
    model = fit_tournament_copula([longer, shorter], max_rho=0.,
                                 sobol_power=5, replicates=2, seed=8)
    assert model.rho == 0.
    assert model.provenance["fit"]["normalized_nll"] == pytest.approx(
        (-math.log(.3) - math.log(.8)) / 2, abs=1e-14)
