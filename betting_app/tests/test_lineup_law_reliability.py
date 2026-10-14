"""R1 RED behavioral contract; synthetic identities only, no DB/native worker/fits.

The orchestrator runs these before implementation. Final-map appearances and
first-map labels are intentionally different. Ages are elapsed UTC days since
conservative effective release; played-day provenance is retained separately.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import math

import numpy as np
import pytest

from betting_app.core.models.contract import UnifiedPredictionResult


def analysis():
    return importlib.import_module("src.analysis.lineup_law_reliability")


def model():
    return importlib.import_module("src.models.lineup_law_reliability")


def roster(prefix):
    return [f"{prefix}{i}" for i in range(5)]


def game(day, a="a", b="b", *, reversed_sides=False):
    teams, prefixes = (("B", "A"), (b, a)) if reversed_sides else (("A", "B"), (a, b))
    return {"date": day, "t1_id": teams[0], "t2_id": teams[1],
            "t1_players": {str(i): {"player_id": p} for i, p in enumerate(roster(prefixes[0]))},
            "t2_players": {str(i): {"player_id": p} for i, p in enumerate(roster(prefixes[1]))}}


def raw(match, day, a="a", b="b", *, context="League", best_of=3):
    return {"match_id": str(match), "date": day, "tournament_name": context,
            "best_of": best_of, "games": [game(day, a, b)]}


def event(match, played, prefix="a", *, release=None, team="A", context="League"):
    return {"team_id": team, "competition_context": context,
            "tournament_name": context, "source_match_id": str(match),
            "source_series_order": (0, int(match)) if str(match).isascii() and str(match).isdecimal() else (1, str(match)),
            "map_index": 1, "original_map_day": played,
            "effective_release_day": release or played, "roster": roster(prefix)}


def state(events, **overrides):
    args = {"team_id": "A", "competition_context": "League",
            "cutoff_day": "2026-04-20", "target_match_id": "999"}
    args.update(overrides)
    return analysis().candidate_state(events, **args)


def test_final_map_is_one_series_appearance_but_target_label_is_first_map():
    source = raw("20", "2026-04-01", "first-")
    source["games"].append(game("2026-04-03", "last-", reversed_sides=True))
    before = deepcopy(source)
    appearances = analysis().build_series_appearances([source], {"20": "2026-04-06"})
    assert len(appearances) == 2
    own = next(row for row in appearances if row["team_id"] == "A")
    assert own["roster"] == roster("last-")
    assert own["map_index"] == 2
    assert own["original_map_day"] == "2026-04-03"
    assert own["effective_release_day"] == "2026-04-06"
    result = state(appearances)
    assert result["candidates"][0]["count"] == 1
    assert result["candidates"][0]["age_days"] == 14  # Release, not played, age.
    assert analysis().released_target_label(source, {"20": "2026-04-06"}, team_id="A", available_before="2026-04-06") is None
    assert analysis().released_target_label(source, {"20": "2026-04-06"}, team_id="A", available_before="2026-04-07") == roster("first-")
    assert source == before


def test_release_override_cannot_precede_final_source_map_and_bo2_is_history_only():
    source = raw("20", "2026-04-01", best_of=2)
    source["games"].append(game("2026-04-03"))
    appearances = analysis().build_series_appearances([source], {"20": "2026-04-02"})
    assert {row["effective_release_day"] for row in appearances} == {"2026-04-03"}
    assert state(appearances)["history_ids"] == ["20"]
    assert analysis().released_target_label(source, {}, team_id="A", available_before="2026-04-20") is None


def test_last_ten_are_series_not_unique_rosters_and_repetitions_count():
    events = [event(i, f"2026-04-{i:02d}", "latest-" if i >= 6 else "previous-") for i in range(1, 13)]
    result = state(events)
    assert result["history_ids"] == [str(i) for i in range(3, 13)]
    latest, previous = result["candidates"]
    assert latest == {"roster": roster("latest-"), "count": 7, "age_days": 8,
                      "last_source_match_id": "12", "last_played_day": "2026-04-12", "last_release_day": "2026-04-12"}
    assert previous == {"roster": roster("previous-"), "count": 3, "age_days": 15,
                        "last_source_match_id": "5", "last_played_day": "2026-04-05", "last_release_day": "2026-04-05"}


def test_unordered_input_and_source_id_ties_are_deterministic_and_nonmutating():
    events = [event("10", "2026-04-02", "ten-"), event("2", "2026-04-02", "two-"),
              event("alpha", "2026-04-02", "alpha-"), event("beta", "2026-04-02", "beta-")]
    before = deepcopy(events)
    first = state(events)
    second = state(list(reversed(events)))
    assert first == second
    assert first["history_ids"] == ["2", "10", "alpha", "beta"]
    assert [c["roster"] for c in first["candidates"]] == [roster("beta-"), roster("alpha-")]
    assert events == before


def test_strict_prior_release_team_and_exact_context_cannot_be_swapped():
    events = [event("1", "2026-04-01", "valid-"),
              event("2", "2026-04-19", "same-day-", release="2026-04-20"),
              event("3", "2026-04-19", "future-", release="2026-04-21"),
              event("4", "2026-04-19", "cup-", context="Cup"),
              event("5", "2026-04-19", "opponent-", team="B")]
    result = state(events)
    assert result["history_ids"] == ["1"]
    assert result["candidates"][0]["roster"] == roster("valid-")
    assert state(events, competition_context="Cup")["candidates"][0]["roster"] == roster("cup-")
    assert state(events, competition_context="Unknown")["absence"] == "NO_EXACT_CONTEXT_HISTORY"
    assert state(events, competition_context=None)["absence"] == "MISSING_EXACT_CONTEXT"


def test_target_identity_is_excluded_even_with_erroneous_prior_release():
    events = [event("1", "2026-04-01", "old-"), event("999", "2026-04-02", "target-")]
    with pytest.raises(ValueError, match="target.*released"):
        state(events)


def test_target_and_future_roster_mutations_do_not_change_prior_candidates():
    sources = [raw("1", "2026-04-01", "old-"), raw("999", "2026-04-20", "target-")]
    before = state(analysis().build_series_appearances(sources, {}))
    changed = deepcopy(sources)
    changed[1]["games"][0] = game("2026-04-20", "different-")
    changed.append(raw("1000", "2026-05-01", "future-"))
    assert state(analysis().build_series_appearances(changed, {})) == before


def test_latest_incomplete_appearance_withholds_instead_of_resurrecting_older_five():
    latest = event("2", "2026-04-02", "partial-")
    latest["roster"][2] = None
    result = state([event("1", "2026-04-01", "old-"), latest])
    assert result["history_ids"] == ["1", "2"]
    assert result["candidates"] == []
    assert result["absence"] == "INCOMPLETE_LATEST_APPEARANCE"


def test_older_incomplete_appearance_occupies_window_but_is_not_an_exact_candidate():
    incomplete = event("1", "2026-04-01")
    incomplete["roster"][0] = None
    result = state([incomplete, event("2", "2026-04-02", "new-")])
    assert result["history_ids"] == ["1", "2"]
    assert result["incomplete_history_count"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["count"] == 1


@pytest.mark.parametrize("corruption", ["duplicate_series", "duplicate_player", "cross_side_overlap", "missing_release", "invalid_release"])
def test_raw_source_corruption_fails_closed(corruption):
    source = raw("1", "2026-04-01")
    sources, releases = [source], {}
    if corruption == "duplicate_series":
        sources.append(deepcopy(source))
    elif corruption == "duplicate_player":
        source["games"][0]["t1_players"]["1"]["player_id"] = "a0"
    elif corruption == "cross_side_overlap":
        source["games"][0]["t2_players"]["1"]["player_id"] = "a0"
    elif corruption == "missing_release":
        del source["games"][0]["date"]
    else:
        releases["1"] = "not-a-day"
    with pytest.raises(ValueError):
        analysis().build_series_appearances(sources, releases)


def test_duplicate_appearance_is_not_double_counted_and_invalid_age_clock_errors():
    item = event("1", "2026-04-01")
    with pytest.raises(ValueError, match="duplicate"):
        state([item, deepcopy(item)])
    item["original_map_day"] = "2026-04-21"
    with pytest.raises(ValueError):
        state([item])


def test_exact_sets_ignore_slot_order_but_not_missing_or_unknown_identity():
    item = event("1", "2026-04-01")
    other = event("2", "2026-04-02")
    other["roster"].reverse()
    result = state([item, other])
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["count"] == 2
    assert analysis().classify_lineup(result, list(reversed(roster("a")))) == "LATEST"
    assert analysis().classify_lineup(result, roster("never-seen-")) == "OTHER"
    for actual in ([None] * 5, ["a0"] * 5, roster("a")[:4]):
        with pytest.raises(ValueError):
            analysis().classify_lineup(result, actual)


def test_previous_and_third_distinct_roster_labels_and_single_candidate_mask():
    result = state([event("1", "2026-04-01", "third-"), event("2", "2026-04-02", "prev-"), event("3", "2026-04-03", "latest-")])
    assert analysis().classify_lineup(result, roster("prev-")) == "PREVIOUS"
    assert analysis().classify_lineup(result, roster("third-")) == "OTHER"
    single = state([event("1", "2026-04-01")])
    assert analysis().classify_lineup(single, roster("new-")) == "OTHER"


def features():
    return np.array([[3., 2.], [5., 0.]]), np.array([[4., 7.], [1., 0.]]), np.array([True, False])


def test_conditional_six_parameters_use_log_counts_and_release_ages_with_structural_mask():
    counts, ages, mask = features()
    theta = np.array([.2, .5, -.3, -.4, .7, .1])
    before = (theta.copy(), counts.copy(), ages.copy(), mask.copy())
    actual = model().lineup_probabilities(theta, counts, ages, mask, conditional=True)
    z0 = .2 + .5 * math.log(4) - .3 * math.log(5)
    z1 = -.4 + .7 * math.log(3) + .1 * math.log(8)
    denominator = math.exp(z0) + math.exp(z1) + 1
    expected_first = [math.exp(z0) / denominator, math.exp(z1) / denominator, 1 / denominator]
    z_single = .2 + .5 * math.log(6) - .3 * math.log(2)
    np.testing.assert_allclose(actual, [expected_first, [1 / (1 + math.exp(-z_single)), 0, 1 / (1 + math.exp(z_single))]], atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-14)
    assert np.all(actual[:, 2] > 0)
    for old, current in zip(before, (theta, counts, ages, mask)):
        np.testing.assert_array_equal(old, current)


def test_global_two_parameter_control_has_same_structural_masks_without_covariates():
    counts, ages, mask = features()
    theta = np.log([2., 3.])
    expected = [[1/3, 1/2, 1/6], [2/3, 0, 1/3]]
    actual = model().lineup_probabilities(theta, counts, ages, mask, conditional=False)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    changed = model().lineup_probabilities(theta, counts * 100, ages + 50, mask, conditional=False)
    np.testing.assert_array_equal(changed, actual)


@pytest.mark.parametrize("conditional,size", [(True, 6), (False, 2)])
def test_objective_and_gradient_include_mean_training_nll_and_fixed_ridge(conditional, size):
    counts, ages, mask = features()
    theta = np.linspace(-.3, .4, size)
    labels = np.array([1, 2])
    objective, gradient = model().lineup_objective(theta, counts, ages, mask, labels, conditional=conditional)
    q = model().lineup_probabilities(theta, counts, ages, mask, conditional=conditional)
    assert objective == pytest.approx(-np.log(q[np.arange(2), labels]).mean() + .1 * np.dot(theta, theta))
    # Independent centered finite differences check the analytic gradient, no fit.
    numeric = []
    for index in range(size):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        hi = model().lineup_objective(plus, counts, ages, mask, labels, conditional=conditional)[0]
        lo = model().lineup_objective(minus, counts, ages, mask, labels, conditional=conditional)[0]
        numeric.append((hi - lo) / 2e-6)
    np.testing.assert_allclose(gradient, numeric, atol=2e-8, rtol=2e-7)


def test_zero_logits_masked_probabilities_and_nll_are_not_fake_three_way_classes():
    counts, ages, mask = features()
    objective, gradient = model().lineup_objective(np.zeros(6), counts, ages, mask, np.array([0, 2]), conditional=True)
    assert objective == pytest.approx((math.log(3) + math.log(2)) / 2)
    assert np.isfinite(gradient).all()
    with pytest.raises(ValueError, match="unavailable|mask|previous"):
        model().lineup_objective(np.zeros(6), counts, ages, mask, np.array([0, 1]), conditional=True)


@pytest.mark.parametrize("kind", ["theta_size", "nonfinite", "negative_count", "negative_age", "empty"])
def test_probability_contract_rejects_invalid_or_unsupported_features(kind):
    counts, ages, mask = features()
    theta = np.zeros(6)
    if kind == "theta_size":
        theta = np.zeros(7)
    elif kind == "nonfinite":
        theta[0] = np.nan
    elif kind == "negative_count":
        counts[0, 0] = -1
    elif kind == "negative_age":
        ages[0, 0] = -1
    else:
        counts, ages, mask = counts[:0], ages[:0], mask[:0]
    with pytest.raises(ValueError):
        model().lineup_probabilities(theta, counts, ages, mask, conditional=True)


def test_primary_scores_are_side_mean_within_parent_series_not_side_weighted():
    rows = [{"match_id": "one", "side": "A", "label": "LATEST", "q": [.6, .3, .1]},
            {"match_id": "one", "side": "B", "label": "OTHER", "q": [.2, .3, .5]},
            {"match_id": "two", "side": "A", "label": "OTHER", "q": [.4, 0., .6]}]
    scored = analysis().score_side_laws(rows)
    indexed = {row["match_id"]: row for row in scored}
    assert indexed["one"]["lineup_nll"] == pytest.approx((-math.log(.6) - math.log(.5)) / 2)
    assert indexed["one"]["other_brier"] == pytest.approx((.1**2 + .5**2) / 2)
    assert indexed["two"]["lineup_nll"] == pytest.approx(-math.log(.6))
    assert indexed["two"]["other_brier"] == pytest.approx(.4**2)
    with pytest.raises(ValueError, match="duplicate"):
        analysis().score_side_laws(rows + [rows[0]])


def test_support_stage_requires_class_counts_and_months_without_cal_fitting():
    counts = {"train_side_labels": 1000, "train_previous_labels": 100, "train_other_labels": 100,
              "assessment_series": 200, "assessment_months": 6, "cal_side_labels": 3,
              "cal_latest_labels": 1, "cal_previous_labels": 1, "cal_other_labels": 1,
              "identity_release_valid": True}
    assert analysis().support_gate(counts)["status"] == "READY_FOR_TWO_TRAIN_FITS"
    for field in ("train_side_labels", "train_previous_labels", "train_other_labels", "assessment_series", "assessment_months", "cal_previous_labels"):
        changed = dict(counts)
        changed[field] -= 1
        assert analysis().support_gate(changed)["status"] != "READY_FOR_TWO_TRAIN_FITS"
    broken = dict(counts, identity_release_valid=False)
    assert analysis().support_gate(broken)["status"] == "BLOCKED_IDENTITY_RELEASE"


def prediction(probability, best_of=3):
    return UnifiedPredictionResult(canonical_match_id=None, prob_a=probability, prob_b=1-probability,
                                   model_name="Causal-C0", model_version="c0-native-2026-w32-e12-v1",
                                   feature_version="native-c0-history16-v1", best_of=best_of,
                                   diagnostics={"mode": "full", "probability_target": "series_win_already_projected", "uncertainty_available": False})


def test_partial_bounds_use_unconditional_product_weights_and_no_second_bo_transform():
    q_a, q_b = [.6, .3, .1], [.5, .2, .3]
    native = {(0, 0): prediction(.8), (0, 1): prediction(.6),
              (1, 0): prediction(.4), (1, 1): prediction(.2)}
    before = deepcopy(native)
    result = analysis().supported_roster_bounds(q_a, q_b, native, best_of=3)
    assert result["pair_weights"] == pytest.approx({(0, 0): .30, (0, 1): .12, (1, 0): .15, (1, 1): .06})
    # .30*.8 + .12*.6 + .15*.4 + .06*.2 = .384; mass=.63.
    assert result["lower"] == pytest.approx(.384)
    assert result["missing_mass"] == pytest.approx(.37)
    assert result["upper"] == pytest.approx(.754)
    assert result["prob_a"] is None
    assert result["scalar_absence"] == "POSITIVE_UNSUPPORTED_ROSTER_MASS"
    assert result["lower"] != pytest.approx(.384 / .63)
    assert result["lower"] != pytest.approx(.30*(3*.8**2-2*.8**3) + .12*(3*.6**2-2*.6**3) + .15*(3*.4**2-2*.4**3) + .06*(3*.2**2-2*.2**3))
    assert native == before


def test_side_reversal_complements_bounds_and_preserves_missing_mass():
    native = {(0, 0): prediction(.81, 5), (0, 1): prediction(.64, 5),
              (1, 0): prediction(.37, 5), (1, 1): prediction(.12, 5)}
    original = analysis().supported_roster_bounds([.7, .2, .1], [.4, .35, .25], native, best_of=5)
    swapped_native = {(b, a): prediction(1-p.prob_a, 5) for (a, b), p in native.items()}
    swapped = analysis().supported_roster_bounds([.4, .35, .25], [.7, .2, .1], swapped_native, best_of=5)
    assert original["lower"] == pytest.approx(.7*.4*.81 + .7*.35*.64 + .2*.4*.37 + .2*.35*.12)
    assert swapped["lower"] == pytest.approx(1-original["upper"])
    assert swapped["upper"] == pytest.approx(1-original["lower"])
    assert swapped["missing_mass"] == pytest.approx(original["missing_mass"])


def test_tiny_positive_other_mass_never_becomes_point_five_or_scalar():
    result = analysis().supported_roster_bounds([1-1e-8, 0., 1e-8], [1., 0., 0.], {(0, 0): prediction(.73)}, best_of=3)
    assert result["missing_mass"] > 0
    assert result["prob_a"] is None
    assert result["upper"] - result["lower"] == pytest.approx(1e-8, abs=1e-15)


def test_evidenced_complete_point_mass_retains_native_probability_without_shrinkage():
    result = analysis().supported_roster_bounds([1., 0., 0.], [1., 0., 0.], {(0, 0): prediction(.73)}, best_of=3)
    assert result["lower"] == result["upper"] == result["prob_a"] == pytest.approx(.73)
    assert result["missing_mass"] == 0


@pytest.mark.parametrize("kind", ["missing_pair", "format", "mode", "nonseries", "unknown_pair", "bad_q"])
def test_matched_native_case_and_joint_weight_errors_are_not_silent_fallbacks(kind):
    q_a, q_b = [.7, .2, .1], [.4, .35, .25]
    native = {(a, b): prediction(.7) for a in range(2) for b in range(2)}
    if kind == "missing_pair":
        del native[(1, 1)]
    elif kind == "format":
        native[(0, 0)] = prediction(.7, 5)
    elif kind == "mode":
        native[(0, 0)].diagnostics["mode"] = "no_w20"
    elif kind == "nonseries":
        native[(0, 0)].diagnostics["probability_target"] = "map_win"
    elif kind == "unknown_pair":
        native[(2, 0)] = prediction(.7)
    else:
        q_a = [.7, .2, .2]
    with pytest.raises(ValueError):
        analysis().supported_roster_bounds(q_a, q_b, native, best_of=3)


def test_native_candidate_requests_reject_either_already_released_target_clock():
    a = state([event("1", "2026-04-01", "a-old-"), event("2", "2026-04-02", "a-new-")])
    b = state([event("3", "2026-04-01", "b-old-", team="B"), event("4", "2026-04-02", "b-new-", team="B")], team_id="B")
    target = {"match_id": "999", "team1_id": "A", "team2_id": "B", "best_of": 5,
              "competition_context": "League", "decision_at": "2026-04-20T08:00:00Z",
              "start_at": None, "native_release_day": "2026-04-20", "label_release_day": "2026-04-21"}
    for release_field in ("native_release_day", "label_release_day"):
        leaking = dict(target, **{release_field: "2026-04-19"})
        with pytest.raises(ValueError, match="released"):
            analysis().build_native_pair_requests(leaking, a, b)


def test_overlapping_native_pair_and_missing_target_context_cannot_be_inferred():
    a = state([event("1", "2026-04-01", "shared-")])
    b = state([event("2", "2026-04-01", "shared-", team="B")], team_id="B")
    target = {"match_id": "999", "team1_id": "A", "team2_id": "B", "best_of": 3,
              "competition_context": "League", "decision_at": "2026-04-20T08:00:00Z",
              "start_at": None, "native_release_day": "2026-04-20", "label_release_day": "2026-04-20"}
    with pytest.raises(ValueError, match="overlap|share"):
        analysis().build_native_pair_requests(target, a, b)
    b = state([event("2", "2026-04-01", "b-", team="B")], team_id="B")
    target["competition_context"] = None
    with pytest.raises(ValueError, match="context"):
        analysis().build_native_pair_requests(target, a, b)


def test_incomplete_first_target_is_not_replaced_by_complete_final_map_actual_roster():
    source = raw("20", "2026-04-01", "first-")
    del source["games"][0]["t1_players"]["2"]
    source["games"].append(game("2026-04-03", "last-"))
    assert analysis().released_target_label(source, {}, team_id="A", available_before="2026-04-20") is None
    appearances = analysis().build_series_appearances([source], {})
    assert state(appearances)["candidates"][0]["roster"] == roster("last-")


def test_candidate_availability_uses_known_canonical_bound_without_inventing_unknown_clock():
    sources = [raw("1", "2026-04-01", "old-"), raw("2", "2026-04-02", "new-")]
    events = analysis().build_series_appearances(sources, {}, canonical_release_days={"2": "2026-04-22"})
    own = next(row for row in events if row["team_id"] == "A" and row["source_match_id"] == "2")
    assert own["native_release_day"] == "2026-04-02"
    assert own["canonical_release_day"] == "2026-04-22"
    assert own["effective_release_day"] == "2026-04-22"
    assert state(events)["history_ids"] == ["1"]
    older = next(row for row in events if row["team_id"] == "A" and row["source_match_id"] == "1")
    assert older["canonical_release_day"] is None
    assert older["effective_release_day"] == older["native_release_day"] == "2026-04-01"
    later = state(events, cutoff_day="2026-04-23")
    assert later["candidates"][0]["age_days"] == 1
    assert later["candidates"][0]["last_played_day"] == "2026-04-02"


def test_native_clock_conflicts_cover_all_source_contexts_not_only_candidate_team_history():
    sources = [raw("1", "2026-04-01", context="Other League"), raw("2", "2026-04-02")]
    events = analysis().build_series_appearances(sources, {}, canonical_release_days={"1": "2026-04-22", "2": "2026-04-03"})
    assert analysis().native_clock_conflicts(events, cutoff_day="2026-04-20") == ["1"]
    assert analysis().native_clock_conflicts(events, cutoff_day="2026-04-22") == ["1"]
    assert analysis().native_clock_conflicts(events, cutoff_day="2026-04-23") == []
    assert state(events)["history_ids"] == ["2"]


def test_unordered_raw_series_preserve_chronological_candidate_counts_and_input_bytes():
    sources = [raw("10", "2026-04-03", "latest-"), raw("2", "2026-04-01", "previous-"),
               raw("3", "2026-04-02", "latest-")]
    before = deepcopy(sources)
    first = state(analysis().build_series_appearances(sources, {}))
    second = state(analysis().build_series_appearances(list(reversed(sources)), {}))
    assert first == second
    assert first["history_ids"] == ["2", "3", "10"]
    assert [c["count"] for c in first["candidates"]] == [2, 1]
    assert sources == before
