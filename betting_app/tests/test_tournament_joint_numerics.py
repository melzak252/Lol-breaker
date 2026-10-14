"""Joint targets and integration error are distinct from marginal sports skill."""
import math

import pytest

from src.analysis import tournament_metrics as metrics
from src.analysis.tournament_numerics import (
    query_tournament_joint_numerics,
    summarize_tournament_numerics,
)


TEAMS = ["A", "B", "C", "D"]


def _score(law, observed=("A", "B")):
    event = {"id": "toy", "edition_id": "toy-edition", "teams": TEAMS}
    forecast = {"final_prob": dict.fromkeys(TEAMS, .5), "joint_final_prob": law,
                "advance_prob": {"group": dict.fromkeys(TEAMS, .5)},
                "joint_advance_prob": {"group": law}}
    labels = {"event_id": "toy", "edition_id": "toy-edition", "finalists": list(observed),
              "qualifiers": {"group": list(observed)}}
    return metrics.score_research_event(event, forecast, labels)


def test_identical_inclusion_marginals_do_not_identify_joint_finalist_or_qualifier_loss():
    first = _score({'["A","B"]': .5, '["C","D"]': .5})
    second = _score({'["A","B"]': .25, '["C","D"]': .25,
                     '["A","C"]': .25, '["B","D"]': .25})
    assert first["finalists_brier"] == second["finalists_brier"]
    assert first["advance_brier"] == second["advance_brier"]
    assert first["joint_finalists_log_loss"] == pytest.approx(math.log(2))
    assert second["joint_finalists_log_loss"] == pytest.approx(math.log(4))
    assert first["joint_finalists_brier"] == pytest.approx(.5)
    assert second["joint_finalists_brier"] == pytest.approx(.75)
    assert first["joint_qualification_brier"] == pytest.approx(.5)
    assert second["joint_qualification_brier"] == pytest.approx(.75)
    assert first["actual_joint_targets"] == {
        "finalists": '["A","B"]', "qualifiers": {"group": '["A","B"]'}}


@pytest.mark.parametrize("law", [
    {'["A","A"]': 1.}, {'["A","Z"]': 1.}, {'["A"]': 1.},
    {'["A","B"]': .5, '["B","A"]': .5}, {'"AB"': 1.},
    {'{"A":1,"B":1}': 1.}, {'["A",[]]': 1.}, {'not-json': 1.},
])
def test_joint_categories_reject_malformed_members_cardinality_and_duplicate_encodings(law):
    with pytest.raises(ValueError):
        _score(law)


def test_missing_legal_sparse_category_is_zero_not_an_invented_uniform_law():
    scored = _score({'["A","C"]': .5, '["B","D"]': .5})
    assert scored["joint_finalists_log_loss"] == math.inf
    assert scored["joint_qualification_log_loss"] == math.inf
    assert scored["joint_finalists_brier"] == pytest.approx(1.5)
    assert scored["joint_qualification_brier"] == pytest.approx(1.5)


def test_unavailable_joint_laws_keep_actual_targets_and_explicit_reasons():
    scored = _score(None)
    for key in ("joint_finalists_log_loss", "joint_finalists_brier",
                "joint_qualification_log_loss", "joint_qualification_brier"):
        assert scored[key] is None
        assert scored["unavailable_reasons"][key]
    assert scored["actual_joint_targets"]["finalists"] == '["A","B"]'


def _forecast(kind="monte_carlo", n=100):
    return {"champion_prob": {"A": .6, "B": .4, "C": 0., "D": 0.},
            "structural_zero_categories": {"champion": ["D"]},
            "integration": {"kind": kind, "fixed_n": n, "iid": True,
                            "stopping_rule": "fixed_n",
                            "effective_sampling_unit": ("independent_prefix" if kind == "hybrid"
                                                        else "independent_complete_path")}}


def test_zero_mc_hits_have_positive_upper_bound_and_are_not_structural_zeros():
    report = summarize_tournament_numerics(_forecast())
    sampled = report["targets"]["champion_prob"]["C"]
    structural = report["targets"]["champion_prob"]["D"]
    assert sampled["zero_kind"] == "sampling_zero"
    assert sampled["interval"][1] > 0
    assert sampled["zero_hit_upper_bound"] == pytest.approx(1 - .05 ** .01)
    assert structural["zero_kind"] == "structural_zero"
    assert structural["interval"] == [0., 0.]


def test_exact_reports_no_mc_error_without_claiming_model_accuracy():
    report = summarize_tournament_numerics(_forecast("exact"))
    leaf = report["targets"]["champion_prob"]["A"]
    assert leaf["sampling_error"] == 0.
    assert leaf["interval"] is None
    assert report["targets"]["champion_prob"]["C"]["zero_kind"] == "exact_law_zero"
    assert "model" in " ".join(report["caveats"]).lower()


def test_hybrid_fractional_conditional_means_use_bounded_not_binomial_error():
    forecast = _forecast("hybrid", n=3)
    forecast["champion_prob"] = {"A": .5, "B": .5}
    forecast["integration"]["prefix_moments"] = {
        "champion_prob": {"A": {"sum": 1.5, "sum_squares": .875, "n": 3}}}
    report = summarize_tournament_numerics(forecast)
    leaf = report["targets"]["champion_prob"]["A"]
    assert leaf["method"] == "bounded_hoeffding_fixed_n"
    assert leaf["standard_error"] == pytest.approx(math.sqrt(1 / 48))
    assert "hits" not in leaf
    assert "zero_hit_upper_bound" not in leaf


@pytest.mark.parametrize("changes", [
    {"effective_sampling_unit": "finite_indexed_vector"}, {"iid": False},
    {"stopping_rule": "adaptive"},
])
def test_non_iid_or_adaptive_designs_do_not_receive_binomial_intervals(changes):
    forecast = _forecast()
    forecast["integration"].update(changes)
    report = summarize_tournament_numerics(forecast)
    leaf = report["targets"]["champion_prob"]["A"]
    assert leaf["interval"] is None
    assert leaf["unsupported_reason"]


def test_malformed_empirical_fraction_does_not_masquerade_as_integer_hits():
    forecast = _forecast(n=3)
    with pytest.raises(ValueError, match="integer"):
        summarize_tournament_numerics(forecast)


def test_hybrid_moments_must_represent_bounded_probabilities_and_reported_mean():
    forecast = _forecast("hybrid", n=3)
    forecast["integration"]["prefix_moments"] = {
        "champion_prob": {"A": {"sum": 1., "sum_squares": 2., "n": 3}}}
    with pytest.raises(ValueError, match="moments"):
        summarize_tournament_numerics(forecast)


@pytest.mark.parametrize("observed", ["AB", ["A", "A"], ["A", "Z"], ["A"], ["A", []]])
def test_joint_finalist_target_rejects_malformed_members_and_cardinality(observed):
    with pytest.raises(ValueError):
        metrics.compute_joint_finalists_brier({'["A","B"]': 1.}, observed, TEAMS)


@pytest.mark.parametrize("kind", ["monte_carlo", "exact", "hybrid"])
def test_unknown_future_outcomes_do_not_disable_numerical_probability_reports(kind):
    forecast = _forecast(kind)
    forecast["target_resolved"] = {"champion": False, "champion_by_team": {"A": False}}
    forecast["stage_rank_prob"] = {"group": {"all": {"A": [.5, .5]}}}
    forecast["stage_rank_resolved"] = {"group": {"all": {"A": False}}}
    report = summarize_tournament_numerics(forecast)
    champion = report["targets"]["champion_prob"]["A"]
    rank = report["targets"]["stage_rank_prob"]["group"]["all"]["A"]["1"]
    if kind == "exact":
        assert champion["sampling_error"] == rank["sampling_error"] == 0.
    else:
        assert champion["interval"][0] < .6 < champion["interval"][1]
        assert rank["interval"][0] < .5 < rank["interval"][1]
    assert "unsupported_reason" not in champion
    assert "unsupported_reason" not in rank


def test_explicit_unmodeled_target_has_a_reason_instead_of_an_interval():
    forecast = _forecast()
    forecast["integration"]["unsupported_targets"] = {
        "champion_prob/A": "Declared target not integrated by this design"}
    report = summarize_tournament_numerics(forecast)
    unsupported = report["targets"]["champion_prob"]["A"]
    assert unsupported["interval"] is None
    assert unsupported["unsupported_reason"] == "Declared target not integrated by this design"
    assert report["targets"]["champion_prob"]["B"]["interval"] is not None


def test_explicit_structural_zero_cannot_override_positive_probability():
    forecast = _forecast()
    forecast["structural_zero_categories"]["champion"] = ["A"]
    with pytest.raises(ValueError, match="Structural-zero"):
        summarize_tournament_numerics(forecast)


def _joint_forecast(kind="monte_carlo", n=100):
    forecast = _forecast(kind, n)
    forecast["joint_final_prob"] = {'["A","C"]': .5, '["B","D"]': .5}
    forecast["joint_advance_prob"] = {"group": dict(forecast["joint_final_prob"])}
    forecast["integration"]["joint_target_design"] = {
        "teams": TEAMS, "finalist_count": 2, "advance_counts": {"group": 2}}
    if kind == "monte_carlo":
        counts = {'["A","C"]': n // 2, '["B","D"]': n // 2}
        forecast["hits"] = {"joint_final": counts, "joint_advance": {"group": dict(counts)}}
    return forecast


@pytest.mark.parametrize("law,stage", [
    ("joint_final_prob", None), ("joint_advance_prob", "group")])
def test_unsampled_legal_joint_event_has_pointwise_positive_mc_bound(law, stage):
    forecast = _joint_forecast()
    queried = query_tournament_joint_numerics(forecast, law, ["A", "B"], stage=stage)
    assert queried["estimate"] == 0.
    assert queried["zero_kind"] == "sampling_zero"
    assert queried["hits"] == 0
    assert queried["n"] == 100
    assert queried["zero_hit_upper_bound"] == pytest.approx(1 - .05 ** .01)
    assert queried["interval"][1] > 0
    assert queried["coverage_scope"] == "pointwise_fixed_n_numerical_only"
    assert '["A","B"]' not in forecast["joint_final_prob"]
    # A positive underlying event can be absent from a finite empirical law.
    underlying = _joint_forecast("exact")
    underlying["joint_final_prob"] = {
        '["A","B"]': .25, '["A","C"]': .25, '["B","D"]': .5}
    underlying["joint_advance_prob"]["group"] = dict(underlying["joint_final_prob"])
    truth = query_tournament_joint_numerics(underlying, law, ["A", "B"], stage=stage)
    assert truth["estimate"] == .25


def test_joint_exact_zero_and_structural_zero_are_distinct_from_mc_sampling_zero():
    exact = query_tournament_joint_numerics(
        _joint_forecast("exact"), "joint_final_prob", ["A", "B"])
    assert exact["zero_kind"] == "exact_law_zero"
    assert exact["sampling_error"] == 0.
    assert "zero_hit_upper_bound" not in exact
    forecast = _joint_forecast()
    forecast["structural_zero_categories"]["joint_final"] = ['["A","B"]']
    structural = query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "B"])
    assert structural["zero_kind"] == "structural_zero"
    assert structural["interval"] == [0., 0.]
    assert structural["sampling_error"] == 0.


def test_joint_summary_reports_observed_sets_without_enumerating_missing_sets():
    forecast = _joint_forecast()
    report = summarize_tournament_numerics(forecast)
    finalists = report["targets"]["joint_final_prob"]
    qualifiers = report["targets"]["joint_advance_prob"]["group"]
    assert set(finalists) == {'["A","C"]', '["B","D"]'}
    for leaf in (finalists['["A","C"]'], qualifiers['["A","C"]']):
        assert leaf["hits"] == 50
        assert leaf["standard_error"] == pytest.approx(.05)
        assert leaf["interval"][0] < .5 < leaf["interval"][1]


def test_integer_looking_hybrid_joint_mean_uses_fractional_prefix_moments_not_hits():
    forecast = _joint_forecast("hybrid", n=4)
    forecast["integration"]["prefix_moments"] = {
        "joint_final_prob": {'["A","C"]': {"sum": 2., "sum_squares": 1.25, "n": 4}}}
    queried = query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "C"])
    assert queried["method"] == "bounded_hoeffding_fixed_n"
    assert queried["observation_unit"] == "per_prefix_conditional_probability"
    assert queried["standard_error"] == pytest.approx(math.sqrt(1 / 48))
    assert "hits" not in queried
    zero = query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "B"])
    assert zero["zero_kind"] == "zero_sampled_conditional_mean"
    assert zero["interval"][1] == pytest.approx(math.sqrt(math.log(40) / 8))
    assert "zero_hit_upper_bound" not in zero


def test_joint_query_does_not_infer_iid_from_finite_vectors_or_integer_probabilities():
    forecast = _joint_forecast()
    forecast["integration"].update(iid=False, effective_sampling_unit="finite_indexed_vector")
    queried = query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "B"])
    assert queried["method"] == "unsupported"
    assert queried["interval"] is None
    assert queried["zero_kind"] == "unclassified_zero"
    assert "finite-vector" in queried["unsupported_reason"]


@pytest.mark.parametrize("members", [
    ["A", "A"], ["A", "Z"], ["A"], ["A", []], "AB",
    '["B","A"]', '["A", "B"]', '{"A":1,"B":1}',
])
def test_joint_query_rejects_invalid_members_and_noncanonical_encoded_sets(members):
    with pytest.raises(ValueError):
        query_tournament_joint_numerics(_joint_forecast(), "joint_final_prob", members)


@pytest.mark.parametrize("key", ['["C","A"]', '["A", "C"]', '["A","Z"]', '["A","A"]'])
def test_numerical_joint_reporting_rejects_corrupt_sparse_category_encodings(key):
    forecast = _joint_forecast()
    forecast["joint_final_prob"] = {key: .5, '["B","D"]': .5}
    with pytest.raises(ValueError):
        summarize_tournament_numerics(forecast)


def test_joint_query_validates_stage_and_declared_capacity_not_observed_keys():
    forecast = _joint_forecast()
    forecast["joint_advance_prob"]["invented"] = {'["A"]': 1.}
    with pytest.raises(ValueError, match="stage"):
        query_tournament_joint_numerics(forecast, "joint_advance_prob", ["A"], stage="invented")
    with pytest.raises(ValueError, match="members"):
        query_tournament_joint_numerics(forecast, "joint_advance_prob", ["A"], stage="group")
    with pytest.raises(ValueError, match="stage"):
        query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "B"], stage="group")


def test_joint_query_requires_actual_empirical_counts_and_rejects_inconsistent_counts():
    forecast = _joint_forecast()
    forecast.pop("hits")
    queried = query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "C"])
    assert queried["method"] == "unsupported"
    assert queried["interval"] is None
    assert "counts" in queried["unsupported_reason"]
    forecast = _joint_forecast()
    forecast["hits"]["joint_final"]['["A","C"]'] = 49
    forecast["hits"]["joint_final"]['["B","D"]'] = 51
    with pytest.raises(ValueError, match="counts"):
        query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "C"])


def test_joint_query_requires_authoritative_design_not_probability_keys():
    forecast = _joint_forecast()
    forecast["integration"].pop("joint_target_design")
    with pytest.raises(ValueError, match="design"):
        query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "C"])


def test_actual_joint_sampling_diagnostic_does_not_floor_infinite_observed_loss():
    forecast = _joint_forecast()
    forecast["final_prob"] = dict.fromkeys(TEAMS, .5)
    forecast["advance_prob"] = {"group": dict.fromkeys(TEAMS, .5)}
    event = {"id": "toy", "edition_id": "toy-edition", "teams": TEAMS}
    labels = {"event_id": "toy", "edition_id": "toy-edition", "finalists": ["A", "B"],
              "qualifiers": {"group": ["A", "B"]}}
    scored = metrics.score_research_event(event, forecast, labels)
    assert scored["joint_finalists_log_loss"] == math.inf
    assert scored["joint_qualification_log_loss"] == math.inf
    diagnostic = scored["actual_joint_numerics"]["finalists"]
    assert diagnostic["estimate"] == 0.
    assert diagnostic["zero_hit_upper_bound"] > 0.
    assert diagnostic["coverage_scope"] == "pointwise_fixed_n_numerical_only"


def test_structural_joint_zero_cannot_override_a_positive_empirical_event():
    forecast = _joint_forecast()
    forecast["structural_zero_categories"]["joint_final"] = ['["A","C"]']
    with pytest.raises(ValueError, match="Structural-zero"):
        query_tournament_joint_numerics(forecast, "joint_final_prob", ["A", "C"])


def test_joint_target_exclusion_preserves_issued_estimate_without_numeric_interval():
    forecast = _joint_forecast()
    forecast["integration"]["unsupported_targets"] = {
        'joint_advance_prob/group/["A","C"]': "Stage not included in the supplied iid design"}
    queried = query_tournament_joint_numerics(
        forecast, "joint_advance_prob", ["A", "C"], stage="group")
    assert queried["estimate"] == .5
    assert queried["method"] == "unsupported"
    assert queried["interval"] is None
    assert queried["unsupported_reason"] == "Stage not included in the supplied iid design"


def test_hybrid_joint_moments_cannot_describe_unbounded_or_inconsistent_contributions():
    forecast = _joint_forecast("hybrid", n=4)
    forecast["integration"]["prefix_moments"] = {
        "joint_advance_prob": {"group": {
            '["A","C"]': {"sum": 2., "sum_squares": 3., "n": 4}}}}
    with pytest.raises(ValueError, match="moments"):
        query_tournament_joint_numerics(
            forecast, "joint_advance_prob", ["A", "C"], stage="group")
