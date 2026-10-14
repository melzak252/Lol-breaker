"""Native tournament score contracts: strict probability laws and edition-level gates.

Small probability/score fixtures are analytic examples, not historical evidence.
No artifacts, database access, model fitting, or bracket simulation are required.
"""
from copy import deepcopy
import math

import pytest

from src.analysis import tournament_metrics as metrics
from src.models.tournament_scorelines import ConditionalScorelineModel


@pytest.mark.parametrize(
    "champion,final,advance",
    [
        ({"A": .4, "B": .2}, {"A": .5, "B": .3}, None),
        ({"A": .6, "B": .4}, {"A": .8, "B": .8}, None),
        ({"A": 1.1, "B": -.1}, {"A": 1., "B": 1.}, None),
        ({"A": .6, "B": .4}, {"A": 1.2, "B": .8}, None),
        ({"A": .6, "B": .4}, {"A": 1., "C": 1.}, None),
        ({"A": .6, "B": .4}, {"A": 1.}, None),
        ({"A": .6, "B": .4}, {"A": 1., "B": 1.}, {"A": 1., "C": 1.}),
        ({"A": .6, "B": .4}, {"A": 1., "B": 1.}, {"A": 1., "B": .9}),
    ],
    ids=["leaked-both-masses", "leaked-final-mass", "invalid-champion-bounds",
         "invalid-final-bounds", "wrong-final-members", "missing-final-member",
         "wrong-advance-members", "fractional-advance-capacity"],
)
def test_flow_lp_rejects_the_supplied_marginals_without_repair(champion, final, advance):
    before = deepcopy((champion, final, advance))
    result = metrics.verify_network_flow_conservation(champion, final, advance)
    assert result["feasible"] is False
    assert (champion, final, advance) == before


def test_flow_lp_rejects_reach_inequality_even_when_all_masses_are_correct():
    result = metrics.verify_network_flow_conservation(
        {"A": .8, "B": .1, "C": .1},
        {"A": .7, "B": .7, "C": .6},
    )
    assert result["feasible"] is False


def test_flow_lp_accepts_fixed_consistent_marginals_including_zero_champion_mass():
    result = metrics.verify_network_flow_conservation(
        {"A": .7, "B": .3, "C": 0., "D": 0.},
        {"A": .9, "B": .8, "C": .2, "D": .1},
        {"A": 1., "B": 1., "C": .6, "D": .4},
    )
    assert result["feasible"] is True
    assert result["max_flow_violation"] == pytest.approx(0.)


def test_real_zero_champion_probability_is_infinite_not_epsilon_floored():
    assert metrics.compute_champion_log_loss({"A": 0., "B": 1.}, "A") == math.inf
    assert metrics.compute_champion_brier({"A": 0., "B": 1.}, "A") == pytest.approx(2.)


def test_real_zero_score_probability_is_infinite_not_epsilon_floored():
    pmf = {(2, 0): 0., (2, 1): .4, (1, 2): .3, (0, 2): .3}
    assert metrics.compute_scoreline_log_loss(pmf, (2, 0)) == math.inf


@pytest.mark.parametrize("scorer", [metrics.compute_champion_log_loss, metrics.compute_champion_brier])
def test_absent_actual_champion_is_malformed_not_a_zero_probability(scorer):
    with pytest.raises(ValueError):
        scorer({"A": .4, "B": .6}, "C")


def test_absent_observed_score_category_is_malformed_not_a_zero_probability():
    with pytest.raises(ValueError):
        metrics.compute_scoreline_log_loss({(2, 0): .5, (0, 2): .5}, (2, 1))


@pytest.mark.parametrize("probabilities", [{"A": 1.2, "B": -.2}, {"A": math.nan, "B": .5}])
def test_champion_scoring_rejects_invalid_probability_laws(probabilities):
    with pytest.raises(ValueError):
        metrics.compute_champion_log_loss(probabilities, "A")


def test_advancement_brier_divides_by_all_teams_not_qualifier_count():
    probabilities = {"A": .8, "B": .6, "C": .4, "D": .2}
    # (.2**2 + .4**2 + .4**2 + .2**2) / 4, not / 2.
    assert metrics.compute_advancement_brier(probabilities, ["A", "B"]) == pytest.approx(.1)


def test_finalists_brier_keeps_the_two_finalist_normalization():
    assert metrics.compute_finalists_brier(
        {"A": .9, "B": .8, "C": .2, "D": .1}, ["A", "B"]
    ) == pytest.approx(.05)


def test_swiss_rps_uses_official_record_order_not_mapping_insertion_order():
    probabilities = {"A": {"0-3": .1, "1-3": .2, "3-2": .3,
                            "3-0": .1, "2-3": .1, "3-1": .2}}
    # CDF errors: .1, .3, -.4, -.3, -.1; normalized by five boundaries.
    assert metrics.compute_swiss_record_rps(probabilities, {"A": "3-2"}) == pytest.approx(.072)


def test_series_length_rps_uses_ordered_games_played():
    # Actual four maps: CDF errors .2 and -.5 at lengths three and four.
    assert metrics.compute_series_length_rps({5: .5, 3: .2, 4: .3}, 4, 5) == pytest.approx(.145)


def _pair_table(probability, *, reverse=False):
    return {
        "probability_unit": "series_win",
        "rows": [{"team_a": "B" if reverse else "A", "team_b": "A" if reverse else "B",
                  "best_of": 5, "p": probability}],
    }


def _series_case():
    event = {"id": "analytic-final", "edition_id": "analytic-edition", "year": 2026,
             "teams": ["A", "B"], "scope": "full_tournament"}
    forecast = {"champion_prob": {"A": .7, "B": .3},
                "final_prob": {"A": 1., "B": 1.}, "advance_prob": {},
                "placement_prob": None, "swiss_record_prob": None}
    labels = {"event_id": event["id"], "edition_id": event["edition_id"],
              "champion": "A", "finalists": ["A", "B"], "qualifiers": {},
              "placement_bands": None, "swiss_records": None,
              "matches": [{"team_a": "A", "team_b": "B", "best_of": 5,
                           "wins_a": 3, "wins_b": 1, "date": "2026-06-02",
                           "match_id": "analytic-final-match"}]}
    return event, forecast, labels


def test_research_scoreline_loss_uses_actual_pair_probability_with_the_same_real_law():
    event, forecast, labels = _series_case()
    law = ConditionalScorelineModel(beta_momentum=.35)
    low = metrics.score_research_event(event, forecast, labels, _pair_table(.25), law)
    high = metrics.score_research_event(event, forecast, labels, _pair_table(.8), law)
    assert low["scoreline_log_loss"] == pytest.approx(-math.log(law.scoreline_pmf(.25, 5)[3, 1]))
    assert high["scoreline_log_loss"] == pytest.approx(-math.log(law.scoreline_pmf(.8, 5)[3, 1]))
    assert high["scoreline_log_loss"] < low["scoreline_log_loss"]


def test_research_scoreline_scoring_complements_reverse_pair_orientation():
    event, forecast, labels = _series_case()
    law = ConditionalScorelineModel(beta_momentum=.35)
    direct = metrics.score_research_event(event, forecast, labels, _pair_table(.8), law)
    reverse = metrics.score_research_event(event, forecast, labels, _pair_table(.2, reverse=True), law)
    assert reverse["scoreline_log_loss"] == pytest.approx(direct["scoreline_log_loss"])
    assert reverse["series_length_rps"] == pytest.approx(direct["series_length_rps"])


def test_legacy_edition_evaluator_does_not_replace_supplied_pair_probability_with_half():
    _, forecast, _ = _series_case()
    truth = metrics.TournamentGroundTruth(
        event_title="Analytic final", champion="A", finalists=("A", "B"),
        swiss_qualifiers=None, swiss_records=None,
        playoff_matches=(metrics.TournamentMatchOutcome(
            stage="playoffs", round_name="Final", team1="A", team2="B",
            score1=3, score2=1, best_of=5, winner="A"),),
    )
    law = ConditionalScorelineModel(beta_momentum=.35)
    low = metrics.evaluate_tournament_edition(
        "analytic-edition", forecast, truth, pair_table=_pair_table(.25), scoreline_model=law)
    high = metrics.evaluate_tournament_edition(
        "analytic-edition", forecast, truth, pair_table=_pair_table(.8), scoreline_model=law)
    assert low["playoff_scoreline_log_loss"] == pytest.approx(
        -math.log(law.scoreline_pmf(.25, 5)[3, 1]), abs=1e-6)
    assert high["playoff_scoreline_log_loss"] == pytest.approx(
        -math.log(law.scoreline_pmf(.8, 5)[3, 1]), abs=1e-6)
    assert high["playoff_scoreline_log_loss"] < low["playoff_scoreline_log_loss"]


def _placement_case():
    event = {"id": "analytic-playoffs", "edition_id": "analytic-four-team", "year": 2026,
             "teams": ["A", "B", "C", "D"], "scope": "full_tournament"}
    forecast = {
        "champion_prob": {"A": .6, "B": .3, "C": .05, "D": .05},
        "final_prob": {"A": .9, "B": .9, "C": .1, "D": .1},
        "advance_prob": {"qualify": {"A": .9, "B": .9, "C": .1, "D": .1}},
        "placement_categories": ["1", "2", "3-4"],
        "placement_prob": {
            "A": {"3-4": .1, "2": .3, "1": .6},
            "B": {"3-4": .1, "2": .6, "1": .3},
            "C": {"3-4": .9, "2": .05, "1": .05},
            "D": {"3-4": .9, "2": .05, "1": .05},
        },
        "swiss_record_prob": None,
    }
    labels = {"event_id": event["id"], "edition_id": event["edition_id"],
              "champion": "A", "finalists": ["A", "B"],
              "qualifiers": {"qualify": ["A", "B"]},
              "placement_bands": {"A": [1, 1], "B": [2, 2], "C": [3, 4], "D": [3, 4]},
              "swiss_records": None, "matches": []}
    return event, forecast, labels


def test_research_placement_rps_scores_official_ties_as_one_ordered_band():
    event, forecast, labels = _placement_case()
    result = metrics.score_research_event(event, forecast, labels)
    # Team RPS: A=.085, B=.05, C=D=.00625, with two ordered band boundaries.
    assert result["placement_rps"] == pytest.approx(.036875)
    assert result["advance_brier"] == pytest.approx(.01)


def test_qualification_route_has_no_invented_champion_or_finalist_target():
    event, forecast, labels = _placement_case()
    event["scope"] = "qualification_route"
    forecast.update(champion_prob=None, final_prob=None, placement_prob=None)
    labels.update(champion=None, finalists=None, placement_bands=None)
    result = metrics.score_research_event(event, forecast, labels)
    assert result["advance_brier"] == pytest.approx(.01)
    for key in ("champion_log_loss", "champion_brier", "finalists_brier", "placement_rps"):
        assert result[key] is None
        assert result["unavailable_reasons"][key]


def test_research_missing_champion_member_is_rejected_instead_of_scored_as_zero():
    event, forecast, labels = _series_case()
    forecast["champion_prob"] = {"B": 1.}
    with pytest.raises(ValueError):
        metrics.score_research_event(event, forecast, labels, _pair_table(.7), ConditionalScorelineModel())


def test_research_observed_champion_with_zero_forecast_remains_infinite():
    event, forecast, labels = _series_case()
    forecast["champion_prob"] = {"A": 0., "B": 1.}
    result = metrics.score_research_event(event, forecast, labels, _pair_table(.7), ConditionalScorelineModel())
    assert result["champion_log_loss"] == math.inf
    assert result["champion_brier"] == pytest.approx(2.)


PRIMARY_METRICS = ("advance_brier", "placement_rps")
MODELS = ("native_c0_unchanged", "residual_independent_matched_marginals", "residual_coherent")


def _edition_scores(count=8):
    rows = []
    for edition in range(count):
        # Distinct edition losses and improvements; equal edition weighting is intentional.
        base = .24 + .007 * edition
        for model, gain in zip(MODELS, (0., .02, .04 + .001 * edition), strict=True):
            rows.append({
                "event_id": f"edition-{edition}-main", "edition_id": f"edition-{edition}",
                "year": 2026, "model": model, "eligible": True,
                "metrics": {"advance_brier": base - gain,
                            "placement_rps": .16 + .004 * edition - .5 * gain,
                            "champion_log_loss": .8 + .03 * edition - gain},
                "target_counts": {"advance_brier": 4 + 2 * edition,
                                  "placement_rps": 4 + 2 * edition,
                                  "champion_log_loss": 1},
                "structural_checks_passed": True, "chronology_checks_passed": True,
            })
    return rows


def _series_guard():
    return {"log_loss_delta_ci_upper": .001, "brier_delta": -.0005,
            "coverage_complete": True, "structural_checks_passed": True,
            "chronology_checks_passed": True}


def _decide(rows, guard=None):
    return metrics.tournament_study_decision(
        rows, _series_guard() if guard is None else guard,
        minimum_editions=8, bootstraps=200, seed=20261006)


def _comparison(decision, metric, reference):
    return next(row for row in decision["comparisons"]
                if row["model"] == "residual_coherent" and row["reference"] == reference
                and row["metric"] == metric)


def test_eight_distinct_supported_editions_can_retain_only_a_historical_candidate():
    decision = _decide(_edition_scores())
    assert decision["decision"] == "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"
    assert decision["eligible_editions"] == 8
    for metric, expected in (("advance_brier", -.0435), ("placement_rps", -.02175)):
        contrast = _comparison(decision, metric, "native_c0_unchanged")
        assert contrast["mean_delta"] == pytest.approx(expected)
        assert contrast["ci_upper"] < 0.
        assert contrast["editions"] == 8
        matched = _comparison(decision, metric, "residual_independent_matched_marginals")
        assert matched["mean_delta"] == pytest.approx(expected + (.02 if metric == "advance_brier" else .01))


def test_seven_editions_cannot_promote_even_with_strictly_better_primary_metrics():
    decision = _decide(_edition_scores(7))
    assert decision["decision"] == "NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE"
    assert decision["eligible_editions"] == 7


def test_two_genuine_phases_of_each_edition_do_not_double_independent_support():
    rows = _edition_scores(4)
    for original in list(rows):
        phase = deepcopy(original)
        phase["event_id"] = original["event_id"].replace("-main", "-qualification")
        # Different phase targets/scores, not duplicate forecasts of the same target.
        phase["metrics"]["advance_brier"] += .011
        phase["metrics"]["placement_rps"] += .006
        phase["target_counts"] = {"advance_brier": 2, "placement_rps": 2, "champion_log_loss": 1}
        rows.append(phase)
    decision = _decide(rows)
    assert decision["eligible_editions"] == 4
    assert decision["decision"] == "NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE"


@pytest.mark.parametrize("metric", PRIMARY_METRICS)
def test_one_failed_primary_contrast_blocks_retention_despite_the_other_improving(metric):
    rows = _edition_scores()
    baseline = {row["edition_id"]: row["metrics"][metric]
                for row in rows if row["model"] == "native_c0_unchanged"}
    for row in rows:
        if row["model"] == "residual_coherent":
            row["metrics"][metric] = baseline[row["edition_id"]] + .01
    assert _decide(rows)["decision"] == "NOT_ADOPTED"


@pytest.mark.parametrize("metric", PRIMARY_METRICS)
@pytest.mark.parametrize("value", [None, math.inf])
def test_unavailable_or_infinite_primary_metric_is_not_dropped_to_create_a_passing_cohort(metric, value):
    rows = _edition_scores()
    affected = next(row for row in rows if row["model"] == "residual_coherent")
    affected["metrics"][metric] = value
    if value is None:
        affected["unavailable_reasons"] = {metric: "No supported target law"}
    assert _decide(rows)["decision"] != "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


def test_infinite_available_secondary_loss_blocks_retention_instead_of_being_floored():
    rows = _edition_scores()
    next(row for row in rows if row["model"] == "residual_coherent")["metrics"]["champion_log_loss"] = math.inf
    assert _decide(rows)["decision"] != "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


@pytest.mark.parametrize("reference", ["native_c0_unchanged", "residual_independent_matched_marginals"])
def test_a_missing_mandatory_comparator_cannot_be_treated_as_success(reference):
    rows = [row for row in _edition_scores() if row["model"] != reference]
    assert _decide(rows)["decision"] != "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


@pytest.mark.parametrize("metric", PRIMARY_METRICS)
def test_primary_target_support_count_is_required_and_must_agree_across_arms(metric):
    rows = _edition_scores()
    affected = next(row for row in rows if row["model"] == "residual_coherent")
    affected["target_counts"][metric] += 1
    assert _decide(rows)["decision"] != "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


@pytest.mark.parametrize("field,value", [
    ("log_loss_delta_ci_upper", .00201),
    ("log_loss_delta_ci_upper", math.inf),
    ("log_loss_delta_ci_upper", None),
    ("brier_delta", .00001),
    ("brier_delta", None),
    ("coverage_complete", False),
    ("structural_checks_passed", False),
    ("chronology_checks_passed", False),
])
def test_every_canonical_series_guard_is_mandatory(field, value):
    guard = _series_guard()
    guard[field] = value
    assert _decide(_edition_scores(), guard)["decision"] == "NOT_ADOPTED"


def test_series_log_loss_limit_is_inclusive_and_zero_brier_delta_is_allowed():
    guard = _series_guard()
    guard.update(log_loss_delta_ci_upper=.002, brier_delta=0.)
    assert _decide(_edition_scores(), guard)["decision"] == "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


@pytest.mark.parametrize("field", ["structural_checks_passed", "chronology_checks_passed"])
def test_one_failed_event_gate_cannot_be_hidden_by_the_rest_of_the_editions(field):
    rows = _edition_scores()
    next(row for row in rows if row["model"] == "residual_coherent")[field] = False
    assert _decide(rows)["decision"] != "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


@pytest.mark.parametrize("support", [None, 0, -1])
def test_numeric_primary_metrics_without_positive_target_counts_cannot_promote(support):
    rows = _edition_scores()
    for row in rows:
        if support is None:
            row["target_counts"].pop("advance_brier")
        else:
            row["target_counts"]["advance_brier"] = support
    assert _decide(rows)["decision"] != "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED"


@pytest.mark.parametrize("field", [
    "log_loss_delta_ci_upper", "brier_delta", "coverage_complete",
    "structural_checks_passed", "chronology_checks_passed",
])
def test_omitted_series_guard_cannot_default_to_a_passing_gate(field):
    guard = _series_guard()
    del guard[field]
    assert _decide(_edition_scores(), guard)["decision"] == "NOT_ADOPTED"


def test_fractional_advancement_mass_is_not_rounded_into_a_stage_capacity():
    result = metrics.verify_network_flow_conservation(
        {"A": .4, "B": .3, "C": .2, "D": .1},
        {"A": .8, "B": .6, "C": .4, "D": .2},
        {"A": .9, "B": .8, "C": .7, "D": .5},
    )
    assert result["feasible"] is False


def test_calibration_editions_do_not_fill_the_assessment_support_requirement():
    rows = _edition_scores()
    for row in rows:
        if row["edition_id"] == "edition-0":
            row["year"] = 2025
    decision = _decide(rows)
    assert decision["eligible_editions"] == 7
    assert decision["decision"] == "NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE"


def test_adverse_coherent_vs_matched_contrast_is_reported_not_masked():
    rows = _edition_scores()
    for row in rows:
        if row["model"] == "residual_independent_matched_marginals":
            row["metrics"]["advance_brier"] -= .06
            row["metrics"]["placement_rps"] -= .03
    decision = _decide(rows)
    for metric, expected in (("advance_brier", .0365), ("placement_rps", .01825)):
        contrast = _comparison(decision, metric, "residual_independent_matched_marginals")
        assert contrast["mean_delta"] == pytest.approx(expected)
        assert contrast["ci_lower"] > 0.
        assert contrast["editions"] == 8


@pytest.mark.parametrize("failure", ["missing_value", "missing_key", "mismatched_count"])
def test_declared_secondary_target_failure_cannot_retain_candidate(failure):
    rows = _edition_scores()
    row = next(item for item in rows if item["model"] == "residual_coherent")
    if failure == "missing_value":
        row["metrics"]["champion_log_loss"] = None
        row["unavailable_reasons"] = {"champion_log_loss": "Forecast law unavailable"}
    elif failure == "missing_key":
        del row["metrics"]["champion_log_loss"]
    else:
        row["target_counts"]["champion_log_loss"] = 2
    decision = _decide(rows)
    assert decision["decision"] == "NOT_ADOPTED"
    assert decision["eligible_editions"] == 8
