"""Observable contracts for side-specific ensemble betting probabilities."""

import math
import json

import pytest

from src.analysis.unified_evaluation_engine import (
    compute_ensemble_p_low,
    compute_quantile_p_low,
)
from betting_app.services.bet_qualification_service import is_bet_eligible


def test_ensemble_lower_bounds_are_finite_monotone_and_side_specific():
    for p in (1e-8, 0.4, 0.9, 1 - 1e-8):
        for mode in ("logit", "linear"):
            lo = compute_ensemble_p_low(p, 0.05, 0.5, mode)
            hi = compute_ensemble_p_low(p, 0.1, 1.25, mode)
            assert math.isfinite(hi) and 0 < hi <= lo <= p
    assert compute_ensemble_p_low(0.4, 0.1, 0) == pytest.approx(0.4)
    assert compute_quantile_p_low(0.6, 0.55) == 0.55
    assert compute_quantile_p_low(0.4, 1 - 0.7) == pytest.approx(0.3)


def test_conservative_score_can_reject_mean_positive_ev():
    params = dict(prob_model=0.6, odds=2.1, tax_rate=0.12, min_ev_net=0.03,
                  dynamic_hurdle=False, max_ev_longshot=0.2,
                  max_coinflip_market_gap=None, max_bo1_market_gap=None,
                  max_tier1_market_gap=None, max_negative_clv_drift=None)
    mean = is_bet_eligible(uncertainty_required=False, **params)
    cautious = is_bet_eligible(
        prob_conservative=compute_ensemble_p_low(0.6, 0.3, 1),
        uncertainty_required=True, **params,
    )
    assert mean[0] is True
    assert cautious[0] is False
    assert cautious[2]["ev_net"] < mean[2]["ev_net"]


def test_quantile_b_side_uses_complement_of_q90():
    from src.analysis.ev_approach_policies import policy_probability

    assert policy_probability(
        0.6, "P4", side="A", p_q10=0.5, p_q90=0.8
    ) == 0.5
    assert policy_probability(
        0.4, "P4", side="B", p_q10=0.5, p_q90=0.8
    ) == pytest.approx(0.2)




def test_empty_policy_portfolio_has_zero_turnover_without_crashing():
    from src.analysis.unified_evaluation_engine import UnifiedBettingEngine

    result = UnifiedBettingEngine().run_simulation([], strategy="quarter_kelly")
    assert result.bets_count == 0
    assert result.total_staked == 0
    assert result.total_pnl == 0
    assert result.final_bankroll == result.initial_bankroll
    assert result.realized_ev_pct is None
    assert result.roi_pct is None
    assert result.expected_ev_pct is None
    assert result.raw_ev_pct is None
    assert result.expected_profit_per_bet is None
    assert result.realized_profit_per_bet is None
    assert result.calibration_gap_pln is None
    assert result.win_rate_pct is None
    assert result.avg_odds is None
    assert result.avg_stake is None
    assert result.to_dict()["result_status"] == "no_eligible_bets"
    assert result.bins == []
    assert result.by_odds_bracket == []


@pytest.mark.parametrize(
    "basis,reason",
    [
        pytest.param(
            "measured_bounds", "provided_lower_bound", id="measured",
        ),
        pytest.param(
            "assumed_research_haircut", "explicit_research_assumption",
            id="assumed",
        ),
    ],
)
def test_qualification_keeps_identical_scores_distinct_by_evidence(basis, reason):
    eligible, rejection, diagnostics = is_bet_eligible(
        prob_model=0.65,
        odds=2.0,
        prob_conservative=0.62,
        uncertainty_required=True,
        uncertainty_basis=basis,
    )

    assert eligible is True
    assert rejection == "eligible"
    assert diagnostics["prob_conservative"] == 0.62
    assert diagnostics["prob_for_ev"] == 0.62
    assert diagnostics["ev_net"] == pytest.approx(0.0912)
    assert diagnostics["uncertainty_basis"] == basis
    assert diagnostics["uncertainty_reason"] == reason


@pytest.fixture
def saved_c0_quote(tmp_path):
    """One persisted mean-only forecast; scenarios supply evidence, not new means."""
    from betting_app.core.models import C0_NATIVE

    path = tmp_path / "c0_quote.json"
    path.write_text(json.dumps({
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "match_id": 1,
        "side": "team_a",
        "prob_model": 0.65,
        "odds": 2.0,
        "won": True,
        "diagnostics": {
            "base_model_name": C0_NATIVE.name,
            "uncertainty_required": False,
            "epistemic_sigma_z": None,
            "p_low_a": None,
            "p_low_b": None,
        },
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "basis,required,lower_bound,eligible,reason",
    [
        pytest.param("mean", False, None, True, None, id="mean"),
        pytest.param("measured_bounds", True, 0.62, True, None, id="bounded"),
        pytest.param(
            "assumed_research_haircut", True, "assumed", False,
            "insufficient_ev", id="assumed",
        ),
        pytest.param(
            "missing_bounds", True, None, False, "missing_uncertainty",
            id="missing",
        ),
    ],
)
def test_saved_c0_quote_preserves_uncertainty_identity_and_selection(
    saved_c0_quote, basis, required, lower_bound, eligible, reason
):
    from src.analysis.unified_evaluation_engine import UnifiedBettingEngine

    quote = json.loads(saved_c0_quote.read_text(encoding="utf-8"))
    probability = quote["prob_model"]
    if lower_bound == "assumed":
        # Explicit historical research assumption, NOT measured native C0 sigma.
        lower_bound = 1 / (1 + math.exp(
            -(math.log(probability / (1 - probability)) - 0.75 * 0.30)
        ))
    engine = UnifiedBettingEngine(uncertainty_required=required)
    kwargs = {key: quote[key] for key in ("match_id", "side", "odds", "prob_model", "won")}
    kwargs["prob_conservative"] = lower_bound
    if basis in {"measured_bounds", "assumed_research_haircut"}:
        kwargs["uncertainty_basis"] = basis

    selected, bet = engine.qualify_quote(**kwargs)

    assert selected is eligible
    assert bet.prob_model == probability
    assert bet.prob_conservative == lower_bound
    assert bet.quarantine_reason == reason
    payload = bet.to_dict()
    assert payload["uncertainty_basis"] == basis
    assert payload["uncertainty_reason"] == {
        "mean": "uncertainty_not_requested",
        "measured_bounds": "provided_lower_bound",
        "assumed_research_haircut": "explicit_research_assumption",
        "missing_bounds": "missing_uncertainty",
    }[basis]
    if basis == "missing_bounds":
        assert bet.ev_net is None
        assert bet.ev_calibrated is None
        assert bet.prob_for_ev is None
    else:
        selected_probability = probability if lower_bound is None else lower_bound
        assert bet.prob_for_ev == selected_probability
        assert bet.ev_net == pytest.approx(
            round(selected_probability * quote["odds"] * 0.88 - 1, 4)
        )
    # Qualification does not rewrite or supplement the saved native forecast.
    assert json.loads(saved_c0_quote.read_text(encoding="utf-8")) == quote


@pytest.mark.parametrize("required", [False, True], ids=["mean", "missing"])
def test_qualification_diagnostics_do_not_present_mean_as_measured_bound(required):
    eligible, reason, diagnostics = is_bet_eligible(
        prob_model=0.65, odds=2.0, uncertainty_required=required,
    )

    assert eligible is not required
    assert reason == ("missing_uncertainty" if required else "eligible")
    assert diagnostics["uncertainty_basis"] == ("missing_bounds" if required else "mean")
    assert diagnostics["uncertainty_reason"] == (
        "missing_uncertainty" if required else "uncertainty_not_requested"
    )
    assert diagnostics["prob_conservative"] is None
    assert diagnostics["prob_for_ev"] == (None if required else 0.65)
    assert diagnostics.get("ev_net") == (None if required else pytest.approx(0.144))


def test_native_c0_read_side_keeps_absent_uncertainty_absent(saved_c0_quote):
    from betting_app.services.bet_qualification_service import qualify_prediction_sides

    quote = json.loads(saved_c0_quote.read_text(encoding="utf-8"))
    sides = qualify_prediction_sides(
        prob_a=quote["prob_model"],
        prob_b=1 - quote["prob_model"],
        diagnostics=quote["diagnostics"],
        model_name=quote["model_name"],
        odds_a=quote["odds"],
        odds_b=quote["odds"],
        market_prob_a=0.5,
    )

    assert sides["a"][0:2] == (True, "eligible")
    assert sides["b"][0:2] == (False, "insufficient_ev")
    for _eligible, _reason, diagnostics in sides.values():
        assert diagnostics["uncertainty_basis"] == "mean"
        assert diagnostics["prob_conservative"] is None
        assert diagnostics["uncertainty_required"] is False
    assert sides["a"][2]["prob_for_ev"] == quote["prob_model"]
    assert sides["b"][2]["prob_for_ev"] == 1 - quote["prob_model"]


@pytest.mark.parametrize("basis", ["measured_bounds", "assumed_research_haircut"])
def test_prediction_side_qualification_preserves_declared_bound_evidence(
    saved_c0_quote, basis,
):
    from betting_app.services.bet_qualification_service import qualify_prediction_sides

    quote = json.loads(saved_c0_quote.read_text(encoding="utf-8"))
    sides = qualify_prediction_sides(
        prob_a=0.65, prob_b=0.35,
        diagnostics={
            **quote["diagnostics"],
            "uncertainty_required": True,
            "p_low_a": 0.62, "p_low_b": 0.30,
            "rating_disagreement": 0.0,
            "uncertainty_basis": basis,
            "uncertainty_reason": "stored_side_specific_evidence",
        },
        model_name=quote["model_name"],
        odds_a=2.0, odds_b=2.0, market_prob_a=0.5,
    )

    assert sides["a"][0:2] == (True, "eligible")
    assert sides["b"][0:2] == (False, "insufficient_ev")
    for side, lower in (("a", 0.62), ("b", 0.30)):
        decision = sides[side][2]
        assert decision["prob_conservative"] == lower
        assert decision["prob_for_ev"] == lower
        assert decision["uncertainty_basis"] == basis
        assert decision["uncertainty_reason"] == "stored_side_specific_evidence"


def test_prediction_sides_missing_required_bounds_report_exclusion_without_ev(
    saved_c0_quote,
):
    from betting_app.services.bet_qualification_service import qualify_prediction_sides

    quote = json.loads(saved_c0_quote.read_text(encoding="utf-8"))
    sides = qualify_prediction_sides(
        prob_a=0.65, prob_b=0.35,
        diagnostics={**quote["diagnostics"], "uncertainty_required": True},
        model_name=quote["model_name"],
        odds_a=2.0, odds_b=2.0, market_prob_a=0.5,
    )

    for eligible, reason, decision in sides.values():
        assert eligible is False
        assert reason == "missing_uncertainty"
        assert decision["uncertainty_basis"] == "missing_bounds"
        assert decision["uncertainty_reason"] == "missing_uncertainty"
        assert decision["prob_conservative"] is None
        assert decision["prob_for_ev"] is None
        assert decision["ev_net"] is None


def test_prediction_sides_preserve_required_rating_disagreement_safety(
    saved_c0_quote,
):
    from betting_app.services.bet_qualification_service import qualify_prediction_sides

    quote = json.loads(saved_c0_quote.read_text(encoding="utf-8"))
    sides = qualify_prediction_sides(
        prob_a=0.65, prob_b=0.35,
        diagnostics={
            **quote["diagnostics"], "uncertainty_required": True,
            "p_low_a": 0.62, "p_low_b": 0.30,
        },
        model_name=quote["model_name"],
        odds_a=2.0, odds_b=2.0, market_prob_a=0.5,
    )

    for eligible, reason, _decision in sides.values():
        assert eligible is False
        assert reason == "invalid_prediction_safety"


def test_runner_groups_preserve_cohort_horizon_as_the_group_key():
    from src.analysis.ev_approach_runner import _opportunity_groups
    import pandas as pd

    frame = pd.DataFrame([
        {"cohort": "archival_market", "horizon": "OPEN", "row_id": "a", "date": "2026-01-01"},
        {"cohort": "production", "horizon": "48h", "row_id": "b", "date": "2026-01-02"},
    ])
    groups = list(_opportunity_groups(frame))

    assert [key for key, _ in groups] == [
        ("archival_market", "OPEN"), ("production", "48h"),
    ]
    assert [part.row_id.tolist() for _, part in groups] == [["a"], ["b"]]


def test_asof_cohort_uses_native_replay_and_retains_unavailable_rows():
    from src.analysis.ev_approach_runner import _asof_cohort_inputs
    import numpy as np
    import pandas as pd

    opportunities = pd.DataFrame([
        {
            "row_id": "production:m1:OPEN", "golgg_match_id": "m1",
            "cohort": "production", "horizon": "OPEN", "date": "2026-01-02",
            "decision_at": pd.NaT, "scenario_decision_at": "2026-01-01T00:00:00Z",
            "start_at": "2026-01-02T12:00:00Z", "label_available_at": pd.NaT,
            "scenario_label_available_at": "2026-01-03T00:00:00Z", "y": 1,
            "p_s": np.nan, "p_s_annual": 0.61, "best_of": 3,
            "feature_time_violation": True,
        },
        {
            "row_id": "production:m2:OPEN", "golgg_match_id": "m2",
            "cohort": "production", "horizon": "OPEN", "date": "2026-01-03",
            "decision_at": pd.NaT, "scenario_decision_at": "2026-01-02T00:00:00Z",
            "start_at": "2026-01-03T12:00:00Z", "label_available_at": pd.NaT,
            "scenario_label_available_at": "2026-01-04T00:00:00Z", "y": 0,
            "p_s": np.nan, "p_s_annual": 0.42, "best_of": 3,
            "feature_time_violation": True,
        },
    ])
    replay = pd.DataFrame([
        {
            "row_id": "production:m1:OPEN", "golgg_match_id": "m1",
            "cohort": "production_asof_previous_roster", "horizon": "OPEN",
            "date": "2026-01-02", "decision_at": pd.NaT,
            "scenario_decision_at": "2026-01-01T00:00:00Z",
            "start_at": "2026-01-02T12:00:00Z", "label_available_at": pd.NaT,
            "y": 1, "best_of": 3, "model_year": 2026,
            "forecaster": "native_annual_a0", "p": 0.7, "status": "available",
            "scenario_fit_origin": "2026-01-01T00:00:00Z",
            "scenario_timestamp_assumption": "daily-release roster replay",
        },
        {
            "row_id": "production:m1:OPEN", "golgg_match_id": "m1",
            "cohort": "production_asof_previous_roster", "horizon": "OPEN",
            "date": "2026-01-02", "decision_at": pd.NaT,
            "scenario_decision_at": "2026-01-01T00:00:00Z",
            "start_at": "2026-01-02T12:00:00Z", "label_available_at": pd.NaT,
            "y": 1, "best_of": 3, "model_year": 2026,
            "forecaster": "seed_only", "p": 0.68, "sigma_z": 0.2,
            "p_std": 0.04, "p_q10": 0.63, "p_q90": 0.72,
            "status": "available",
        },
    ])
    cohort, source_rows, receipt = _asof_cohort_inputs(opportunities, replay)

    assert len(cohort) == 2
    assert cohort.row_id.tolist() == opportunities.row_id.tolist()
    assert cohort.cohort.unique().tolist() == ["production_asof_previous_roster"]
    assert cohort.p_s.iloc[0] == pytest.approx(0.7)
    assert pd.isna(cohort.p_s.iloc[1])
    assert cohort.p_s_annual.tolist() == opportunities.p_s_annual.tolist()
    assert cohort.label_available_at.isna().all()
    assert receipt["retained_unavailable_rows"] == 1
    assert source_rows["forecaster"].nunique() == 2


def test_asof_cohort_rejects_cutoff_and_target_identity_conflicts():
    from src.analysis.ev_approach_runner import _asof_cohort_inputs
    import pandas as pd

    opportunity = pd.DataFrame([{
        "row_id": "production:m1:OPEN", "golgg_match_id": "m1",
        "cohort": "production", "horizon": "OPEN", "date": "2026-01-02",
        "decision_at": pd.NaT, "scenario_decision_at": "2026-01-01T00:00:00Z",
        "start_at": "2026-01-02T12:00:00Z", "label_available_at": pd.NaT,
        "scenario_label_available_at": "2026-01-03T00:00:00Z", "y": 1,
        "p_s": pd.NA, "p_s_annual": 0.61, "best_of": 3,
    }])
    prediction = pd.DataFrame([{
        "row_id": "production:m1:OPEN", "golgg_match_id": "m1",
        "cohort": "production_asof_previous_roster", "horizon": "OPEN",
        "date": "2026-01-02", "decision_at": pd.NaT,
        "scenario_decision_at": "2026-01-01T01:00:00Z",
        "start_at": "2026-01-02T12:00:00Z", "label_available_at": pd.NaT,
        "y": 1, "best_of": 3, "model_year": 2026,
        "forecaster": "native_annual_a0", "p": 0.7, "status": "available",
    }])

    with pytest.raises(ValueError, match="scenario_decision_at"):
        _asof_cohort_inputs(opportunity, prediction)


def test_native_target_predictions_keep_objective_and_score_semantics(tmp_path):
    from src.analysis.ev_approach_runner import _external_prediction_rows
    import pandas as pd

    targets = pd.DataFrame([
        {
            "row_id": "canonical_sports:m1", "golgg_match_id": "m1",
            "cohort": "canonical_sports", "horizon": "annual",
            "date": "2026-01-02", "best_of": 3, "y": 1,
        },
        {
            "row_id": "archival_market:m1:OPEN", "golgg_match_id": "m1",
            "cohort": "archival_market", "horizon": "OPEN",
            "date": "2026-01-02", "best_of": 3, "y": 1,
        },
    ])
    path = tmp_path / "native.csv"
    pd.DataFrame([
        {
            "golgg_match_id": "m1", "date": "2026-01-02", "best_of": 3,
            "y": 1, "objective": "terminal", "p": 0.71,
            "score_kind": "probability",
        },
        {
            "golgg_match_id": "m1", "date": "2026-01-02", "best_of": 3,
            "y": 1, "objective": "winner", "p": 0.69,
            "score_kind": "probability",
        },
    ]).to_csv(path, index=False)
    import hashlib
    import json

    input_hashes = {}
    for name in (
        "terminal_test.parquet", "winner_test.parquet",
        "target_comparison_aggregation_receipt.json",
    ):
        input_path = tmp_path / name
        input_path.write_bytes(name.encode())
        input_hashes[name] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    sidecar = {
        "status": "VERIFIED_COMPLETE_NATIVE_TARGET_INPUT",
        "rows": 2,
        "rows_per_objective": {"terminal": 1, "winner": 1},
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "inputs_sha256": input_hashes,
    }
    path.with_suffix(".receipt.json").write_text(json.dumps(sidecar))

    records, receipt = _external_prediction_rows(targets, path, "native_target")

    assert receipt["identity_matches"] == 4
    assert receipt["source_receipt"]["status"] == "VERIFIED"
    assert {record["method"] for record in records} == {
        "native_target:terminal", "native_target:winner",
    }
    assert {(record["cohort"], record["horizon"]) for record in records} == {
        ("canonical_sports", "annual"), ("archival_market", "OPEN"),
    }
    assert all(record["p"] in (0.69, 0.71) for record in records)

    (tmp_path / "terminal_test.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source receipt input does not match"):
        _external_prediction_rows(targets, path, "native_target")


def test_raw_ranking_has_discrimination_without_probability_scores():
    from src.analysis.ev_approach_runner import _ranking_metrics, _score_metrics
    import numpy as np
    import pandas as pd

    raw_ranking = pd.DataFrame({
        "y": [0, 1, 0, 1],
        "p": [np.nan] * 4,
        "raw_score": [-2.0, 3.0, -1.0, 1.0],
    })
    calibrated = pd.DataFrame({
        "y": [0, 1, 0, 1],
        "p": [0.2, 0.8, 0.3, 0.7],
    })

    assert _score_metrics(raw_ranking)["n_scored"] == 0
    assert _ranking_metrics(raw_ranking)["rank_auc"] == 1.0
    assert _score_metrics(calibrated)["n_scored"] == 4


def test_action_outputs_do_not_turn_decision_scores_into_ev():
    from src.analysis.ev_approach_runner import _action_bets
    import pandas as pd

    bets = _action_bets(pd.DataFrame([{
        "row_id": "r1", "cohort": "asof", "horizon": "OPEN",
        "method": "kelly_utility", "action_side": "A",
        "action_fraction": 0.02, "odds_a": 2.0, "bookmaker_a": "book",
        "quote_id_a": "q1", "decision_at": pd.NaT,
        "scenario_decision_at": "2026-01-01T00:00:00Z",
        "scenario_timestamp_assumption": "scenario",
        "start_at": "2026-01-02T00:00:00Z", "label_available_at": pd.NaT,
        "scenario_label_available_at": "2026-01-03T00:00:00Z",
        "y": 1, "raw_score": 0.9,
    }]), 0.12)

    assert len(bets) == 1
    assert bets.action_fraction.iloc[0] == pytest.approx(0.02)
    assert pd.isna(bets.p.iloc[0])
    assert pd.isna(bets.ev.iloc[0])
    assert bets.unit_return.iloc[0] == pytest.approx(0.76)


def test_external_prediction_artifact_is_optional():
    from src.analysis.ev_approach_runner import _external_prediction_rows
    import pandas as pd

    records, receipt = _external_prediction_rows(pd.DataFrame(), None, "native_target")

    assert records == []
    assert receipt == {"status": "NOT_SUPPLIED", "path": None, "attached": 0}




def test_p5_cal_reference_needs_explicit_release_assumption_when_actual_is_unknown():
    from src.analysis.ev_approach_runner import _p5_threshold_receipts
    import pandas as pd

    calibration = pd.DataFrame([
        {
            "variant": "seed_only",
            "model_year": 2026,
            "partition": "CAL",
            "prediction_role": "SIGMA_REFERENCE",
            "row_id": "cal-1",
            "sigma_z": 0.2,
            "label_available_at": pd.NaT,
            "scenario_label_available_at": "2025-12-31T00:00:00Z",
        },
        {
            "variant": "seed_only",
            "model_year": 2026,
            "partition": "CAL",
            "prediction_role": "SIGMA_REFERENCE",
            "row_id": "cal-unknown",
            "sigma_z": 0.99,
            "label_available_at": pd.NaT,
            "scenario_label_available_at": "2025-12-31T00:00:00Z",
        },
    ])

    assert _p5_threshold_receipts(calibration) == []

    calibration["scenario_timestamp_assumption"] = ["daily-release scenario", None]
    [receipt] = _p5_threshold_receipts(calibration)
    assert receipt["cal_reference_id"] == "seed_only:2026"
    assert receipt["cal_rows"] == 1
    assert receipt["sigma_z_q75"] == pytest.approx(0.2)
    assert receipt["cal_row_ids"] == ["cal-1"]

def test_forecast_records_separate_own_fit_origin_from_native_source_origin():
    import pandas as pd
    from src.analysis.ev_approach_runner import _probability_record

    row = pd.Series({
        "row_id": "head-1", "cohort": "annual", "horizon": "OPEN",
        "fit_origin": pd.NaT, "scenario_fit_origin": "2025-01-01T00:00:00Z",
    })
    head = _probability_record(
        row, "head:logistic", 0.6, head_fit_origin="2025-07-01T00:00:00Z",
    )
    market = _probability_record(row, "market_raw", 0.5)
    annual = _probability_record(row, "native_annual_a0", 0.6)

    assert head["fit_kind"] == "monthly_head"
    assert pd.isna(head["fit_origin"])
    assert head["scenario_fit_origin"] == "2025-07-01T00:00:00Z"
    assert head["source_scenario_fit_origin"] == "2025-01-01T00:00:00Z"
    risk = _probability_record(
        row, "risk:sequential_residual", 0.6,
        effective_decision_at="2025-06-01T00:00:00Z",
        feedback_count=1, max_feedback_release_at="2025-05-31T00:00:00Z",
        feedback_ids_sha256="abc",
    )

    assert risk["fit_kind"] == "online_feedback"
    assert risk["scenario_fit_origin"] == "2025-06-01T00:00:00Z"
    assert risk["feedback_count"] == 1
    assert risk["max_feedback_release_at"] == "2025-05-31T00:00:00Z"
    assert risk["feedback_ids_sha256"] == "abc"
    assert market["fit_kind"] == "stateless"
    assert pd.isna(market["scenario_fit_origin"])
    assert annual["fit_kind"] == "frozen_annual"
    assert annual["scenario_fit_origin"] == "2025-01-01T00:00:00Z"


def test_deep_loss_count_depends_on_outcome_probability_not_odds():
    import pandas as pd
    from src.analysis.ev_approach_runner import _score_metrics

    frame = pd.DataFrame({
        "y": [0, 1, 1, 0],
        "p": [0.95, 0.05, 0.5, float("nan")],
        "odds_a": [1.1, 1.2, 10.0, 2.0],
        "odds_b": [1.2, 1.1, 10.0, 2.0],
    })
    result = _score_metrics(frame)
    assert result["n_scored"] == 3
    assert result["tail_log_loss_ge_2_5"] == 2
    assert _score_metrics(frame.drop(columns=["odds_a", "odds_b"]))[
        "tail_log_loss_ge_2_5"
    ] == 2


def test_paired_score_intervals_match_their_weighting_estimands():
    import numpy as np
    import pandas as pd
    from src.analysis.ev_approach_runner import _score_pair_bootstrap

    counts = np.array([80, 2, 3, 4])
    probabilities = np.array([0.9, 0.1, 0.4, 0.6])
    row_ids = [f"row-{i}" for i in range(counts.sum())]
    universe = pd.DataFrame({
        "row_id": row_ids,
        "date": np.repeat(
            ["2024-01-10", "2024-02-10", "2024-03-10", "2024-04-10"],
            counts,
        ),
        "y": 1,
    })
    method_a = pd.DataFrame({"row_id": row_ids, "p": np.repeat(probabilities, counts)})
    method_b = pd.DataFrame({"row_id": row_ids, "p": 0.5})
    draws, seed = 5000, 773
    result = _score_pair_bootstrap(universe, method_a, method_b, draws, seed)
    sampled = np.random.default_rng(seed).integers(0, 4, size=(draws, 4))
    for metric, monthly_delta in (
        ("log_loss", np.log(0.5 / probabilities)),
        ("brier", (probabilities - 1.0) ** 2 - 0.25),
    ):
        weighted = (
            (monthly_delta * counts)[sampled].sum(axis=1)
            / counts[sampled].sum(axis=1)
        )
        equal_month = monthly_delta[sampled].mean(axis=1)
        assert result[f"delta_{metric}_a_minus_b"] == pytest.approx(
            np.average(monthly_delta, weights=counts)
        )
        assert result[f"{metric}_ci"] == pytest.approx(
            np.quantile(weighted, [0.025, 0.975])
        )
        assert result[f"equal_month_{metric}_ci"] == pytest.approx(
            np.quantile(equal_month, [0.025, 0.975])
        )


def test_paired_score_bootstrap_keeps_unforecasted_months_without_zero_loss():
    import numpy as np
    import pandas as pd
    from src.analysis.ev_approach_runner import _score_pair_bootstrap

    universe = pd.DataFrame({
        "row_id": ["a", "b", "c", "d"],
        "date": ["2024-01-10", "2024-02-10", "2024-03-10", "2024-04-10"],
        "y": [1, 1, 1, 1],
    })
    method_a = pd.DataFrame({"row_id": universe.row_id, "p": [np.nan, np.nan, 0.6, 0.7]})
    method_b = pd.DataFrame({"row_id": universe.row_id, "p": [np.nan, np.nan, 0.5, 0.5]})
    draws, seed = 5000, 774
    result = _score_pair_bootstrap(universe, method_a, method_b, draws, seed)
    sampled = np.random.default_rng(seed).integers(0, 4, size=(draws, 4))
    empty = int(np.count_nonzero((sampled < 2).all(axis=1)))
    assert result["n_paired"] == 2
    assert result["months"] == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert result["delta_log_loss_a_minus_b"] == pytest.approx(
        np.log(0.5 / np.array([0.6, 0.7])).mean()
    )
    assert result["log_loss_bootstrap"]["empty_replicates"] == empty
    assert result["status"] == "UNDEFINED_REPLICATES"
    assert result["log_loss_ci"] == [None, None]


def test_score_calibration_uses_shared_right_closed_bins_and_includes_endpoints():
    import pandas as pd
    from src.analysis.ev_approach_runner import _score_metrics

    frame = pd.DataFrame({"y": [0, 1, 0], "p": [0.45, 0.5, 0.51]})
    assert _score_metrics(frame)["ece10"] == pytest.approx(
        (2.0 / 3.0) * abs(0.5 - 0.475) + 0.51 / 3.0
    )
    endpoints = _score_metrics(pd.DataFrame({"y": [1, 0], "p": [0.0, 1.0]}))
    assert endpoints["ece10"] == pytest.approx(1.0)
    assert endpoints["brier"] == 1.0
    assert endpoints["tail_log_loss_ge_2_5"] == 2


def test_every_uncertainty_policy_has_a_paired_financial_comparison(tmp_path):
    import ast
    import pandas as pd
    from src.analysis.ev_approach_runner import _evaluate_forecasts, UNCERTAINTY_POLICIES

    dates = pd.to_datetime(["2026-01-10", "2026-02-10", "2026-03-10"], utc=True)
    opportunities = pd.DataFrame({
        "row_id": ["a", "b", "c"], "cohort": "archival_market", "horizon": "OPEN",
        "date": dates, "decision_at": dates, "start_at": dates + pd.Timedelta(hours=12),
        "label_available_at": dates + pd.Timedelta(days=1), "y": [1, 0, 1],
        "p_s": 0.59, "q": 0.55, "odds_a": 2.1, "odds_b": 1.8,
        "bookmaker_a": "book", "bookmaker_b": "book",
        "quote_id_a": ["a:A", "b:A", "c:A"], "quote_id_b": ["a:B", "b:B", "c:B"],
    })
    forecasts = opportunities[["cohort", "horizon", "row_id"]].assign(
        method="uncertainty:seed_only", p=0.59, status="available",
        score_kind="probability", variant="seed_only", model_year=2026,
        sigma_z=0.2, p_std=0.05, p_q10=0.52, p_q90=0.68,
    )
    thresholds = pd.DataFrame([{
        "variant": "seed_only", "model_year": 2026, "sigma_z_q75": 0.1,
        "cal_reference_id": "closed-2025-cal", "cal_rows": 3, "cal_row_ids": [],
    }])
    protocol = {
        "seed": 91, "statistics": {"bootstrap_draws": 5000},
        "decision_evaluation": {
            "initial_bankroll": 10000, "flat_stake": 100,
            "kelly_fraction": 0.25, "kelly_max_free_capital_fraction": 0.025,
        },
    }
    stage07, stage08 = tmp_path / "stage07", tmp_path / "stage08"
    stage07.mkdir()
    stage08.mkdir()
    _evaluate_forecasts(
        opportunities, opportunities.iloc[:0], forecasts, thresholds,
        protocol, stage07, stage08,
    )
    paired = pd.read_csv(stage08 / "paired_intervals.csv")
    policies = paired.loc[paired.comparison_type.eq("uncertainty_policy_finance")]
    assert set(policies.method_b) == {
        f"uncertainty:seed_only:{policy}" for policy in UNCERTAINTY_POLICIES
    }
    quarantine = policies.loc[policies.method_b.eq("uncertainty:seed_only:P5")].iloc[0]
    assert ast.literal_eval(quarantine.months) == ["2026-01", "2026-02", "2026-03"]
    assert ast.literal_eval(quarantine.policy_b)["bets"] == 0
    assert ast.literal_eval(quarantine.bootstrap)["empty_replicates"] == 5000
    assert ast.literal_eval(quarantine.mean_weighted)["delta_pnl_per_opportunity"] == pytest.approx(-23.2)


def test_p5_receipts_retain_source_ids_before_the_origin():
    import pandas as pd
    from src.analysis.ev_approach_runner import _p5_threshold_receipts

    cal = pd.DataFrame({
        "golgg_match_id": [101, 102, 103], "variant": "seed_only", "model_year": 2025,
        "partition": "CAL", "prediction_role": "SIGMA_REFERENCE",
        "sigma_z": [0.1, 0.2, 100.0], "label_available_at": pd.NaT,
        "scenario_label_available_at": pd.to_datetime(
            ["2024-12-29", "2024-12-30", "2025-01-01"], utc=True,
        ),
        "scenario_timestamp_assumption": "daily released label scenario",
    })
    receipt, = _p5_threshold_receipts(cal)
    assert receipt["cal_row_ids"] == ["101", "102"]
    assert receipt["cal_rows"] == 2
    assert receipt["sigma_z_q75"] == pytest.approx(0.175)
    assert receipt["max_calibration_label_release_at"] == "2024-12-30T00:00:00+00:00"


def test_canonical_native_confidence_cannot_train_a_market_control():
    import pandas as pd
    from src.analysis.ev_approach_runner import _canonical_sports_frame, _chronological_head_matrix

    dates = pd.to_datetime([
        "2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02", "2024-03-01",
    ], utc=True)
    canonical = pd.DataFrame({
        "golgg_match_id": ["a", "b", "c", "d", "e"], "date": dates, "y": [0, 1, 0, 1, 1],
        "p_s": [0.2, 0.8, 0.3, 0.7, 0.6], "q": [0.8, 0.8, 0.7, 0.7, 0.6],
        "scenario_prediction_at": dates,
        "scenario_label_available_at": dates + pd.Timedelta(days=1),
        "scenario_timestamp_assumption": "daily released label scenario",
        "information_status": "historical_scenario",
    })
    sports_only = _canonical_sports_frame(canonical)
    predictions, receipts = _chronological_head_matrix(
        sports_only, methods=["market_cal"], lambdas=(0.01,), min_train=2,
    )
    assert predictions.p.isna().all()
    assert all(not receipt["train_ids"] for receipt in receipts)
