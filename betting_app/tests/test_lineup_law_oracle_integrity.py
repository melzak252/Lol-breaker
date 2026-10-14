"""Independent oracle regression cases; no fitting or native execution."""
from copy import deepcopy
from importlib import import_module
import json

import pytest


def oracle():
    return import_module("scripts.verify_lineup_law_reliability")


def roster(prefix):
    return [prefix + str(i) for i in range(5)]


def opportunity_fixture():
    canonical = dict(match_id="1", date="2026-02-01", release_day="2026-02-03", best_of=1,
                     team_a="A", team_b="B", roster_a=roster("a"), roster_b=roster("b"), tournament_name="League")
    source = dict(teams=["A", "B"], best_of=1, forecast="2026-02-01", context="League",
                  rosters={"A":roster("a"), "B":roster("b")}, wins={"A":1, "B":0},
                  native_release_day="2026-02-03", canonical_release_day="2026-02-03")
    meta = dict(golgg_match_id="1", team1_id="A", team2_id="B", best_of=1, y=1,
                roster_a=roster("a"), roster_b=roster("b"), unknown_players=0)
    return canonical, source, meta


@pytest.mark.parametrize("reverse", [False, True])
def test_incomplete_raw_first_map_remains_unlabelled_with_complete_native_metadata(reverse):
    module = oracle()
    canonical, source, meta = opportunity_fixture()
    source["rosters"]["A"][-1] = None
    if reverse:
        meta.update(team1_id="B", team2_id="A", y=0,
                    roster_a=roster("b"), roster_b=roster("a"))
    before = deepcopy(source)
    audit = module.Audit()
    audit.taxonomy_rules = []
    rows, _ = module.opportunity_replay([canonical], [meta], [], {"1":source}, [], audit)
    own = next(row for row in rows if row["side"] == "A")
    assert own["label"] is None
    assert own["label_absence"] == "INCOMPLETE_FIRST_MAP_TARGET"
    assert own["actual_roster"][-1] is None
    assert source == before


@pytest.mark.parametrize("incomplete", [False, True])
def test_contradictory_known_raw_identity_is_rejected(incomplete):
    module = oracle()
    canonical, source, meta = opportunity_fixture()
    source["rosters"]["A"][0] = "intruder"
    if incomplete:
        source["rosters"]["A"][-1] = None
    audit = module.Audit()
    audit.taxonomy_rules = []
    with pytest.raises(ValueError, match="identity mismatch"):
        module.opportunity_replay([canonical], [meta], [], {"1":source}, [], audit)


def test_sub_ulp_other_mass_preserves_exact_missing_mass_and_scalar_abstention():
    module = oracle()
    tiny = 2**-60
    actual = module.partial([1-tiny, 0, tiny], [1, 0, 0], {(0, 0):.75})
    assert actual["missing_mass"] == tiny
    assert actual["prob_a"] is None
    assert actual["scalar_absence"] == "POSITIVE_UNSUPPORTED_ROSTER_MASS"


def test_no_fit_reporting_contrasts_archive_is_rejected(tmp_path):
    module = oracle()
    roots = {stage:tmp_path/stage for stage in ("06_models", "07_model_output", "08_reporting")}
    for root in roots.values():
        root.mkdir()
    (roots["08_reporting"]/"contrasts.npz").write_bytes(b"unauthorized")
    audit = module.Audit()
    module.no_fit_outputs_replay(audit, roots)
    assert [item["check_id"] for item in audit.failures] == ["prepare.no_predictions.contrasts.npz"]


def test_late_failure_invalidates_applicable_fitted_success_but_preserves_not_run():
    module = oracle()
    audit = module.Audit()
    audit.check("protected.after", "changed", "pinned")
    complete = {"status":"PASS", "fitted_prediction_oracle_status":"PASS"}
    module.finalize_statuses(audit, complete)
    assert complete == {"status":"FAIL", "fitted_prediction_oracle_status":"FAIL"}
    partial = {"status":"FAIL", "fitted_prediction_oracle_status":"NOT_RUN_PARTIAL_HARD_STOP"}
    module.finalize_statuses(audit, partial)
    assert partial["fitted_prediction_oracle_status"] == "NOT_RUN_PARTIAL_HARD_STOP"


@pytest.mark.parametrize("stop_name", ["run_hard_stop.json","recover_evaluation_hard_stop.json"])
def test_partial_dispatch_requires_hard_stop_and_explicit_availability(tmp_path,stop_name):
    module = oracle()
    reporting = tmp_path/"08_reporting"
    reporting.mkdir()
    index = {"branch_status":"INCONCLUSIVE", "artifact_descriptors":[]}
    assert module.partial_hard_stop(reporting, index) is None
    receipt = dict(status="EXECUTION_FAILED", failure_class="SOFTWARE_FAILURE", failure_type="RuntimeError",
                   failure_reason="capture failed", formal_fits=2, no_restart=True)
    (reporting/stop_name).write_text(json.dumps(receipt))
    assert module.partial_hard_stop(reporting, index) == receipt


def test_partial_unavailable_artifact_cannot_hide_present_output(tmp_path):
    module = oracle()
    path = tmp_path/"side_laws.parquet"
    path.write_bytes(b"present")
    audit = module.Audit()
    descriptor = dict(role="side_laws", repo_relative_path="side_laws.parquet", absence_status="NOT_RUN", absence_reason="interrupted")
    assert not module.artifact_available(audit, tmp_path, [descriptor], path)
    assert audit.failures


def partial_zero_fit_fixture(tmp_path, kind="INTERRUPTED"):
    roots = {stage:tmp_path/stage for stage in ("06_models","07_model_output","08_reporting")}
    for root in roots.values():
        root.mkdir()
    descriptors = []
    for stage,names in (("06_models",("conditional.json","global.json")),
                        ("07_model_output",("side_laws.parquet","series_scores.parquet","native_pairs.parquet","native_blocked_pairs.parquet","roster_bounds.parquet")),
                        ("08_reporting",("metrics.json","contrasts.npz","cal_audit.json","native_execution.json","query_nonmutation.json","saved_output_binding.json","public_cases.json"))):
        for name in names:
            descriptors.append(dict(role=name,repo_relative_path=str((roots[stage]/name).relative_to(tmp_path)),
                                    absence_status="NOT_RUN",absence_reason="interrupted"))
    stop = dict(status="INCONCLUSIVE",failure_kind=kind,failure_type="KeyboardInterrupt",
                failure_reason="during capture",formal_fits=0,completed_fits=0,unused_TRAIN_slots=2,no_restart=True)
    decision = dict(status="INCONCLUSIVE",formal_fits=0,completed_fits=0,unused_TRAIN_slots=2,
                    CAL_fits=0,C0_fits=0,ASSESS_fits=0)
    (roots["08_reporting"]/"decision.json").write_text(json.dumps(decision))
    receipt = dict(raw_state_count=0,scope_not_exercised=[],scope_exercised=[])
    return roots,descriptors,stop,receipt


def test_actual_partial_receipt_retains_failure_and_explicit_not_run_checks(tmp_path):
    module = oracle()
    roots,descriptors,stop,receipt = partial_zero_fit_fixture(tmp_path)
    audit = module.Audit()
    module.partial_outputs_replay(audit,receipt,stop,tmp_path,roots,{"artifact_descriptors":descriptors},
                                  [],[],{}, {},[],[],{},tmp_path)
    assert receipt["status"] == "FAIL"
    assert receipt["fitted_prediction_oracle_status"] == "NOT_RUN_PARTIAL_HARD_STOP"
    assert receipt["primary_slot_statuses"] == {"7":"NOT_RUN_PARTIAL_HARD_STOP","8":"NOT_RUN_PARTIAL_HARD_STOP"}
    skipped = [check for check in audit.checks if check["status"]=="NOT_RUN"]
    assert any(check["check_id"]=="partial.contrasts" for check in skipped)
    assert all(check["reason"]=="KeyboardInterrupt: during capture" for check in skipped)
    assert [check["check_id"] for check in audit.failures] == ["partial.execution_failure"]


def lifecycle_fixture():
    pair = dict(match_id="1",candidate_a_index=0,candidate_b_index=0,status="NATIVE_PAIR_UNAVAILABLE",
                failure_kind="AVAILABILITY",failure_type="C0WorkerTimeout",failure_reason="timed out",native_requested=True)
    pair.update(session_index=0,cutoff_day="2026-02-01",failure_phase="PREDICTION")
    execution = dict(worker_sessions=[dict(session_index=0,pid=123,established_at="2026-10-10T00:00:00+00:00",
                                         closed_at="2026-10-10T00:00:01+00:00",close_reason="timeout")],
                     worker_startup_attempts=[dict(attempt_index=0,status="ESTABLISHED",session_index=0)],
                     worker_pids=[123],native_worker_sessions=1,native_worker_recycles=0,native_request_count=1,
                     native_valid_pairs=0,native_failed_pairs=1,native_blocked_pairs=0,raw_snapshots_persisted=0,
                     worker_closed=True,requests_by_series={"1":1})
    execution.update(native_worker_startup_attempts=1,native_worker_startup_failures=0,state_advance_origin_count=0,
                     state_advance_origin_attempt_count=1,state_advance_origin_unconfirmed_count=1)
    return execution,pair


def test_timeout_only_session_is_counted_even_without_accepted_prediction():
    module = oracle()
    execution,pair = lifecycle_fixture()
    audit = module.Audit()
    module.native_execution_replay(audit,execution,[pair],[])
    assert not audit.failures
    execution["native_worker_sessions"] = 0
    audit = module.Audit()
    module.native_execution_replay(audit,execution,[pair],[])
    assert [check["check_id"] for check in audit.failures] == ["native.execution.native_worker_sessions"]


def test_native_integrity_failure_cannot_be_recorded_as_recoverable_availability():
    module = oracle()
    execution,pair = lifecycle_fixture()
    pair["failure_type"] = "C0InferenceError"
    audit = module.Audit()
    module.native_execution_replay(audit,execution,[pair],[])
    assert any(check["check_id"].endswith(".authorized") for check in audit.failures)


def test_partial_consumed_attempt_prefix_checks_retained_marker_and_counter(tmp_path):
    module = oracle()
    roots,descriptors,stop,receipt = partial_zero_fit_fixture(tmp_path)
    import hashlib
    operation = dict(operation_id="R1_TRAIN_conditional",study="R1",kind="bounded_TRAIN_lineup_law",
                     arm="conditional",partition="TRAIN",seed=None,authorized_ceiling=2,counter_before=0,counter_after=1,
                     input_ids_sha256=hashlib.sha256(json.dumps([]).encode()).hexdigest(),
                     started_at_utc="2026-10-10T00:00:00+00:00",finished_at_utc="2026-10-10T00:00:01+00:00",
                     status="INTERRUPTED_OR_FAILED",parameters_sha256=None,iterations=None,function_evaluations=None,
                     failure_reason="KeyboardInterrupt")
    attempt = dict(operation,status="ATTEMPTED",finished_at_utc=None)
    (roots["08_reporting"]/"conditional_attempt.json").write_text(json.dumps(attempt))
    decision_path = roots["08_reporting"]/"decision.json"
    decision = json.loads(decision_path.read_text())
    decision.update(formal_fits=1,unused_TRAIN_slots=1)
    decision_path.write_text(json.dumps(decision))
    stop.update(formal_fits=1,unused_TRAIN_slots=1)
    operation["counter_after"] = 2
    audit = module.Audit()
    module.partial_outputs_replay(audit,receipt,stop,tmp_path,roots,{"artifact_descriptors":descriptors},
                                  [operation],[],{}, {},[],[],{},tmp_path)
    failures = {check["check_id"] for check in audit.failures}
    assert "budget.conditional.counter_after" in failures
    assert "budget.conditional.immutable_attempt.counter_after" in failures
    assert receipt["fitted_prediction_oracle_status"] == "NOT_RUN_PARTIAL_HARD_STOP"


def test_failed_startup_does_not_invent_established_session_or_prediction(tmp_path):
    module = oracle()
    execution,pair = lifecycle_fixture()
    execution.update(worker_sessions=[],worker_pids=[],native_worker_sessions=0,native_worker_recycles=0,
                     native_request_count=0,requests_by_series={},state_advance_origin_count=0,
                     state_advance_origin_attempt_count=0,state_advance_origin_unconfirmed_count=0,
                     native_worker_startup_failures=1,
                     worker_startup_attempts=[dict(attempt_index=0,status="FAILED",session_index=None,
                                                  failure_type="C0WorkerTimeout",failure_reason="startup timeout")])
    pair.update(native_requested=False,session_index=None,failure_phase="STARTUP")
    audit = module.Audit()
    module.native_execution_replay(audit,execution,[pair],[])
    assert not audit.failures


def test_late_hard_stop_overrides_retained_decision_without_rewriting_it(tmp_path):
    module = oracle()
    roots,descriptors,stop,receipt = partial_zero_fit_fixture(tmp_path,kind="IO")
    path = roots["08_reporting"]/"decision.json"
    decision = json.loads(path.read_text())
    decision["status"] = "NO_LINEUP_LAW_GAIN"
    path.write_text(json.dumps(decision))
    before = path.read_bytes()
    audit = module.Audit()
    module.partial_outputs_replay(audit,receipt,stop,tmp_path,roots,{"artifact_descriptors":descriptors},
                                  [],[],{}, {},[],[],{},tmp_path)
    assert receipt["status"] == "FAIL"
    assert [check["check_id"] for check in audit.failures] == ["partial.execution_failure"]
    assert path.read_bytes() == before


def test_archived_runtime_exception_requires_exact_old_digest_and_allowed_path(tmp_path):
    module = oracle()
    current = tmp_path/"scripts/verify_lineup_law_reliability.py"
    current.parent.mkdir()
    current.write_text("corrected runtime")
    snapshot = tmp_path/"old.py"
    snapshot.write_text("failed runtime")
    authorization = {"allowed_source_changes":["scripts/verify_lineup_law_reliability.py"],
                     "old_runtime_snapshots":{"scripts/verify_lineup_law_reliability.py":
                                              {"snapshot":"old.py","sha256":module.sha(snapshot)}}}
    pin = {"path":str(current),"sha256":module.sha(snapshot)}
    audit = module.Audit()
    module.runtime_pin_replay(audit,tmp_path,"verifier",pin,authorization,allow_archived=True)
    assert not audit.failures
    pin["sha256"] = "unrelated old digest"
    audit = module.Audit()
    module.runtime_pin_replay(audit,tmp_path,"verifier",pin,authorization,allow_archived=True)
    assert audit.failures
    pin.update(path=str(snapshot),sha256="not the actual digest")
    audit = module.Audit()
    module.runtime_pin_replay(audit,tmp_path,"scientific",pin,authorization,allow_archived=True)
    assert audit.failures


def test_recovery_budget_rejects_new_fit_and_native_retry_without_reclassifying_parent():
    module = oracle()
    _,pair = lifecycle_fixture()
    pair.update(match_id="73048",failure_kind="PROGRAMMING",failure_type="AttributeError",
                status="NATIVE_PAIR_FATAL")
    audit = module.Audit()
    module.recovery_counts_replay(audit,[{"kind":"bounded_TRAIN_lineup_law"}],[pair],[pair],["73048"])
    assert {check["check_id"] for check in audit.failures} >= {
        "recovery.zero_new_fit_trace","recovery.no_original_native_retry"}
    before = deepcopy(pair)
    audit = module.Audit()
    module.recovery_counts_replay(audit,[],[pair],[],["73048"])
    assert not audit.failures
    assert pair == before


@pytest.mark.parametrize("tamper", [None,"failure_kind","inherited_request_count"])
def test_recovery_native_audit_preserves_original_fatal_failure_and_counts(tmp_path,tamper):
    module = oracle()
    parent = {stage:tmp_path/"parent"/stage for stage in ("07_model_output","08_reporting")}
    roots = {"08_reporting":tmp_path/"recovery"}
    for path in (*parent.values(),*roots.values()):
        path.mkdir(parents=True)
    inherited_execution,pair = lifecycle_fixture()
    pair.update(failure_kind="SOFTWARE",failure_type="AttributeError",status="NATIVE_PAIR_FATAL")
    module.pd.DataFrame([pair]).to_parquet(parent["07_model_output"]/"native_pairs.parquet",index=False)
    (parent["08_reporting"]/"native_execution.json").write_text(json.dumps(inherited_execution))
    (roots["08_reporting"]/"new_operation_trace.json").write_text("[]")
    (roots["08_reporting"]/"decision.json").write_text(json.dumps({
        "inherited_TRAIN_fits":2,"new_fit_counts":{"TRAIN":0,"CAL":0,"C0":0,"ASSESS":0}}))
    execution = deepcopy(inherited_execution)
    execution.update(worker_sessions=[],worker_startup_attempts=[],worker_pids=[],native_worker_sessions=0,
                     native_request_count=0,native_failed_pairs=0,requests_by_series={},
                     native_worker_startup_attempts=0,state_advance_origin_attempt_count=0,
                     state_advance_origin_unconfirmed_count=0,inherited_execution=inherited_execution,
                     inherited_request_count=1,new_native_request_count=0,no_retry_native_match_ids=["1"])
    inherited_pair = dict(pair,execution_scope="PARENT_INHERITED")
    if tamper=="failure_kind":
        inherited_pair["failure_kind"] = "AVAILABILITY"
    if tamper=="inherited_request_count":
        execution["inherited_request_count"] = 0
    audit = module.Audit()
    receipt = {}
    module.recovery_native_replay(audit,receipt,tmp_path,roots,
                                 (parent,{}, {},{"failure_kind":"SOFTWARE"}),
                                 {"no_retry_native_match_ids":["1"]},[inherited_pair],[],execution)
    assert bool(audit.failures) == (tamper is not None)
    assert receipt["native_operation_counts"] == {
        "inherited_requests":1,"new_requests":0,"inherited_sessions":1,"new_sessions":0}
