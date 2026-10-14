"""Pre-fit integrity boundaries using temporary bytes, never candidate fitting."""
import json

import numpy as np
import pytest

from src.analysis import champion_oov_reliability as runner


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def bound_fixture(tmp_path):
    evidence = tmp_path / "authorization"
    for prepared in ("original", "compatible_but_unpinned"):
        path = tmp_path / prepared
        path.mkdir()
        np.savez(path / "arrays.npz", mask=np.ones((1, 2, 5, 16), dtype=bool))
    contract = {"input_paths": {"old_prepared_root": "original"}}
    put(evidence / "w1_contract.json", contract)
    put(evidence / "authorization_and_initial_pins.json", {"protected_before": {}})
    destination = tmp_path / "data/05_model_input/champion_oov_reliability/example"
    protocol = {"evidence_root": str(evidence.resolve()), "w1_contract": contract}
    put(destination / "protocol.json", protocol)
    put(destination / "binding.json", {"protocol_sha256": runner._sha(destination / "protocol.json"), "pins": {}})
    return evidence, contract


@pytest.mark.parametrize("public_query", [False, True])
def test_bound_load_rejects_compatible_contract_from_different_root_before_arrays(tmp_path, public_query):
    evidence, contract = bound_fixture(tmp_path)
    other = tmp_path / "substituted"
    put(other / "w1_contract.json", contract)
    with pytest.raises(ValueError, match="evidence root differs"):
        if public_query:
            runner.predict_saved("example", ["target"], repo_root=tmp_path, evidence_root=other)
        else:
            runner._load_bound("example", repo_root=tmp_path, evidence_root=other)


def test_bound_load_rejects_changed_contract_at_original_root_before_arrays(tmp_path):
    evidence, contract = bound_fixture(tmp_path)
    contract["input_paths"]["old_prepared_root"] = "compatible_but_unpinned"
    put(evidence / "w1_contract.json", contract)
    with pytest.raises(ValueError, match="contract differs"):
        runner._load_bound("example", repo_root=tmp_path, evidence_root=evidence)


def historical_fixture(tmp_path):
    models = ["D_seed_1.joblib", "D_seed_2.joblib", "D_seed_3.joblib", "D_ensemble.joblib"]
    outputs = ["D_cal.parquet", "D_assess.parquet", "C0_assess.parquet"]
    for locator in models + outputs:
        (tmp_path / locator).write_bytes(locator.encode())
    hashes = {locator: runner._sha(tmp_path / locator) for locator in models + outputs}
    scoring = {"additional_pins": {p: hashes[p] for p in models}, "models_before_scoring": {p: hashes[p] for p in models}}
    put(tmp_path / "scoring.json", scoring)
    put(tmp_path / "decision.json", {"output_hashes": {p: hashes[p] for p in outputs}})
    inputs = {"old_scoring_binding": "scoring.json", "old_decision": "decision.json",
              "saved_d0_models": models[:3], "saved_d0_ensemble": models[3],
              "saved_d0_cal_outputs": outputs[:1], "saved_d0_assessment_outputs": outputs[1:2],
              "saved_c0_assessment_outputs": outputs[2:]}
    return inputs, scoring, models, outputs


@pytest.mark.parametrize("artifact", ["D_seed_1.joblib", "D_ensemble.joblib", "D_cal.parquet", "D_assess.parquet", "C0_assess.parquet"])
def test_historical_reuse_rejects_stale_bytes_instead_of_repinning(tmp_path, artifact):
    inputs, _, _, _ = historical_fixture(tmp_path)
    runner._validate_historical_references(tmp_path, tmp_path, inputs)
    (tmp_path / artifact).write_bytes(b"changed-compatible-artifact")
    with pytest.raises(ValueError, match="historical"):
        runner._validate_historical_references(tmp_path, tmp_path, inputs)


@pytest.mark.parametrize("receipt_field", ["additional_pins", "models_before_scoring"])
def test_historical_reuse_requires_both_model_receipts(tmp_path, receipt_field):
    inputs, scoring, models, _ = historical_fixture(tmp_path)
    scoring[receipt_field].pop(models[0])
    put(tmp_path / "scoring.json", scoring)
    with pytest.raises(ValueError, match="historical"):
        runner._validate_historical_references(tmp_path, tmp_path, inputs)


def test_final_protected_drift_closes_without_positive_decision_or_lost_budget(tmp_path):
    evidence = tmp_path / "evidence"
    protected = tmp_path / "old_negative.json"
    protected.write_bytes(b"original-negative-evidence")
    put(evidence / "authorization_and_initial_pins.json", {"protected_before": {str(protected): runner._sha(protected)}})
    paths = {stage: tmp_path / stage for stage in ("06_models", "07_model_output", "08_reporting")}
    for path in paths.values():
        path.mkdir()
    trace = [{"kind": "W1_NEURAL_FIT", "status": "COMPLETED", "epochs": 12, "fit_receipt": {"epochs_completed": 12}}]
    execution = {"run_id": "example", "status": "RUNNING", "operation_trace": trace,
                 "neural_fits": 9, "CAL_terminal_scalars": 3, "seed_epochs": 108}
    put(paths["08_reporting"] / "execution.json", execution)
    protected.write_bytes(b"drift-after-all-successful-fits")
    with pytest.raises(ValueError, match="protected original bytes changed") as failure:
        runner._publish_completed("example", tmp_path, tmp_path, evidence, paths, execution, True)
    assert not (paths["08_reporting"] / "decision.json").exists()
    assert json.loads((paths["08_reporting"] / "execution.json").read_text())["status"] == "RUNNING"
    runner._record_failure(paths, "example", execution, None, failure.value, "FINAL_INTEGRITY")
    decision = json.loads((paths["08_reporting"] / "decision.json").read_text())
    journal = json.loads((paths["08_reporting"] / "execution.json").read_text())
    assert decision["decision"] == "NO_ADOPTION"
    assert journal["status"] == decision["status"] == "FIT_FAILED_WITHIN_BUDGET"
    assert journal["failure_phase"] == "FINAL_INTEGRITY"
    assert decision["operation_trace"] == trace
    assert (decision["neural_fits"], decision["CAL_terminal_scalars"], decision["seed_epochs"]) == (9, 3, 108)
    assert not (paths["08_reporting"] / "final_integrity.json").exists()


def test_completed_publication_checks_integrity_and_remains_create_only(tmp_path):
    evidence = tmp_path / "evidence"
    put(evidence / "authorization_and_initial_pins.json", {"protected_before": {}})
    paths = {stage: tmp_path / stage for stage in ("06_models", "07_model_output", "08_reporting")}
    for path in paths.values():
        path.mkdir()
    execution = {"status": "RUNNING", "neural_fits": 9, "CAL_terminal_scalars": 3, "seed_epochs": 108}
    put(paths["08_reporting"] / "execution.json", execution)
    decision = runner._publish_completed("example", tmp_path, tmp_path, evidence, paths, execution, True)
    assert decision["decision"] == "DEVELOPMENT_CANDIDATE"
    assert execution["status"] == "COMPLETED"
    assert json.loads((paths["08_reporting"] / "final_integrity.json").read_text()) == {"protected_after": {}}
    before = (paths["08_reporting"] / "decision.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner._publish_completed("example", tmp_path, tmp_path, evidence, paths, execution, False)
    assert (paths["08_reporting"] / "decision.json").read_bytes() == before


def test_failed_attempt_retains_prior_success_and_unknown_progress(tmp_path):
    paths = {"08_reporting": tmp_path / "reporting"}
    successful = {"kind": "W1_NEURAL_FIT", "status": "COMPLETED", "epochs": 12,
                  "fit_receipt": {"epochs_completed": 12}}
    failed = {"kind": "W1_NEURAL_FIT", "status": "ATTEMPTED", "epochs": 12,
              "iterations": None, "function_evaluations": None}
    execution = {"status": "RUNNING", "neural_fits": 2, "CAL_terminal_scalars": 0, "seed_epochs": 12,
                 "operation_trace": [successful, failed]}
    runner._record_failure(paths, "example", execution, failed, ValueError("nonfinite objective"), "FIT_OR_EXPORT")
    decision = json.loads((paths["08_reporting"] / "decision.json").read_text())
    assert execution["neural_fits"] == 2
    assert execution["seed_epochs"] == 12
    assert successful["status"] == "COMPLETED"
    assert failed["status"] == "FAILED"
    assert failed["iterations"] is None
    assert "fit_receipt" not in failed
    assert decision["operation_trace"] == [successful, failed]
    assert decision["decision"] == "NO_ADOPTION"


def test_historical_reference_missing_old_output_pin_is_rejected(tmp_path):
    inputs, _, _, outputs = historical_fixture(tmp_path)
    receipt = json.loads((tmp_path / "decision.json").read_text())
    receipt["output_hashes"].pop(outputs[0])
    put(tmp_path / "decision.json", receipt)
    with pytest.raises(ValueError, match="historical reference"):
        runner._validate_historical_references(tmp_path, tmp_path, inputs)


def test_export_failure_retains_recorded_completed_training_epochs(tmp_path):
    paths = {"08_reporting": tmp_path / "reporting"}
    completed_training = {"kind": "W1_NEURAL_FIT", "status": "ATTEMPTED", "epochs": 12,
                          "fit_receipt": {"epochs_completed": 12, "optimizer_steps": 4}}
    execution = {"status": "RUNNING", "neural_fits": 1, "CAL_terminal_scalars": 0, "seed_epochs": 12,
                 "operation_trace": [completed_training]}
    runner._record_failure(paths, "example", execution, completed_training, OSError("state export failed"), "FIT_OR_EXPORT")
    decision = json.loads((paths["08_reporting"] / "decision.json").read_text())
    assert completed_training["status"] == "FAILED"
    assert completed_training["fit_receipt"]["epochs_completed"] == 12
    assert decision["seed_epochs"] == execution["seed_epochs"] == 12
    assert decision["decision"] == "NO_ADOPTION"
