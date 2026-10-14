"""Source integrity, edition support and marginal-preservation runner boundaries."""
import json

import pandas as pd
import pytest

from src.models import tournament_copula_runner as runner


def test_freeze_rejects_owned_execution_namespace_without_touching_user_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    owned = tmp_path / "data/07_model_output/tournament_copula/owned/result.json"
    owned.parent.mkdir(parents=True)
    owned.write_bytes(b'"user-owned"\n')
    with pytest.raises(FileExistsError):
        runner.freeze_copula_study("owned")
    assert owned.read_bytes() == b'"user-owned"\n'
    assert not (tmp_path / "data/05_model_input").exists()


def test_run_checks_changed_source_before_claiming_execution_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "pairs.json"
    source.write_text('{"p":0.5}')
    protocol = {"schema": runner.SCHEMA, "run_id": "bound", "bindings": {
        "project:pairs.json": runner._sha(source)}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    source.write_text('{"p":0.8}')
    with pytest.raises(ValueError, match="binding changed"):
        runner.run_copula_study(path)
    assert not (tmp_path / "data/06_models").exists()


def test_bound_research_bytes_can_move_without_changing_the_frozen_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    original = tmp_path / "original-research"
    relocated = tmp_path / "relocated-research"
    original.mkdir()
    relocated.mkdir()
    source = original / "bound.json"
    source.write_text('{"native_probability":0.73}')
    digest = runner._sha(source)
    moved = relocated / source.name
    moved.write_bytes(source.read_bytes())
    source.unlink()
    design_path = tmp_path / runner.DESIGN_PATH
    design_path.parent.mkdir(parents=True)
    design_path.write_text("{}")
    protocol = {"schema": runner.SCHEMA, "run_id": "relocated",
                "bindings": {"research:bound.json": digest},
                "runtime": runner._runtime(), "design": {},
                "research_root_local_provenance": str(original)}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    anchor = path.with_name("protocol_binding.json")
    anchor.write_text(json.dumps({"protocol_sha256": runner._sha(path)}))
    verified, root = runner._verify(path, research_root=relocated)
    assert verified == protocol
    assert root == relocated.resolve()
    moved.write_text('{"native_probability":0.91}')
    with pytest.raises(ValueError):
        runner._verify(path, research_root=relocated)


def score(event, partition, model, value):
    return {"event_id": event, "edition_id": event, "partition": partition,
            "model": model, "eligible": True,
            "structural_checks_passed": True, "chronology_checks_passed": True,
            "metrics": {"normalized_joint_sports_path_nll": value,
                        "champion_brier": .2, "placement_rps": .1},
            "target_counts": {"normalized_joint_sports_path_nll": 1,
                              "champion_brier": 1, "placement_rps": 4},
            "unavailable_reasons": {}}


def decision_rows():
    rows = []
    for event, part in (("cal-a", "CAL"), ("cal-b", "CAL"), ("primary", "ASSESS")):
        for model in runner.CONTROLS:
            rows.append(score(event, part, model, .4 if model == runner.CANDIDATE else .5))
    return rows


def test_one_assessment_edition_does_not_produce_fake_bootstrap_interval_or_adoption():
    decision = runner._decision(decision_rows(), {"passed": True},
        {"CAL": ["cal-a", "cal-b"], "ASSESS": ["primary", "missing"]}, numerical_passed=True)
    assert decision["calibration_gate"]["passed"] is True
    assert decision["eligible_primary_editions"] == 1
    assert decision["minimum_primary_editions"] == 8
    assert decision["adopted"] is False
    assert decision["goal_complete"] is False
    primary = [c for c in decision["comparisons"] if c["partition"] == "ASSESS"]
    assert primary
    assert all(c["ci_upper"] is None and c["ci_lower"] is None for c in primary)
    assert all(c["interval_reason"] for c in primary)
    assert "missing" in decision["primary_support_reasons"][0]


def test_cal_response_improvement_cannot_hide_championship_regression():
    rows = decision_rows()
    for row in rows:
        if row["partition"] == "CAL" and row["model"] == runner.CANDIDATE:
            row["metrics"]["champion_brier"] = .3
    decision = runner._decision(rows, {"passed": True},
        {"CAL": ["cal-a", "cal-b"], "ASSESS": ["primary", "missing"]}, numerical_passed=True)
    assert decision["calibration_gate"]["passed"] is False
    assert decision["retained_development_only"] is False
    assert any("champion_brier" in reason for reason in decision["calibration_gate"]["reasons"])


def test_nonfinite_path_loss_is_not_replaced_by_finite_intersection():
    rows = decision_rows()
    rows[0]["metrics"]["normalized_joint_sports_path_nll"] = float("inf")
    decision = runner._decision(rows, {"passed": True},
        {"CAL": ["cal-a", "cal-b"], "ASSESS": ["primary", "missing"]}, numerical_passed=True)
    assert decision["calibration_gate"]["passed"] is False
    comparison = next(c for c in decision["comparisons"] if c["partition"] == "CAL"
                      and c["reference"] == "native_c0_unchanged"
                      and c["metric"] == "normalized_joint_sports_path_nll")
    assert comparison["mean_delta"] is None
    assert comparison["eligible_editions"] == 2
    assert comparison["nonfinite_editions"] == ["cal-a"]


def test_gaussian_marginal_guard_handles_certainties_without_clipping():
    frame = pd.DataFrame({"p_c0": [0., .01, .2, .5, .9, .99, 1.],
                          "y": [0, 1, 0, 1, 1, 0, 1], "model_year": [2026] * 7})
    guard = runner._marginal_guard(frame, .5, require_canonical=False)
    assert guard["passed"] is True
    assert guard["rows"] == 7
    assert guard["quadrature_max_absolute_error"] < 1e-10
    assert guard["log_loss_delta"] == 0.
    assert guard["brier_delta"] == 0.


def test_canonical_guard_cannot_pass_a_reduced_cohort():
    frame = pd.DataFrame({"p_c0": [.5], "y": [1], "model_year": [2026]})
    with pytest.raises(ValueError, match="11550"):
        runner._marginal_guard(frame, .2)


def test_training_scores_are_never_counted_as_primary_assessment():
    rows = decision_rows()
    for i in range(12):
        for model in runner.CONTROLS:
            rows.append(score(f"train-{i}", "TRAIN", model, .01))
    decision = runner._decision(rows, {"passed": True},
        {"CAL": ["cal-a", "cal-b"], "ASSESS": ["primary", "missing"]}, numerical_passed=True)
    assert decision["eligible_primary_editions"] == 1
    assert all(c["partition"] != "TRAIN" for c in decision["comparisons"])
