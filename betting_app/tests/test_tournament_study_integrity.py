"""Freeze-time admission and corruption detection without native fits or archives."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.models import tournament_improvement_runner as runner
from src.models import tournament_native as native
from src.models import tournament_research_cohort as cohort


RUN_ID = "tournament_c0_20261006_01"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def study_inputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    research = tmp_path / "research"
    research.mkdir()
    # Preserve real implementation bytes in an isolated project so corruption
    # scenarios cannot touch the dirty checkout or production/native artifacts.
    for relative in runner.CODE:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((runner.PROJECT_ROOT / relative).read_bytes())
    input_dir = project / "data/05_model_input/tournament_improvement" / RUN_ID
    design = {"base_forecasts": "data/reference.parquet"}
    _write_json(input_dir / "design.json", design)
    protected = project / "conf/base/c0_serving.json"
    _write_json(protected, {"immutable": "native serving configuration"})
    _write_json(input_dir / "preimplementation_evidence.json", {
        "protected_sha256": {str(protected.relative_to(project)): _digest(protected)},
    })
    _write_json(input_dir / "union_gate_before_corrections.json", {
        "exit_code": 1, "failed": ["prior source-bound regression"],
    })
    _write_json(input_dir / "independent_adf_smoke.json", {
        "formal": False, "receipt": "retained independent prefit evidence",
    })
    frame = pd.DataFrame({
        "model_year": [2024, 2025, 2026],
        "golgg_match_id": ["train-1", "cal-1", "assess-1"],
    })
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    monkeypatch.setattr(runner, "DESIGN_PATH", input_dir / "design.json")
    monkeypatch.setattr(runner, "_canonical_frame", lambda design, root: frame)

    def prepare_empty_cohort(output_dir, **kwargs):
        manifest = {"events": {}, "exclusions": []}
        _write_json(output_dir / "manifest.json", manifest)
        return manifest

    class UnavailableNativeProvider:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def artifact_contract(self, year, model_kind):
            raise native.NativeArtifactUnavailable("isolated test has no fitted assets")

    # Only heavyweight cohort/native preparation is substituted. Admission,
    # exclusive creation, file hashing and verification execute unchanged.
    monkeypatch.setattr(cohort, "prepare_research_cohort", prepare_empty_cohort)
    monkeypatch.setattr(native, "ResearchNativePairProvider", UnavailableNativeProvider)
    return project, research, input_dir, protected


def test_freeze_preserves_prefit_evidence_and_refuses_a_second_freeze(study_inputs):
    project, research, input_dir, protected = study_inputs
    retained = {path: path.read_bytes() for path in input_dir.iterdir()}
    retained[protected] = protected.read_bytes()

    protocol_path = runner.freeze_study(RUN_ID, research)
    runner._verify(protocol_path, research)
    assert {path: path.read_bytes() for path in retained} == retained

    frozen = {path: path.read_bytes() for path in input_dir.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="occupied"):
        runner.freeze_study(RUN_ID, research)
    assert {path: path.read_bytes() for path in frozen} == frozen
    assert not (project / "data/06_models/tournament_improvement" / RUN_ID).exists()


@pytest.mark.parametrize("relative", [
    "src/ratings/glicko2_core.py",
    "src/models/frozen_tournament.py",
])
def test_verify_rejects_executing_dependency_changed_after_freeze(study_inputs, relative):
    project, research, _, _ = study_inputs
    protocol_path = runner.freeze_study(RUN_ID, research)
    runner._verify(protocol_path, research)
    dependency = project / relative
    dependency.write_bytes(dependency.read_bytes() + b"\n# changed after freeze\n")
    with pytest.raises(ValueError, match="Frozen study binding changed") as error:
        runner._verify(protocol_path, research)
    assert str(dependency) in str(error.value)


@pytest.mark.parametrize("filename", [
    "preimplementation_evidence.json",
    "union_gate_before_corrections.json",
    "independent_adf_smoke.json",
])
def test_verify_rejects_retained_prefit_receipt_corruption(study_inputs, filename):
    _, research, input_dir, _ = study_inputs
    protocol_path = runner.freeze_study(RUN_ID, research)
    receipt = input_dir / filename
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen study binding changed") as error:
        runner._verify(protocol_path, research)
    assert str(receipt) in str(error.value)


def test_freeze_rejects_changed_protected_artifact_before_preparation(study_inputs):
    _, research, input_dir, protected = study_inputs
    protected.write_bytes(protected.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Protected preimplementation artifact changed"):
        runner.freeze_study(RUN_ID, research)
    assert not (input_dir / "cohort").exists()
    assert not (input_dir / "protocol.json").exists()


@pytest.mark.parametrize("occupied", [
    "05_model_input/protocol.json",
    "05_model_input/memberships.json",
    "05_model_input/cohort",
    "05_model_input/unowned_receipt.json",
    "06_models",
    "07_model_output",
    "08_reporting",
])
def test_freeze_refuses_occupied_formal_namespace_without_changing_it(study_inputs, occupied):
    project, research, input_dir, _ = study_inputs
    stage, _, child = occupied.partition("/")
    target = project / "data" / stage / "tournament_improvement" / RUN_ID
    if child:
        target /= child
    if child and child.endswith(".json"):
        _write_json(target, {"owned_existing": True})
    else:
        target.mkdir(parents=True, exist_ok=True)
        (target / "existing.json").write_text("{}", encoding="utf-8")
    snapshots = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        runner.freeze_study(RUN_ID, research)
    assert {path: path.read_bytes() for path in project.rglob("*") if path.is_file()} == snapshots
    if child != "protocol.json":
        assert not (input_dir / "protocol.json").exists()


def test_candidate_recipe_boundaries_include_later_native_base_training(tmp_path):
    frame = pd.DataFrame({
        "golgg_match_id": ["train", "cal", "assess"],
        "model_year": [2024, 2025, 2026],
        "date": ["2024-11-02", "2025-12-27", "2026-01-05"],
        "result_day": ["2024-11-02", "2025-12-27", "2026-01-05"],
        "y": [1, 0, 1],
        "p_c0": [.6, .4, .6],
        "best_of": [3, 3, 3],
        "roster_a": [["a0", "a1", "a2", "a3", "a4"]] * 3,
        "roster_b": [["b0", "b1", "b2", "b3", "b4"]] * 3,
    })
    artifacts = {
        f"{year}:c0": {
            "train_end": f"{year - 1}-06-30T00:00:00Z",
            "calibration_end": f"{year - 1}-11-06T00:00:00Z",
        }
        for year in (2024, 2025, 2026)
    }
    paths = {stage: tmp_path / stage for stage in
             ("06_models", "07_model_output", "08_reporting")}
    for path in paths.values():
        path.mkdir()
    runner._canonical_candidate(frame, 0., paths, artifacts)
    exported = pd.read_parquet(paths["07_model_output"] / "candidate.parquet")
    assessed = exported.loc[exported.model_year.eq(2026)].iloc[0]
    assert assessed.train_end == "2025-06-30"
    assert assessed.train_end_exclusive == "2025-07-01"
    assert assessed.calibration_end == "2025-12-27"
    assert assessed.calibration_end_exclusive == "2026-01-01"
    assert assessed.base_train_end == "2025-06-30"
    assert assessed.base_calibration_end == "2025-11-06"
