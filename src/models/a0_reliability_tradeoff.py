"""Hash-pinned candidate-only native6 duration experiment.

This historical study reuses the frozen Plan07 C0/shared experts and fits only
one six-epoch candidate per seed and annual origin. It never mutates Plan07.
"""
from __future__ import annotations

import json
import hashlib
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import reliability_tradeoff as analysis
from src.models import a0_data_focus as native
from src.models.a0_native_refit import (
    SEEDS, _atomic_checkpoint, _calculate_forecast, _contract_digest,
    _sha256, _torch_state_hash, _versions, _write_json_exclusive,
)
from src.models.a0_target_objectives import fit_beta_target_on_device
from src.models.a0_reliability import YEARS, annual_partitions
from src.models.competition_tiers import classify_competition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_RUN = "model_reliability_20261001_055321"
REFERENCE_CANONICAL = (
    "data/08_reporting/model_reliability/model_reliability_20261001_055321/"
    "assessment_v2/joined.parquet"
)
GOAL_RELATIVE = "data/05_model_input/reliability_tradeoff/reliability_tradeoff_20261001_092401_integrity_v2/initial_goal.md"
FAMILY_PROTOCOL_RELATIVE = "data/05_model_input/model_reliability/model_reliability_20261001_055321/evaluation_protocol_v2.json"
EXPECTED_A0_MANIFEST_SHA256 = "dd1c60bbc85329a21018e8f044dad50fef02e1095964c0a39462635dc925be6e"
VARIANTS = ("native12", "native6")
COMPONENTS = ("full", "neural", "raw_neural")
MODES_OUT = ("full", "no_w20", "no_organization")


def _paths(run_id: str):
    return {
        name: PROJECT_ROOT / f"data/{stage}/reliability_tradeoff" / run_id
        for name, stage in (("input", "05_model_input"), ("models", "06_models"),
                            ("output", "07_model_output"), ("report", "08_reporting"))
    }


def _source_hashes(backend_hashes):
    result = dict(backend_hashes)
    for relative in (
        "src/models/a0_reliability_tradeoff.py", "scripts/train_a0_reliability_tradeoff.py",
        "src/models/a0_reliability.py", "scripts/train_a0_reliability.py",
        "src/models/a0_data_focus.py", "scripts/train_a0_data_focus.py",
        "src/models/a0_native_refit.py", "src/models/a0_target_experiment.py",
        "src/models/a0_target_objectives.py", "src/models/a0_history_reconstruction.py",
        "src/models/competition_tiers.py", "src/analysis/reliability_tradeoff.py",
        "scripts/run_model_benchmark.py", "src/analysis/ev_approach_data.py",
        "src/analysis/model_reliability.py", "src/analysis/a0_data_focus.py",
        "src/analysis/production_market_benchmark.py", "src/analysis/research_benchmark.py",
        "src/analysis/model_benchmark.py", "src/analysis/metrics.py", "src/analysis/bootstrap.py",
        "src/models/a0_uncertainty.py",
    ):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen source missing: {relative}")
        result[relative] = _sha256(path)
    return result


def _source_path(key, research_root):
    if key.startswith("research:"):
        return research_root / key.removeprefix("research:")
    return PROJECT_ROOT / key.removeprefix("project:")


def _relative_inputs(data, root):
    return native._relative_input_hashes(data["input_hashes"], root)


def _split_contract(data):
    frame = data["frame"]
    partitions = {}
    for year in YEARS:
        split = annual_partitions(frame, year)
        expected = {"train": 27705, "cal": 1379, "validation": 4642} if year == 2024 else (
            {"train": 32625, "cal": 1103, "validation": 3805} if year == 2025 else
            {"train": 36205, "cal": 1328, "validation": 3103})
        counts = {key: len(value) for key, value in split.items()}
        if counts != expected:
            raise ValueError(f"annual partition counts differ for {year}: {counts}")
        partitions[str(year)] = {
            "counts": counts,
            "membership_sha256": {key: _contract_digest(frame.iloc[ix].golgg_match_id.astype(str).tolist())
                                   for key, ix in split.items()},
            "boundaries": {
                "train_event_and_release_before": f"{year - 1}-07-01",
                "cal_event_start_inclusive": f"{year - 1}-07-01",
                "cal_event_and_release_before": f"{year}-01-01",
                "assessment_event_start_inclusive": f"{year}-01-01",
                "assessment_event_before": f"{year + 1}-01-01",
            },
        }
    return partitions


def _validate_canonical(data, canonical):
    frame = data["frame"]
    if (len(canonical) != 11550
            or canonical.golgg_match_id.astype(str).nunique() != len(canonical)):
        raise ValueError("canonical assessment must contain exactly 11,550 unique IDs")
    canonical_ids = canonical.golgg_match_id.astype(str).tolist()
    bank = frame.assign(_id=frame.golgg_match_id.astype(str)).set_index("_id")
    if bank.index.duplicated().any() or not set(canonical_ids).issubset(set(bank.index.astype(str))):
        raise ValueError("canonical IDs do not resolve uniquely in the feature bank")
    expected_ids = set()
    for year in YEARS:
        expected_ids.update(frame.iloc[annual_partitions(frame, year)["validation"]].golgg_match_id.astype(str))
    if set(canonical_ids) != expected_ids:
        raise ValueError("canonical IDs differ from complete annual assessment memberships")
    rows = bank.loc[canonical_ids]
    for column in ("team1_id", "team2_id", "y", "best_of"):
        if column not in canonical or column not in rows:
            raise ValueError(f"canonical or feature bank lacks identity column {column}")
        if not np.array_equal(canonical[column].astype(str).to_numpy(), rows[column].astype(str).to_numpy()):
            raise ValueError(f"canonical side/label identity mismatch: {column}")
    if not np.array_equal(pd.to_datetime(canonical.date).to_numpy(), pd.to_datetime(rows.date).to_numpy()):
        raise ValueError("canonical dates do not align to feature bank")
    if "result_day" in canonical and not np.array_equal(
            pd.to_datetime(canonical.result_day).dt.strftime("%Y-%m-%d").to_numpy(),
            pd.to_datetime(rows.result_day).dt.strftime("%Y-%m-%d").to_numpy()):
        raise ValueError("canonical label-release dates do not align")
def _validate_manifest_canonical(raw, joined):
    if len(raw) != 11550 or len(joined) != 11550:
        raise ValueError("manifest and Plan07 canonical inputs must contain 11,550 rows")
    if raw.golgg_match_id.astype(str).tolist() != joined.golgg_match_id.astype(str).tolist():
        raise ValueError("manifest and Plan07 canonical IDs/order differ")
    for column in ("team1_id", "team2_id", "y", "best_of"):
        if not np.array_equal(raw[column].astype(str).to_numpy(),
                              joined[column].astype(str).to_numpy()):
            raise ValueError(f"manifest and Plan07 canonical {column} differ")
    for column in ("date", "result_day"):
        if not np.array_equal(pd.to_datetime(raw[column]).to_numpy(),
                              pd.to_datetime(joined[column]).to_numpy()):
            raise ValueError(f"manifest and Plan07 canonical {column} differ")
    if not np.array_equal(raw.p_full.to_numpy(float), joined.p_a0_full.to_numpy(float)):
        raise ValueError("Plan07 canonical p_a0_full differs from manifest p_full")


def _validate_run_id(run_id):
    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single nonempty path component")
def _verify_redirected_control(execution):
    """Check the pinned completion's artifacts before loading a redirected C0."""
    reference = execution.get("reference")
    if reference is None:
        return None
    run_id = reference["run_id"]
    _validate_run_id(run_id)
    relative = f"data/08_reporting/a0_data_focus/{run_id}/completed.json"
    completion_path = PROJECT_ROOT / relative
    if _sha256(completion_path) != reference["completed_sha256"]:
        raise ValueError("redirected control completion hash mismatch")
    completion = json.loads(completion_path.read_text())
    if completion["status"] != "TRAINING_COMPLETE_NOT_SCORED":
        raise ValueError("redirected control is not complete")
    artifacts = completion["artifacts"]
    model_root = f"data/06_models/a0_data_focus/{run_id}"
    components = ["shared", "control/calibration",
                  *(f"control/seed_{seed}" for seed in SEEDS)]
    required = {f"{model_root}/{component}/model.joblib{suffix}"
                for component in components for suffix in ("", ".receipt.json")}
    if not required.issubset(artifacts):
        raise ValueError("redirected control artifacts do not bind every consumed checkpoint and receipt")
    for path, expected in artifacts.items():
        artifact = PROJECT_ROOT / path
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"redirected control artifact hash mismatch: {path}")
    return {"completed_path": relative, "completed_sha256": reference["completed_sha256"],
            "artifacts": artifacts}


def freeze_protocol(run_id: str, research_root=None) -> Path:
    """Freeze source, inputs, controls, canonical identities and partitions before fitting."""
    root = native.resolve_research_root(research_root)
    _validate_run_id(run_id)
    paths = _paths(run_id)
    protocol_path = paths["input"] / "protocol.json"
    if any(path.exists() and any(path.iterdir()) for path in paths.values() if path.is_dir()):
        raise FileExistsError("reliability-tradeoff output namespace must be empty before freeze")
    manifest = PROJECT_ROOT / "conf/base/research_benchmark.json"
    manifest_sha = _sha256(manifest)
    if manifest_sha != EXPECTED_A0_MANIFEST_SHA256:
        raise ValueError("canonical A0 manifest differs from authorized pre-study version")
    manifest_data = json.loads(manifest.read_text())
    manifest_input_path = manifest_data["input"]["path"]
    manifest_input_sha256 = manifest_data["input"]["sha256"]
    raw_canonical_path = root / manifest_input_path
    if _sha256(raw_canonical_path) != manifest_input_sha256:
        raise ValueError("manifest raw canonical input differs from its pinned SHA-256")
    goal_source = PROJECT_ROOT / GOAL_RELATIVE
    goal_bytes = goal_source.read_bytes()
    goal_snapshot = paths["input"] / "initial_goal.md"

    backend, backend_hashes = native.uncertainty._activate_archive_backend(root)
    sources = _source_hashes(backend_hashes)
    data = native.uncertainty._load_training_data(root)
    native.partition_indices(data["frame"], history=data["history"], raw=data["raw"])
    partitions = _split_contract(data)
    canonical_path = PROJECT_ROOT / REFERENCE_CANONICAL
    canonical = pd.read_parquet(canonical_path)
    _validate_canonical(data, canonical)
    raw_canonical = pd.read_parquet(raw_canonical_path)
    _validate_manifest_canonical(raw_canonical, canonical)

    reference_reports = {}
    for year in YEARS:
        base = PROJECT_ROOT / f"data/08_reporting/model_reliability/{REFERENCE_RUN}/neural/{year}/completed.json"
        receipt = json.loads(base.read_text())
        if receipt["status"] != "TRAINING_COMPLETE_NOT_SCORED":
            raise ValueError(f"Plan07 native12 control is not complete for {year}")
        reference_contract = receipt["contract"]
        if (reference_contract["input_sha256"] != _relative_inputs(data, root)
                or reference_contract["split_ids_sha256"] != partitions[str(year)]["membership_sha256"]):
            raise ValueError(f"Plan07 control inputs or partitions differ for {year}")
        for relative, digest in receipt["artifacts"].items():
            file = PROJECT_ROOT / relative
            if not file.is_file() or _sha256(file) != digest:
                raise ValueError(f"Plan07 artifact hash mismatch: {relative}")
        control_root = PROJECT_ROOT / f"data/06_models/model_reliability/{REFERENCE_RUN}/neural/{year}"
        control_execution = json.loads((control_root / "execution.json").read_text())
        reference = control_execution.get("reference")
        redirected_completion = _verify_redirected_control(control_execution)
        control_artifact_root = (PROJECT_ROOT / "data/06_models/a0_data_focus" / reference["run_id"]
                                 if reference else control_root)
        train_rows = partitions[str(year)]["counts"]["train"]
        first_phase_receipts = {}
        for seed in SEEDS:
            trajectory_path = control_artifact_root / f"supervised_18/seed_{seed}/model.joblib.receipt.json"
            trajectory_details = json.loads(trajectory_path.read_text())["details"]
            phase_key = "details.first_phase" if year == 2026 else "details.phases[0]"
            phase = trajectory_details["first_phase"] if year == 2026 else trajectory_details["phases"][0]
            expected_updates = 6 * math.ceil(train_rows / 128)
            if (len(phase["main_losses"]) != 6 or len(phase["auxiliary_losses"]) != 6
                    or phase["optimizer_updates"] != expected_updates):
                raise ValueError(f"saved supervised_18 first-phase receipt differs from frozen schedule: {year}/{seed}")
            first_phase_receipts[str(seed)] = {
                "path": str(trajectory_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(trajectory_path), "phase_key": phase_key,
                "optimizer_updates": phase["optimizer_updates"],
            }
        native12_output_path = (
            f"data/07_model_output/model_reliability/{REFERENCE_RUN}/neural/{year}/control.parquet")
        native12_output_sha256 = receipt["artifacts"].get(native12_output_path)
        if native12_output_sha256 is None:
            raise ValueError(f"Plan07 control receipt does not pin native12 output for {year}")
        reference_reports[str(year)] = {
            "completed_path": str(base.relative_to(PROJECT_ROOT)),
            "completed_sha256": _sha256(base), "artifacts": receipt["artifacts"],
            "control_execution_path": str((control_root / "execution.json").relative_to(PROJECT_ROOT)),
            "control_execution_sha256": _sha256(control_root / "execution.json"),
            "control_artifact_root": str(control_artifact_root.relative_to(PROJECT_ROOT)),
            "native12_output_path": native12_output_path,
            "native12_output_sha256": native12_output_sha256,
            "first_phase_receipts": first_phase_receipts,
            "redirected_completion": redirected_completion,
        }
    family_protocol_file = PROJECT_ROOT / FAMILY_PROTOCOL_RELATIVE
    family_protocol_sha256 = _sha256(family_protocol_file)
    family_profiles = json.loads(family_protocol_file.read_text())["family_profiles"]
    for key, digest in sorted(sources.items()):
        source = _source_path(key, root)
        if _sha256(source) != digest:
            raise ValueError(f"source changed during freeze: {key}")
    for origin in reference_reports.values():
        for trajectory in origin["first_phase_receipts"].values():
            trajectory_path = PROJECT_ROOT / trajectory["path"]
            if _sha256(trajectory_path) != trajectory["sha256"]:
                raise ValueError(f"saved supervised_18 receipt changed before freeze: {trajectory['path']}")
    paths["input"].mkdir(parents=True, exist_ok=True)
    with goal_snapshot.open("xb") as target:
        target.write(goal_bytes)
    source_zip = paths["input"] / "sources.zip"
    with zipfile.ZipFile(source_zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("goal/initial_goal.md", goal_bytes)
        archive.write(manifest, "project/conf/base/research_benchmark.json")
        archive.write(raw_canonical_path, "research/" + manifest_input_path)
        archive.write(family_protocol_file, "project/" + FAMILY_PROTOCOL_RELATIVE)
        for key in sorted(sources):
            archive.write(_source_path(key, root), key.replace(":", "/", 1))
        for origin in reference_reports.values():
            for trajectory in origin["first_phase_receipts"].values():
                archive.write(PROJECT_ROOT / trajectory["path"], "project/" + trajectory["path"])
            redirected = origin["redirected_completion"]
            if redirected is not None:
                archive.write(PROJECT_ROOT / redirected["completed_path"],
                              "project/" + redirected["completed_path"])
    protocol = {
        "schema": "a0_reliability_tradeoff_v2", "run_id": run_id,
        "status": "FROZEN_BEFORE_FITTING", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_goal_path": GOAL_RELATIVE, "initial_goal_sha256": hashlib.sha256(goal_bytes).hexdigest(),
        "goal_snapshot_path": str(goal_snapshot.relative_to(PROJECT_ROOT)),
        "goal_snapshot_sha256": hashlib.sha256(goal_bytes).hexdigest(),
        "family_profile_protocol_path": FAMILY_PROTOCOL_RELATIVE,
        "family_profile_protocol_sha256": family_protocol_sha256,
        "family_profiles": family_profiles,
        "source_sha256": sources, "source_snapshot_path": str(source_zip.relative_to(PROJECT_ROOT)),
        "versions": _versions(),
        "source_snapshot_sha256": _sha256(source_zip), "manifest_sha256": manifest_sha,
        "manifest_path": "conf/base/research_benchmark.json",
        "manifest_input_path": manifest_input_path, "manifest_input_sha256": manifest_input_sha256,
        "inputs": _relative_inputs(data, root), "bank_receipt_sha256": data["bank_receipt_sha256"],
        "canonical_path": REFERENCE_CANONICAL, "canonical_sha256": _sha256(canonical_path),
        "partitions": partitions, "years": list(YEARS), "seeds": list(SEEDS),
        "reference_run": REFERENCE_RUN, "reference_origins": reference_reports,
        "training": {"candidate_epochs": 6, "control_epochs": 12, "batch_size": 128,
                     "shuffle": False, "device": "cpu", "threads": 2,
                     "optimizer": "single AdamW call, lr=0.001, network decay=0.01, eta decay=0",
                     "objective": "native terminal beta NLL plus 0.1 masked auxiliary SmoothL1",
                     "calibration": "native CAL beta parameters; saved C0 shared-expert slopes"},
        "evaluation": {"status": "NOT_RUN", "resamples": 5000, "seed": 20261001,
                       "reliability_rms_improvement_min": 0.005,
                       "monthly_squared_gap_ci_upper_max": 0.0,
                       "global_ll_ci_upper_max": 0.005, "global_brier_ci_upper_max": 0.001,
                       "critical_profile_ll_ci_upper_max": 0.005,
                       "canonical_a0_reliability_rms_improvement_min": 0.005,
                       "canonical_a0_monthly_squared_gap_ci_upper_max": 0.0,
                       "canonical_a0_global_ll_ci_upper_max": 0.005,
                       "canonical_a0_global_brier_ci_upper_max": 0.001,
                       "canonical_a0_critical_profile_ll_ci_upper_max": 0.005,
                       "optional_mode_ll_ci_upper_max": 0.005,
                       "year_rms_worsening_max": 0.005, "favorite_gap_tolerance": 0.05,
                       "favorite_min_rows": 200, "favorite_min_months": 6,
                       "required_profiles": ["all", "BO3", "BO5", "major", "source_unseen_player"]},
        "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_ONLY_NOT_PRODUCTION_OR_FORECAST",
    }
    _write_json_exclusive(protocol_path, protocol)
    return protocol_path


def _read_protocol(path, root):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    run_id = protocol.get("run_id")
    _validate_run_id(run_id)
    if protocol.get("schema") != "a0_reliability_tradeoff_v2":
        raise ValueError("unsupported reliability-tradeoff protocol")
    if path != (_paths(run_id)["input"] / "protocol.json").resolve():
        raise ValueError("protocol is not stored in its run-specific input namespace")
    expected_training = {
        "candidate_epochs": 6, "control_epochs": 12, "batch_size": 128,
        "shuffle": False, "device": "cpu", "threads": 2,
        "optimizer": "single AdamW call, lr=0.001, network decay=0.01, eta decay=0",
        "objective": "native terminal beta NLL plus 0.1 masked auxiliary SmoothL1",
        "calibration": "native CAL beta parameters; saved C0 shared-expert slopes",
    }
    if protocol.get("training") != expected_training:
        raise ValueError("unsupported native6 training recipe")
    expected_paths = {
        "initial_goal_path": GOAL_RELATIVE,
        "goal_snapshot_path": str((_paths(run_id)["input"] / "initial_goal.md").relative_to(PROJECT_ROOT)),
        "source_snapshot_path": str((_paths(run_id)["input"] / "sources.zip").relative_to(PROJECT_ROOT)),
        "family_profile_protocol_path": FAMILY_PROTOCOL_RELATIVE,
        "canonical_path": REFERENCE_CANONICAL,
        "manifest_path": "conf/base/research_benchmark.json",
    }
    if any(protocol.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("protocol artifact path differs from authorized study")
    if protocol.get("versions") != _versions():
        raise ValueError("native runtime versions differ from frozen protocol")
    if (protocol.get("years") != list(YEARS) or protocol.get("seeds") != list(SEEDS)
            or protocol.get("reference_run") != REFERENCE_RUN):
        raise ValueError("frozen origins, seeds, or Plan07 reference differ")
    fixed_evaluation = {
        "status": "NOT_RUN", "resamples": 5000, "seed": 20261001,
        "reliability_rms_improvement_min": 0.005, "monthly_squared_gap_ci_upper_max": 0.0,
        "global_ll_ci_upper_max": 0.005, "global_brier_ci_upper_max": 0.001,
        "canonical_a0_reliability_rms_improvement_min": 0.005,
        "canonical_a0_monthly_squared_gap_ci_upper_max": 0.0,
        "canonical_a0_global_ll_ci_upper_max": 0.005,
        "canonical_a0_global_brier_ci_upper_max": 0.001,
        "canonical_a0_critical_profile_ll_ci_upper_max": 0.005,
        "critical_profile_ll_ci_upper_max": 0.005, "optional_mode_ll_ci_upper_max": 0.005,
        "year_rms_worsening_max": 0.005, "favorite_gap_tolerance": 0.05,
        "favorite_min_rows": 200, "favorite_min_months": 6,
        "required_profiles": ["all", "BO3", "BO5", "major", "source_unseen_player"],
    }
    if protocol.get("evaluation") != fixed_evaluation:
        raise ValueError("frozen evaluation criteria differ from Plan08")
    if protocol.get("initial_goal_sha256") != protocol.get("goal_snapshot_sha256"):
        raise ValueError("initial goal hash does not match archived snapshot")
    snapshot = PROJECT_ROOT / protocol["goal_snapshot_path"]
    if _sha256(snapshot) != protocol["goal_snapshot_sha256"]:
        raise ValueError("frozen Markdown goal snapshot changed")
    archive = PROJECT_ROOT / protocol["source_snapshot_path"]
    if _sha256(archive) != protocol["source_snapshot_sha256"]:
        raise ValueError("frozen source archive changed")
    archived_trajectories = {}
    with zipfile.ZipFile(archive) as source_zip:
        if source_zip.read("goal/initial_goal.md") != snapshot.read_bytes():
            raise ValueError("archived Markdown goal differs from pinned snapshot")
        manifest_bytes = source_zip.read("project/conf/base/research_benchmark.json")
        raw_input_bytes = source_zip.read("research/" + protocol["manifest_input_path"])
        family_protocol_bytes = source_zip.read("project/" + FAMILY_PROTOCOL_RELATIVE)
        for origin in protocol["reference_origins"].values():
            for trajectory in origin["first_phase_receipts"].values():
                archived_trajectories[trajectory["path"]] = source_zip.read(
                    "project/" + trajectory["path"])
    for year, origin in protocol["reference_origins"].items():
        report_path = PROJECT_ROOT / origin["completed_path"]
        output_path = PROJECT_ROOT / origin["native12_output_path"]
        if (_sha256(report_path) != origin["completed_sha256"]
                or _sha256(output_path) != origin["native12_output_sha256"]):
            raise ValueError(f"Plan07 native12 reference changed: {year}")
        execution_path = PROJECT_ROOT / origin["control_execution_path"]
        if _sha256(execution_path) != origin["control_execution_sha256"]:
            raise ValueError(f"Plan07 native12 execution redirect changed: {year}")
        if _verify_redirected_control(json.loads(execution_path.read_text())) != origin["redirected_completion"]:
            raise ValueError(f"Plan07 redirected control provenance changed: {year}")
        for relative, digest in origin["artifacts"].items():
            if _sha256(PROJECT_ROOT / relative) != digest:
                raise ValueError(f"Plan07 artifact hash mismatch: {relative}")
        for trajectory in origin["first_phase_receipts"].values():
            relative = trajectory["path"]
            payload = archived_trajectories[relative]
            if (_sha256(PROJECT_ROOT / relative) != trajectory["sha256"]
                    or hashlib.sha256(payload).hexdigest() != trajectory["sha256"]):
                raise ValueError(f"saved supervised_18 receipt changed: {relative}")
    family_protocol = PROJECT_ROOT / FAMILY_PROTOCOL_RELATIVE
    if (_sha256(family_protocol) != protocol.get("family_profile_protocol_sha256")
            or hashlib.sha256(family_protocol_bytes).hexdigest() != protocol.get("family_profile_protocol_sha256")
            or json.loads(family_protocol_bytes)["family_profiles"] != protocol.get("family_profiles")):
        raise ValueError("Plan07 family profiles differ from frozen protocol")
    manifest = PROJECT_ROOT / protocol["manifest_path"]
    manifest_data = json.loads(manifest_bytes)
    raw_path = root / protocol["manifest_input_path"]
    if (hashlib.sha256(manifest_bytes).hexdigest() != protocol["manifest_sha256"]
            or _sha256(manifest) != protocol["manifest_sha256"]
            or manifest_data["input"]["path"] != protocol["manifest_input_path"]
            or manifest_data["input"]["sha256"] != protocol["manifest_input_sha256"]
            or _sha256(raw_path) != protocol["manifest_input_sha256"]
            or hashlib.sha256(raw_input_bytes).hexdigest() != protocol["manifest_input_sha256"]):
        raise ValueError("canonical A0 manifest or raw manifest input differs from frozen protocol")
    _validate_manifest_canonical(pd.read_parquet(raw_path),
                                 pd.read_parquet(PROJECT_ROOT / protocol["canonical_path"]))
    _, backend_hashes = native.uncertainty._activate_archive_backend(root)
    if _source_hashes(backend_hashes) != protocol["source_sha256"]:
        raise ValueError("frozen experiment sources changed")
    if _sha256(PROJECT_ROOT / protocol["canonical_path"]) != protocol["canonical_sha256"]:
        raise ValueError("canonical assessment input changed")
    return path, protocol


def _origin_paths(run_id, year):
    return {name: _paths(run_id)[name] / str(year)
            for name in ("models", "output", "report")}




def load_origin(model_root: Path, research_root=None):
    """Integrity-check and deserialize the native6 candidate plus reused C0/shared experts."""
    model_root = Path(model_root)
    execution = json.loads((model_root / "execution.json").read_text())
    contract = execution["contract"]
    root = native.resolve_research_root(research_root)
    protocol_path = PROJECT_ROOT / contract["protocol_path"]
    if _sha256(protocol_path) != contract["protocol_sha256"]:
        raise ValueError("protocol changed since origin training")
    _, protocol = _read_protocol(protocol_path, root)
    backend, hashes = native.uncertainty._activate_archive_backend(root)
    if _source_hashes(hashes) != contract["source_sha256"]:
        raise ValueError("source hashes differ from origin receipt")
    if _sha256(PROJECT_ROOT / contract["canonical_path"]) != contract["canonical_sha256"]:
        raise ValueError("canonical data changed since training")
    year = int(contract["year"])
    if (contract["run_id"] != protocol["run_id"] or contract["scope"] != protocol["scope"]
            or contract["versions"] != protocol["versions"]
            or contract["input_sha256"] != protocol["inputs"]
            or contract["source_sha256"] != protocol["source_sha256"]
            or contract["protocol_path"] != str(protocol_path.relative_to(PROJECT_ROOT))
            or contract["protocol_sha256"] != _sha256(protocol_path)
            or contract["canonical_path"] != protocol["canonical_path"]
            or contract["canonical_sha256"] != protocol["canonical_sha256"]
            or contract["split_ids_sha256"] != protocol["partitions"][str(year)]["membership_sha256"]
            or model_root.resolve() != _origin_paths(contract["run_id"], year)["models"].resolve()):
        raise ValueError("native6 origin contract does not match its frozen protocol")
    completion_path = _paths(contract["run_id"])["report"] / "training_completed.json"
    completion = json.loads(completion_path.read_text())
    if (completion["status"] != "TRAINING_COMPLETE_NOT_SCORED"
            or completion["evaluation_status"] != "NOT_RUN"
            or completion["protocol_sha256"] != contract["protocol_sha256"]):
        raise ValueError("native6 study is not a hash-matched unscored completion")
    for relative, digest in completion["artifacts"].items():
        artifact = PROJECT_ROOT / relative
        if not artifact.is_file() or _sha256(artifact) != digest:
            raise ValueError(f"native6 study artifact integrity mismatch: {relative}")
    # All checkpoint receipts are checked before any joblib deserialization.
    shared_path = model_root / "shared/model.joblib"
    model_paths = [model_root / f"native6/seed_{seed}/model.joblib" for seed in SEEDS]
    calibration_path = model_root / "native6/calibration/model.joblib"
    expected = [(shared_path, {**contract, "component": "shared"})]
    expected += [(path, {**contract, "component": "neural", "variant": "native6", "seed": seed})
                 for path, seed in zip(model_paths, SEEDS)]
    expected += [(calibration_path, {**contract, "component": "calibration", "variant": "native6"})]
    for path, expected_contract in expected:
        receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text())
        from src.models.a0_native_refit import _contract_digest, _json_safe
        if (receipt["contract"] != _json_safe(expected_contract)
                or receipt["contract_sha256"] != _contract_digest(expected_contract)
                or receipt["checkpoint_sha256"] != _sha256(path)):
            raise ValueError(f"native6 checkpoint receipt/hash mismatch: {path}")
    shared = joblib_load(shared_path)
    models = [joblib_load(path) for path in model_paths]
    calibration = joblib_load(calibration_path)
    return backend, shared, models, calibration


def joblib_load(path):
    import joblib
    return joblib.load(path)


def _predict(backend, shared, models, calibration, data, indices):
    raw, history = native._subset_inputs(data, indices)
    best_of = data["frame"].best_of.to_numpy(dtype=int)[indices]
    forecast_data = {"frame": data["frame"], "raw": data["raw"], "history": data["history"]}
    logits, rho, raw_seed = _calculate_forecast(models, forecast_data, indices, backend)
    other = native._other_logits(shared, raw, history, best_of)
    full, neural, _ = native._predict_mixture(backend, logits, other, calibration["slopes"],
                                             raw["gates"], best_of, calibration)
    return {"full": full, "neural": neural, "raw_neural": raw_seed.mean(axis=2)}, logits, rho, raw, history, other


def _probability_block(variant, values):
    return {f"{variant}__{component}__{mode}": values[component][:, index]
            for component in COMPONENTS for index, mode in enumerate(MODES_OUT)}


def _endpoint_frame(frame, indices, partition, model_year, variant_outputs):
    result = frame.iloc[indices][["golgg_match_id", "date", "y", "best_of"]].copy().reset_index(drop=True)
    result["model_year"] = int(model_year)
    result["label_release_day"] = pd.to_datetime(frame.iloc[indices].result_day).dt.strftime("%Y-%m-%d").to_numpy()
    result["partition"] = partition
    result["origin_date"] = f"{int(model_year)}-01-01"
    result["release_scenario"] = "historical_actual_roster_daily_release_not_issued_forecast"
    result["training_output_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    for variant, outputs in variant_outputs.items():
        for column, values in _probability_block(variant, outputs["probabilities"]).items():
            result[column] = values
        for score_kind, values in outputs["scoreline"].items():
            if score_kind.startswith("raw_"):
                result[f"{variant}__raw_scoreline_nll__{score_kind.removeprefix('raw_')}"] = values
            else:
                result[f"{variant}__scoreline_nll__{score_kind}"] = values
    return result


def _scoreline(backend, logits, rho, calibration, raw, best_of, wins_a, wins_b):
    return native._neural_scoreline_nll(backend, logits, rho, calibration, raw["gates"],
                                        best_of, wins_a, wins_b)
def _scoreline_export(scores):
    result = {}
    for key, values in scores.items():
        if key.startswith("nll_neural_"):
            result[key.removeprefix("nll_neural_")] = values
        elif key.startswith("nll_raw_neural_"):
            result["raw_" + key.removeprefix("nll_raw_neural_")] = values
    return result


def _fit_origin(paths, protocol_path, protocol, root, data, split, smoke):
    from src.models.a0_reliability import load_origin
    origin_started_at_utc = datetime.now(timezone.utc).isoformat()
    year = int(paths["models"].name)
    reference_record = protocol["reference_origins"][str(year)]
    reference_report = PROJECT_ROOT / reference_record["completed_path"]
    if _sha256(reference_report) != reference_record["completed_sha256"]:
        raise ValueError(f"Plan07 native12 receipt changed for {year}")
    for relative, digest in reference_record["artifacts"].items():
        artifact_path = PROJECT_ROOT / relative
        if not artifact_path.is_file() or _sha256(artifact_path) != digest:
            raise ValueError(f"Plan07 native12 artifact changed: {relative}")
    control_root = PROJECT_ROOT / f"data/06_models/model_reliability/{REFERENCE_RUN}/neural/{year}"
    control_execution = json.loads((control_root / "execution.json").read_text())
    if _sha256(control_root / "execution.json") != reference_record["control_execution_sha256"]:
        raise ValueError(f"Plan07 native12 execution redirect changed for {year}")
    reference = control_execution.get("reference")
    control_artifact_root = PROJECT_ROOT / reference_record["control_artifact_root"]
    expected_control_artifact_root = (
        PROJECT_ROOT / "data/06_models/a0_data_focus" / reference["run_id"] if reference else control_root)
    if control_artifact_root.resolve() != expected_control_artifact_root.resolve():
        raise ValueError(f"Plan07 saved-control redirect changed for {year}")
    if _verify_redirected_control(control_execution) != reference_record["redirected_completion"]:
        raise ValueError(f"Plan07 redirected control provenance changed for {year}")
    backend, shared, control_models, control_calibration = load_origin(control_root, "control", root)
    frame = data["frame"]
    train, cal, assessment = (split[key] for key in ("train", "cal", "validation"))
    initial_hashes, control_initial = {}, {}
    for seed, model in zip(SEEDS, control_models):
        receipt_path = control_artifact_root / f"control/seed_{seed}/model.joblib.receipt.json"
        control_initial[str(seed)] = json.loads(receipt_path.read_text())["details"]["initial_state_sha256"]
        initial_model = backend["nonlinear_history"].NonlinearHistoryModel(
            shared["baseline"], shared["normalizer"], seed=seed, epochs=6)
        initial_model.main_objective = "terminal"
        initial_hashes[str(seed)] = _torch_state_hash(initial_model)
    if initial_hashes != control_initial:
        raise ValueError(f"native6 constructor initialization differs from saved C0: {year}")

    contract = {
        "schema": "a0_reliability_tradeoff_origin_v1", "run_id": protocol["run_id"], "year": year,
        "scope": "SMOKE_DIAGNOSTIC_ONLY" if smoke else protocol["scope"],
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)), "protocol_sha256": _sha256(protocol_path),
        "source_sha256": protocol["source_sha256"], "input_sha256": protocol["inputs"],
        "canonical_path": protocol["canonical_path"], "canonical_sha256": protocol["canonical_sha256"],
        "versions": _versions(), "split_ids_sha256": protocol["partitions"][str(year)]["membership_sha256"],
        "reference": protocol["reference_origins"][str(year)],
    }
    _write_json_exclusive(paths["models"] / "execution.json", {
        "schema": "a0_reliability_tradeoff_execution_v1", "contract": contract,
        "started_at_utc": origin_started_at_utc,
        "execution_scope": "SMOKE_DIAGNOSTIC_ONLY" if smoke else "FULL_FIT",
        "smoke": bool(smoke), "candidate_epochs": 1 if smoke else 6,
    })
    shared_path = control_artifact_root / "shared/model.joblib"
    _atomic_checkpoint(paths["models"] / "shared/model.joblib", shared,
                       {**contract, "component": "shared"},
                       {"source_path": str(shared_path.relative_to(PROJECT_ROOT)),
                        "source_sha256": _sha256(shared_path),
                        "reuse": "Plan07 saved C0 shared experts; no refit"})
    epochs = 1 if smoke else 6
    models, training_receipts = [], []
    for seed in SEEDS:
        started = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        model = backend["nonlinear_history"].NonlinearHistoryModel(
            shared["baseline"], shared["normalizer"], seed=seed, epochs=epochs)
        model.main_objective = "terminal"
        initial = _torch_state_hash(model)
        if initial != control_initial[str(seed)]:
            raise ValueError(f"candidate initial state mismatch: year={year}, seed={seed}")
        fit_beta_target_on_device(
            model, data["raw"], data["history"], None, train, (frame.best_of.to_numpy(dtype=int)-1)//2,
            data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
            device="cpu", objective="terminal",
        )
        expected_updates = epochs * math.ceil(len(train) / 128)
        if len(model.train_losses) != epochs or len(model.epoch_diagnostics) != epochs:
            raise ValueError(f"candidate fit did not execute exactly {epochs} epochs")
        old_receipt = control_artifact_root / f"supervised_18/seed_{seed}/model.joblib.receipt.json"
        frozen_trajectory = reference_record["first_phase_receipts"][str(seed)]
        if (not smoke and
                (str(old_receipt.relative_to(PROJECT_ROOT)) != frozen_trajectory["path"]
                 or _sha256(old_receipt) != frozen_trajectory["sha256"])):
            raise ValueError(f"saved supervised_18 trajectory changed after freeze: {year}/{seed}")
        if smoke:
            first_phase = {"available": False, "reason": "SMOKE_TRAJECTORY_COMPARISON_SKIPPED"}
        else:
            if not old_receipt.is_file():
                raise FileNotFoundError(f"saved supervised_18 trajectory receipt missing: {old_receipt}")
            old_details = json.loads(old_receipt.read_text())["details"]
            old_phase = old_details["first_phase"] if year == 2026 else old_details["phases"][0]
            old_main = np.asarray(old_phase["main_losses"], dtype=float)
            old_aux = np.asarray(old_phase["auxiliary_losses"], dtype=float)
            expected_first_phase_updates = 6 * math.ceil(len(train) / 128)
            if (len(old_main) != 6 or len(old_aux) != 6
                    or old_phase["optimizer_updates"] != expected_first_phase_updates):
                raise ValueError(f"saved supervised_18 first phase is not the frozen six-epoch trajectory: {year}/{seed}")
            first_phase = {
                "available": True,
                "receipt_path": str(old_receipt.relative_to(PROJECT_ROOT)),
                "receipt_sha256": _sha256(old_receipt),
                "receipt_phase_key": "details.first_phase" if year == 2026 else "details.phases[0]",
                "saved_optimizer_updates": old_phase["optimizer_updates"],
                "main_losses_exact_match": bool(np.array_equal(old_main, np.asarray(model.train_losses))),
                "auxiliary_losses_exact_match": bool(np.array_equal(old_aux, np.asarray(model.auxiliary_losses))),
                "main_losses_max_abs_difference": float(np.max(np.abs(old_main - model.train_losses))),
                "auxiliary_losses_max_abs_difference": float(np.max(np.abs(old_aux - model.auxiliary_losses))),
                "saved_first_phase_rng_state": False,
            }
        details = {"seed": seed, "initial_state_sha256": initial,
                   "first_six_phase_comparison": first_phase,
                   "final_state_sha256": _torch_state_hash(model),
                   "optimizer_updates": expected_updates,
                   "phase": {"epochs": epochs, "optimizer_updates": expected_updates,
                             "main_losses": model.train_losses,
                             "auxiliary_losses": model.auxiliary_losses,
                             "diagnostics": model.epoch_diagnostics},
                   "gate_rng_final_state": model.dropout_rng_final_state,
                   "started_at_utc": started,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "runtime_seconds": time.monotonic() - started_monotonic}
        _atomic_checkpoint(paths["models"] / f"native6/seed_{seed}/model.joblib", model,
                           {**contract, "component": "neural", "variant": "native6", "seed": seed}, details)
        models.append(model)
        training_receipts.append(details)

    # Fit calibration on candidate CAL logits only, carrying forward saved C0 slopes.
    if len(cal) == 0:
        raise ValueError("candidate calibration population is empty")
    raw, history = native._subset_inputs(data, cal)
    best_of = frame.best_of.to_numpy(dtype=int)[cal]
    logits, _, _ = _calculate_forecast(
        models, {"frame": frame, "raw": data["raw"], "history": data["history"]}, cal, backend)
    other = native._other_logits(shared, raw, history, best_of)
    candidate_calibration = native._calibrate(
        backend, logits, other, raw["gates"], frame.iloc[cal],
        data["wins_a"][cal], data["wins_b"][cal], slopes=control_calibration["slopes"])
    _atomic_checkpoint(paths["models"] / "native6/calibration/model.joblib", candidate_calibration,
                       {**contract, "component": "calibration", "variant": "native6"},
                       {"cal_rows": len(cal), "cal_indices_sha256": _contract_digest(
                           frame.iloc[cal].golgg_match_id.astype(str).tolist()),
                        "slopes_source": "saved C0 calibration",
                        "beta_records": candidate_calibration["beta_records"]})
    calibration_details = {"cal_rows": len(cal), "slopes_source": "saved C0 calibration"}

    all_endpoints, probabilities_by_partition = [], {}
    candidate_logits = {}
    for part, indices in (("train", train), ("cal", cal), ("assessment", assessment)):
        candidate, logits, rho, raw, history, other = _predict(
            backend, shared, models, candidate_calibration, data, indices)
        control, c_logits, c_rho, c_raw, _, control_other = _predict(
            backend, shared, control_models, control_calibration, data, indices)
        if part == "assessment" and not smoke:
            reference_file = PROJECT_ROOT / reference_record["native12_output_path"]
            if _sha256(reference_file) != reference_record["native12_output_sha256"]:
                raise ValueError(f"saved Plan07 native12 output changed for {year}")
            saved = pd.read_parquet(reference_file)
            saved = saved.set_index(saved.golgg_match_id.astype(str))
            ids = frame.iloc[indices].golgg_match_id.astype(str)
            expected = saved.loc[ids, [f"p_{kind}_{mode}" for kind in COMPONENTS for mode in MODES_OUT]]
            for component in COMPONENTS:
                observed = control[component]
                expected_values = expected[[f"p_{component}_{mode}" for mode in MODES_OUT]].to_numpy(float)
                if not np.allclose(observed, expected_values, atol=1e-7, rtol=0):
                    raise ValueError("replayed native12 assessment differs from saved Plan07 control")
        c_score = _scoreline(backend, c_logits, c_rho, control_calibration, c_raw,
                             frame.iloc[indices].best_of.to_numpy(int), data["wins_a"][indices], data["wins_b"][indices])
        n_score = _scoreline(backend, logits, rho, candidate_calibration, raw,
                             frame.iloc[indices].best_of.to_numpy(int), data["wins_a"][indices], data["wins_b"][indices])
        variant_outputs = {
            "native12": {"probabilities": control, "scoreline": _scoreline_export(c_score)},
            "native6": {"probabilities": candidate, "scoreline": _scoreline_export(n_score)},
        }
        all_endpoints.append(_endpoint_frame(frame, indices, part.upper(), year, variant_outputs))
        probabilities_by_partition[part] = {"native12": control, "native6": candidate}
        candidate_logits[part] = {
            "indices": indices, "native6_logits": logits, "native6_rho": rho,
            "native12_logits": c_logits, "native12_rho": c_rho,
            "native6_other_logits": other, "native12_other_logits": control_other,
        }
    calibration_details["scales"] = {
        mode: analysis.confidence_match_scale(probabilities_by_partition["cal"]["native6"]["full"][:, index],
                                              probabilities_by_partition["cal"]["native12"]["full"][:, index])
        for index, mode in enumerate(MODES_OUT)
    }
    for mode_index, mode in enumerate(MODES_OUT):
        probabilities_by_partition["assessment"]["confidence_matched__full__" + mode] = analysis.apply_confidence_scale(
            probabilities_by_partition["assessment"]["native12"]["full"][:, mode_index],
            calibration_details["scales"][mode])

    endpoint = pd.concat(all_endpoints, ignore_index=True)
    for part in ("train", "cal", "assessment"):
        rows = endpoint.partition.eq(part.upper()).to_numpy()
        for mode_index, mode in enumerate(MODES_OUT):
            endpoint.loc[rows, f"confidence_matched__full__{mode}"] = analysis.apply_confidence_scale(
                probabilities_by_partition[part]["native12"]["full"][:, mode_index],
                calibration_details["scales"][mode])
    endpoint.to_parquet(paths["output"] / "endpoints.parquet", index=False)
    np.savez_compressed(paths["output"] / "native6_logits.npz",
                        **{f"{part}_{key}": value for part, values in candidate_logits.items() for key, value in values.items()})
    assessment_frame = frame.iloc[assessment][["golgg_match_id", "date", "y", "best_of", "tournament_name"]].copy().reset_index(drop=True)
    assessment_frame["model_year"] = year
    assessment_frame["label_release_day"] = pd.to_datetime(frame.iloc[assessment].result_day).dt.strftime("%Y-%m-%d").to_numpy()
    assessment_frame["origin_date"] = f"{year}-01-01"
    assessment_frame["release_scenario"] = "historical_actual_roster_daily_release_not_issued_forecast"
    assessment_frame["training_output_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    for variant, values in probabilities_by_partition["assessment"].items():
        if variant.startswith("confidence_matched"):
            assessment_frame[variant] = values
        else:
            for column, values_column in _probability_block(variant, values).items():
                assessment_frame[column] = values_column
    return {"year": year, "partition_counts": {key: len(value) for key, value in split.items()},
            "boundaries": protocol["partitions"][str(year)]["boundaries"],
            "origin_started_at_utc": origin_started_at_utc,
            "origin_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_state_sha256": control_initial, "training": training_receipts,
            "calibration": calibration_details, "assessment": assessment_frame,
            "endpoint_path": str((paths["output"] / "endpoints.parquet").relative_to(PROJECT_ROOT)),
            "native6_logits_path": str((paths["output"] / "native6_logits.npz").relative_to(PROJECT_ROOT))}


def _verify_runtime(protocol_path, protocol, root, data):
    backend, hashes = native.uncertainty._activate_archive_backend(root)
    if _source_hashes(hashes) != protocol["source_sha256"]:
        raise ValueError("frozen sources changed before fitting")
    if _relative_inputs(data, root) != protocol["inputs"] or data["bank_receipt_sha256"] != protocol["bank_receipt_sha256"]:
        raise ValueError("frozen feature bank or inputs changed")
    current = _split_contract(data)
    if current != protocol["partitions"]:
        raise ValueError("frozen annual memberships changed")
    return backend


def run_experiment(protocol_path: Path, research_root=None, smoke: bool = False) -> dict:
    root = native.resolve_research_root(research_root)
    protocol_path = Path(protocol_path).resolve()
    protocol_path, protocol = _read_protocol(protocol_path, root)
    if _sha256(PROJECT_ROOT / protocol["initial_goal_path"]) != protocol["initial_goal_sha256"]:
        raise ValueError("live Markdown goal changed since protocol freeze")
    protocol_sha256 = _sha256(protocol_path)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    backend, _ = native.uncertainty._activate_archive_backend(root)
    data = native.uncertainty._load_training_data(root)
    native.partition_indices(data["frame"], history=data["history"], raw=data["raw"])
    _verify_runtime(protocol_path, protocol, root, data)
    _validate_canonical(data, pd.read_parquet(PROJECT_ROOT / protocol["canonical_path"]))
    training_started_at_utc = datetime.now(timezone.utc).isoformat()
    training_started_monotonic = time.monotonic()
    run_id = protocol["run_id"] + ("_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") if smoke else "")
    if smoke:
        smoke_paths = _paths(run_id)
        for path in smoke_paths.values():
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"refusing nonempty smoke diagnostic directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        frame = data["frame"]
        split = annual_partitions(frame, 2024)
        rng = np.random.default_rng(20261001)
        split["train"] = np.sort(rng.choice(split["train"], 1024, replace=False))
        split["cal"] = np.sort(rng.choice(split["cal"], 192, replace=False))
        split["validation"] = np.asarray([
            next(index for index in split["validation"] if frame.iloc[index].best_of == best_of)
            for best_of in (1, 3, 5)
        ])
        identities = [classify_competition(name, pd.Timestamp(day).date())
                      for name, day in zip(frame.tournament_name, frame.date)]
        frame["tier"] = [item.tier.value for item in identities]
        frame["family"] = [item.family for item in identities]
        paths = _origin_paths(run_id, 2024)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=False)
        result = _fit_origin(paths, protocol_path, protocol, root, data, split, True)
        smoke_finished_at_utc = datetime.now(timezone.utc).isoformat()
        smoke_result = {
            "status": "SMOKE_DIAGNOSTIC_COMPLETE", "scope": "ISOLATED_DIAGNOSTIC_PREFLIGHT",
            "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)),
            "protocol_sha256": _sha256(protocol_path), "run_id": run_id,
            "started_at_utc": training_started_at_utc, "completed_at_utc": smoke_finished_at_utc,
            "runtime_seconds": time.monotonic() - training_started_monotonic,
            "origins": [2024], "candidate_models": len(result["training"]),
            "epochs_per_model": [len(item["phase"]["main_losses"]) for item in result["training"]],
            "optimizer_updates_total": sum(item["optimizer_updates"] for item in result["training"]),
            "rows": {"train": 1024, "cal": 192, "assessment": 3},
            "purpose": "exercise saved-control loading, initialization, one-epoch fit, CAL calibration, endpoint probability and scoreline-likelihood diagnostics, and isolated serialization; not the six-epoch recipe, nine-fit completion, or a formal study decision",
            "artifacts": {str(file.relative_to(PROJECT_ROOT)): _sha256(file)
                          for name in ("models", "output") for file in sorted(smoke_paths[name].rglob("*"))
                          if file.is_file()},
        }
        _write_json_exclusive(smoke_paths["report"] / "smoke_diagnostic.json", smoke_result)
        return smoke_result
    paths = _paths(run_id)
    for key in ("models", "output", "report"):
        if paths[key].exists() and any(paths[key].iterdir()):
            raise FileExistsError(f"refusing nonempty final output directory: {paths[key]}")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    frame = data["frame"]
    identities = [classify_competition(name, pd.Timestamp(day).date())
                  for name, day in zip(frame.tournament_name, frame.date)]
    frame["tier"] = [item.tier.value for item in identities]
    frame["family"] = [item.family for item in identities]
    results, assessment_parts, endpoint_parts = {}, [], []
    for year in YEARS:
        split = annual_partitions(frame, year)
        expected = protocol["partitions"][str(year)]
        if ({key: len(value) for key, value in split.items()} != expected["counts"]
                or {key: _contract_digest(frame.iloc[value].golgg_match_id.astype(str).tolist())
                    for key, value in split.items()} != expected["membership_sha256"]):
            raise ValueError(f"frozen membership changed for {year}")
        origin_paths = _origin_paths(run_id, year)
        for path in origin_paths.values():
            path.mkdir(parents=True, exist_ok=False)
        result = _fit_origin(origin_paths, protocol_path, protocol, root, data, split, False)
        report_record = {key: value for key, value in result.items() if key != "assessment"}
        _write_json_exclusive(origin_paths["report"] / "training_receipt.json", report_record)
        _write_json_exclusive(origin_paths["report"] / "confidence_scales.json", result["calibration"]["scales"])
        results[str(year)] = result
        assessment_parts.append(results[str(year)].pop("assessment"))
        endpoint_parts.append(pd.read_parquet(PROJECT_ROOT / results[str(year)]["endpoint_path"]))
    canonical = pd.read_parquet(PROJECT_ROOT / protocol["canonical_path"])
    assessment = pd.concat(assessment_parts, ignore_index=True)
    if (assessment.golgg_match_id.astype(str).duplicated().any()
            or len(assessment) != 11550
            or set(assessment.golgg_match_id.astype(str)) != set(canonical.golgg_match_id.astype(str))):
        raise ValueError("candidate assessment is not complete and unique")
    canonical_ids = canonical.golgg_match_id.astype(str).tolist()
    assessment["_id"] = assessment.golgg_match_id.astype(str)
    canonical["_id"] = canonical.golgg_match_id.astype(str)
    mapped = assessment.set_index("_id").loc[canonical_ids].reset_index(drop=True)
    bank = frame.assign(_id=frame.golgg_match_id.astype(str)).set_index("_id")
    bank_rows = bank.loc[canonical_ids]
    for column in ("team1_id", "team2_id", "y", "best_of"):
        if column in canonical and column in bank_rows:
            if not np.array_equal(canonical[column].astype(str).to_numpy(),
                                  bank_rows[column].astype(str).to_numpy()):
                raise ValueError(f"canonical side/label identity mismatch: {column}")
    if not np.array_equal(pd.to_datetime(canonical.date).to_numpy(),
                          pd.to_datetime(bank_rows.date).to_numpy()):
        raise ValueError("canonical dates do not align to bank assessment")
    if not np.array_equal(canonical.y.to_numpy(), mapped.y.to_numpy()):
        raise ValueError("canonical labels do not align to bank assessment")
    if "result_day" in canonical and not np.array_equal(
            pd.to_datetime(canonical.result_day).dt.strftime("%Y-%m-%d").to_numpy(),
            mapped.label_release_day.to_numpy()):
        raise ValueError("canonical label-release dates do not align")
    for column in mapped.columns:
        if column not in canonical.columns and column != "_id":
            canonical[column] = mapped[column].to_numpy()
    canonical["label_release_day"] = mapped.label_release_day.to_numpy()
    canonical["model_year"] = mapped.model_year.to_numpy()
    canonical.drop(columns=["_id"], inplace=True)
    canonical["reliability_tradeoff_scenario"] = "HISTORICAL_DEVELOPMENT_NOT_ISSUED_FORECAST"
    if sum(item["optimizer_updates"] for result in results.values() for item in result["training"]) != 13590:
        raise ValueError("the nine native6 fits did not execute the frozen 13,590 updates")
    for year in YEARS:
        result = results[str(year)]
        result["assessment_rows"] = protocol["partitions"][str(year)]["counts"]["validation"]
        result["confidence_scale"] = result["calibration"].get("scales", {})
    exported_at_utc = datetime.now(timezone.utc).isoformat()
    canonical["reliability_tradeoff_exported_at_utc"] = exported_at_utc
    canonical.to_parquet(paths["output"] / "predictions.parquet", index=False)
    endpoint_export = pd.concat(endpoint_parts, ignore_index=True)
    endpoint_export["reliability_tradeoff_exported_at_utc"] = exported_at_utc
    endpoint_export.to_parquet(paths["output"] / "endpoint_predictions.parquet", index=False)
    _, final_backend_hashes = native.uncertainty._activate_archive_backend(root)
    if (_source_hashes(final_backend_hashes) != protocol["source_sha256"]
            or _sha256(protocol_path) != protocol_sha256):
        raise ValueError("source or protocol changed during native6 training")
    if any(_sha256(root / relative) != digest for relative, digest in protocol["inputs"].items()):
        raise ValueError("a frozen model input changed during native6 training")
    if (_sha256(PROJECT_ROOT / protocol["canonical_path"]) != protocol["canonical_sha256"]
            or _sha256(PROJECT_ROOT / protocol["manifest_path"]) != protocol["manifest_sha256"]
            or _sha256(root / protocol["manifest_input_path"]) != protocol["manifest_input_sha256"]):
        raise ValueError("a frozen canonical, manifest, or raw manifest input changed during native6 training")
    for reference_record in protocol["reference_origins"].values():
        if _sha256(PROJECT_ROOT / reference_record["completed_path"]) != reference_record["completed_sha256"]:
            raise ValueError("a Plan07 control receipt changed during native6 training")
        for relative, digest in reference_record["artifacts"].items():
            if _sha256(PROJECT_ROOT / relative) != digest:
                raise ValueError(f"a Plan07 control artifact changed during native6 training: {relative}")
        execution = json.loads((PROJECT_ROOT / reference_record["control_execution_path"]).read_text())
        if _verify_redirected_control(execution) != reference_record["redirected_completion"]:
            raise ValueError("a redirected control artifact changed during native6 training")
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    artifacts = {str(file.relative_to(PROJECT_ROOT)): _sha256(file)
                 for stage in ("models", "output", "report")
                 for file in sorted(paths[stage].rglob("*")) if file.is_file()}
    completion = {
        "schema": "a0_reliability_tradeoff_training_v1", "status": "TRAINING_COMPLETE_NOT_SCORED",
        "evaluation_status": "NOT_RUN", "scope": protocol["scope"],
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)), "protocol_sha256": protocol_sha256,
        "source_sha256": protocol["source_sha256"], "versions": protocol["versions"],
        "training_started_at_utc": training_started_at_utc, "training_completed_at_utc": completed_at_utc,
        "runtime_seconds": time.monotonic() - training_started_monotonic,
        "origins": results, "confidence_scales": {year: record["calibration"]["scales"]
                                                   for year, record in results.items()},
        "memberships": protocol["partitions"], "artifacts": artifacts,
        "predictions_path": str((paths["output"] / "predictions.parquet").relative_to(PROJECT_ROOT)),
        "endpoint_predictions_path": str((paths["output"] / "endpoint_predictions.parquet").relative_to(PROJECT_ROOT)),
    }
    _write_json_exclusive(paths["report"] / "training_completed.json", completion)
    return completion
