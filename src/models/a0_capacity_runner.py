"""Frozen orchestration for native A0 capacity and conditional distillation studies."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import importlib.metadata
import platform
import joblib
import numpy as np
import pandas as pd
import torch

from src.models import a0_uncertainty as uncertainty
from src.models import a0_data_focus as native
from datetime import date, datetime
import math
from src.models.a0_target_experiment import resolve_research_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN08_CONTROL = Path("data/07_model_output/model_reliability/model_reliability_20261001_055321/neural/2026/control.parquet")
ANNUAL_A0_PREDICTIONS = Path("data/07_model_output/a0_sports_correction/c0_raw_connector_20261002_033658/predictions.parquet")
REFERENCE_RUN = "a0_data_focus_20260929_192443"
SEEDS = (20260910, 20260911, 20260912)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NaT or value is pd.NA:
        return None
    return value

def _runtime_environment() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib")
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in packages}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(values: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(values), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(root.rglob("*.py")) if path.is_file()}


def frozen_contract(run_id: str, research_root: Path) -> dict[str, Any]:
    return {"schema": "a0_capacity_v1", "run_id": run_id,
            "initialization": "native_c0_seeded_first_head_row",
            "assembly": "full_precision_probability_mixture",
            "reference_run": REFERENCE_RUN, "widths": [16, 32, 64],
            "seeds": list(SEEDS), "epochs": [12, 24],
            "training": {"batch_size": 128, "device": "cpu", "threads": 2,
                         "deterministic": True, "optimizer": "AdamW", "learning_rate": .001,
                         "network_weight_decay": .01, "eta_weight_decay": 0.0},
            "calibration": "native_joint_fit_on_CAL; preserve saved C0 slopes and mixture objects",
            "mixture_contract": "saved_full_precision_per_mode_probability_mixture",
            "protocol_correction": "retain first row of native seeded three-output head; full-precision saved probability mixture",
            "distillation": {"student_cell": "w16_e12", "teacher_cell": "w32_e12",
                "teacher_column": "w32_e12__full", "lambda": .5, "temperature": 1.,
                "direct_control": "existing_w16_e12_checkpoint",
                "verified_teacher_advantage_candidates": [
                    "w64_e12-w32_e12", "w32_e24-w32_e12"],
                "kd_column": "w16_e12_kd", "direct_column": "w16_e12_direct"},
            "research_root": str(Path(research_root).resolve())}


def verify_protocol_hashes(protocol: dict[str, Any], root: Path) -> None:
    root = Path(root).resolve()
    for relative, expected in protocol.get("inputs", {}).items():
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {relative}")
    for relative, expected in protocol.get("source_snapshot", {}).items():
        if relative.startswith("research:"):
            path = Path(protocol["research_root"]) / relative.removeprefix("research:")
        else:
            path = PROJECT_ROOT / relative.removeprefix("project:")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")


def _source_snapshot(research_root: Path, backend_hashes: dict[str, str]) -> dict[str, str]:
    focus = native._source_hashes(backend_hashes)
    result = {}
    for relative, digest in focus.items():
        key = relative if relative.startswith(("project:", "research:")) else f"project:{relative}"
        result[key] = digest
    for relative in ("src/models/a0_capacity_experiment.py", "src/models/a0_capacity_runner.py",
                     "src/analysis/a0_capacity_experiment.py",
                     "src/analysis/production_market_benchmark.py", "src/analysis/bootstrap.py",
                     "scripts/train_a0_capacity_experiment.py"):
        path = PROJECT_ROOT / relative
        result[f"project:{relative}"] = _sha256(path)
    return result


def _validate_run_id(run_id: str) -> str:
    if (not run_id or run_id in {".", ".."} or Path(run_id).name != run_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id)):
        raise ValueError("run_id must be a single safe path component")
    return run_id


def _execution_id(run_id: str, smoke: bool) -> str:
    return f"{run_id}_smoke" if smoke else run_id


def _market_protocol_rows() -> tuple[dict[str, Any], dict[str, str]]:
    report_root = PROJECT_ROOT / "data/08_reporting/model_redesign_research/model_redesign_research_20261004_01"
    matched_path = report_root / "matched_diagnostic_rows.parquet"
    recipe_path = report_root / "reproduction_recipe.json"
    redesign_protocol_path = PROJECT_ROOT / "data/05_model_input/model_redesign_research/model_redesign_research_20261004_01/protocol.json"
    betting_root = PROJECT_ROOT / "data/05_model_input/betting_ev_reliability/c0_betting_20261002_085551"
    selected_path = betting_root / "selected_v2.parquet"
    inventory_path = betting_root / "opportunity_inventory_v2.parquet"
    opportunities_path = betting_root / "opportunities_v2.parquet"
    redesign_protocol = json.loads(redesign_protocol_path.read_text())
    listed_sources = [PROJECT_ROOT / source["path"] for source in redesign_protocol["sources"].values()]
    for source in listed_sources:
        expected = next(item["sha256"] for item in redesign_protocol["sources"].values()
                        if PROJECT_ROOT / item["path"] == source)
        if _sha256(source) != expected:
            raise ValueError(f"frozen betting source hash mismatch: {source}")
    market = pd.read_parquet(matched_path)
    selected = pd.read_parquet(selected_path)
    inventory = pd.read_parquet(inventory_path)
    if not market.row_id.astype(str).is_unique:
        raise ValueError("matched diagnostic fixtures have duplicate row identities")
    accepted = market.loc[market.source.eq("pinnacle")].copy()
    raw = selected.loc[selected.source.eq("pinnacle") & selected.model_key.eq("new_connector")
                       & selected.policy_name.eq("raw_ev_gt_5")].copy()
    if len(accepted) != 1615 or len(raw) != 330:
        raise ValueError("frozen diagnostic accepted or selected cohort changed")
    if set(raw.row_id.astype(str)) - set(accepted.row_id.astype(str)):
        raise ValueError("selected bets are not inside accepted market fixtures")
    if len(inventory.loc[inventory.source.eq("pinnacle")]) != 1619:
        raise ValueError("frozen Pinnacle inventory no longer contains 1619 rows")
    accept_rows = accepted.to_dict("records")
    near_count = int((raw.ev.astype(float).gt(.05) & raw.ev.astype(float).le(.10)).sum())
    if near_count != 76:
        raise ValueError(f"saved selected near-band cohort changed: expected 76, got {near_count}")
    selected_rows = []
    for record in raw.to_dict("records"):
        near_band = .05 < float(record["ev"]) <= .10
        selected_rows.append({**record, "selected_group": "connector_raw_ev_gt_5",
                              "near_band": near_band,
                              "market_q": float(record["q"]), "y": int(record["y"]),
                              "decimal_odds": float(record["odds"]),
                              "market_q_chosen": float(record["q"] if record["side"] == "A" else 1 - record["q"]),
                              "won": bool(record["won"])})
    rows = {"enabled": True, "accepted_fixture_count": 1615, "inventory_count": 1619,
            "raw_ev_selected_count": 330, "near_band_count": 76, "take_rate": .12,
            "accepted_rows": accept_rows, "selection_rows": selected_rows,
            "inventory_rows": inventory.loc[inventory.source.eq("pinnacle")].to_dict("records"),
            "status": "FROZEN_SAVED_SELECTIONS"}
    source_paths = [matched_path, recipe_path, redesign_protocol_path, *listed_sources,
                    selected_path, inventory_path, opportunities_path]
    hashes = {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in set(source_paths)}
    return rows, hashes


def _partition(frame: pd.DataFrame, *, smoke: bool, history: Any, raw: Any) -> dict[str, np.ndarray]:
    split = native.partition_indices(frame, history=history, raw=raw)
    if not smoke:
        for name, count in {"train": 36205, "cal": 1328, "validation": 3103}.items():
            if len(split[name]) != count:
                raise ValueError(f"frozen {name} population differs: expected {count}, got {len(split[name])}")
    return split


def _reference_paths() -> tuple[Path, Path]:
    models = PROJECT_ROOT / "data/06_models/a0_data_focus" / REFERENCE_RUN
    calibration = models / "control/calibration/model.joblib"
    if not calibration.is_file():
        raise FileNotFoundError(f"pinned C0 calibration missing: {calibration}")
    return models, calibration


def freeze_run(run_id: str, research_root: str | Path | None = None) -> Path:
    run_id = _validate_run_id(run_id)
    research_root = resolve_research_root(research_root)
    backend, backend_hashes = uncertainty._activate_archive_backend(research_root)
    data = uncertainty._load_training_data(research_root)
    split = _partition(data["frame"], smoke=False, history=data["history"], raw=data["raw"])
    model_root, calibration_path = _reference_paths()
    execution_path = model_root / "execution.json"
    execution = json.loads(execution_path.read_text())
    reference_protocol = execution["protocol"]
    completion_path = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / REFERENCE_RUN / "completed.json"
    completion = json.loads(completion_path.read_text())
    if (execution["protocol"].get("run_id") != REFERENCE_RUN
            or completion.get("status") != "TRAINING_COMPLETE_NOT_SCORED"
            or completion.get("source_unchanged") is not True
            or execution["contract"].get("source_sha256") != completion["contract"].get("source_sha256")
            or execution["contract"].get("protocol_sha256") != completion["contract"].get("protocol_sha256")):
        raise ValueError("pinned C0 execution/completion evidence is invalid")
    canonical = research_root / reference_protocol["inputs"]["canonical_relative_to_research_root"]
    if not canonical.is_file() or _sha256(canonical) != reference_protocol["inputs"]["canonical_sha256"]:
        raise ValueError(f"pinned canonical assessment input differs: {canonical}")
    canonical_frame = pd.read_parquet(canonical)
    if "golgg_match_id" not in canonical_frame:
        raise ValueError("canonical assessment input has no match identities")
    canonical_frame["golgg_match_id"] = canonical_frame.golgg_match_id.astype(str)
    canonical_frame = canonical_frame.loc[
        (pd.to_datetime(canonical_frame.date) >= "2026-01-01")
        & (pd.to_datetime(canonical_frame.date) < "2027-01-01")].copy()
    canonical_ids = set(canonical_frame.golgg_match_id)
    assessment = data["frame"].iloc[split["validation"]].copy()
    assessment_ids = assessment.golgg_match_id.astype(str).tolist()
    if (len(canonical_frame) != 3103 or canonical_frame.golgg_match_id.duplicated().any()
            or set(assessment_ids) != canonical_ids):
        raise ValueError("canonical assessment identity set differs from feature bank")
    canonical_frame = canonical_frame.set_index("golgg_match_id").loc[assessment_ids]
    for field in ("team1_id", "team2_id", "best_of", "y"):
        if not np.array_equal(canonical_frame[field].astype(str).to_numpy(),
                              assessment[field].astype(str).to_numpy()):
            raise ValueError(f"canonical assessment orientation/date target mismatch: {field}")
    if not np.array_equal(pd.to_datetime(canonical_frame.date).astype(str).to_numpy(),
                          pd.to_datetime(assessment.date).astype(str).to_numpy()):
        raise ValueError("canonical assessment date alignment differs")
    plan08_path = PROJECT_ROOT / PLAN08_CONTROL
    annual_a0_path = PROJECT_ROOT / ANNUAL_A0_PREDICTIONS
    plan08 = pd.read_parquet(plan08_path)
    annual_a0 = pd.read_parquet(annual_a0_path)
    expected_ids = set(assessment_ids)
    if (len(plan08) != 3103 or set(plan08.golgg_match_id.astype(str)) != expected_ids
            or len(annual_a0) != 11550 or not expected_ids.issubset(set(annual_a0.golgg_match_id.astype(str)))):
        raise ValueError("frozen Plan07/08 or annual A0 assessment identity coverage differs")
    market_rows, market_hashes = _market_protocol_rows()
    shared_path = model_root / "shared/model.joblib"
    shared_receipt = shared_path.with_suffix(shared_path.suffix + ".receipt.json")
    calibration_receipt = calibration_path.with_suffix(calibration_path.suffix + ".receipt.json")
    ref_files = [shared_path, shared_receipt, calibration_path, calibration_receipt,
                 execution_path, completion_path]
    ref_files.extend(path for seed in SEEDS for path in (
        model_root / "control" / f"seed_{seed}/model.joblib",
        model_root / "control" / f"seed_{seed}/model.joblib.receipt.json"))
    completed_artifacts = completion.get("artifacts", {})
    for path in ref_files:
        relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        if path == completion_path:
            continue
        if completed_artifacts.get(relative) != _sha256(path):
            raise ValueError(f"pinned C0 completed-artifact hash mismatch: {relative}")
        if path.name.endswith(".receipt.json"):
            artifact_path = Path(str(path).removesuffix(".receipt.json"))
            sidecar = json.loads(path.read_text())
            if (sidecar.get("checkpoint_sha256") != _sha256(artifact_path)
                    or sidecar.get("contract", {}).get("source_sha256")
                    != completion["contract"].get("source_sha256")):
                raise ValueError(f"pinned C0 model receipt disagrees with completed run: {path}")
    for filename, expected in completion["contract"]["input_sha256"].items():
        path = Path(filename)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"pinned C0 completion input changed: {filename}")
    manifest_path = PROJECT_ROOT / "conf/base/research_benchmark.json"
    ref_files.append(manifest_path)
    inputs = {str(path.resolve().relative_to(PROJECT_ROOT)): _sha256(path) for path in ref_files}
    inputs.update(data["input_hashes"])
    inputs[str(canonical.resolve())] = _sha256(canonical)
    inputs[PLAN08_CONTROL.as_posix()] = _sha256(plan08_path)
    inputs[ANNUAL_A0_PREDICTIONS.as_posix()] = _sha256(annual_a0_path)
    inputs.update(market_hashes)
    snapshot = _source_snapshot(research_root, backend_hashes)
    protocol = frozen_contract(run_id, research_root)
    reference_calibration = joblib.load(calibration_path)
    protocol["reference_mixture_theta"] = {
        mode: [float(value) for value in reference_calibration["mixtures"][mode].theta]
        for mode in native.MODES}
    protocol.update({"status": "FROZEN", "research_root": str(research_root), "inputs": inputs,
                     "source_snapshot": snapshot, "backend_source_hashes": backend_hashes,
                     "canonical": {"path": str(canonical), "sha256": _sha256(canonical)},
                     "cohort": {"rows": len(split["validation"]), "ids_sha256": _digest(assessment_ids)},
                     "split": {name: {"rows": len(indices), "ids_sha256": _digest(
                         data["frame"].iloc[indices].golgg_match_id.astype(str).tolist())}
                               for name, indices in split.items()},
                     "evaluation": {"bootstrap": {"draws": 5000, "seed": 20261004,
                                                        "block": "month", "confidence": .983333}},
                     "reference": {"shared_model": str(shared_path.relative_to(PROJECT_ROOT)),
                                   "calibration": str(calibration_path.relative_to(PROJECT_ROOT)),
                                   "model_root": str(model_root.relative_to(PROJECT_ROOT)),
                                   "completed_receipt": str(completion_path.relative_to(PROJECT_ROOT)),
                                   "completed_receipt_sha256": _sha256(completion_path),
                                   "training_source_sha256": completion["contract"]["source_sha256"],
                                   "plan08_control": PLAN08_CONTROL.as_posix(),
                                   "annual_a0_predictions": ANNUAL_A0_PREDICTIONS.as_posix()},
                     "diagnostic_market": market_rows,
                     "smoke": {"train_rows": 1024, "cal_rows": 192, "assessment_rows": 192,
                               "best_of": [1, 3, 5]}})
    base = PROJECT_ROOT / "data/05_model_input/a0_capacity" / run_id
    base.mkdir(parents=True, exist_ok=False)
    (base / "source_snapshot").mkdir()
    for name in snapshot:
        if name.startswith("project:"):
            relative = name.removeprefix("project:")
            source = PROJECT_ROOT / relative
            destination = base / "source_snapshot" / relative
        else:
            relative = name.removeprefix("research:")
            source = research_root / relative
            destination = base / "source_snapshot" / "research" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    protocol["source_snapshot_archive"] = str((base / "source_snapshot").relative_to(PROJECT_ROOT))
    protocol_path = base / "protocol.json"
    protocol_path.write_text(json.dumps(_json_safe(protocol), indent=2, sort_keys=True, allow_nan=False) + "\n")
    protocol_hash = _sha256(protocol_path)
    recipe = {"run_id": run_id, "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
              "protocol_sha256": protocol_hash,
              "input_manifest_sha256": _digest(protocol["inputs"]),
              "source_manifest_sha256": _digest(protocol["source_snapshot"]),
              "environment": _runtime_environment(),
              "execution": {"device": "cpu", "torch_threads": 2, "deterministic_algorithms": True},
              "formal_command": f"python scripts/train_a0_capacity_experiment.py --protocol {protocol_path.relative_to(PROJECT_ROOT)} --distill",
              "smoke_command": f"python scripts/train_a0_capacity_experiment.py --protocol {protocol_path.relative_to(PROJECT_ROOT)} --smoke"}
    recipe_path = PROJECT_ROOT / "data/08_reporting/a0_capacity" / run_id / "reproduction_recipe.json"
    recipe_path.parent.mkdir(parents=True, exist_ok=False)
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    return protocol_path


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")

def _subset_inputs(data: dict[str, Any], indices: np.ndarray):
    return uncertainty._raw_subset(data["raw"], indices), {
        key: uncertainty._IndexedRows(value, indices) for key, value in data["history"].items()
    }


def _cell_name(width: int, epoch: int) -> str:
    return f"w{width}_e{epoch}"


def evaluate_endpoints(*, backend: dict[str, Any], shared: dict[str, Any],
                       models: dict[tuple[int, int], list[Any]], reference_calibration: dict[str, Any],
                       data: dict[str, Any], cal: np.ndarray, assess: np.ndarray,
                       smoke: bool = False) -> dict[str, Any]:
    """Fit only native neural CAL Beta parameters and assemble frozen C0 experts."""
    from src.models.a0_native_refit import _calculate_forecast
    from src.models.a0_target_experiment import _predict_mixture

    frame = data["frame"]
    wins_a, wins_b = data["wins_a"], data["wins_b"]
    bo = frame.best_of.to_numpy(dtype=np.int64)
    logits: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    rhos: dict[tuple[int, int], np.ndarray] = {}
    for cell, members in models.items():
        for part_name, ix in (("cal", cal), ("assessment", assess)):
            part = data
            z, rho, _ = _calculate_forecast(members, part, ix, backend)
            logits.setdefault(cell, {})[part_name] = z
            rhos[cell] = rho

    # Strict C0 reproduction gate runs before any challenger assembly.
    control_difference = None
    control_logit_difference = control_rho_difference = None
    if not smoke and (32, 12) in models:
        reference_dir = PROJECT_ROOT / "data/06_models/a0_data_focus" / REFERENCE_RUN
        old_models = [joblib.load(reference_dir / "control" / f"seed_{seed}/model.joblib") for seed in SEEDS]
        max_difference = 0.0
        saved_plan08 = pd.read_parquet(PROJECT_ROOT / PLAN08_CONTROL).set_index("golgg_match_id")
        for part_name, ix in (("cal", cal), ("assessment", assess)):
            old_z, old_rho, old_raw = _calculate_forecast(old_models, data, ix, backend)
            new_z = logits[(32, 12)][part_name]
            new_rho = rhos[(32, 12)]
            raw_subset, hist_subset = _subset_inputs(data, ix)
            other = native._other_logits(shared, raw_subset, hist_subset, bo[ix])
            old_full, _, _ = _predict_mixture(backend, old_z, other, reference_calibration["slopes"],
                                               raw_subset["gates"], bo[ix], reference_calibration)
            _, _, new_raw = _calculate_forecast(models[(32, 12)], data, ix, backend)
            new_full, _, _ = _predict_mixture(backend, new_z, other, reference_calibration["slopes"],
                                               raw_subset["gates"], bo[ix], reference_calibration)
            control_logit_difference = max(control_logit_difference or 0.,
                float(np.max(np.abs(old_z - new_z))))
            control_rho_difference = max(control_rho_difference or 0.,
                float(np.max(np.abs(old_rho - new_rho))))
            max_difference = max(max_difference,
                float(np.max(np.abs(old_raw - new_raw))),
                float(np.max(np.abs(old_full - new_full))))
            if part_name == "assessment":
                ids = frame.iloc[ix].golgg_match_id.astype(str)
                saved = saved_plan08.loc[ids]
                for mode_index, mode in enumerate(native.MODES):
                    max_difference = max(max_difference, float(np.max(np.abs(
                        old_full[:, mode_index] - saved[f"p_full_{mode}"].to_numpy(float)))))
                    max_difference = max(max_difference, float(np.max(np.abs(
                        old_raw[:, mode_index].mean(axis=1) - saved[f"p_raw_neural_{mode}"].to_numpy(float)))))
        control_difference = max_difference
        if max_difference > 2e-6:
            raise RuntimeError(f"native C0 probability parity failed: {max_difference:.9g}")
    output = pd.DataFrame({
        "golgg_match_id": frame.iloc[assess].golgg_match_id.astype(str).to_numpy(),
        "date": frame.iloc[assess].date.to_numpy(), "y": frame.iloc[assess].y.to_numpy(),
        "best_of": frame.iloc[assess].best_of.to_numpy(),
        "tournament_name": frame.iloc[assess].tournament_name.to_numpy(),
        "unknown_players": frame.iloc[assess].unknown_players.to_numpy(),
        "competition_tier": frame.iloc[assess].get("tier", pd.Series("unknown", index=frame.iloc[assess].index)).to_numpy(),
    })
    cal_output = pd.DataFrame({
        "golgg_match_id": frame.iloc[cal].golgg_match_id.astype(str).to_numpy(),
        "date": frame.iloc[cal].date.to_numpy(), "y": frame.iloc[cal].y.to_numpy(),
        "best_of": frame.iloc[cal].best_of.to_numpy(),
    })
    assess_frame = frame.iloc[assess]
    raw_cal, history_cal = _subset_inputs(data, cal)
    raw_assess, history_assess = _subset_inputs(data, assess)
    other_cal = native._other_logits(shared, raw_cal, history_cal, bo[cal])
    other_assess = native._other_logits(shared, raw_assess, history_assess, bo[assess])
    metrics, calibration_receipts = {}, []
    calibrations: dict[tuple[int, int], dict[str, Any]] = {}
    assembled: dict[tuple[int, int], np.ndarray] = {}
    for cell, members in models.items():
        cal_z = logits[cell]["cal"]
        parameters, records = [], []
        for mode in range(3):
            fitted = backend["beta_terminal"].fit_joint(
                cal_z[:, mode], wins_a[cal], wins_b[cal], "beta")
            parameters.append((fitted["scale"], fitted["rho"]))
            records.append(fitted)
        candidate_calibration = {
            "beta_parameters": np.asarray(parameters, dtype=float),
            "slopes": reference_calibration["slopes"],
            "mixtures": reference_calibration["mixtures"],
        }
        calibrations[cell] = candidate_calibration
        probs, _, _ = _predict_mixture(
            backend, logits[cell]["assessment"], other_assess, candidate_calibration["slopes"],
            raw_assess["gates"], bo[assess], candidate_calibration)
        assembled[cell] = probs
        cal_probs, _, _ = _predict_mixture(
            backend, logits[cell]["cal"], other_cal, candidate_calibration["slopes"],
            raw_cal["gates"], bo[cal], candidate_calibration)
        nll = native._neural_scoreline_nll(
            backend, logits[cell]["assessment"], rhos[cell], candidate_calibration,
            raw_assess["gates"], bo[assess], wins_a[assess], wins_b[assess])
        cal_nll = native._neural_scoreline_nll(
            backend, logits[cell]["cal"], rhos[cell], candidate_calibration,
            raw_cal["gates"], bo[cal], wins_a[cal], wins_b[cal])
        name = _cell_name(*cell)
        for mode, mode_name in enumerate(native.MODES):
            output[f"{name}__{mode_name}"] = probs[:, mode]
            output[f"{name}__nll__{mode_name}"] = nll[f"nll_neural_{mode_name}"]
            cal_output[f"{name}__{mode_name}"] = cal_probs[:, mode]
            cal_output[f"{name}__nll__{mode_name}"] = cal_nll[f"nll_neural_{mode_name}"]
        metrics[name] = {
            "cell": name, "width": cell[0], "epoch": cell[1],
            "calibration": [{key: float(record[key]) for key in ("scale", "rho")} for record in records],
            "calibration_rows": len(cal), "assessment_rows": len(assess),
            "assessment_series": {
                mode: {"log_loss": float(np.mean(-(assess_frame.y.to_numpy() * np.log(probs[:, i])
                    + (1-assess_frame.y.to_numpy()) * np.log1p(-probs[:, i])))),
                       "brier": float(np.mean(np.square(probs[:, i] - assess_frame.y.to_numpy())))}
                for i, mode in enumerate(native.MODES)
            },
            "scoreline_nll": {mode: float(np.mean(nll[f"nll_neural_{mode}"])) for mode in native.MODES},
            "cal_scoreline_nll": {mode: float(np.mean(cal_nll[f"nll_neural_{mode}"])) for mode in native.MODES},
        }
        calibration_receipts.append({"cell": name, "rows": int(len(cal)),
                                     "fit_records": _json_safe(records),
                                     "mixture_reused": True, "slopes_reused": True})
    ids = frame.iloc[assess].golgg_match_id.astype(str).to_numpy()
    saved_plan08 = pd.read_parquet(PROJECT_ROOT / PLAN08_CONTROL)
    saved_plan08["golgg_match_id"] = saved_plan08.golgg_match_id.astype(str)
    saved_plan08 = saved_plan08.set_index("golgg_match_id").loc[ids]
    annual_a0 = pd.read_parquet(PROJECT_ROOT / ANNUAL_A0_PREDICTIONS)
    annual_a0["golgg_match_id"] = annual_a0.golgg_match_id.astype(str)
    annual_a0 = annual_a0.drop_duplicates("golgg_match_id").set_index("golgg_match_id").loc[ids]
    for mode in native.MODES:
        output[f"p_c0__{mode}"] = saved_plan08[f"p_full_{mode}"].to_numpy(float)
    output["p_c0__full"] = saved_plan08["p_full_full"].to_numpy(float)
    output["p_a0__full"] = annual_a0["p_a0"].to_numpy(float)
    return {"predictions": output, "cal_predictions": cal_output, "metrics": metrics,
            "control_max_abs_probability_difference": control_difference,
            "control_logit_max_abs_difference": control_logit_difference,
            "control_rho_max_abs_difference": control_rho_difference,
            "calibration_receipts": calibration_receipts, "calibrations": calibrations,
            "assembled": assembled, "logits": logits, "rhos": rhos}


def _runtime_invariants(backend: dict[str, Any], shared: dict[str, Any],
                        models: dict[tuple[int, int], list[Any]], calibrations: dict,
                        data: dict[str, Any], assess: np.ndarray, assembled: dict) -> dict[str, Any]:
    from src.models.a0_native_refit import _calculate_forecast
    from src.models.a0_target_experiment import _predict_mixture

    bo = data["frame"].best_of.to_numpy(dtype=np.int64)[assess]
    raw, history = _subset_inputs(data, assess)
    raw_swapped = {key: np.asarray(value)[:, ::-1].copy() for key, value in raw.items()}
    history_swapped = {key: np.asarray(value)[:, ::-1].copy() for key, value in history.items()}
    local_frame = data["frame"].iloc[assess].reset_index(drop=True)
    local_data = {"frame": local_frame, "raw": raw_swapped, "history": history_swapped}
    other_swapped = native._other_logits(shared, raw_swapped, history_swapped, bo)
    symmetry = 0.
    symmetry_samples = 0
    for cell, members in models.items():
        reverse_z, _, _ = _calculate_forecast(members, local_data, np.arange(len(assess)), backend)
        reverse, _, _ = _predict_mixture(backend, reverse_z, other_swapped,
            calibrations[cell]["slopes"], raw_swapped["gates"], bo, calibrations[cell])
        difference = np.abs(assembled[cell] + reverse - 1)
        symmetry = max(symmetry, float(np.max(difference)))
        symmetry_samples += int(difference.size)
    neutral_error = 0.
    cold_samples = 0
    cold_raw = {key: np.zeros_like(np.asarray(value)[:1]) for key, value in data["raw"].items()}
    cold_history = {key: np.zeros_like(np.asarray(value)[:1]) for key, value in data["history"].items()}
    for cell, members in models.items():
        for best_of in (1, 3, 5):
            cold_frame = data["frame"].iloc[:1].copy()
            cold_frame["best_of"] = best_of
            cold_data = {"frame": cold_frame, "raw": cold_raw, "history": cold_history}
            cold_z, _, _ = _calculate_forecast(members, cold_data, np.asarray([0]), backend)
            cold_bo = np.asarray([best_of], dtype=np.int64)
            cold_other = native._other_logits(shared, cold_raw, cold_history, cold_bo)
            cold_probabilities, _, _ = _predict_mixture(
                backend, cold_z, cold_other, calibrations[cell]["slopes"],
                cold_raw["gates"], cold_bo, calibrations[cell])
            if not np.isfinite(cold_probabilities).all():
                raise ValueError("empty-history assembly produced non-finite probabilities")
            neutral_error = max(neutral_error, float(np.max(np.abs(cold_probabilities - .5))))
            cold_samples += int(cold_probabilities.size)
    if neutral_error > 1e-4:
        raise ValueError(f"empty-history assembly is not neutral: max error {neutral_error:.9g}")
    return {"side_symmetry_max_abs_error": symmetry, "side_symmetry_samples": symmetry_samples,
            "empty_history_probability": .5 + neutral_error,
            "empty_history_max_abs_error": neutral_error, "empty_history_samples": cold_samples,
            "cold_best_of_cases": [1, 3, 5], "cold_modes_per_case": len(native.MODES)}



def run_protocol(protocol_path: str | Path, *, smoke: bool = False, distill: bool = False,
                 research_root: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "a0_capacity_v1" or protocol.get("status") != "FROZEN":
        raise ValueError("protocol is not a frozen a0_capacity_v1 protocol")
    verify_protocol_hashes(protocol, PROJECT_ROOT)
    frozen_research_root = Path(protocol["research_root"]).resolve()
    if research_root is not None and Path(research_root).expanduser().resolve() != frozen_research_root:
        raise ValueError("execution research root differs from the frozen protocol")
    research_root = frozen_research_root
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError("frozen native capacity protocol requires deterministic CPU execution")
    backend, backend_hashes = uncertainty._activate_archive_backend(research_root)
    if backend_hashes != protocol["backend_source_hashes"]:
        raise ValueError("activated archive/backend hashes differ from frozen protocol")
    data = uncertainty._load_training_data(research_root)
    for absolute, digest in data["input_hashes"].items():
        if protocol["inputs"].get(absolute) != digest:
            raise ValueError(f"feature-bank transitive input is absent or changed: {absolute}")
    frame = data["frame"]
    identities = [native.classify_competition(name, pd.Timestamp(day).date())
                  for name, day in zip(frame.tournament_name, frame.date)]
    frame["tier"] = [identity.tier.value for identity in identities]
    frame["family"] = [identity.family for identity in identities]
    split = _partition(frame, smoke=smoke, history=data["history"], raw=data["raw"])
    for name, indices in split.items():
        expected = protocol["split"][name]
        ids = frame.iloc[indices].golgg_match_id.astype(str).tolist()
        if len(indices) != expected["rows"] or _digest(ids) != expected["ids_sha256"]:
            raise ValueError(f"frozen {name} partition membership changed")
    model = __import__("src.models.a0_capacity_experiment", fromlist=["ParametricNonlinearHistoryModel"])
    reference_root = PROJECT_ROOT / protocol["reference"]["model_root"]
    shared = joblib.load(reference_root / "shared/model.joblib")
    reference_calibration = joblib.load(PROJECT_ROOT / protocol["reference"]["calibration"])
    run_id = _execution_id(protocol["run_id"], smoke)
    model_root = PROJECT_ROOT / "data/06_models/a0_capacity" / run_id
    output_root = PROJECT_ROOT / "data/07_model_output/a0_capacity" / run_id
    report_root = PROJECT_ROOT / "data/08_reporting/a0_capacity" / run_id
    frozen_recipe = report_root / "reproduction_recipe.json"
    allowed_frozen_report = False
    if report_root.exists():
        existing = {path.name for path in report_root.iterdir()}
        if existing == {"reproduction_recipe.json"} and frozen_recipe.is_file():
            recipe = json.loads(frozen_recipe.read_text())
            allowed_frozen_report = Path(recipe.get("protocol", "")).resolve() == protocol_path
        if not allowed_frozen_report:
            raise FileExistsError(f"refusing to reuse stale capacity report state: {run_id}")
    if model_root.exists() or output_root.exists():
        raise FileExistsError(f"refusing to reuse stale capacity run state: {run_id}")
    model_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True, exist_ok=True)
    receipt = {"run_id": run_id, "status": "RUNNING", "smoke": smoke,
               "device": "cpu", "threads": 2, "cuda_available": False,
               "protocol_sha256": _sha256(protocol_path),
               "input_manifest_sha256": _digest(protocol["inputs"]),
               "source_manifest_sha256": _digest(protocol["source_snapshot"]),
               "environment": _runtime_environment(),
               "backend_source_hashes": backend_hashes, "trajectories": [], "failures": []}
    _save_json(report_root / "receipt.json", receipt)
    try:
        train = split["train"][:1024] if smoke else split["train"]
        cal = split["cal"][:192] if smoke else split["cal"]
        assess = split["validation"][:192] if smoke else split["validation"]
        if smoke:
            assess = _representative_smoke_rows(frame, split["validation"], 192)
        bo_index = (frame.best_of.to_numpy(dtype=np.int64) - 1) // 2
        seeds, checkpoints = (SEEDS[:1], (1,)) if smoke else (SEEDS, (12, 24))
        endpoints: dict[tuple[int, int], list[Any]] = {}
        initialization_receipts = {}
        def train_width(width: int) -> None:
            trajectories = []
            for seed in seeds:
                instance = model.ParametricNonlinearHistoryModel(shared["baseline"], shared["normalizer"],
                    width=width, seed=seed, epochs=1 if smoke else 24, coefficient=.1)
                initial_network_hash = _model_hash(instance.network.state_dict())
                initial_rng = {"numpy_state_sha256": _digest(np.random.get_state()),
                               "torch_state_sha256": hashlib.sha256(
                                   torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                               "eta": float(instance.eta.detach().cpu())}
                initialization_receipts[(width, seed)] = {
                    "initial_network_sha256": initial_network_hash, **initial_rng}
                snapshots = instance.fit_trajectory(
                    data["raw"], data["history"], None, train, bo_index, data["wins_a"], data["wins_b"],
                    data["targets"], data["target_mask"],
                    checkpoint_dir=model_root / f"w{width}_seed{seed}",
                    checkpoint_epochs=checkpoints, device="cpu",
                    progress=lambda epoch, loss, aux, record: print(
                        f"capacity width={width} seed={seed} epoch={epoch} "
                        f"train_loss={loss:.8g} auxiliary_loss={aux:.8g}", flush=True))
                trajectories.append(snapshots)
                for epoch, snapshot in snapshots.items():
                    path = model_root / f"w{width}_seed{seed}" / f"model_epoch{epoch}.joblib"
                    joblib.dump(snapshot, path)
                receipt["trajectories"].append({"width": width, "seed": seed,
                    "checkpoints": list(checkpoints), "status": "complete",
                    "initialization": initialization_receipts[(width, seed)]})
            for epoch in checkpoints:
                members = [trajectory[epoch] for trajectory in trajectories]
                endpoints[(width, epoch)] = members * 3 if smoke else members
            _save_json(report_root / "receipt.json", receipt)

        if smoke:
            train_width(32)
        else:
            # The control is the first continuous trajectory and must pass before challengers train.
            train_width(32)
            gate_payload = evaluate_endpoints(backend=backend, shared=shared,
                models={(32, 12): endpoints[(32, 12)]}, reference_calibration=reference_calibration,
                data=data, cal=cal, assess=assess, smoke=False)
            beta_difference = float(np.max(np.abs(
                gate_payload["calibrations"][(32, 12)]["beta_parameters"]
                - reference_calibration["beta_parameters"])))
            postcal_difference = max(
                float(np.max(np.abs(gate_payload["predictions"][f"w32_e12__{mode}"]
                                    - gate_payload["predictions"][f"p_c0__{mode}"])))
                for mode in native.MODES)
            gate_probability_difference = max(
                gate_payload["control_max_abs_probability_difference"], postcal_difference)
            if gate_probability_difference > 2e-6 or beta_difference > 2e-6:
                raise RuntimeError(
                    f"refit-CAL C0 control parity failed: probability={gate_probability_difference:.9g}, "
                    f"beta={beta_difference:.9g}")
            receipt["control_gate"] = {"status": "PASS",
                "max_probability_difference": gate_probability_difference,
                "max_logit_difference": gate_payload["control_logit_max_abs_difference"],
                "max_rho_difference": gate_payload["control_rho_max_abs_difference"],
                "beta_parameters_difference": beta_difference,
                "comparisons": ["CAL/ASSESS raw terminal probabilities",
                                "CAL/ASSESS all-mode full-precision assembled probabilities",
                                "CAL-refit beta parameters against saved C0"]}
            _save_json(report_root / "receipt.json", receipt)
            for width in (16, 64):
                train_width(width)

        payload = evaluate_endpoints(backend=backend, shared=shared, models=endpoints,
            reference_calibration=reference_calibration, data=data, cal=cal, assess=assess, smoke=smoke)
        endpoint_arrays = {}
        endpoint_provenance = {}
        for cell, calibration in payload["calibrations"].items():
            cell_name = _cell_name(*cell)
            cell_dir = model_root / cell_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(calibration, cell_dir / "calibration.joblib")
            endpoint_arrays[f"{cell_name}_cal_logits"] = payload["logits"][cell]["cal"]
            endpoint_arrays[f"{cell_name}_assessment_logits"] = payload["logits"][cell]["assessment"]
            endpoint_arrays[f"{cell_name}_rho"] = payload["rhos"][cell]
            endpoint_provenance[cell_name] = {
                "calibration_sha256": _sha256(cell_dir / "calibration.joblib"),
                "member_state_sha256": [_model_hash(member.network.state_dict())
                                        for member in endpoints[cell]],
                "cal_rows": len(cal), "assessment_rows": len(assess)}
        np.savez_compressed(output_root / "endpoint_arrays.npz", **endpoint_arrays)
        _save_json(report_root / "endpoint_provenance.json", endpoint_provenance)
        payload["runtime_proof"] = _runtime_invariants(backend, shared, endpoints,
            payload["calibrations"], data, assess, payload["assembled"])
        payload["runtime_proof"]["control_parity_max_abs_probability_difference"] = payload[
            "control_max_abs_probability_difference"]
        payload["runtime_proof"]["max_control_probability_difference"] = payload[
            "control_max_abs_probability_difference"]
        predictions = payload["predictions"]
        predictions.to_parquet(output_root / "predictions.parquet", index=False)
        payload["cal_predictions"].to_parquet(output_root / "calibration_predictions.parquet", index=False)
        _save_json(report_root / "metrics.json", payload["metrics"])
        _save_json(report_root / "calibration_receipts.json", payload["calibration_receipts"])
        receipt.update({"status": "PREDICTIONS_COMPLETE",
            "control_max_abs_probability_difference": payload["control_max_abs_probability_difference"],
            "calibration_receipts": payload["calibration_receipts"]})
        _save_json(report_root / "receipt.json", receipt)
        if smoke:
            _save_json(report_root / "decision.json", {"status": "SMOKE_DIAGNOSTIC_ONLY",
                "calibration_cells": list(payload["metrics"]), "assessment_rows": len(predictions)})
        else:
            from src.analysis.a0_capacity_experiment import evaluate_experiment
            evaluated = evaluate_experiment(predictions, protocol, report_root,
                                            runtime_proof=payload["runtime_proof"])
            if distill:
                _run_distillation(model, backend, shared, endpoints, data, split, payload,
                                  reference_calibration, evaluated, protocol,
                                  model_root, output_root, report_root, initialization_receipts)
        receipt["status"] = "COMPLETE"
        artifact_hashes = {}
        for root in (model_root, output_root, report_root):
            for path in root.rglob("*"):
                if path.is_file() and path.name != "receipt.json":
                    artifact_hashes[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)
        receipt["artifact_sha256"] = artifact_hashes
        _save_json(report_root / "receipt.json", receipt)
        return {"status": "COMPLETE", "run_id": run_id, "predictions": str(output_root / "predictions.parquet")}
    except Exception as error:
        receipt["status"] = "FAILED"
        receipt["failures"].append({"type": type(error).__name__, "message": str(error)})
        _save_json(report_root / "receipt.json", receipt)
        raise


def _representative_smoke_rows(frame: pd.DataFrame, indices: np.ndarray, target: int) -> np.ndarray:
    selected: list[int] = []
    for best_of in (1, 3, 5):
        matches = indices[frame.iloc[indices].best_of.to_numpy(dtype=int) == best_of]
        if not len(matches):
            raise ValueError(f"smoke assessment lacks representative BO{best_of}")
        selected.append(int(matches[0]))
    selected_set = set(selected)
    selected.extend(int(index) for index in indices if int(index) not in selected_set and len(selected) < target)
    return np.asarray(selected, dtype=np.int64)


def _run_distillation(model: Any, backend: dict[str, Any], shared: dict[str, Any],
                      endpoints: dict[tuple[int, int], list[Any]], data: dict[str, Any],
                      split: dict[str, np.ndarray], payload: dict[str, Any],
                      reference_calibration: dict[str, Any], evaluated: dict[str, Any],
                      protocol: dict[str, Any], model_root: Path, output_root: Path,
                      report_root: Path, initialization_receipts: dict) -> None:
    if (32, 12) not in endpoints or (16, 12) not in endpoints:
        raise ValueError("fixed W32/E12 teacher and W16/E12 direct controls are required")
    eligibility = evaluated.get("decision", {}).get("distillation", {})
    if not eligibility:
        raise ValueError("frozen evaluator did not return the distillation eligibility gate")
    teacher_advantage = bool(
        eligibility.get("trigger", {}).get("verified_width32_or_width64_teacher_advantage", False))
    if not eligibility.get("eligible", False):
        _save_json(report_root / "distillation.json", {
            "status": "SKIPPED_BY_FROZEN_EVALUATION_GATE",
            "eligibility": eligibility})
        return

    frame = data["frame"]

    train = split["train"]
    bo = frame.best_of.to_numpy(dtype=np.int64)
    bo_index = (bo - 1) // 2
    teacher_members = endpoints[(32, 12)]
    teacher_probs = np.zeros((len(train), 6), dtype=np.float64)
    teacher_hashes = [_model_hash(member.network.state_dict()) for member in teacher_members]
    for member in teacher_members:
        z = np.asarray(member.map_logits(data["raw"], data["history"], None, train,
                                         bo_index), dtype=np.float64)
        rho = float(.5 * torch.sigmoid(member.eta.detach()).cpu())
        teacher_probs += np.exp(backend["beta_terminal"].terminal_logs(z, bo[train], rho)) / len(teacher_members)
    if (not np.isfinite(teacher_probs).all()
            or not np.allclose(teacher_probs.sum(axis=1), 1., atol=1e-10, rtol=0)):
        raise ValueError("TRAIN-only teacher terminal distribution is invalid")

    direct_members = endpoints[(16, 12)]
    kd_members, initial_hashes, final_direct_hashes, final_kd_hashes = [], [], [], []
    direct_rng_hashes, kd_rng_hashes = [], []
    for seed_index, seed in enumerate(SEEDS):
        direct_endpoint = direct_members[seed_index]
        direct_initial_hash = initialization_receipts[(16, seed)]["initial_network_sha256"]
        direct_state = torch.load(
            model_root / f"w16_seed{seed}/checkpoint_epoch12.pt",
            map_location="cpu", weights_only=False)
        kd = model.ParametricNonlinearHistoryModel(shared["baseline"], shared["normalizer"],
            width=16, seed=seed, epochs=12, coefficient=.1)
        kd_initial_hash = _model_hash(kd.network.state_dict())
        if kd_initial_hash != direct_initial_hash:
            raise ValueError("KD student initialization differs from the saved direct-label trajectory")
        kd_snapshots = kd.fit_trajectory(data["raw"], data["history"], None, train, bo_index,
            data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
            checkpoint_dir=model_root / f"distill_kd_seed{seed}", checkpoint_epochs=(12,), device="cpu",
            teacher_probabilities=teacher_probs, distillation_weight=.5,
            progress=lambda epoch, loss, aux, record: print(
                f"capacity distill seed={seed} epoch={epoch} "
                f"train_loss={loss:.8g} auxiliary_loss={aux:.8g}", flush=True))
        kd_endpoint = kd_snapshots[12]
        kd_state = torch.load(model_root / f"distill_kd_seed{seed}/checkpoint_epoch12.pt",
                              map_location="cpu", weights_only=False)
        if _json_safe(direct_state["numpy_rng_state"]) != _json_safe(kd_state["numpy_rng_state"]):
            raise ValueError("matched direct/KD data-mask RNG states diverged")
        initial_hashes.append(direct_initial_hash)
        final_direct_hashes.append(_model_hash(direct_endpoint.network.state_dict()))
        final_kd_hashes.append(_model_hash(kd_endpoint.network.state_dict()))
        direct_rng_hashes.append(_digest(direct_state["numpy_rng_state"]))
        kd_rng_hashes.append(_digest(kd_state["numpy_rng_state"]))
        kd_members.append(kd_endpoint)

    assess = split["validation"]
    direct_eval = evaluate_endpoints(backend=backend, shared=shared,
        models={(16, 12): direct_members}, reference_calibration=reference_calibration,
        data=data, cal=split["cal"], assess=assess)
    kd_eval = evaluate_endpoints(backend=backend, shared=shared,
        models={(16, 12): kd_members}, reference_calibration=reference_calibration,
        data=data, cal=split["cal"], assess=assess)
    direct_probability = direct_eval["predictions"]["w16_e12__full"].to_numpy(float)
    kd_probability = kd_eval["predictions"]["w16_e12__full"].to_numpy(float)
    direct_runtime = _measure_runtime(direct_members, data, assess)
    kd_runtime = _measure_runtime(kd_members, data, assess)
    teacher_runtime = _measure_runtime(teacher_members, data, assess)
    runtime_proof = {
        "direct_latency_ms": direct_runtime["latency_ms"], "kd_latency_ms": kd_runtime["latency_ms"],
        "direct_peak_memory_bytes": direct_runtime["peak_memory_bytes"],
        "kd_peak_memory_bytes": kd_runtime["peak_memory_bytes"],
        "teacher_latency_ms": teacher_runtime["latency_ms"],
        "teacher_peak_memory_bytes": teacher_runtime["peak_memory_bytes"],
        **payload["runtime_proof"],
    }
    full_predictions = pd.read_parquet(output_root / "predictions.parquet")
    full_predictions["w16_e12_kd"] = kd_probability
    full_predictions["w16_e12_direct"] = direct_probability
    full_predictions.to_parquet(output_root / "predictions.parquet", index=False)
    paired = full_predictions[["golgg_match_id", "date", "y", "best_of",
                               "tournament_name", "unknown_players", "competition_tier",
                               "w16_e12_kd", "w16_e12_direct"]]
    paired.to_parquet(output_root / "distillation_predictions.parquet", index=False)
    distill_protocol = dict(protocol)
    from src.analysis.a0_capacity_experiment import evaluate_experiment
    final_evaluation = evaluate_experiment(full_predictions, distill_protocol, report_root,
                                           runtime_proof=runtime_proof)
    _save_json(report_root / "distillation.json", {
        "status": "COMPLETE", "eligibility": eligibility,
        "teacher": "fixed_w32_e12_equal_seed_uncalibrated_terminal_mixture",
        "teacher_partition": "TRAIN_only", "teacher_rows": len(train),
        "teacher_initial_member_hashes": teacher_hashes, "lambda": .5, "temperature": 1.,
        "seeds": list(SEEDS), "student_initial_hashes": initial_hashes,
        "direct_final_hashes": final_direct_hashes, "kd_final_hashes": final_kd_hashes,
        "direct_numpy_rng_hashes": direct_rng_hashes, "kd_numpy_rng_hashes": kd_rng_hashes,
        "runtime_proof": runtime_proof,
        "paired_prediction_rows": len(paired),
        "evaluation": final_evaluation["decision"].get("distillation"),
    })


def _model_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _measure_runtime(members: list[Any], data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    import time
    from torch.profiler import ProfilerActivity, profile
    raw, history = _subset_inputs(data, indices)
    local_frame = data["frame"].iloc[indices].reset_index(drop=True)
    best_of_index = (local_frame.best_of.to_numpy(dtype=np.int64) - 1) // 2
    local_ix = np.arange(len(indices), dtype=np.int64)

    def infer():
        return [member.map_logits(raw, history, None, local_ix, best_of_index) for member in members]

    infer()
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        infer()
        durations.append((time.perf_counter() - start) * 1000)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        infer()
    memory_events = [event for event in prof.events()
                     if event.name == "[memory]" and event.self_cpu_memory_usage != 0]
    current = peak = 0
    for event in memory_events:
        current += int(event.self_cpu_memory_usage)
        peak = max(peak, current)
    if peak <= 0:
        raise RuntimeError("CPU inference profiler did not report positive peak memory")
    return {"latency_ms": float(np.median(durations)), "peak_memory_bytes": int(peak),
            "memory_measurement": "chronological_profiler_memory_events_self_cpu_usage"}


