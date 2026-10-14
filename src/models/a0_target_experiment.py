"""Paired native A0 terminal-versus-series-winner objective experiment.

The frozen raw-rating/full-MLP experts and annual Beta calibration/mixing recipe
are shared unchanged by both arms. Only the neural TRAIN main loss differs.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "ev_all_approaches_20260927_220902"
PROTOCOL_SHA256 = "7bca9fc39bad77d3ecb84f4cd607a076861630dd14cb16467abb30f32db2b0cd"
OBJECTIVES = ("terminal", "winner")
YEARS = (2024, 2025, 2026)
SEEDS = (20260910, 20260911, 20260912)
MODES = ("full", "no_w20", "no_organization")
MODE_FLAGS = ((True, True), (False, True), (False, False))
BANK_RELATIVE = Path("a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def resolve_research_root(research_root: str | Path | None = None) -> Path:
    """Resolve a research root using CLI, environment, then repository pointer."""
    from src.models.a0_uncertainty import _resolve_research_root

    if research_root is None and os.environ.get("ENSEMBLE_RESEARCH_ROOT"):
        research_root = os.environ["ENSEMBLE_RESEARCH_ROOT"]
    return _resolve_research_root(research_root)


def _load_protocol(protocol_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(protocol_path).expanduser().resolve(strict=True)
    if _sha256(path) != PROTOCOL_SHA256:
        raise ValueError("native target experiment requires the frozen protocol SHA-256")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    native = protocol.get("native_target_experiment", {})
    if (
        protocol.get("run_id") != RUN_ID
        or protocol.get("status") != "PREREGISTERED_DEVELOPMENT_EXPERIMENT"
        or native.get("objectives") != list(OBJECTIVES)
        or native.get("years") != list(YEARS)
        or native.get("seeds") != list(SEEDS)
        or native.get("epochs") != 12
    ):
        raise ValueError("frozen native-target protocol does not match the required experiment")
    return path, protocol


def _activate(research_root: Path):
    from src.models.a0_uncertainty import _activate_archive_backend

    return _activate_archive_backend(research_root)


def _load_data(research_root: Path) -> dict[str, Any]:
    from src.models.a0_uncertainty import _load_training_data

    return _load_training_data(research_root)


def _verify_input_hashes(input_hashes: dict[str, str]) -> None:
    for filename, expected in input_hashes.items():
        if _sha256(Path(filename)) != expected:
            raise ValueError(f"frozen training input changed: {filename}")


def _project_source_hashes(backend_hashes: dict[str, str]) -> dict[str, str]:
    sources = (
        PROJECT_ROOT / "src/models/a0_target_objectives.py",
        PROJECT_ROOT / "src/models/a0_target_experiment.py",
        PROJECT_ROOT / "src/models/a0_accelerator.py",
        PROJECT_ROOT / "src/models/a0_uncertainty.py",
        PROJECT_ROOT / "scripts/train_a0_uncertainty.py",
    )
    project_hashes = {path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path) for path in sources}
    return {**project_hashes, **backend_hashes}


def _versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib")
        },
    }


def _source_contract(research_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    _, backend_hashes = _activate(research_root)
    return _project_source_hashes(backend_hashes), backend_hashes


def _member_ids(frame: pd.DataFrame, indices: np.ndarray) -> list[str]:
    return frame.iloc[indices].golgg_match_id.astype(str).tolist()


def _contract_digest(contract: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(contract), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verified_sidecar(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint and its receipt must both exist: {path}")
    receipt = json.loads(sidecar.read_text())
    if receipt.get("contract") != _json_safe(contract):
        raise ValueError(f"checkpoint contract differs from this run: {path}")
    if receipt.get("checkpoint_sha256") != _sha256(path):
        raise ValueError(f"checkpoint hash differs from its receipt: {path}")
    return receipt


def _save_checkpoint(path: Path, model: Any, contract: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix + ".receipt.json").exists():
        raise FileExistsError(f"refusing to overwrite checkpoint artifact: {path}")
    joblib.dump(model, path)
    receipt = {
        "contract": _json_safe(contract),
        "contract_sha256": _contract_digest(contract),
        "checkpoint_sha256": _sha256(path),
        "details": _json_safe(details),
    }
    _write_json(path.with_suffix(path.suffix + ".receipt.json"), receipt)
    return receipt


def _restore_neural(path: Path, contract: dict[str, Any], model_type: type, seed: int, epochs: int):
    receipt = _verified_sidecar(path, contract)
    model = joblib.load(path)
    if type(model) is not model_type or model.seed != seed or model.epochs != epochs:
        raise ValueError(f"saved neural checkpoint class, seed, or epoch contract differs: {path}")
    if getattr(model, "main_objective", None) != contract["objective"]:
        raise ValueError(f"saved neural checkpoint objective differs: {path}")
    if len(model.train_losses) != epochs or len(model.auxiliary_losses) != epochs:
        raise ValueError(f"saved neural checkpoint has incomplete epoch histories: {path}")
    if not np.isfinite(model.train_losses + model.auxiliary_losses).all():
        raise ValueError(f"saved neural checkpoint contains nonfinite losses: {path}")
    return model, receipt




def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.network.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(model.eta.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _training_frame(data: dict[str, Any], year: int):
    from src.models.a0_uncertainty import annual_membership

    frame = data["frame"]
    train, cal, test = annual_membership(frame, year)
    if not len(train) or not len(cal) or not len(test):
        raise ValueError(f"empty annual TRAIN/CAL/TEST partition for {year}")
    return frame, train, cal, test


def _raw_subset(raw: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(values)[indices] for key, values in raw.items()}


def _indexed_history(history: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    from src.models.a0_uncertainty import _IndexedRows

    return {key: _IndexedRows(values, indices) for key, values in history.items()}


def _map_logits(models: list[Any], raw: dict[str, Any], history: dict[str, Any], indices: np.ndarray, bo_index: np.ndarray) -> np.ndarray:
    values = np.empty((len(indices), len(MODES), len(models)), dtype=float)
    for member, model in enumerate(models):
        for mode_index, (w20, org) in enumerate(MODE_FLAGS):
            values[:, mode_index, member] = model.map_logits(
                raw, history, None, indices, bo_index, w20, org,
            )
    if not np.isfinite(values).all():
        raise ValueError("native neural inference produced nonfinite map logits")
    return values


def _frozen_expert_source_hashes(research_root: Path) -> dict[str, str]:
    run = research_root / "a0-phase-walkforward-20260914"
    paths = (
        run / "code/retrain_other_experts.py",
        run / "CAUSAL_REBUILD_PLAN.md",
        research_root / "unified-ratings-20260911/code/scripts/benchmark_unified_ratings.py",
        research_root / "nonlinear-history-20260912/code/src/models/unified_ratings.py",
        research_root / "nonlinear-history-20260912/code/src/models/roster_optional.py",
        research_root / "nonlinear-history-20260912/code/src/models/temporal_training.py",
        research_root / "nonlinear-history-20260912/code/src/models/temporal_architectures.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("frozen expert source contract is incomplete: " + ", ".join(missing))
    return {str(path): _sha256(path) for path in paths}


def _load_frozen_expert(
    research_root: Path, backend: dict[str, Any], frame: pd.DataFrame,
    year: int, name: str, seed: int, kind: str, full: bool,
    membership: dict[str, list[str]], selected: np.ndarray,
    source_hashes: dict[str, str], bank_receipt_sha256: str,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Load one already-trained archive expert and its audited raw logits."""
    folder = research_root / "a0-phase-walkforward-20260914/data/06_models/causal_a0" / str(year) / "other"
    checkpoint = folder / f"{name}.joblib"
    forecast = folder / f"{name}.npz"
    audit_path = folder / f"{name}.json"
    if not all(path.is_file() for path in (checkpoint, forecast, audit_path)):
        raise FileNotFoundError(f"required frozen A0 expert artifacts are missing for {year}/{name}")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "COMPLETE"
        or audit.get("name") != name
        or audit.get("bank_receipt_sha256") != bank_receipt_sha256
        or audit.get("code_sha256") != source_hashes
        or audit.get("membership_ids") != membership
        or audit.get("checkpoint_sha256") != _sha256(checkpoint)
        or audit.get("predictions_sha256") != _sha256(forecast)
        or audit.get("kind") != kind or audit.get("full") is not full
        or audit.get("seed") != seed
    ):
        raise ValueError(f"frozen expert audit/identity/hash contract mismatch: {audit_path}")
    warm_checkpoint = None
    if kind == "mlp" and full:
        warm_checkpoint = folder / f"{year}_ratings_mlp_{seed}.joblib"
        if (
            not warm_checkpoint.is_file()
            or audit.get("initial_sha256") != _sha256(warm_checkpoint)
        ):
            raise ValueError(f"frozen full-MLP warmup provenance differs from its audit: {warm_checkpoint}")
    elif audit.get("initial_sha256") is not None:
        raise ValueError(f"unexpected warmup checkpoint in frozen expert audit: {audit_path}")
    model_type = backend["unified_ratings"].UnifiedModel
    model = joblib.load(checkpoint)
    if (
        type(model) is not model_type or model.seed != seed
        or model.kind != kind or model.full is not full
        or (
            kind == "mlp" and full
            and (model.epochs != 12 or audit.get("epochs") != 12)
        )
        or not isinstance(model.slopes, dict) or set(model.slopes) != set(MODES)
    ):
        raise ValueError(f"frozen expert checkpoint has a different model contract: {checkpoint}")
    if audit.get("slopes", {}) != model.slopes:
        raise ValueError(f"frozen expert checkpoint calibration differs from its saved audit: {checkpoint}")
    with np.load(forecast, allow_pickle=False) as archive:
        columns = []
        for mode in MODES:
            key = f"z_{mode}"
            if key not in archive or archive[key].shape != (len(frame),):
                raise ValueError(f"frozen expert raw-logit output is not aligned: {forecast}:{key}")
            columns.append(np.asarray(archive[key][selected], dtype=float))
    logits = np.column_stack(columns)
    if not np.isfinite(logits).all():
        raise ValueError(f"frozen expert raw logits are nonfinite: {forecast}")
    files = {
        checkpoint.relative_to(research_root).as_posix(): _sha256(checkpoint),
        forecast.relative_to(research_root).as_posix(): _sha256(forecast),
        audit_path.relative_to(research_root).as_posix(): _sha256(audit_path),
    }
    if warm_checkpoint is not None:
        files[warm_checkpoint.relative_to(research_root).as_posix()] = _sha256(warm_checkpoint)
    audit_summary = {
        "path": audit_path.relative_to(research_root).as_posix(),
        "sha256": files[audit_path.relative_to(research_root).as_posix()],
        "name": name,
        "status": audit["status"],
        "kind": kind,
        "full": full,
        "seed": seed,
        "counts": audit["counts"],
        "membership_ids_sha256": _contract_digest(audit["membership_ids"]),
        "initial_sha256": audit.get("initial_sha256"),
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "predictions_sha256": audit["predictions_sha256"],
        "epochs": audit["epochs"],
        "slopes": audit["slopes"],
    }
    return model, logits, {"audit": audit_summary, "external_files_sha256": files}


def _load_shared_experts(
    research_root: Path, data: dict[str, Any], backend: dict[str, Any],
    year: int, train: np.ndarray, cal: np.ndarray, test: np.ndarray,
    selected: np.ndarray, shared_root: Path, shared_contract: dict[str, Any],
) -> tuple[Any, list[Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse the archived, identity-verified raw-rating and full-MLP experts."""
    frame = data["frame"]
    membership = {
        "train": _member_ids(frame, train),
        "calibration": _member_ids(frame, cal),
        "test": _member_ids(frame, test),
    }
    source_hashes = _frozen_expert_source_hashes(research_root)
    bank_receipt_sha256 = data["bank_receipt_sha256"]
    shared_root.mkdir(parents=True, exist_ok=True)
    receipt_path = shared_root / "experts_receipt.json"
    previous = None
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if previous.get("contract") != _json_safe(shared_contract):
            raise ValueError("frozen shared-expert receipt has a different input/source contract")
        for relative, expected in previous.get("external_artifacts_sha256", {}).items():
            path = research_root / relative
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"frozen shared expert changed after its receipt: {path}")
        for name, expected in previous.get("artifacts_sha256", {}).items():
            path = shared_root / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"derived shared-expert artifact changed: {path}")

    rating_name = f"{year}_ratings_raw_{SEEDS[0]}"
    rating, rating_z, rating_record = _load_frozen_expert(
        research_root, backend, frame, year, rating_name, SEEDS[0],
        "raw", False, membership, selected, source_hashes, bank_receipt_sha256,
    )
    mlps, mlp_predictions = [], []
    expert_records = {rating_name: rating_record}
    for seed in SEEDS:
        name = f"{year}_full_mlp_{seed}"
        model, logits, record = _load_frozen_expert(
            research_root, backend, frame, year, name, seed, "mlp", True,
            membership, selected, source_hashes, bank_receipt_sha256,
        )
        mlps.append(model)
        mlp_predictions.append(logits)
        expert_records[name] = record
    mlp_z = np.stack(mlp_predictions, axis=2)
    if rating_z.shape != (len(selected), 3) or mlp_z.shape != (len(selected), 3, 3):
        raise ValueError("frozen expert predictions do not match annual CAL/TEST rows")

    logits_path = shared_root / "expert_logits.npz"
    external_hashes = {
        key: value
        for record in expert_records.values()
        for key, value in record["external_files_sha256"].items()
    }
    logits_contract = {
        **shared_contract,
        "artifact": "reused_frozen_raw_rating_and_full_mlp_logits",
        "indices": selected.tolist(),
        "external_artifacts_sha256": external_hashes,
    }
    logits_sidecar = logits_path.with_suffix(logits_path.suffix + ".receipt.json")
    if logits_path.exists() or logits_sidecar.exists():
        if not logits_path.is_file() or not logits_sidecar.is_file():
            raise FileNotFoundError("partial reused-expert logit artifact")
        sidecar = json.loads(logits_sidecar.read_text())
        if sidecar.get("contract") != _json_safe(logits_contract) or sidecar.get("sha256") != _sha256(logits_path):
            raise ValueError("reused frozen-expert logits do not match their source contract")
        with np.load(logits_path, allow_pickle=False) as archive:
            if (
                not np.array_equal(archive["indices"], selected)
                or not np.array_equal(archive["rating_z"], rating_z)
                or not np.array_equal(archive["mlp_z"], mlp_z)
            ):
                raise ValueError("reused frozen-expert logits differ from their archived predictions")
    else:
        np.savez_compressed(logits_path, indices=selected, rating_z=rating_z, mlp_z=mlp_z)
        _write_json(logits_sidecar, {
            "contract": _json_safe(logits_contract), "sha256": _sha256(logits_path),
        })
    current = {
        "status": "FROZEN_EXPERTS_REUSED",
        "contract": _json_safe(shared_contract),
        "contract_sha256": _contract_digest(shared_contract),
        "frozen_source_sha256": source_hashes,
        "membership": membership,
        "expert_names": [rating_name, *[f"{year}_full_mlp_{seed}" for seed in SEEDS]],
        "expert_artifacts": expert_records,
        "external_artifacts_sha256": external_hashes,
        "logit_contract": _json_safe(logits_contract),
        "artifacts_sha256": {
            logits_path.name: _sha256(logits_path),
            logits_sidecar.name: _sha256(logits_sidecar),
        },
        "calibration_rule": "reuse original annual raw-rating/full-MLP checkpoints and their CAL-fitted slopes; no expert retraining in either target arm",
    }
    if previous is not None and previous != _json_safe(current):
        raise ValueError("frozen shared-expert receipt differs from the verified archive")
    if previous is None:
        _write_json(receipt_path, current)
    else:
        current = previous
    return rating, mlps, rating_z, mlp_z, current

def _neural_contract(
    *, base_contract: dict[str, Any], objective: str, seed: int,
    train_ids: list[str], cal_ids: list[str], test_ids: list[str],
) -> dict[str, Any]:
    return {
        **base_contract,
        "objective": objective,
        "seed": int(seed),
        "epochs": 12,
        "auxiliary_coefficient": 0.1,
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "cal_ids_sha256": hashlib.sha256("\n".join(cal_ids).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "network_weight_decay": 0.01, "eta_weight_decay": 0.0},
        "schedule": "12 fixed epochs; batch 128; gradient norm clip 5; seed-matched dropout/masks",
    }



def _predict_mixture(
    backend: dict[str, Any], neural_z: np.ndarray, experts_z: np.ndarray,
    slopes: np.ndarray, gates: np.ndarray, best_of: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta_calibration = backend["beta_calibration"]
    reliability = backend["tail_reliability"]
    z = np.concatenate((neural_z, experts_z), axis=2)
    prediction = np.empty((len(best_of), 3), dtype=float)
    p_native = np.empty_like(prediction)
    p_raw_nn = expit(neural_z).mean(axis=2)
    for mode_index, mode in enumerate(MODES):
        expert_probability = reliability.calibrated_experts(z, slopes, gates, mode_index)
        mass = beta_calibration.calibrated_distribution(
            neural_z, calibration["beta_parameters"], gates,
            best_of, mode_index,
        )
        p_native[:, mode_index] = np.exp(logsumexp(mass[:, :3], axis=1))
        expert_probability[:, 0] = p_native[:, mode_index]
        mix = calibration["mixtures"][mode]
        context = np.zeros((len(best_of), len(mix.mean)), dtype=float)
        prediction[:, mode_index] = mix.predict(expert_probability, context)
    if not np.isfinite(prediction).all() or np.any((prediction <= 0) | (prediction >= 1)):
        raise ValueError("native target mixture produced invalid probabilities")
    return prediction, p_native, p_raw_nn




def _fit_calibration(
    backend: dict[str, Any], frame: pd.DataFrame, neural_z: np.ndarray,
    experts_z: np.ndarray, expert_slopes: np.ndarray, gates: np.ndarray,
    year: int, wins_a: np.ndarray, wins_b: np.ndarray, raw_features: dict[str, Any],
):
    beta_calibration = backend["beta_calibration"]
    beta_terminal = backend["beta_terminal"]
    reliability = backend["tail_reliability"]
    from src.models.competition_tiers import classify_competition

    z = np.concatenate((neural_z, experts_z), axis=2)
    if z.shape[1:] != (3, 7) or expert_slopes.shape != (3, 7):
        raise ValueError("native neural/expert calibration arrays are not aligned")
    cf_frame = frame.copy()
    cf_frame["wins_a"] = wins_a
    cf_frame["wins_b"] = wins_b
    cf, other_proof = reliability.crossfit_probabilities(z, cf_frame, raw_features["gates"], year - 1)
    neural_cf, nn_proof = beta_calibration.joint_crossfit(neural_z, cf_frame, raw_features["gates"], year - 1, "beta")
    cf[:, :, 0] = neural_cf
    ci = np.flatnonzero(np.isfinite(cf).all(axis=(1, 2)))
    dates = cf_frame.date.to_numpy(dtype=str)
    releases = cf_frame.result_day.to_numpy(dtype=str)
    fi = np.flatnonzero((dates >= f"{year - 1:04d}-01-01") & (dates < f"{year:04d}-01-01") & (releases < f"{year:04d}-01-01"))
    if not len(fi) or not len(ci):
        raise ValueError("native annual CAL or quarter-crossfit mixture support is empty")
    for proof in other_proof + nn_proof:
        fit_ix = np.asarray(proof.get("fit_indices", []), dtype=np.int64)
        if len(fit_ix) and not (cf_frame.iloc[fit_ix].result_day < proof["cutoff"]).all():
            raise ValueError("late result release entered native quarter calibration")
    if not (cf_frame.iloc[ci].result_day < f"{year:04d}-01-01").all():
        raise ValueError("late annual CAL label entered native mixture calibration")

    beta_records = []
    beta_parameters = np.empty((3, 2), dtype=float)
    for mode_index in range(3):
        record = beta_terminal.fit_joint(neural_z[fi, mode_index], wins_a[fi], wins_b[fi], "beta")
        beta_records.append(record)
        beta_parameters[mode_index] = (record["scale"], record["rho"])
    tiers = np.asarray([
        classify_competition(tournament, pd.Timestamp(day).date()).tier.value
        for tournament, day in zip(cf_frame.tournament_name, cf_frame.date)
    ], dtype=object)
    mixtures: dict[str, Any] = {}
    mix_records = []
    for mode_index, mode in enumerate(MODES):
        experts = reliability.calibrated_experts(z, expert_slopes, raw_features["gates"], mode_index)
        mass = beta_calibration.calibrated_distribution(
            neural_z, beta_parameters, raw_features["gates"],
            cf_frame.best_of.to_numpy(dtype=np.int64), mode_index,
        )
        experts[:, 0] = np.exp(logsumexp(mass[:, :3], axis=1))
        crossfit_experts = experts.copy()
        crossfit_experts[ci] = cf[ci, mode_index]
        context, context_names = reliability.context_features(
            raw_features, cf_frame.best_of.to_numpy(dtype=np.int64), tiers,
            crossfit_experts, w20=mode == "full", org=mode != "no_organization",
        )
        mixture = reliability.ReliabilityLayer("constant_mixture").fit(
            cf[ci, mode_index], context[ci], cf_frame.y.to_numpy(dtype=int)[ci],
        )
        mixtures[mode] = mixture
        mix_records.append({
            "mode": mode, "theta": mixture.theta.tolist(),
            "fit_ids": cf_frame.iloc[ci].golgg_match_id.astype(str).tolist(),
            "context_columns": context_names, "fit_info": mixture.fit_info,
        })
    report = {
        "objective": "common native terminal Beta calibration, identical for both TRAIN objective arms",
        "native_neural_calibration": "joint three-seed Beta calibration; no per-seed/K5 uncertainty calibration",
        "beta": beta_records, "mixtures": mix_records,
        "quarter_other": other_proof, "quarter_nn": nn_proof,
        "annual_cal_ids": cf_frame.iloc[fi].golgg_match_id.astype(str).tolist(),
        "mixture_cal_ids": cf_frame.iloc[ci].golgg_match_id.astype(str).tolist(),
        "quarter_calibrated_rows": int(len(ci)),
    }
    state = {
        "beta_parameters": beta_parameters, "beta_records": beta_records,
        "slopes": expert_slopes, "mixtures": mixtures, "report": report,
    }
    return state, report




def train_annual_target_comparison(
    protocol_path: str | Path,
    research_root: str | Path | None = None,
    *,
    objective: str,
    year: int,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Fit one frozen objective/year arm (three paired seeds) and export TEST rows."""
    if objective not in OBJECTIVES or year not in YEARS:
        raise ValueError("objective and annual model year must match the frozen protocol")
    if device != "cpu":
        raise ValueError("the frozen native target protocol requires CPU execution")
    protocol_file, protocol = _load_protocol(protocol_path)
    root = resolve_research_root(research_root)
    from src.models import a0_uncertainty as native
    from src.models.a0_target_objectives import fit_beta_target_on_device

    native._configure_training_device(device, 2)
    backend, backend_hashes = _activate(root)
    source_hashes = _project_source_hashes(backend_hashes)
    data = _load_data(root)
    frame, train, cal, test = _training_frame(data, year)
    selected = np.sort(np.concatenate((cal, test))).astype(np.int64)
    test_local = np.flatnonzero(np.char.startswith(frame.iloc[selected].date.to_numpy(dtype=str), str(year)))
    if len(test_local) != len(test):
        raise ValueError("annual prediction membership disagrees with the corrected-bank TEST split")
    train_ids, cal_ids, test_ids = _member_ids(frame, train), _member_ids(frame, cal), _member_ids(frame, test)
    input_hashes = dict(data["input_hashes"])
    _verify_input_hashes(input_hashes)
    base_contract = {
        "run_id": RUN_ID,
        "protocol_sha256": _sha256(protocol_file),
        "research_root": str(root),
        "year": int(year),
        "epochs": 12,
        "device": device,
        "threads": 2,
        "seeds": list(SEEDS),
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "bank_receipt_sha256": data["bank_receipt_sha256"],
        "training_membership_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
    }
    base = "a0_target_experiment"
    model_root = PROJECT_ROOT / f"data/06_models/{base}" / RUN_ID
    prediction_root = PROJECT_ROOT / f"data/07_model_output/{base}" / RUN_ID
    report_root = PROJECT_ROOT / f"data/08_reporting/{base}" / RUN_ID
    arm_root = model_root / objective / str(year)
    arm_prediction = prediction_root / objective / f"annual_{year}_test.parquet"
    arm_receipt_path = report_root / objective / f"annual_{year}_receipt.json"
    arm_contract = {**base_contract, "objective": objective}
    if arm_receipt_path.is_file():
        old = json.loads(arm_receipt_path.read_text())
        if not resume:
            raise FileExistsError(f"annual target arm already exists: {arm_receipt_path}")
        if old.get("contract") != _json_safe(arm_contract):
            raise ValueError("completed native target arm has a different execution contract")
        for rel, expected in old["artifacts_sha256"].items():
            path = PROJECT_ROOT / rel
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"completed native target artifact changed: {path}")
        shared = old.get("shared_experts", {}).get("receipt", {})
        if shared.get("status") != "FROZEN_EXPERTS_REUSED":
            raise ValueError("completed native target arm lacks the frozen-expert reuse receipt")
        if shared.get("frozen_source_sha256") != _frozen_expert_source_hashes(root):
            raise ValueError("completed native frozen expert source hash contract changed")
        shared_dir = model_root / "shared_experts" / str(year)
        shared_receipt_path = shared_dir / "experts_receipt.json"
        if (
            not shared_receipt_path.is_file()
            or _sha256(shared_receipt_path) != old.get("shared_experts", {}).get("receipt_sha256")
            or json.loads(shared_receipt_path.read_text()) != shared
        ):
            raise ValueError("completed shared-expert receipt changed after the annual run")
        for rel, expected in shared.get("external_artifacts_sha256", {}).items():
            path = root / rel
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"completed native frozen expert changed: {path}")
        for name, expected in shared.get("artifacts_sha256", {}).items():
            path = shared_dir / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"completed derived shared expert artifact changed: {path}")
        return old
    for path in (arm_root, arm_prediction, arm_receipt_path):
        if path.exists() and not resume:
            if path.is_dir() and not any(path.iterdir()):
                continue
            raise FileExistsError(f"refusing to overwrite native target experiment artifact: {path}")
    arm_root.mkdir(parents=True, exist_ok=True)
    arm_prediction.parent.mkdir(parents=True, exist_ok=True)
    arm_receipt_path.parent.mkdir(parents=True, exist_ok=True)

    shared_root = model_root / "shared_experts" / str(year)
    shared_contract = {
        "run_id": RUN_ID,
        "protocol_sha256": _sha256(protocol_file),
        "research_root": str(root),
        "year": int(year),
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "bank_receipt_sha256": data["bank_receipt_sha256"],
        "training_membership_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "expert_recipe": "reuse identity-verified archived raw-rating and three full-MLP experts with their frozen TRAIN/CAL fit",
    }
    rating, mlps, rating_z, mlp_z, shared_receipt = _load_shared_experts(
        root, data, backend, year, train, cal, test, selected, shared_root,
        shared_contract,
    )
    labels = frame.y.to_numpy(dtype=np.float64)
    best_of = frame.best_of.to_numpy(dtype=np.int64)
    bo_index = (best_of - 1) // 2
    wins_a, wins_b = data["wins_a"], data["wins_b"]

    from src.models.roster_optional import RosterModel
    from src.models.roster_optional import subset as roster_subset
    from src.models.temporal_training import HistoryNormalizer
    baseline = RosterModel("linear").fit(roster_subset(data["raw"], train), labels[train], best_of[train])
    normalizer = HistoryNormalizer().fit(data["history"], train)
    neural_type = backend["nonlinear_history"].NonlinearHistoryModel
    models: list[Any] = []
    neural_receipts = []
    for seed in SEEDS:
        checkpoint = arm_root / f"neural_seed_{seed}.joblib"
        checkpoint_contract = _neural_contract(
            base_contract=base_contract, objective=objective, seed=seed,
            train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
        )
        if checkpoint.exists() or checkpoint.with_suffix(checkpoint.suffix + ".receipt.json").exists():
            if not resume:
                raise FileExistsError(f"native neural checkpoint exists; pass --resume to verify: {checkpoint}")
            model, checkpoint_receipt = _restore_neural(checkpoint, checkpoint_contract, neural_type, seed, 12)
        else:
            model = neural_type(baseline, normalizer, seed=seed, epochs=12)
            initial_hash = _state_hash(model)
            started = time.monotonic()

            def progress(epoch, main_loss, auxiliary_loss, diagnostic, *, _seed=seed):
                print(json.dumps({
                    "status": "EPOCH", "objective": objective, "year": year,
                    "seed": _seed, "epoch": epoch, "main_loss": main_loss,
                    "auxiliary_loss": auxiliary_loss, "rho": diagnostic["rho"],
                }), flush=True)

            fit_beta_target_on_device(
                model, data["raw"], data["history"], None, train, bo_index,
                wins_a, wins_b, data["targets"], data["target_mask"], progress,
                device=device, objective=objective,
            )
            model.main_objective = objective
            if len(model.train_losses) != 12 or len(model.auxiliary_losses) != 12:
                raise ValueError("native target model did not complete exactly 12 epochs")
            if not np.isfinite(model.train_losses + model.auxiliary_losses).all():
                raise ValueError("native target model has nonfinite training losses")
            mask_hash = hashlib.sha256(np.asarray(data["target_mask"][train], dtype=bool).tobytes()).hexdigest()
            details = {
                "objective": objective,
                "seed": seed,
                "epochs": model.epochs,
                "main_loss_by_epoch": list(model.train_losses),
                "auxiliary_loss_by_epoch": list(model.auxiliary_losses),
                "total_loss_by_epoch": [float(m + model.coefficient * a) for m, a in zip(model.train_losses, model.auxiliary_losses)],
                "epoch_diagnostics": model.epoch_diagnostics,
                "dropout_mask_rng_final_state_sha256": hashlib.sha256(
                    json.dumps(_json_safe(model.dropout_rng_final_state), sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "initialization_sha256": initial_hash,
                "training_mask_seed": seed,
                "training_target_mask_sha256": mask_hash,
                "auxiliary_coefficient": model.coefficient,
                "target_scaler": {
                    "mean": model.target_scaler.mean,
                    "scale": model.target_scaler.scale,
                    "counts": model.target_scaler.counts,
                },
                "training_seconds": time.monotonic() - started,
                "training_ids_sha256": checkpoint_contract["train_ids_sha256"],
                "cal_ids_sha256": checkpoint_contract["cal_ids_sha256"],
                "test_ids_sha256": checkpoint_contract["test_ids_sha256"],
            }
            checkpoint_receipt = _save_checkpoint(checkpoint, model, checkpoint_contract, details)
        models.append(model)
        neural_receipts.append({
            "seed": seed,
            "checkpoint": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
            "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
            "contract_sha256": checkpoint_receipt["contract_sha256"],
            "details": checkpoint_receipt["details"],
        })

    other_objective = "winner" if objective == "terminal" else "terminal"
    other_receipt = report_root / other_objective / f"annual_{year}_receipt.json"
    if other_receipt.is_file():
        other = json.loads(other_receipt.read_text())
        by_seed = {
            entry["seed"]: entry["details"]
            for entry in other.get("neural", [])
        }
        paired_fields = (
            "initialization_sha256",
            "training_mask_seed",
            "training_target_mask_sha256",
            "dropout_mask_rng_final_state_sha256",
            "auxiliary_coefficient",
        )
        for entry in neural_receipts:
            previous = by_seed.get(entry["seed"], {})
            current = entry["details"]
            for field in paired_fields:
                if previous.get(field) is not None and current.get(field) is not None and previous[field] != current[field]:
                    raise ValueError(f"paired {field} differs for seed {entry['seed']}")

    neural_z = _map_logits(models, data["raw"], data["history"], selected, bo_index)
    expert_z = np.concatenate((rating_z[:, :, None], mlp_z), axis=2)
    expert_slopes = np.ones((3, 7), dtype=float)
    for mode_index, mode in enumerate(MODES):
        expert_slopes[mode_index, 3] = float(rating.slopes[mode])
        for j, model in enumerate(mlps):
            expert_slopes[mode_index, 4 + j] = float(model.slopes[mode])

    calibration_path = arm_root / "calibration_state.joblib"
    calibration_sidecar_path = calibration_path.with_suffix(calibration_path.suffix + ".receipt.json")
    calibration_contract = {**arm_contract, "artifact": "native_three_neural_beta_and_quarter_crossfit_constant_mixture"}
    if calibration_path.exists() or calibration_sidecar_path.exists():
        if not resume or not calibration_path.is_file() or not calibration_sidecar_path.is_file():
            raise FileExistsError(f"partial calibration state requires verified resume: {calibration_path}")
        calibration_receipt = _verified_sidecar(calibration_path, calibration_contract)
        calibration = joblib.load(calibration_path)
        calibration_report = calibration_receipt["details"]["calibration_report"]
    else:
        selected_frame = frame.iloc[selected].reset_index(drop=True).copy()
        raw_selected = _raw_subset(data["raw"], selected)
        calibration, calibration_report = _fit_calibration(
            backend, selected_frame, neural_z, expert_z, expert_slopes,
            np.asarray(data["raw"]["gates"])[selected], year,
            wins_a[selected], wins_b[selected], raw_features=raw_selected,
        )
        calibration["objective"] = objective
        calibration["year"] = year
        calibration["report"] = calibration_report
        joblib.dump(calibration, calibration_path)
        calibration_receipt = {
            "contract": _json_safe(calibration_contract),
            "contract_sha256": _contract_digest(calibration_contract),
            "checkpoint_sha256": _sha256(calibration_path),
            "details": {"calibration_report": calibration_report},
        }
        _write_json(calibration_sidecar_path, calibration_receipt)

    selected_frame = frame.iloc[selected].reset_index(drop=True).copy()
    prediction, native_probability, raw_map_probability = _predict_mixture(
        backend, neural_z, expert_z, expert_slopes,
        np.asarray(data["raw"]["gates"])[selected],
        best_of[selected], calibration,
    )
    test_rows = test_local
    result = selected_frame.iloc[test_rows].reset_index(drop=True).copy()
    result["model_year"] = int(year)
    result["objective"] = objective
    result["train_end"] = f"{year - 2:04d}-12-31"
    result["calibration_end"] = f"{year - 1:04d}-12-31"
    result["main_objective"] = objective
    result["main_loss_final"] = [float(np.mean([model.train_losses[-1] for model in models]))] * len(result)
    for mode_index, mode in enumerate(MODES):
        result[f"p_{mode}"] = prediction[test_rows, mode_index]
        result[f"p_native_beta_{mode}"] = native_probability[test_rows, mode_index]
        result[f"p_nn_raw_{mode}"] = raw_map_probability[test_rows, mode_index]
    result["p"] = result["p_full"]
    if result.golgg_match_id.astype(str).duplicated().any() or len(result) != len(test):
        raise ValueError("annual native target TEST output lost or duplicated target rows")
    probability_columns = ["p", *[f"p_{mode}" for mode in MODES], *[f"p_native_beta_{mode}" for mode in MODES], *[f"p_nn_raw_{mode}" for mode in MODES]]
    probabilities = result[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("annual native target predictions contain invalid probabilities")
    required_candidate = ["golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y", "result_day", "feature_source_max_day", "train_end", "calibration_end", "p"]
    missing = sorted(set(required_candidate) - set(result.columns))
    if missing:
        raise ValueError(f"candidate output is missing canonical benchmark columns: {missing}")

    if arm_prediction.exists():
        if not resume:
            raise FileExistsError(f"annual target prediction output exists: {arm_prediction}")
        prior_sidecar_path = arm_prediction.with_suffix(arm_prediction.suffix + ".receipt.json")
        if not prior_sidecar_path.is_file():
            raise FileNotFoundError("existing annual prediction output has no contract receipt")
        prior = json.loads(prior_sidecar_path.read_text())
        if prior.get("contract") != _json_safe(arm_contract) or prior.get("sha256") != _sha256(arm_prediction):
            raise ValueError("existing annual candidate output differs from the requested arm")
        stored = pd.read_parquet(arm_prediction)
        if not np.array_equal(stored.golgg_match_id.astype(str), result.golgg_match_id.astype(str)):
            raise ValueError("existing annual candidate IDs differ from the requested output")
        if not np.allclose(stored.p.to_numpy(float), result.p.to_numpy(float), rtol=0, atol=1e-12):
            raise ValueError("existing annual candidate predictions differ from reloaded checkpoints")
    else:
        result.to_parquet(arm_prediction, index=False)
        _write_json(arm_prediction.with_suffix(arm_prediction.suffix + ".receipt.json"), {
            "contract": _json_safe(arm_contract), "sha256": _sha256(arm_prediction),
        })

    if any(_sha256(Path(filename)) != value for filename, value in input_hashes.items()):
        raise ValueError("a frozen research input changed during native target training")
    current_source_hashes, _ = _source_contract(root)
    if current_source_hashes != source_hashes:
        raise ValueError("native target source code changed during training")
    _verify_input_hashes(input_hashes)

    saved_inference = saved_checkpoint_inference_smoke(
        arm_root, shared_root, root, year=year, predictions=result,
        backend=backend, max_rows=9,
    )
    # Verify the other arm's paired initialization and training masks when it has
    # already been run; neither arm is required to wait for the other.
    artifacts = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in (arm_root.rglob("*") if arm_root.exists() else [])
        if path.is_file()
    }
    artifacts[arm_prediction.relative_to(PROJECT_ROOT).as_posix()] = _sha256(arm_prediction)
    artifacts[arm_prediction.with_suffix(arm_prediction.suffix + ".receipt.json").relative_to(PROJECT_ROOT).as_posix()] = _sha256(arm_prediction.with_suffix(arm_prediction.suffix + ".receipt.json"))
    receipt = {
        "status": "ANNUAL_COMPLETE",
        "research_status": "DEVELOPMENT_ONLY",
        "run_id": RUN_ID,
        "objective": objective,
        "year": year,
        "contract": _json_safe(arm_contract),
        "contract_sha256": _contract_digest(arm_contract),
        "research_root": str(root),
        "inputs_sha256": input_hashes,
        "source_sha256": source_hashes,
        "backend_source_sha256": backend_hashes,
        "memberships": {
            "train_ids": train_ids, "cal_ids": cal_ids,
            "test_ids": test_ids, "prediction_ids": result.golgg_match_id.astype(str).tolist(),
        },
        "counts": {"train": len(train), "cal": len(cal), "test": len(test)},
        "neural": neural_receipts,
        "shared_experts": {
            "receipt": shared_receipt,
            "receipt_sha256": _sha256(shared_root / "experts_receipt.json"),
            "rating_checkpoint_path": (
                f"a0-phase-walkforward-20260914/data/06_models/causal_a0/"
                f"{year}/other/{year}_ratings_raw_{SEEDS[0]}.joblib"
            ),
            "rating_checkpoint_sha256": shared_receipt["external_artifacts_sha256"][
                f"a0-phase-walkforward-20260914/data/06_models/causal_a0/{year}/other/{year}_ratings_raw_{SEEDS[0]}.joblib"
            ],
            "full_mlp_checkpoint_sha256": {
                str(seed): shared_receipt["external_artifacts_sha256"][
                    f"a0-phase-walkforward-20260914/data/06_models/causal_a0/{year}/other/{year}_full_mlp_{seed}.joblib"
                ]
                for seed in SEEDS
            },
        },
        "training_contract": {
            "only_main_loss_changes": True,
            "objective": objective,
            "auxiliary_coefficient": 0.1,
            "optimizer": "AdamW(lr=0.001, network weight_decay=0.01, eta weight_decay=0.0)",
            "batch_size": 128,
            "gradient_clip_norm": 5,
            "epochs": 12,
            "seed_matched_initialization_and_masks": True,
        },
        "calibration": calibration_report,
        "calibration_state_sha256": _sha256(calibration_path),
        "prediction_path": arm_prediction.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_sha256": _sha256(arm_prediction),
        "prediction_rows": len(result),
        "prediction_fields": probability_columns,
        "canonical_benchmark_candidate_column": "p",
        "saved_inference_smoke": saved_inference,
        "versions": _versions(),
        "device": device,
        "threads": 2,
        "artifacts_sha256": artifacts,
    }
    _write_json(arm_receipt_path, receipt)
    del data, models, rating, mlps, neural_z, expert_z, prediction, native_probability, raw_map_probability
    gc.collect()
    return _json_safe(receipt)


def _inference_components(
    model_root: Path, shared_root: Path, research_root: Path,
    data: dict[str, Any], backend: dict[str, Any], year: int, objective: str,
):
    neural_type = backend["nonlinear_history"].NonlinearHistoryModel
    neural_models = []
    for seed in SEEDS:
        path = model_root / f"neural_seed_{seed}.joblib"
        sidecar = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text())
        _verified_sidecar(path, sidecar.get("contract", {}))
        model = joblib.load(path)
        if (
            type(model) is not neural_type or model.seed != seed
            or model.epochs != 12 or model.main_objective != objective
        ):
            raise ValueError(f"saved neural checkpoint does not match inference contract: {path}")
        neural_models.append(model)

    shared_receipt_path = shared_root / "experts_receipt.json"
    if not shared_receipt_path.is_file():
        raise FileNotFoundError(f"frozen shared-expert receipt is missing: {shared_receipt_path}")
    shared_receipt = json.loads(shared_receipt_path.read_text())
    if shared_receipt.get("status") != "FROZEN_EXPERTS_REUSED":
        raise ValueError("saved inference requires the verified frozen-expert reuse receipt")
    _, train, cal, test = _training_frame(data, year)
    selected = np.sort(np.concatenate((cal, test))).astype(np.int64)
    rating, mlps, _, _, verified_receipt = _load_shared_experts(
        research_root, data, backend, year, train, cal, test, selected,
        shared_root, shared_receipt["contract"],
    )
    if verified_receipt != shared_receipt:
        raise ValueError("saved shared-expert receipt changed during inference verification")

    calibration_path = model_root / "calibration_state.joblib"
    calibration_sidecar = json.loads(calibration_path.with_suffix(calibration_path.suffix + ".receipt.json").read_text())
    _verified_sidecar(calibration_path, calibration_sidecar.get("contract", {}))
    calibration = joblib.load(calibration_path)
    if calibration.get("year") != year or calibration.get("objective") != objective:
        raise ValueError("saved calibration belongs to a different annual target arm")
    return neural_models, rating, mlps, calibration


def saved_checkpoint_inference_smoke(
    model_root: str | Path,
    shared_root: str | Path,
    research_root: str | Path,
    *,
    year: int,
    predictions: pd.DataFrame | None = None,
    objective: str | None = None,
    backend: dict[str, Any] | None = None,
    max_rows: int = 9,
) -> dict[str, Any]:
    """Reload checkpoints and check stored, swapped-side, cold, and context outputs."""
    if max_rows < 3:
        raise ValueError("inference smoke needs at least three rows for all best-of values")
    root = resolve_research_root(research_root)
    model_dir, shared_dir = Path(model_root), Path(shared_root)
    if objective is None:
        receipt_candidates = sorted(model_dir.glob("annual_*_receipt.json"))
        if receipt_candidates:
            objective = json.loads(receipt_candidates[0].read_text()).get("objective")
        else:
            objective = model_dir.parent.name
    if objective not in OBJECTIVES:
        raise ValueError("inference smoke requires terminal or winner objective")
    if backend is None:
        backend, _ = _activate(root)
    data = _load_data(root)
    frame = data["frame"]
    _, _, _, test = _training_frame(data, year)
    test_frame = frame.iloc[test].reset_index(drop=True)
    if predictions is not None:
        lookup = pd.Series(np.arange(len(test_frame)), index=test_frame.golgg_match_id.astype(str))
        try:
            positions = lookup.loc[predictions.golgg_match_id.astype(str)].to_numpy(dtype=np.int64)
        except KeyError as error:
            raise ValueError("saved-inference candidates are absent from annual TEST membership") from error
        chosen = pd.DataFrame({"position": positions, "best_of": test_frame.iloc[positions].best_of.to_numpy()})
    else:
        chosen = pd.DataFrame({"position": np.arange(len(test_frame)), "best_of": test_frame.best_of.to_numpy()})
    samples = []
    for best_of in (1, 3, 5):
        selected = chosen.loc[chosen.best_of == best_of, "position"].to_numpy(dtype=np.int64)[:max(1, max_rows // 3)]
        if not len(selected):
            raise ValueError(f"saved-inference sample lacks best-of-{best_of}")
        samples.extend(selected.tolist())
    sample = np.asarray(samples, dtype=np.int64)
    global_indices = test[sample]
    frame_sample = frame.iloc[global_indices].reset_index(drop=True)
    raw = _raw_subset(data["raw"], global_indices)
    history = {key: values[global_indices] for key, values in data["history"].items()}
    before = {
        f"raw:{key}": hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        for key, value in raw.items()
    }
    before.update({f"history:{key}": hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for key, value in history.items()})
    models, rating, mlps, calibration = _inference_components(
        model_dir, shared_dir, root, data, backend, year, objective,
    )
    bo_index = (frame_sample.best_of.to_numpy(dtype=np.int64) - 1) // 2
    neural_z = _map_logits(models, raw, history, np.arange(len(sample), dtype=np.int64), bo_index)
    native = __import__("src.models.a0_uncertainty", fromlist=["_unified_logits"])
    raw_local = raw
    hist_local = _indexed_history(history, np.arange(len(sample), dtype=np.int64))
    rating_z = native._unified_logits(rating, raw_local, hist_local, bo_index)
    mlp_z = np.stack([native._unified_logits(model, raw_local, hist_local, bo_index) for model in mlps], axis=2)
    slopes = np.asarray(calibration["slopes"], dtype=float)
    gates = np.asarray(raw["gates"], dtype=float)
    p, p_native, _ = _predict_mixture(
        backend, neural_z, np.concatenate((rating_z[:, :, None], mlp_z), axis=2),
        slopes, gates, frame_sample.best_of.to_numpy(dtype=np.int64), calibration,
    )
    checks = {}
    for mode_index, mode in enumerate(MODES):
        candidate_col = f"p_{mode}"
        if predictions is not None:
            expected = predictions.set_index("golgg_match_id").loc[frame_sample.golgg_match_id.astype(str), candidate_col].to_numpy(float)
            checks[f"{mode}_stored_error"] = float(np.max(np.abs(p[:, mode_index] - expected)))
        swapped_raw = {key: np.asarray(value)[:, ::-1].copy() for key, value in raw.items()}
        swapped_history = {key: np.asarray(value)[:, ::-1].copy() for key, value in history.items()}
        reverse_nn = _map_logits(models, swapped_raw, swapped_history, np.arange(len(sample), dtype=np.int64), bo_index)
        reverse_rating = native._unified_logits(rating, swapped_raw, _indexed_history(swapped_history, np.arange(len(sample), dtype=np.int64)), bo_index)
        reverse_mlps = np.stack([
            native._unified_logits(model, swapped_raw, _indexed_history(swapped_history, np.arange(len(sample), dtype=np.int64)), bo_index)
            for model in mlps
        ], axis=2)
        reverse_p, _, _ = _predict_mixture(
            backend, reverse_nn, np.concatenate((reverse_rating[:, :, None], reverse_mlps), axis=2),
            slopes, swapped_raw["gates"], frame_sample.best_of.to_numpy(dtype=np.int64), calibration,
        )
        checks[f"{mode}_side_swap_error"] = float(np.max(np.abs(p[:, mode_index] + reverse_p[:, mode_index] - 1.0)))
    cold_raw = {key: np.zeros_like(value) for key, value in raw.items()}
    cold_history = {key: np.zeros_like(value) for key, value in history.items()}
    cold_nn = _map_logits(models, cold_raw, cold_history, np.arange(len(sample), dtype=np.int64), bo_index)
    cold_rating = native._unified_logits(rating, cold_raw, _indexed_history(cold_history, np.arange(len(sample), dtype=np.int64)), bo_index)
    cold_mlps = np.stack([
        native._unified_logits(model, cold_raw, _indexed_history(cold_history, np.arange(len(sample), dtype=np.int64)), bo_index)
        for model in mlps
    ], axis=2)
    cold_p, cold_native, _ = _predict_mixture(
        backend, cold_nn, np.concatenate((cold_rating[:, :, None], cold_mlps), axis=2),
        slopes, cold_raw["gates"], frame_sample.best_of.to_numpy(dtype=np.int64), calibration,
    )
    checks["cold_neural_map_centering_error"] = float(np.max(np.abs(expit(cold_nn) - 0.5)))
    checks["cold_native_beta_centering_error"] = float(np.max(np.abs(cold_native - 0.5)))
    checks["cold_mixture_centering_error"] = float(np.max(np.abs(cold_p - 0.5)))
    after = {
        f"raw:{key}": hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        for key, value in raw.items()
    }
    after.update({f"history:{key}": hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for key, value in history.items()})
    checks["input_mutation"] = before != after
    numeric = [value for key, value in checks.items() if key != "input_mutation"]
    if checks["input_mutation"] or any(not np.isfinite(value) or value > 2e-6 for value in numeric):
        raise ValueError(f"saved native target inference contract failed: {checks}")
    return {"rows": int(len(sample)), "best_of": sorted(set(frame_sample.best_of.astype(int))), "checks": checks, "objective": objective, "year": year}


def aggregate_target_comparison(
    protocol_path: str | Path,
    research_root: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Join three annual held-out slices into complete canonical candidates."""
    protocol_file, protocol = _load_protocol(protocol_path)
    root = resolve_research_root(research_root)
    from src.models.a0_uncertainty import _load_training_data

    canonical_path = root / protocol["inputs"]["canonical_relative"]
    if _sha256(canonical_path) != protocol["inputs"]["canonical_sha256"]:
        raise ValueError("frozen canonical benchmark input hash changed")
    manifest = PROJECT_ROOT / protocol["inputs"]["canonical_manifest"]
    if _sha256(manifest) != protocol["inputs"]["canonical_manifest_sha256"]:
        raise ValueError("frozen canonical benchmark manifest hash changed")
    canonical = pd.read_parquet(canonical_path)
    canonical["golgg_match_id"] = canonical.golgg_match_id.astype(str)
    if canonical.golgg_match_id.isna().any() or canonical.golgg_match_id.duplicated().any():
        raise ValueError("canonical benchmark IDs are null or duplicated")
    prediction_root = PROJECT_ROOT / f"data/07_model_output/a0_target_experiment/{RUN_ID}"
    report_root = PROJECT_ROOT / f"data/08_reporting/a0_target_experiment/{RUN_ID}"
    destination = Path(output_dir).expanduser().resolve() if output_dir else prediction_root
    destination.mkdir(parents=True, exist_ok=True)
    data = _load_training_data(root)
    bank_frame = data["frame"].loc[
        pd.to_datetime(data["frame"]["date"], utc=True).dt.year.isin(YEARS)
    ].copy()
    bank_frame["golgg_match_id"] = bank_frame.golgg_match_id.astype(str)
    if len(bank_frame) != 11550 or len(canonical) != 11550:
        raise ValueError("frozen native target and canonical cohorts must both contain 11550 targets")
    if set(bank_frame.golgg_match_id) != set(canonical.golgg_match_id):
        raise ValueError("corrected-bank target identities differ from the locked canonical cohort")
    input_hashes = data["input_hashes"]
    _verify_input_hashes(input_hashes)
    source_hashes, _ = _source_contract(root)
    frozen_expert_source_hashes = _frozen_expert_source_hashes(root)
    training_source_hashes = None
    executed_source_snapshots: dict[str, str] = {}
    for objective in OBJECTIVES:
        parts = []
        for year in YEARS:
            receipt_path = report_root / objective / f"annual_{year}_receipt.json"
            prediction_path = prediction_root / objective / f"annual_{year}_test.parquet"
            if not receipt_path.is_file() or not prediction_path.is_file():
                raise FileNotFoundError(f"missing annual native target output for {objective}/{year}")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "ANNUAL_COMPLETE" or receipt.get("objective") != objective or receipt.get("year") != year:
                raise ValueError(f"annual receipt contract mismatch: {receipt_path}")
            if training_source_hashes is None:
                training_source_hashes = receipt.get("source_sha256", {})
                if set(training_source_hashes) != set(source_hashes):
                    raise ValueError("annual native target source inventory changed")
                for source, expected in training_source_hashes.items():
                    if source_hashes[source] == expected:
                        continue
                    # Scoring-only repairs need not repeat training. Verify the exact
                    # executed file and require every non-aggregation AST node unchanged.
                    if source != "src/models/a0_target_experiment.py":
                        raise ValueError(f"annual native target training source changed: {source}")
                    snapshot = PROJECT_ROOT / (
                        f"data/06_models/a0_target_experiment/{RUN_ID}/executed_sources/{expected}.py"
                    )
                    if not snapshot.is_file() or _sha256(snapshot) != expected:
                        raise ValueError("executed native target source snapshot missing or changed")
                    import ast

                    trees = [
                        ast.parse(path.read_text()) for path in (snapshot, PROJECT_ROOT / source)
                    ]
                    for tree in trees:
                        tree.body = [
                            node for node in tree.body
                            if not (
                                isinstance(node, ast.FunctionDef)
                                and node.name == "aggregate_target_comparison"
                            )
                        ]
                    if ast.dump(trees[0]) != ast.dump(trees[1]):
                        raise ValueError("native target training/inference changed beyond aggregation")
                    executed_source_snapshots[source] = str(snapshot.relative_to(PROJECT_ROOT))
            if (
                receipt.get("run_id") != RUN_ID
                or receipt.get("contract", {}).get("protocol_sha256") != _sha256(protocol_file)
                or receipt.get("source_sha256") != training_source_hashes
                or receipt.get("inputs_sha256") != input_hashes
            ):
                raise ValueError(f"annual native target source/input contract changed: {receipt_path}")
            for relative, expected in receipt.get("artifacts_sha256", {}).items():
                path = PROJECT_ROOT / relative
                if not path.is_file() or _sha256(path) != expected:
                    raise ValueError(f"annual target model artifact changed: {path}")
            shared_receipt = receipt.get("shared_experts", {}).get("receipt", {})
            if shared_receipt.get("status") != "FROZEN_EXPERTS_REUSED":
                raise ValueError(f"annual target receipt lacks frozen shared experts: {receipt_path}")
            if shared_receipt.get("frozen_source_sha256") != frozen_expert_source_hashes:
                raise ValueError(f"frozen shared expert source contract changed: {receipt_path}")
            shared_dir = PROJECT_ROOT / f"data/06_models/a0_target_experiment/{RUN_ID}/shared_experts/{year}"
            shared_path = shared_dir / "experts_receipt.json"
            shared_details = receipt.get("shared_experts", {})
            if (
                not shared_path.is_file()
                or _sha256(shared_path) != shared_details.get("receipt_sha256")
                or json.loads(shared_path.read_text()) != shared_receipt
            ):
                raise ValueError(f"annual shared-expert receipt changed: {shared_path}")
            for relative, expected in shared_receipt.get("external_artifacts_sha256", {}).items():
                path = root / relative
                if not path.is_file() or _sha256(path) != expected:
                    raise ValueError(f"frozen shared expert artifact changed: {path}")
            for name, expected in shared_receipt.get("artifacts_sha256", {}).items():
                path = shared_dir / name
                if not path.is_file() or _sha256(path) != expected:
                    raise ValueError(f"derived shared expert artifact changed: {path}")
            logits_path = shared_dir / "expert_logits.npz"
            logits_sidecar_path = logits_path.with_suffix(logits_path.suffix + ".receipt.json")
            if not logits_sidecar_path.is_file():
                raise FileNotFoundError(f"shared expert logits lack a sidecar: {logits_path}")
            logits_sidecar = json.loads(logits_sidecar_path.read_text())
            if (
                logits_sidecar.get("contract") != shared_receipt.get("logit_contract")
                or logits_sidecar.get("sha256") != _sha256(logits_path)
            ):
                raise ValueError(f"shared expert logit sidecar changed: {logits_sidecar_path}")
            prediction_sidecar_path = prediction_path.with_suffix(prediction_path.suffix + ".receipt.json")
            if not prediction_sidecar_path.is_file():
                raise FileNotFoundError(f"annual candidate prediction lacks its contract receipt: {prediction_path}")
            prediction_sidecar = json.loads(prediction_sidecar_path.read_text())
            if prediction_sidecar.get("contract") != receipt.get("contract") or prediction_sidecar.get("sha256") != _sha256(prediction_path):
                raise ValueError(f"annual candidate sidecar contract/hash mismatch: {prediction_path}")
            if receipt.get("prediction_sha256") != _sha256(prediction_path):
                raise ValueError(f"annual prediction hash mismatch: {prediction_path}")
            part = pd.read_parquet(prediction_path)
            if len(part) != receipt["counts"]["test"] or part.golgg_match_id.astype(str).duplicated().any():
                raise ValueError(f"annual predictions lost or duplicated IDs: {prediction_path}")
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)
        combined["golgg_match_id"] = combined.golgg_match_id.astype(str)
        if combined.golgg_match_id.duplicated().any() or set(combined.golgg_match_id) != set(bank_frame.golgg_match_id):
            raise ValueError(f"{objective} annual TEST outputs do not cover all 11550 corrected-bank targets")
        ordered = combined.set_index("golgg_match_id").loc[bank_frame.golgg_match_id].reset_index()
        for column in ("team1_id", "team2_id", "date", "best_of", "y", "result_day"):
            if not np.array_equal(ordered[column].astype(str).to_numpy(), bank_frame[column].astype(str).to_numpy()):
                raise ValueError(f"annual {objective} candidate identity differs on {column}")
        candidates = ordered[[
            "golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y",
            "result_day", "feature_source_max_day", "train_end", "calibration_end", "p",
            "p_full", "p_no_w20", "p_no_organization", "p_native_beta_full",
            "p_nn_raw_full", "model_year", "objective",
        ]].copy()
        candidates["golgg_match_id"] = candidates.golgg_match_id.astype(str)
        candidate_path = destination / f"{objective}_test.parquet"
        if candidate_path.exists():
            raise FileExistsError(f"refusing to overwrite aggregate candidate: {candidate_path}")
        candidates.to_parquet(candidate_path, index=False)
    terminal = pd.read_parquet(destination / "terminal_test.parquet")
    winner = pd.read_parquet(destination / "winner_test.parquet")
    canonical["golgg_match_id"] = canonical.golgg_match_id.astype(str)
    # The same candidate schema and identity checks used by the locked benchmark
    # are applied here without running a benchmark or producing scored results.
    from src.analysis.research_benchmark import attach_candidate

    for candidate in (terminal, winner):
        attach_candidate(canonical, candidate, "p")
    paired = terminal[["golgg_match_id", "p"]].merge(
        winner[["golgg_match_id", "p"]], on="golgg_match_id", suffixes=("_terminal", "_winner"),
        validate="one_to_one", sort=False,
    )
    if len(paired) != len(canonical) or paired.golgg_match_id.duplicated().any():
        raise ValueError("paired target comparison failed full canonical ID coverage")
    paired_path = destination / "paired_target_predictions.parquet"
    if paired_path.exists():
        raise FileExistsError(f"refusing to overwrite paired target export: {paired_path}")
    paired.to_parquet(paired_path, index=False)
    artifacts = {
        path.name: _sha256(path)
        for path in (destination / "terminal_test.parquet", destination / "winner_test.parquet", paired_path)
    }
    receipt = {
        "status": "AGGREGATED_CANDIDATES_COMPLETE",
        "research_status": "DEVELOPMENT_ONLY",
        "run_id": RUN_ID,
        "protocol_sha256": _sha256(protocol_file),
        "canonical_path": str(canonical_path),
        "canonical_sha256": _sha256(canonical_path),
        "canonical_rows": len(canonical),
        "target_bank_rows": len(bank_frame),
        "full_bank_rows": len(data["frame"]),
        "training_source_sha256": training_source_hashes,
        "aggregation_source_sha256": source_hashes,
        "executed_source_snapshots": executed_source_snapshots,
        "objectives": list(OBJECTIVES),
        "prediction_scope": "annual held-out TEST only; concatenated years provide exactly one forecast per canonical target; no scoring or retrospective publication claim",
        "candidate_column": "p",
        "artifacts_sha256": artifacts,
        "candidate_schema": list(terminal.columns),
    }
    receipt_path = destination / "target_comparison_aggregation_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate receipt: {receipt_path}")
    _write_json(receipt_path, receipt)
    return _json_safe(receipt)


def smoke_target_objective(
    protocol_path: str | Path,
    research_root: str | Path | None = None,
    *, objective: str,
    year: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit exactly one seed and one epoch in an unscored, separate smoke namespace."""
    if objective not in OBJECTIVES or year not in YEARS:
        raise ValueError("smoke objective/year is outside the frozen contract")
    if device != "cpu":
        raise ValueError("the frozen native target protocol requires CPU execution")
    protocol_file, _ = _load_protocol(protocol_path)
    root = resolve_research_root(research_root)
    from src.models import a0_uncertainty as native
    from src.models.a0_target_objectives import fit_beta_target_on_device
    native._configure_training_device(device, 2)
    backend, source_hashes = _activate(root)
    source_hashes = _project_source_hashes(source_hashes)
    data = _load_data(root)
    frame, train, cal, test = _training_frame(data, year)
    seed = SEEDS[0]
    ids = _member_ids(frame, train)
    contract = {
        "run_id": RUN_ID, "objective": objective, "year": year, "seed": seed,
        "epochs": 1, "scoring_eligible": False,
        "protocol_sha256": _sha256(protocol_file), "input_sha256": data["input_hashes"],
        "source_sha256": source_hashes, "train_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "device": device, "threads": 2,
    }
    smoke_root = PROJECT_ROOT / f"data/06_models/a0_target_experiment_smoke/{RUN_ID}/{objective}/{year}"
    model_path = smoke_root / "one_epoch_smoke.joblib"
    receipt_path = PROJECT_ROOT / f"data/08_reporting/a0_target_experiment_smoke/{RUN_ID}/{objective}/{year}/receipt.json"
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError("smoke output namespace already exists; refusing to overwrite")
    from src.models.roster_optional import RosterModel
    from src.models.roster_optional import subset as roster_subset
    from src.models.temporal_training import HistoryNormalizer
    labels = frame.y.to_numpy(dtype=np.float64)
    best_of = frame.best_of.to_numpy(dtype=np.int64)
    bo_index = (best_of - 1) // 2
    baseline = RosterModel("linear").fit(roster_subset(data["raw"], train), labels[train], best_of[train])
    normalizer = HistoryNormalizer().fit(data["history"], train)
    model = backend["nonlinear_history"].NonlinearHistoryModel(baseline, normalizer, seed=seed, epochs=1)
    fit_beta_target_on_device(
        model, data["raw"], data["history"], None, train, bo_index,
        data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
        device=device, objective=objective,
    )
    model.main_objective = objective
    if len(model.train_losses) != 1 or not np.isfinite(model.train_losses + model.auxiliary_losses).all():
        raise ValueError("one-epoch smoke did not produce finite main and auxiliary losses")
    _verify_input_hashes(data["input_hashes"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    saved_model = joblib.load(model_path)
    if (
        type(saved_model) is not type(model)
        or saved_model.seed != seed
        or saved_model.epochs != 1
        or saved_model.main_objective != objective
        or _state_hash(saved_model) != _state_hash(model)
    ):
        raise ValueError("saved one-epoch smoke checkpoint does not match its objective contract")
    inference_indices = np.concatenate([
        test[frame.iloc[test].best_of.to_numpy(dtype=np.int64) == best_of][:3]
        for best_of in (1, 3, 5)
        if np.any(frame.iloc[test].best_of.to_numpy(dtype=np.int64) == best_of)
    ])
    if not len(inference_indices):
        raise ValueError("saved smoke inference has no annual TEST rows")
    smoke_logits = saved_model.map_logits(
        data["raw"], data["history"], None, inference_indices, bo_index, True, True,
    )
    if not np.isfinite(smoke_logits).all():
        raise ValueError("saved one-epoch smoke inference produced nonfinite logits")
    _verify_input_hashes(data["input_hashes"])
    receipt = {
        "status": "UNSCORED_ONE_EPOCH_SMOKE",
        "scoring_eligible": False,
        "contract": contract,
        "checkpoint_sha256": _sha256(model_path),
        "main_loss": model.train_losses,
        "auxiliary_loss": model.auxiliary_losses,
        "main_objective": objective,
        "seed": seed,
        "year": year,
        "train_rows": len(train),
        "cal_rows_not_used_for_smoke": len(cal),
        "test_rows_not_scored": len(test),
        "saved_inference": {
            "rows": int(len(inference_indices)),
            "logits_sha256": hashlib.sha256(np.asarray(smoke_logits).tobytes()).hexdigest(),
        },
    }
    _write_json(receipt_path, receipt)
    stored_receipt = json.loads(receipt_path.read_text())
    if (
        stored_receipt.get("contract") != contract
        or stored_receipt.get("checkpoint_sha256") != _sha256(model_path)
        or stored_receipt.get("scoring_eligible") is not False
        or stored_receipt.get("saved_inference", {}).get("rows") != len(inference_indices)
    ):
        raise ValueError("saved one-epoch smoke receipt does not match its execution contract")
    return _json_safe(receipt)
