"""Prequential native-A0 neural refits under the frozen cadence/decay protocol.

This module deliberately owns only the native neural expert.  It does not fit
or replace the annual full-A0 mixture and never consumes bookmaker data.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTINGS = ("uniform", "h365")
MODES = ("full", "no_w20", "no_organization")
MODE_FLAGS = ((True, True), (False, True), (False, False))
SEEDS = (20260910, 20260911, 20260912)
EPOCHS = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (float,)):
        if not math.isfinite(value):
            raise ValueError("nonfinite value cannot be written to an artifact receipt")
        return value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


@contextmanager
def _artifact_bundle(path: Path):
    """Publish an artifact and its receipt together, never an incomplete pair."""
    destination = path.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace an artifact bundle: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        staged = staging / path.name
        yield staged
        if not staged.is_file() or not staged.with_suffix(staged.suffix + ".receipt.json").is_file():
            raise ValueError("artifact bundle requires both data and receipt before publication")
        if destination.exists():
            raise FileExistsError(f"artifact bundle appeared during publication: {destination}")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _contract_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ids_digest(values: Any) -> str:
    ids = [str(value) for value in values]
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _date_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise ValueError(f"native refit data is missing required {name}")
    parsed = pd.to_datetime(frame[name], errors="raise")
    if parsed.isna().any():
        raise ValueError(f"native refit data contains null {name}")
    return parsed.dt.normalize()


def origin_train_indices(frame: pd.DataFrame, origin: str | pd.Timestamp) -> np.ndarray:
    """Return the full TRAIN history with event and release strictly before origin."""
    cutoff = pd.Timestamp(origin).normalize()
    dates = _date_column(frame, "date")
    releases = _date_column(frame, "result_day")
    return np.flatnonzero(((dates < cutoff) & (releases < cutoff)).to_numpy()).astype(np.int64)


def training_row_weights(
    frame: pd.DataFrame,
    indices: np.ndarray,
    origin: str | pd.Timestamp,
    half_life_days: float | None = None,
) -> np.ndarray:
    """Build globally normalized event-age weights for the complete TRAIN rows."""
    ix = np.asarray(indices, dtype=np.int64)
    if ix.ndim != 1 or not len(ix) or (ix < 0).any() or (ix >= len(frame)).any():
        raise ValueError("TRAIN indices must be a nonempty in-range vector")
    if len(np.unique(ix)) != len(ix):
        raise ValueError("TRAIN indices must not contain duplicate rows")
    cutoff = pd.Timestamp(origin).normalize()
    dates = _date_column(frame, "date").iloc[ix]
    if "result_day" in frame:
        releases = _date_column(frame, "result_day").iloc[ix]
        if not (releases < cutoff).all():
            raise ValueError("training result releases must be strictly before origin")
    age = (cutoff - dates).dt.days.to_numpy(dtype=np.float64)
    if not np.isfinite(age).all() or np.any(age <= 0):
        raise ValueError("training target event dates must be strictly before origin with positive age")
    if half_life_days is None:
        return np.ones(len(ix), dtype=np.float64)
    half_life = float(half_life_days)
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("half-life must be a finite positive number of days")
    raw = np.exp2(-age / half_life)
    mean = float(raw.mean())
    if not math.isfinite(mean) or mean <= 0 or not np.isfinite(raw).all():
        raise ValueError("event-age weights cannot be normalized")
    result = raw / mean
    if not np.isfinite(result).all() or np.any(result <= 0) or not np.isclose(result.mean(), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("TRAIN row weights failed global normalization")
    return result


def calibration_indices(
    frame: pd.DataFrame,
    source_cutoffs: Any,
    origin: str | pd.Timestamp,
    window_months: int = 12,
) -> np.ndarray:
    """Select released, prequential OOS rows from a trailing calendar-month window."""
    if int(window_months) != window_months or window_months <= 0:
        raise ValueError("calibration window must be a positive whole number of months")
    cutoff = pd.Timestamp(origin).normalize()
    dates = _date_column(frame, "date")
    releases = _date_column(frame, "result_day")
    fit_values = pd.Series(source_cutoffs)
    if len(fit_values) != len(frame):
        raise ValueError("source fit cutoffs must align with the canonical frame")
    fit_dates = pd.to_datetime(fit_values, errors="coerce").dt.normalize()
    window_start = cutoff - pd.DateOffset(months=int(window_months))
    candidate = (dates >= window_start) & (dates < cutoff) & (releases < cutoff)
    candidate_ix = np.flatnonzero(candidate.to_numpy())
    if not len(candidate_ix):
        return candidate_ix.astype(np.int64)
    source = fit_dates.iloc[candidate_ix]
    target = dates.iloc[candidate_ix]
    if source.isna().any():
        raise ValueError("eligible CAL rows are missing an out-of-sample source model fit origin")
    if (source.to_numpy() > target.to_numpy()).any():
        raise ValueError("CAL source fit origin is after its target date; forecast is not out-of-sample")
    if (source >= cutoff).any():
        raise ValueError("CAL source fit origin must be strictly before calibration origin")
    return candidate_ix.astype(np.int64)


def _protocol(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    protocol_path = Path(path).expanduser().resolve(strict=True)
    digest = _sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run_id = protocol.get("run_id")
    safe_run_id = isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is not None
    expected_origins = pd.date_range("2023-01-01", "2026-09-01", freq="MS").strftime("%Y-%m-%d").tolist()
    if (
        protocol.get("schema") != "native_neural_cadence_decay_v1"
        or protocol.get("status") != "FROZEN_BEFORE_TRAINING_AND_CANDIDATE_SCORING"
        or not safe_run_id
        or protocol.get("model_identity") != "Native A0 neural expert only; unchanged annual full Causal A0 is external reference. No non-neural expert or mixture objective changes; this is not an assembled full-A0 replacement."
        or protocol.get("origins") != expected_origins
        or protocol.get("warmup_year") != 2023
        or protocol.get("evaluation_years") != [2024, 2025, 2026]
        or protocol.get("expected_canonical_rows") != 11550
    ):
        raise ValueError("protocol does not match the frozen native neural cadence/decay experiment")
    arms = protocol.get("arms", {})
    expected_arms = {
        "A": {"cadence": "annual", "weighting": "uniform"},
        "B": {"cadence": "annual", "weighting": "h365"},
        "C": {"cadence": "monthly", "weighting": "uniform"},
        "D": {"cadence": "monthly", "weighting": "h365"},
    }
    if arms != expected_arms:
        raise ValueError("frozen protocol factorial arms differ from the native refit contract")
    training = protocol.get("training", {})
    if (
        training.get("seeds") != list(SEEDS) or training.get("epochs") != EPOCHS
        or training.get("batch_size") != 128 or training.get("fresh_initialization") is not True
        or training.get("shuffle") is not False or training.get("device") != "cpu"
        or training.get("threads_per_process") != 2
        or training.get("main_objective") != "native stopped-series Beta terminal-score NLL; learned eta included in weighted main loss"
        or training.get("auxiliary_coefficient") != 0.1
        or training.get("half_life_days") != 365
        or training.get("optimizer") != {
            "name": "AdamW", "lr": 0.001, "network_weight_decay": 0.01,
            "eta_weight_decay": 0.0, "gradient_clip": 5,
        }
    ):
        raise ValueError("frozen native training settings are unsupported")
    calibration = protocol.get("calibration", {})
    if (
        calibration.get("cadence") != "monthly for all arms" or calibration.get("window_months") != 12
        or calibration.get("min_rows") != 200
        or calibration.get("objective") != "Native joint three-seed stopped-series Beta calibration independently for all3optional-context modes, preserving gate routing"
    ):
        raise ValueError("frozen monthly calibration settings are unsupported")
    if protocol.get("forecast", {}).get("modes") != list(MODES):
        raise ValueError("frozen forecast modes differ from the native refit contract")
    return protocol_path, protocol, digest


def _research_root(research_root: str | Path | None) -> Path:
    from src.models.a0_target_experiment import resolve_research_root

    return resolve_research_root(research_root)


def _load_environment(root: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    from src.models.a0_uncertainty import _activate_archive_backend, _load_training_data

    backend, backend_hashes = _activate_archive_backend(root)
    data = _load_training_data(root)
    frame = data["frame"]
    required = ("golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y", "feature_source_max_day")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"corrected-bank metadata lacks native canonical fields: {missing}")
    if frame.golgg_match_id.astype(str).duplicated().any() or frame.golgg_match_id.isna().any():
        raise ValueError("corrected-bank target identities are null or duplicated")
    feature_day = pd.to_datetime(frame["feature_source_max_day"], errors="raise").dt.normalize()
    date = _date_column(frame, "date")
    dated_features = feature_day.notna()
    if not (feature_day[dated_features] < date[dated_features]).all():
        raise ValueError("corrected feature history is not strictly released before its target date")
    evaluation_rows = date.dt.year.isin((2024, 2025, 2026))
    if feature_day[evaluation_rows].isna().any() or not (feature_day[evaluation_rows] < date[evaluation_rows]).all():
        raise ValueError("canonical evaluation target lacks strictly prior corrected feature history")
    frame["feature_source_max_day"] = feature_day.dt.strftime("%Y-%m-%d")
    if not frame.best_of.isin([1, 3, 5]).all() or not frame.y.isin([0, 1]).all():
        raise ValueError("corrected-bank canonical outcomes contain unsupported best-of or labels")
    return data, {**backend_hashes, **_project_source_hashes()}, backend


def _project_source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "src/models/a0_native_refit.py",
        PROJECT_ROOT / "src/models/a0_target_objectives.py",
        PROJECT_ROOT / "src/models/a0_target_experiment.py",
        PROJECT_ROOT / "src/models/a0_uncertainty.py",
        PROJECT_ROOT / "src/models/a0_accelerator.py",
        PROJECT_ROOT / "scripts/train_a0_uncertainty.py",
    )
    return {f"project:{path.relative_to(PROJECT_ROOT).as_posix()}": _sha256(path) for path in paths}


@lru_cache(maxsize=1)
def _versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib")
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in names}}


def _run_paths(run_id: str, weighting: str) -> dict[str, Path]:
    base_model = PROJECT_ROOT / "data/06_models/a0_native_refit" / run_id / weighting
    base_output = PROJECT_ROOT / "data/07_model_output/a0_native_refit" / run_id / weighting
    base_report = PROJECT_ROOT / "data/08_reporting/a0_native_refit" / run_id / weighting
    return {
        "models": base_model,
        "origins": base_output / "origins",
        "output": base_output,
        "report": base_report,
    }


def _origin_month_indices(frame: pd.DataFrame, origin: str) -> np.ndarray:
    start = pd.Timestamp(origin)
    stop = start + pd.offsets.MonthBegin(1)
    dates = _date_column(frame, "date")
    return np.flatnonzero(((dates >= start) & (dates < stop)).to_numpy()).astype(np.int64)


def _forecast_indices(frame: pd.DataFrame, origin: str, eval_years: set[int]) -> np.ndarray:
    month = _origin_month_indices(frame, origin)
    start = pd.Timestamp(origin)
    if start.month == 1:
        end = start + pd.DateOffset(years=1)
        dates = _date_column(frame, "date")
        annual = np.flatnonzero(((dates >= start) & (dates < end)).to_numpy())
        return np.unique(np.concatenate((month, annual))).astype(np.int64)
    return month


def _weight_stats(weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(weights, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "effective_sample_size": float(values.sum() ** 2 / np.square(values).sum()),
    }


def _torch_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.network.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(model.eta.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def _dropout_receipt(model: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {"class": module.__class__.__name__, "p": float(module.p)}
        for name, module in model.network.named_modules()
        if "Dropout" in module.__class__.__name__ and hasattr(module, "p")
    }


def _atomic_checkpoint(path: Path, model: Any, contract: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    with _artifact_bundle(path) as staged:
        with staged.open("xb") as stream:
            joblib.dump(model, stream)
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "contract": _json_safe(contract),
            "contract_sha256": _contract_digest(contract),
            "checkpoint_sha256": _sha256(staged),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "details": _json_safe(details),
        }
        _write_json_exclusive(staged.with_suffix(staged.suffix + ".receipt.json"), receipt)
    return receipt


def _restore_checkpoint(path: Path, contract: dict[str, Any], neural_type: type, seed: int):
    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"native checkpoint and its receipt are both required: {path}")
    receipt = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        receipt.get("contract") != _json_safe(contract)
        or receipt.get("contract_sha256") != _contract_digest(contract)
        or receipt.get("checkpoint_sha256") != _sha256(path)
    ):
        raise ValueError(f"native checkpoint contract or checksum differs: {path}")
    model = joblib.load(path)
    if (
        type(model) is not neural_type or int(model.seed) != int(seed)
        or int(model.epochs) != EPOCHS or getattr(model, "main_objective", None) != "terminal"
        or len(model.train_losses) != EPOCHS or len(model.auxiliary_losses) != EPOCHS
        or len(model.epoch_diagnostics) != EPOCHS
        or not np.isfinite(model.train_losses + model.auxiliary_losses).all()
    ):
        raise ValueError(f"native checkpoint class, objective, seed, epoch, or loss contract differs: {path}")
    return model, receipt




def _save_forecast(path: Path, contract: dict[str, Any], indices: np.ndarray, frame: pd.DataFrame, logits: np.ndarray, rho: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    ids = frame.iloc[indices].golgg_match_id.astype(str).to_numpy(dtype=str)
    with _artifact_bundle(path) as staged:
        with staged.open("xb") as stream:
            np.savez_compressed(
                stream,
                indices=np.asarray(indices, dtype=np.int64),
                ids=ids,
                logits=np.asarray(logits, dtype=np.float64),
                rho=np.asarray(rho, dtype=np.float64),
                seed_probabilities=np.asarray(probabilities, dtype=np.float64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "contract": _json_safe(contract),
            "contract_sha256": _contract_digest(contract),
            "forecast_sha256": _sha256(staged),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(indices)),
        }
        _write_json_exclusive(staged.with_suffix(staged.suffix + ".receipt.json"), receipt)
    return receipt


def _load_forecast(path: Path, contract: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"required native OOS forecast or receipt is missing: {path}")
    receipt = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        receipt.get("contract") != _json_safe(contract)
        or receipt.get("contract_sha256") != _contract_digest(contract)
    ):
        raise ValueError(f"forecast artifact contract differs: {path}")
    if receipt.get("forecast_sha256") != _sha256(path):
        raise ValueError(f"forecast checksum differs from receipt: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        result = {name: artifact[name] for name in artifact.files}
    if (
        result["logits"].shape != (len(result["indices"]), 3, 3)
        or result["seed_probabilities"].shape != (len(result["indices"]), 3, 3)
        or result["rho"].shape != (3,)
        or len(result["ids"]) != len(result["indices"])
        or not np.isfinite(result["logits"]).all()
        or not np.isfinite(result["seed_probabilities"]).all()
        or np.any((result["seed_probabilities"] <= 0) | (result["seed_probabilities"] >= 1))
    ):
        raise ValueError(f"native forecast arrays are malformed: {path}")
    return result, receipt


def _source_origin(date: Any, cadence: str) -> str:
    stamp = pd.Timestamp(date)
    if cadence == "annual":
        return f"{stamp.year:04d}-01-01"
    if cadence == "monthly":
        return stamp.to_period("M").start_time.strftime("%Y-%m-%d")
    raise ValueError("cadence must be annual or monthly")


def _training_contract(
    protocol: dict[str, Any], protocol_hash: str, source_hashes: dict[str, str],
    data: dict[str, Any], weighting: str, origin: str, train: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    frame = data["frame"]
    train_frame = frame.iloc[train]
    return {
        "schema": "native_refit_neural_checkpoint_v1",
        "run_id": protocol["run_id"],
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hashes,
        "input_sha256": data["input_hashes"],
        "versions": _versions(),
        "weighting": weighting,
        "fit_origin": origin,
        "objective": "terminal",
        "seed": None,
        "epochs": EPOCHS,
        "device": "cpu",
        "threads": 2,
        "train_rows": int(len(train)),
        "train_ids_sha256": _ids_digest(train_frame.golgg_match_id),
        "train_membership_sha256": hashlib.sha256(np.asarray(train, dtype=np.int64).tobytes()).hexdigest(),
        "train_max_event_date": str(train_frame.date.max()),
        "train_max_release_date": str(train_frame.result_day.max()),
        "row_weights_sha256": hashlib.sha256(np.asarray(weights, dtype=np.float64).tobytes()).hexdigest(),
        "weight_statistics": _weight_stats(weights),
        "fresh_initialization": True,
        "optimizer": protocol["training"]["optimizer"],
        "batch_size": 128,
        "shuffle": False,
        "dropout_seed": "per-seed NumPy PCG64; native CoherentModel torch seed",
        "auxiliary_coefficient": 0.1,
    }


def _fit_baseline_normalizer(backend: dict[str, Any], data: dict[str, Any], indices: np.ndarray):
    from src.models.roster_optional import RosterModel, subset
    from src.models.temporal_training import HistoryNormalizer

    frame = data["frame"]
    labels = frame.y.to_numpy(dtype=np.float64)
    best_of = frame.best_of.to_numpy(dtype=np.int64)
    baseline = RosterModel("linear").fit(subset(data["raw"], indices), labels[indices], best_of[indices])
    normalizer = HistoryNormalizer().fit(data["history"], indices)
    return baseline, normalizer


def _calculate_forecast(
    models: list[Any], data: dict[str, Any], indices: np.ndarray, backend: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = data["frame"]
    bo = frame.best_of.to_numpy(dtype=np.int64)
    bo_index = (bo - 1) // 2
    logits = np.empty((len(indices), 3, 3), dtype=np.float64)
    rho = np.empty(3, dtype=np.float64)
    probabilities = np.empty_like(logits)
    terminal_logs = backend["beta_terminal"].terminal_logs
    for seed_index, model in enumerate(models):
        rho[seed_index] = float(0.5 * __import__("torch").sigmoid(model.eta.detach()).cpu())
        for mode_index, (w20, organization) in enumerate(MODE_FLAGS):
            values = np.asarray(model.map_logits(
                data["raw"], data["history"], None, indices, bo_index, w20, organization,
            ), dtype=np.float64)
            if values.shape != (len(indices),) or not np.isfinite(values).all():
                raise ValueError("native saved model returned malformed map logits")
            logits[:, mode_index, seed_index] = values
            log_scores = terminal_logs(values, bo[indices], rho[seed_index])
            probabilities[:, mode_index, seed_index] = np.exp(logsumexp(log_scores[:, :3], axis=1))
    if not np.isfinite(rho).all() or np.any((rho <= 0) | (rho >= 0.5)):
        raise ValueError("native learned dependence parameter is outside (0, 0.5)")
    return logits, rho, probabilities


def train_weighting(
    protocol_path: str | Path,
    weighting: str,
    research_root: str | Path | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Train every frozen monthly origin for one weighting and export raw OOS rows."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    started = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    protocol_file, protocol, protocol_hash = _protocol(protocol_path)
    root = _research_root(research_root)
    from src.models import a0_uncertainty as native

    native._configure_training_device("cpu", 2)
    data, source_hashes, backend = _load_environment(root)
    frame = data["frame"]
    canonical = frame.loc[pd.to_datetime(frame.date).dt.year.isin(protocol["evaluation_years"])]
    if len(canonical) != protocol["expected_canonical_rows"]:
        raise ValueError(f"corrected bank has {len(canonical)} evaluation rows, expected {protocol['expected_canonical_rows']}")
    paths = _run_paths(protocol["run_id"], weighting)
    if not resume and any(path.exists() and any(path.rglob("*")) for path in (paths["models"], paths["origins"], paths["report"])):
        raise FileExistsError("native weighting namespace is not empty; pass --resume only for verified artifacts")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    origins = protocol["origins"]
    eval_years = set(protocol["evaluation_years"])
    from src.models.a0_target_objectives import fit_beta_target_on_device

    training_receipts = []
    forecast_receipts = []
    neural_type = backend["nonlinear_history"].NonlinearHistoryModel
    bo_index = (frame.best_of.to_numpy(dtype=np.int64) - 1) // 2
    for origin in origins:
        origin_started = time.monotonic()
        train = origin_train_indices(frame, origin)
        if not len(train):
            raise ValueError(f"empty strictly eligible TRAIN at origin {origin}")
        half_life = None if weighting == "uniform" else protocol["training"]["half_life_days"]
        full_weights = training_row_weights(frame, train, origin, half_life_days=half_life)
        weight_stats = _weight_stats(full_weights)
        train_contract_base = _training_contract(
            protocol, protocol_hash, source_hashes, data, weighting, origin, train, full_weights,
        )
        origin_dir = paths["models"] / origin
        origin_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_paths = [origin_dir / f"seed_{seed}" / "model.joblib" for seed in SEEDS]
        origin_models: list[Any] = [None] * len(SEEDS)

        # A complete saved member carries the shared unweighted baseline and
        # normalizer, so resumptions never refit either against a different cohort.
        if resume:
            for seed_index, seed in enumerate(SEEDS):
                checkpoint = checkpoint_paths[seed_index]
                sidecar = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
                if checkpoint.exists() or sidecar.exists():
                    contract = {**train_contract_base, "seed": int(seed)}
                    origin_models[seed_index], _ = _restore_checkpoint(checkpoint, contract, neural_type, seed)
        baseline_normalizer = next((model for model in origin_models if model is not None), None)
        if baseline_normalizer is None:
            baseline, normalizer = _fit_baseline_normalizer(
                backend, data, train,
            )
        else:
            baseline, normalizer = baseline_normalizer.baseline, baseline_normalizer.normalizer
        source_receipts = []
        for seed_index, seed in enumerate(SEEDS):
            checkpoint = checkpoint_paths[seed_index]
            contract = {**train_contract_base, "seed": int(seed)}
            if origin_models[seed_index] is None:
                if checkpoint.exists() or checkpoint.with_suffix(checkpoint.suffix + ".receipt.json").exists():
                    raise FileExistsError(f"partial native checkpoint requires explicit valid resume: {checkpoint}")
                model = neural_type(baseline, normalizer, seed=seed, epochs=EPOCHS)
                model.main_objective = "terminal"
                initialization_hash = _torch_state_hash(model)
                row_weights = full_weights if weighting == "h365" else None
                fit_started = time.monotonic()

                def progress(epoch, main_loss, auxiliary_loss, diagnostic):
                    print(json.dumps({
                        "event": "native_refit_epoch", "weighting": weighting,
                        "origin": origin, "seed": seed, "epoch": epoch,
                        "main_loss": float(main_loss), "auxiliary_loss": float(auxiliary_loss),
                        "elapsed_fit_seconds": time.monotonic() - fit_started,
                    }, allow_nan=False), flush=True)

                fit_beta_target_on_device(
                    model, data["raw"], data["history"], None, train, bo_index,
                    data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
                    progress=progress, device="cpu", objective="terminal", row_weights=row_weights,
                )
                model.main_objective = "terminal"
                fit_seconds = time.monotonic() - fit_started
                if (
                    len(model.train_losses) != EPOCHS or len(model.auxiliary_losses) != EPOCHS
                    or len(model.epoch_diagnostics) != EPOCHS
                    or not np.isfinite(model.train_losses + model.auxiliary_losses).all()
                ):
                    raise ValueError(f"native terminal fit did not complete all epochs at {origin}, seed {seed}")
                details = {
                    "initial_state_sha256": initialization_hash,
                    "final_state_sha256": _torch_state_hash(model),
                    "fit_seconds": fit_seconds,
                    "train_losses": list(model.train_losses),
                    "auxiliary_losses": list(model.auxiliary_losses),
                    "epoch_diagnostics": model.epoch_diagnostics,
                    "dropout_rng_final_state": getattr(model, "dropout_rng_final_state", None),
                    "dropout_layers": _dropout_receipt(model),
                    "learned_eta": float(model.eta.detach()),
                    "learned_rho": float(0.5 * __import__("torch").sigmoid(model.eta.detach()).cpu()),
                    "weight_statistics": weight_stats,
                }
                receipt = _atomic_checkpoint(checkpoint, model, contract, details)
                origin_models[seed_index] = model
                print(json.dumps({
                    "event": "native_refit_fit_complete", "weighting": weighting,
                    "origin": origin, "seed": seed, "fit_seconds": fit_seconds,
                    "train_rows": len(train), "weight_ess": weight_stats["effective_sample_size"],
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                }, allow_nan=False), flush=True)
            else:
                receipt_path = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("contract") != _json_safe(contract):
                    raise ValueError(f"resumed checkpoint contract differs: {checkpoint}")
            source_receipts.append({
                "seed": int(seed),
                "path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "receipt_path": str(checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")),
                "fit_seconds": None if origin_models[seed_index] is not None and resume and checkpoint.exists() and not receipt.get("details", {}).get("fit_seconds") else receipt.get("details", {}).get("fit_seconds"),
                "rho": float(0.5 * __import__("torch").sigmoid(origin_models[seed_index].eta.detach()).cpu()),
            })

        forecast_indices = _forecast_indices(frame, origin, eval_years)
        if not len(forecast_indices):
            raise ValueError(f"no target rows available for origin {origin}")
        forecast_path = paths["origins"] / origin / "forecast.npz"
        forecast_contract = {
            "schema": "native_refit_origin_forecast_v1", "run_id": protocol["run_id"],
            "protocol_sha256": protocol_hash, "source_sha256": source_hashes,
            "input_sha256": data["input_hashes"], "weighting": weighting,
            "versions": _versions(),
            "source_fit_origin": origin,
            "forecast_indices_sha256": hashlib.sha256(np.asarray(forecast_indices, dtype=np.int64).tobytes()).hexdigest(),
            "forecast_ids_sha256": _ids_digest(frame.iloc[forecast_indices].golgg_match_id),
            "rows": int(len(forecast_indices)), "modes": list(MODES), "seed_order": list(SEEDS),
            "checkpoint_sha256": [record["checkpoint_sha256"] for record in source_receipts],
        }
        if resume and (forecast_path.exists() or forecast_path.with_suffix(".npz.receipt.json").exists()):
            forecast, forecast_receipt = _load_forecast(forecast_path, forecast_contract)
            if not np.array_equal(forecast["indices"], forecast_indices) or not np.array_equal(
                forecast["ids"], frame.iloc[forecast_indices].golgg_match_id.astype(str).to_numpy(dtype=str),
            ):
                raise ValueError(f"resumed forecast membership differs at {origin}")
        else:
            if forecast_path.exists() or forecast_path.with_suffix(".npz.receipt.json").exists():
                raise FileExistsError(f"partial forecast requires valid resume: {forecast_path}")
            logits, rho, probabilities = _calculate_forecast(origin_models, data, forecast_indices, backend)
            forecast_receipt = _save_forecast(
                forecast_path, forecast_contract, forecast_indices, frame, logits, rho, probabilities,
            )
        forecast_receipts.append({
            "origin": origin, "path": str(forecast_path),
            "forecast_sha256": forecast_receipt["forecast_sha256"],
            "rows": int(len(forecast_indices)),
            "ids_sha256": _ids_digest(frame.iloc[forecast_indices].golgg_match_id),
            "created_at_utc": forecast_receipt.get("created_at_utc"),
        })
        training_receipts.append({
            "origin": origin, "train_rows": int(len(train)),
            "train_ids_sha256": train_contract_base["train_ids_sha256"],
            "train_max_event_date": train_contract_base["train_max_event_date"],
            "train_max_release_date": train_contract_base["train_max_release_date"],
            "row_weights_sha256": train_contract_base["row_weights_sha256"],
            "weight_statistics": weight_stats, "members": source_receipts,
            "origin_elapsed_seconds": time.monotonic() - origin_started,
        })
        print(json.dumps({
            "event": "native_refit_origin_complete", "weighting": weighting,
            "origin": origin, "elapsed_seconds": time.monotonic() - origin_started,
            "train_rows": len(train), "forecast_rows": len(forecast_indices),
        }, allow_nan=False), flush=True)

    if _sha256(protocol_file) != protocol_hash:
        raise ValueError("frozen protocol changed while native training was running")
    # The expensive, immutable bank inputs are hashed once by _load_training_data
    # at process start.  Do not repeatedly reread the multi-gigabyte input arrays.
    receipt_path = paths["report"] / "training_receipt.json"
    receipt = {
        "status": "COMPLETE", "model_identity": protocol["model_identity"],
        "run_id": protocol["run_id"], "weighting": weighting,
        "protocol_path": str(protocol_file), "protocol_sha256": protocol_hash,
        "research_root": str(root), "source_sha256": source_hashes,
        "input_sha256": data["input_hashes"], "versions": _versions(),
        "origins": training_receipts, "forecasts": forecast_receipts,
        "started_at_utc": started_at_utc,
        "elapsed_seconds": time.monotonic() - started,
    }
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"completed training receipt already exists: {receipt_path}")
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") != "COMPLETE" or previous.get("protocol_sha256") != protocol_hash
            or previous.get("weighting") != weighting or previous.get("input_sha256") != data["input_hashes"]
            or previous.get("source_sha256") != source_hashes or previous.get("versions") != _versions()
        ):
            raise ValueError("completed native training receipt differs from this run")
        receipt = previous
    else:
        _write_json_exclusive(receipt_path, receipt)
    return {
        "status": "COMPLETE", "weighting": weighting,
        "training_receipt": str(receipt_path),
        "model_root": str(paths["models"]), "forecast_root": str(paths["origins"]),
        "origins": len(origins), "seed_fits": len(origins) * len(SEEDS),
        "elapsed_seconds": float(time.monotonic() - started),
    }


class _ForecastCache:
    def __init__(self, paths: dict[str, Path], protocol: dict[str, Any], protocol_hash: str,
                 source_hashes: dict[str, str], input_hashes: dict[str, str], frame: pd.DataFrame,
                 weighting: str):
        self.paths = paths
        self.protocol = protocol
        self.protocol_hash = protocol_hash
        self.source_hashes = source_hashes
        self.input_hashes = input_hashes
        self.frame = frame
        self.weighting = weighting
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    def get(self, origin: str) -> dict[str, np.ndarray]:
        if origin in self.cache:
            return self.cache[origin]
        indices = _forecast_indices(self.frame, origin, set(self.protocol["evaluation_years"]))
        path = self.paths["origins"] / origin / "forecast.npz"
        checkpoints = []
        for seed in SEEDS:
            checkpoint = self.paths["models"] / origin / f"seed_{seed}" / "model.joblib"
            receipt_path = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
            if not checkpoint.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(f"missing required monthly native checkpoint support: {checkpoint}")
            checkpoints.append(_sha256(checkpoint))
        contract = {
            "schema": "native_refit_origin_forecast_v1", "run_id": self.protocol["run_id"],
            "protocol_sha256": self.protocol_hash, "source_sha256": self.source_hashes,
            "input_sha256": self.input_hashes, "weighting": self.weighting,
            "versions": _versions(),
            "source_fit_origin": origin,
            "forecast_indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
            "forecast_ids_sha256": _ids_digest(self.frame.iloc[indices].golgg_match_id),
            "rows": int(len(indices)), "modes": list(MODES), "seed_order": list(SEEDS),
            "checkpoint_sha256": checkpoints,
        }
        value, _ = _load_forecast(path, contract)
        if not np.array_equal(value["indices"], indices):
            raise ValueError(f"forecast rows differ from expected origin membership: {path}")
        expected_ids = self.frame.iloc[indices].golgg_match_id.astype(str).to_numpy(dtype=str)
        if not np.array_equal(value["ids"], expected_ids):
            raise ValueError(f"forecast identities differ from corrected bank: {path}")
        self.cache[origin] = value
        return value


def _verified_training_source(
    data: dict[str, Any], paths: dict[str, Path], protocol: dict[str, Any],
    protocol_hash: str, source_hashes: dict[str, str], weighting: str,
) -> tuple[str, _ForecastCache]:
    """Bind the run manifest to every checkpoint and OOS forecast, not a sample."""
    frame = data["frame"]
    origins = protocol["origins"]
    receipt_path = paths["report"] / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("protocol_sha256") != protocol_hash
        or receipt.get("source_sha256") != source_hashes
        or receipt.get("input_sha256") != data["input_hashes"]
        or receipt.get("versions") != _versions()
        or receipt.get("weighting") != weighting
        or [row.get("origin") for row in receipt.get("origins", [])] != origins
        or [row.get("origin") for row in receipt.get("forecasts", [])] != origins
    ):
        raise ValueError(f"native training manifest is incomplete or mismatched for {weighting}")
    cache = _ForecastCache(paths, protocol, protocol_hash, source_hashes, data["input_hashes"], frame, weighting)
    for position, origin in enumerate(origins):
        train = origin_train_indices(frame, origin)
        half_life = None if weighting == "uniform" else protocol["training"]["half_life_days"]
        weights = training_row_weights(frame, train, origin, half_life_days=half_life)
        expected = _training_contract(protocol, protocol_hash, source_hashes, data, weighting, origin, train, weights)
        recorded = receipt["origins"][position]
        for key in (
            "train_rows", "train_ids_sha256", "train_max_event_date", "train_max_release_date",
            "row_weights_sha256", "weight_statistics",
        ):
            if recorded.get(key) != _json_safe(expected[key]):
                raise ValueError(f"training manifest membership differs at {origin}: {key}")
        members = recorded.get("members", [])
        if [member.get("seed") for member in members] != list(SEEDS):
            raise ValueError(f"training manifest seed membership differs at {origin}")
        for seed, member in zip(SEEDS, members):
            checkpoint = paths["models"] / origin / f"seed_{seed}" / "model.joblib"
            sidecar = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            contract = {**expected, "seed": seed}
            checksum = _sha256(checkpoint)
            if (
                saved.get("contract") != _json_safe(contract)
                or saved.get("contract_sha256") != _contract_digest(contract)
                or saved.get("checkpoint_sha256") != checksum
                or member.get("checkpoint_sha256") != checksum
            ):
                raise ValueError(f"native checkpoint differs from its verified training manifest: {checkpoint}")
        forecast = cache.get(origin)
        recorded_forecast = receipt["forecasts"][position]
        if (
            recorded_forecast.get("forecast_sha256") != _sha256(paths["origins"] / origin / "forecast.npz")
            or recorded_forecast.get("rows") != len(forecast["ids"])
            or recorded_forecast.get("ids_sha256") != _ids_digest(pd.Series(forecast["ids"]))
        ):
            raise ValueError(f"native OOS forecast differs from its training manifest at {origin}")
    return _sha256(receipt_path), cache




def _raw_oos_for_indices(
    indices: np.ndarray, frame: pd.DataFrame, cadence: str, cache: _ForecastCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    logits = np.empty((len(indices), 3, 3), dtype=np.float64)
    rho = np.empty((len(indices), 3), dtype=np.float64)
    seed_prob = np.empty_like(logits)
    origins = []
    groups: dict[str, list[tuple[int, int]]] = {}
    for local_index, global_index in enumerate(indices):
        origin = _source_origin(frame.iloc[global_index].date, cadence)
        groups.setdefault(origin, []).append((local_index, int(global_index)))
        origins.append(origin)
    for source_origin, pairs in groups.items():
        artifact = cache.get(source_origin)
        location = {int(global_index): row for row, global_index in enumerate(artifact["indices"])}
        missing = [global_index for _, global_index in pairs if global_index not in location]
        if missing:
            raise ValueError(f"source forecast {source_origin} lacks {len(missing)} required rows")
        for local_index, global_index in pairs:
            source_row = location[global_index]
            logits[local_index] = artifact["logits"][source_row]
            rho[local_index] = artifact["rho"]
            seed_prob[local_index] = artifact["seed_probabilities"][source_row]
    return logits, rho, seed_prob, origins


def _timestamp_max(values: pd.Series) -> str:
    if len(values) == 0:
        raise ValueError("cannot record maximum of empty causal membership")
    return pd.to_datetime(values, errors="raise").max().strftime("%Y-%m-%d")


def _identity_columns(frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    columns = ["golgg_match_id", "team1_id", "team2_id", "date", "result_day", "feature_source_max_day", "best_of", "y"]
    return frame.iloc[indices][columns].reset_index(drop=True).copy()


def _fit_monthly_calibrator(
    origin: str, cal_indices: np.ndarray, cal_logits: np.ndarray,
    data: dict[str, Any], backend: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    terminal = backend["beta_terminal"]
    frame = data["frame"]
    parameters = np.empty((3, 2), dtype=np.float64)
    records = []
    for mode_index, mode in enumerate(MODES):
        record = terminal.fit_joint(
            cal_logits[:, mode_index, :], data["wins_a"][cal_indices], data["wins_b"][cal_indices], "beta",
        )
        parameters[mode_index] = (record["scale"], record["rho"])
        records.append({"mode": mode, **record})
    return parameters, records


def _calibrated_modes(
    logits: np.ndarray, parameters: np.ndarray, gates: np.ndarray,
    best_of: np.ndarray, backend: dict[str, Any],
) -> np.ndarray:
    result = np.empty((len(logits), 3), dtype=np.float64)
    for mode_index, mode in enumerate(MODES):
        distribution = backend["beta_calibration"].calibrated_distribution(
            logits, parameters, gates, best_of, mode_index,
        )
        result[:, mode_index] = np.exp(logsumexp(distribution[:, :3], axis=1))
    if not np.isfinite(result).all() or np.any((result <= 0) | (result >= 1)):
        raise ValueError("monthly calibrated native series probability is invalid")
    return result


def _expected_calibration_contract(
    protocol: dict[str, Any], protocol_hash: str, weighting: str, cadence: str,
    source_hashes: dict[str, str], input_hashes: dict[str, str], origin: str,
    training_source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "native_refit_monthly_calibration_v1", "run_id": protocol["run_id"],
        "protocol_sha256": protocol_hash, "weighting": weighting, "cadence": cadence,
        "origin": origin, "window_months": 12, "min_rows": 200,
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "versions": _versions(),
        "training_source_sha256": training_source_sha256,
        "objective": "joint_three_seed_native_beta_all_modes",
    }


def _write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Parquet writers require a path on some backends; use a unique sibling and
    # hard-link into place so the final name is created exclusively.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_calibration(path: Path, contract: dict[str, Any], parameters: np.ndarray,
                      records: list[dict[str, Any]], details: dict[str, Any]) -> dict[str, Any]:
    with _artifact_bundle(path) as staged:
        with staged.open("xb") as stream:
            np.savez_compressed(stream, parameters=np.asarray(parameters, dtype=np.float64))
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "contract": _json_safe(contract), "contract_sha256": _contract_digest(contract),
            "calibration_sha256": _sha256(staged), "parameters": _json_safe(parameters),
            "records": _json_safe(records), "details": _json_safe(details),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_exclusive(staged.with_suffix(staged.suffix + ".receipt.json"), receipt)
    return receipt


def _load_calibration(path: Path, contract: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    sidecar = path.with_suffix(path.suffix + ".receipt.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"required monthly native calibration support is missing: {path}")
    receipt = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        receipt.get("contract") != _json_safe(contract)
        or receipt.get("contract_sha256") != _contract_digest(contract)
        or receipt.get("calibration_sha256") != _sha256(path)
    ):
        raise ValueError(f"monthly calibration receipt or checksum differs: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        parameters = artifact["parameters"]
    if parameters.shape != (3, 2) or not np.isfinite(parameters).all():
        raise ValueError(f"monthly calibration parameters are malformed: {path}")
    return parameters, receipt


def _saved_inference_check(
    data: dict[str, Any], backend: dict[str, Any], paths: dict[str, Path], protocol: dict[str, Any],
    protocol_hash: str, source_hashes: dict[str, str], weighting: str, cadence: str,
    sample_indices: np.ndarray, raw_forecast_logits: np.ndarray, raw_forecast_probs: np.ndarray,
    candidate: pd.DataFrame, calibration_parameters: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Verify actual saved members against forecasts, modes, swaps, cold start, and exports."""

    frame = data["frame"]
    raw = data["raw"]
    history = data["history"]
    source_origins = [_source_origin(frame.iloc[i].date, cadence) for i in sample_indices]
    sample_raw = {key: np.asarray(value)[sample_indices].copy() for key, value in raw.items()}
    sample_history = {key: np.asarray(value)[sample_indices].copy() for key, value in history.items()}
    before = {
        **{f"raw:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in sample_raw.items()},
        **{f"history:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in sample_history.items()},
    }
    bo = frame.iloc[sample_indices].best_of.to_numpy(dtype=np.int64)
    bo_index = (bo - 1) // 2
    columns: dict[str, np.ndarray] = {}
    checks: dict[str, float] = {}
    passed_inputs_before: dict[str, str] = {}
    passed_inputs_after: dict[str, str] = {}
    neural_type = backend["nonlinear_history"].NonlinearHistoryModel
    for source_origin in sorted(set(source_origins)):
        rows = np.flatnonzero(np.asarray(source_origins, dtype=object) == source_origin)
        model_rows = np.arange(len(rows), dtype=np.int64)
        raw_part = {key: value[rows].copy() for key, value in sample_raw.items()}
        history_part = {key: value[rows].copy() for key, value in sample_history.items()}
        passed_inputs_before.update({
            **{f"{source_origin}:raw:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in raw_part.items()},
            **{f"{source_origin}:history:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in history_part.items()},
        })
        for seed_index, seed in enumerate(SEEDS):
            checkpoint = paths["models"] / source_origin / f"seed_{seed}" / "model.joblib"
            sidecar = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
            if not sidecar.is_file() or not checkpoint.is_file():
                raise FileNotFoundError(f"saved inference requires real native checkpoint: {checkpoint}")
            receipt = json.loads(sidecar.read_text(encoding="utf-8"))
            checkpoint_contract = receipt.get("contract")
            if (
                checkpoint_contract is None or checkpoint_contract.get("run_id") != protocol["run_id"]
                or checkpoint_contract.get("protocol_sha256") != protocol_hash
                or checkpoint_contract.get("weighting") != weighting
                or checkpoint_contract.get("fit_origin") != source_origin
                or checkpoint_contract.get("seed") != seed
                or receipt.get("checkpoint_sha256") != _sha256(checkpoint)
            ):
                raise ValueError(f"saved inference checkpoint provenance differs: {checkpoint}")
            model = joblib.load(checkpoint)
            if type(model) is not neural_type or model.main_objective != "terminal":
                raise ValueError(f"saved inference loaded a non-native-terminal model: {checkpoint}")
            for mode_index, (w20, organization) in enumerate(MODE_FLAGS):
                z = model.map_logits(raw_part, history_part, None, model_rows, bo_index[rows], w20, organization)
                expected_z = raw_forecast_logits[rows, mode_index, seed_index]
                error = float(np.max(np.abs(z - expected_z)))
                checks[f"{source_origin}_{seed}_{MODES[mode_index]}_saved_logit_error"] = error
                columns.setdefault(f"{MODES[mode_index]}_{seed}", np.empty(len(sample_indices), dtype=float))[rows] = z
                swapped_raw = {key: value[:, ::-1].copy() for key, value in raw_part.items()}
                swapped_history = {key: value[:, ::-1].copy() for key, value in history_part.items()}
                reverse = model.map_logits(swapped_raw, swapped_history, None, model_rows, bo_index[rows], w20, organization)
                checks[f"{source_origin}_{seed}_{MODES[mode_index]}_side_swap_error"] = float(np.max(np.abs(z + reverse)))
                rho = float(0.5 * __import__("torch").sigmoid(model.eta.detach()).cpu())
                logs = backend["beta_terminal"].terminal_logs(z, bo[rows], rho)
                probability = np.exp(logsumexp(logs[:, :3], axis=1))
                checks[f"{source_origin}_{seed}_{MODES[mode_index]}_saved_raw_probability_error"] = float(
                    np.max(np.abs(probability - raw_forecast_probs[rows, mode_index, seed_index]))
                )
                reverse_logs = backend["beta_terminal"].terminal_logs(reverse, bo[rows], rho)
                reverse_probability = np.exp(logsumexp(reverse_logs[:, :3], axis=1))
                checks[f"{source_origin}_{seed}_{MODES[mode_index]}_probability_complement_error"] = float(
                    np.max(np.abs(probability + reverse_probability - 1.0))
                )
        passed_inputs_after.update({
            **{f"{source_origin}:raw:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in raw_part.items()},
            **{f"{source_origin}:history:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in history_part.items()},
        })
    if passed_inputs_before != passed_inputs_after:
        raise ValueError("saved native inference mutated arrays passed to the model")
    cold_raw = {key: np.zeros_like(value) for key, value in sample_raw.items()}
    cold_history = {key: np.zeros_like(value) for key, value in sample_history.items()}
    # Every saved checkpoint represented in the sample gets a cold-start check.
    for source_origin in sorted(set(source_origins)):
        rows = np.flatnonzero(np.asarray(source_origins, dtype=object) == source_origin)
        raw_zero = {key: value[rows].copy() for key, value in cold_raw.items()}
        history_zero = {key: value[rows].copy() for key, value in cold_history.items()}
        for seed in SEEDS:
            model = joblib.load(paths["models"] / source_origin / f"seed_{seed}" / "model.joblib")
            for mode_index, (w20, organization) in enumerate(MODE_FLAGS):
                zero_logits = model.map_logits(raw_zero, history_zero, None, np.arange(len(rows), dtype=np.int64), bo_index[rows], w20, organization)
                checks[f"{source_origin}_{seed}_{MODES[mode_index]}_cold_logit_error"] = float(np.max(np.abs(zero_logits)))
                logs = backend["beta_terminal"].terminal_logs(
                    zero_logits, bo[rows], float(0.5 * __import__("torch").sigmoid(model.eta.detach()).cpu()),
                )
                pzero = np.exp(logsumexp(logs[:, :3], axis=1))
                checks[f"{source_origin}_{seed}_{MODES[mode_index]}_cold_probability_error"] = float(np.max(np.abs(pzero - 0.5)))
    for local, global_index in enumerate(sample_indices):
        candidate_row = candidate.iloc[int(np.flatnonzero(candidate.golgg_match_id.astype(str).to_numpy() == str(frame.iloc[global_index].golgg_match_id))[0])]
        params = calibration_parameters[str(candidate_row.calibration_origin)]
        mode_logits = np.stack([
            np.column_stack([columns[f"{mode}_{seed}"][local:local + 1] for seed in SEEDS])
            for mode in MODES
        ], axis=1)
        gates = np.asarray(raw["gates"])[global_index:global_index + 1]
        probs = _calibrated_modes(
            mode_logits, params, gates, np.asarray([bo[local]], dtype=np.int64), backend,
        )[0]
        for mode_index, mode in enumerate(MODES):
            expected = float(candidate_row[f"p_{mode}"])
            checks[f"candidate_{global_index}_{mode}_calibrated_error"] = float(abs(probs[mode_index] - expected))
    after = {
        **{f"raw:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in sample_raw.items()},
        **{f"history:{key}": hashlib.sha256(value.tobytes()).hexdigest() for key, value in sample_history.items()},
    }
    if before != after:
        raise ValueError("saved native inference mutated input arrays")
    if not checks or any(not math.isfinite(value) or value > 2e-6 for value in checks.values()):
        failures = {key: value for key, value in checks.items() if not math.isfinite(value) or value > 2e-6}
        raise ValueError(f"saved native inference verification failed: {failures}")
    return {
        "status": "PASS", "rows": int(len(sample_indices)),
        "best_of": sorted(set(bo.tolist())), "checks": checks,
        "source_fit_origins": sorted(set(source_origins)), "input_mutation": False,
    }


def saved_checkpoint_inference_verification(
    protocol_path: str | Path,
    weighting: str,
    cadence: str,
    research_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exported candidate inference using three real BO1/3/5 rows."""
    if weighting not in WEIGHTINGS or cadence not in ("annual", "monthly"):
        raise ValueError("saved inference requires native weighting and annual/monthly cadence")
    protocol_file, protocol, protocol_hash = _protocol(protocol_path)
    root = _research_root(research_root)
    from src.models import a0_uncertainty as native
    native._configure_training_device("cpu", 2)
    data, source_hashes, backend = _load_environment(root)
    paths = _run_paths(protocol["run_id"], weighting)
    return _verify_exported_arm(
        data, backend, paths, protocol, protocol_hash, source_hashes,
        weighting, cadence,
    )


def _verify_exported_arm(
    data: dict[str, Any], backend: dict[str, Any], paths: dict[str, Path],
    protocol: dict[str, Any], protocol_hash: str, source_hashes: dict[str, str],
    weighting: str, cadence: str,
) -> dict[str, Any]:
    frame = data["frame"]
    candidate_path = paths["output"] / f"{weighting}_{cadence}" / "candidate.parquet"
    sidecar_path = candidate_path.with_suffix(candidate_path.suffix + ".receipt.json")
    if not candidate_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"candidate is required for saved inference verification: {candidate_path}")
    receipt = json.loads(sidecar_path.read_text(encoding="utf-8"))
    candidate = pd.read_parquet(candidate_path)
    training_source_sha256, cache = _verified_training_source(
        data, paths, protocol, protocol_hash, source_hashes, weighting,
    )
    contract = {
        "schema": "native_neural_refit_candidate_v1", "run_id": protocol["run_id"],
        "protocol_sha256": protocol_hash, "weighting": weighting, "cadence": cadence,
        "source_sha256": source_hashes, "input_sha256": data["input_hashes"],
        "model_identity": protocol["model_identity"],
        "canonical_ids_sha256": _ids_digest(candidate.golgg_match_id),
        "training_source_sha256": training_source_sha256, "versions": _versions(),
    }
    if (
        receipt.get("contract") != _json_safe(contract)
        or receipt.get("contract_sha256") != _contract_digest(contract)
        or receipt.get("sha256") != _sha256(candidate_path)
    ):
        raise ValueError("candidate contract or checksum does not match its inference receipt")
    canonical = np.flatnonzero(pd.to_datetime(frame.date).dt.year.isin(protocol["evaluation_years"]).to_numpy())
    expected_ids = frame.iloc[canonical].golgg_match_id.astype(str).to_numpy(dtype=str)
    candidate_ids = candidate.golgg_match_id.astype(str).to_numpy(dtype=str)
    if (
        len(candidate_ids) != len(expected_ids)
        or len(np.unique(candidate_ids)) != len(candidate_ids)
        or not np.array_equal(candidate_ids, expected_ids)
    ):
        raise ValueError("candidate identities do not match the complete canonical evaluation cohort")
    by_bo = []
    for best_of in (1, 3, 5):
        rows = canonical[frame.iloc[canonical].best_of.to_numpy(dtype=np.int64) == best_of]
        if not len(rows):
            raise ValueError(f"canonical targets have no real best-of-{best_of} rows")
        by_bo.append(int(rows[0]))
    sample = np.asarray(by_bo, dtype=np.int64)
    expected_logits, _, expected_probabilities, _ = _raw_oos_for_indices(sample, frame, cadence, cache)
    aggregate_path = PROJECT_ROOT / "data/08_reporting/a0_native_refit" / protocol["run_id"] / "aggregate_receipt.json"
    aggregate_receipt = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        aggregate_receipt.get("status") != "COMPLETE"
        or aggregate_receipt.get("run_id") != protocol["run_id"]
        or aggregate_receipt.get("protocol_sha256") != protocol_hash
        or aggregate_receipt.get("source_sha256") != source_hashes
        or aggregate_receipt.get("input_sha256") != data["input_hashes"]
        or aggregate_receipt.get("versions") != _versions()
        or aggregate_receipt.get("arms", {}).get(f"{weighting}_{cadence}", {}).get("candidate_sha256") != receipt["sha256"]
        or aggregate_receipt.get("arms", {}).get(f"{weighting}_{cadence}", {}).get("candidate_receipt_sha256") != _sha256(sidecar_path)
        or aggregate_receipt.get("canonical_ids_sha256") != _ids_digest(frame.iloc[canonical].golgg_match_id)
    ):
        raise ValueError("aggregate receipt does not match the saved candidate protocol and inputs")
    monthly = aggregate_receipt.get("arms", {}).get(f"{weighting}_{cadence}", {}).get("monthly_calibrations", {})
    if not monthly:
        raise ValueError("aggregate receipt lacks monthly native calibration support")
    params_by_origin = {}
    for origin, membership in monthly.items():
        calibration_path = paths["report"] / "calibrations" / cadence / origin / "calibration.npz"
        calibration_contract = _expected_calibration_contract(
            protocol, protocol_hash, weighting, cadence, source_hashes, data["input_hashes"], origin,
            training_source_sha256,
        )
        parameters, _ = _load_calibration(calibration_path, calibration_contract)
        recorded = np.asarray(membership.get("parameters"), dtype=np.float64)
        if not np.array_equal(parameters, recorded):
            raise ValueError(f"aggregate and calibration artifact differ at {origin}")
        params_by_origin[str(origin)] = parameters
    return _saved_inference_check(
        data, backend, paths, protocol, protocol_hash, source_hashes,
        weighting, cadence, sample, expected_logits, expected_probabilities,
        candidate, params_by_origin,
    )


def aggregate(
    protocol_path: str | Path,
    research_root: str | Path | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Calibrate monthly OOS raw forecasts and export all four canonical arms."""
    started = time.monotonic()
    protocol_file, protocol, protocol_hash = _protocol(protocol_path)
    root = _research_root(research_root)
    from src.models import a0_uncertainty as native
    native._configure_training_device("cpu", 2)
    data, source_hashes, backend = _load_environment(root)
    frame = data["frame"]
    eval_indices = np.flatnonzero(pd.to_datetime(frame.date).dt.year.isin(protocol["evaluation_years"]).to_numpy())
    if len(eval_indices) != protocol["expected_canonical_rows"]:
        raise ValueError(f"canonical corrected-bank cohort has {len(eval_indices)} rows; expected {protocol['expected_canonical_rows']}")
    canonical = frame.iloc[eval_indices].copy().reset_index(drop=True)
    ids = canonical.golgg_match_id.astype(str)
    if ids.duplicated().any():
        raise ValueError("canonical output membership has duplicate IDs")
    actual_origins = protocol["origins"]
    origin_set = set(actual_origins)
    output_arms: dict[str, Any] = {}
    candidate_frames: dict[str, pd.DataFrame] = {}
    calibration_params_by_arm: dict[str, dict[str, np.ndarray]] = {}
    training_sources = {}
    forecast_caches = {}
    for weighting in WEIGHTINGS:
        paths = _run_paths(protocol["run_id"], weighting)
        training_source_sha256, cache = _verified_training_source(
            data, paths, protocol, protocol_hash, source_hashes, weighting,
        )
        training_sources[weighting] = training_source_sha256
        forecast_caches[weighting] = cache
        for cadence in ("annual", "monthly"):
            arm_key = f"{weighting}_{cadence}"
            candidate_path = paths["output"] / arm_key / "candidate.parquet"
            candidate_sidecar = candidate_path.with_suffix(candidate_path.suffix + ".receipt.json")
            if candidate_path.exists() or candidate_sidecar.exists():
                if not resume or not candidate_path.is_file() or not candidate_sidecar.is_file():
                    raise FileExistsError(f"candidate exists without verified resume: {candidate_path}")
                existing = json.loads(candidate_sidecar.read_text(encoding="utf-8"))
                existing_contract = {
                    "schema": "native_neural_refit_candidate_v1", "run_id": protocol["run_id"],
                    "protocol_sha256": protocol_hash, "weighting": weighting, "cadence": cadence,
                    "source_sha256": source_hashes, "input_sha256": data["input_hashes"],
                    "model_identity": protocol["model_identity"],
                    "canonical_ids_sha256": _ids_digest(canonical.golgg_match_id),
                    "training_source_sha256": training_source_sha256, "versions": _versions(),
                }
                existing_frame = pd.read_parquet(candidate_path)
                if (
                    existing.get("contract") != _json_safe(existing_contract)
                    or existing.get("contract_sha256") != _contract_digest(existing_contract)
                    or existing.get("sha256") != _sha256(candidate_path)
                    or not np.array_equal(
                        existing_frame.golgg_match_id.astype(str).to_numpy(),
                        canonical.golgg_match_id.astype(str).to_numpy(),
                    )
                ):
                    raise ValueError(f"existing candidate contract, membership, or checksum differs: {candidate_path}")
                candidate_frames[arm_key] = existing_frame
                details = existing.get("details", {})
                output_arms[arm_key] = {
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": existing["sha256"],
                    "candidate_receipt_sha256": _sha256(candidate_sidecar),
                    "rows": len(existing_frame),
                    "monthly_calibrations": details.get("monthly_calibrations", {}),
                    "source_fit_origins": sorted(existing_frame.source_fit_origin.astype(str).unique()),
                    "weighting": weighting, "cadence": cadence,
                }
                continue
            route_origins = [_source_origin(day, cadence) for day in canonical.date]
            if any(origin not in origin_set for origin in route_origins):
                absent = sorted(set(route_origins) - origin_set)
                raise ValueError(f"canonical rows lack required {cadence} fit origins: {absent}")
            source_logits, source_rho, raw_seed_probs, source_origins = _raw_oos_for_indices(
                eval_indices, frame, cadence, cache,
            )
            if source_logits.shape != (len(canonical), 3, 3):
                raise ValueError(f"{arm_key} raw forecast coverage is incomplete")
            source_cutoffs = pd.Series(pd.NaT, index=np.arange(len(frame)), dtype="datetime64[ns]")
            for global_index, source_origin in zip(eval_indices, source_origins):
                source_cutoffs.iloc[global_index] = pd.Timestamp(source_origin)
            # The calibration population includes released OOS rows over the
            # prior year, not just evaluation rows. Obtain raw predictions from
            # the arm's correct historical model for every possible CAL row.
            date_values = _date_column(frame, "date")
            release_values = _date_column(frame, "result_day")
            scored_start = pd.Timestamp("2023-01-01")
            scored_stop = pd.Timestamp("2026-10-01")
            possible_cal = np.flatnonzero(((date_values >= scored_start) & (date_values < scored_stop)).to_numpy())
            cal_all_logits, _, _, cal_source_origins = _raw_oos_for_indices(
                possible_cal, frame, cadence, cache,
            )
            source_cutoffs.iloc[possible_cal] = pd.to_datetime(cal_source_origins)
            # `source_cutoffs` is complete for every possible 2023–2026 CAL
            # row; earlier rows cannot enter a frozen 12-month calibration.
            created_at = datetime.now(timezone.utc).isoformat()
            records_by_global = {int(index): local for local, index in enumerate(possible_cal)}
            month_starts = [origin for origin in actual_origins if origin[:4] in {"2024", "2025", "2026"}]
            calibrated_eval = np.full((len(canonical), 3), np.nan, dtype=np.float64)
            calibration_by_origin: dict[str, dict[str, Any]] = {}
            params_by_origin: dict[str, np.ndarray] = {}
            cal_membership_by_origin: dict[str, dict[str, Any]] = {}
            for calibration_origin in month_starts:
                origin_stamp = pd.Timestamp(calibration_origin)
                cal_indices = calibration_indices(
                    frame, source_cutoffs, calibration_origin,
                    window_months=protocol["calibration"]["window_months"],
                )
                if len(cal_indices) < protocol["calibration"]["min_rows"]:
                    raise ValueError(
                        f"native monthly calibration {arm_key} at {calibration_origin} has "
                        f"{len(cal_indices)} OOS released rows; minimum is {protocol['calibration']['min_rows']}"
                    )
                cal_local = np.asarray([records_by_global[int(i)] for i in cal_indices], dtype=np.int64)
                cal_logits = cal_all_logits[cal_local]
                params, records = _fit_monthly_calibrator(
                    calibration_origin, cal_indices, cal_logits, data, backend,
                )
                month_rows = np.flatnonzero(
                    (pd.to_datetime(canonical.date).dt.to_period("M").dt.to_timestamp() == origin_stamp).to_numpy()
                )
                if len(month_rows):
                    mode_prob = _calibrated_modes(
                        source_logits[month_rows], params,
                        np.asarray(data["raw"]["gates"])[eval_indices[month_rows]],
                        canonical.iloc[month_rows].best_of.to_numpy(dtype=np.int64), backend,
                    )
                    calibrated_eval[month_rows] = mode_prob
                params_by_origin[calibration_origin] = params
                cal_frame = frame.iloc[cal_indices]
                source_entries = []
                for global_index in cal_indices:
                    source_origin = str(pd.Timestamp(source_cutoffs.iloc[global_index]).date())
                    source_entries.append({
                        "golgg_match_id": str(frame.iloc[global_index].golgg_match_id),
                        "source_fit_origin": source_origin,
                    })
                calibration_path = paths["report"] / "calibrations" / cadence / calibration_origin / "calibration.npz"
                calibration_contract = _expected_calibration_contract(
                    protocol, protocol_hash, weighting, cadence, source_hashes,
                    data["input_hashes"], calibration_origin,
                    training_source_sha256,
                )
                details = {
                    "row_count": int(len(cal_indices)),
                    "calibration_ids_sha256": _ids_digest(frame.iloc[cal_indices].golgg_match_id),
                    "calibration_ids": frame.iloc[cal_indices].golgg_match_id.astype(str).tolist(),
                    "source_rows": source_entries,
                    "max_event_date": _timestamp_max(cal_frame.date),
                    "max_release_date": _timestamp_max(cal_frame.result_day),
                    "fit_origins_max": max(entry["source_fit_origin"] for entry in source_entries),
                    "source_fit_origins": sorted({entry["source_fit_origin"] for entry in source_entries}),
                    "parameters": params.tolist(), "fit_records": records,
                    "minimum_rows": protocol["calibration"]["min_rows"],
                }
                if calibration_path.exists() or calibration_path.with_suffix(".npz.receipt.json").exists():
                    if not resume:
                        raise FileExistsError(f"monthly calibration exists without verified resume: {calibration_path}")
                    prior_params, prior_receipt = _load_calibration(calibration_path, calibration_contract)
                    if not np.array_equal(prior_params, params):
                        raise ValueError(f"resumed monthly calibration differs from recalculation: {calibration_path}")
                    calibration_receipt = prior_receipt
                else:
                    calibration_receipt = _save_calibration(
                        calibration_path, calibration_contract, params, records, details,
                    )
                params_by_origin[calibration_origin] = params
                cal_membership_by_origin[calibration_origin] = {
                    **details,
                    "calibration_artifact": str(calibration_path),
                    "calibration_artifact_sha256": calibration_receipt["calibration_sha256"],
                    "created_at_utc": calibration_receipt.get("created_at_utc"),
                }
            if not np.isfinite(calibrated_eval).all():
                missing_rows = np.flatnonzero(~np.isfinite(calibrated_eval).all(axis=1))
                raise ValueError(f"{arm_key} lacks monthly calibration support for {len(missing_rows)} canonical rows")

            # Provenance is per target row: source model membership maxima and
            # the actual monthly calibration support are both explicit.
            source_training_bounds = {}
            for origin in sorted(set(source_origins)):
                origin_receipt_path = paths["models"] / origin / f"seed_{SEEDS[0]}" / "model.joblib.receipt.json"
                origin_receipt = json.loads(origin_receipt_path.read_text(encoding="utf-8"))
                contract = origin_receipt["contract"]
                source_training_bounds[origin] = (
                    contract["train_max_event_date"], contract["train_max_release_date"],
                )
            source_train_event_max = [source_training_bounds[origin][0] for origin in source_origins]
            source_train_release_max = [source_training_bounds[origin][1] for origin in source_origins]
            source_train_end = [max(source_training_bounds[origin]) for origin in source_origins]
            cal_event_max = [cal_membership_by_origin[str(origin)]["max_event_date"] for origin in canonical.date.map(lambda d: pd.Timestamp(d).to_period("M").start_time.strftime("%Y-%m-%d"))]
            cal_release_max = [cal_membership_by_origin[str(origin)]["max_release_date"] for origin in canonical.date.map(lambda d: pd.Timestamp(d).to_period("M").start_time.strftime("%Y-%m-%d"))]
            result = _identity_columns(frame, eval_indices)
            result["train_end"] = source_train_end
            result["calibration_end"] = cal_release_max
            result["train_max_event_date"] = source_train_event_max
            result["train_max_release_date"] = source_train_release_max
            result["calibration_max_event_date"] = cal_event_max
            result["calibration_max_release_date"] = cal_release_max
            result["source_fit_origin"] = source_origins
            result["calibration_origin"] = [pd.Timestamp(day).to_period("M").start_time.strftime("%Y-%m-%d") for day in canonical.date]
            result["known_roster_scenario"] = True
            result["roster_asof_certified"] = False
            result["historical_reconstruction"] = True
            result["issued_historically"] = False
            result["model_identity"] = "native_a0_neural_expert_only"
            result["cadence"] = cadence
            result["weighting"] = weighting
            result["prediction_created_at_utc"] = created_at
            result["p"] = calibrated_eval[:, 0]
            result["p_full"] = calibrated_eval[:, 0]
            result["p_no_w20"] = calibrated_eval[:, 1]
            result["p_no_organization"] = calibrated_eval[:, 2]
            for mode_index, mode in enumerate(MODES):
                result[f"p_native_raw_{mode}"] = raw_seed_probs[:, mode_index, :].mean(axis=1)
                result[f"p_map_logit_{mode}"] = expit(source_logits[:, mode_index, :]).mean(axis=1)
                for seed_index, seed in enumerate(SEEDS):
                    result[f"p_native_raw_{mode}_seed_{seed}"] = raw_seed_probs[:, mode_index, seed_index]
                    result[f"map_logit_{mode}_seed_{seed}"] = source_logits[:, mode_index, seed_index]
                    result[f"rho_seed_{seed}"] = source_rho[:, seed_index]
            for column in ("p", "p_full", "p_no_w20", "p_no_organization"):
                values = result[column].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
                    raise ValueError(f"candidate export has invalid {column}")
            if len(result) != protocol["expected_canonical_rows"] or result.golgg_match_id.astype(str).duplicated().any():
                raise ValueError(f"candidate {arm_key} lost or duplicated canonical rows")
            for key in ("team1_id", "team2_id", "date", "result_day", "best_of", "y"):
                if not np.array_equal(result[key].astype(str).to_numpy(), canonical[key].astype(str).to_numpy()):
                    raise ValueError(f"candidate {arm_key} canonical identity mismatch on {key}")
            if not (pd.to_datetime(result.train_end) < pd.to_datetime(result.date)).all():
                raise ValueError(f"candidate {arm_key} has a noncausal TRAIN maximum")
            if not (pd.to_datetime(result.calibration_end) < pd.to_datetime(result.date)).all():
                raise ValueError(f"candidate {arm_key} has a noncausal CAL maximum")
            if not (pd.to_datetime(result.feature_source_max_day) < pd.to_datetime(result.date)).all():
                raise ValueError(f"candidate {arm_key} uses features released on/after target date")
            # Verify with the same identity/time/coverage contract as the locked scorer.
            from src.analysis.research_benchmark import validate_frame
            validate_frame(result, ["p"])
            contract = {
                "schema": "native_neural_refit_candidate_v1", "run_id": protocol["run_id"],
                "protocol_sha256": protocol_hash, "weighting": weighting, "cadence": cadence,
                "source_sha256": source_hashes, "input_sha256": data["input_hashes"],
                "model_identity": protocol["model_identity"],
                "canonical_ids_sha256": _ids_digest(result.golgg_match_id),
                "training_source_sha256": training_source_sha256, "versions": _versions(),
            }
            with _artifact_bundle(candidate_path) as staged:
                _write_parquet_exclusive(staged, result)
                candidate_receipt = {
                    "contract": _json_safe(contract), "contract_sha256": _contract_digest(contract),
                    "sha256": _sha256(staged),
                    "details": {"rows": len(result), "candidate_path": str(candidate_path),
                                "monthly_calibrations": cal_membership_by_origin,
                                "candidate_created_at_utc": created_at},
                }
                _write_json_exclusive(staged.with_suffix(staged.suffix + ".receipt.json"), candidate_receipt)
            candidate_frames[arm_key] = result
            calibration_params_by_arm[arm_key] = params_by_origin
            output_arms[arm_key] = {
                "candidate_path": str(candidate_path), "candidate_sha256": candidate_receipt["sha256"],
                "candidate_receipt_sha256": _sha256(candidate_sidecar),
                "rows": len(result), "monthly_calibrations": cal_membership_by_origin,
                "source_fit_origins": sorted(set(source_origins)),
                "weighting": weighting, "cadence": cadence,
            }

    # If candidates were resumed, recover calibration parameters from their
    # complete receipt and artifact sidecars before proving restored inference.
    for weighting in WEIGHTINGS:
        paths = _run_paths(protocol["run_id"], weighting)
        for cadence in ("annual", "monthly"):
            arm_key = f"{weighting}_{cadence}"
            if arm_key not in calibration_params_by_arm:
                receipt = json.loads((paths["output"] / arm_key / "candidate.parquet.receipt.json").read_text(encoding="utf-8"))
                monthly = receipt.get("details", {}).get("monthly_calibrations", {})
                expected_months = {origin for origin in actual_origins if int(origin[:4]) in protocol["evaluation_years"]}
                if set(monthly) != expected_months:
                    raise ValueError(f"resumed candidate lacks complete monthly calibration support: {arm_key}")
                values = {}
                for origin in monthly:
                    calibration_path = paths["report"] / "calibrations" / cadence / origin / "calibration.npz"
                    contract = _expected_calibration_contract(
                        protocol, protocol_hash, weighting, cadence, source_hashes, data["input_hashes"], origin,
                        training_sources[weighting],
                    )
                    params, _ = _load_calibration(calibration_path, contract)
                    if not np.array_equal(params, np.asarray(monthly[origin].get("parameters"), dtype=np.float64)):
                        raise ValueError(f"resumed candidate calibration differs from its artifact: {arm_key}/{origin}")
                    values[origin] = params
                calibration_params_by_arm[arm_key] = values

    verification = {}
    for weighting in WEIGHTINGS:
        paths = _run_paths(protocol["run_id"], weighting)
        for cadence in ("annual", "monthly"):
            key = f"{weighting}_{cadence}"
            candidate = candidate_frames.get(key)
            if candidate is None:
                candidate = pd.read_parquet(paths["output"] / key / "candidate.parquet")
            canonical_positions = {str(value): int(i) for i, value in enumerate(frame.golgg_match_id.astype(str))}
            by_bo = []
            for best_of in (1, 3, 5):
                choices = candidate.loc[candidate.best_of.astype(int) == best_of, "golgg_match_id"].astype(str)
                if choices.empty:
                    raise ValueError(f"candidate {key} lacks real best-of-{best_of} rows")
                by_bo.append(canonical_positions[choices.iloc[0]])
            sample = np.asarray(by_bo, dtype=np.int64)
            cache = forecast_caches[weighting]
            raw_logits, _, raw_probs, _ = _raw_oos_for_indices(sample, frame, cadence, cache)
            verification[key] = _saved_inference_check(
                data, backend, paths, protocol, protocol_hash, source_hashes,
                weighting, cadence, sample, raw_logits, raw_probs, candidate,
                calibration_params_by_arm[key],
            )

    report_path = PROJECT_ROOT / "data/08_reporting/a0_native_refit" / protocol["run_id"] / "aggregate_receipt.json"
    aggregate_receipt = {
        "status": "COMPLETE", "model_identity": protocol["model_identity"],
        "run_id": protocol["run_id"], "protocol_path": str(protocol_file),
        "protocol_sha256": protocol_hash, "research_root": str(root),
        "source_sha256": source_hashes, "input_sha256": data["input_hashes"],
        "versions": _versions(), "canonical_rows": len(canonical),
        "canonical_ids_sha256": _ids_digest(canonical.golgg_match_id),
        "arms": output_arms, "saved_inference_verification": verification,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
    }
    if report_path.exists():
        if not resume:
            raise FileExistsError(f"aggregate receipt already exists: {report_path}")
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        execution_fields = {"created_at_utc", "elapsed_seconds"}
        prior_semantics = {key: value for key, value in prior.items() if key not in execution_fields}
        current_semantics = {key: value for key, value in aggregate_receipt.items() if key not in execution_fields}
        if prior_semantics != _json_safe(current_semantics):
            raise ValueError("existing aggregate receipt differs from the verified artifacts or inference")
        aggregate_receipt = prior
    else:
        _write_json_exclusive(report_path, aggregate_receipt)
    return {
        "status": "COMPLETE", "canonical_rows": len(canonical),
        "candidate_paths": {
            key: str(_run_paths(protocol["run_id"], key.split("_")[0])["output"] / key / "candidate.parquet")
            for key in sorted(output_arms)
        },
        "aggregate_receipt": str(report_path),
        "saved_inference_verification": verification,
        "elapsed_seconds": time.monotonic() - started,
    }


def _smoke_one(
    protocol: dict[str, Any], protocol_hash: str, data: dict[str, Any], backend: dict[str, Any],
    source_hashes: dict[str, str], weighting: str, origin: str,
) -> dict[str, Any]:
    from src.models.a0_target_objectives import fit_beta_target_on_device

    frame = data["frame"]
    eligible = origin_train_indices(frame, origin)
    if not len(eligible):
        raise ValueError(f"no actual eligible TRAIN rows for bounded preflight at {origin}")
    labels = frame.y.to_numpy(dtype=np.int8)
    class_positions = [
        eligible[np.flatnonzero(labels[eligible] == value)[0]]
        for value in (0, 1) if np.any(labels[eligible] == value)
    ]
    if len(class_positions) != 2:
        raise ValueError("bounded native preflight TRAIN must contain both outcomes")
    remainder = eligible[~np.isin(eligible, class_positions)]
    chosen = np.sort(np.concatenate((np.asarray(class_positions, dtype=np.int64), remainder[:2046])))
    weights = training_row_weights(
        frame, chosen, origin,
        half_life_days=None if weighting == "uniform" else protocol["training"]["half_life_days"],
    )
    baseline, normalizer = _fit_baseline_normalizer(backend, data, chosen)
    seed = SEEDS[0]
    model = backend["nonlinear_history"].NonlinearHistoryModel(baseline, normalizer, seed=seed, epochs=1)
    init_hash = _torch_state_hash(model)
    bo = frame.best_of.to_numpy(dtype=np.int64)
    bo_index = (bo - 1) // 2
    fit_started = time.monotonic()
    fit_beta_target_on_device(
        model, data["raw"], data["history"], None, chosen, bo_index,
        data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
        device="cpu", objective="terminal", row_weights=weights if weighting == "h365" else None,
    )
    fit_seconds = time.monotonic() - fit_started
    model.main_objective = "terminal"
    if len(model.train_losses) != 1 or not np.isfinite(model.train_losses + model.auxiliary_losses).all():
        raise ValueError(f"bounded real-data smoke failed for {weighting}")
    source_tag = _contract_digest(source_hashes)[:12]
    smoke_root = PROJECT_ROOT / "data/06_models/a0_native_refit_smoke" / protocol["run_id"] / source_tag / weighting / origin
    checkpoint = smoke_root / "seed_20260910_one_epoch" / "model.joblib"
    receipt_path = PROJECT_ROOT / "data/08_reporting/a0_native_refit_smoke" / protocol["run_id"] / source_tag / f"{weighting}_{origin.replace('-', '')}.json"
    smoke_contract = {
        "schema": "unscored_native_real_data_smoke_v1", "run_id": protocol["run_id"],
        "protocol_sha256": protocol_hash, "source_sha256": source_hashes,
        "input_sha256": data["input_hashes"], "weighting": weighting,
        "versions": _versions(),
        "fit_origin": origin, "seed": seed, "epochs": 1,
        "train_ids_sha256": _ids_digest(frame.iloc[chosen].golgg_match_id),
        "train_membership_sha256": hashlib.sha256(np.asarray(chosen, dtype=np.int64).tobytes()).hexdigest(),
        "row_weights_sha256": hashlib.sha256(np.asarray(weights, dtype=np.float64).tobytes()).hexdigest(),
        "rows": int(len(chosen)), "scoring_eligible": False,
    }
    if checkpoint.exists() or checkpoint.with_suffix(checkpoint.suffix + ".receipt.json").exists() or receipt_path.exists():
        raise FileExistsError(f"unscored smoke namespace already exists: {smoke_root}")
    details = {
        "initial_state_sha256": init_hash, "final_state_sha256": _torch_state_hash(model),
        "main_losses": model.train_losses, "auxiliary_losses": model.auxiliary_losses,
        "dropout_layers": _dropout_receipt(model),
        "epoch_diagnostics": model.epoch_diagnostics,
        "weight_statistics": _weight_stats(weights),
        "fit_seconds": fit_seconds,
    }
    checkpoint_receipt = _atomic_checkpoint(checkpoint, model, smoke_contract, details)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "status": "UNSCORED_REAL_DATA_PREFLIGHT", "scoring_eligible": False,
        "run_id": protocol["run_id"], "weighting": weighting,
        "source_fit_origin": origin, "rows": int(len(chosen)),
        "selected_train_ids": frame.iloc[chosen].golgg_match_id.astype(str).tolist(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
        "checkpoint_receipt": str(checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")),
        "fit_seconds": details["fit_seconds"],
    }
    _write_json_exclusive(receipt_path, output)
    return output


def smoke(protocol_path: str | Path, research_root: str | Path | None = None) -> dict[str, Any]:
    """Run bounded one-seed/one-epoch real-data preflight for both weight paths."""
    protocol_file, protocol, protocol_hash = _protocol(protocol_path)
    root = _research_root(research_root)
    from src.models import a0_uncertainty as native
    native._configure_training_device("cpu", 2)
    data, source_hashes, backend = _load_environment(root)
    results = {}
    # Two isolated preflight namespaces prove both objective-weighting routes;
    # none is ever visible to the scored 45-origin model directory.
    for weighting in WEIGHTINGS:
        result = _smoke_one(
            protocol, protocol_hash, data,
            backend, source_hashes,
            weighting, protocol["origins"][0],
        )
        results[weighting] = result
    return {
        "status": "PREFLIGHT_COMPLETE", "scoring_eligible": False,
        "protocol": str(protocol_file), "preflights": results,
    }


