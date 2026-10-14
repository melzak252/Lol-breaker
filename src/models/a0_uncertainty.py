"""Genuine causal A0 member training and uncertainty forecasts.

Archive model implementations are loaded only from the pinned research archive;
this module's pure chronology/bootstrap/summary helpers have no archive import.
"""
from __future__ import annotations

import gc
from functools import partial
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp


SEEDS = (20260910, 20260911, 20260912, 20260913, 20260914)
VARIANTS = ("seed_only", "block_bootstrap")
MODES = ("full", "no_w20", "no_organization")
YEARS = (2024, 2025, 2026)


def _date_strings(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError("date columns cannot contain null values")
    return parsed.dt.strftime("%Y-%m-%d")


def annual_membership(frame: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closed TRAIN, released prior-year CAL, and calendar-year TEST indices."""
    if year not in YEARS:
        raise ValueError(f"unsupported annual model year: {year}")
    required = {"golgg_match_id", "date", "result_day"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"annual membership missing columns: {sorted(missing)}")
    ids = frame["golgg_match_id"]
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise ValueError("duplicate or null target ID")
    dates = _date_strings(frame["date"])
    released = _date_strings(frame["result_day"])
    if (released < dates).any():
        raise ValueError("label release precedes source match date")
    train_cutoff = f"{year - 1:04d}-01-01"
    cal_start = train_cutoff
    cal_end = f"{year:04d}-01-01"
    test_end = f"{year + 1:04d}-01-01"
    masks = (
        (dates < train_cutoff) & (released < train_cutoff),
        (dates >= cal_start) & (dates < cal_end) & (released < cal_end),
        (dates >= cal_end) & (dates < test_end),
    )
    return tuple(np.flatnonzero(mask.to_numpy()) for mask in masks)


def bootstrap_training_indices(
    frame: pd.DataFrame, train: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample nonempty TRAIN calendar weeks, carrying each week's full multiplicity."""
    indices = np.asarray(train)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("nonempty integer TRAIN indices are required")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(frame)):
        raise ValueError("TRAIN indices must be unique valid frame positions")
    if "date" not in frame or "golgg_match_id" not in frame:
        raise ValueError("weekly bootstrap requires dates and target IDs")
    if frame.iloc[indices]["golgg_match_id"].isna().any():
        raise ValueError("TRAIN target IDs cannot be null")
    dates = pd.to_datetime(frame.iloc[indices]["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("TRAIN dates cannot be null")
    week_start = dates.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    blocks: dict[str, list[int]] = {}
    for row, week in zip(indices.tolist(), week_start.tolist()):
        blocks.setdefault(week, []).append(int(row))
    weeks = sorted(blocks)
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(weeks), size=len(weeks))
    draw_order = [weeks[int(i)] for i in draw]
    sampled = np.fromiter(
        (row for week in draw_order for row in blocks[week]), dtype=np.int64
    )
    # Preserve native TRAIN traversal; resampling changes weights, not shuffle order.
    sampled.sort()
    week_counts = dict(zip(weeks, np.bincount(draw, minlength=len(weeks)).tolist(), strict=True))
    row_counts: dict[str, int] = {}
    for week, multiplicity in week_counts.items():
        if multiplicity:
            for row in blocks[week]:
                target_id = str(frame.iloc[row]["golgg_match_id"])
                row_counts[target_id] = int(multiplicity)
    receipt = {
        "scheme": "nonempty_calendar_weeks_monday_sunday_with_replacement",
        "seed": int(seed),
        "eligible_weeks": weeks,
        "draw_order": draw_order,
        "week_multiplicities": week_counts,
        "row_multiplicities": row_counts,
        "eligible_rows": int(len(indices)),
        "sampled_rows": int(len(sampled)),
    }
    return sampled, receipt


def summarize_members(members: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize NxK strictly interior probabilities as population dispersion."""
    values = np.asarray(members, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("member probabilities must be a nonempty NxK matrix")
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("member probabilities must be finite and strictly between zero and one")
    mean = values.mean(axis=1)
    epistemic = values.var(axis=1, ddof=0)
    aleatoric = (values * (1.0 - values)).mean(axis=1)
    total = mean * (1.0 - mean)
    entropy = -(mean * np.log(mean) + (1.0 - mean) * np.log1p(-mean))
    member_entropy = -(
        values * np.log(values) + (1.0 - values) * np.log1p(-values)
    ).mean(axis=1)
    logits = np.log(values) - np.log1p(-values)
    mutual_information = np.maximum(entropy - member_entropy, 0.0)
    return {
        "p": mean,
        "p_std": np.sqrt(epistemic),
        "sigma_z": logits.std(axis=1, ddof=0),
        "epistemic_variance": epistemic,
        "aleatoric_variance": aleatoric,
        "total_variance": total,
        "entropy": entropy,
        "mutual_information": mutual_information,
        "p_q10": np.quantile(values, 0.10, axis=1),
        "p_q90": np.quantile(values, 0.90, axis=1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _resolve_research_root(research_root: str | Path | None) -> Path:
    if research_root is not None:
        root = Path(research_root).expanduser()
    elif os.environ.get("RESEARCH_ROOT"):
        root = Path(os.environ["RESEARCH_ROOT"]).expanduser()
    else:
        project_root = Path(__file__).resolve().parents[2]
        pointer = project_root / "data/research_root.txt"
        if not pointer.is_file():
            raise FileNotFoundError(f"research root pointer not found: {pointer}")
        root = Path(pointer.read_text().strip()).expanduser()
    root = root.resolve(strict=True)
    required = (
        root / "a0-phase-walkforward-20260914/code/causal_nn.py",
        root / "nonlinear-history-20260912/code/src/models/nonlinear_history.py",
        root / "auxiliary-player-20260912/data/03_primary/targets.npy",
        root / "unified-ratings-20260911/data/04_feature/roster-optional-v1/metadata.jsonl",
    )
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError("research root is missing pinned A0 inputs: " + ", ".join(absent))
    return root


def _activate_archive_backend(research_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Extend the research namespace without shadowing any existing local module."""
    sys.dont_write_bytecode = True
    import src.models

    code_root = research_root / "nonlinear-history-20260912/code"
    models_root = code_root / "src/models"
    if not models_root.is_dir():
        raise FileNotFoundError(models_root)
    package_paths = list(src.models.__path__)
    if str(models_root) not in package_paths:
        src.models.__path__.append(str(models_root))
    importlib.invalidate_caches()
    required_modules = (
        "roster_optional",
        "temporal_architectures",
        "temporal_training",
        "stopped_series_torch",
        "auxiliary_player",
        "coherent_series",
        "beta_training_torch",
        "beta_training",
        "nonlinear_history",
        "unified_ratings",
        "beta_terminal",
        "beta_calibration",
        "tail_reliability",
    )
    loaded: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for short_name in required_modules:
        qualified = f"src.models.{short_name}"
        existing = sys.modules.get(qualified)
        if existing is not None:
            origin = Path(existing.__file__).resolve() if getattr(existing, "__file__", None) else None
            if origin is None or not origin.is_relative_to(models_root.resolve()):
                raise ImportError(f"refusing non-archive backend module {qualified}: {origin}")
            module = existing
        else:
            module = importlib.import_module(qualified)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(models_root.resolve()):
            raise ImportError(f"backend module resolved outside pinned archive: {qualified} -> {origin}")
        loaded[short_name] = module
    archive_tiers = models_root / "competition_tiers.py"
    local_models_root = Path(src.models.__path__[0]).resolve()
    local_tiers = local_models_root / "competition_tiers.py"
    archived_tiers_hash = _sha256(archive_tiers)
    if local_tiers.is_file():
        if _sha256(local_tiers) != archived_tiers_hash:
            raise ImportError("local competition_tiers differs from the pinned A0 taxonomy")
        tiers_module = importlib.import_module("src.models.competition_tiers")
        if Path(tiers_module.__file__).resolve() != local_tiers.resolve():
            raise ImportError("competition_tiers did not resolve to the checked local module")
    else:
        tiers_module = importlib.import_module("src.models.competition_tiers")
        if not Path(tiers_module.__file__).resolve().is_relative_to(models_root.resolve()):
            raise ImportError("competition_tiers resolved outside the pinned A0 archive")
    loaded["competition_tiers"] = tiers_module
    for module in loaded.values():
        path = archive_tiers if module is tiers_module else Path(module.__file__).resolve()
        source_hashes[f"research:{path.relative_to(research_root)}"] = _sha256(path)
    reference_files = (
        research_root / "a0-phase-walkforward-20260914/code/causal_nn.py",
        research_root / "a0-phase-walkforward-20260914/code/causal_calibrate.py",
        research_root / "a0-phase-walkforward-20260914/code/retrain_other_experts.py",
    )
    for path in reference_files:
        source_hashes[f"research:{path.relative_to(research_root)}"] = _sha256(path)
    for path in (Path(__file__).resolve(), Path(__file__).with_name("a0_accelerator.py").resolve()):
        source_hashes[f"project:src/models/{path.name}"] = _sha256(path)
    return loaded, source_hashes


class _IndexedRows:
    """A row-indexed view that keeps large history arrays memory-mapped."""

    def __init__(self, base: Any, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), *base.shape[1:])
        self.dtype = base.dtype

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: Any) -> Any:
        return self.base[self.indices[item]]


def _raw_subset(raw: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in raw.items()}


def _identity_values(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("roster identity must be a five-player sequence")
    roster = [str(value) for value in values]
    if len(roster) != 5 or len(set(roster)) != 5:
        raise ValueError("roster identity must contain five unique players")
    return roster


def _load_training_data(research_root: Path) -> dict[str, Any]:
    bank = research_root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    bank_receipt_path = bank / "completed.json"
    bank_receipt = json.loads(bank_receipt_path.read_text())
    if bank_receipt.get("status") not in ("COMPLETE", "PASS"):
        raise ValueError("corrected A0 feature bank is not complete")
    inventory = bank_receipt.get("outputs_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("corrected bank receipt has no immutable output inventory")
    input_hashes = {str(bank_receipt_path): _sha256(bank_receipt_path)}
    for filename, expected in inventory.items():
        expected_hash = expected.get("sha256") if isinstance(expected, dict) else expected
        path = bank / filename
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"corrected bank output hash mismatch: {path}")
        input_hashes[str(path)] = str(expected_hash)

    metadata_path = bank / "metadata.parquet"
    frame = pd.read_parquet(metadata_path)
    if "golgg_match_id" not in frame:
        raise ValueError("corrected metadata has no target IDs")
    if frame["golgg_match_id"].isna().any():
        raise ValueError("corrected metadata contains a null target ID")
    frame["golgg_match_id"] = frame["golgg_match_id"].astype(str)
    if frame["golgg_match_id"].duplicated().any():
        raise ValueError("corrected metadata contains duplicate target IDs")
    for name in ("team1_id", "team2_id"):
        if name not in frame:
            raise ValueError(f"corrected metadata missing {name}")
        frame[name] = frame[name].astype(str)
    for name in ("roster_a", "roster_b"):
        if name not in frame:
            raise ValueError(f"corrected metadata missing {name}")
        frame[name] = frame[name].map(_identity_values)
    frame["date"] = _date_strings(frame["date"])
    frame["result_day"] = _date_strings(frame["result_day"])
    ids = frame["golgg_match_id"].tolist()

    original_path = research_root / "unified-ratings-20260911/data/04_feature/roster-optional-v1/metadata.jsonl"
    bank_original_path = bank / "original_metadata.jsonl"
    if _sha256(original_path) != _sha256(bank_original_path):
        raise ValueError("corrected bank target metadata differs from the auxiliary-target identity source")
    original = pd.read_json(
        original_path,
        lines=True,
        convert_dates=False,
        dtype={"golgg_match_id": str, "team1_id": str, "team2_id": str},
    )
    if original["golgg_match_id"].astype(str).tolist() != ids:
        raise ValueError("corrected bank order differs from the auxiliary target metadata order")
    for name in ("team1_id", "team2_id"):
        if original[name].astype(str).tolist() != frame[name].tolist():
            raise ValueError(f"target identity side mismatch in {name}")
    for name in ("roster_a", "roster_b"):
        original_rosters = original[name].map(_identity_values).tolist()
        if original_rosters != frame[name].tolist():
            raise ValueError(f"target identity roster mismatch in {name}")
    for name in ("date", "best_of", "y"):
        if not np.array_equal(original[name].astype(str).to_numpy(), frame[name].astype(str).to_numpy()):
            raise ValueError(f"target metadata mismatch in {name}")

    label_path = research_root / "nonlinear-history-20260912/data/03_primary/aligned_scoreline_labels.parquet"
    labels = pd.read_parquet(label_path)
    labels["golgg_match_id"] = labels["golgg_match_id"].astype(str)
    if labels["golgg_match_id"].duplicated().any() or set(labels["golgg_match_id"]) != set(ids):
        raise ValueError("scoreline labels do not uniquely cover the feature-bank targets")
    labels = labels.set_index("golgg_match_id").loc[ids].reset_index()
    for name in ("date", "best_of", "y"):
        if not np.array_equal(labels[name].astype(str).to_numpy(), frame[name].astype(str).to_numpy()):
            raise ValueError(f"scoreline labels are misaligned on {name}")
    wins_a = labels["wins_a"].to_numpy(dtype=np.int64)
    wins_b = labels["wins_b"].to_numpy(dtype=np.int64)
    if not np.isin(frame["y"].to_numpy(), [0, 1]).all():
        raise ValueError("A0 target labels must be binary")
    if not np.array_equal((wins_a > wins_b).astype(np.int8), labels["y"].to_numpy(dtype=np.int8)):
        raise ValueError("scoreline outcome orientation does not match team-1 target labels")
    rounds = (frame["best_of"].to_numpy(dtype=np.int64) + 1) // 2
    if not np.array_equal(np.maximum(wins_a, wins_b), rounds) or np.any(np.minimum(wins_a, wins_b) >= rounds):
        raise ValueError("scoreline labels are not legal stopped outcomes for best-of")

    arrays_path = bank / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    history = {
        key: np.load(bank / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ("history", "mask", "champions")
    }
    n = len(frame)
    if any(len(values) != n for values in [*raw.values(), *history.values()]):
        raise ValueError("feature-bank arrays are not aligned to metadata row order")

    aux_dir = research_root / "auxiliary-player-20260912/data/03_primary"
    schema_path = aux_dir / "schema.json"
    schema = json.loads(schema_path.read_text())
    expected_shape = [n, 2, 5, 12]
    if schema.get("shape") != expected_shape or schema.get("target_only") is not True:
        raise ValueError("auxiliary target schema or identity cohort differs")
    targets = np.load(aux_dir / "targets.npy", mmap_mode="r", allow_pickle=False)
    target_mask = np.load(aux_dir / "target_mask.npy", mmap_mode="r", allow_pickle=False)
    if targets.shape != tuple(expected_shape) or target_mask.shape != targets.shape:
        raise ValueError("auxiliary player targets are not aligned to canonical target metadata")
    if not np.array_equal(original["golgg_match_id"].astype(str).to_numpy(), ids):
        raise ValueError("auxiliary target row identities differ from corrected bank order")

    source_paths = (
        label_path,
        original_path,
        schema_path,
        aux_dir / "targets.npy",
        aux_dir / "target_mask.npy",
    )
    for path in source_paths:
        input_hashes[str(path)] = _sha256(path)
    input_hashes[str(metadata_path)] = _sha256(metadata_path)
    input_hashes[str(arrays_path)] = _sha256(arrays_path)
    for key in history:
        path = bank / f"{key}.npy"
        input_hashes[str(path)] = _sha256(path)
    frame["y"] = frame["y"].astype(np.int8)
    return {
        "frame": frame,
        "raw": raw,
        "history": history,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "targets": targets,
        "target_mask": target_mask,
        "bank_receipt_sha256": _sha256(bank_receipt_path),
        "input_hashes": input_hashes,
    }

def _row_modes(gates: np.ndarray, requested_mode: str) -> np.ndarray:
    values = np.asarray(gates, dtype=float).copy()
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("optional gates must have shape [N,2,2]")
    if requested_mode == "no_w20":
        values[..., 1] = 0
    elif requested_mode == "no_organization":
        values[:] = 0
    elif requested_mode != "full":
        raise ValueError(f"mode must be one of {MODES}")
    no_w20 = np.all(values[..., 1] == 0, axis=1)
    no_organization = no_w20 & np.all(values[..., 0] == 0, axis=1)
    return np.where(no_organization, 2, np.where(no_w20, 1, 0)).astype(np.int64)


def _fit_dynamic_slopes(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The archive's bounded logit-slope fit generalized to any member count."""
    values = np.asarray(logits, dtype=float)
    target = np.asarray(labels, dtype=float)
    if values.ndim != 2 or target.shape != (len(values),) or not len(values):
        raise ValueError("aligned nonempty logits and labels are required for calibration")
    if not np.isfinite(values).all() or not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("invalid expert calibration inputs")
    slopes = np.empty(values.shape[1], dtype=float)
    for member in range(values.shape[1]):
        z = values[:, member]
        fitted = minimize_scalar(
            lambda slope: np.mean(np.logaddexp(0.0, z * slope) - target * z * slope),
            bounds=(0.15, 5.0),
            method="bounded",
        )
        if not fitted.success or not np.isfinite(fitted.x):
            raise RuntimeError("A0 expert slope calibration failed")
        slopes[member] = float(fitted.x)
    return slopes


def _fit_beta_member(fit_joint: Any, logits: np.ndarray, wins_a: np.ndarray, wins_b: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("one member's nonempty calibration logits are required")
    # fit_joint's native contract is a three-component terminal ensemble. Repeating
    # one member makes that exact likelihood equal to the member's own likelihood.
    repeated = np.repeat(values[:, None], 3, axis=1)
    record = fit_joint(repeated, wins_a, wins_b, "beta")
    return np.asarray([record["scale"], record["rho"]], dtype=float), record


def _terminal_probabilities(
    terminal_logs: Any,
    logits: np.ndarray,
    beta_parameters: np.ndarray,
    gates: np.ndarray,
    best_of: np.ndarray,
    requested_mode: str,
) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    params = np.asarray(beta_parameters, dtype=float)
    bo = np.asarray(best_of, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("map logits must have shape [N,K]")
    n, members = values.shape
    if params.shape != (members, 3, 2) or bo.shape != (n,):
        raise ValueError("member calibration is not aligned to logits")
    row_mode = _row_modes(gates, requested_mode)
    scales = params[:, row_mode, 0].T
    rhos = params[:, row_mode, 1].T
    mass = terminal_logs(
        (values * scales).reshape(-1),
        np.repeat(bo, members),
        rhos.reshape(-1),
    ).reshape(n, members, 6)
    result = np.exp(logsumexp(mass[:, :, :3], axis=2))
    if not np.isfinite(result).all() or np.any((result <= 0.0) | (result >= 1.0)):
        raise ValueError("terminal projection produced a non-interior probability")
    return result


def _expert_probabilities(
    logits: np.ndarray,
    slopes: np.ndarray,
    gates: np.ndarray,
    requested_mode: str,
) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    calibrators = np.asarray(slopes, dtype=float)
    if values.ndim != 2 or calibrators.shape != (3, values.shape[1]):
        raise ValueError("expert logits and mode calibrators are not aligned")
    routed = calibrators[_row_modes(gates, requested_mode)]
    result = expit(np.clip(values * routed, -20.0, 20.0))
    if not np.isfinite(result).all() or np.any((result <= 0.0) | (result >= 1.0)):
        raise ValueError("expert calibration produced a non-interior probability")
    return result


class _SavedA0UncertaintyModel:
    """Complete inference bundle; all learned objects are embedded, never path-loaded."""

    def __init__(self, members: list[dict[str, Any]], mixtures: dict[str, Any], metadata: dict[str, Any]):
        self.members = members
        self.mixtures = mixtures
        self.metadata = metadata

    def predict_members(
        self,
        raw: dict[str, np.ndarray],
        hist: dict[str, Any],
        best_of: np.ndarray,
        mode: str = "full",
    ) -> np.ndarray:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        bo = np.asarray(best_of)
        if bo.ndim != 1 or bo.dtype.kind not in "iu" or not np.isin(bo, [1, 3, 5]).all():
            raise ValueError("best_of must contain one integer in {1,3,5} per row")
        n = len(bo)
        if any(len(value) != n for value in raw.values()) or any(len(value) != n for value in hist.values()):
            raise ValueError("inference features are not row-aligned")
        if "gates" not in raw:
            raise ValueError("inference requires the optional-feature gate array")
        gates = np.asarray(raw["gates"], dtype=float)
        if gates.shape != (n, 2, 2):
            raise ValueError("gates must have shape [N,2,2]")
        bo_index = (bo.astype(np.int64) - 1) // 2
        nn_logits = np.empty((n, len(self.members)), dtype=float)
        rating_logits = np.empty_like(nn_logits)
        mlp_logits = np.empty_like(nn_logits)
        query_indices = np.arange(n, dtype=np.int64)
        for member_index, member in enumerate(self.members):
            neural = member["neural"]
            nn_logits[:, member_index] = neural.map_logits(
                raw, hist, None, query_indices, bo_index,
                mode == "full", mode != "no_organization",
            )
        unified_cache: dict[tuple[int, str], np.ndarray] = {}
        for member_index, member in enumerate(self.members):
            for expert_key, output in (("rating", rating_logits), ("mlp", mlp_logits)):
                expert = member[expert_key]
                cache_key = (id(expert), expert_key)
                if cache_key not in unified_cache:
                    unified_cache[cache_key] = _unified_logits(
                        expert, raw, hist, bo_index, modes=(mode,),
                    )[:, 0]
                output[:, member_index] = unified_cache[cache_key]
        return self._predict_from_logits(
            {"nn": nn_logits, "rating": rating_logits, "mlp": mlp_logits}, gates, bo, mode,
        )

    def _predict_from_logits(
        self, logits: dict[str, np.ndarray], gates: np.ndarray, bo: np.ndarray, mode: str,
    ) -> np.ndarray:
        n = len(bo)
        final = np.empty((n, len(self.members)), dtype=float)
        mixture = self.mixtures[mode]
        mixture_context = np.zeros((n, len(mixture.mean)), dtype=float)
        for member_index, member in enumerate(self.members):
            beta = _terminal_probabilities(
                self.metadata["terminal_logs"],
                logits["nn"][:, member_index : member_index + 1],
                np.asarray(member["beta_parameters"], dtype=float)[None, :, :],
                gates,
                bo,
                mode,
            )[:, 0]
            rating = _expert_probabilities(
                logits["rating"][:, member_index : member_index + 1],
                np.asarray(member["rating_slopes"], dtype=float)[:, None],
                gates,
                mode,
            )[:, 0]
            mlp = _expert_probabilities(
                logits["mlp"][:, member_index : member_index + 1],
                np.asarray(member["mlp_slopes"], dtype=float)[:, None],
                gates,
                mode,
            )[:, 0]
            experts = np.column_stack((beta, rating, mlp))
            final[:, member_index] = mixture.predict(experts, mixture_context)
        if not np.isfinite(final).all() or np.any((final <= 0.0) | (final >= 1.0)):
            raise ValueError("saved A0 model produced invalid member probabilities")
        return final


def _unified_logits(
    model: Any, raw: dict[str, np.ndarray], hist: dict[str, Any], bo: np.ndarray,
    *, modes: tuple[str, ...] = MODES,
) -> np.ndarray:
    n = len(bo)
    result = np.empty((n, len(modes)), dtype=float)
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        indices = np.arange(start, stop, dtype=np.int64)
        raw_part = _raw_subset(raw, indices)
        history_part = {key: _IndexedRows(value, indices) for key, value in hist.items()}
        cache = model.prepare(raw_part, history_part if model.full else None)
        local = np.arange(stop - start, dtype=np.int64)
        for mode_index, mode in enumerate(modes):
            result[start:stop, mode_index] = model.logits_cached(
                cache, local, bo[start:stop], mode == "full", mode != "no_organization",
            )
        del cache, raw_part, history_part
    if not np.isfinite(result).all():
        raise ValueError("nonfinite A0 rating expert logits")
    return result


def _calibration_frame_indices(frame: pd.DataFrame, year: int) -> list[dict[str, Any]]:
    cal_year = year - 1
    dates = frame["date"].to_numpy(dtype=str)
    releases = frame["result_day"].to_numpy(dtype=str)
    folds = []
    for quarter, (start_month, end_month) in enumerate(((4, 7), (7, 10), (10, 13))):
        start = f"{cal_year:04d}-{start_month:02d}-01"
        end = f"{year:04d}-01-01" if end_month == 13 else f"{cal_year:04d}-{end_month:02d}-01"
        fit = np.flatnonzero((dates >= f"{cal_year:04d}-01-01") & (dates < start) & (releases < start))
        test = np.flatnonzero((dates >= start) & (dates < end) & (releases < f"{year:04d}-01-01"))
        if not len(fit) or not len(test):
            raise ValueError(f"empty chronological calibration quarter: {start}–{end}")
        folds.append({"quarter": quarter, "start": start, "end": end, "fit": fit, "test": test})
    return folds


def _fit_fold_calibrators(
    fit: np.ndarray,
    logits: dict[str, np.ndarray],
    labels: np.ndarray,
    wins_a: np.ndarray,
    wins_b: np.ndarray,
    fit_joint: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    member_count = logits["nn"].shape[2]
    beta = np.empty((member_count, 3, 2), dtype=float)
    beta_records: dict[str, Any] = {}
    for member in range(member_count):
        records = []
        for mode_index, mode in enumerate(MODES):
            beta[member, mode_index], record = _fit_beta_member(
                fit_joint,
                logits["nn"][fit, mode_index, member],
                wins_a[fit],
                wins_b[fit],
            )
            records.append({"mode": mode, **record})
        beta_records[str(member)] = records
    slopes: dict[str, np.ndarray] = {}
    for expert in ("rating", "mlp"):
        by_mode = np.empty((3, member_count), dtype=float)
        for mode_index in range(3):
            by_mode[mode_index] = _fit_dynamic_slopes(
                logits[expert][fit, mode_index, :], labels[fit]
            )
        slopes[expert] = by_mode
    return beta, slopes, {"beta": beta_records, "slopes": {k: v.tolist() for k, v in slopes.items()}}


def _restore_fitted_checkpoint(
    path: Path, model_type: type, seed: int, epochs: int, device: str,
    *, kind: str | None = None, full: bool | None = None,
) -> Any:
    """Restore a trusted checkpoint after the runner verifies its execution inputs."""
    import joblib

    model = joblib.load(path)
    if type(model) is not model_type or model.seed != seed:
        raise ValueError(f"checkpoint class or member seed differs: {path}")
    if kind is not None and (model.kind != kind or model.full != full):
        raise ValueError(f"checkpoint expert contract differs: {path}")
    if len(model.train_losses) != epochs or (epochs and model.epochs != epochs):
        raise ValueError(f"checkpoint does not contain the completed epoch schedule: {path}")
    if getattr(model, "training_device", "cpu") != device:
        raise ValueError(f"checkpoint training device differs: {path}")
    if kind is None and any(
        len(getattr(model, field, ())) != epochs
        for field in ("auxiliary_losses", "epoch_diagnostics")
    ):
        raise ValueError(f"checkpoint neural epoch histories are incomplete: {path}")
    losses = list(model.train_losses) + list(getattr(model, "auxiliary_losses", []))
    if not np.isfinite(losses).all():
        raise ValueError(f"checkpoint training losses are nonfinite: {path}")
    return model


def _fit_training_component(
    backend: dict[str, Any],
    raw: dict[str, np.ndarray],
    history: dict[str, Any],
    sampled: np.ndarray,
    labels: np.ndarray,
    bo_index: np.ndarray,
    seed: int,
    *,
    kind: str,
    full: bool,
    initial: Any = None,
    epochs: int | None = None,
    device: str = "cpu",
    checkpoint: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    expected_epochs = 0 if kind == "raw" else (int(epochs) if epochs is not None else (12 if full else 24))
    if checkpoint is not None and checkpoint.exists():
        model = _restore_fitted_checkpoint(
            checkpoint, backend["unified_ratings"].UnifiedModel, seed,
            expected_epochs, device, kind=kind, full=full,
        )
        return model, {
            "kind": kind, "full": full, "seed": seed,
            "device": device, "epochs": expected_epochs,
            "train_losses": list(model.train_losses), "seconds": None,
            "sampled_rows": int(len(sampled)),
            "reused_checkpoint_sha256": _sha256(checkpoint),
        }
    train_raw = _raw_subset(raw, sampled)
    train_history = {key: _IndexedRows(value, sampled) for key, value in history.items()}
    local = np.arange(len(sampled), dtype=np.int64)


    # This is the archive UnifiedModel.fit path, called on a row-indexed view so
    # preprocessing and caches cover sampled TRAIN rows rather than the whole bank.
    UnifiedModel = backend["unified_ratings"].UnifiedModel
    model = UnifiedModel(kind=kind, full=full, seed=seed)
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("epoch override must be positive")
        model.epochs = int(epochs)
    started = time.monotonic()
    fit = model.fit
    if device == "cuda":
        from src.models.a0_accelerator import fit_unified_on_device

        fit = partial(fit_unified_on_device, model, device=device)
    cache = fit(
        train_raw,
        train_history if full else None,
        local,
        labels[sampled],
        bo_index[sampled],
        initial=initial,
    )
    duration = time.monotonic() - started
    losses = list(model.train_losses)
    if len(losses) != expected_epochs or not np.isfinite(losses).all():
        raise ValueError(f"invalid {kind} training loss/epoch receipt")
    details = {
        "kind": kind,
        "full": bool(full),
        "seed": int(seed),
        "device": getattr(model, "training_device", "cpu"),
        "epochs": int(model.epochs),
        "train_losses": losses,
        "seconds": duration,
        "sampled_rows": int(len(sampled)),
    }
    del cache, train_raw, train_history, local
    gc.collect()
    return model, details


def _fit_rating_model(
    backend: dict[str, Any],
    raw: dict[str, np.ndarray],
    sampled: np.ndarray,
    labels: np.ndarray,
    bo_index: np.ndarray,
    seed: int,
    *,
    checkpoint: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    model, details = _fit_training_component(
        backend,
        raw,
        {},
        sampled,
        labels,
        bo_index,
        seed,
        kind="raw",
        full=False,
        checkpoint=checkpoint,
    )
    details["epochs"] = 0
    details["train_losses"] = []
    return model, details


def _fit_calibrated_unified(model: Any, raw: dict[str, np.ndarray], history: dict[str, Any], bo_index: np.ndarray, cal_indices: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    raw_cal = _raw_subset(raw, cal_indices)
    history_cal = {key: _IndexedRows(value, cal_indices) for key, value in history.items()}
    cache = model.prepare(raw_cal, history_cal if model.full else None)
    local = np.arange(len(cal_indices), dtype=np.int64)
    start = time.monotonic()
    model.calibrate_cached(cache, local, labels[cal_indices], bo_index[cal_indices])
    elapsed = time.monotonic() - start
    details = {"slopes": dict(model.slopes), "seconds": elapsed, "calibration_rows": int(len(cal_indices))}
    del cache, raw_cal, history_cal, local
    gc.collect()
    return details


def _training_environment() -> dict[str, Any]:
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _configure_training_device(device: str, threads: int) -> None:
    if device not in ("cpu", "cuda"):
        raise ValueError("training device must be cpu or cuda")
    import torch

    torch.set_num_threads(threads)
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_per_process_memory_fraction(0.75)
        torch.use_deterministic_algorithms(True)


def fit_real_data_preflight(
    research_root: Path,
    *,
    year: int,
    variant: str,
    seed: int,
    threads: int = 2,
    max_rows: int = 512,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit one real-data epoch of each native A0 component without saving artifacts."""
    if year not in YEARS or variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("preflight requires a frozen year, variant, and member seed")
    if threads != 2 or max_rows < 2:
        raise ValueError("preflight requires two threads and at least two rows")
    _configure_training_device(device, threads)
    root = _resolve_research_root(research_root)
    backend, backend_hashes = _activate_archive_backend(root)
    data = _load_training_data(root)
    frame = data["frame"]
    train, _, _ = annual_membership(frame, year)
    labels = frame["y"].to_numpy(dtype=np.float64)
    best_of = frame["best_of"].to_numpy(dtype=np.int64)
    bo_index = (best_of - 1) // 2
    if variant == "block_bootstrap":
        sampled, sample_receipt = bootstrap_training_indices(frame, train, seed + 100000)
    else:
        sampled = train.copy()
        sample_receipt = {
            "scheme": "closed_train_rows_without_resampling",
            "seed": int(seed),
            "eligible_rows": int(len(train)),
            "sampled_rows": int(len(sampled)),
        }
    class_positions = []
    for outcome in (0.0, 1.0):
        matches = np.flatnonzero(labels[sampled] == outcome)
        if not len(matches):
            raise ValueError("preflight TRAIN sample does not contain both outcomes")
        class_positions.append(int(matches[0]))
    chosen_positions = np.asarray(class_positions, dtype=np.int64)
    remainder = np.setdiff1d(np.arange(len(sampled), dtype=np.int64), chosen_positions, assume_unique=True)
    remainder = np.random.default_rng(seed).permutation(remainder)
    selected = sampled[np.concatenate((chosen_positions, remainder))[:max_rows]]
    from src.models.roster_optional import RosterModel, subset
    from src.models.temporal_training import HistoryNormalizer

    started = time.monotonic()
    baseline = RosterModel("linear").fit(
        subset(data["raw"], selected), labels[selected], best_of[selected]
    )
    baseline_seconds = time.monotonic() - started
    started = time.monotonic()
    normalizer = HistoryNormalizer().fit(data["history"], selected)
    normalizer_seconds = time.monotonic() - started
    neural = backend["nonlinear_history"].NonlinearHistoryModel(
        baseline, normalizer, seed=seed, epochs=1
    )
    started = time.monotonic()
    fit_neural = neural.fit_beta
    if device == "cuda":
        from src.models.a0_accelerator import fit_beta_on_device

        fit_neural = partial(fit_beta_on_device, neural, device=device)
    fit_neural(
        data["raw"],
        data["history"],
        None,
        selected,
        bo_index,
        data["wins_a"],
        data["wins_b"],
        data["targets"],
        data["target_mask"],
    )
    neural_seconds = time.monotonic() - started
    rating, rating_fit = _fit_rating_model(
        backend, data["raw"], selected, labels, bo_index, seed
    )
    warm, warm_fit = _fit_training_component(
        backend, data["raw"], data["history"], selected, labels, bo_index, seed,
        kind="mlp", full=False, epochs=1, device=device,
    )
    full, full_fit = _fit_training_component(
        backend, data["raw"], data["history"], selected, labels, bo_index, seed,
        kind="mlp", full=True, initial=warm, epochs=1, device=device,
    )
    result = {
        "status": "PREFLIGHT_FIT_COMPLETE",
        "year": year,
        "variant": variant,
        "seed": seed,
        "device": device,
        "rows": len(selected),
        "training_ids": frame.iloc[selected]["golgg_match_id"].tolist(),
        "bootstrap": sample_receipt,
        "linear_roster_baseline_seconds": baseline_seconds,
        "history_normalizer_seconds": normalizer_seconds,
        "neural": {
            "epochs": len(neural.train_losses),
            "train_losses": neural.train_losses,
            "auxiliary_losses": neural.auxiliary_losses,
            "seconds": neural_seconds,
        },
        "raw_rating": rating_fit,
        "ratings_mlp_warm": warm_fit,
        "full_mlp": full_fit,
        "backend_source_sha256": backend_hashes,
    }
    del neural, rating, warm, full, baseline, normalizer
    gc.collect()
    return _json_safe(result)


def train_year(
    research_root: Path,
    model_dir: Path,
    prediction_path: Path,
    *,
    year: int,
    variant: str,
    seeds: tuple[int, ...],
    threads: int = 2,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Fit a genuine annual K=5 A0 ensemble and write its model and CAL/TEST rows."""
    if year not in YEARS:
        raise ValueError(f"unsupported annual model year: {year}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if tuple(seeds) != SEEDS:
        raise ValueError(f"the frozen A0 uncertainty protocol requires seeds {SEEDS}")
    if threads != 2:
        raise ValueError("the frozen training contract requires two torch threads per worker")
    _configure_training_device(device, threads)
    root = _resolve_research_root(research_root)
    run_started = time.monotonic()
    destination = Path(model_dir).expanduser().resolve()
    prediction = Path(prediction_path).expanduser().resolve()
    if prediction.exists() and not (resume and (destination / "receipt.json").is_file()):
        raise FileExistsError(f"refusing to overwrite completed prediction artifact: {prediction}")
    if destination.exists() and (not destination.is_dir() or (not resume and any(destination.iterdir()))):
        raise FileExistsError(f"model directory must be fresh unless explicitly resuming: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    prediction.parent.mkdir(parents=True, exist_ok=True)

    backend, backend_hashes = _activate_archive_backend(root)
    import joblib
    data_load_started = time.monotonic()
    data = _load_training_data(root)
    data_load_seconds = time.monotonic() - data_load_started
    if resume and (destination / "receipt.json").is_file():
        receipt = json.loads((destination / "receipt.json").read_text())
        expected_order = [{"index": i, "seed": int(seed)} for i, seed in enumerate(seeds)]
        if (
            receipt.get("status") != "COMPLETE" or receipt.get("year") != year
            or receipt.get("variant") != variant or receipt.get("member_order") != expected_order
            or receipt.get("backend_source_sha256") != backend_hashes
            or receipt.get("input_sha256") != data["input_hashes"]
        ):
            raise ValueError("completed annual model has a different execution contract")
        for name, expected in receipt["outputs_sha256"].items():
            artifact = prediction if name == "prediction_parquet" else destination / name
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise ValueError(f"completed annual artifact hash differs: {artifact}")
        return receipt
    frame = data["frame"]
    train, cal, test = annual_membership(frame, year)
    if not all(len(indices) for indices in (train, cal, test)):
        raise ValueError("annual TRAIN/CAL/TEST partitions must all be nonempty")
    cal_test = np.sort(np.concatenate((cal, test))).astype(np.int64)
    cal_local = np.searchsorted(cal_test, cal)
    labels = frame["y"].to_numpy(dtype=np.float64)
    best_of = frame["best_of"].to_numpy(dtype=np.int64)
    bo_index = (best_of - 1) // 2
    if not np.isin(best_of, [1, 3, 5]).all():
        raise ValueError("A0 supports only best-of 1, 3, or 5")
    wins_a = data["wins_a"]
    wins_b = data["wins_b"]
    cal_frame = frame.iloc[cal].reset_index(drop=True)
    folds = _calibration_frame_indices(cal_frame, year)

    # Seed-only samples have identical TRAIN membership, so the deterministic
    # baseline, history normalizer and raw-rating fit are fit once and shared.
    shared_baseline = None
    shared_history_normalizer = None
    shared_rating = None
    shared_rating_calibration = None
    shared_baseline_seconds = None
    restored_first_neural = None
    first_neural_path = destination / "member_0_nn.joblib"
    if resume and first_neural_path.exists():
        restored_first_neural = _restore_fitted_checkpoint(
            first_neural_path, backend["nonlinear_history"].NonlinearHistoryModel,
            seeds[0], 12, device,
        )
    if variant == "seed_only":
        from src.models.roster_optional import RosterModel
        from src.models.roster_optional import subset as archive_subset
        from src.models.temporal_training import HistoryNormalizer

        baseline_seconds = history_normalizer_seconds = None
        if restored_first_neural is not None:
            shared_baseline = restored_first_neural.baseline
            shared_history_normalizer = restored_first_neural.normalizer
        else:
            started = time.monotonic()
            shared_baseline = RosterModel("linear").fit(
                archive_subset(data["raw"], train), labels[train], best_of[train]
            )
            baseline_seconds = time.monotonic() - started
            started = time.monotonic()
            shared_history_normalizer = HistoryNormalizer().fit(data["history"], train)
            history_normalizer_seconds = time.monotonic() - started
        shared_rating, shared_rating_fit = _fit_rating_model(
            backend, data["raw"], train, labels, bo_index, seeds[0],
            checkpoint=destination / "rating_raw_shared.joblib" if resume else None,
        )
        shared_rating_calibration = _fit_calibrated_unified(
            shared_rating, data["raw"], data["history"], bo_index, cal, labels
        )
        shared_baseline_seconds = {
            "linear_roster_baseline_seconds": baseline_seconds,
            "history_normalizer_seconds": history_normalizer_seconds,
            "raw_rating_fit": shared_rating_fit,
            "raw_rating_calibration": shared_rating_calibration,
        }

    member_objects: list[dict[str, Any]] = []
    member_receipts: list[dict[str, Any]] = []
    sampled_receipts: list[dict[str, Any]] = []
    for member_index, seed in enumerate(seeds):
        member_training_started = time.monotonic()
        if variant == "block_bootstrap":
            sampled, draw_receipt = bootstrap_training_indices(frame, train, seed + 100000)
        else:
            sampled = train.copy()
            row_mult = {str(frame.iloc[row].golgg_match_id): 1 for row in train}
            draw_receipt = {
                "scheme": "closed_train_rows_without_resampling",
                "seed": int(seed),
                "eligible_rows": int(len(train)),
                "sampled_rows": int(len(sampled)),
                "row_multiplicities": row_mult,
            }
        sampled_receipts.append(draw_receipt)
        neural_checkpoint = destination / f"member_{member_index}_nn.joblib"
        neural = None
        if resume and neural_checkpoint.exists():
            neural = restored_first_neural if member_index == 0 else _restore_fitted_checkpoint(
                neural_checkpoint, backend["nonlinear_history"].NonlinearHistoryModel,
                seed, 12, device,
            )
        if neural is not None:
            baseline = neural.baseline
            history_normalizer = neural.normalizer
            baseline_report = {
                "reused_neural_checkpoint_sha256": _sha256(neural_checkpoint),
                "linear_roster_baseline_seconds": None,
                "history_normalizer_seconds": None,
            }
        elif variant == "seed_only":
            baseline = shared_baseline
            history_normalizer = shared_history_normalizer
            baseline_report = shared_baseline_seconds
        else:
            from src.models.roster_optional import RosterModel
            from src.models.roster_optional import subset as archive_subset
            from src.models.temporal_training import HistoryNormalizer

            started = time.monotonic()
            baseline = RosterModel("linear").fit(
                archive_subset(data["raw"], sampled), labels[sampled], best_of[sampled]
            )
            baseline_time = time.monotonic() - started
            started = time.monotonic()
            history_normalizer = HistoryNormalizer().fit(data["history"], sampled)
            normalizer_time = time.monotonic() - started
            baseline_report = {
                "linear_roster_baseline_seconds": baseline_time,
                "history_normalizer_seconds": normalizer_time,
            }
        nn_seconds = None
        if neural is None:
            started = time.monotonic()
            neural = backend["nonlinear_history"].NonlinearHistoryModel(
                baseline, history_normalizer, seed=seed, epochs=12
            )
            fit_neural = neural.fit_beta
            if device == "cuda":
                from src.models.a0_accelerator import fit_beta_on_device

                fit_neural = partial(fit_beta_on_device, neural, device=device)
            fit_neural(
                data["raw"],
                data["history"],
                None,
                sampled,
                bo_index,
                wins_a,
                wins_b,
                data["targets"],
                data["target_mask"],
            )
            nn_seconds = time.monotonic() - started
        if (
            any(len(trace) != 12 for trace in (
                neural.train_losses, neural.auxiliary_losses, neural.epoch_diagnostics,
            ))
            or not np.isfinite(neural.train_losses + neural.auxiliary_losses).all()
        ):
            raise ValueError("invalid nonlinear-history member fit")
        if not neural_checkpoint.exists():
            joblib.dump(neural, neural_checkpoint)

        if variant == "seed_only":
            rating = shared_rating
            rating_fit = shared_rating_fit
            rating_calibration = shared_rating_calibration
            if member_index == 0:
                rating_checkpoint = destination / "rating_raw_shared.joblib"
                if not rating_checkpoint.exists():
                    joblib.dump(rating, rating_checkpoint)
        else:
            rating_checkpoint = destination / f"member_{member_index}_rating_raw.joblib"
            rating, rating_fit = _fit_rating_model(
                backend, data["raw"], sampled, labels, bo_index, seed,
                checkpoint=rating_checkpoint if resume else None,
            )
            rating_calibration = _fit_calibrated_unified(
                rating, data["raw"], data["history"], bo_index, cal, labels
            )
            if not rating_checkpoint.exists():
                joblib.dump(rating, rating_checkpoint)

        warm_checkpoint = destination / f"member_{member_index}_ratings_mlp_warm.joblib"
        full_checkpoint = destination / f"member_{member_index}_full_mlp.joblib"
        warm, warm_fit = _fit_training_component(
            backend,
            data["raw"],
            data["history"],
            sampled,
            labels,
            bo_index,
            seed,
            kind="mlp",
            full=False,
            device=device,
            checkpoint=warm_checkpoint if resume else None,
        )
        if not warm_checkpoint.exists():
            joblib.dump(warm, warm_checkpoint)
        full_mlp, full_fit = _fit_training_component(
            backend,
            data["raw"],
            data["history"],
            sampled,
            labels,
            bo_index,
            seed,
            kind="mlp",
            full=True,
            initial=warm,
            device=device,
            checkpoint=full_checkpoint if resume else None,
        )
        del warm
        mlp_calibration = _fit_calibrated_unified(
            full_mlp, data["raw"], data["history"], bo_index, cal, labels
        )
        if not full_checkpoint.exists():
            joblib.dump(full_mlp, full_checkpoint)

        member_objects.append(
            {
                "seed": int(seed),
                "neural": neural,
                "rating": rating,
                "mlp": full_mlp,
                "beta_parameters": None,
                "rating_slopes": None,
                "mlp_slopes": None,
            }
        )
        max_sampled_release = frame.iloc[np.unique(sampled)]["result_day"].max()
        max_sampled_feature_day = frame.iloc[np.unique(sampled)]["feature_source_max_day"].dropna().max() if "feature_source_max_day" in frame else None
        files = [neural_checkpoint, warm_checkpoint, full_checkpoint]
        if variant == "block_bootstrap" or member_index == 0:
            files.append(rating_checkpoint)
        member_receipts.append(
            {
                "member_index": member_index,
                "seed": int(seed),
                "variant": variant,
                "training_seconds": time.monotonic() - member_training_started,
                "sample": draw_receipt,
                "train_ids": frame.iloc[train]["golgg_match_id"].tolist(),
                "sampled_train_row_count": int(len(sampled)),
                "max_train_release": str(max_sampled_release),
                "max_train_feature_source_day": None if pd.isna(max_sampled_feature_day) else str(max_sampled_feature_day),
                "cal_ids": frame.iloc[cal]["golgg_match_id"].tolist(),
                "test_ids": frame.iloc[test]["golgg_match_id"].tolist(),
                "max_cal_release": str(frame.iloc[cal]["result_day"].max()),
                "max_test_release": str(frame.iloc[test]["result_day"].max()),
                "training": {
                    **baseline_report,
                    "neural": {
                        "epochs": int(neural.epochs),
                        "device": getattr(neural, "training_device", "cpu"),
                        "train_losses": list(neural.train_losses),
                        "auxiliary_losses": list(neural.auxiliary_losses),
                        "epoch_diagnostics": list(neural.epoch_diagnostics),
                        "seconds": nn_seconds,
                    },
                    "raw_rating": rating_fit,
                    "raw_rating_calibration": rating_calibration,
                    "ratings_mlp_warm": warm_fit,
                    "full_mlp": full_fit,
                    "full_mlp_calibration": mlp_calibration,
                },
                "checkpoints": {str(path): _sha256(path) for path in files},
            }
        )
        del neural, full_mlp, baseline, history_normalizer, sampled
        gc.collect()

    # Evaluate only the annual CAL+TEST cohort, preserving bank orientation/order.
    query_logits = {"nn": None, "rating": None, "mlp": None}
    n_query = len(cal_test)
    query_logits["nn"] = np.empty((n_query, 3, len(seeds)), dtype=float)
    query_logits["rating"] = np.empty_like(query_logits["nn"])
    query_logits["mlp"] = np.empty_like(query_logits["nn"])
    query_bo = bo_index[cal_test]
    for member_index, member in enumerate(member_objects):
        for mode_index, (w20, organization) in enumerate(
            ((True, True), (False, True), (False, False))
        ):
            query_logits["nn"][:, mode_index, member_index] = member["neural"].map_logits(
                data["raw"], data["history"], None, cal_test, bo_index, w20, organization
            )
        query_logits["mlp"][:, :, member_index] = _unified_logits(
            member["mlp"], _raw_subset(data["raw"], cal_test),
            {key: _IndexedRows(value, cal_test) for key, value in data["history"].items()}, query_bo
        )
    rating_predictions: dict[int, np.ndarray] = {}
    for member_index, member in enumerate(member_objects):
        rating_id = id(member["rating"])
        if rating_id not in rating_predictions:
            rating_predictions[rating_id] = _unified_logits(
                member["rating"], _raw_subset(data["raw"], cal_test), {}, query_bo
            )
        query_logits["rating"][:, :, member_index] = rating_predictions[rating_id]

    # Fit final annual parameters member-by-member, then refit mixture weights only
    # on the archive's Q2/Q3/Q4 forecasts from preceding released CAL quarters.
    from src.models.beta_terminal import fit_joint, terminal_logs
    from src.models.competition_tiers import classify_competition
    from src.models.tail_reliability import ReliabilityLayer, context_features

    calibration_started = time.monotonic()
    cal_logits = {name: values[cal_local] for name, values in query_logits.items()}
    cal_labels = labels[cal]
    beta_parameters = np.empty((len(seeds), 3, 2), dtype=float)
    beta_records: dict[str, Any] = {}
    for member_index in range(len(seeds)):
        records = []
        for mode_index, mode in enumerate(MODES):
            parameters, record = _fit_beta_member(
                fit_joint,
                cal_logits["nn"][:, mode_index, member_index],
                wins_a[cal],
                wins_b[cal],
            )
            beta_parameters[member_index, mode_index] = parameters
            records.append({"mode": mode, **record})
        beta_records[str(member_index)] = records
    slope_parameters: dict[str, np.ndarray] = {}
    for expert in ("rating", "mlp"):
        slopes = np.empty((3, len(seeds)), dtype=float)
        for mode_index in range(3):
            slopes[mode_index] = _fit_dynamic_slopes(
                cal_logits[expert][:, mode_index, :], cal_labels
            )
        slope_parameters[expert] = slopes

    crossfit = np.full((len(cal_frame), 3, 3), np.nan, dtype=float)
    fold_receipts: list[dict[str, Any]] = []
    cal_gates = np.asarray(data["raw"]["gates"])[cal]
    cal_best_of = best_of[cal]
    for fold in folds:
        fit_ix, test_ix = fold["fit"], fold["test"]
        fold_beta, fold_slopes, fold_parameters = _fit_fold_calibrators(
            fit_ix,
            {name: values[cal_local] for name, values in query_logits.items()},
            cal_labels,
            wins_a[cal],
            wins_b[cal],
            fit_joint,
        )
        for mode_index, mode in enumerate(MODES):
            nn_probability = _terminal_probabilities(
                terminal_logs,
                cal_logits["nn"][test_ix, mode_index, :],
                fold_beta,
                cal_gates[test_ix],
                cal_best_of[test_ix],
                mode,
            )
            rating_probability = _expert_probabilities(
                cal_logits["rating"][test_ix, mode_index, :],
                fold_slopes["rating"],
                cal_gates[test_ix],
                mode,
            )
            mlp_probability = _expert_probabilities(
                cal_logits["mlp"][test_ix, mode_index, :],
                fold_slopes["mlp"],
                cal_gates[test_ix],
                mode,
            )
            crossfit[test_ix, mode_index, 0] = nn_probability.mean(axis=1)
            crossfit[test_ix, mode_index, 1] = rating_probability.mean(axis=1)
            crossfit[test_ix, mode_index, 2] = mlp_probability.mean(axis=1)
        fold_receipts.append(
            {
                "start": fold["start"],
                "end": fold["end"],
                "fit_ids": cal_frame.iloc[fit_ix]["golgg_match_id"].tolist(),
                "prediction_ids": cal_frame.iloc[test_ix]["golgg_match_id"].tolist(),
                "max_fit_release": str(cal_frame.iloc[fit_ix]["result_day"].max()),
                "parameters": fold_parameters,
            }
        )
    if not np.isfinite(crossfit).all(axis=(1, 2)).any():
        raise ValueError("no finite quarter-cross-fitted forecasts for mixture calibration")
    mix_indices = np.flatnonzero(np.isfinite(crossfit).all(axis=(1, 2)))
    if not (cal_frame.iloc[mix_indices]["result_day"] < f"{year:04d}-01-01").all():
        raise ValueError("late CAL label entered the annual mixture fit")
    tiers = np.asarray(
        [
            classify_competition(tournament, pd.Timestamp(day).date()).tier.value
            for tournament, day in zip(cal_frame["tournament_name"], cal_frame["date"])
        ],
        dtype=object,
    )
    mixtures: dict[str, Any] = {}
    mixture_receipts = []
    for mode_index, mode in enumerate(MODES):
        expert_cal = crossfit[mix_indices, mode_index]
        context, context_names = context_features(
            _raw_subset(data["raw"], cal[mix_indices]),
            cal_best_of[mix_indices],
            tiers[mix_indices],
            expert_cal,
            w20=mode == "full",
            org=mode != "no_organization",
        )
        mix = ReliabilityLayer("constant_mixture").fit(
            expert_cal,
            context,
            cal_labels[mix_indices],
        )
        mixtures[mode] = mix
        mixture_receipts.append(
            {
                "mode": mode,
                "theta": mix.theta.tolist(),
                "weights": mix.distribution(
                    mix.theta, expert_cal[:1], mix.design(context[:1]),
                )[1][0].tolist(),
                "fit_ids": cal_frame.iloc[mix_indices]["golgg_match_id"].tolist(),
                "context_columns": context_names,
                "fit_info": mix.fit_info,
            }
        )

    calibration_seconds = time.monotonic() - calibration_started
    for member_index, member in enumerate(member_objects):
        member["beta_parameters"] = beta_parameters[member_index].copy()
        member["rating_slopes"] = slope_parameters["rating"][:, member_index].copy()
        member["mlp_slopes"] = slope_parameters["mlp"][:, member_index].copy()

    inference = _SavedA0UncertaintyModel(
        member_objects,
        mixtures,
        {"terminal_logs": terminal_logs, "year": year, "variant": variant, "seeds": list(seeds)},
    )
    prediction_started = time.monotonic()
    predictions = {
        mode: inference._predict_from_logits(
            {key: values[:, mode_index, :] for key, values in query_logits.items()},
            np.asarray(data["raw"]["gates"])[cal_test],
            best_of[cal_test],
            mode,
        )
        for mode_index, mode in enumerate(MODES)
    }
    prediction_seconds = time.monotonic() - prediction_started
    full_summary = summarize_members(predictions["full"])
    result = frame.iloc[cal_test].reset_index(drop=True).copy()
    result["model_year"] = int(year)
    result["variant"] = variant
    result["partition"] = np.where(np.isin(cal_test, cal), "cal", "test")
    result["train_end"] = f"{year - 2:04d}-12-31"
    result["calibration_end"] = f"{year - 1:04d}-12-31"
    for member_index in range(len(seeds)):
        result[f"p_member_{member_index}"] = predictions["full"][:, member_index]
    for name, values in full_summary.items():
        result[name] = values
    result["p_no_w20"] = predictions["no_w20"].mean(axis=1)
    result["p_no_organization"] = predictions["no_organization"].mean(axis=1)
    if not np.isfinite(result[["p", "p_std", "sigma_z", "epistemic_variance", "aleatoric_variance", "total_variance", "entropy", "mutual_information", "p_q10", "p_q90", "p_no_w20", "p_no_organization"]].to_numpy(dtype=float)).all():
        raise ValueError("prediction table has nonfinite forecast or uncertainty values")
    if np.any((result[[f"p_member_{i}" for i in range(len(seeds))]].to_numpy() <= 0.0) | (result[[f"p_member_{i}" for i in range(len(seeds))]].to_numpy() >= 1.0)):
        raise ValueError("prediction table contains a member probability outside (0,1)")
    if result["golgg_match_id"].astype(str).duplicated().any():
        raise ValueError("duplicate CAL/TEST prediction targets")
    result["golgg_match_id"] = result["golgg_match_id"].astype(str)
    result["team1_id"] = result["team1_id"].astype(str)
    result["team2_id"] = result["team2_id"].astype(str)
    artifact_write_started = time.monotonic()
    result.to_parquet(prediction, index=False)

    # Preserve fitted objects individually as well as in the self-contained bundle.
    model_path = destination / "inference_model.joblib"
    joblib.dump(inference, model_path)
    artifact_write_seconds = time.monotonic() - artifact_write_started
    outputs = {path.name: _sha256(path) for path in destination.iterdir() if path.is_file()}
    outputs["prediction_parquet"] = _sha256(prediction)
    max_train_release = str(frame.iloc[train]["result_day"].max())
    max_cal_release = str(frame.iloc[cal]["result_day"].max())
    total_seconds = time.monotonic() - run_started
    receipt = {
        "status": "COMPLETE",
        "model": "genuine_nonlinear_history_causal_a0",
        "year": int(year),
        "variant": variant,
        "member_order": [{"index": i, "seed": int(seed)} for i, seed in enumerate(seeds)],
        "train_ids": frame.iloc[train]["golgg_match_id"].tolist(),
        "cal_ids": frame.iloc[cal]["golgg_match_id"].tolist(),
        "test_ids": frame.iloc[test]["golgg_match_id"].tolist(),
        "counts": {"train": len(train), "cal": len(cal), "test": len(test)},
        "train_end": f"{year - 2:04d}-12-31",
        "calibration_end": f"{year - 1:04d}-12-31",
        "max_train_release": max_train_release,
        "max_cal_release": max_cal_release,
        "max_test_release": str(frame.iloc[test]["result_day"].max()),
        "bootstrap_seed_rule": "member_seed_plus_100000",
        "sampled_multiplicities": sampled_receipts,
        "members": member_receipts,
        "shared_deterministic_training": shared_baseline_seconds,
        "annual_beta_calibration": beta_records,
        "annual_expert_slopes": {key: value.tolist() for key, value in slope_parameters.items()},
        "quarter_crossfit": fold_receipts,
        "mixtures": mixture_receipts,
        "calibration_interpretation": "CAL rows are used to fit annual member parameters and quarter-crossfit is used to fit the fixed mixture; CAL output rows are not out-of-sample final-mixture forecasts",
        "prediction_rows": len(result),
        "prediction_path": str(prediction),
        "inference_model_path": str(model_path),
        "input_sha256": data["input_hashes"],
        "backend_source_sha256": backend_hashes,
        "versions": _training_environment(),
        "threads": threads,
        "device": device,
        "timings_seconds": {
            "data_load": data_load_seconds,
            "calibration": calibration_seconds,
            "prediction": prediction_seconds,
            "artifact_write": artifact_write_seconds,
            "total_before_receipt": total_seconds,
        },
        "outputs_sha256": outputs,
    }
    receipt_path = destination / "receipt.json"
    _write_json(receipt_path, receipt)
    return _json_safe(receipt)


def load_model(model_dir: Path, research_root: Path) -> _SavedA0UncertaintyModel:
    """Load a complete serialized annual inference bundle after pinning its backend."""
    root = _resolve_research_root(research_root)
    backend, backend_hashes = _activate_archive_backend(root)
    del backend
    directory = Path(model_dir).expanduser().resolve()
    receipt_path = directory / "receipt.json"
    model_path = directory / "inference_model.joblib"
    if not receipt_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("saved A0 model is missing its receipt or inference bundle")
    receipt = json.loads(receipt_path.read_text())
    current_hashes = dict(backend_hashes)
    if receipt.get("backend_source_sha256") != current_hashes:
        raise ValueError("pinned A0 backend source changed since training")
    if receipt.get("outputs_sha256", {}).get("inference_model.joblib") != _sha256(model_path):
        raise ValueError("saved inference model hash differs from its training receipt")
    import joblib

    model = joblib.load(model_path)
    if not isinstance(model, _SavedA0UncertaintyModel):
        raise TypeError("saved artifact is not a complete A0 uncertainty inference model")
    return model
