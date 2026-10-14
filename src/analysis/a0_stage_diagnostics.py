"""Trace frozen Causal A0 inference stages without fitting or changing selections."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from src.analysis import ev_asof_replay as replay
from src.analysis.a0_error_audit import ROOT, save_json, sha256
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.probability_metrics import (
    binary_log_loss_vector,
    calculate_ece,
    evaluate_binary_probabilities,
)

DRAWS = 5000
SEED = 20260930
EPSILON = 1e-12
TAX_RATE = 0.12
PRIMARY_N = 1852
FIXED_N = 498
MAP_STAGE = "neural_raw_map_probability_bo1_only"
PROBABILITY_COLUMNS = {
    "neural_unscaled_iid": "p_neural_unscaled_iid",
    "neural_scaled_iid": "p_neural_scaled_iid",
    "neural_unscaled_frozen_beta": "p_neural_unscaled_frozen_beta",
    "neural_calibrated_beta": "p_neural_calibrated_beta",
    "neural_baseline_only_frozen_beta": "p_neural_baseline_only_frozen_beta",
    "rating_raw_series": "p_rating_raw_series",
    "rating_calibrated_series": "p_rating_calibrated_series",
    "mlp_raw_series": "p_mlp_raw_series",
    "mlp_calibrated_series": "p_mlp_calibrated_series",
    "native_full_mixture": "p_native_full_mixture",
}
CONTRASTS = (
    ("neural_baseline_only_frozen_beta", "neural_calibrated_beta", "linear_offset_to_learned_residual"),
    ("neural_unscaled_iid", "neural_scaled_iid", "neural_scale_under_iid"),
    ("neural_unscaled_frozen_beta", "neural_calibrated_beta", "neural_scale_under_beta"),
    ("neural_unscaled_iid", "neural_unscaled_frozen_beta", "iid_to_beta_unscaled"),
    ("neural_scaled_iid", "neural_calibrated_beta", "iid_to_beta_scaled"),
    ("rating_raw_series", "rating_calibrated_series", "rating_calibration_amplification"),
    ("mlp_raw_series", "mlp_calibrated_series", "mlp_calibration_amplification"),
    ("neural_calibrated_beta", "native_full_mixture", "mixture_vs_calibrated_neural"),
    ("rating_calibrated_series", "native_full_mixture", "mixture_vs_calibrated_rating"),
    ("mlp_calibrated_series", "native_full_mixture", "mixture_vs_calibrated_mlp"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Changed {label}: {path}")
    return actual


def _source_path(protocol: dict, name: str) -> Path:
    value = protocol.get("paths", {}).get(name)
    if value is None:
        raise ValueError(f"Protocol has no resolved source path for {name}")
    return Path(value)


def _load_snapshot(protocol: dict, snapshot: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any], dict[str, str]]:
    feature_dir = ROOT / protocol["feature_dir"]
    prediction_dir = ROOT / protocol["prediction_dir"]
    paths = {
        "arrays": feature_dir / f"{snapshot}_arrays.npz",
        "targets": feature_dir / f"{snapshot}_targets.parquet",
        "snapshot_receipt": feature_dir / f"{snapshot}_receipt.json",
        "predictions": prediction_dir / f"{snapshot}.parquet",
    }
    receipt = _read_json(paths["snapshot_receipt"])
    if receipt.get("status") != "COMPLETE_FROZEN_SNAPSHOT" or receipt.get("snapshot") != snapshot:
        raise ValueError(f"Invalid {snapshot} snapshot receipt")
    expected_hashes = receipt.get("output_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError(f"{snapshot} snapshot receipt omits output hashes")
    input_hashes: dict[str, str] = {}
    for label in ("arrays", "targets", "predictions"):
        path = paths[label]
        key = str(path.relative_to(ROOT))
        expected = expected_hashes.get(key)
        if not expected:
            raise ValueError(f"{snapshot} receipt does not hash {key}")
        input_hashes[key] = _verify_hash(path, str(expected), f"{snapshot} snapshot {label}")
    target = pd.read_parquet(paths["targets"])
    predictions = pd.read_parquet(paths["predictions"])
    if "origin_id" not in target or "origin_id" not in predictions:
        raise ValueError(f"{snapshot} snapshots must retain origin_id")
    target["origin_id"] = target.origin_id.astype("string")
    predictions["origin_id"] = predictions.origin_id.astype("string")
    if target.origin_id.isna().any() or target.origin_id.duplicated().any():
        raise ValueError(f"{snapshot} target origin IDs are not unique")
    if predictions.origin_id.isna().any() or predictions.origin_id.duplicated().any():
        raise ValueError(f"{snapshot} prediction origin IDs are not unique")
    if set(target.origin_id) != set(predictions.origin_id):
        raise ValueError(f"{snapshot} target/prediction inventories differ")
    with np.load(paths["arrays"], allow_pickle=False) as archive:
        if "origin_id" not in archive.files:
            raise ValueError(f"{snapshot} arrays omit origin_id")
        array_ids = archive["origin_id"].astype(str)
        raw = {key.removeprefix("raw_"): archive[key] for key in archive.files if key.startswith("raw_")}
        history = {key.removeprefix("hist_"): archive[key] for key in archive.files if key.startswith("hist_")}
    if not np.array_equal(array_ids, target.origin_id.astype(str).to_numpy()):
        raise ValueError(f"{snapshot} NPZ and target row order differ")
    if len(target) != len(array_ids):
        raise ValueError(f"{snapshot} arrays do not cover every target")
    for name, shape in replay._RAW_SHAPES.items():
        if name not in raw or raw[name].shape != (len(target), *shape):
            raise ValueError(f"{snapshot} raw feature shape mismatch for {name}")
    for name, shape in replay._TEMPORAL_SHAPES.items():
        if name not in history or history[name].shape != (len(target), *shape):
            raise ValueError(f"{snapshot} history feature shape mismatch for {name}")
    feature_available = target.get("feature_available", pd.Series(False, index=target.index)).fillna(False).astype(bool)
    saved = predictions.set_index("origin_id").reindex(target.origin_id)
    saved_available = saved.status.astype("string").eq("available").to_numpy()
    if not np.array_equal(feature_available.to_numpy(), saved_available):
        raise ValueError(f"{snapshot} feature and forecast availability disagree")
    return target, raw, history, saved.reset_index(), receipt, input_hashes


def _json_values(value: Any) -> list[str]:
    if value is None or value is pd.NA:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value if item is not None and item is not pd.NA]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, (list, tuple)):
            return [str(item) for item in decoded if item is not None]
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return []
    return []


def _same_five(value: Any, other: Any) -> bool:
    left, right = _json_values(value), _json_values(other)
    return len(left) == 5 and len(set(left)) == 5 and len(right) == 5 and len(set(right)) == 5 and set(left) == set(right)


def _row_mode(gates: np.ndarray) -> np.ndarray:
    values = np.asarray(gates, dtype=float).copy()
    no_w20 = np.all(values[..., 1] == 0, axis=1)
    no_organization = no_w20 & np.all(values[..., 0] == 0, axis=1)
    return np.where(no_organization, 2, np.where(no_w20, 1, 0)).astype(np.int64)


def _win_probability(log_distribution: np.ndarray) -> np.ndarray:
    values = np.asarray(log_distribution, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Native series projection must return six stopped-score log masses")
    return np.exp(logsumexp(values[:, :3], axis=1))


def _project_map_members(terminal_logs: Any, logits: np.ndarray, best_of: np.ndarray, scale: np.ndarray, rho: np.ndarray) -> np.ndarray:
    n = len(best_of)
    z = np.asarray(logits, dtype=float)
    if z.shape != (n, 3):
        raise ValueError("Three aligned neural map logits are required per series")
    projected = terminal_logs(
        (z * np.asarray(scale, dtype=float)[:, None]).reshape(-1),
        np.repeat(np.asarray(best_of, dtype=int), 3),
        np.repeat(np.asarray(rho, dtype=float), 3),
    ).reshape(n, 3, 6)
    return np.exp(logsumexp(projected[:, :, :3], axis=(1, 2)) - np.log(3.0))


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    result = frame.copy()
    for column in result:
        if not pd.api.types.is_object_dtype(result[column].dtype):
            continue
        if any(isinstance(value, (dict, list, tuple, np.ndarray)) for value in result[column]):
            result[column] = result[column].map(
                lambda value: json.dumps(replay._json_safe(value), allow_nan=False)
                if isinstance(value, (dict, list, tuple, np.ndarray)) else value
            )
    result.to_parquet(path, index=False)


def _capture_stages(
    target: pd.DataFrame,
    raw: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    saved: pd.DataFrame,
    protocol: dict,
    research_root: Path,
    snapshot: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    backend, feature_hashes = replay._activate_feature_backend(research_root)
    native_source_hashes = replay._verify_native_sources(research_root)
    native_module = replay._activate_native_predictor(research_root)
    model_backend = backend["model_backend"]
    beta_calibration = model_backend["beta_calibration"]
    beta_terminal = model_backend["beta_terminal"]
    tail_reliability = model_backend["tail_reliability"]
    repair_receipt_path = _source_path(protocol, "repair_receipt")
    repair_receipt = _read_json(repair_receipt_path)
    stage = target.copy().reset_index(drop=True)
    stage["snapshot"] = snapshot
    stage["snapshot_available"] = stage.feature_available.fillna(False).astype(bool)
    mask = np.asarray(history["mask"], dtype=bool)
    values = np.asarray(history["history"])
    token_count = mask.sum(axis=(1, 2, 3))
    complete_stats = np.zeros(len(stage), dtype=bool)
    for index in np.flatnonzero(token_count == 160):
        present = values[index, ..., 12:24][mask[index]]
        complete_stats[index] = bool(present.shape == (160, 12) and np.all(present > 0))
    stage["complete_history_160"] = (token_count == 160) & complete_stats
    float_columns = ["native_full_max_abs_difference"]
    float_columns.extend(PROBABILITY_COLUMNS.values())
    float_columns.extend(["p_neural_raw_map_probability", "neural_scale", "neural_rho"])
    for seed in range(3):
        float_columns.extend((f"neural_map_logit_seed_{seed}", f"neural_map_probability_seed_{seed}",
                              f"neural_linear_offset_seed_{seed}", f"neural_learned_residual_seed_{seed}"))
    for seed in range(4):
        prefix = "rating" if seed == 0 else f"mlp_seed_{seed-1}"
        float_columns.extend((f"{prefix}_raw_series_logit", f"{prefix}_calibrated_series_logit", f"{prefix}_calibration_slope"))
    for column in float_columns:
        stage[column] = np.nan
    stage["native_available"] = False
    stage["native_full_probability"] = np.nan
    stage["native_no_w20_probability"] = np.nan
    stage["native_no_organization_probability"] = np.nan
    saved_index = saved.set_index("origin_id")
    native_parity: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    model_hashes: dict[str, dict[str, str]] = {}
    mixture_records: dict[str, Any] = {}
    available = stage.snapshot_available.to_numpy(dtype=bool)
    years = pd.to_numeric(stage.model_year, errors="coerce")
    import torch

    for year in sorted(years.loc[available].dropna().astype(int).unique()):
        positions = np.flatnonzero(available & years.eq(year).to_numpy())
        predictor = native_module.CausalA0Predictor(int(year))
        prefix = f"{replay.CAUSAL_ARCHIVE_RELATIVE}/data/06_models/causal_a0/{year}/"
        observed = {str(path).split(prefix, 1)[-1]: value for path, value in predictor.hashes.items()}
        repair_models = repair_receipt.get("models", {}).get(str(year), {}).get("artifact_hashes")
        if not isinstance(repair_models, dict):
            raise ValueError(f"Frozen model artifact hashes are absent for {year}")
        expected_models = {str(path).split(prefix, 1)[-1]: value for path, value in repair_models.items()}
        if expected_models != observed:
            raise ValueError(f"Frozen model artifacts changed for {year}")
        artifact_hashes[str(year)] = observed
        model_hashes[str(year)] = {str(path): str(digest) for path, digest in predictor.hashes.items()}
        parity = {
            "native_full": {"maximum_absolute_difference": 0.0, "rows": int(len(positions))},
            "native_no_w20": {"maximum_absolute_difference": 0.0, "rows": int(len(positions))},
            "native_no_organization": {"maximum_absolute_difference": 0.0, "rows": int(len(positions))},
        }
        for start in range(0, len(positions), 128):
            indices = positions[start:start + 128]
            batch_raw, batch_history = replay._batch_features(raw, history, indices)
            best_of = pd.to_numeric(stage.iloc[indices].best_of, errors="coerce").to_numpy(dtype=int)
            bo_index = (best_of - 1) // 2
            if not np.isin(best_of, (1, 3, 5)).all():
                raise ValueError(f"Unsupported best-of in stage trace for {year}")
            native = np.asarray(predictor.predict(batch_raw, batch_history, best_of), dtype=float)
            if native.shape != (len(indices), 3) or not np.isfinite(native).all():
                raise ValueError(f"Native A0 returned invalid stage-trace output for {year}")
            expected = saved_index.reindex(stage.iloc[indices].origin_id.astype("string"))
            expected_columns = ("p_golgg", "p_no_w20", "p_no_organization")
            if any(column not in expected for column in expected_columns):
                raise ValueError("Saved snapshot predictions omit native optional-mode probabilities")
            reference = expected.loc[:, list(expected_columns)].to_numpy(dtype=float)
            if not np.isfinite(reference).all():
                raise ValueError("Saved snapshot contains non-finite available predictions")
            differences = np.abs(native - reference)
            for mode_index, name in enumerate(parity):
                maximum = float(differences[:, mode_index].max(initial=0.0))
                parity[name]["maximum_absolute_difference"] = max(parity[name]["maximum_absolute_difference"], maximum)
                if maximum > EPSILON:
                    raise ValueError(f"Final native prediction parity failed for {year}/{name}: {maximum}")

            n = len(indices)
            row_ix = np.arange(n, dtype=np.int64)
            map_logits = np.empty((n, 3, 3), dtype=float)
            offsets = np.empty((n, 3), dtype=float)
            residuals = np.empty((n, 3), dtype=float)
            for seed, model in enumerate(predictor.nn):
                model.network.eval()
                for mode_index, (w20, organization) in enumerate(((True, True), (False, True), (False, False))):
                    map_logits[:, mode_index, seed] = np.asarray(
                        model.map_logits(batch_raw, batch_history, None, row_ix, bo_index, w20, organization), dtype=float)
                model_inputs, model_offset = model.batch(batch_raw, batch_history, None, row_ix, bo_index, True, True)
                with torch.no_grad():
                    residual = model.network(model_inputs, torch.as_tensor(bo_index, dtype=torch.long))
                offsets[:, seed] = model_offset.detach().cpu().numpy().astype(float)
                residuals[:, seed] = residual.detach().cpu().numpy().astype(float)
                recomposed = (model_offset + residual).detach().cpu().numpy().astype(float)
                if not np.array_equal(recomposed, map_logits[:, 0, seed]):
                    raise ValueError(f"Neural offset/residual decomposition failed for seed {seed}")
            gates = np.asarray(batch_raw["gates"], dtype=float)
            gate_mode = _row_mode(gates)
            parameters = np.asarray(predictor.parameters, dtype=float)
            if parameters.shape != (3, 2) or not np.isfinite(parameters).all():
                raise ValueError(f"Invalid saved frozen neural calibration parameters for {year}")
            scale = parameters[gate_mode, 0]
            rho = parameters[gate_mode, 1]
            p_unscaled_iid = _project_map_members(beta_terminal.terminal_logs, map_logits[:, 0, :], best_of,
                                                   np.ones(n, dtype=float), np.zeros(n, dtype=float))
            p_scaled_iid = _project_map_members(beta_terminal.terminal_logs, map_logits[:, 0, :], best_of,
                                                 scale, np.zeros(n, dtype=float))
            p_unscaled_beta = _project_map_members(beta_terminal.terminal_logs, map_logits[:, 0, :], best_of,
                                                    np.ones(n, dtype=float), rho)
            beta_distribution = beta_calibration.calibrated_distribution(map_logits, parameters, gates, best_of, 0)
            p_neural = _win_probability(beta_distribution)
            baseline_logits = np.repeat(offsets[:, None, :], 3, axis=1)
            baseline_distribution = beta_calibration.calibrated_distribution(baseline_logits, parameters, gates, best_of, 0)
            p_baseline_only = _win_probability(baseline_distribution)
            p_raw_map = expit(map_logits[:, 0, :]).mean(axis=1)

            n_experts = np.zeros((n, 3, 7), dtype=float)
            slopes = np.ones((3, 7), dtype=float)
            other_models = predictor.other
            if len(other_models) != 4 or len(predictor.mixes) != 3:
                raise ValueError("Frozen native A0 must expose four experts and three optional-mode mixtures")
            modes = ((True, True), (False, True), (False, False))
            for model_index, model in enumerate(other_models):
                cache = model.prepare(batch_raw, batch_history)
                expert_index = model_index + 3
                for mode_index, (w20, organization) in enumerate(modes):
                    series_logits = np.asarray(
                        model.logits_cached(cache, row_ix, bo_index, w20, organization), dtype=float)
                    if series_logits.shape != (n,):
                        raise ValueError("Native cached expert must return one series logit per origin")
                    n_experts[:, mode_index, expert_index] = series_logits
                    if mode_index == 0:
                        if model_index == 0:
                            stage.loc[indices, "rating_raw_series_logit"] = series_logits
                        else:
                            stage.loc[indices, f"mlp_seed_{model_index-1}_raw_series_logit"] = series_logits
                slopes[:, expert_index] = [float(model.slopes[name]) for name in tail_reliability.MODES]
                del cache
            calibrated_groups_by_mode: list[np.ndarray] = []
            native_experts_by_mode: list[np.ndarray] = []
            neural_by_mode: list[np.ndarray] = []
            for mode_index in range(3):
                groups = tail_reliability.calibrated_experts(n_experts, slopes, gates, mode_index)
                distribution = beta_calibration.calibrated_distribution(
                    map_logits, parameters, gates, best_of, mode_index)
                neural_probability = _win_probability(distribution)
                experts = groups.copy()
                experts[:, 0] = neural_probability
                calibrated_groups_by_mode.append(groups)
                native_experts_by_mode.append(experts)
                neural_by_mode.append(neural_probability)
            calibrated_groups = calibrated_groups_by_mode[0]
            p_neural = neural_by_mode[0]
            p_rating_raw = expit(n_experts[:, 0, 3])
            p_mlp_raw = expit(n_experts[:, 0, 4:7]).mean(axis=1)
            p_rating_calibrated = calibrated_groups[:, 1]
            p_mlp_calibrated = calibrated_groups[:, 2]
            row_slopes = slopes[gate_mode]

            for mix_index, mixture in enumerate(predictor.mixes):
                mode_name = ("full", "no_w20", "no_organization")[mix_index]
                mix_context = np.zeros((n, len(mixture.mean)), dtype=float)
                native_mode_probability = np.asarray(
                    mixture.predict(native_experts_by_mode[mix_index], mix_context), dtype=float)
                if not np.allclose(native_mode_probability, native[:, mix_index], atol=EPSILON, rtol=0):
                    raise ValueError(f"Captured expert stages do not reproduce native {mode_name} mixture for {year}")
                _, weights = mixture.distribution(
                    mixture.theta, native_experts_by_mode[mix_index], mixture.design(mix_context))
                if weights is None:
                    raise ValueError("Native mixture did not expose its unchanged expert weights")
                weights = np.asarray(weights, dtype=float)
                if not np.isfinite(weights).all() or weights.shape != (n, 3):
                    raise ValueError("Native mixture weights are invalid")
                weight_key = f"{year}/{mode_name}"
                current_weights = weights[0].tolist()
                prior = mixture_records.get(weight_key)
                if prior is not None and not np.allclose(prior["weights"], current_weights, atol=1e-12, rtol=0):
                    raise ValueError(f"Native mixture weights changed between capture batches for {weight_key}")
                mixture_records[weight_key] = {
                    "kind": str(mixture.kind),
                    "weights": current_weights,
                    "max_row_weight_difference": max(
                        float(prior["max_row_weight_difference"]) if prior else 0.0,
                        float(np.max(np.abs(weights - weights[0]))) if n else 0.0),
                    "constant_weights": bool(np.allclose(weights, weights[0], atol=1e-12, rtol=0)),
                    "context": "all-zero native inference context",
                }
                if not mixture_records[weight_key]["constant_weights"]:
                    raise ValueError(f"Native mixture weights are not constant for {weight_key}")
            mixture_records[f"{year}/experts"] = {
                "raw_rating_probability": "expit(logits_cached series logit); no additional best-of projection",
                "calibrated_rating_probability": "native calibrated_experts output",
                "raw_mlp_probability": "mean member expit(logits_cached series logits); no additional best-of projection",
                "calibrated_mlp_probability": "native calibrated_experts output",
                "neural_probability": "native calibrated_distribution stopped-series projection",
            }

            native_full = native[:, 0]
            stage.loc[indices, "native_available"] = True
            stage.loc[indices, "native_full_probability"] = native[:, 0]
            stage.loc[indices, "native_no_w20_probability"] = native[:, 1]
            stage.loc[indices, "native_no_organization_probability"] = native[:, 2]
            stage.loc[indices, "native_full_max_abs_difference"] = np.abs(native[:, 0] - reference[:, 0])
            stage.loc[indices, "p_neural_unscaled_iid"] = p_unscaled_iid
            stage.loc[indices, "p_neural_scaled_iid"] = p_scaled_iid
            stage.loc[indices, "p_neural_unscaled_frozen_beta"] = p_unscaled_beta
            stage.loc[indices, "p_neural_calibrated_beta"] = p_neural
            stage.loc[indices, "p_neural_baseline_only_frozen_beta"] = p_baseline_only
            stage.loc[indices, "p_neural_raw_map_probability"] = p_raw_map
            stage.loc[indices, "p_rating_raw_series"] = p_rating_raw
            stage.loc[indices, "p_rating_calibrated_series"] = p_rating_calibrated
            stage.loc[indices, "p_mlp_raw_series"] = p_mlp_raw
            stage.loc[indices, "p_mlp_calibrated_series"] = p_mlp_calibrated
            stage.loc[indices, "p_native_full_mixture"] = native_full
            stage.loc[indices, "neural_scale"] = scale
            stage.loc[indices, "neural_rho"] = rho
            for seed in range(3):
                stage.loc[indices, f"neural_map_logit_seed_{seed}"] = map_logits[:, 0, seed]
                stage.loc[indices, f"neural_map_probability_seed_{seed}"] = expit(map_logits[:, 0, seed])
                stage.loc[indices, f"neural_linear_offset_seed_{seed}"] = offsets[:, seed]
                stage.loc[indices, f"neural_learned_residual_seed_{seed}"] = residuals[:, seed]
            stage.loc[indices, "rating_raw_series_logit"] = n_experts[:, 0, 3]
            stage.loc[indices, "rating_calibrated_series_logit"] = np.clip(
                n_experts[:, 0, 3] * row_slopes[:, 3], -20.0, 20.0)
            stage.loc[indices, "rating_calibration_slope"] = row_slopes[:, 3]
            stage.loc[indices, "mlp_raw_series_logit"] = n_experts[:, 0, 4:7].mean(axis=1)
            calibrated_mlp_logits = np.clip(n_experts[:, 0, 4:7] * row_slopes[:, 4:7], -20.0, 20.0)
            stage.loc[indices, "mlp_calibrated_series_logit"] = calibrated_mlp_logits.mean(axis=1)
            for model_index in range(3):
                stage.loc[indices, f"mlp_seed_{model_index}_calibrated_series_logit"] = calibrated_mlp_logits[:, model_index]
                stage.loc[indices, f"mlp_seed_{model_index}_calibration_slope"] = row_slopes[:, 4 + model_index]
            del batch_raw, batch_history, native
        native_parity[str(year)] = parity
        del predictor
        gc.collect()
    receipt = {
        "snapshot": snapshot,
        "native_source_hashes": native_source_hashes,
        "feature_backend_hashes": feature_hashes,
        "repair_receipt_sha256": sha256(repair_receipt_path),
        "artifact_hashes_by_year": artifact_hashes,
        "model_hashes_by_year": model_hashes,
        "native_final_parity_by_year": native_parity,
        "mixtures": mixture_records,
    }
    return stage, receipt, {"native_source_hashes": native_source_hashes, "feature_backend_hashes": feature_hashes,
                            "artifact_hashes_by_year": artifact_hashes, "model_hashes_by_year": model_hashes,
                            "native_final_parity_by_year": native_parity, "mixtures": mixture_records}


def _flatten_interval(prefix: str, interval: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_estimate": interval.get("estimate"),
        f"{prefix}_equal_block_estimate": interval.get("equal_block_estimate"),
        f"{prefix}_blocks": interval.get("blocks"),
        f"{prefix}_active_blocks": interval.get("active_blocks"),
        f"{prefix}_ci_lower": interval.get("ci_lower"),
        f"{prefix}_ci_upper": interval.get("ci_upper"),
        f"{prefix}_draws": interval.get("draws"),
        f"{prefix}_valid_replicates": interval.get("valid_replicates"),
        f"{prefix}_undefined_replicates": interval.get("empty_replicates"),
        f"{prefix}_bootstrap_status": interval.get("status"),
    }


def _aligned_interval(universe: pd.DataFrame, rows: pd.DataFrame, values: np.ndarray, valid: np.ndarray,
                      block_column: str, draws: int, seed: int) -> dict[str, Any]:
    ids = universe.origin_id.astype(str).to_numpy()
    index = {origin_id: position for position, origin_id in enumerate(ids)}
    numerator = np.zeros(len(universe), dtype=float)
    denominator = np.zeros(len(universe), dtype=float)
    for origin_id, value, keep in zip(rows.origin_id.astype(str), np.asarray(values, dtype=float), np.asarray(valid, dtype=bool)):
        if origin_id not in index:
            raise ValueError(f"Scored origin is outside the fixed preselection universe: {origin_id}")
        if keep:
            position = index[origin_id]
            if denominator[position] != 0:
                raise ValueError(f"Repeated scored origin in block interval: {origin_id}")
            numerator[position] = float(value)
            denominator[position] = 1.0
    blocks = universe[block_column].fillna("unknown").astype(str).to_numpy()
    return block_ratio_interval(numerator, denominator, blocks, draws, seed)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y, dtype=float)
    probabilities = np.asarray(p, dtype=float)
    if not len(labels):
        return {"n_scored": 0, "auc": None, "logloss": None, "brier": None, "ece": None, "accuracy_0_5": None}
    if not np.isin(labels, (0.0, 1.0)).all() or not np.isfinite(probabilities).all():
        raise ValueError("Stage metrics require finite probabilities and binary outcomes")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Stage probability is outside [0,1]")
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    if np.unique(labels).size == 2:
        return {"n_scored": len(labels), **evaluate_binary_probabilities(
            labels.astype(int), probabilities, epsilon=EPSILON, include_accuracy=True)}
    return {
        "n_scored": len(labels), "sample_size": float(len(labels)), "auc": None,
        "logloss": float(binary_log_loss_vector(labels.astype(int), probabilities, epsilon=EPSILON).mean()),
        "brier": float(np.mean((labels - probabilities) ** 2)),
        "ece": float(calculate_ece(labels.astype(int), probabilities)),
        "accuracy_0_5": float(np.mean((probabilities >= 0.5) == labels)),
    }


def _audit_inputs(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    sources = protocol.get("sources", {})
    for name in ("audit_origins", "audit_opportunities", "audit_bets"):
        path = _source_path(protocol, name)
        record = sources.get(name, {})
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not expected:
            raise ValueError(f"Protocol does not pin {name}")
        hashes[str(path)] = _verify_hash(path, str(expected), f"frozen {name}")
        frames[name] = pd.read_parquet(path)
    return frames["audit_origins"], frames["audit_opportunities"], frames["audit_bets"], hashes


def _primary_and_selections(stage: pd.DataFrame, audit_origins: pd.DataFrame,
                            opportunities: pd.DataFrame, bets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_mask = (opportunities.cohort.astype("string").eq("oddspapi_common")
                    & opportunities.source.astype("string").eq("best_price")
                    & opportunities.horizon.astype("string").eq("T-1h")
                    & opportunities.audit_primary_eligible.fillna(False).astype(bool))
    primary_source = opportunities.loc[primary_mask].copy()
    if len(primary_source) != PRIMARY_N or primary_source.origin_id.astype(str).duplicated().any():
        raise ValueError(f"Attempt9 primary cohort changed: expected {PRIMARY_N} unique fixtures")
    fixed_mask = (bets.cohort.astype("string").eq("oddspapi_common")
                  & bets.source.astype("string").eq("best_price")
                  & bets.horizon.astype("string").eq("T-1h")
                  & bets.quality.astype("string").eq("exclude_q25_conflicts")
                  & bets.policy.astype("string").eq("raw_ev_gt_5"))
    selected = bets.loc[fixed_mask].copy()
    if len(selected) != FIXED_N or selected.origin_id.astype(str).duplicated().any():
        raise ValueError(f"Attempt9 fixed selections changed: expected {FIXED_N} unique fixtures")
    if primary_source.origin_id.isna().any() or selected.origin_id.isna().any():
        raise ValueError("Primary origins and fixed selections must have origin_id")
    if not set(selected.origin_id.astype(str)).issubset(set(primary_source.origin_id.astype(str))):
        raise ValueError("A fixed attempt9 selection is outside the primary cohort")
    if stage.origin_id.astype(str).duplicated().any():
        raise ValueError("Stage capture has duplicate origin IDs")
    stage_ids = set(stage.origin_id.astype(str))
    for label, frame in (("primary", primary_source), ("fixed selections", selected)):
        unmatched = set(frame.origin_id.astype(str)) - stage_ids
        if unmatched:
            raise ValueError(f"{label} origins are missing from the frozen snapshot: {len(unmatched)}")
    primary_source["origin_id"] = primary_source.origin_id.astype("string")
    selected["origin_id"] = selected.origin_id.astype("string")

    stage_fields = [
        "origin_id", *PROBABILITY_COLUMNS.values(), "p_neural_raw_map_probability",
        "native_available", "native_full_probability", "native_no_w20_probability",
        "native_no_organization_probability", "complete_history_160",
        "missing_reason", "team1_id", "team2_id", "best_of", "y", "roster_a", "roster_b",
        "tournament_name", "competition_family",
    ]
    stage_join = stage[[column for column in stage_fields if column in stage]].copy()
    rename = {
        column: f"snapshot_{column}"
        for column in ("team1_id", "team2_id", "best_of", "y", "roster_a", "roster_b",
                       "tournament_name", "competition_family", "missing_reason")
        if column in stage_join
    }
    stage_join = stage_join.rename(columns=rename)
    origin_fields = [column for column in (
        "origin_id", "canonical_roster_a", "canonical_roster_b", "canonical_same_cutoff_day",
        "tournament_name", "edition_proxy", "competition_family",
    ) if column in audit_origins.columns]
    origin_meta = audit_origins[origin_fields].drop_duplicates("origin_id").copy()
    origin_meta["origin_id"] = origin_meta.origin_id.astype("string")
    origin_meta = origin_meta.rename(columns={
        "canonical_roster_a": "audit_canonical_roster_a",
        "canonical_roster_b": "audit_canonical_roster_b",
        "canonical_same_cutoff_day": "audit_canonical_same_cutoff_day",
        "tournament_name": "origin_tournament_name",
        "edition_proxy": "origin_edition_proxy",
        "competition_family": "origin_competition_family",
    })

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        merged = frame.merge(stage_join, on="origin_id", how="left", validate="one_to_one")
        merged = merged.merge(origin_meta, on="origin_id", how="left", validate="one_to_one")
        if "native_available" not in merged or "native_full_probability" not in merged:
            raise ValueError("Stage join omits native availability or the full native forecast")
        for column, preferred in (
            ("tournament_name", "snapshot_tournament_name"),
            ("competition_family", "snapshot_competition_family"),
        ):
            original = merged.get(column, pd.Series(np.nan, index=merged.index))
            snapshot_value = merged.get(preferred, pd.Series(np.nan, index=merged.index))
            origin_value = merged.get(f"origin_{column}", pd.Series(np.nan, index=merged.index))
            merged[column] = snapshot_value.combine_first(original).combine_first(origin_value)
        if "snapshot_missing_reason" in merged:
            merged["missing_reason"] = merged.snapshot_missing_reason
        cutoff = merged.get("audit_canonical_same_cutoff_day", pd.Series(False, index=merged.index))
        cutoff = cutoff.fillna(False).astype(bool)
        roster_a = merged.get("snapshot_roster_a", pd.Series(None, index=merged.index, dtype=object))
        roster_b = merged.get("snapshot_roster_b", pd.Series(None, index=merged.index, dtype=object))
        canonical_a = merged.get("audit_canonical_roster_a", pd.Series(None, index=merged.index, dtype=object))
        canonical_b = merged.get("audit_canonical_roster_b", pd.Series(None, index=merged.index, dtype=object))
        merged["same_players_and_cutoff"] = [
            bool(cut and _same_five(a, ca) and _same_five(b, cb))
            for a, b, ca, cb, cut in zip(roster_a, roster_b, canonical_a, canonical_b, cutoff)
        ]
        for column in ("team1_id", "team2_id", "best_of", "y"):
            snapshot_column = f"snapshot_{column}"
            if column in merged and snapshot_column in merged:
                if column in {"best_of", "y"}:
                    same = np.array_equal(pd.to_numeric(merged[column], errors="raise").to_numpy(float),
                                          pd.to_numeric(merged[snapshot_column], errors="raise").to_numpy(float))
                else:
                    same = merged[column].astype("string").equals(merged[snapshot_column].astype("string"))
                if not same:
                    raise ValueError(f"Attempt9 and snapshot orientation/label changed: {column}")
        if "date" not in merged:
            raise ValueError("Frozen attempt9 rows omit their event date")
        dates = pd.to_datetime(merged.date, utc=True, errors="coerce")
        merged["month"] = dates.dt.strftime("%Y-%m").fillna("unknown")
        existing_proxy = merged.get("edition_proxy", pd.Series(np.nan, index=merged.index))
        origin_proxy = merged.get("origin_edition_proxy", pd.Series(np.nan, index=merged.index))
        proxy = existing_proxy.combine_first(origin_proxy).fillna("unknown").astype(str)
        unknown = proxy.eq("unknown")
        proxy.loc[unknown] = (
            merged.loc[unknown, "tournament_name"].fillna("unknown").astype(str)
            + ":" + dates.loc[unknown].dt.strftime("%Y").fillna("unknown")
        )
        merged["edition_proxy"] = proxy
        return merged

    primary = attach(primary_source)
    if len(primary) != PRIMARY_N or primary.origin_id.isna().any() or primary.native_full_probability.isna().all():
        raise ValueError("Primary stage join changed or failed to align with the frozen inventory")
    fixed = attach(selected)
    if len(fixed) != FIXED_N:
        raise ValueError("Fixed selection join changed its denominator")
    if not fixed.side.astype("string").isin(("A", "B")).all():
        raise ValueError("Saved selection side must be A or B")
    if "won" not in fixed or fixed.won.isna().any():
        raise ValueError("Saved fixed selections must retain settled outcomes")
    target_column = "snapshot_y" if "snapshot_y" in fixed else "y"
    if target_column not in fixed:
        raise ValueError("Snapshot target labels are unavailable for fixed selections")
    target_y = pd.to_numeric(fixed[target_column], errors="coerce").to_numpy(float)
    if not np.isfinite(target_y).all() or not np.isin(target_y, (0.0, 1.0)).all():
        raise ValueError("Snapshot labels for fixed selections must be finite binary outcomes")
    selected_y = np.where(fixed.side.astype(str).to_numpy() == "A", target_y, 1.0 - target_y)
    if not np.array_equal(selected_y.astype(bool), fixed.won.astype(bool).to_numpy()):
        raise ValueError("Saved selection outcome does not match snapshot team orientation")
    fixed["selected_y"] = selected_y
    fixed["p_saved_selected"] = pd.to_numeric(fixed.p_model_bet, errors="coerce")
    fixed["p_native_selected"] = np.where(
        fixed.side.astype(str).eq("A"), fixed.native_full_probability, 1.0 - fixed.native_full_probability)
    if not set(fixed.origin_id.astype(str)).issubset(set(primary.origin_id.astype(str))):
        raise ValueError("Fixed selections are not represented in the attempt9 primary denominator")
    return primary, fixed


def _population_frames(primary: pd.DataFrame, fixed: pd.DataFrame) -> list[tuple[str, pd.DataFrame, str, str, bool]]:
    populations: list[tuple[str, pd.DataFrame, str, str, bool]] = [
        ("full_primary", primary, "y", "series", False),
        ("same_players_and_cutoff", primary.loc[primary.same_players_and_cutoff], "y", "series", False),
        ("nacl", primary.loc[primary.tournament_name.astype("string").str.contains("NACL", case=False, na=False)], "y", "series", False),
    ]
    best_of = pd.to_numeric(primary.best_of, errors="coerce")
    for value in (1, 3, 5):
        populations.append((f"bo{value}", primary.loc[best_of.eq(value)], "y", "series", value == 1))
    populations.extend((
        ("selected_all", fixed, "selected_y", "selection", False),
        ("selected_wins", fixed.loc[fixed.won.astype(bool)], "selected_y", "selection", False),
        ("selected_losses", fixed.loc[~fixed.won.astype(bool)], "selected_y", "selection", False),
    ))
    e_spring = fixed.tournament_name.astype("string").str.contains(r"(?i)(?=.*emea)(?=.*spring)", na=False, regex=True)
    complete = fixed.get("complete_history_160", pd.Series(False, index=fixed.index)).fillna(False).astype(bool)
    populations.append(("emea_spring_complete_history_selections", fixed.loc[e_spring & complete], "selected_y", "selection", False))
    return populations


def _stage_probability(frame: pd.DataFrame, stage_name: str, kind: str, best_of_one_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    column = "p_neural_raw_map_probability" if stage_name == MAP_STAGE else PROBABILITY_COLUMNS[stage_name]
    probabilities = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    allowed = np.isfinite(probabilities)
    if best_of_one_only:
        allowed &= pd.to_numeric(frame.best_of, errors="coerce").to_numpy(dtype=float) == 1.0
    if kind == "selection":
        side_a = frame.side.astype("string").eq("A").to_numpy()
        probabilities = np.where(side_a, probabilities, 1.0 - probabilities)
    return probabilities, allowed


def _score_table(primary: pd.DataFrame, fixed: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores: list[dict[str, Any]] = []
    populations = _population_frames(primary, fixed)
    stage_names = list(PROBABILITY_COLUMNS)
    for population, frame, target_column, kind, bo1_only in populations:
        if frame.empty:
            continue
        labels = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
        valid_y = np.isfinite(labels) & np.isin(labels, (0.0, 1.0))
        names = [*stage_names]
        if population in {"bo1", "selected_all", "selected_wins", "selected_losses"}:
            names.append(MAP_STAGE)
        for stage_name in names:
            stage_map_only = stage_name == MAP_STAGE
            p, allowed = _stage_probability(frame, stage_name, kind, best_of_one_only=stage_map_only)
            scored = allowed & valid_y
            metric = _metrics(labels[scored], p[scored])
            loss = np.full(len(frame), np.nan, dtype=float)
            brier = np.full(len(frame), np.nan, dtype=float)
            if scored.any():
                loss[scored] = binary_log_loss_vector(labels[scored].astype(int), p[scored], epsilon=EPSILON)
                brier[scored] = (labels[scored] - np.clip(p[scored], EPSILON, 1.0 - EPSILON)) ** 2
            row: dict[str, Any] = {
                "population": population,
                "stage": stage_name,
                "stage_semantics": "raw map probability; scored only on BO1" if stage_map_only else "series-winner probability",
                "n_inventory": int(len(frame)),
                "n_scored": int(scored.sum()),
                "n_missing_or_ineligible": int(len(frame) - scored.sum()),
                "n_missing_forecast": int((~frame.native_available.fillna(False).to_numpy(dtype=bool)).sum()),
                **metric,
            }
            for measure, values in (("logloss", loss), ("brier", brier)):
                keep = scored & np.isfinite(values)
                for block_column, label in (("month", "month"), ("edition_proxy", "edition_proxy")):
                    if block_column in primary:
                        interval = _aligned_interval(primary, frame, np.where(keep, values, 0.0), keep,
                                                     block_column, draws, seed)
                        row.update(_flatten_interval(f"{measure}_{label}", interval))
            if "missing_reason" in frame and not frame.empty:
                missing = frame.loc[~frame.native_available.fillna(False).astype(bool), "missing_reason"].astype("string").fillna("unknown")
                row["missing_reasons_json"] = json.dumps(missing.value_counts(dropna=False).to_dict(), sort_keys=True)
            scores.append(row)
    score_frame = pd.DataFrame(scores)
    fixed_rows = _fixed_selection_rows(fixed)
    fixed_summary = _fixed_selection_summary(primary, fixed_rows, draws, seed)
    return score_frame, fixed_summary, fixed_rows


def _fixed_selection_rows(fixed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    factor = 1.0 - TAX_RATE
    for stage_name, column in {**PROBABILITY_COLUMNS, MAP_STAGE: "p_neural_raw_map_probability"}.items():
        values = pd.to_numeric(fixed[column], errors="coerce").to_numpy(dtype=float)
        eligible = np.isfinite(values)
        if stage_name == MAP_STAGE:
            eligible &= pd.to_numeric(fixed.best_of, errors="coerce").to_numpy(dtype=float) == 1.0
        selected_probability = np.where(fixed.side.astype(str).to_numpy() == "A", values, 1.0 - values)
        selected_probability[~eligible] = np.nan
        odds = pd.to_numeric(fixed.odds, errors="coerce").to_numpy(dtype=float)
        won = fixed.won.to_numpy(dtype=float)
        expected_ev = factor * odds * selected_probability - 1.0
        realized_ev = factor * odds * won - 1.0
        ev_error = expected_ev - realized_ev
        for index, source in fixed.reset_index(drop=True).iterrows():
            rows.append({
                "origin_id": str(source.origin_id),
                "fixture_id": source.fixture_id,
                "golgg_match_id": source.golgg_match_id,
                "date": source.date,
                "month": source.get("month"),
                "edition_proxy": source.get("edition_proxy"),
                "tournament_name": source.get("tournament_name"),
                "competition_family": source.get("competition_family"),
                "best_of": source.best_of,
                "side": source.side,
                "odds": odds[index],
                "won": bool(won[index]),
                "stage": stage_name,
                "stage_semantics": "raw map probability; eligible only on BO1" if stage_name == MAP_STAGE else "series-winner probability",
                "native_available": bool(source.native_available),
                "stage_available": bool(eligible[index]),
                "missing_reason": source.get("missing_reason"),
                "p_team1": values[index] if eligible[index] else np.nan,
                "p_selected": selected_probability[index],
                "expected_ev_net_per_unit": expected_ev[index] if eligible[index] else np.nan,
                "realized_ev_net_per_unit": realized_ev[index] if eligible[index] else np.nan,
                "ev_error_expected_minus_realized": ev_error[index] if eligible[index] else np.nan,
                "probability_error_selected_side": selected_probability[index] - won[index] if eligible[index] else np.nan,
                "complete_history_160": bool(source.get("complete_history_160", False)),
                "same_players_and_cutoff": bool(source.get("same_players_and_cutoff", False)),
            })
    return pd.DataFrame(rows)


def _fixed_selection_summary(primary: pd.DataFrame, rows: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    result: list[dict[str, Any]] = []
    for stage_name, frame in rows.groupby("stage", sort=False):
        valid = frame.stage_available.to_numpy(dtype=bool)
        forecast_available = frame.native_available.to_numpy(dtype=bool)
        p = pd.to_numeric(frame.p_selected, errors="coerce").to_numpy(dtype=float)
        ev_error = pd.to_numeric(frame.ev_error_expected_minus_realized, errors="coerce").to_numpy(dtype=float)
        expected = pd.to_numeric(frame.expected_ev_net_per_unit, errors="coerce").to_numpy(dtype=float)
        realized = pd.to_numeric(frame.realized_ev_net_per_unit, errors="coerce").to_numpy(dtype=float)
        row: dict[str, Any] = {
            "stage": stage_name,
            "stage_semantics": str(frame.stage_semantics.iloc[0]),
            "n_inventory": int(len(frame)),
            "n_scored": int(valid.sum()),
            "n_missing_or_ineligible": int(len(frame) - valid.sum()),
            "n_missing_forecast": int((~forecast_available).sum()),
            "n_stage_ineligible_available_forecast": int((forecast_available & ~valid).sum()),
            "mean_selected_probability": float(np.mean(p[valid])) if valid.any() else None,
            "mean_expected_ev_net_per_unit": float(np.mean(expected[valid])) if valid.any() else None,
            "mean_realized_ev_net_per_unit": float(np.mean(realized[valid])) if valid.any() else None,
            "mean_ev_error_expected_minus_realized": float(np.mean(ev_error[valid])) if valid.any() else None,
            "sum_ev_error_expected_minus_realized": float(np.sum(ev_error[valid])) if valid.any() else None,
            "mean_absolute_ev_error": float(np.mean(np.abs(ev_error[valid]))) if valid.any() else None,
            "tax_rate": TAX_RATE,
            "n_wins_inventory": int(frame.won.sum()),
            "n_losses_inventory": int((~frame.won).sum()),
        }
        for measure, values in (("ev_error", ev_error), ("expected_ev", expected), ("realized_ev", realized)):
            measure_valid = valid & np.isfinite(values)
            for block_column, label in (("month", "month"), ("edition_proxy", "edition_proxy")):
                interval = _aligned_interval(primary, frame, np.where(measure_valid, values, 0.0), measure_valid,
                                             block_column, draws, seed)
                row.update(_flatten_interval(f"{measure}_{label}", interval))
        missing = frame.loc[~forecast_available, "missing_reason"].astype("string").fillna("unknown")
        row["missing_reasons_json"] = json.dumps(missing.value_counts(dropna=False).to_dict(), sort_keys=True)
        result.append(row)
    return pd.DataFrame(result)


def _contrast_tables(primary: pd.DataFrame, fixed: pd.DataFrame, fixed_rows: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed_by_stage = {stage: group for stage, group in fixed_rows.groupby("stage", sort=False)}
    for population, frame, target_column, kind, _ in _population_frames(primary, fixed):
        if frame.empty:
            continue
        labels = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float)
        valid_y = np.isfinite(labels) & np.isin(labels, (0.0, 1.0))
        for base_name, alt_name, contrast_name in CONTRASTS:
            if kind == "selection":
                base_frame = fixed_by_stage.get(base_name)
                alt_frame = fixed_by_stage.get(alt_name)
                if base_frame is None or alt_frame is None:
                    continue
                joined = frame[["origin_id", target_column]].copy()
                joined = joined.merge(base_frame[["origin_id", "p_selected"]].rename(columns={"p_selected": "base_p"}), on="origin_id", how="left", validate="one_to_one")
                joined = joined.merge(alt_frame[["origin_id", "p_selected"]].rename(columns={"p_selected": "alt_p"}), on="origin_id", how="left", validate="one_to_one")
                base_p = pd.to_numeric(joined.base_p, errors="coerce").to_numpy(dtype=float)
                alt_p = pd.to_numeric(joined.alt_p, errors="coerce").to_numpy(dtype=float)
            else:
                base_p = pd.to_numeric(frame[PROBABILITY_COLUMNS[base_name]], errors="coerce").to_numpy(dtype=float)
                alt_p = pd.to_numeric(frame[PROBABILITY_COLUMNS[alt_name]], errors="coerce").to_numpy(dtype=float)
            common = valid_y & np.isfinite(base_p) & np.isfinite(alt_p)
            if not common.any():
                continue
            base_loss = binary_log_loss_vector(labels[common].astype(int), base_p[common], epsilon=EPSILON)
            alt_loss = binary_log_loss_vector(labels[common].astype(int), alt_p[common], epsilon=EPSILON)
            base_brier = (labels[common] - np.clip(base_p[common], EPSILON, 1.0 - EPSILON)) ** 2
            alt_brier = (labels[common] - np.clip(alt_p[common], EPSILON, 1.0 - EPSILON)) ** 2
            record: dict[str, Any] = {
                "population": population,
                "sample": kind,
                "contrast": contrast_name,
                "base_stage": base_name,
                "alt_stage": alt_name,
                "delta_direction": "alt_minus_base; positive logloss/brier/EV error is worse",
                "n_inventory": int(len(frame)),
                "n_common_scored": int(common.sum()),
                "n_unpaired_or_missing": int(len(frame) - common.sum()),
                "mean_probability_change_alt_minus_base": float(np.mean(alt_p[common] - base_p[common])),
                "mean_logloss_delta_alt_minus_base": float(np.mean(alt_loss - base_loss)),
                "mean_brier_delta_alt_minus_base": float(np.mean(alt_brier - base_brier)),
                "mean_absolute_probability_change": float(np.mean(np.abs(alt_p[common] - base_p[common]))),
                "bootstrap_universe": "all 1852 attempt9 primary fixtures, including months/editions with no scored rows",
            }
            for measure, delta in (("logloss", alt_loss - base_loss), ("brier", alt_brier - base_brier),
                                   ("probability_change", alt_p[common] - base_p[common])):
                aligned_values = np.zeros(len(frame), dtype=float)
                aligned_values[common] = delta
                for block_column, label in (("month", "month"), ("edition_proxy", "tournament_name_year_proxy")):
                    interval = _aligned_interval(primary, frame, aligned_values, common, block_column, draws, seed)
                    record.update(_flatten_interval(f"{measure}_{label}", interval))
            rows.append(record)
    # Selected-side EV error contrasts use only fixed saved sides and the same complete primary universe.
    for base_name, alt_name, contrast_name in CONTRASTS:
        base_frame = fixed_by_stage.get(base_name)
        alt_frame = fixed_by_stage.get(alt_name)
        if base_frame is None or alt_frame is None:
            continue
        joined = base_frame[["origin_id", "won", "odds", "p_selected"]].rename(columns={"p_selected": "base_p"})
        joined = joined.merge(alt_frame[["origin_id", "p_selected"]].rename(columns={"p_selected": "alt_p"}), on="origin_id", how="inner", validate="one_to_one")
        odds = pd.to_numeric(joined.odds, errors="coerce").to_numpy(dtype=float)
        y = joined.won.to_numpy(dtype=float)
        base_p = pd.to_numeric(joined.base_p, errors="coerce").to_numpy(dtype=float)
        alt_p = pd.to_numeric(joined.alt_p, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(odds) & np.isfinite(y) & np.isfinite(base_p) & np.isfinite(alt_p)
        if not valid.any():
            continue
        multiplier = (1.0 - TAX_RATE) * odds
        base_error = multiplier * (base_p - y)
        alt_error = multiplier * (alt_p - y)
        delta = alt_error - base_error
        record = {
            "population": "fixed_selection_ev_error",
            "sample": "selection",
            "contrast": contrast_name,
            "base_stage": base_name,
            "alt_stage": alt_name,
            "delta_direction": "alt_minus_base; positive expected-minus-realized EV error is worse",
            "n_inventory": FIXED_N,
            "n_common_scored": int(valid.sum()),
            "n_unpaired_or_missing": int(FIXED_N - valid.sum()),
            "mean_ev_error_base": float(base_error[valid].mean()),
            "mean_ev_error_alt": float(alt_error[valid].mean()),
            "mean_ev_error_delta_alt_minus_base": float(delta[valid].mean()),
            "sum_ev_error_delta_alt_minus_base": float(delta[valid].sum()),
            "bootstrap_universe": "all 1852 attempt9 primary fixtures; denominator is the paired fixed saved selections only",
        }
        for block_column, label in (("month", "month"), ("edition_proxy", "tournament_name_year_proxy")):
            interval = _aligned_interval(primary, joined, np.where(valid, delta, 0.0), valid, block_column, draws, seed)
            record.update(_flatten_interval(f"ev_error_delta_{label}", interval))
        rows.append(record)
    return pd.DataFrame(rows)




def _named_cases(primary: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    ids = {
        "81029": "Gen.G–Hanwha Life, match 81029",
        "80660": "BNK FearX–DN SOOPers, match 80660",
        "80743": "Dplus KIA–Hanwha Life, match 80743",
        "78657": "Maryville–Winthrop, match 78657",
    }
    named: list[dict[str, Any]] = []
    fixed_by_match = {str(value): row for value, row in zip(fixed.golgg_match_id.astype(str), fixed.to_dict("records"))}
    primary_by_match = {str(value): row for value, row in zip(primary.golgg_match_id.astype(str), primary.to_dict("records"))}
    for match_id, label in ids.items():
        item = fixed_by_match.get(match_id)
        if item is not None:
            named.append({**item, "case_type": "predeclared_named_match", "case_label": label,
                          "matched_to_match_id": None, "winning_control": bool(item.get("won", False))})
        elif match_id in primary_by_match:
            named.append({**primary_by_match[match_id], "case_type": "named_match_not_a_saved_bet",
                          "case_label": label, "matched_to_match_id": None, "winning_control": False})
    e_spring = fixed.tournament_name.astype("string").str.contains(
        r"(?i)(?=.*emea)(?=.*spring)", na=False, regex=True)
    for item in fixed.loc[e_spring].to_dict("records"):
        named.append({**item, "case_type": "emea_spring_fixed_selection",
                      "case_label": "EMEA Spring fixed selection; complete history flagged separately",
                      "matched_to_match_id": None, "winning_control": bool(item.get("won", False))})
    targets = fixed.loc[fixed.golgg_match_id.astype(str).isin(ids) | e_spring].copy()
    winners = fixed.loc[fixed.won.astype(bool)].copy()
    if not winners.empty:
        labels = [".5-.6", ".6-.7", ".7-.8", ".8-.9", ".9-1"]
        edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
        winner_market = np.maximum(
            pd.to_numeric(winners.q_bet, errors="coerce"),
            1.0 - pd.to_numeric(winners.q_bet, errors="coerce"),
        )
        winners = winners.assign(
            _market_band=pd.cut(winner_market, edges, right=False, labels=labels).astype("string"))
        for target in targets.loc[~targets.won.astype(bool)].itertuples(index=False):
            confidence = pd.to_numeric(pd.Series([target.p_model_bet]), errors="coerce").iloc[0]
            if not np.isfinite(confidence):
                continue
            target_q = pd.to_numeric(pd.Series([target.q_bet]), errors="coerce").iloc[0]
            target_band = None
            if np.isfinite(target_q):
                market = max(float(target_q), 1.0 - float(target_q))
                target_band = pd.cut(pd.Series([market]), edges, right=False, labels=labels).astype("string").iloc[0]
            target_best_of = pd.to_numeric(pd.Series([target.best_of]), errors="coerce").iloc[0]
            same_family_bo = winners.loc[
                winners.competition_family.astype("string").eq(str(target.competition_family))
                & pd.to_numeric(winners.best_of, errors="coerce").eq(target_best_of)
            ]
            if target_band is not None and not pd.isna(target_band):
                family_band = same_family_bo.loc[same_family_bo._market_band.eq(str(target_band))]
                any_band = winners.loc[winners._market_band.eq(str(target_band))]
            else:
                family_band = same_family_bo.iloc[0:0]
                any_band = winners.iloc[0:0]
            if not family_band.empty:
                candidates = family_band
            elif not same_family_bo.empty:
                candidates = same_family_bo
            elif not any_band.empty:
                candidates = any_band
            else:
                candidates = winners
            candidates = candidates.assign(
                _distance=(pd.to_numeric(candidates.p_model_bet, errors="coerce") - float(confidence)).abs())
            control = candidates.sort_values(["_distance", "fixture_id"], kind="mergesort").iloc[0]
            named.append({**control.drop(labels=["_market_band", "_distance"], errors="ignore").to_dict(),
                          "case_type": "matched_winning_control",
                          "case_label": "descriptive nearest-confidence saved win",
                          "matched_to_match_id": str(target.golgg_match_id), "winning_control": True})
    result = pd.DataFrame(named)
    if not result.empty:
        return result.drop_duplicates(subset=["case_type", "origin_id", "matched_to_match_id"], keep="first")
    return pd.DataFrame(columns=["case_type", "case_label", "origin_id", "fixture_id", "golgg_match_id", "winning_control"])


def _write_outputs(output: Path, stage: pd.DataFrame, score_frame: pd.DataFrame,
                   fixed_summary: pd.DataFrame, fixed_rows: pd.DataFrame,
                   contrasts: pd.DataFrame, named: pd.DataFrame, results: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stage diagnostics: {output}")
    output.mkdir(parents=True, exist_ok=False)
    _save_frame(stage, output / "stages.parquet")
    score_frame.to_csv(output / "stage_scores.csv", index=False)
    fixed_summary.to_csv(output / "fixed_selection_scores.csv", index=False)
    _save_frame(fixed_rows, output / "fixed_selection_rows.parquet")
    contrasts.to_csv(output / "stage_contrasts.csv", index=False)
    named.to_csv(output / "named_cases.csv", index=False)
    save_json(output / "results.json", results)


def run(protocol: dict, research_root: Path, snapshot: str) -> dict[str, Any]:
    """Capture native A0 internals and score frozen stages for one snapshot."""
    if snapshot not in {"baseline", "context"}:
        raise ValueError("snapshot must be baseline or context")
    root = Path(research_root).resolve()
    protocol_root = Path(protocol.get("resolved_research_root", root)).resolve()
    if root != protocol_root:
        raise ValueError("research_root differs from the resolved frozen protocol root")
    contract = protocol.get("fixed_contract", {})
    draws = int(contract.get("bootstrap_draws", DRAWS))
    seed = int(contract.get("seed", SEED))
    epsilon = float(contract.get("probability_epsilon", EPSILON))
    tax_rate = float(contract.get("tax_rate", TAX_RATE))
    if (draws, seed, epsilon, tax_rate) != (DRAWS, SEED, EPSILON, TAX_RATE):
        raise ValueError("Stage diagnostics require the frozen attempt10 scoring contract")
    output = ROOT / protocol["report_dir"] / f"stages_{snapshot}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stage diagnostics: {output}")
    target, raw, history, saved, snapshot_receipt, snapshot_hashes = _load_snapshot(protocol, snapshot)
    stage, capture_receipt, capture = _capture_stages(target, raw, history, saved, protocol, root, snapshot)
    if len(stage) != int(snapshot_receipt.get("origins", -1)):
        raise ValueError("Stage capture dropped a snapshot origin")
    receipt_models = snapshot_receipt.get("models", {})
    if set(map(str, receipt_models)) != set(capture["artifact_hashes_by_year"]):
        raise ValueError("Snapshot receipt and native stage capture cover different model years")
    for year, actual in capture["artifact_hashes_by_year"].items():
        expected = receipt_models.get(year, {}).get("artifact_hashes")
        if not isinstance(expected, dict) or expected != actual:
            raise ValueError(f"Saved snapshot model hashes differ from native A0 artifacts for {year}")
    audit_origins, opportunities, bets, audit_hashes = _audit_inputs(protocol)
    primary, fixed = _primary_and_selections(stage, audit_origins, opportunities, bets)
    if snapshot == "baseline":
        baseline_difference = np.abs(pd.to_numeric(primary.p, errors="coerce").to_numpy(float)
                                     - pd.to_numeric(primary.native_full_probability, errors="coerce").to_numpy(float))
        known = np.isfinite(baseline_difference)
        if known.sum() != PRIMARY_N or float(baseline_difference[known].max(initial=0.0)) > EPSILON:
            raise ValueError("Baseline native output does not match attempt9 primary probabilities")
        selection_difference = np.abs(
            pd.to_numeric(fixed.p_saved_selected, errors="coerce").to_numpy(float)
            - pd.to_numeric(fixed.p_native_selected, errors="coerce").to_numpy(float))
        selection_known = fixed.native_available.fillna(False).to_numpy(dtype=bool) & np.isfinite(selection_difference)
        if selection_known.sum() != int(fixed.native_available.fillna(False).sum()) or (
            selection_known.any() and float(selection_difference[selection_known].max(initial=0.0)) > EPSILON
        ):
            raise ValueError("Baseline saved selection probabilities differ from native side orientation")
    origin_fields = [column for column in (
        "origin_id", "canonical_roster_a", "canonical_roster_b", "canonical_same_cutoff_day",
    ) if column in audit_origins.columns]
    origin_meta = audit_origins[origin_fields].drop_duplicates("origin_id").copy()
    origin_meta["origin_id"] = origin_meta.origin_id.astype("string")
    origin_meta = origin_meta.rename(columns={
        "canonical_roster_a": "audit_canonical_roster_a",
        "canonical_roster_b": "audit_canonical_roster_b",
        "canonical_same_cutoff_day": "audit_canonical_same_cutoff_day",
    })
    stage = stage.merge(origin_meta, on="origin_id", how="left", validate="one_to_one")
    if "missing_reason" not in stage:
        stage["missing_reason"] = pd.NA
    stage["same_players_and_cutoff"] = [
        bool(cutoff and _same_five(roster_a, canonical_a) and _same_five(roster_b, canonical_b))
        for roster_a, roster_b, canonical_a, canonical_b, cutoff in zip(
            stage.roster_a, stage.roster_b, stage.audit_canonical_roster_a,
            stage.audit_canonical_roster_b,
            stage.audit_canonical_same_cutoff_day.fillna(False).astype(bool),
        )
    ]
    # Stage summaries use the full attempt9 primary fixture/month/edition universe for every bootstrap.
    score_frame, fixed_summary, fixed_rows = _score_table(primary, fixed, draws, seed)
    contrasts = _contrast_tables(primary, fixed, fixed_rows, draws, seed)
    named = _named_cases(primary, fixed)
    missing_counts = stage.loc[~stage.native_available.astype(bool), "missing_reason"].astype("string").fillna("unknown").value_counts(dropna=False).to_dict()
    native_diffs = [record["maximum_absolute_difference"] for year in capture["native_final_parity_by_year"].values()
                    for name, record in year.items() if not name.startswith("_")]
    results: dict[str, Any] = {
        "status": "COMPLETE_NATIVE_STAGE_TRACE",
        "snapshot": snapshot,
        "run_id": protocol.get("run_id"),
        "classification": protocol.get("classification"),
        "coverage": {
            "snapshot_origin_inventory": int(len(stage)),
            "available_forecasts": int(stage.native_available.sum()),
            "unavailable_forecasts": int((~stage.native_available).sum()),
            "missing_reasons": missing_counts,
            "attempt9_primary_inventory": int(len(primary)),
            "primary_native_available": int(primary.native_available.fillna(False).sum()),
            "attempt9_fixed_selection_inventory": int(len(fixed)),
            "fixed_selection_native_available": int(fixed.native_available.fillna(False).sum()),
            "primary_ids_exactly_matched_to_snapshot": int(primary.origin_id.nunique()),
            "fixed_selection_ids_exactly_matched_to_snapshot": int(fixed.origin_id.nunique()),
            "same_players_and_cutoff_primary_n": int(primary.same_players_and_cutoff.sum()),
            "complete_history_160_fixed_selection_n": int((fixed.complete_history_160.fillna(False).astype(bool)
                                                               & fixed.tournament_name.astype("string").str.contains(r"(?i)(?=.*emea)(?=.*spring)", na=False, regex=True)).sum()),
        },
        "parity": {
            "native_full_and_optional_modes_reproduced_from_saved_snapshot": True,
            "maximum_absolute_difference_across_native_modes": max(native_diffs, default=0.0),
            "absolute_tolerance": EPSILON,
            "baseline_attempt9_primary_probability_max_difference": (
                float(np.nanmax(np.abs(pd.to_numeric(primary.p, errors="coerce").to_numpy(float)
                                       - pd.to_numeric(primary.native_full_probability, errors="coerce").to_numpy(float))))
                if snapshot == "baseline" else None
            ),
            "by_year": capture["native_final_parity_by_year"],
        },
        "stage_contract": {
            "neural_map_logits": "three saved neural seeds; map-level raw logits, linear baseline offset and learned residual retained separately",
            "neural_series_variants": ["unscaled iid", "scaled iid", "unscaled frozen-beta", "scaled frozen-beta (native calibrated neural)",
                                       "baseline-offset-only under the same frozen full calibration (diagnostic intervention only)"],
            "rating_mlp": "logits_cached outputs are already series logits; raw and native calibrated probabilities are recorded without additional best-of projection",
            "native_mixture": "actual frozen full mixture, unchanged weights and all-zero native context",
            "map_scoring_rule": "raw map probability is directly scored only for BO1; other raw neural diagnostics are first projected to series probability",
            "fixed_selection_rule": "attempt9 side and odds remain fixed; no threshold search or reselection",
            "bootstrap": {"method": "block_ratio_interval", "draws": draws, "seed": seed,
                          "month_and_tournament_name_year_proxy": True, "equal_block_estimates_reported": True,
                          "undefined_replicates_reported": True, "universe": "all attempt9 primary 1852 fixtures"},
            "tax_rate": tax_rate,
            "probability_epsilon": epsilon,
            "mixtures": capture["mixtures"],
        },
        "source_hashes": {
            "snapshot_outputs": snapshot_hashes,
            "audit_sources": audit_hashes,
            "snapshot_receipt": sha256(ROOT / protocol["feature_dir"] / f"{snapshot}_receipt.json"),
            "repair_receipt": capture_receipt["repair_receipt_sha256"],
            "stage_module": sha256(Path(__file__).resolve()),
            "native_source_files": capture["native_source_hashes"],
            "feature_backend_sources": capture["feature_backend_hashes"],
            "model_artifacts_by_year": capture["model_hashes_by_year"],
            "artifact_hashes_by_year": capture["artifact_hashes_by_year"],
        },
        "interpretation": {
            "raw_strength": "Raw series-score evidence is summarized separately for neural iid projections and uncalibrated rating/MLP series probabilities.",
            "calibration_amplification": "Frozen beta scale/rho and expert slopes are compared against their own raw stages with paired same-ID score and fixed-side EV effects.",
            "specific_missing_sports_factor": "No specific missing sports factor is identified by this inference-stage trace; stage attribution is not an omitted-variable test.",
            "baseline_only": "Linear-offset-only projection uses frozen full calibration as a diagnostic intervention, not a trained replacement model.",
        },
        "limitations": [
            "Historical scenario labels and tournament-name/year blocks are not certified competition units or lineup announcements.",
            "No fitting, recalibration, deployment, selection change, quote issuance, wager execution, or bookmaker binding validation occurs.",
            "Selected EV errors are on the fixed saved attempt9 sides/odds; missing context forecasts remain explicit and are excluded from score denominators, never treated as wins or zero errors.",
            "Stage comparisons are descriptive; matching does not establish a causal treatment or identify a specific omitted sports variable.",
        ],
        "outputs": {
            "stages": "stages.parquet: one row per snapshot origin, including unavailable rows and raw stage columns",
            "stage_scores": "stage_scores.csv: proper series-probability metrics and monthly/edition proxy intervals by frozen population",
            "fixed_selection_scores": "fixed_selection_scores.csv: stage-wise selected-side EV error summaries on the available saved selection denominator",
            "fixed_selection_rows": "fixed_selection_rows.parquet: one row per saved selection and stage, with fixed side, odds, outcome, coverage and EV error",
            "stage_contrasts": "stage_contrasts.csv: paired same-origin stage effects; primary and subgroup intervals retain the complete preselection universe",
            "named_cases": "named_cases.csv: four predeclared matches, EMEA Spring fixed selections and descriptive nearest-winning controls",
        },
    }
    results["output_dir"] = str(output)
    _write_outputs(output, stage, score_frame, fixed_summary, fixed_rows, contrasts, named, results)
    return results
