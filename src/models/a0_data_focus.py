"""Single-origin, pre-2026 A0 neural population/masking experiment.

The archived implementations and previous experiments remain unchanged. All
arms share the non-neural experts, input scalers, and fixed CAL population.
This is a research assembly, not the frozen annual A0 or a live adapter.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from src.models import a0_uncertainty as uncertainty
from src.models.a0_native_refit import (
    MODES, SEEDS, _atomic_checkpoint, _calculate_forecast, _contract_digest,
    _fit_baseline_normalizer, _json_safe, _sha256, _torch_state_hash,
    _versions, _write_json_exclusive,
)
from src.models.a0_target_experiment import _predict_mixture, resolve_research_root
from src.models.a0_target_objectives import fit_beta_target_on_device
from src.models.a0_history_reconstruction import pretrain_history
from src.models.competition_tiers import classify_competition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("control", "high_tier_only", "market_weighted", "token_dropout", "market_weighted_dropout")
RECONSTRUCTION_VARIANTS = ("control", "supervised_18", "reconstruction")


def experiment_variants(protocol):
    if protocol["schema"] == "a0_data_focus_v1":
        expected = VARIANTS
    elif protocol["schema"] == "a0_history_reconstruction_v1":
        expected = RECONSTRUCTION_VARIANTS
        recipe = protocol["reconstruction"]
        if (recipe["probability"] != .25 or recipe["stats"] != ["csd@15", "xpd@15"]
                or recipe["value_indices"] != [8, 9] or recipe["presence_indices"] != [20, 21]
                or recipe["rng_namespace"] != 20261001
                or protocol["variants"]["supervised_18"]["first_phase_epochs"] != 6
                or protocol["variants"]["reconstruction"]["first_phase_epoch_equivalents"] != 6):
            raise ValueError("unsupported frozen reconstruction recipe")
        if any(config["population"] != "all" or config["token_dropout"] != 0.
               for config in protocol["variants"].values()):
            raise ValueError("reconstruction requires the full uniform TRAIN population")
    else:
        raise ValueError("unknown fixed experiment schema")
    if set(protocol["variants"]) != set(expected):
        raise ValueError("unexpected fixed experiment variants")
    return expected


def _relative_input_hashes(hashes, root):
    return {Path(path).relative_to(root).as_posix(): value for path, value in hashes.items()}


def _reuse_reference(protocol, source_hashes, data, split, research_root):
    """Reuse the pinned C0 and shared experts, never silently retrain a control."""
    run_id = protocol["reference"]["run_id"]
    report = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / run_id / "completed.json"
    if _sha256(report) != protocol["reference"]["completed_sha256"]:
        raise ValueError("reference completion checksum differs")
    receipt = json.loads(report.read_text())
    contract = receipt["contract"]
    if receipt["status"] != "TRAINING_COMPLETE_NOT_SCORED":
        raise ValueError("reference training did not complete")
    if (_relative_input_hashes(contract["input_sha256"], research_root)
            != _relative_input_hashes(data["input_hashes"], research_root)):
        raise ValueError("reference training inputs differ")
    expected_ids = {key: _contract_digest(data["frame"].iloc[ix].golgg_match_id.astype(str).tolist())
                    for key, ix in split.items()}
    if expected_ids != contract["split_ids_sha256"]:
        raise ValueError("reference split membership differs")
    # These two files orchestrate this new experiment; archived inference
    # implementations, native losses, experts and calibration must be unchanged.
    for path, expected_hash in contract["source_sha256"].items():
        if path not in ("src/models/a0_data_focus.py", "scripts/train_a0_data_focus.py"):
            if source_hashes.get(path) != expected_hash:
                raise ValueError(f"reference implementation changed: {path}")
    model_root = PROJECT_ROOT / "data/06_models/a0_data_focus" / run_id
    for relative, expected_hash in receipt["artifacts"].items():
        if ("/control/" in relative or "/shared/" in relative or relative.endswith("/execution.json")
                or relative.endswith("/control.parquet") or relative.endswith("/control_logits.npz")):
            if _sha256(PROJECT_ROOT / relative) != expected_hash:
                raise ValueError(f"reference artifact changed: {relative}")
    shared = _load_checkpoint(model_root / "shared/model.joblib", {**contract, "component": "shared"})
    models, details = [], []
    for seed in SEEDS:
        path = model_root / "control" / f"seed_{seed}/model.joblib"
        models.append(_load_checkpoint(path, {**contract, "component": "neural", "variant": "control", "seed": seed}))
        details.append(json.loads(path.with_suffix(".joblib.receipt.json").read_text())["details"])
    calibration = _load_checkpoint(model_root / "control/calibration/model.joblib",
                                   {**contract, "component": "calibration", "variant": "control"})
    return {"shared": shared, "models": models, "details": details, "calibration": calibration}


def _history_coverage(data, indices):
    coverage, selected_coverage = [], []
    for begin in range(0, len(indices), 256):
        ix = indices[begin:begin + 256]
        valid = np.asarray(data["history"]["mask"][ix], dtype=bool)
        present = (np.asarray(data["history"]["history"][ix])[..., 12:24] > 0) & valid[..., None]
        counts = valid.sum((1, 2, 3))
        for destination, numerator, fields in (
                (coverage, present.sum((1, 2, 3, 4)), 12),
                (selected_coverage, present[..., 8:10].sum((1, 2, 3, 4)), 2)):
            destination.extend(np.divide(numerator, counts * fields,
                                         out=np.full(len(ix), np.nan), where=counts > 0))
    return np.asarray(coverage), np.asarray(selected_coverage)


def _neural_scoreline_nll(backend, logits, rho, calibration, gates, best_of, wins_a, wins_b):
    columns = np.where(wins_a > wins_b, wins_b, 3 + wins_a).astype(int)
    rows = np.arange(len(logits))
    result = {}
    for mode_index, mode in enumerate(MODES):
        members = backend["beta_terminal"].terminal_logs(
            logits[:, mode_index].ravel(), np.repeat(best_of, 3), np.tile(rho, len(logits)),
        ).reshape(len(logits), 3, 6)
        raw_mass = logsumexp(members, axis=1) - np.log(3)
        calibrated_mass = backend["beta_calibration"].calibrated_distribution(
            logits, calibration["beta_parameters"], gates, best_of, mode_index,
        )
        for kind, mass in (("raw_neural", raw_mass), ("neural", calibrated_mass)):
            values = -mass[rows, columns]
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("invalid observed scoreline probability")
            result[f"nll_{kind}_{mode}"] = values
    return result


def partition_indices(frame: pd.DataFrame, *, history=None, raw=None) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(frame.date)
    releases = pd.to_datetime(frame.result_day)
    feature_days = pd.to_datetime(frame.feature_source_max_day)
    if dates.isna().any() or releases.isna().any() or (releases < dates).any():
        raise ValueError("invalid event or release dates")
    if (feature_days >= dates).any():
        raise ValueError("feature cutoff must precede the prediction day")
    missing = feature_days.isna().to_numpy()
    if missing.any():
        # The corrected bank uses null only before any result has been consumed.
        # Cold-start absence is not an unknown cutoff; require the matching state.
        if history is None or raw is None:
            raise ValueError("missing feature cutoff requires verified empty history")
        cold = frame.loc[missing]
        source_free = (
            np.all(history["mask"][missing] == 0)
            and np.all(raw["gates"][missing] == 0)
            and cold.unknown_players.eq(10).all()
            and cold.min_player_series.eq(0).all()
            and cold.new_organization.eq(True).all()
        )
        if not source_free:
            raise ValueError("missing feature cutoff despite historical evidence")
    masks = {
        "train": (dates < "2025-07-01") & (releases < "2025-07-01"),
        "cal": (dates >= "2025-07-01") & (dates < "2026-01-01") & (releases < "2026-01-01"),
        "validation": (dates >= "2026-01-01") & (dates < "2027-01-01"),
    }
    return {name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()}


def market_tier_weights(frame: pd.DataFrame, train: np.ndarray, market_ids: set[str]):
    selected = frame.iloc[train]
    counts = selected.tier.value_counts()
    covered = selected.loc[selected.golgg_match_id.astype(str).isin(market_ids), "tier"].value_counts()
    if not len(covered):
        raise ValueError("no market coverage inside TRAIN")
    covered = covered.reindex(counts.index, fill_value=0)
    ratios = (covered / covered.sum()) / (counts / counts.sum())
    capped = ratios.clip(.5, 3.)
    by_tier = capped / ((capped * counts).sum() / counts.sum())
    weights = selected.tier.map(by_tier).to_numpy(dtype=np.float64)
    return weights, {
        "covered_rows": int(covered.sum()), "train_counts": counts.to_dict(),
        "covered_counts": covered.to_dict(), "raw_ratios": ratios.to_dict(),
        "tier_weights": by_tier.to_dict(),
        "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def drop_history_tokens(inputs: dict[str, torch.Tensor], keep: torch.Tensor):
    """Hide complete normalized tokens; padding cannot become evidence."""
    original = inputs["mask"]
    history = inputs["history"]
    if original.dtype != torch.bool or keep.dtype != torch.bool:
        raise ValueError("history masks must be boolean")
    if keep.shape != original.shape or history.shape[:-1] != original.shape:
        raise ValueError("history mask shape mismatch")
    mask = original & keep
    return {
        **inputs, "mask": mask,
        "history": torch.where(mask[..., None], history, history.new_zeros(())),
    }


@contextmanager
def training_token_dropout(model: Any, probability: float, seed: int):
    """Temporarily wrap the native batch hook; never pickle a closure."""
    if not 0 <= probability <= 1:
        raise ValueError("token dropout probability must be in [0,1]")
    receipt = {"probability": probability, "observed_tokens": 0, "dropped_tokens": 0}
    if probability == 0:
        yield receipt
        return
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260929]))
    original = model.batch
    sentinel = object()
    previous_override = model.__dict__.get("batch", sentinel)

    def batch(raw, data, ids, ix, bo, w20=True, org=True, rng=None):
        inputs, offset = original(raw, data, ids, ix, bo, w20, org, rng)
        if rng is not None:
            keep = torch.as_tensor(history_rng.random(tuple(inputs["mask"].shape)) >= probability,
                                   device=inputs["mask"].device)
            receipt["observed_tokens"] += int(inputs["mask"].sum())
            receipt["dropped_tokens"] += int((inputs["mask"] & ~keep).sum())
            inputs = drop_history_tokens(inputs, keep)
        return inputs, offset

    history_rng = rng
    model.batch = batch
    try:
        yield receipt
    finally:
        if previous_override is sentinel:
            del model.__dict__["batch"]
        else:
            model.batch = previous_override
        receipt["rng_final_state"] = history_rng.bit_generator.state


def _source_hashes(backend_hashes):
    result = dict(backend_hashes)
    for relative in (
        "src/models/a0_data_focus.py", "scripts/train_a0_data_focus.py",
        "src/models/a0_native_refit.py", "src/models/a0_target_experiment.py",
        "src/models/a0_target_objectives.py", "src/models/competition_tiers.py",
        "src/models/a0_history_reconstruction.py",
    ):
        result[relative] = _sha256(PROJECT_ROOT / relative)
    return result


def _subset_inputs(data, indices):
    return (
        uncertainty._raw_subset(data["raw"], indices),
        {key: uncertainty._IndexedRows(value, indices) for key, value in data["history"].items()},
    )


def _other_logits(shared, raw, history, best_of):
    return np.stack([
        uncertainty._unified_logits(model, raw, history, (best_of - 1) // 2)
        for model in [shared["rating"], *shared["mlps"]]
    ], axis=2)


def _calibrate(backend, neural_z, other_z, gates, frame, wins_a, wins_b, slopes=None):
    reliability = backend["tail_reliability"]
    z = np.concatenate((neural_z, other_z), axis=2)
    labels = frame.y.to_numpy(dtype=float)
    best_of = frame.best_of.to_numpy(dtype=int)
    if slopes is None:
        slopes = reliability.fit_slopes(z, labels)
    parameters, records = [], []
    for mode in range(3):
        record = backend["beta_terminal"].fit_joint(
            neural_z[:, mode], wins_a, wins_b, "beta",
        )
        parameters.append((record["scale"], record["rho"]))
        records.append(record)
    parameters = np.asarray(parameters)
    mixtures = {}
    for mode in range(3):
        expert_p = reliability.calibrated_experts(z, slopes, gates, mode)
        mass = backend["beta_calibration"].calibrated_distribution(
            neural_z, parameters, gates, best_of, mode,
        )
        expert_p[:, 0] = np.exp(logsumexp(mass[:, :3], axis=1))
        mixtures[MODES[mode]] = reliability.ReliabilityLayer("constant_mixture").fit(
            expert_p, np.empty((len(frame), 0)), labels,
        )
    return {"beta_parameters": parameters, "beta_records": records, "slopes": slopes, "mixtures": mixtures}


def _load_checkpoint(path: Path, contract: dict):
    receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text())
    if (receipt["contract"] != _json_safe(contract)
            or receipt["contract_sha256"] != _contract_digest(contract)
            or receipt["checkpoint_sha256"] != _sha256(path)):
        raise ValueError(f"checkpoint contract or checksum mismatch: {path}")
    return joblib.load(path)


def load_artifacts(model_root: Path, variant: str, research_root: Path | None = None):
    """Load only this run's hash-verified shared/variant/calibration artifacts."""
    root = Path(model_root)
    execution = json.loads((root / "execution.json").read_text())
    if variant not in experiment_variants(execution["protocol"]):
        raise ValueError("unknown data-focus variant")
    contract = execution["contract"]
    backend, hashes = uncertainty._activate_archive_backend(resolve_research_root(research_root))
    if _source_hashes(hashes) != contract["source_sha256"]:
        raise ValueError("data-focus source hashes no longer match this run")
    shared = _load_checkpoint(root / "shared/model.joblib", {**contract, "component": "shared"})
    models = [_load_checkpoint(root / variant / f"seed_{seed}/model.joblib",
                              {**contract, "component": "neural", "variant": variant, "seed": seed})
              for seed in SEEDS]
    calibration = _load_checkpoint(root / variant / "calibration/model.joblib",
                                   {**contract, "component": "calibration", "variant": variant})
    return backend, shared, models, calibration


def predict_models(backend, shared, models, calibration, raw, history, best_of):
    """Three context modes; native series probabilities, never double BO projection."""
    best_of = np.asarray(best_of, dtype=np.int64)
    data = {"frame": pd.DataFrame({"best_of": best_of}), "raw": raw, "history": history}
    neural_z, _, raw_probability = _calculate_forecast(models, data, np.arange(len(best_of)), backend)
    other_z = _other_logits(shared, raw, history, best_of)
    full, neural, _ = _predict_mixture(
        backend, neural_z, other_z, calibration["slopes"], raw["gates"], best_of, calibration,
    )
    return {"full": full, "neural": neural, "raw_neural": raw_probability.mean(2)}


def verify_saved(model_root, variant, research_root, data, indices, expected):
    backend, shared, models, calibration = load_artifacts(model_root, variant, research_root)
    raw, indexed = _subset_inputs(data, indices)
    history = {key: value[:] for key, value in indexed.items()}
    best_of = data["frame"].best_of.to_numpy(dtype=int)[indices]
    reference = predict_models(backend, shared, models, calibration, raw, history, best_of)
    for key, value in reference.items():
        np.testing.assert_allclose(value, expected[key], atol=2e-6, rtol=0)
    repeated = predict_models(backend, shared, models, calibration, raw, history, best_of)
    for key in reference:
        np.testing.assert_array_equal(reference[key], repeated[key])
    swapped = predict_models(backend, shared, models, calibration,
                             {key: value[:, ::-1].copy() for key, value in raw.items()},
                             {key: value[:, ::-1].copy() for key, value in history.items()}, best_of)
    symmetry_error = max(float(np.max(np.abs(reference[key] + swapped[key] - 1))) for key in reference)
    if symmetry_error > 2e-6:
        raise ValueError("saved A0 side symmetry failed")
    cold = predict_models(backend, shared, models, calibration,
                          {key: np.zeros_like(value) for key, value in raw.items()},
                          {key: np.zeros_like(value) for key, value in history.items()}, best_of)
    for value in cold.values():
        np.testing.assert_allclose(value, .5, atol=2e-6, rtol=0)
    empty = predict_models(backend, shared, models, calibration, raw,
                           {key: np.zeros_like(value) for key, value in history.items()}, best_of)
    if not all(np.isfinite(value).all() for value in empty.values()):
        raise ValueError("empty-history inference is nonfinite")
    after = predict_models(backend, shared, models, calibration, raw, history, best_of)
    for key in reference:
        np.testing.assert_array_equal(reference[key], after[key])
    return {"rows": len(indices), "best_of": best_of.tolist(), "symmetry_max_error": symmetry_error,
            "saved_forecast_max_error": max(float(np.max(np.abs(reference[key] - expected[key]))) for key in reference),
            "repeatable": True, "cold_start": True, "empty_histories": True, "query_nonmutation": True}


def run_experiment(protocol_path: Path, *, research_root: Path | None = None, smoke: bool = False):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    variants = experiment_variants(protocol)
    reconstruction = protocol["schema"] == "a0_history_reconstruction_v1"
    if protocol["training"]["seeds"] != list(SEEDS) or protocol["training"]["epochs"] != 12:
        raise ValueError("unexpected fixed experiment protocol")
    expected_split = {
        "train_event_and_release_before": "2025-07-01", "cal_event_start": "2025-07-01",
        "cal_event_and_release_before": "2026-01-01", "validation_event_start": "2026-01-01",
        "validation_event_before": "2027-01-01",
    }
    if any(protocol["split"][key] != value for key, value in expected_split.items()):
        raise ValueError("unsupported split; the declared dates must match the implementation")
    fixed_training = {"batch_size": 128, "shuffle": False, "device": "cpu", "threads": 2}
    if any(protocol["training"][key] != value for key, value in fixed_training.items()):
        raise ValueError("unsupported native training configuration")
    if _sha256(protocol_path.parent / "data_quality.json") != protocol["inputs"]["data_quality_sha256"]:
        raise ValueError("data-quality evidence differs from the reviewed protocol")
    research_root = resolve_research_root(research_root)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    backend, source_hashes = uncertainty._activate_archive_backend(research_root)
    source_hashes = _source_hashes(source_hashes)
    data = uncertainty._load_training_data(research_root)
    frame = data["frame"]
    identities = [classify_competition(name, pd.Timestamp(day).date())
                  for name, day in zip(frame.tournament_name, frame.date)]
    frame["tier"] = [identity.tier.value for identity in identities]
    frame["family"] = [identity.family for identity in identities]
    split = partition_indices(frame, history=data["history"], raw=data["raw"])
    for key, value in split.items():
        if len(value) != protocol["split"][key + "_rows"]:
            raise ValueError(f"frozen {key} population differs")
    input_hashes = (_relative_input_hashes(data["input_hashes"], research_root)
                    if reconstruction else data["input_hashes"])
    if (input_hashes != protocol["inputs"]["data_sha256"]
            or data["bank_receipt_sha256"] != protocol["inputs"]["bank_receipt_sha256"]):
        raise ValueError("feature/target inputs differ from frozen protocol")
    canonical_path = research_root / protocol["inputs"]["canonical_relative_to_research_root"]
    if _sha256(canonical_path) != protocol["inputs"]["canonical_sha256"]:
        raise ValueError("canonical input checksum differs")
    canonical = pd.read_parquet(canonical_path)
    if reconstruction:
        reference = _reuse_reference(protocol, source_hashes, data, split, research_root)
        weights, weight_receipt = None, {"population": "all", "row_weights": "uniform",
                                        "reference": protocol["reference"]}
    else:
        reference = None
        market_ids = set(canonical.loc[canonical.market_open.notna(), "golgg_match_id"].astype(str))
        weights, weight_receipt = market_tier_weights(frame, split["train"], market_ids)
        for tier, weight in weight_receipt["tier_weights"].items():
            if not np.isclose(weight, protocol["market_weights"]["tier_weights"][tier], atol=1e-12, rtol=0):
                raise ValueError("market weighting differs from frozen protocol")
    run_id = protocol["run_id"] + (
        "_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") if smoke else ""
    )
    model_root = PROJECT_ROOT / "data/06_models/a0_data_focus" / run_id
    output_root = PROJECT_ROOT / "data/07_model_output/a0_data_focus" / run_id
    report_root = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / run_id
    for path in (model_root, output_root, report_root):
        path.mkdir(parents=True, exist_ok=False)
    if smoke:
        rng = np.random.default_rng(20260929)
        positions = np.sort(rng.choice(len(split["train"]), 1024, replace=False))
        split["train"] = split["train"][positions]
        if weights is not None:
            weights = weights[positions]
            weights /= weights.mean()
        split["cal"] = np.sort(rng.choice(split["cal"], 192, replace=False))
        split["validation"] = np.array([next(i for i in split["validation"] if frame.iloc[i].best_of == bo)
                                         for bo in (1, 3, 5)])
    contract = {"schema": "a0_data_focus_execution_v1", "protocol_sha256": _sha256(protocol_path),
                "source_sha256": source_hashes, "input_sha256": data["input_hashes"], "versions": _versions(),
                "scope": "SMOKE_UNSCORED" if smoke else "DEVELOPMENT_ONLY", "run_id": run_id,
                "split_ids_sha256": {key: _contract_digest(frame.iloc[ix].golgg_match_id.astype(str).tolist())
                                     for key, ix in split.items()}}
    _write_json_exclusive(model_root / "execution.json", {"contract": contract, "protocol": protocol})
    _write_json_exclusive(report_root / "weighting.json", weight_receipt)
    membership = []
    for name, indices in split.items():
        part = frame.iloc[indices][["golgg_match_id", "date", "result_day", "tier"]].copy()
        part["partition"] = name
        membership.append(part)
    pd.concat(membership, ignore_index=True).to_parquet(report_root / "partitions.parquet", index=False)
    train, cal, validation = split["train"], split["cal"], split["validation"]
    labels = frame.y.to_numpy(dtype=float)
    best_of = frame.best_of.to_numpy(dtype=int)
    bo_index = (best_of - 1) // 2
    print(f"{run_id}: TRAIN={len(train)} CAL={len(cal)} VALIDATION={len(validation)}", flush=True)
    if reference is not None:
        shared = reference["shared"]
        baseline, normalizer = shared["baseline"], shared["normalizer"]
        shared_details = {"reused_reference": protocol["reference"], "new_optimizer_updates": 0}
    else:
        baseline, normalizer = _fit_baseline_normalizer(backend, data, train)
        rating, rating_details = uncertainty._fit_rating_model(backend, data["raw"], train, labels, bo_index, SEEDS[0])
        shared = {"baseline": baseline, "normalizer": normalizer, "rating": rating, "mlps": []}
        shared_details = {"rating": rating_details, "mlp": []}
        for seed in SEEDS:
            warm, warm_details = uncertainty._fit_training_component(
                backend, data["raw"], data["history"], train, labels, bo_index, seed,
                kind="mlp", full=False, epochs=1 if smoke else 24,
            )
            full, full_details = uncertainty._fit_training_component(
                backend, data["raw"], data["history"], train, labels, bo_index, seed,
                kind="mlp", full=True, initial=warm, epochs=1 if smoke else 12,
            )
            shared["mlps"].append(full)
            shared_details["mlp"].append({"seed": seed, "warm": warm_details, "full": full_details})
            del warm
            print(f"shared MLP seed={seed} fitted", flush=True)
    _atomic_checkpoint(model_root / "shared/model.joblib", shared, {**contract, "component": "shared"}, shared_details)
    score_indices = np.concatenate((cal, validation))
    score_raw, score_history = _subset_inputs(data, score_indices)
    other_z = _other_logits(shared, score_raw, score_history, best_of[score_indices])
    np.savez_compressed(output_root / "shared_logits.npz", indices=score_indices, other_logits=other_z)
    slopes = reference["calibration"]["slopes"] if reference is not None else None
    verifications = {}
    initial_hashes = {}
    gate_rng_states = {}
    for variant in variants:
        config = protocol["variants"][variant]
        selected = train
        row_weights = None
        if config["population"] == "major_and_international":
            selected = train[frame.iloc[train].tier.isin(["major", "international"]).to_numpy()]
        elif config["population"] == "all_market_tier_weighted":
            row_weights = weights
        elif config["population"] != "all":
            raise ValueError("unknown frozen training population")
        train_rows = frame.iloc[selected][["golgg_match_id", "date", "result_day", "tier"]].copy()
        train_rows["weight"] = np.ones(len(selected)) if row_weights is None else row_weights
        train_rows.to_parquet(report_root / f"{variant}_train.parquet", index=False)
        models = []
        for seed_index, seed in enumerate(SEEDS):
            started = time.monotonic()
            phase = None
            reused = reference is not None and variant == "control"
            if reused:
                model = reference["models"][seed_index]
                details = {**reference["details"][seed_index], "reused_reference": protocol["reference"],
                           "new_optimizer_updates": 0, "runtime_seconds": 0.}
                initial_hash = details["initial_state_sha256"]
            else:
                model = backend["nonlinear_history"].NonlinearHistoryModel(
                    baseline, normalizer, seed=seed, epochs=1 if smoke else 12,
                )
                model.main_objective = "terminal"
                initial_hash = _torch_state_hash(model)
                if variant == "supervised_18":
                    model.epochs = 1 if smoke else 6
                    fit_beta_target_on_device(
                        model, data["raw"], data["history"], None, selected, bo_index,
                        data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
                        device="cpu", objective="terminal",
                        progress=lambda epoch, main, aux, diagnostics: print(
                            f"{variant} seed={seed} first_phase={epoch} main={main:.6f}", flush=True),
                    )
                    phase = {"kind": "native_supervision", "optimizer_updates": model.epochs * math.ceil(len(selected) / 128),
                             "main_losses": model.train_losses, "auxiliary_losses": model.auxiliary_losses,
                             "epoch_diagnostics": model.epoch_diagnostics}
                elif variant == "reconstruction":
                    phase = pretrain_history(
                        model, data["raw"], data["history"], selected, bo_index,
                        epoch_equivalents=1 if smoke else 6, probability=.25, device="cpu",
                        progress=lambda record: print(
                            f"{variant} seed={seed} pretext={record['epoch_equivalent']} loss={record['loss']:.6f}", flush=True),
                    )
                model.epochs = 1 if smoke else 12
                with training_token_dropout(model, float(config["token_dropout"]), seed) as mask_receipt:
                    fit_beta_target_on_device(
                        model, data["raw"], data["history"], None, selected, bo_index,
                        data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
                        device="cpu", objective="terminal", row_weights=row_weights,
                        progress=lambda epoch, main, aux, diagnostics: print(
                            f"{variant} seed={seed} epoch={epoch} main={main:.6f} aux={aux:.6f}", flush=True),
                    )
                updates = model.epochs * math.ceil(len(selected) / 128) + (phase["optimizer_updates"] if phase else 0)
                details = {"train_rows": len(selected), "initial_state_sha256": initial_hash,
                           "final_state_sha256": _torch_state_hash(model), "main_losses": model.train_losses,
                           "auxiliary_losses": model.auxiliary_losses, "epoch_diagnostics": model.epoch_diagnostics,
                           "history_mask": mask_receipt, "gate_rng_final_state": model.dropout_rng_final_state,
                           "first_phase": phase, "new_optimizer_updates": updates,
                           "runtime_seconds": time.monotonic() - started}
            if seed in initial_hashes and initial_hashes[seed] != initial_hash:
                raise ValueError("paired neural initialization differs")
            initial_hashes[seed] = initial_hash
            if config["population"] != "major_and_international" and not (reused and smoke):
                state = model.dropout_rng_final_state
                if seed in gate_rng_states and state != gate_rng_states[seed]:
                    raise ValueError("intervention changed the final native gate RNG stream")
                gate_rng_states[seed] = state
            _atomic_checkpoint(model_root / variant / f"seed_{seed}/model.joblib", model,
                               {**contract, "component": "neural", "variant": variant, "seed": seed}, details)
            models.append(model)
        neural_z, rho, raw_probability = _calculate_forecast(models, data, score_indices, backend)
        calibration = (reference["calibration"] if reference is not None and variant == "control" else
                       _calibrate(backend, neural_z[:len(cal)], other_z[:len(cal)], score_raw["gates"][:len(cal)],
                                  frame.iloc[cal], data["wins_a"][cal], data["wins_b"][cal], slopes))
        slopes = calibration["slopes"]
        _atomic_checkpoint(model_root / variant / "calibration/model.joblib", calibration,
                           {**contract, "component": "calibration", "variant": variant},
                           {"beta": calibration["beta_records"], "mixtures": {
                               mode: {"theta": item.theta, "fit": item.fit_info}
                               for mode, item in calibration["mixtures"].items()}})
        full_p, neural_p, _ = _predict_mixture(backend, neural_z, other_z, slopes, score_raw["gates"],
                                              best_of[score_indices], calibration)
        np.savez_compressed(output_root / f"{variant}_logits.npz", indices=score_indices, logits=neural_z,
                            rho=rho, raw_seed_probability=raw_probability)
        predictions = frame.iloc[validation].copy()
        if reconstruction:
            predictions["history_stat_coverage"], predictions["csd_xpd_coverage"] = _history_coverage(data, validation)
            predictions["wins_a"], predictions["wins_b"] = data["wins_a"][validation], data["wins_b"][validation]
            scoreline = _neural_scoreline_nll(
                backend, neural_z[len(cal):], rho, calibration, score_raw["gates"][len(cal):],
                best_of[validation], data["wins_a"][validation], data["wins_b"][validation],
            )
            for column, values in scoreline.items():
                predictions[column] = values
        predictions["train_end"] = str(frame.iloc[train].result_day.max())
        predictions["calibration_end"] = str(frame.iloc[cal].result_day.max())
        predictions["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        predictions["scenario"] = "historical_actual_roster_daily_release_not_issued_forecast"
        expected = {"full": full_p[len(cal):], "neural": neural_p[len(cal):],
                    "raw_neural": raw_probability[len(cal):].mean(2)}
        for kind, matrix in expected.items():
            if not np.isfinite(matrix).all() or np.any((matrix <= 0) | (matrix >= 1)):
                raise ValueError("nonfinite or noninterior validation probability")
            for mode_index, mode in enumerate(MODES):
                predictions[f"p_{kind}_{mode}"] = matrix[:, mode_index]
        predictions.to_parquet(output_root / f"{variant}.parquet", index=False)
        probe_positions = np.array([next(i for i, row in enumerate(validation) if best_of[row] == bo) for bo in (1, 3, 5)])
        verifications[variant] = verify_saved(model_root, variant, research_root, data, validation[probe_positions],
                                              {key: value[probe_positions] for key, value in expected.items()})
        print(f"{variant}: saved and restored all-mode BO1/3/5 predictions", flush=True)
    _, final_hashes = uncertainty._activate_archive_backend(research_root)
    if _source_hashes(final_hashes) != source_hashes or _sha256(protocol_path) != contract["protocol_sha256"]:
        raise ValueError("source or protocol changed during execution")
    completion = {"status": "SMOKE_PASSED" if smoke else "TRAINING_COMPLETE_NOT_SCORED", "contract": contract,
                  "verification": verifications, "source_unchanged": True,
                  "artifacts": {str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                                for root in (model_root, output_root, report_root)
                                for path in sorted(root.rglob("*")) if path.is_file()}}
    _write_json_exclusive(report_root / "completed.json", completion)
    print(json.dumps({"status": completion["status"], "run_id": run_id, "report_root": str(report_root)}), flush=True)
    return completion
