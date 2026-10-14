"""Matched annual replication of native12 versus native6+12.

The hash-pinned single-origin implementations are reused unchanged. Previously
trained 2026 artifacts are references, never forecasts for earlier years.
"""
from __future__ import annotations

import json
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import a0_data_focus as native
from src.models.a0_native_refit import (
    MODES, SEEDS, _atomic_checkpoint, _calculate_forecast, _contract_digest,
    _fit_baseline_normalizer, _sha256, _torch_state_hash, _versions,
    _write_json_exclusive,
)
from src.models.a0_target_objectives import fit_beta_target_on_device
from src.models.competition_tiers import classify_competition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YEARS = (2024, 2025, 2026)
VARIANTS = ("control", "supervised_18")
REFERENCE_RUN = "a0_history_reconstruction_20261001_010503"
CANONICAL = "rating-foundation-20260915/data/08_reporting/paired.parquet"


def annual_partitions(frame: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    """Closed TRAIN/CAL partitions; delayed labels never cross a fit origin."""
    if isinstance(year, bool) or year not in YEARS:
        raise ValueError("unsupported annual origin")
    dates = pd.to_datetime(frame.date, errors="raise")
    released = pd.to_datetime(frame.result_day, errors="raise")
    if dates.isna().any() or released.isna().any() or (released < dates).any():
        raise ValueError("invalid event/release chronology")
    train_end = pd.Timestamp(year=year - 1, month=7, day=1)
    origin = pd.Timestamp(year=year, month=1, day=1)
    stop = pd.Timestamp(year=year + 1, month=1, day=1)
    masks = {
        "train": (dates < train_end) & (released < train_end),
        "cal": (dates >= train_end) & (dates < origin) & (released < origin),
        "validation": (dates >= origin) & (dates < stop),
    }
    return {key: np.flatnonzero(value.to_numpy()) for key, value in masks.items()}


def _sources(backend_hashes):
    result = native._source_hashes(backend_hashes)
    for relative in ("src/models/a0_reliability.py", "scripts/train_a0_reliability.py"):
        result[relative] = _sha256(PROJECT_ROOT / relative)
    return result


def _inputs(data, research_root):
    return native._relative_input_hashes(data["input_hashes"], research_root)


def _source_path(key, research_root):
    if key.startswith("research:"):
        return research_root / key.removeprefix("research:")
    return PROJECT_ROOT / key.removeprefix("project:")


def freeze_protocol(run_id: str, research_root: Path | None = None) -> Path:
    """Pin inputs, code, closed memberships and fixed hypotheses before fitting."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be one path component")
    root = native.resolve_research_root(research_root)
    destination = PROJECT_ROOT / "data/05_model_input/model_reliability" / run_id
    path = destination / "neural_protocol.json"
    if path.exists():
        raise FileExistsError(path)
    goal = json.loads((destination / "saved_goal.json").read_text())
    if goal["run_id"] != run_id:
        raise ValueError("goal belongs to a different run")
    _, backend_hashes = native.uncertainty._activate_archive_backend(root)
    data = native.uncertainty._load_training_data(root)
    native.partition_indices(data["frame"], history=data["history"], raw=data["raw"])
    sources = _sources(backend_hashes)
    canonical = pd.read_parquet(root / CANONICAL)
    canonical_ids = set(canonical.golgg_match_id.astype(str))
    if len(canonical) != 11550 or len(canonical_ids) != len(canonical):
        raise ValueError("canonical population differs")
    partitions = {}
    for year in YEARS:
        split = annual_partitions(data["frame"], year)
        assessed = data["frame"].iloc[split["validation"]]
        if not set(assessed.golgg_match_id).issubset(canonical_ids):
            raise ValueError("annual assessment has noncanonical targets")
        partitions[str(year)] = {
            "train_end_exclusive": f"{year - 1}-07-01",
            "cal_end_exclusive": f"{year}-01-01",
            "assessment_end_exclusive": f"{year + 1}-01-01",
            "counts": {key: len(ix) for key, ix in split.items()},
            "membership_sha256": {key: _contract_digest(data["frame"].iloc[ix].golgg_match_id.tolist())
                                   for key, ix in split.items()},
        }
    previous = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / REFERENCE_RUN / "completed.json"
    source_zip = destination / "neural_sources.zip"
    with zipfile.ZipFile(source_zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, digest in sources.items():
            source = _source_path(key, root)
            if _sha256(source) != digest:
                raise ValueError("source changed during freeze")
            archive.write(source, key.replace(":", "/", 1))
    protocol = {
        "schema": "a0_reliability_annual_v1", "run_id": run_id,
        "status": "FROZEN_BEFORE_FITTING", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_sha256": _sha256(destination / "saved_goal.json"),
        "source_sha256": sources, "source_snapshot_sha256": _sha256(source_zip),
        "inputs": _inputs(data, root), "bank_receipt_sha256": data["bank_receipt_sha256"],
        "canonical_path": CANONICAL, "canonical_sha256": _sha256(root / CANONICAL),
        "partitions": partitions, "years": list(YEARS), "variants": list(VARIANTS),
        "training": {"seeds": list(SEEDS), "control_epochs": 12, "candidate_epochs": [6, 12],
                     "batch_size": 128, "shuffle": False, "threads": 2,
                     "phase_boundary": "reset AdamW and native gate RNG, retain learned parameters",
                     "shared_experts": "same full-TRAIN rating and three native warm24/full12 MLPs",
                     "objective": "native terminal beta NLL plus0.1 masked auxiliary SmoothL1",
                     "calibration": "unchanged native CAL beta, expert slopes and regularized constant mixture"},
        "reference_2026": {"run_id": REFERENCE_RUN, "completed_sha256": _sha256(previous)},
        "primary": {"years": [2024, 2025], "candidate": "supervised_18", "reference": "control",
                    "rows": 8447, "metric": "full assembled winner LL"},
        "evaluation": {"resamples": 5000, "seed": 20261001, "probability_epsilon": None,
                       "ll_interval_upper_below": 0., "brier_delta_at_most": 0.,
                       "full_population_ll_delta_at_most": 0., "guard_ll_margin": .005,
                       "guard_min_months": 6, "calibration_gap_tolerance": .05,
                       "calibration_min_rows": 200, "calibration_min_months": 6},
        "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_NOT_PROSPECTIVE",
    }
    _write_json_exclusive(path, protocol)
    print(json.dumps({"protocol": str(path), "sha256": _sha256(path), "partitions": partitions}), flush=True)
    return path


def _read_protocol(path: Path, root: Path):
    protocol = json.loads(Path(path).read_text())
    expected_training = {"seeds": list(SEEDS), "control_epochs": 12, "candidate_epochs": [6, 12],
                         "batch_size": 128, "shuffle": False, "threads": 2}
    if (protocol["schema"] != "a0_reliability_annual_v1" or protocol["years"] != list(YEARS)
            or protocol["variants"] != list(VARIANTS)
            or any(protocol["training"][key] != value for key, value in expected_training.items())):
        raise ValueError("unsupported frozen annual recipe")
    if _sha256(root / protocol["canonical_path"]) != protocol["canonical_sha256"]:
        raise ValueError("canonical evidence changed")
    return protocol


def _origin_paths(run_id, year):
    return {name: PROJECT_ROOT / f"data/{stage}/model_reliability" / run_id / "neural" / str(year)
            for name, stage in (("models", "06_models"), ("output", "07_model_output"), ("report", "08_reporting"))}


def load_origin(model_root: Path, variant: str, research_root: Path | None = None):
    root = native.resolve_research_root(research_root)
    execution = json.loads((Path(model_root) / "execution.json").read_text())
    contract = execution["contract"]
    if variant not in VARIANTS:
        raise ValueError("unknown annual variant")
    backend, hashes = native.uncertainty._activate_archive_backend(root)
    if _sources(hashes) != contract["source_sha256"]:
        raise ValueError("annual implementation changed since fitting")
    reference = execution.get("reference")
    if reference:
        completion_path = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / reference["run_id"] / "completed.json"
        if _sha256(completion_path) != reference["completed_sha256"]:
            raise ValueError("reused completion changed")
        previous_root = PROJECT_ROOT / "data/06_models/a0_data_focus" / reference["run_id"]
        return native.load_artifacts(previous_root, variant, root)
    shared = native._load_checkpoint(Path(model_root) / "shared/model.joblib", {**contract, "component": "shared"})
    models = [native._load_checkpoint(Path(model_root) / variant / f"seed_{seed}/model.joblib",
                                     {**contract, "component": "neural", "variant": variant, "seed": seed})
              for seed in SEEDS]
    calibration = native._load_checkpoint(Path(model_root) / variant / "calibration/model.joblib",
                                         {**contract, "component": "calibration", "variant": variant})
    return backend, shared, models, calibration


def _verify_origin(paths, variant, root, data, indices, predictions):
    backend, shared, models, calibration = load_origin(paths["models"], variant, root)
    raw, indexed = native._subset_inputs(data, indices)
    history = {key: value[:] for key, value in indexed.items()}
    best_of = data["frame"].iloc[indices].best_of.to_numpy(dtype=int)
    replay = native.predict_models(backend, shared, models, calibration, raw, history, best_of)
    errors = []
    for kind, values in replay.items():
        expected = predictions[[f"p_{kind}_{mode}" for mode in MODES]].to_numpy(dtype=float)
        np.testing.assert_allclose(values, expected, atol=2e-6, rtol=0)
        errors.append(float(np.max(np.abs(values - expected))))
    probes = np.array([np.flatnonzero(best_of == bo)[0] for bo in (1, 3, 5)])
    probe_raw = {key: value[probes] for key, value in raw.items()}
    probe_history = {key: value[probes] for key, value in history.items()}
    probe_bo = best_of[probes]
    original = native.predict_models(backend, shared, models, calibration, probe_raw, probe_history, probe_bo)
    reverse = native.predict_models(backend, shared, models, calibration,
                                    {key: value[:, ::-1].copy() for key, value in probe_raw.items()},
                                    {key: value[:, ::-1].copy() for key, value in probe_history.items()}, probe_bo)
    symmetry = max(float(np.max(np.abs(original[key] + reverse[key] - 1))) for key in original)
    if symmetry > 2e-6:
        raise ValueError("annual side symmetry failed")
    cold = native.predict_models(backend, shared, models, calibration,
                                 {key: np.zeros_like(value) for key, value in probe_raw.items()},
                                 {key: np.zeros_like(value) for key, value in probe_history.items()}, probe_bo)
    for values in cold.values():
        np.testing.assert_allclose(values, .5, atol=2e-6, rtol=0)
    empty = native.predict_models(backend, shared, models, calibration, probe_raw,
                                  {key: np.zeros_like(value) for key, value in probe_history.items()}, probe_bo)
    if not all(np.isfinite(value).all() for value in empty.values()):
        raise ValueError("empty-history inference failed")
    repeated = native.predict_models(backend, shared, models, calibration, probe_raw, probe_history, probe_bo)
    for key in original:
        np.testing.assert_array_equal(original[key], repeated[key])
    return {"rows": len(indices), "probabilities_replayed": len(indices) * 9,
            "max_replay_error": max(errors), "max_symmetry_error": symmetry,
            "cold_start": True, "empty_histories": True, "repeatable_nonmutating_queries": True}


def _reuse_2026(paths, protocol, contract, root, data, split):
    reference = protocol["reference_2026"]
    previous_report = PROJECT_ROOT / "data/08_reporting/a0_data_focus" / reference["run_id"] / "completed.json"
    if _sha256(previous_report) != reference["completed_sha256"]:
        raise ValueError("2026 reference checksum differs")
    receipt = json.loads(previous_report.read_text())
    if (native._relative_input_hashes(receipt["contract"]["input_sha256"], root) != _inputs(data, root)
            or receipt["contract"]["split_ids_sha256"] != contract["split_ids_sha256"]):
        raise ValueError("2026 reference population or inputs differ")
    _write_json_exclusive(paths["models"] / "execution.json", {"contract": contract, "reference": reference})
    result = {}
    previous_output = PROJECT_ROOT / "data/07_model_output/a0_data_focus" / reference["run_id"]
    for variant in VARIANTS:
        path = previous_output / f"{variant}.parquet"
        relative = str(path.relative_to(PROJECT_ROOT))
        if _sha256(path) != receipt["artifacts"][relative]:
            raise ValueError("reference predictions changed")
        predictions = pd.read_parquet(path)
        expected = data["frame"].iloc[split["validation"]]
        for key in ("golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y"):
            if predictions[key].astype(str).tolist() != expected[key].astype(str).tolist():
                raise ValueError(f"reference identity mismatch: {key}")
        predictions.to_parquet(paths["output"] / f"{variant}.parquet", index=False)
        result[variant] = {**_verify_origin(paths, variant, root, data, split["validation"], predictions),
                           "new_optimizer_updates": [0, 0, 0], "reference": reference}
    return result


def _train_origin(paths, protocol, contract, backend, root, data, split, smoke):
    _write_json_exclusive(paths["models"] / "execution.json", {"contract": contract})
    frame = data["frame"]
    train, cal, validation = (split[name] for name in ("train", "cal", "validation"))
    labels = frame.y.to_numpy(dtype=float)
    best_of = frame.best_of.to_numpy(dtype=int)
    bo_index = (best_of - 1) // 2
    baseline, normalizer = _fit_baseline_normalizer(backend, data, train)
    rating, rating_details = native.uncertainty._fit_rating_model(backend, data["raw"], train, labels, bo_index, SEEDS[0])
    shared = {"baseline": baseline, "normalizer": normalizer, "rating": rating, "mlps": []}
    shared_details = {"rating": rating_details, "mlps": []}
    for seed in SEEDS:
        warm, warm_details = native.uncertainty._fit_training_component(
            backend, data["raw"], data["history"], train, labels, bo_index, seed,
            kind="mlp", full=False, epochs=1 if smoke else 24)
        full, full_details = native.uncertainty._fit_training_component(
            backend, data["raw"], data["history"], train, labels, bo_index, seed,
            kind="mlp", full=True, initial=warm, epochs=1 if smoke else 12)
        shared["mlps"].append(full)
        shared_details["mlps"].append({"seed": seed, "warm": warm_details, "full": full_details})
    _atomic_checkpoint(paths["models"] / "shared/model.joblib", shared, {**contract, "component": "shared"}, shared_details)
    score_indices = np.concatenate((cal, validation))
    score_raw, score_history = native._subset_inputs(data, score_indices)
    other_z = native._other_logits(shared, score_raw, score_history, best_of[score_indices])
    np.savez_compressed(paths["output"] / "shared_logits.npz", indices=score_indices, other_logits=other_z)
    slopes = None
    initial_hashes, gate_states, verification = {}, {}, {}
    for variant in VARIANTS:
        models, updates = [], []
        for seed in SEEDS:
            started = time.monotonic()
            model = backend["nonlinear_history"].NonlinearHistoryModel(baseline, normalizer, seed=seed, epochs=1 if smoke else 12)
            model.main_objective = "terminal"
            initial = _torch_state_hash(model)
            if seed in initial_hashes and initial != initial_hashes[seed]:
                raise ValueError("paired initialization differs")
            initial_hashes[seed] = initial
            phases = []
            schedule = [1] if smoke else [12]
            if variant == "supervised_18":
                schedule = [1, 1] if smoke else [6, 12]
            for phase_index, epochs in enumerate(schedule):
                model.epochs = epochs
                fit_beta_target_on_device(
                    model, data["raw"], data["history"], None, train, bo_index,
                    data["wins_a"], data["wins_b"], data["targets"], data["target_mask"],
                    device="cpu", objective="terminal",
                    progress=lambda epoch, main, aux, diagnostics: print(
                        f"year={contract['year']} arm={variant} seed={seed} phase={phase_index + 1} epoch={epoch} main={main:.6f}", flush=True))
                phases.append({"epochs": epochs, "optimizer_updates": epochs * math.ceil(len(train) / 128),
                               "main_losses": model.train_losses, "auxiliary_losses": model.auxiliary_losses,
                               "diagnostics": model.epoch_diagnostics})
            if seed in gate_states and gate_states[seed] != model.dropout_rng_final_state:
                raise ValueError("paired final gate RNG streams differ")
            gate_states[seed] = model.dropout_rng_final_state
            update_count = sum(item["optimizer_updates"] for item in phases)
            _atomic_checkpoint(paths["models"] / variant / f"seed_{seed}/model.joblib", model,
                               {**contract, "component": "neural", "variant": variant, "seed": seed},
                               {"initial_state_sha256": initial, "final_state_sha256": _torch_state_hash(model),
                                "phases": phases, "optimizer_updates": update_count,
                                "gate_rng_final_state": model.dropout_rng_final_state,
                                "runtime_seconds": time.monotonic() - started})
            models.append(model)
            updates.append(update_count)
        neural_z, rho, raw_seed_probability = _calculate_forecast(models, data, score_indices, backend)
        calibration = native._calibrate(backend, neural_z[:len(cal)], other_z[:len(cal)], score_raw["gates"][:len(cal)],
                                        frame.iloc[cal], data["wins_a"][cal], data["wins_b"][cal], slopes)
        slopes = calibration["slopes"]
        _atomic_checkpoint(paths["models"] / variant / "calibration/model.joblib", calibration,
                           {**contract, "component": "calibration", "variant": variant},
                           {"beta": calibration["beta_records"], "cal_rows": len(cal)})
        full_p, neural_p, _ = native._predict_mixture(backend, neural_z, other_z, slopes, score_raw["gates"], best_of[score_indices], calibration)
        np.savez_compressed(paths["output"] / f"{variant}_logits.npz", indices=score_indices, logits=neural_z,
                            rho=rho, raw_seed_probability=raw_seed_probability)
        predictions = frame.iloc[validation].copy()
        predictions["history_stat_coverage"], predictions["csd_xpd_coverage"] = native._history_coverage(data, validation)
        predictions["wins_a"], predictions["wins_b"] = data["wins_a"][validation], data["wins_b"][validation]
        scoreline = native._neural_scoreline_nll(backend, neural_z[len(cal):], rho, calibration,
                                                score_raw["gates"][len(cal):], best_of[validation],
                                                data["wins_a"][validation], data["wins_b"][validation])
        for key, values in scoreline.items():
            predictions[key] = values
        predictions["train_end"] = str(frame.iloc[train].result_day.max())
        predictions["calibration_end"] = str(frame.iloc[cal].result_day.max())
        predictions["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        predictions["scenario"] = "historical_actual_roster_daily_release_not_issued_forecast"
        for kind, values in (("full", full_p[len(cal):]), ("neural", neural_p[len(cal):]),
                             ("raw_neural", raw_seed_probability[len(cal):].mean(2))):
            if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
                raise ValueError("noninterior assessment probability")
            for index, mode in enumerate(MODES):
                predictions[f"p_{kind}_{mode}"] = values[:, index]
        predictions.to_parquet(paths["output"] / f"{variant}.parquet", index=False)
        verification[variant] = {**_verify_origin(paths, variant, root, data, validation, predictions),
                                 "new_optimizer_updates": updates}
    return verification


def run_experiment(protocol_path: Path, *, research_root: Path | None = None, smoke: bool = False):
    root = native.resolve_research_root(research_root)
    protocol_path = Path(protocol_path).resolve()
    protocol = _read_protocol(protocol_path, root)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    backend, backend_hashes = native.uncertainty._activate_archive_backend(root)
    sources = _sources(backend_hashes)
    if sources != protocol["source_sha256"]:
        raise ValueError("frozen neural sources changed")
    data = native.uncertainty._load_training_data(root)
    native.partition_indices(data["frame"], history=data["history"], raw=data["raw"])
    if _inputs(data, root) != protocol["inputs"] or data["bank_receipt_sha256"] != protocol["bank_receipt_sha256"]:
        raise ValueError("frozen neural inputs changed")
    frame = data["frame"]
    identities = [classify_competition(name, pd.Timestamp(day).date()) for name, day in zip(frame.tournament_name, frame.date)]
    frame["tier"] = [item.tier.value for item in identities]
    frame["family"] = [item.family for item in identities]
    run_id = protocol["run_id"] + ("_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") if smoke else "")
    all_results = {}
    for year in ([2024] if smoke else YEARS):
        split = annual_partitions(frame, year)
        expected = protocol["partitions"][str(year)]
        memberships = {key: _contract_digest(frame.iloc[ix].golgg_match_id.tolist()) for key, ix in split.items()}
        if (memberships != expected["membership_sha256"] or {key: len(ix) for key, ix in split.items()} != expected["counts"]):
            raise ValueError("frozen annual membership changed")
        if smoke:
            rng = np.random.default_rng(20261001)
            split["train"] = np.sort(rng.choice(split["train"], 1024, replace=False))
            split["cal"] = np.sort(rng.choice(split["cal"], 192, replace=False))
            split["validation"] = np.array([next(i for i in split["validation"] if frame.iloc[i].best_of == bo) for bo in (1, 3, 5)])
        paths = _origin_paths(run_id, year)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=False)
        contract = {"schema": "a0_reliability_origin_v1", "run_id": run_id, "year": year,
                    "scope": "SMOKE_UNSCORED" if smoke else "EXPOSED_HISTORICAL_DEVELOPMENT",
                    "protocol_sha256": _sha256(protocol_path), "source_sha256": sources,
                    "input_sha256": protocol["inputs"], "versions": _versions(),
                    "split_ids_sha256": {key: _contract_digest(frame.iloc[ix].golgg_match_id.tolist()) for key, ix in split.items()}}
        members = []
        for key, ix in split.items():
            rows = frame.iloc[ix][["golgg_match_id", "date", "result_day", "tier"]].copy()
            rows["partition"] = key
            members.append(rows)
        pd.concat(members, ignore_index=True).to_parquet(paths["report"] / "partitions.parquet", index=False)
        print(f"{run_id} year={year}: " + ", ".join(f"{key}={len(ix)}" for key, ix in split.items()), flush=True)
        verification = (_reuse_2026(paths, protocol, contract, root, data, split) if year == 2026 and not smoke
                        else _train_origin(paths, protocol, contract, backend, root, data, split, smoke))
        completion = {"status": "SMOKE_PASSED" if smoke else "TRAINING_COMPLETE_NOT_SCORED",
                      "contract": contract, "verification": verification,
                      "artifacts": {str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                                    for directory in paths.values() for path in sorted(directory.rglob("*")) if path.is_file()}}
        _write_json_exclusive(paths["report"] / "completed.json", completion)
        all_results[str(year)] = {"completed_path": str((paths["report"] / "completed.json").relative_to(PROJECT_ROOT)),
                                  "completed_sha256": _sha256(paths["report"] / "completed.json"), "verification": verification}
    if _sources(backend_hashes) != sources or _sha256(protocol_path) != contract["protocol_sha256"]:
        raise ValueError("source/protocol changed during training")
    result = {"status": "SMOKE_PASSED" if smoke else "TRAINING_COMPLETE_NOT_SCORED", "run_id": run_id,
              "protocol_sha256": _sha256(protocol_path), "origins": all_results}
    _write_json_exclusive(PROJECT_ROOT / "data/08_reporting/model_reliability" / run_id / "neural_completed.json", result)
    print(json.dumps(result), flush=True)
    return result
