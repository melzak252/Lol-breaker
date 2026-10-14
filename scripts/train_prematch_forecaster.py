#!/usr/bin/env python3
"""Train and evaluate fresh pre-draft models under a frozen temporal protocol.

Stages are deliberately separate: prepare -> develop/select -> final -> predict.
The final stage cannot choose a different model using its scored outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

from betting_app.ml.calibration.candidate_calibration import TemperatureScalingCalibrator
from src.analysis.bootstrap import block_ratio_interval
from src.analysis.model_benchmark import compute_benchmark_summary
from src.analysis.probability_metrics import binary_log_loss_vector, clip_probabilities
from src.models.prematch_features import PLAYER_FEATURE_NAMES, build_features
from src.models.prematch_neural import NeuralModel
from src.models.prematch_predictor import ArchitectureForecastArtifact, ForecastArtifact, FittedMember, partition_indices, take_batch
from src.models.prematch_tabular import TabularModel

CODE_FILES = [
    "scripts/train_prematch_forecaster.py",
    "scripts/export_prospective_features.py",
    "src/utils/golgg_schema.py",
    "betting_app/ml/calibration/candidate_calibration.py",
    "src/analysis/model_benchmark.py",
    "src/analysis/research_benchmark.py",
    "src/analysis/probability_metrics.py",
    "src/analysis/metrics.py",
    "src/analysis/bootstrap.py",
    "src/models/prematch_features.py",
    "src/models/prematch_tabular.py",
    "src/models/prematch_neural.py",
    "src/models/prematch_predictor.py",
    "src/models/prematch_history_bank.py",
    "src/models/prematch_history_fusion.py",
    "src/models/prematch_stack.py",
]


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n")


def code_hashes() -> dict[str, str]:
    return {path: sha256(path) for path in CODE_FILES}


def prepare(protocol: dict, protocol_path: Path) -> None:
    for field in ("source", "release_days"):
        if sha256(protocol[field]) != protocol[field + "_sha256"]:
            raise ValueError(f"Pinned {field} hash does not match")
    destination = Path(protocol["feature_dir"])
    report_dir = Path(protocol["report_dir"])
    if (report_dir / "prepared.json").exists():
        raise FileExistsError(f"Preparation receipt already exists: {report_dir}")
    source_protocol_path = protocol.get("feature_source_protocol")
    if source_protocol_path:
        source_protocol_path = Path(source_protocol_path)
        source_protocol = json.loads(source_protocol_path.read_text())
        for key in ("source_sha256", "release_days_sha256"):
            if source_protocol[key] != protocol[key]:
                raise ValueError(f"Feature reuse changes pinned {key}")
        if Path(source_protocol["feature_dir"]).resolve() != destination.resolve():
            raise ValueError("Feature reuse must reference the original feature directory")
        _, _, source_receipt = load_features(source_protocol, source_protocol_path)
        result = source_receipt["artifacts"]
    else:
        if destination.exists():
            raise FileExistsError(f"Feature run already exists: {destination}")
        releases = json.loads(Path(protocol["release_days"]).read_text())
        result = build_features(Path(protocol["source"]), releases, destination)
    receipt = {
        "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "artifacts": result, "feature_artifact_hashes": {key: sha256(path) for key, path in result.items() if key.endswith("_path")},
        "code_sha256": code_hashes(), "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "joblib", "pyarrow")}},
    }
    if source_protocol_path:
        receipt["feature_source"] = {
            "protocol": str(source_protocol_path),
            "protocol_sha256": sha256(source_protocol_path),
            "receipt_sha256": sha256(Path(source_protocol["report_dir"]) / "prepared.json"),
        }
    write_json(report_dir / "prepared.json", receipt)
    print(json.dumps({"stage": "prepare", "artifacts": result}), flush=True)


def load_features(protocol: dict, protocol_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    receipt = json.loads((Path(protocol["report_dir"]) / "prepared.json").read_text())
    if receipt["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Protocol changed after feature preparation")
    if receipt["code_sha256"]["src/models/prematch_features.py"] != sha256("src/models/prematch_features.py"):
        raise ValueError("Feature implementation changed after feature preparation")
    for key, expected in receipt["feature_artifact_hashes"].items():
        if sha256(receipt["artifacts"][key]) != expected:
            raise ValueError(f"Feature artifact changed: {key}")
    metadata = pd.read_parquet(receipt["artifacts"]["metadata_path"])
    with np.load(receipt["artifacts"]["arrays_path"]) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if metadata.match_id.duplicated().any() or not metadata.best_of.isin([1, 3, 5]).all():
        raise ValueError("Invalid target identities or binary series formats")
    history = metadata.max_history_release_day.notna()
    if not metadata.loc[history, "max_history_release_day"].lt(metadata.loc[history, "date"]).all():
        raise ValueError("Feature history violates strict release chronology")
    if any(len(value) != len(metadata) for value in arrays.values()):
        raise ValueError("Feature arrays do not align with target metadata")
    experience_column = PLAYER_FEATURE_NAMES.index("log_appearance_count")
    history_counts = np.rint(np.expm1(arrays["players"][..., experience_column])).astype(np.int64)
    metadata["minimum_player_history_maps"] = history_counts.min(axis=(1, 2))
    metadata["mean_player_history_maps"] = history_counts.mean(axis=(1, 2))
    return metadata, arrays, receipt


def metrics(y: np.ndarray, probability: np.ndarray, epsilon: float) -> dict:
    summary = asdict(compute_benchmark_summary(y, probability, probability_epsilon=epsilon))
    losses = binary_log_loss_vector(y, probability, epsilon=epsilon)
    summary.update(deep_losses=int((losses >= 2.5).sum()), mean_probability=float(np.mean(probability)), observed_frequency=float(np.mean(y)))
    return summary


def fit_member(configuration: dict, batch: dict, y: np.ndarray, weights: np.ndarray, calibration: dict, calibration_y: np.ndarray, protocol: dict, origin: str, cutoff: str) -> FittedMember:
    raw_only = protocol["calibration_months"] == 0
    if raw_only:
        if protocol["candidate_variants"] != ["raw"] or len(calibration_y):
            raise ValueError("Disabled calibration requires raw-only forecasts and an empty CAL partition")
    elif len(np.unique(calibration_y)) != 2:
        raise ValueError("CAL must contain both outcomes; refusing a silent identity calibrator")
    kind = configuration["kind"]
    estimator = None
    if kind != "rating":
        if kind == "neural":
            estimator = NeuralModel(configuration["params"], seed=protocol["seed"])
        elif kind == "history_fusion":
            from src.models.prematch_history_fusion import HistoryFusionModel

            estimator = HistoryFusionModel(configuration["params"], seed=protocol["seed"])
        else:
            estimator = TabularModel(kind, configuration["params"], seed=protocol["seed"])
        estimator.fit(batch, y, weights)
    if raw_only:
        return FittedMember(configuration, estimator, None, protocol["probability_epsilon"], origin, cutoff, 0)
    raw = calibration["baseline_series"][:, configuration["params"]["index"]] if kind == "rating" else estimator.predict_proba(calibration)
    raw = clip_probabilities(raw, epsilon=protocol["probability_epsilon"])
    lower, upper = protocol["temperature_bounds"]
    calibrator = TemperatureScalingCalibrator(min_temperature=lower, max_temperature=upper).fit(logit(raw), calibration_y)
    if not np.isfinite(calibrator.temperature_) or not lower <= calibrator.temperature_ <= upper:
        raise ValueError("Invalid fitted calibration temperature")
    return FittedMember(configuration, estimator, calibrator, protocol["probability_epsilon"], origin, cutoff, len(calibration_y))


def run_origins(protocol: dict, metadata: pd.DataFrame, arrays: dict, origins: list[str], configurations: list[dict], phase: str) -> tuple[pd.DataFrame, dict[str, FittedMember], list[dict]]:
    calibration_enabled = protocol["calibration_months"] != 0
    if not calibration_enabled and protocol["candidate_variants"] != ["raw"]:
        raise ValueError("Disabled calibration requires raw-only candidate variants")
    parts, receipts = [], []
    last_members: dict[str, FittedMember] = {}
    prediction_dir = Path(protocol["prediction_dir"])
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for origin in origins:
        train, cal, test = partition_indices(metadata, origin, protocol["calibration_months"])
        if train.sum() < 1000 or (calibration_enabled and cal.sum() < 100) or not test.any():
            raise ValueError(f"Insufficient chronological partition at {origin}: {train.sum()}, {cal.sum()}, {test.sum()}")
        cutoff = (pd.Timestamp(origin) - pd.DateOffset(months=protocol["calibration_months"])).strftime("%Y-%m-%d")
        train_batch, cal_batch, test_batch = (take_batch(arrays, mask) for mask in (train, cal, test))
        prior_batch = take_batch(arrays, test, prior_roster=True) if phase == "final" else None
        train_y, cal_y = metadata.loc[train, "y"].to_numpy(dtype=float), metadata.loc[cal, "y"].to_numpy(dtype=float)
        age = (pd.Timestamp(cutoff) - pd.to_datetime(metadata.loc[train, "date"])).dt.days.to_numpy()
        weights = np.exp2(-age / protocol["training_half_life_days"])
        part = metadata.loc[test].copy()
        part["model_origin"] = origin
        part["training_cutoff"] = cutoff
        part["generated_at"] = datetime.now(timezone.utc).isoformat()
        cal_predictions = metadata.loc[cal, ["match_id", "date", "release_day", "y"]].copy()
        directory = Path(protocol["model_dir"]) / phase / origin
        directory.mkdir(parents=True, exist_ok=False)
        members = {}
        fit_receipt = {"origin": origin, "training_cutoff": cutoff, "n_train": int(train.sum()), "n_calibration": int(cal.sum()), "n_forecasts": int(test.sum()), "train_last_label_release": metadata.loc[train, "release_day"].max(), "cal_last_label_release": metadata.loc[cal, "release_day"].max(), "members": {}}
        for configuration in configurations:
            began = time.monotonic()
            identifier = configuration["id"]
            member = fit_member(configuration, train_batch, train_y, weights, cal_batch, cal_y, protocol, origin, cutoff)
            members[identifier] = member
            for variant in protocol["candidate_variants"]:
                column = f"{identifier}__{variant}"
                part[column] = member.predict_proba(test_batch, variant)
                if calibration_enabled:
                    cal_predictions[column] = member.predict_proba(cal_batch, variant)
                if prior_batch is not None:
                    part["prior__" + column] = member.predict_proba(prior_batch, variant)
            artifact = directory / f"{identifier}.joblib"
            joblib.dump(member, artifact, compress=3)
            temperature = member.calibrator.temperature_ if member.calibrator is not None else None
            fit_receipt["members"][identifier] = {"configuration": configuration, "temperature": temperature, "seconds": time.monotonic() - began, "artifact": str(artifact), "sha256": sha256(artifact)}
            print(json.dumps({"phase": phase, "origin": origin, "candidate": identifier, "n_train": int(train.sum()), "n_cal": int(cal.sum()), "n_forecasts": int(test.sum()), "temperature": temperature, "seconds": round(time.monotonic() - began, 2)}), flush=True)
        if calibration_enabled:
            cal_predictions.to_parquet(prediction_dir / f"{phase}_calibration_{origin}.parquet", index=False)
        part.to_parquet(prediction_dir / f"{phase}_forecasts_{origin}.parquet", index=False)
        write_json(directory / "receipt.json", fit_receipt)
        receipts.append(fit_receipt)
        parts.append(part)
        last_members = members
    forecasts = pd.concat(parts, ignore_index=True)
    if forecasts.match_id.duplicated().any():
        raise ValueError("Quarterly forecast blocks overlap")
    return forecasts, last_members, receipts


def member_recipe(label: str) -> list[dict]:
    identifier, variant = label.split("__")
    return [{"candidate": identifier, "variant": variant, "weight": 1.0}]


def average_recipes(labels: list[str]) -> list[dict]:
    return [{**member_recipe(label)[0], "weight": 1.0 / len(labels)} for label in labels]


def recipe_probability(frame: pd.DataFrame, recipe: list[dict], *, prior: bool = False) -> np.ndarray:
    prefix = "prior__" if prior else ""
    result = np.zeros(len(frame), dtype=float)
    for part in recipe:
        result += part["weight"] * frame[prefix + part["candidate"] + "__" + part["variant"]].to_numpy()
    return result


def develop(protocol: dict, protocol_path: Path) -> None:
    initial_calibration_start = (pd.Timestamp(protocol["final_origins"][0]) - pd.DateOffset(months=protocol["calibration_months"])).strftime("%Y-%m-%d")
    if protocol["selection_end_exclusive"] > initial_calibration_start:
        raise ValueError("Model selection overlaps the initial final calibration window")
    metadata, arrays, prepared = load_features(protocol, protocol_path)
    frame, _, receipts = run_origins(protocol, metadata, arrays, protocol["development_origins"], protocol["candidates"], "development")
    eligible = frame.date.lt(protocol["selection_end_exclusive"]) & frame.release_day.lt(protocol["selection_end_exclusive"])
    frame["selection_eligible"] = eligible
    if not eligible.any():
        raise ValueError("No development labels were available before selection closed")
    y = frame.loc[eligible, "y"].to_numpy()
    rows, recipes = [], {}
    for config in protocol["candidates"]:
        for variant in protocol["candidate_variants"]:
            label = config["id"] + "__" + variant
            recipes[label] = member_recipe(label)
            rows.append({"model": label, "family": config["family"], **metrics(y, frame.loc[eligible, label].to_numpy(), protocol["probability_epsilon"])})
    ranked = pd.DataFrame(rows).sort_values(["log_loss", "brier_score", "model"])
    family_best = ranked.groupby("family", sort=False).first()["model"].to_dict()
    supervised_families = ["logistic", "lightgbm", "neural"]
    best_supervised = ranked.loc[ranked.family.isin(supervised_families), "model"].iloc[0]
    recipes["ensemble_families"] = average_recipes([family_best[family] for family in supervised_families])
    recipes["ensemble_player_anchor"] = average_recipes([best_supervised, family_best["player_control"]])
    for label in ("ensemble_families", "ensemble_player_anchor"):
        frame[label] = recipe_probability(frame, recipes[label])
        rows.append({"model": label, "family": "ensemble", **metrics(y, frame.loc[eligible, label].to_numpy(), protocol["probability_epsilon"])})
    ranked = pd.DataFrame(rows).sort_values(["log_loss", "brier_score", "model"])
    selected = str(ranked.iloc[0].model)
    reference = str(ranked.loc[ranked.family.isin(["team_control", "player_control"]), "model"].iloc[0])
    report_dir = Path(protocol["report_dir"])
    ranked.to_csv(report_dir / "development_metrics.csv", index=False)
    frame.to_parquet(Path(protocol["prediction_dir"]) / "development_predictions.parquet", index=False)
    selection = {
        "selected": selected, "recipe": recipes[selected], "family_best": family_best,
        "reference": reference, "reference_recipe": recipes[reference], "recipes": recipes,
        "protocol_sha256": sha256(protocol_path),
        "feature_artifact_hashes": prepared["feature_artifact_hashes"],
        "code_sha256": code_hashes(), "selected_at": datetime.now(timezone.utc).isoformat(),
        "selection_period": [str(frame.loc[eligible, "date"].min()), str(frame.loc[eligible, "date"].max())],
        "selection_n": int(eligible.sum()), "selection_excluded_unreleased": int((~eligible).sum()),
        "selection_label_deadline": protocol["selection_end_exclusive"],
        "final_outcomes_used_for_selection": False, "fit_receipts": receipts,
    }
    write_json(report_dir / "selection.json", selection)
    print(ranked[["model", "log_loss", "brier_score", "deep_losses"]].to_string(index=False), flush=True)
    print(json.dumps({"selected": selected, "recipe": recipes[selected], "reference": reference}), flush=True)


def final_report(protocol: dict, frame: pd.DataFrame, selection: dict) -> dict:
    epsilon = protocol["probability_epsilon"]
    y = frame.y.to_numpy()
    scored = {column: metrics(y, frame[column].to_numpy(), epsilon) for column in frame if column.startswith("p_")}
    reference_loss = binary_log_loss_vector(y, frame.p_reference.to_numpy(), epsilon=epsilon)
    selected_loss = binary_log_loss_vector(y, frame.p_selected.to_numpy(), epsilon=epsilon)
    delta = selected_loss - reference_loss
    monthly_blocks = frame.date.astype(str).str[:7].to_numpy()
    tournament_blocks = (frame.date.astype(str).str[:4] + "::" + frame.tournament.astype(str)).to_numpy()
    intervals = {
        unit: block_ratio_interval(delta, np.ones(len(delta)), blocks, draws=protocol["uncertainty"]["draws"], seed=protocol["seed"])
        for unit, blocks in (("calendar_month", monthly_blocks), ("source_tournament_label_within_year", tournament_blocks))
    }
    slices = []
    profile = frame.copy()
    profile["quarter"] = pd.to_datetime(profile.date).dt.to_period("Q").astype(str)
    profile["confidence"] = pd.cut(np.maximum(profile.p_selected, 1 - profile.p_selected), [0.499999, .6, .7, .8, .9, 1.0], right=False).astype(str)
    profile["history_support"] = pd.cut(profile.minimum_player_history_maps, [-.5, .5, 10.5, 50.5, np.inf], labels=["includes debutant", "minimum 1-10 maps", "minimum 11-50 maps", "all players >50 maps"]).astype(str)
    for dimension in ("quarter", "best_of", "source_provenance", "confidence", "history_support"):
        for name, group in profile.groupby(dimension, dropna=False, observed=True):
            slices.append({"dimension": dimension, "slice": str(name), **metrics(group.y.to_numpy(), group.p_selected.to_numpy(), epsilon)})
    bins = pd.cut(frame.p_selected, np.linspace(0, 1, 11), include_lowest=True)
    calibration = frame.groupby(bins, observed=False).agg(matches=("y", "size"), forecast=("p_selected", "mean"), observed=("y", "mean"))
    calibration.index.name = "probability_bin"
    report_dir = Path(protocol["report_dir"])
    pd.DataFrame(slices).to_csv(report_dir / "final_profiles.csv", index=False)
    calibration.to_csv(report_dir / "final_calibration.csv")
    monthly = pd.DataFrame({"month": monthly_blocks, "selected_loss": selected_loss, "reference_loss": reference_loss, "delta": delta}).groupby("month").agg(matches=("delta", "size"), selected_ll=("selected_loss", "mean"), reference_ll=("reference_loss", "mean"), delta_ll=("delta", "mean"))
    monthly.to_csv(report_dir / "final_monthly_effects.csv")
    return {"status": "HISTORICAL_LOCKED_RECIPE_ASSESSMENT", "selected": selection["selected"], "reference": selection["reference"], "rows": len(frame), "date_min": frame.date.min(), "date_max": frame.date.max(), "metrics": scored, "paired_delta_log_loss": intervals, "probability_epsilon": epsilon, "final_selection_performed": False, "limitations": ["Historical years have prior project exposure: this is not an independent prospective confirmation.", "Realized first-map lineups are a declared known-lineup scenario, not timestamp-verified announcements.", "Source remains incomplete and mixed-provenance; exact publication times are unavailable.", "Month blocks preserve local dependence, not all repeated-team dependence; tournament-label groups are not independently verified editions.", "No financial, betting-selection calibration, production or tournament-simulation certification."]}


def finalize(protocol: dict, protocol_path: Path) -> None:
    metadata, arrays, prepared = load_features(protocol, protocol_path)
    report_dir = Path(protocol["report_dir"])
    selection_path = report_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["protocol_sha256"] != sha256(protocol_path) or selection["code_sha256"] != code_hashes():
        raise ValueError("Protocol or implementation changed after development selection")
    if selection["feature_artifact_hashes"] != prepared["feature_artifact_hashes"]:
        raise ValueError("Feature inputs changed after development selection")
    labels = list(selection["family_best"].values())
    needed = {part["candidate"] for part in selection["recipe"]}
    needed.update(label.split("__")[0] for label in labels)
    configurations = [config for config in protocol["candidates"] if config["id"] in needed]
    frame, last_members, receipts = run_origins(protocol, metadata, arrays, protocol["final_origins"], configurations, "final")
    frame["p_selected"] = recipe_probability(frame, selection["recipe"])
    frame["p_reference"] = recipe_probability(frame, selection["reference_recipe"])
    frame["p_prior_roster"] = recipe_probability(frame, selection["recipe"], prior=True)
    for family, label in selection["family_best"].items():
        frame["p_" + family] = frame[label]
    frame["p_chance"] = .5
    expected = metadata.loc[metadata.date.ge(protocol["final_origins"][0]), "match_id"].astype(str)
    if set(expected) != set(frame.match_id.astype(str)):
        raise ValueError("Final forecasts do not cover every eligible final target")
    prediction_path = Path(protocol["prediction_dir"]) / "final_predictions.parquet"
    frame.to_parquet(prediction_path, index=False)
    summary = final_report(protocol, frame, selection)
    summary["fit_receipts"] = receipts
    summary["selection_sha256_before_final"] = sha256(selection_path)
    summary["final_predictions_sha256"] = sha256(prediction_path)
    state = joblib.load(prepared["artifacts"]["state_path"])
    selected_members = {part["candidate"]: last_members[part["candidate"]] for part in selection["recipe"]}
    artifact = ForecastArtifact(selection["recipe"], selected_members, state, {"run_id": protocol["run_id"], "selected_recipe": selection["selected"], "model_origin": protocol["final_origins"][-1], "source_sha256": protocol["source_sha256"], "release_days_sha256": protocol["release_days_sha256"], "protocol_sha256": sha256(protocol_path), "code_sha256": code_hashes(), "feature_artifact_hashes": prepared["feature_artifact_hashes"]})
    artifact_path = Path(protocol["model_dir"]) / "selected_forecaster.joblib"
    artifact.save(artifact_path)
    summary["artifact"] = str(artifact_path)
    summary["artifact_sha256"] = sha256(artifact_path)
    write_json(report_dir / "final_summary.json", summary)
    print(json.dumps(_jsonable({"selected": selection["selected"], "rows": len(frame), "metrics": summary["metrics"], "paired_delta_log_loss": summary["paired_delta_log_loss"], "artifact": str(artifact_path)}), indent=2, allow_nan=False), flush=True)


def architecture_prepare(protocol: dict, protocol_path: Path, research_root: Path | None = None) -> None:
    """Pin and align existing raw histories; never copy fitted A0 representations."""
    from src.models.prematch_history_bank import align_history

    root_file = PROJECT_ROOT / "data/research_root.txt"
    root_value = research_root or os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    if root_value is None and root_file.exists():
        root_value = root_file.read_text().strip()
    if not root_value:
        raise ValueError("Specify --research-root, ENSEMBLE_RESEARCH_ROOT or data/research_root.txt")
    root = Path(root_value).resolve()
    source_path = Path(protocol["source_protocol"])
    if sha256(source_path) != protocol["source_protocol_sha256"]:
        raise ValueError("Fresh feature protocol changed")
    source = json.loads(source_path.read_text())
    metadata, arrays, source_receipt = load_features(source, source_path)
    del arrays
    bank = root / protocol["history_bank_relative"]
    input_hashes = {str(source_path): sha256(source_path)}
    for field in ("source", "release_days"):
        if sha256(source[field]) != source[field + "_sha256"]:
            raise ValueError(f"Original {field} changed")
        input_hashes[source[field]] = source[field + "_sha256"]
    for name, expected in protocol["history_input_sha256"].items():
        path = bank / name
        if sha256(path) != expected:
            raise ValueError(f"Historical feature input changed: {name}")
        input_hashes[str(path)] = expected
    manifest_path = Path(protocol["benchmark_manifest"])
    if sha256(manifest_path) != protocol["benchmark_manifest_sha256"]:
        raise ValueError("Canonical benchmark manifest changed")
    manifest = json.loads(manifest_path.read_text())
    canonical_path = root / manifest["input"]["path"]
    if sha256(canonical_path) != manifest["input"]["sha256"]:
        raise ValueError("Canonical reference predictions changed")
    input_hashes[str(manifest_path)] = sha256(manifest_path)
    input_hashes[str(canonical_path)] = manifest["input"]["sha256"]
    report_dir = Path(protocol["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=False)
    old_metadata = pd.read_parquet(bank / "metadata.parquet")
    _, _, alignment = align_history(
        metadata, old_metadata, np.load(bank / "history.npy", mmap_mode="r"),
        np.load(bank / "mask.npy", mmap_mode="r"), output_dir=Path(protocol["feature_dir"]),
    )
    ordered = old_metadata.set_index(old_metadata.golgg_match_id.astype(str)).loc[metadata.match_id.astype(str)]
    history_days = pd.DataFrame({
        "fresh": metadata.max_history_release_day.fillna("").to_numpy(),
        "history": ordered.feature_source_max_day.fillna("").to_numpy(),
    })
    metadata["feature_source_max_day"] = history_days.max(axis=1).replace("", None)
    metadata_path = Path(protocol["feature_dir"]) / "metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    feature_paths = [metadata_path, Path(protocol["feature_dir"]) / "history.npy",
                     Path(protocol["feature_dir"]) / "history_mask.npy"]
    receipt = {
        "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "research_root": str(root), "source_protocol": str(source_path),
        "input_hashes": input_hashes, "source_feature_hashes": source_receipt["feature_artifact_hashes"],
        "feature_hashes": {str(path): sha256(path) for path in feature_paths},
        "code_sha256": code_hashes(), "alignment": alignment,
        "canonical_path": str(canonical_path), "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": source_receipt["environment"],
    }
    receipt["environment"] = {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "joblib", "pyarrow")},
    }
    write_json(report_dir / "architecture_prepared.json", receipt)
    with zipfile.ZipFile(report_dir / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as snapshot:
        for path in CODE_FILES:
            snapshot.write(path, arcname=path)
    print(json.dumps({"stage": "architecture-prepare", "rows": len(metadata), "alignment": alignment}), flush=True)


def load_architecture_features(protocol: dict, protocol_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    receipt = json.loads((Path(protocol["report_dir"]) / "architecture_prepared.json").read_text())
    if receipt["protocol_sha256"] != sha256(protocol_path) or receipt["code_sha256"] != code_hashes():
        raise ValueError("Architecture protocol or implementation changed after preparation")
    for path, expected in {**receipt["input_hashes"], **receipt["feature_hashes"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Architecture input changed: {path}")
    source_path = Path(receipt["source_protocol"])
    source = json.loads(source_path.read_text())
    original, arrays, source_receipt = load_features(source, source_path)
    if source_receipt["feature_artifact_hashes"] != receipt["source_feature_hashes"]:
        raise ValueError("Fresh feature inputs changed")
    directory = Path(protocol["feature_dir"])
    metadata = pd.read_parquet(directory / "metadata.parquet")
    if not np.array_equal(metadata.match_id.astype(str), original.match_id.astype(str)):
        raise ValueError("Aligned metadata no longer matches fresh feature rows")
    arrays["history"] = np.load(directory / "history.npy", mmap_mode="r")
    arrays["history_mask"] = np.load(directory / "history_mask.npy", mmap_mode="r")
    return metadata, arrays, receipt


def architecture_train(protocol: dict, protocol_path: Path) -> None:
    metadata, arrays, prepared = load_architecture_features(protocol, protocol_path)
    warmup = [config for config in protocol["candidates"] if config["id"] in protocol["warmup_candidates"]]
    paths, all_receipts = {}, []
    for phase, origins, configs in (
        ("warmup", protocol["warmup_origins"], warmup),
        ("architecture", protocol["evaluation_origins"], protocol["candidates"]),
    ):
        frame, _, receipts = run_origins(protocol, metadata, arrays, origins, configs, phase)
        if phase == "architecture" and len(frame) != protocol["expected_evaluation_rows"]:
            raise ValueError("Architecture forecasts do not cover the declared full cohort")
        path = Path(protocol["prediction_dir"]) / f"{phase}_predictions.parquet"
        frame.to_parquet(path, index=False)
        paths[str(path)] = sha256(path)
        all_receipts.extend(receipts)
    if prepared["code_sha256"] != code_hashes() or prepared["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Implementation or protocol changed during training")
    write_json(Path(protocol["report_dir"]) / "architecture_training.json", {
        "status": "COMPLETE_BASE_MODELS", "prediction_hashes": paths,
        "fit_receipts": all_receipts, "protocol_sha256": sha256(protocol_path),
    })


def architecture_stack(protocol: dict, protocol_path: Path) -> None:
    from src.models.prematch_stack import SpecialistStack, stack_context, stack_training_indices

    metadata, arrays, prepared = load_architecture_features(protocol, protocol_path)
    report_dir = Path(protocol["report_dir"])
    if (report_dir / "architecture_stacking.json").exists():
        raise FileExistsError("Architecture stack has already been fitted")
    training = json.loads((report_dir / "architecture_training.json").read_text())
    if training["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Training protocol differs from stack protocol")
    frames = []
    for path, expected in training["prediction_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("Out-of-time expert predictions changed")
        frames.append(pd.read_parquet(path))
    forecasts = pd.concat(frames, ignore_index=True)
    if forecasts.match_id.astype(str).duplicated().any():
        raise ValueError("Warmup and evaluation populations overlap")
    fit_by_origin = {receipt["origin"]: receipt for receipt in training["fit_receipts"]}
    forecasts["calibration_last_label_release"] = forecasts.model_origin.map(
        {origin: receipt["cal_last_label_release"] for origin, receipt in fit_by_origin.items()}
    )
    spec = protocol["stack"]
    expert_columns = [name + "__" + spec["variant"] for name in spec["experts"]]
    probabilities = forecasts[expert_columns].to_numpy(float)
    row_lookup = pd.Index(metadata.match_id.astype(str)).get_indexer(forecasts.match_id.astype(str))
    if (row_lookup < 0).any():
        raise ValueError("Unknown expert forecast identity")
    # The connector needs summary features, not another large history copy.
    contexts = stack_context({"players": arrays["players"][row_lookup]}, probabilities)
    forecasts["specialist_stack"] = np.nan
    stack_receipts = []
    for origin in protocol["evaluation_origins"]:
        eligible = stack_training_indices(forecasts, origin)
        target = forecasts.model_origin.eq(origin).to_numpy()
        if eligible.sum() < 100 or not target.any():
            raise ValueError("Insufficient genuine out-of-time connector history")
        model = SpecialistStack(C=spec["C"]).fit(
            probabilities[eligible], contexts[eligible], forecasts.loc[eligible, "y"].to_numpy(float)
        )
        forecasts.loc[target, "specialist_stack"] = model.predict_proba(probabilities[target], contexts[target])
        fit = fit_by_origin[origin]
        members = {}
        for name, member_receipt in fit["members"].items():
            path = member_receipt["artifact"]
            if sha256(path) != member_receipt["sha256"]:
                raise ValueError("Base checkpoint changed before stacking")
            members[name] = joblib.load(path)
        bundle = ArchitectureForecastArtifact(
            members, model, tuple(spec["experts"]), spec["variant"],
            {"run_id": protocol["run_id"], "origin": origin, "protocol_sha256": sha256(protocol_path),
             "feature_hashes": prepared["feature_hashes"], "code_sha256": prepared["code_sha256"],
             "scope": "aligned feature-batch research inference; no live history adapter"},
        )
        artifact_path = Path(protocol["model_dir"]) / "architecture" / origin / "architecture_bundle.joblib"
        if artifact_path.exists():
            raise FileExistsError(artifact_path)
        bundle.save(artifact_path)
        stack_receipts.append({
            "origin": origin, "n_train": int(eligible.sum()), "n_forecasts": int(target.sum()),
            "training_last_date": forecasts.loc[eligible, "date"].max(),
            "training_last_label_release": forecasts.loc[eligible, "release_day"].max(),
            "training_ids": forecasts.loc[eligible, "match_id"].astype(str).tolist(),
            "artifact": str(artifact_path), "sha256": sha256(artifact_path),
        })
        print(json.dumps({"stage": "architecture-stack", "origin": origin,
                          "training_rows": int(eligible.sum()), "forecast_rows": int(target.sum())}), flush=True)
    evaluated = forecasts.loc[forecasts.model_origin.isin(protocol["evaluation_origins"])].copy()
    if len(evaluated) != protocol["expected_evaluation_rows"] or not np.isfinite(evaluated.specialist_stack).all():
        raise ValueError("Incomplete learned-stack coverage")
    output = Path(protocol["prediction_dir"]) / "connected_predictions.parquet"
    evaluated.to_parquet(output, index=False)
    write_json(report_dir / "architecture_stacking.json", {
        "status": "COMPLETE_LEARNED_CONNECTOR", "protocol_sha256": sha256(protocol_path),
        "prediction_path": str(output), "prediction_sha256": sha256(output),
        "fit_receipts": stack_receipts, "expert_input_hashes": training["prediction_hashes"],
    })


def architecture_report(protocol: dict, protocol_path: Path) -> None:
    from src.analysis.research_benchmark import attach_candidate, validate_frame

    _, _, prepared = load_architecture_features(protocol, protocol_path)
    report_dir = Path(protocol["report_dir"])
    stacked = json.loads((report_dir / "architecture_stacking.json").read_text())
    if sha256(stacked["prediction_path"]) != stacked["prediction_sha256"]:
        raise ValueError("Connected forecasts changed")
    frame = pd.read_parquet(stacked["prediction_path"])
    base = pd.read_parquet(prepared["canonical_path"])
    if set(frame.match_id.astype(str)) != set(base.golgg_match_id.astype(str)):
        raise ValueError("Candidate must cover the exact canonical cohort")
    base = base.set_index(base.golgg_match_id.astype(str)).loc[frame.match_id.astype(str)].reset_index(drop=True)
    reverse = base.team1_id.astype(str).to_numpy() != frame.team_a.astype(str).to_numpy()
    for fresh_key, canonical_key in (("date", "date"), ("release_day", "result_day"), ("best_of", "best_of")):
        if not np.array_equal(frame[fresh_key].astype(str), base[canonical_key].astype(str)):
            raise ValueError(f"Canonical identity mismatch: {fresh_key}")
    if not np.array_equal(frame.team_a.astype(str), np.where(reverse, base.team2_id.astype(str), base.team1_id.astype(str))):
        raise ValueError("Unaligned canonical team A")
    if not np.array_equal(frame.team_b.astype(str), np.where(reverse, base.team1_id.astype(str), base.team2_id.astype(str))):
        raise ValueError("Unaligned canonical team B")
    if not np.array_equal(frame.y, np.where(reverse, 1 - base.y, base.y)):
        raise ValueError("Unaligned canonical outcome")
    frame["a0"] = np.where(reverse, 1 - base.p_full, base.p_full)
    epsilon = protocol["probability_epsilon"]
    columns = [config["id"] + "__" + variant for config in protocol["candidates"]
               for variant in protocol["candidate_variants"]] + ["specialist_stack", "a0"]
    scores = {column: metrics(frame.y.to_numpy(), frame[column].to_numpy(), epsilon) for column in columns}
    pd.DataFrame([{"model": name, **score} for name, score in scores.items()]).to_csv(report_dir / "architecture_metrics.csv", index=False)
    intervals = []
    comparisons = [(name, "a0") for name in columns if name != "a0"]
    comparisons.append((protocol["primary_candidate"], protocol["primary_comparison"]))
    for candidate, reference in comparisons:
        for metric_name in ("log_loss", "brier"):
            if metric_name == "log_loss":
                delta = binary_log_loss_vector(frame.y, frame[candidate], epsilon=epsilon) - binary_log_loss_vector(frame.y, frame[reference], epsilon=epsilon)
            else:
                delta = (clip_probabilities(frame[candidate], epsilon=epsilon) - frame.y.to_numpy()) ** 2 - (clip_probabilities(frame[reference], epsilon=epsilon) - frame.y.to_numpy()) ** 2
            for unit, blocks in (
                ("calendar_month", frame.date.astype(str).str[:7]),
                ("source_tournament_label_within_year", frame.date.astype(str).str[:4] + "::" + frame.tournament.astype(str)),
            ):
                interval = block_ratio_interval(delta, np.ones(len(frame)), blocks, draws=protocol["uncertainty"]["draws"], seed=protocol["seed"])
                intervals.append({"candidate": candidate, "reference": reference, "metric": metric_name, "unit": unit, **interval})
    pd.DataFrame(intervals).to_csv(report_dir / "architecture_intervals.csv", index=False)
    profiles = []
    frame["year"] = frame.date.str[:4]
    frame["history_support"] = pd.cut(frame.minimum_player_history_maps, [-.5, .5, 10.5, 50.5, np.inf],
                                      labels=["debutant", "1-10", "11-50", ">50"]).astype(str)
    for dimension in ("year", "best_of", "history_support"):
        for name, group in frame.groupby(dimension, observed=True, dropna=False):
            for column in columns:
                profiles.append({"dimension": dimension, "slice": str(name), "model": column,
                                 **metrics(group.y.to_numpy(), group[column].to_numpy(), epsilon)})
    pd.DataFrame(profiles).to_csv(report_dir / "architecture_profiles.csv", index=False)
    frame.to_parquet(report_dir / "paired_predictions.parquet", index=False)
    canonical = base[["golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y"]].copy()
    canonical["feature_source_max_day"] = frame.feature_source_max_day.to_numpy()
    # These are strict release bounds, not fictitious real-time fit timestamps.
    canonical["train_end"] = (pd.to_datetime(frame.training_cutoff) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").to_numpy()
    canonical["calibration_end"] = (pd.to_datetime(frame.model_origin) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d").to_numpy()
    for column in columns:
        if column not in ("specialist_stack", "a0"):
            canonical[column] = np.where(reverse, 1 - frame[column], frame[column])
    base_columns = [column for column in columns if column not in ("specialist_stack", "a0")]
    validate_frame(canonical, base_columns)
    stack_candidate = canonical[["golgg_match_id", "team1_id", "team2_id", "date", "result_day",
                                 "best_of", "y", "feature_source_max_day", "calibration_end"]].copy()
    stack_ends = {item["origin"]: item["training_last_label_release"] for item in stacked["fit_receipts"]}
    stack_candidate["train_end"] = frame.model_origin.map(stack_ends).to_numpy()
    stack_candidate["specialist_stack"] = np.where(reverse, 1 - frame.specialist_stack, frame.specialist_stack)
    attach_candidate(base, stack_candidate, "specialist_stack")
    stack_candidate_path = report_dir / "canonical_stack_candidate.parquet"
    stack_candidate.to_parquet(stack_candidate_path, index=False)
    for candidate in (protocol["primary_candidate"], protocol["primary_comparison"]):
        attach_candidate(base, canonical, candidate)
    candidate_path = report_dir / "canonical_candidates.parquet"
    canonical.to_parquet(candidate_path, index=False)
    write_json(report_dir / "architecture_summary.json", {
        "status": "HISTORICAL_ARCHITECTURE_COMPARISON_NOT_PROMOTED", "rows": len(frame),
        "date_min": frame.date.min(), "date_max": frame.date.max(),
        "primary_candidate": protocol["primary_candidate"], "primary_comparison": protocol["primary_comparison"],
        "metrics": scores, "paired_intervals": intervals, "side_reversals": int(reverse.sum()),
        "candidate_path": str(candidate_path), "candidate_sha256": sha256(candidate_path),
        "stack_candidate_path": str(stack_candidate_path), "stack_candidate_sha256": sha256(stack_candidate_path),
        "protocol_sha256": sha256(protocol_path), "limitations": protocol["limitations"],
        "hyperparameter_selection_performed": False, "production_changed": False,
    })
    print(json.dumps(_jsonable({"rows": len(frame), "metrics": scores, "candidate_path": str(candidate_path)}), indent=2, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "develop", "final", "predict", "architecture-prepare", "architecture-train", "architecture-stack", "architecture-report", "predict-batch"])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--match-json", type=Path, help="JSON with team_a, roster_a, team_b, roster_b, day, best_of")
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--features", type=Path, help="Aligned NPZ feature batch for architecture inference")
    args = parser.parse_args()
    if args.stage == "predict-batch":
        if args.artifact is None or args.features is None:
            parser.error("predict-batch requires --artifact and --features")
        artifact = ArchitectureForecastArtifact.load(args.artifact)
        with np.load(args.features, allow_pickle=False) as archive:
            predictions = artifact.predict_proba({key: archive[key] for key in archive.files})
        print(json.dumps(_jsonable(predictions), indent=2, allow_nan=False))
        return
    if args.stage == "predict":
        if args.artifact is None or args.match_json is None:
            parser.error("predict requires --artifact and --match-json")
        artifact = ForecastArtifact.load(args.artifact)
        print(json.dumps(artifact.predict_match(**json.loads(args.match_json.read_text())), indent=2))
        return
    if args.protocol is None:
        parser.error("training stages require --protocol")
    protocol = json.loads(args.protocol.read_text())
    if args.stage == "architecture-prepare":
        architecture_prepare(protocol, args.protocol, args.research_root)
        return
    {"prepare": prepare, "develop": develop, "final": finalize,
     "architecture-train": architecture_train, "architecture-stack": architecture_stack,
     "architecture-report": architecture_report}[args.stage](protocol, args.protocol)


if __name__ == "__main__":
    main()
