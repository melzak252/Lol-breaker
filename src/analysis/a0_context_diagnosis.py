"""Frozen roster-context repair and prediction-stage diagnosis; never fit models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import ev_asof_replay as replay
from src.analysis.a0_error_audit import ROOT, load_protocol, save_json, sha256


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


def _native_predictions(targets, raw, history, protocol):
    result = targets.copy()
    for column in ("p_golgg", "p_no_w20", "p_no_organization"):
        result[column] = np.nan
    result["status"] = np.where(result.feature_available, "available", "unavailable")
    module = replay._activate_native_predictor(protocol["resolved_research_root"])
    original_receipt = json.loads(protocol["paths"]["repair_receipt"].read_text())
    models = {}
    for year in sorted(result.loc[result.feature_available, "model_year"].astype(int).unique()):
        predictor = module.CausalA0Predictor(int(year))
        prefix = f"{replay.CAUSAL_ARCHIVE_RELATIVE}/data/06_models/causal_a0/{year}/"
        expected = {str(path).split(prefix, 1)[-1]: value for path, value in original_receipt["models"][str(year)]["artifact_hashes"].items()}
        observed = {str(path).split(prefix, 1)[-1]: value for path, value in predictor.hashes.items()}
        if expected != observed:
            raise ValueError(f"Frozen model artifacts changed for {year}")
        positions = np.flatnonzero(result.feature_available & result.model_year.eq(year))
        for start in range(0, len(positions), 128):
            indices = positions[start:start + 128]
            batch_raw, batch_history = replay._batch_features(raw, history, indices)
            values = predictor.predict(batch_raw, batch_history, result.iloc[indices].best_of.to_numpy(int))
            result.loc[indices, ["p_golgg", "p_no_w20", "p_no_organization"]] = values
        indices = positions[:3]
        batch_raw, batch_history = replay._batch_features(raw, history, indices)
        models[str(year)] = {"artifact_hashes": observed, "smoke": replay._native_smoke(predictor, batch_raw, batch_history, result.iloc[indices].best_of.to_numpy(int))}
        del predictor
        gc.collect()
    return result, models


def build_snapshot(protocol: dict, snapshot: str) -> dict:
    if snapshot not in {"baseline", "context"}:
        raise ValueError("snapshot must be baseline or context")
    output = ROOT / protocol["feature_dir"]
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / f"{snapshot}_arrays.npz"
    target_path = output / f"{snapshot}_targets.parquet"
    receipt_path = output / f"{snapshot}_receipt.json"
    prediction_path = ROOT / protocol["prediction_dir"] / f"{snapshot}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (arrays_path, target_path, receipt_path, prediction_path)):
        raise FileExistsError(f"Refusing to overwrite {snapshot} evidence")
    source_hash = sha256(ROOT / "src/analysis/ev_asof_replay.py")
    if snapshot == "baseline" and source_hash != protocol["replay_source_before_sha256"]:
        raise ValueError("Baseline must be captured before the roster-context source change")
    if snapshot == "context" and replay.REPLAY_CONTRACT_VERSION == "a0_input_repair_asof_v2":
        raise ValueError("Context snapshot requires the repaired context contract")
    capture_started_at = datetime.now(timezone.utc).isoformat()
    print(f"SNAPSHOT_SOURCE_LOADED {snapshot} {source_hash}", flush=True)
    root = protocol["resolved_research_root"]
    legacy = replay._verify_protocol(ROOT / "data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json")
    bank, releases, bank_receipt, bank_hashes = replay._verify_corrected_bank(root, legacy, protocol["paths"]["raw_history"])
    original, bank_arrays, bank_temporal = replay._read_bank_validation_data(bank, bank_receipt)
    reference = pd.read_parquet(protocol["paths"]["repair_predictions"])
    fields = ["cohort", "fixture_id", "horizon", "origin_id", "golgg_match_id", "row_id", "scenario_decision_at", "decision_at", "information_status", "scenario_timestamp_assumption", "start_at"]
    arrays = replay._build_replay_arrays(
        reference[fields], original[["golgg_match_id", "team1_id", "team2_id", "best_of", "date"]],
        root, protocol["paths"]["raw_history"], releases,
        validation_controls=replay._select_oracle_controls(original),
        bank_arrays=bank_arrays, bank_temporal=bank_temporal,
    )
    del original, bank_arrays, bank_temporal
    gc.collect()
    predictions, models = _native_predictions(arrays.targets, arrays.raw, arrays.hist, protocol)
    predictions["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    predictions["replay_contract_version"] = replay.REPLAY_CONTRACT_VERSION
    if set(predictions.origin_id) != set(reference.origin_id) or predictions.origin_id.duplicated().any():
        raise ValueError("Snapshot changed the frozen origin inventory")
    comparison = predictions.merge(reference[["origin_id", "p_golgg", "status"]], on="origin_id", suffixes=("", "_reference"), validate="one_to_one")
    common = comparison.status.eq("available") & comparison.status_reference.eq("available")
    maximum = float(abs(comparison.loc[common, "p_golgg"] - comparison.loc[common, "p_golgg_reference"]).max()) if common.any() else None
    if snapshot == "baseline":
        if not comparison.status.equals(comparison.status_reference):
            raise ValueError("Baseline availability differs from the frozen repair")
        if maximum is None or maximum > 1e-12:
            raise ValueError(f"Baseline prediction parity failed: {maximum}")
    _save_frame(arrays.targets, target_path)
    _save_frame(predictions, prediction_path)
    np.savez_compressed(arrays_path, origin_id=arrays.targets.origin_id.to_numpy(dtype=str),
                        **{"raw_" + key: value for key, value in arrays.raw.items()},
                        **{"hist_" + key: value for key, value in arrays.hist.items()})
    result = {"status": "COMPLETE_FROZEN_SNAPSHOT", "snapshot": snapshot, "run_id": protocol["run_id"],
              "replay_contract": replay.REPLAY_CONTRACT_VERSION, "source_sha256": source_hash,
              "capture_started_at_utc": capture_started_at,
              "predictions_generated_at_utc": predictions.generated_at_utc.iloc[0],
              "runner_source_sha256": sha256(Path(__file__)),
              "origins": len(predictions), "available": int(predictions.status.eq("available").sum()),
              "missing_reasons": predictions.loc[predictions.status.ne("available"), "missing_reason"].value_counts(dropna=False).to_dict(),
              "common_reference_origins": int(common.sum()), "max_abs_probability_difference_from_reference": maximum,
              "replay_validation": arrays.validation_receipt, "models": models, "bank_hashes": bank_hashes,
              "feature_hashes": arrays.feature_source_hashes,
              "output_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (arrays_path, target_path, prediction_path)},
              "limitations": ["Exact competition labels are an archived scenario, not announced lineup-unit evidence.", "All player histories retained; frozen model and calibration unchanged."]}
    save_json(receipt_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--track", required=True, choices=("baseline", "context", "stages-baseline", "stages-context", "evaluate"))
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol, args.research_root)
    if args.track in {"baseline", "context"}:
        result = build_snapshot(protocol, args.track)
    elif args.track.startswith("stages-"):
        from src.analysis.a0_stage_diagnostics import run
        result = run(protocol, protocol["resolved_research_root"], args.track.removeprefix("stages-"))
    else:
        from src.analysis.a0_context_evaluation import run
        result = run(protocol)
    print(json.dumps(replay._json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
