"""Expand a frozen raw sports trio on closed development years, then fit its head.

The inherited architecture was selected on 2024--2025H1. Those years are never
scored as this candidate's assessment; their role is closed head development.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from scripts.train_prematch_forecaster import CODE_FILES as FRESH_CODE, load_features, recipe_probability, run_origins
from src.analysis.ev_approach_data import _key_series
from src.analysis.research_benchmark import validate_frame
from src.models.a0_correction_experiment import (
    CODE_FILES, _json, _verified, directories, sha256, train, write_json,
)
from src.models.a0_uncertainty import _resolve_research_root


def prepare_fresh(protocol_path: Path, research_root: str | Path | None = None) -> dict:
    protocol = _json(protocol_path)
    paths = directories(protocol)
    _verified(protocol["parent_prepared"], protocol["parent_prepared_sha256"])
    parent = _json(protocol["parent_prepared"])
    _verified(parent["joined_path"], parent["joined_sha256"])
    root = _resolve_research_root(research_root)
    hashes = dict(parent["input_hashes"])
    for path, digest in hashes.items():
        _verified(root / path.removeprefix("research:") if path.startswith("research:") else path, digest)
    hashes[protocol["parent_prepared"]] = protocol["parent_prepared_sha256"]
    hashes[parent["joined_path"]] = parent["joined_sha256"]
    _verified(protocol["fresh_protocol"], protocol["fresh_protocol_sha256"])
    hashes[protocol["fresh_protocol"]] = protocol["fresh_protocol_sha256"]
    source = _json(protocol["fresh_protocol"])
    metadata, arrays, feature_receipt = load_features(source, Path(protocol["fresh_protocol"]))
    for key, digest in feature_receipt["feature_artifact_hashes"].items():
        hashes[feature_receipt["artifacts"][key]] = digest
    hashes[protocol["reused_2026_predictions"]] = _verified(protocol["reused_2026_predictions"], protocol["reused_2026_sha256"])
    hashes[protocol["reused_2026_receipt"]] = sha256(protocol["reused_2026_receipt"])
    old = _json(protocol["reused_2026_receipt"])
    if old["final_predictions_sha256"] != protocol["reused_2026_sha256"]:
        raise ValueError("Reused 2026 raw forecasts differ from their fit receipt")
    raw_recipe = protocol["fixed_fresh_recipe"]
    configurations = protocol["fixed_fresh_configurations"]
    names = [part["candidate"] for part in raw_recipe]
    if names != [part["id"] for part in configurations] or any(part["variant"] != "raw" for part in raw_recipe):
        raise ValueError("Invalid fixed raw trio recipe")
    if any(not np.isclose(part["weight"], 1 / 3, rtol=0., atol=0.) for part in raw_recipe):
        raise ValueError("The frozen raw trio requires equal weights")
    for record in old["fit_receipts"]:
        for name in names:
            member = record["members"][name]
            hashes[member["artifact"]] = _verified(member["artifact"], member["sha256"])
    code = {path: sha256(path) for path in dict.fromkeys((*CODE_FILES, *FRESH_CODE, "src/models/a0_fresh_correction.py"))}
    hashes.update(code)
    write_json(paths["input"] / "base_execution.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256(protocol_path),
        "code_sha256": code, "input_hashes": hashes, "new_base_fits": 24,
        "reused_base_fits": 9, "assessment_start": "2026-01-01",
        "status": "SEALED_BEFORE_NEW_BASE_FITS", "recipe_selection": "NONE_IN_THIS_RUN",
    })
    with zipfile.ZipFile(paths["input"] / "base_source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
        for path in code:
            snapshot.write(path, arcname=path)
    fitting = dict(source)
    fitting.update(model_dir=str(paths["model"].with_name(paths["model"].name + "_base")),
                   prediction_dir=str(paths["prediction"].with_name(paths["prediction"].name + "_base")),
                   candidate_variants=["raw"])
    new, _, receipts = run_origins(fitting, metadata, arrays, protocol["new_base_origins"], configurations,
                                    "closed_head_development")
    reused = pd.read_parquet(protocol["reused_2026_predictions"])
    new[protocol["specialist_column"]] = recipe_probability(new, raw_recipe)
    reused[protocol["specialist_column"]] = recipe_probability(reused, raw_recipe)
    if not np.array_equal(reused[protocol["specialist_column"]], reused.p_selected):
        raise ValueError("The reused 2026 fixed raw recipe does not reproduce saved corrected forecasts")
    fresh = pd.concat([new, reused], ignore_index=True)
    fresh["match_id"] = _key_series(fresh.match_id)
    base = pd.read_parquet(parent["joined_path"])
    if fresh.match_id.isna().any() or fresh.match_id.duplicated().any() or set(fresh.match_id) != set(base.golgg_match_id):
        raise ValueError("Fresh expansion does not preserve every canonical fixture")
    fresh = fresh.set_index("match_id").loc[base.golgg_match_id].reset_index(drop=True)
    same = _key_series(fresh.team_a).eq(base.team1_id) & _key_series(fresh.team_b).eq(base.team2_id)
    reverse = _key_series(fresh.team_a).eq(base.team2_id) & _key_series(fresh.team_b).eq(base.team1_id)
    if not (same | reverse).all():
        raise ValueError("Fresh expansion participant orientation mismatch")
    for left, right in (("date", "date"), ("release_day", "result_day"), ("best_of", "best_of")):
        if not fresh[left].astype(str).eq(base[right].astype(str)).all():
            raise ValueError(f"Fresh expansion changed {left}")
    if not np.array_equal(np.where(reverse, 1 - fresh.y, fresh.y), base.y):
        raise ValueError("Fresh expansion target orientation mismatch")
    for i, rosters in enumerate(zip(base.roster_a, base.roster_b, fresh.roster_a, fresh.roster_b)):
        sets = [set(map(str, values)) for values in rosters]
        if any(len(values) != 5 for values in sets) or sets[:2] != (sets[2:][::-1] if reverse.iloc[i] else sets[2:]):
            raise ValueError(f"Fresh expansion roster mismatch: {base.golgg_match_id.iloc[i]}")
    all_receipts = receipts + old["fit_receipts"]
    origins = {}
    for origin, rows in fresh.groupby("model_origin", sort=True):
        found = [item for item in all_receipts if item["origin"] == origin]
        if len(found) != 1:
            raise ValueError(f"No unique raw specialist receipt: {origin}")
        record = found[0]
        if (record["train_last_label_release"] >= record["training_cutoff"]
                or record["cal_last_label_release"] >= origin or len(rows) != record["n_forecasts"]):
            raise ValueError(f"Raw specialist chronology/coverage violation: {origin}")
        group = {"kind": "fixed_raw_member_group", "recipe": raw_recipe,
                 "members": {name: record["members"][name] for name in names},
                 "inherited_architecture_selection_exposure": "2024-2025H1; these years are closed head development"}
        for member in group["members"].values():
            hashes[member["artifact"]] = _verified(member["artifact"], member["sha256"])
        origins[origin] = {key: record[key] for key in (
            "training_cutoff", "n_train", "n_calibration", "n_forecasts", "train_last_label_release", "cal_last_label_release",
        )}
        origins[origin]["member"] = group
    result = base.copy()
    result["p_specialist"] = np.where(reverse, 1 - fresh[protocol["specialist_column"]], fresh[protocol["specialist_column"]])
    result["specialist_reversed"] = reverse
    result["specialist_origin"] = fresh.model_origin
    history = fresh.max_history_release_day.fillna("").astype(str)
    if (history.ge(fresh.date) & history.ne("")).any():
        raise ValueError("Fresh raw history is not strictly prior")
    result["feature_source_max_day"] = pd.concat([result.feature_source_max_day, history], axis=1).max(axis=1)
    result["specialist_release_day"] = fresh.release_day.astype(str)
    result["head_label_release_day"] = result[["result_day", "result_day_a0", "specialist_release_day"]].max(axis=1)
    validate_frame(result, ["p_a0", "p_specialist"])
    joined_path = paths["input"] / "joined.parquet"
    if joined_path.exists():
        raise FileExistsError(joined_path)
    result.to_parquet(joined_path, index=False)
    fresh_path = paths["input"] / "fixed_raw_specialist_predictions.parquet"
    fresh.to_parquet(fresh_path, index=False)
    hashes[str(fresh_path)] = sha256(fresh_path)
    evidence = {
        "status": "PREPARED", "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path), "rows": len(result),
        "reversed_specialist_rows": int(reverse.sum()), "joined_path": str(joined_path),
        "joined_sha256": sha256(joined_path), "input_hashes": hashes,
        "native_oof_provenance": parent["native_oof_provenance"], "specialist_origins": origins,
        "new_base_fits": 24, "new_unused_temperature_fits": 24, "reused_base_fits": 9,
        "information_status": "2026 exposed development; prior years closed head development, not independent selection-unseen evaluation",
    }
    write_json(paths["input"] / "prepared.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    args = parser.parse_args()
    prepared = prepare_fresh(args.protocol, args.research_root)
    print(json.dumps({"stage": "fresh-preparation", "rows": prepared["rows"], "new_base_fits": prepared["new_base_fits"]}), flush=True)
    trained = train(args.protocol, args.research_root)
    print(json.dumps({"stage": "correction-training", "rows": trained["rows"], "correction_fits": trained["correction_fits"]}), flush=True)


if __name__ == "__main__":
    main()
