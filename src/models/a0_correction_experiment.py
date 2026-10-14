"""Chronological, receipt-checked training of a small A0-anchored sports head.

This is historical research. Forecast origins are scenarios, not creation times;
all assessment years were already exposed during earlier model development.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from src.analysis.ev_approach_data import (
    _key_series, _normalize_canonical, _verify_canonical_oof_provenance,
)
from src.analysis.research_benchmark import losses, validate_frame
from src.models.a0_uncertainty import _resolve_research_root

CODE_FILES = (
    "scripts/train_a0_sports_correction.py",
    "src/models/a0_correction_experiment.py",
    "src/models/a0_sports_correction.py",
    "src/analysis/a0_sports_correction.py",
    "src/analysis/ev_approach_data.py",
    "src/models/a0_uncertainty.py",
    "src/analysis/research_benchmark.py",
    "src/analysis/model_benchmark.py",
    "src/analysis/probability_metrics.py",
    "src/analysis/metrics.py",
    "src/analysis/bootstrap.py",
    "src/analysis/production_market_benchmark.py",
    "src/models/competition_tiers.py",
)
FAMILIES = {"control": "p_calibration", "candidate": "p_correction"}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")


def _verified(path: str | Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Changed frozen input: {path}")
    return actual


def _json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def directories(protocol: dict) -> dict[str, Path]:
    run = protocol["run_id"]
    return {key: Path(f"data/{stage}/a0_sports_correction/{run}") for key, stage in (
        ("input", "05_model_input"), ("model", "06_models"),
        ("prediction", "07_model_output"), ("report", "08_reporting"),
    )}


def prepare(protocol_path: Path, research_root: str | Path | None = None) -> dict:
    if _json(protocol_path).get("schema") == "c0_raw_specialist_connector_v1":
        return prepare_c0(protocol_path, research_root)
    protocol = _json(protocol_path)
    paths = directories(protocol)
    if (paths["input"] / "prepared.json").exists() or (paths["input"] / "joined.parquet").exists():
        raise FileExistsError("Prepared experiment already exists; do not overwrite it")
    root = _resolve_research_root(research_root)
    hashes = {protocol["manifest"]: _verified(protocol["manifest"], protocol["manifest_sha256"])}
    canonical_path = root / protocol["canonical_relative"]
    hashes["research:" + protocol["canonical_relative"]] = _verified(canonical_path, protocol["canonical_sha256"])
    base = _normalize_canonical(pd.read_parquet(canonical_path), protocol["evaluation"]["expected_rows"])
    native = _verify_canonical_oof_provenance(base, root, root / "a0-phase-walkforward-20260914")
    specialist_path = Path(protocol["specialist_predictions"])
    hashes[str(specialist_path)] = _verified(specialist_path, protocol["specialist_predictions_sha256"])
    specialist_protocol = _json(protocol["specialist_protocol"])
    receipt = _json(protocol["specialist_receipt"])
    if receipt.get("status") != "COMPLETE_BASE_MODELS":
        raise ValueError("Specialist fitting receipt is incomplete")
    if receipt["prediction_hashes"].get(str(specialist_path)) != hashes[str(specialist_path)]:
        raise ValueError("Specialist predictions do not match their fitting receipt")
    prepared_path = Path(specialist_protocol["report_dir"]) / "architecture_prepared.json"
    prepared = _json(prepared_path)
    _verified(protocol["specialist_protocol"], prepared["protocol_sha256"])
    for path in (protocol["specialist_protocol"], protocol["specialist_receipt"], str(prepared_path)):
        hashes[path] = sha256(path)
    compatibility_path = paths["input"] / "source_compatibility.json"
    compatibility = _json(compatibility_path) if compatibility_path.exists() else {"changes": {}}
    if compatibility_path.exists():
        hashes[str(compatibility_path)] = sha256(compatibility_path)
        hashes[compatibility["source_snapshot"]] = _verified(
            compatibility["source_snapshot"], compatibility["source_snapshot_sha256"],
        )
    for path, expected in prepared["code_sha256"].items():
        reviewed = compatibility["changes"].get(path)
        if reviewed is None:
            hashes[path] = _verified(path, expected)
            continue
        if path != "src/analysis/research_benchmark.py" or reviewed["frozen_sha256"] != expected:
            raise ValueError(f"Unreviewed specialist implementation change: {path}")
        with zipfile.ZipFile(compatibility["source_snapshot"]) as snapshot:
            if hashlib.sha256(snapshot.read(path)).hexdigest() != expected:
                raise ValueError("Historical reporting source differs from its snapshot")
        hashes[path] = _verified(path, reviewed["current_sha256"])
    fresh = pd.read_parquet(specialist_path)
    fresh["match_id"] = _key_series(fresh.match_id)
    if fresh.match_id.isna().any() or fresh.match_id.duplicated().any():
        raise ValueError("Specialist identities are invalid")
    if set(fresh.match_id) != set(base.golgg_match_id):
        raise ValueError("Specialist forecast coverage differs from canonical coverage")
    fresh = fresh.set_index("match_id").loc[base.golgg_match_id].reset_index(drop=True)
    team_a, team_b = _key_series(fresh.team_a), _key_series(fresh.team_b)
    reverse = team_a.eq(base.team2_id) & team_b.eq(base.team1_id)
    same = team_a.eq(base.team1_id) & team_b.eq(base.team2_id)
    if not (reverse | same).all():
        raise ValueError("Specialist participant identities disagree with canonical sides")
    for left, right in (("date", "date"), ("release_day", "result_day"), ("best_of", "best_of")):
        if not fresh[left].astype(str).eq(base[right].astype(str)).all():
            raise ValueError(f"Specialist/canonical {left} disagree")
    if not np.array_equal(np.where(reverse, 1 - fresh.y, fresh.y), base.y):
        raise ValueError("Specialist target orientation disagrees with canonical labels")
    # Both experts must see the same actual-roster scenario, irrespective of slot order.
    for i, (native_a, native_b, fresh_a, fresh_b) in enumerate(zip(
        base.roster_a, base.roster_b, fresh.roster_a, fresh.roster_b,
    )):
        rosters = [list(map(str, values)) for values in (native_a, native_b, fresh_a, fresh_b)]
        if any(len(values) != 5 or len(set(values)) != 5 for values in rosters):
            raise ValueError(f"Invalid five-player roster for {base.golgg_match_id.iloc[i]}")
        native_rosters = [set(values) for values in rosters[:2]]
        fresh_rosters = [set(values) for values in rosters[2:]]
        if reverse.iloc[i]:
            fresh_rosters.reverse()
        if native_rosters != fresh_rosters:
            raise ValueError(f"Specialist/canonical roster mismatch for {base.golgg_match_id.iloc[i]}")
    checked_origins = {}
    for origin, rows in fresh.groupby("model_origin", sort=True):
        fit_records = [r for r in receipt["fit_receipts"] if r["origin"] == origin]
        if len(fit_records) != 1:
            raise ValueError(f"No unique specialist fit receipt for {origin}")
        fit = fit_records[0]
        cutoff = (pd.Timestamp(origin) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        if (fit["training_cutoff"] != cutoff or not rows.training_cutoff.eq(cutoff).all()
                or fit["train_last_label_release"] >= cutoff
                or fit["cal_last_label_release"] >= origin
                or not rows.date.ge(origin).all()
                or not rows.date.lt((pd.Timestamp(origin) + pd.DateOffset(months=3)).strftime("%Y-%m-%d")).all()
                or fit["n_forecasts"] != len(rows)):
            raise ValueError(f"Specialist chronology/coverage receipt failed at {origin}")
        member = fit["members"]["history_representation"]
        hashes[member["artifact"]] = _verified(member["artifact"], member["sha256"])
        checked_origins[origin] = {key: fit[key] for key in (
            "training_cutoff", "n_train", "n_calibration", "n_forecasts",
            "train_last_label_release", "cal_last_label_release",
        )}
        checked_origins[origin]["member"] = member
    result = base.copy()
    result["p_a0"] = result.p_full
    result["p_specialist"] = np.where(reverse, 1 - fresh[protocol["specialist_column"]], fresh[protocol["specialist_column"]])
    result["specialist_reversed"] = reverse
    result["specialist_origin"] = fresh.model_origin
    result["specialist_release_day"] = fresh.release_day.astype(str)
    for field in ("result_day_a0", "specialist_release_day"):
        result[field] = pd.to_datetime(result[field], errors="raise").dt.strftime("%Y-%m-%d")
        if result[field].isna().any() or result[field].lt(result.date).any():
            raise ValueError(f"Invalid result availability: {field}")
    result["head_label_release_day"] = result[["result_day", "result_day_a0", "specialist_release_day"]].max(axis=1)
    fresh_history = fresh.feature_source_max_day.fillna("").astype(str)
    if (fresh_history.ge(fresh.date.astype(str)) & fresh_history.ne("")).any():
        raise ValueError("Specialist feature history is not strictly prior")
    result["feature_source_max_day"] = pd.concat([result.feature_source_max_day, fresh_history], axis=1).max(axis=1)
    validate_frame(result, ["p_a0", "p_specialist"])
    if result.date.ge(protocol["evaluation"]["primary_start"]).sum() != protocol["evaluation"]["primary_expected_rows"]:
        raise ValueError("Primary cohort changed")
    paths["input"].mkdir(parents=True, exist_ok=True)
    joined_path = paths["input"] / "joined.parquet"
    result.to_parquet(joined_path, index=False)
    evidence = {
        "status": "PREPARED", "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path), "rows": len(result),
        "reversed_specialist_rows": int(reverse.sum()),
        "joined_path": str(joined_path), "joined_sha256": sha256(joined_path),
        "input_hashes": hashes, "native_oof_provenance": native,
        "specialist_origins": checked_origins,
        "information_status": "matched actual-roster daily scenario; not certified publication or issuance",
    }
    write_json(paths["input"] / "prepared.json", evidence)
    return evidence


def _validate_c0_protocol(protocol: dict) -> None:
    """Reject a changed Plan09 recipe before input preparation or any head fit."""
    origins = ["2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01", "2025-07-01",
               "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]
    columns = {"annual_a0": "p_a0", "c0": "p_c0", "specialist": "p_specialist",
               "connector03": "p_connector03", "c0_calibration": "p_c0_calibration",
               "c0_connector": "p_c0_connector"}
    required = {
        None: {"schema": "c0_raw_specialist_connector_v1", "probability_columns": columns},
        "cohort": {"rows": 11550, "primary_rows": 3103, "warmup_rows": 8447,
                   "primary_start": "2026-01-01", "primary_end_exclusive": "2026-09-09"},
        "model": {"arms": ["c0_connector", "c0_calibration"], "anchor_identity": "native12__full__full",
                  "alpha_bounds": [0, 1], "slope_bounds": [0, "unbounded"],
                  "penalties": [0.0001, 0.001, 0.01, 0.1], "minimum_fit_rows": 500,
                  "base_fits": 0, "head_fits": 72},
        "chronology": {"cv_origins": origins, "forecast_origins": origins[-3:],
                       "interval": "origin-inclusive and next-quarter-exclusive"},
        "evaluation": {"primary_contrasts": ["p_c0", "p_c0_calibration", "p_connector03"],
                       "guards": ["best_of_3", "best_of_5", "major", "source_unseen_player"],
                       "guard_comparators": ["p_c0", "p_a0"], "resamples": 5000, "seed": 20261002,
                       "shared_calendar_month_draws": True, "equal_month": True,
                       "zero_support_replicates": "retain and mark conditional interval unresolved",
                       "source_tournament_year_proxy_blocks": 143},
        "inference": {"mode": "full_context_series_winner_only", "valid_target_interval_required": True,
                      "anchor_identity_required": True, "strict_interior_probabilities": True,
                      "new_clipping": False},
        "replay_tolerances": {"native_upstream": 2e-6, "head_predictions": 1e-12,
                              "independent_head_optimum": 1e-12},
    }
    for section, fields in required.items():
        actual = protocol if section is None else protocol.get(section, {})
        if not isinstance(actual, dict):
            raise ValueError(f"Invalid frozen Plan09 section: {section}")
        for key, expected in fields.items():
            if actual.get(key) != expected:
                raise ValueError(f"Changed or missing frozen Plan09 field: {section or 'root'}.{key}")
    if protocol.get("classification") not in {"EXPOSED_HISTORICAL_DEVELOPMENT", "SYNTHETIC_SOFTWARE_SMOKE"}:
        raise ValueError("Plan09 must declare exposed history or explicitly synthetic software inputs")
    if protocol.get("output_contract", {}).get("prediction_columns") != list(columns.values()):
        raise ValueError("Plan09 output identities differ from its six declared models")


def prepare_c0(protocol_path: Path, research_root: str | Path | None = None) -> dict:
    """Assemble an exact keyed Plan09 C0/raw/control bundle without refitting sources."""
    protocol = _json(protocol_path)
    _validate_c0_protocol(protocol)
    paths = directories(protocol)
    if any((paths["input"] / name).exists() for name in ("prepared.json", "joined.parquet")):
        raise FileExistsError("Plan09 prepared input already exists")
    if protocol["market_diagnostic"]["enabled"]:
        _verified(protocol["market_diagnostic"]["protocol"], protocol["market_diagnostic"]["sha256"])
    sources = protocol["sources"]
    root = _resolve_research_root(research_root)
    canonical_path = root / sources["canonical"]["path"].removeprefix("research:")
    input_hashes = {"research:" + sources["canonical"]["path"].removeprefix("research:"):
                    _verified(canonical_path, sources["canonical"]["sha256"])}
    source_frames = {}
    input_hashes = dict(input_hashes)
    for name, source in sources.items():
        if name == "canonical":
            continue
        path = Path(source["path"])
        input_hashes[str(path)] = _verified(path, source["sha256"])
        if path.suffix == ".parquet":
            source_frames[name] = pd.read_parquet(path)
    lineage = _json(sources["prepared_lineage"]["path"])
    lineage_inputs = {}
    historical_code_divergences = {}
    lineage_dir = Path(sources["prepared_lineage"]["path"]).parent
    source_archive = lineage_dir / "source_snapshot.zip"
    base_archive = lineage_dir / "base_source_snapshot.zip"
    proof = _json("data/08_reporting/a0_sports_correction/a0_fresh_correction_20261001_02/decision.json")
    for archive in (source_archive, base_archive):
        expected_archive = proof["proof_sha256"].get(str(archive))
        if expected_archive is None:
            raise ValueError(f"No immutable receipt pins source archive {archive}")
        _verified(archive, expected_archive)
    for source, digest in lineage["input_hashes"].items():
        resolved = root / source.removeprefix("research:") if source.startswith("research:") else Path(source)
        current = sha256(resolved)
        if current != digest:
            if Path(source).suffix != ".py":
                raise ValueError(f"Changed frozen lineage input: {resolved}")
            archived = False
            for archive in (source_archive, base_archive):
                try:
                    with zipfile.ZipFile(archive) as snapshot:
                        archived = hashlib.sha256(snapshot.read(source)).hexdigest() == digest
                except KeyError:
                    archived = False
                if archived:
                    break
            if not archived:
                raise ValueError(f"Historical source code is absent from its pinned archive: {source}")
            historical_code_divergences[source] = {"receipt_sha256": digest, "current_sha256": current,
                                                    "archive": str(archive), "status": "ARCHIVED_SOURCE_VERIFIED"}
        else:
            lineage_inputs[source] = digest
    raw_producer = "src/models/a0_fresh_correction.py"
    producer_sha = lineage["input_hashes"].get(raw_producer)
    if producer_sha is None:
        raise ValueError("Prepared lineage lacks the original raw prediction producer hash")
    producer_archive = None
    for archive in (source_archive, base_archive):
        try:
            with zipfile.ZipFile(archive) as snapshot:
                if hashlib.sha256(snapshot.read(raw_producer)).hexdigest() == producer_sha:
                    producer_archive = archive
                    break
        except KeyError:
            continue
    if producer_archive is None:
        raise ValueError("Original raw prediction producer is not present in its pinned source archive")
    raw_alignment_evidence = {"producer": raw_producer, "producer_sha256": producer_sha,
        "producer_archive": str(producer_archive), "producer_archive_sha256": sha256(producer_archive),
        "producer_identity_operation": "match_id is unique-validated, then fresh.set_index(match_id).loc[base.golgg_match_id].reset_index(drop=True) before raw export"}
    input_hashes.update(lineage_inputs)
    c0 = source_frames["native12_c0"]
    a0 = source_frames["annual_a0_specialist_connector03"]
    native_protocol_path = sources["native12_c0"]["protocol"]
    native_protocol_sha = _verified(native_protocol_path, sources["native12_c0"]["protocol_sha256"])
    native_protocol = _json(native_protocol_path)
    native_inputs = {}
    native_inputs[native_protocol_path] = native_protocol_sha

    def native_path(locator: str) -> Path:
        direct = Path(locator)
        candidates = (direct, root / locator, root.parent / locator)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Native C0 receipt input is unavailable: {locator}")

    for locator, digest in native_protocol["inputs"].items():
        resolved = native_path(locator)
        native_inputs[locator] = _verified(resolved, digest)
    native_year_receipts = {}
    for year, origin in native_protocol["reference_origins"].items():
        completion = origin.get("redirected_completion") or origin
        completion_artifacts = completion.get("artifacts", {})
        artifacts = dict(origin.get("artifacts", {}))
        artifacts.update(completion_artifacts)
        verified_artifacts = {locator: _verified(locator, digest)
                              for locator, digest in artifacts.items()}
        completion_path = completion.get("completed_path", origin.get("completed_path"))
        completion_sha = completion.get("completed_sha256", origin.get("completed_sha256"))
        if not completion_path or not completion_sha:
            raise ValueError(f"Native C0 {year} completion receipt is missing")
        completion_receipt_sha = _verified(completion_path, completion_sha)
        completion_receipt = _json(completion_path)
        if completion_receipt.get("artifacts") != completion_artifacts:
            raise ValueError(f"Native C0 {year} completion artifact closure changed")
        output_path = origin["native12_output_path"]
        output_sha = _verified(output_path, origin["native12_output_sha256"])
        execution_sha = _verified(origin["control_execution_path"], origin["control_execution_sha256"])
        native_inputs.update(verified_artifacts)
        native_inputs[completion_path] = completion_receipt_sha
        native_inputs[output_path] = output_sha
        native_inputs[origin["control_execution_path"]] = execution_sha
        native_year_receipts[year] = {"completed_path": completion_path,
                                      "completed_sha256": completion_receipt_sha,
                                      "artifact_count": len(verified_artifacts),
                                      "native12_output_path": output_path,
                                      "native12_output_sha256": output_sha}
    input_hashes.update(native_inputs)
    raw = pd.read_parquet(sources["raw_specialist"]["path"])
    canonical = _normalize_canonical(pd.read_parquet(canonical_path), protocol["cohort"]["rows"])
    keys = protocol["cohort"]["keys"]
    required_c0 = set(keys) | {sources["native12_c0"]["column"]}
    required_a0 = set(keys) | set(sources["annual_a0_specialist_connector03"]["columns"])
    if not required_c0.issubset(c0) or not required_a0.issubset(a0):
        raise ValueError("Pinned C0 or connector source lacks required keys/probabilities")
    if c0.golgg_match_id.astype(str).duplicated().any() or a0.golgg_match_id.astype(str).duplicated().any():
        raise ValueError("Pinned source has duplicate match identities")
    canonical["golgg_match_id"] = _key_series(canonical.golgg_match_id)
    lineage_joined_path = Path(lineage["joined_path"])
    _verified(lineage_joined_path, lineage["joined_sha256"])
    lineage_base = pd.read_parquet(lineage_joined_path)
    if not np.array_equal(canonical.golgg_match_id.astype(str), lineage_base.golgg_match_id.astype(str)):
        raise ValueError("Original prepared row order differs from the canonical cohort")
    raw_a, raw_b = _key_series(raw.team_a).astype(str), _key_series(raw.team_b).astype(str)
    lineage_a, lineage_b = _key_series(lineage_base.team1_id).astype(str), _key_series(lineage_base.team2_id).astype(str)
    raw_keys = list(zip(raw.date.astype(str), raw.release_day.astype(str), raw.best_of.astype(str),
                        np.minimum(raw_a, raw_b), np.maximum(raw_a, raw_b)))
    lineage_keys = list(zip(lineage_base.date.astype(str), lineage_base.result_day.astype(str),
                            lineage_base.best_of.astype(str), np.minimum(lineage_a, lineage_b),
                            np.maximum(lineage_a, lineage_b)))
    if raw_keys != lineage_keys:
        raise ValueError("Raw specialist row order differs from the exact original prepared identity map")
    raw_key_counts = pd.Series(raw_keys, dtype="object").value_counts()
    lineage_key_counts = pd.Series(lineage_keys, dtype="object").value_counts()
    if not raw_key_counts.equals(lineage_key_counts):
        raise ValueError("Raw/canonical duplicate-key profile differs")
    raw_alignment_evidence.update({"raw_rows": len(raw), "canonical_rows": len(lineage_base),
        "ordered_raw_identity_checks": len(raw_keys),
        "duplicate_identity_groups": int(raw_key_counts.gt(1).sum()),
        "rows_in_duplicate_identity_groups": int(raw_key_counts[raw_key_counts.gt(1)].sum()),
        "duplicate_profile_matches_canonical": True,
        "identity_key_fields": ["date", "release_day/result_day", "best_of", "unordered_team_ids"],
        "same_day_doubleheaders_disambiguated_by": "archived_unique_match_id_row_order"})
    raw = raw.assign(golgg_match_id=lineage_base.golgg_match_id.astype(str).to_numpy())
    canonical = canonical.set_index("golgg_match_id", drop=False)
    expected = set(canonical.index)
    c0 = c0.assign(golgg_match_id=_key_series(c0.golgg_match_id)).set_index("golgg_match_id", drop=False)
    a0 = a0.assign(golgg_match_id=_key_series(a0.golgg_match_id)).set_index("golgg_match_id", drop=False)
    raw = raw.set_index("golgg_match_id", drop=False)
    if any(set(frame.index) != expected for frame in (c0, a0, raw)):
        raise ValueError("Pinned source ID coverage differs from canonical")
    ids = canonical.index
    out = canonical.copy()
    for name, frame in (("c0", c0), ("a0", a0)):
        for key in keys:
            left = out[key].astype(str).to_numpy()
            right = frame.loc[ids, key].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"{name} source mismatch in {key}")
    for column, source, field in (
        ("p_a0", a0, "p_a0"), ("p_specialist", a0, "p_specialist"),
        ("p_connector03", a0, "p_correction"),
        ("p_c0", c0, sources["native12_c0"]["column"]),
    ):
        out[column] = source.loc[ids, field].to_numpy(float)
    raw_aligned = raw.loc[ids]
    same = (_key_series(raw_aligned.team_a).to_numpy() == out.team1_id.astype(str).to_numpy()) & (_key_series(raw_aligned.team_b).to_numpy() == out.team2_id.astype(str).to_numpy())
    reverse = (_key_series(raw_aligned.team_a).to_numpy() == out.team2_id.astype(str).to_numpy()) & (_key_series(raw_aligned.team_b).to_numpy() == out.team1_id.astype(str).to_numpy())
    if not np.all(same | reverse):
        raise ValueError("Raw specialist participants disagree with canonical sides")
    for raw_field, canonical_field in (("date", "date"), ("release_day", "result_day"), ("best_of", "best_of")):
        if not raw_aligned[raw_field].astype(str).eq(out[canonical_field].astype(str)).all():
            raise ValueError(f"Raw specialist/canonical {raw_field} mismatch")
    raw_probability = raw_aligned[sources["raw_specialist"]["column"]].to_numpy(float)
    out["p_specialist"] = np.where(reverse, 1. - raw_probability, raw_probability)
    if not np.array_equal(np.where(reverse, 1 - raw_aligned.y.to_numpy(int), raw_aligned.y.to_numpy(int)), out.y.to_numpy(int)):
        raise ValueError("Raw specialist outcomes disagree after side normalization")
    for column in ("p_a0", "p_specialist"):
        if not np.array_equal(out[column].to_numpy(), a0.loc[ids, column].to_numpy(float)):
            raise ValueError(f"Raw lineage differs from retained {column}")
    out["p_c0_calibration"] = out.p_c0
    out["p_c0_connector"] = out.p_c0
    out["head_label_release_day"] = lineage_base.head_label_release_day.astype(str).to_numpy()
    out["head_origin"] = "C0_WARMUP"
    out["head_status"] = "C0_WARMUP"
    primary_rows = int(pd.to_datetime(out.date).ge(protocol["cohort"]["primary_start"]).sum())
    if len(out) != protocol["cohort"]["rows"] or primary_rows != protocol["cohort"]["primary_rows"]:
        raise ValueError("Plan09 complete/primary cohort size changed")
    out.loc[pd.to_datetime(out.date).ge(protocol["cohort"]["primary_start"]), "head_status"] = "PENDING_C0_HEAD"
    out["head_label_release_day"] = pd.to_datetime(out.head_label_release_day).dt.strftime("%Y-%m-%d")
    validate_frame(out, ["p_a0", "p_c0", "p_specialist", "p_connector03", "p_c0_calibration", "p_c0_connector"])
    paths["input"].mkdir(parents=True, exist_ok=True)
    joined = paths["input"] / "joined.parquet"
    out.to_parquet(joined, index=False)
    receipt = {"status": "PREPARED", "schema": "plan09_c0_input_receipt_v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256(protocol_path),
               "rows": len(out), "primary_rows": int(pd.to_datetime(out.date).ge("2026-01-01").sum()),
               "warmup_rows": int(pd.to_datetime(out.date).lt("2026-01-01").sum()),
               "raw_reversed_rows": int(reverse.sum()), "joined_path": str(joined),
               "joined_sha256": sha256(joined), "input_hashes": input_hashes,
               "native_anchor_identity": "native12__full__full",
               "native12_recipe_protocol": sources["native12_c0"]["protocol"],
               "native12_recipe_protocol_sha256": _verified(
                   sources["native12_c0"]["protocol"], sources["native12_c0"]["protocol_sha256"]),
               "native12_year_receipts": native_year_receipts,
               "raw_alignment_evidence": raw_alignment_evidence,
               "historical_code_divergences": historical_code_divergences,
               "lineage": _json("data/05_model_input/a0_sports_correction/a0_fresh_correction_20261001_02/prepared.json")}
    write_json(paths["input"] / "prepared.json", receipt)
    return receipt


def training_indices(frame: pd.DataFrame, origin: str) -> np.ndarray:
    return np.flatnonzero((frame.date.lt(origin) & frame.head_label_release_day.lt(origin)).to_numpy())


def _quarter_end(origin: str) -> str:
    return (pd.Timestamp(origin) + pd.DateOffset(months=3)).strftime("%Y-%m-%d")


def _prediction_column(family: str, penalty: float) -> str:
    return f"{family}__{penalty:g}"


def select_penalty(frame: pd.DataFrame, cv: pd.DataFrame, origin: str, family: str, grid: list[float]) -> tuple[float, dict]:
    """Select on earlier-origin predictions with outcomes released before origin."""
    eligible = (frame.date.lt(origin) & frame.head_label_release_day.lt(origin)
                & cv.cv_origin.notna() & cv.cv_origin.lt(origin))
    count = int(eligible.sum())
    if count == 0:
        return max(grid), {"n": 0, "losses": {}, "reason": "NO_PAST_VALIDATION_SUPPORT"}
    scores = {}
    for penalty in grid:
        p = cv.loc[eligible, _prediction_column(family, penalty)].to_numpy(float)
        scores[str(penalty)] = float(losses(frame.loc[eligible, "y"].to_numpy(), p).mean())
    minimum = min(scores.values())
    chosen = max(penalty for penalty in grid if scores[str(penalty)] <= minimum + 1e-12)
    return chosen, {"n": count, "losses": scores, "reason": "PAST_ROLLING_ORIGIN_LOG_LOSS",
                    "last_label_release": frame.loc[eligible, "head_label_release_day"].max(),
                    "validation_ids": frame.loc[eligible, "golgg_match_id"].tolist()}


def _verify_c0_prepared_sources(prepared: dict, protocol: dict, research_root: Path) -> dict:
    """Verify data bytes and retain exact archived historical producers after code cutover."""
    from src.analysis.ev_approach_data import _research_artifact_path

    entries = dict(prepared["input_hashes"])
    entries.update({name: value["receipt_sha256"]
                    for name, value in prepared.get("historical_code_divergences", {}).items()})
    result = {"files": {}, "archived_sources": {}, "source_archives": {}}
    archives = None
    for source, expected in entries.items():
        if source.startswith("research:"):
            path = _research_artifact_path(source.removeprefix("research:"), research_root)
        else:
            path = Path(source) if Path(source).is_file() else _research_artifact_path(source, research_root)
        actual = sha256(path)
        if actual == expected:
            result["files"][source] = expected
            continue
        if Path(source).suffix != ".py":
            raise ValueError(f"Changed frozen C0 source input: {source}")
        if archives is None:
            lineage = Path(protocol["sources"]["prepared_lineage"]["path"])
            decision_path = Path("data/08_reporting/a0_sports_correction") / lineage.parent.name / "decision.json"
            proof = _json(decision_path)
            result["historical_decision_sha256"] = sha256(decision_path)
            archives = [lineage.parent / name for name in ("source_snapshot.zip", "base_source_snapshot.zip")]
            for archive in archives:
                digest = proof["proof_sha256"].get(str(archive))
                if digest is None:
                    raise ValueError(f"Unpinned historical source archive: {archive}")
                result["source_archives"][str(archive)] = _verified(archive, digest)
        for archive in archives:
            with zipfile.ZipFile(archive) as snapshot:
                if source in snapshot.namelist() and hashlib.sha256(snapshot.read(source)).hexdigest() == expected:
                    result["archived_sources"][source] = {
                        "receipt_sha256": expected, "current_sha256": actual, "archive": str(archive),
                    }
                    break
        else:
            raise ValueError(f"Historical producer absent from verified source archives: {source}")
    return result


def _set_c0_release_bounds(frame: pd.DataFrame) -> None:
    """Attach conservative complete-recipe release bounds, never fit/creation clocks."""
    dates = pd.to_datetime(frame.date, errors="raise")
    warmup = frame.head_status.eq("C0_WARMUP")
    annual = dates.dt.year.astype(str) + "-01-01"
    origins = pd.to_datetime(frame.head_origin.where(~warmup, annual), errors="raise")
    if origins.isna().any() or (origins > dates).any():
        raise ValueError("Complete C0 recipe origin is absent or after its target")
    ends = origins + pd.offsets.QuarterBegin(startingMonth=1)
    if ((~warmup) & (dates >= ends)).any():
        raise ValueError("Active C0 head is outside its fitted quarter")
    frame["complete_recipe_origin"] = origins.dt.strftime("%Y-%m-%d")
    bound = (origins - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    frame["train_end"] = bound
    frame["calibration_end"] = bound
    frame["fit_date_semantics"] = (
        "conservative last-permitted released-label bound from verified annual C0 and "
        "closed raw/head/selection origins; not actual fit or forecast creation time"
    )


def train(protocol_path: Path, research_root: str | Path | None = None) -> dict:
    if _json(protocol_path).get("schema") == "c0_raw_specialist_connector_v1":
        from src.models.a0_probability_connector import train_c0
        return train_c0(protocol_path, research_root)

    protocol = _json(protocol_path)
    paths = directories(protocol)
    prepared = _json(paths["input"] / "prepared.json")
    _verified(protocol_path, prepared["protocol_sha256"])
    _verified(prepared["joined_path"], prepared["joined_sha256"])
    root = _resolve_research_root(research_root)
    for path, digest in prepared["input_hashes"].items():
        _verified(root / path.removeprefix("research:") if path.startswith("research:") else path, digest)
    if paths["model"].exists() or paths["prediction"].exists():
        raise FileExistsError("Experiment model/prediction directory already exists")
    paths["model"].mkdir(parents=True)
    paths["prediction"].mkdir(parents=True)
    execution = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "prepared_sha256": sha256(paths["input"] / "prepared.json"),
        "code_sha256": {path: sha256(path) for path in CODE_FILES},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "status": "SEALED_BEFORE_CORRECTION_FITS", "no_base_expert_refits": True,
    }
    write_json(paths["input"] / "execution.json", execution)
    with zipfile.ZipFile(paths["input"] / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
        for path in CODE_FILES:
            snapshot.write(path, arcname=path)
    frame = pd.read_parquet(prepared["joined_path"])
    cv = pd.DataFrame(index=frame.index)
    cv["cv_origin"] = pd.Series(index=frame.index, dtype="string")
    grid = protocol["model"]["regularization_grid"]
    fitted = {}
    fit_records = []
    for origin in protocol["chronology"]["cv_origins"]:
        indices = training_indices(frame, origin)
        if len(indices) < protocol["chronology"]["minimum_fit_rows"]:
            raise ValueError(f"Insufficient closed training support at {origin}")
        validation = np.flatnonzero((frame.date.ge(origin) & frame.date.lt(_quarter_end(origin))).to_numpy())
        train_frame, validation_frame = frame.iloc[indices], frame.iloc[validation]
        cv.loc[validation, "cv_origin"] = origin
        membership_path = paths["model"] / origin / "training_membership.json"
        write_json(membership_path, {"origin": origin, "ids": train_frame.golgg_match_id.tolist(),
                   "n": len(indices), "last_release": train_frame.head_label_release_day.max()})
        for family in FAMILIES:
            for penalty in grid:
                model = AnchoredSportsCorrection(include_specialist=family == "candidate", penalty=penalty,
                                                 knot_probability=protocol["model"]["knot_probability"])
                model.fit(train_frame.p_a0.to_numpy(), train_frame.p_specialist.to_numpy(),
                          train_frame.best_of.to_numpy(), train_frame.y.to_numpy())
                prediction = model.predict_proba(validation_frame.p_a0.to_numpy(), validation_frame.p_specialist.to_numpy(),
                                                validation_frame.best_of.to_numpy())
                cv.loc[validation, _prediction_column(family, penalty)] = prediction
                model_path = paths["model"] / origin / f"{family}_{penalty:g}.json"
                record = {"origin": origin, "family": family, "penalty": penalty,
                          "training_membership": str(membership_path), "membership_sha256": sha256(membership_path),
                          "model": model.to_dict(), "fit_diagnostics": model.fit_diagnostics}
                write_json(model_path, record)
                fitted[(origin, family, penalty)] = (model, model_path)
                fit_records.append({"path": str(model_path), "sha256": sha256(model_path),
                                    "origin": origin, "family": family, "penalty": penalty,
                                    "n_train": len(indices), "n_validation": len(validation)})
        print(json.dumps({"stage": "fit", "origin": origin, "n_train": len(indices), "n_validation": len(validation), "fits": 8}), flush=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    result = frame.copy()
    for column in FAMILIES.values():
        result[column] = result.p_a0
    result["head_status"] = "WARMUP_IDENTITY_A0"
    result["head_origin"] = result.date.str[:4] + "-01-01"
    selections = []
    for origin in protocol["chronology"]["forecast_origins"]:
        target = frame.date.ge(origin) & frame.date.lt(_quarter_end(origin))
        for family, column in FAMILIES.items():
            penalty, selection = select_penalty(frame, cv, origin, family, grid)
            model, model_path = fitted[(origin, family, penalty)]
            result.loc[target, column] = cv.loc[target, _prediction_column(family, penalty)]
            selected = {"origin": origin, "family": family, "penalty": penalty,
                        "selection": selection, "model_path": str(model_path), "sha256": sha256(model_path),
                        "series_probability_inputs": ["native_annual_A0_full", protocol["specialist_column"]],
                        "specialist_artifact": prepared["specialist_origins"][origin]["member"],
                        "native_recipe": prepared["native_oof_provenance"]["years"][origin[:4]],
                        "valid_target_interval": [origin, _quarter_end(origin)], "generated_at": generated_at,
                        "research_only": True}
            write_json(paths["model"] / origin / f"selected_{family}.json", selected)
            selections.append(selected)
        result.loc[target, "head_status"] = "PAST_ONLY_FITTED_CORRECTION"
        result.loc[target, "head_origin"] = origin
    result["train_end"] = (pd.to_datetime(result.head_origin) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    result["calibration_end"] = result.train_end
    result["fit_date_semantics"] = "conservative last-allowed-day bound, not claimed actual creation time"
    result["generated_at"] = generated_at
    validate_frame(result, ["p_a0", "p_specialist", *FAMILIES.values()])
    cv.insert(0, "golgg_match_id", frame.golgg_match_id)
    cv.to_parquet(paths["prediction"] / "validation_grid.parquet", index=False)
    prediction_path = paths["prediction"] / "predictions.parquet"
    result.to_parquet(prediction_path, index=False)
    evidence = {"status": "COMPLETE_CORRECTION_FITS", "created_at": generated_at,
                "protocol_sha256": sha256(protocol_path), "execution_sha256": sha256(paths["input"] / "execution.json"),
                "fits": fit_records, "selections": selections, "rows": len(result),
                "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
                "warmup_rows": int(result.head_status.eq("WARMUP_IDENTITY_A0").sum()),
                "base_fits": 0, "correction_fits": len(fit_records)}
    write_json(paths["report"] / "training.json", evidence)
    return evidence


def evaluate(protocol_path: Path) -> dict:
    if _json(protocol_path).get("schema") == "c0_raw_specialist_connector_v1":
        from src.analysis.a0_sports_correction import evaluate_c0
        protocol = _json(protocol_path)
        paths = directories(protocol)
        receipt = _json(paths["report"] / "training.json")
        _validate_c0_protocol(protocol)
        _verified(protocol_path, receipt["protocol_sha256"])
        _verified(receipt["prediction_path"], receipt["prediction_sha256"])
        _verified(paths["input"] / "execution.json", receipt["execution_sha256"])
        for path, digest in _json(paths["input"] / "execution.json")["code_sha256"].items():
            _verified(path, digest)
        result = evaluate_c0(pd.read_parquet(paths["prediction"] / "predictions.parquet"), protocol)
        result["protocol_sha256"] = sha256(protocol_path)
        result["training_receipt_sha256"] = sha256(paths["report"] / "training.json")
        result["prediction_sha256"] = receipt["prediction_sha256"]
        write_json(paths["report"] / "results.json", result)
        return result
    from src.analysis.a0_sports_correction import evaluate_correction

    protocol = _json(protocol_path)
    paths = directories(protocol)
    receipt = _json(paths["report"] / "training.json")
    _verified(protocol_path, receipt["protocol_sha256"])
    _verified(receipt["prediction_path"], receipt["prediction_sha256"])
    execution = _json(paths["input"] / "execution.json")
    for path, digest in execution["code_sha256"].items():
        _verified(path, digest)
    result = evaluate_correction(pd.read_parquet(receipt["prediction_path"]), protocol)
    result["protocol_sha256"] = sha256(protocol_path)
    result["prediction_sha256"] = receipt["prediction_sha256"]
    write_json(paths["report"] / "results.json", result)
    return result


def predict(artifact: Path, inputs: Path) -> dict:
    """Exercise the learned connector; callers supply matched real base forecasts."""
    from src.models.a0_sports_correction import AnchoredSportsCorrection

    selected = _json(artifact)
    _verified(selected["model_path"], selected["sha256"])
    state = _json(selected["model_path"])
    if state["model"].get("schema") in ("c0_raw_specialist_probability_connector_v1", "c0_nonnegative_logit_slope_v1"):
        from src.models.a0_probability_connector import predict as predict_c0
        return predict_c0(artifact, inputs)
    model = AnchoredSportsCorrection.from_dict(state["model"])
    rows = _json(inputs)
    date = pd.Series(rows["date"], dtype="string")
    start, end = selected["valid_target_interval"]
    if not (date.ge(start) & date.lt(end)).all():
        raise ValueError("Target date is outside the saved complete-recipe interval")
    p = model.predict_proba(rows["p_a0"], rows["p_specialist"], rows["best_of"])
    if len(date) != len(p):
        raise ValueError("Target dates and probability inputs are misaligned")
    return {"origin": selected["origin"], "family": selected["family"], "p": p.tolist(),
            "research_only": True, "input_contract": "same-side SERIES probabilities from the pinned upstream recipes"}
