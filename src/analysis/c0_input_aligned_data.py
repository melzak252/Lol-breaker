"""Immutable, exposed-development C0 inputs from strictly prior exact-context rosters.

Archive namespace activation and real native computation occur only in a fresh
worker process. Neither preparation nor roster resolution authenticates sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

from src.analysis import ev_asof_replay as replay
from src.analysis.c0_betting_development import build_panel, _load, _source_path, _resolve_research_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "c0_input_aligned_v1"
MODEL_IDENTITY = "Causal-C0:c0-native-2026-w32-e12-v1"
MODES = ("full", "no_w20", "no_organization")
ORDER_FIELDS = tuple(f"{prefix}_player_order_{side}" for prefix in
    ("tensor", "raw_tensor", "temporal_tensor") for side in ("a", "b"))
TENSOR_SHAPES = {"raw.players": (2,5,21), "raw.core": (2,5), "raw.team": (2,11),
    "raw.w20": (2,10), "raw.gates": (2,2), "hist.history": (2,5,16,33),
    "hist.mask": (2,5,16), "hist.champions": (2,5,16)}
DTYPES = {key: np.dtype("float32" if key.startswith("raw.") else
    {"hist.history":"float16", "hist.mask":"uint8", "hist.champions":"uint16"}[key])
    for key in TENSOR_SHAPES}
OUTPUTS = {"04_feature": ("native_snapshots.npz", "feature_rows.parquet", "history_membership.json"),
    "05_model_input": ("prior_panel.parquet", "actual_lineup_panel.parquet",
        "fresh300s_prior_panel.parquet", "native_control_panel.parquet", "preparation_receipt.json")}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(replay._json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _apply_forecast_eligibility(panel, target_native_release_days):
    """Forbid every recipe once either retained target-release clock is known.

    Daily native history is released strictly before the UTC decision day.
    Labels and corrected native series are independent evidence; neither a
    target's outcome nor market/roster support participates in this decision.
    """
    releases = {}
    for identifier, value in target_native_release_days.items():
        identifier = replay._normalize_id(identifier, "native target series ID")
        if identifier is None:
            raise ValueError("Missing native target series ID")
        releases[identifier] = replay._as_day(value, "native target effective release day")
    eligibility, absences, native_days, label_days = [], [], [], []
    for _, row in panel.iterrows():
        cutoff = replay._timestamp(row["scenario_decision_at"], "scenario_decision_at")
        if pd.isna(cutoff):
            raise ValueError("Missing nominal decision clock")
        identifier = replay._normalize_id(row.get("golgg_match_id"), "target match ID")
        native_day = releases.get(identifier)
        label_value = row.get("effective_label_release_day")
        if label_value is None or pd.isna(label_value):
            label_value = row.get("label_release_day")
        label_day = None if label_value is None or pd.isna(label_value) else replay._as_day(
            label_value, "target effective label release day")
        forbidden = any(day is not None and day < cutoff.date() for day in (native_day, label_day))
        eligibility.append(not forbidden)
        absences.append("target_already_released_before_cutoff" if forbidden else None)
        native_days.append(native_day.isoformat() if native_day is not None else None)
        label_days.append(label_day.isoformat() if label_day is not None else None)
    result = panel.copy()
    result["forecast_eligible"] = pd.Series(eligibility, index=result.index, dtype=bool)
    result["forecast_absence"] = absences
    result["target_native_effective_release_day"] = native_days
    result["target_effective_label_release_day"] = label_days
    if "native_absence" not in result:
        result["native_absence"] = None
    if "native_absence_reasons" not in result:
        result["native_absence_reasons"] = [[] for _ in range(len(result))]
    forbidden = ~result.forecast_eligible
    result.loc[forbidden, "native_admitted"] = False
    for name in ["p_C0", *(name for name in result if name.startswith(
            ("p_full_", "p_neural_", "p_raw_neural_")))]:
        result.loc[forbidden, name] = np.nan
    result.loc[forbidden, "native_absence"] = result.loc[forbidden, "forecast_absence"]
    result["native_absence_reasons"] = [
        [absence] if absence is not None else reasons
        for absence, reasons in zip(result.forecast_absence, result.native_absence_reasons)]
    return result


def build_prior_targets(panel, roster_events, *, target_native_release_days=None):
    """Keep every opportunity; select latest state before validating its roster.

    An invalid latest exact-context state blocks an older valid roster. Source
    slot order is retained in evidence; the native request sorts canonical IDs.
    Actual target rosters and probabilities are never a prior-input fallback.
    """
    required = {"row_id", "team1_id", "team2_id", "scenario_decision_at"}
    if not required.issubset(panel.columns) or panel.row_id.duplicated().any():
        raise ValueError("Missing or duplicate opportunity identity")
    indexed = defaultdict(list)
    events = list(event for group in roster_events.values() for event in group) if isinstance(roster_events, Mapping) else list(roster_events)
    event_releases = {}
    target_ids = {replay._normalize_id(value, "target match ID") for value in panel["golgg_match_id"]}
    for event in events:
        identifier = replay._normalize_id(event.get("source_match_id"), "source match ID")
        if identifier is not None and identifier in target_ids:
            day = replay._as_day(event["effective_release_day"], "effective release day")
            if identifier in event_releases and event_releases[identifier] != day:
                raise ValueError(f"Inconsistent roster/history release for series {identifier}")
            event_releases[identifier] = day
    releases = dict(event_releases) if target_native_release_days is None else dict(target_native_release_days)
    panel = _apply_forecast_eligibility(panel, releases)
    for event in events:
        team = replay._normalize_id(event.get("team_id"), "roster event team ID")
        if team is not None:
            indexed[team].append(event)
    records = []
    for _, original in panel.iterrows():
        record = original.to_dict()
        cutoff = replay._timestamp(record["scenario_decision_at"], "scenario_decision_at")
        if pd.isna(cutoff):
            raise ValueError("Missing nominal decision clock")
        context = replay._target_competition_context(record, str(record.get("golgg_match_id", record["row_id"])))
        record["competition_context"] = context
        reasons = [record["forecast_absence"]] if not record["forecast_eligible"] else []
        for side, field in (("a", "team1_id"), ("b", "team2_id")):
            team = replay._normalize_id(record[field], field)
            record[field] = team
            event = replay._select_prior_event(indexed.get(team, ()), team, context, cutoff.date()) if record["forecast_eligible"] else None
            source = replay._json_safe(dict(event)) if event is not None else None
            roster = [None]*5
            if event is None and record["forecast_eligible"]:
                reasons.append(f"prior_roster_{side}_missing")
            elif event is not None:
                source["source_match_id"] = replay._normalize_id(event.get("source_match_id"), "source match ID")
                source["source_event_id"] = "raw-map-roster:"+_canonical_sha(source)
                try:
                    roster = replay._event_roster(event)
                except ValueError:
                    reasons.append(f"prior_roster_{side}_invalid")
                else:
                    if not replay._valid_roster(roster):
                        reasons.append(f"prior_roster_{side}_invalid")
                    if source["source_match_id"] is None:
                        reasons.append(f"prior_roster_{side}_source_missing")
            record[f"prior_source_{side}"] = source
            record[f"prior_roster_{side}"] = roster
            record[f"roster_{side}"] = sorted(roster) if replay._valid_roster(roster) else roster
        overlap = sorted(set(player for player in record["prior_roster_a"] if player is not None)
            & set(player for player in record["prior_roster_b"] if player is not None))
        if overlap:
            reasons.append("prior_rosters_overlap")
        if record["team1_id"] is None or record["team2_id"] is None or record["team1_id"] == record["team2_id"]:
            reasons.append("prior_team_identity_invalid")
        record.update(prior_roster_available=not reasons, overlapping_prior_player_ids=overlap,
            native_absence=reasons[0] if reasons else None, native_absence_reasons=reasons,
            native_admitted=False, p_C0=np.nan, generated_at_utc=None, feature_source_max_day=None,
            roster_policy="strictly_prior_day_exact_context_observed_scenario_not_official_announcement")
        for name in panel.columns:
            if name.startswith(("p_full_", "p_neural_", "p_raw_neural_")):
                record[name] = np.nan
        records.append(record)
    result = pd.DataFrame(records, columns=list(dict.fromkeys([*panel.columns,
        *(key for row in records for key in row)])))
    result.attrs.update(panel.attrs)
    return result


def _root(value):
    return _resolve_research_root(value).resolve(strict=True)


def _verify(protocol, root):
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Wrong input-aligned protocol schema")
    run_id = protocol.get("run_id", "")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("Invalid run ID")
    for key, reference in protocol.get("sources", {}).items():
        path = _source_path(reference, root)
        if _sha(path) != reference["sha256"]:
            raise ValueError(f"Changed frozen source hash: {key}: {path}")
    for path, expected in protocol.get("code_sha256", {}).items():
        if _sha(PROJECT_ROOT/path) != expected:
            raise ValueError(f"Changed frozen code hash: {path}")
    if not {"raw_history", "release_map", "control", "market", "quotes", "availability"}.issubset(protocol.get("sources", {})):
        raise ValueError("Missing native/source closure")
    if not protocol.get("code_sha256"):
        raise ValueError("Missing frozen code hash closure")
    if protocol.get("native_identity") != MODEL_IDENTITY or protocol.get("mode") != "full":
        raise ValueError("Wrong frozen native identity/mode")


def _paths(run_id):
    return {stage: PROJECT_ROOT/"data"/stage/"c0_input_aligned"/run_id for stage in OUTPUTS}


def _preflight(paths):
    for stage, names in OUTPUTS.items():
        for name in names:
            path = paths[stage]/name
            if path.exists():
                raise FileExistsError(f"Preparation output already exists: {path}")


def _atomic(path, writer):
    """Publish complete bytes atomically, without replacing any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".c0-input-", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_json(path, value):
    _atomic(path, lambda temporary: temporary.write_text(_json(value)+"\n", encoding="utf-8"))


def prepare_input_aligned(protocol: Mapping, *, research_root=None, smoke_limit=None):
    """Verify immutable inputs, then run genuine native preparation in isolation."""
    root = _root(research_root)
    _verify(protocol, root)
    if smoke_limit is not None and (type(smoke_limit) is not int or smoke_limit <= 0):
        raise ValueError("smoke_limit must be a positive integer")
    run_id = protocol["run_id"] if smoke_limit is None else f"{protocol['run_id']}_smoke_{smoke_limit}"
    paths = _paths(run_id)
    _preflight(paths)
    with tempfile.TemporaryDirectory(prefix="c0-input-aligned-") as temporary:
        protocol_path = Path(temporary)/"protocol.json"
        protocol_path.write_text(_json(protocol), encoding="utf-8")
        command = [sys.executable, "-m", "src.analysis.c0_input_aligned_data", "--worker",
            str(protocol_path), "--research-root", str(root)]
        if smoke_limit is not None:
            command.extend(["--smoke-limit", str(smoke_limit)])
        environment = dict(os.environ)
        environment.update(BETTING_APP_C0_WORKER="1", C0_RESEARCH_ROOT=str(root))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError("Native input-aligned preparation failed in isolated process:\n"
                + completed.stdout + "\n" + completed.stderr)
    receipt = json.loads((paths["05_model_input"]/"preparation_receipt.json").read_text())
    if receipt["protocol_sha256"] != _canonical_sha(protocol):
        raise ValueError("Worker preparation protocol binding differs")
    return receipt


def _attach_start(panel, quotes):
    selected = quotes.loc[(quotes.horizon == "T-1h") & quotes.strict_prior_day]
    clocks = {}
    for fixture, group in selected.groupby("fixture_id", sort=False):
        starts = {replay._timestamp(value, "start_cutoff_at").isoformat() for value in group.start_cutoff_at}
        if len(starts) != 1:
            raise ValueError("Conflicting historical start clock")
        clocks[str(fixture)] = next(iter(starts))
    result = panel.copy()
    result["start_at"] = result.fixture_id.astype(str).map(clocks)
    if result.start_at.isna().any():
        raise ValueError("Missing bound scenario start clock")
    return result


def _smoke_indices(prior, limit):
    """Input-only stratification; labels/probabilities never influence selection."""
    ordered = prior.sort_values(["scenario_decision_at", "row_id"])
    supported = ordered.loc[ordered.prior_roster_available]
    chosen = []
    for phase in ("TRAIN", "CAL", "ASSESS"):
        group = supported.loc[supported.phase == phase]
        if not group.empty and len(chosen) < limit:
            chosen.append(group.index[0])
    for bo in (1, 3, 5):
        if any(prior.loc[index, "best_of"] == bo for index in chosen):
            continue
        group = supported.loc[supported.best_of == bo]
        if not group.empty and len(chosen) < limit:
            chosen.append(group.index[0])
    unavailable = ordered.loc[~ordered.prior_roster_available]
    if not unavailable.empty and len(chosen) < limit:
        chosen.append(unavailable.index[0])
    for index in ordered.index:
        if len(chosen) >= limit:
            break
        if index not in chosen:
            chosen.append(index)
    return [index for index in ordered.index if index in chosen]


def _counts(panel):
    return {phase: {"denominator": len(group), "native_admitted": int(group.native_admitted.sum()),
        "market_eligible": int(group.market_eligible.sum()), "financial_eligible": int(group.financial_eligible.sum()),
        "fit_eligible": int(group.fit_eligible.sum()),
        "forecast_eligible": int(group.forecast_eligible.sum()),
        "forecast_absence": dict(Counter(group.forecast_absence.dropna())),
        "common_fit": int((group.forecast_eligible & group.native_admitted & group.market_eligible & group.fit_eligible).sum()),
        "native_absence": dict(Counter(group.native_absence.dropna())) if "native_absence" in group else {}}
        for phase, group in panel.groupby("phase", sort=True)}


def _prepare_worker(protocol, root, smoke_limit):
    if os.environ.get("BETTING_APP_C0_WORKER") != "1":
        raise RuntimeError("Preparation worker requires fresh isolated native namespace")
    started = _now()
    _verify(protocol, root)
    run_id = protocol["run_id"] if smoke_limit is None else f"{protocol['run_id']}_smoke_{smoke_limit}"
    paths = _paths(run_id)
    _preflight(paths)
    control, market, quotes = _load(protocol, root)
    actual, archival_coverage = build_panel(control, market, quotes, max_age_seconds=86400)
    fresh, fresh_coverage = build_panel(control, market, quotes, max_age_seconds=300)
    actual = _attach_start(actual, quotes)
    fresh = _attach_start(fresh, quotes)
    if len(control) != 3103 or len(actual) != 1668 or actual.groupby("phase").size().to_dict() != {"TRAIN":775,"CAL":370,"ASSESS":523}:
        raise ValueError("Frozen provider/native/phase denominator changed")
    from betting_app.core.models.native_c0 import NativeC0Adapter, encode_snapshot
    adapter = NativeC0Adapter()
    identity = adapter.load()
    raw_path, raw_sha, release_sha, prepared = adapter._load_history()
    if (raw_path.resolve() != _source_path(protocol["sources"]["raw_history"], root)
            or raw_sha != protocol["sources"]["raw_history"]["sha256"]
            or Path(prepared["release_path"]).resolve() != _source_path(protocol["sources"]["release_map"], root)
            or release_sha != protocol["sources"]["release_map"]["sha256"]):
        raise ValueError("Actual native history/release differs from frozen study sources")
    if f"{identity['name']}:{identity['version']}" != MODEL_IDENTITY:
        raise ValueError("Native adapter identity differs")
    # Corrected series release is independent of provider labels and roster
    # selection. Check it before any target query or temporal-state advance.
    target_native_release_days = {str(item.identifier): item.games[-1].day
        for item in prepared["series"]}
    actual["actual_saved_p_C0"] = actual.p_C0
    actual = _apply_forecast_eligibility(actual, target_native_release_days)
    fresh = _apply_forecast_eligibility(fresh, target_native_release_days)
    prior = build_prior_targets(actual, prepared["roster_events"],
        target_native_release_days=target_native_release_days)
    if smoke_limit is not None:
        indices = _smoke_indices(prior, smoke_limit)
        prior = prior.loc[indices].reset_index(drop=True)
        actual = actual.set_index("row_id", drop=False).loc[prior.row_id].reset_index(drop=True)
        fresh = fresh.set_index("row_id", drop=False).loc[prior.row_id].reset_index(drop=True)
    arrays = {key: np.empty((int((prior.forecast_eligible & prior.prior_roster_available).sum()),*shape), dtype=DTYPES[key])
        for key, shape in TENSOR_SHAPES.items()}
    feature_records = []
    available_indices = []
    queries = []
    history_series = [{"source_match_id": str(item.identifier),
        "effective_release_day": item.games[-1].day.isoformat(),
        "source_series_order": replay._json_safe(item.order)} for item in prepared["series"]]
    membership_by_cutoff = {}
    for index in prior.sort_values(["scenario_decision_at","row_id"]).index:
        row = prior.loc[index]
        if not row.forecast_eligible or not row.prior_roster_available:
            continue
        request = adapter._validate_request({"team1_id": row.team1_id,"team2_id":row.team2_id,
            "roster_a":row.roster_a,"roster_b":row.roster_b,"best_of":int(row.best_of),
            "decision_at":row.scenario_decision_at,"start_at":row.start_at,
            "competition_context":row.competition_context,"mode":"full"})
        state = adapter._advance_history(request["decision_at"].date(), prepared)
        marker = replay._logical_state_marker(state)
        target, raw, hist, provenance = adapter._raw_snapshot(request)
        if replay._logical_state_marker(state) != marker:
            raise ValueError("Feature query mutated chronological native state")
        prior.at[index,"feature_source_max_day"] = target["feature_source_max_day"]
        position = len(feature_records)
        tensor_hashes = {}
        for key, array in (("raw."+name,value) for name,value in raw.items()):
            arrays[key][position] = array
        for name, value in hist.items():
            arrays["hist."+name][position] = value
        for key in TENSOR_SHAPES:
            tensor_hashes[key] = hashlib.sha256(arrays[key][position].tobytes()).hexdigest()
        cutoff = target["cutoff_day"]
        if cutoff not in membership_by_cutoff:
            membership_by_cutoff[cutoff] = {"history_series_consumed":prepared["cursor"],
                "source_membership_sha256":_canonical_sha(history_series[:prepared["cursor"]]),
                "feature_source_max_day":target["feature_source_max_day"]}
        feature_records.append({"row_id":row.row_id,"request":_json(request),"target":_json(target),
            "prior_source_a":_json(row.prior_source_a),"prior_source_b":_json(row.prior_source_b),
            "forecast_eligible":bool(row.forecast_eligible),"forecast_absence":row.forecast_absence,
            "target_native_effective_release_day":row.target_native_effective_release_day,
            "target_effective_label_release_day":row.target_effective_label_release_day,
            "feature_source_max_day":target["feature_source_max_day"],"cutoff_day":cutoff,
            "history_membership_id":cutoff,"history_series_consumed":prepared["cursor"],
            "tensor_sha256":_json(tensor_hashes),"native_history_provenance":_json(provenance),
            **{key:target[key] for key in ORDER_FIELDS}})
        available_indices.append(index)
        queries.append((request,target,provenance))
    npz_ids = [feature["row_id"] for feature in feature_records]
    parity = []
    before = {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in arrays.items()}
    if queries:
        from src.models import a0_data_focus
        best_of = np.asarray([request["best_of"] for request,_,_ in queries],dtype=np.int64)
        predictions = a0_data_focus.predict_models(adapter._backend,adapter._shared,adapter._models,
            adapter._calibration,{key[4:]:value for key,value in arrays.items() if key.startswith("raw.")},
            {key[5:]:value for key,value in arrays.items() if key.startswith("hist.")},best_of)
        generated = _now()
        for component in ("full","neural","raw_neural"):
            values = np.asarray(predictions[component],dtype=np.float64)
            if values.shape != (len(queries),len(MODES)) or not np.isfinite(values).all() or ((values<=0)|(values>=1)).any():
                raise ValueError("Invalid genuine native all-mode probabilities")
            for mode_index, mode in enumerate(MODES):
                column = f"p_{component}_{mode}"
                prior[column] = np.nan
                prior.loc[available_indices,column] = values[:,mode_index]
        prior.loc[available_indices,"p_C0"] = np.asarray(predictions["full"])[:,0]
        prior.loc[available_indices,"native_admitted"] = True
        prior.loc[available_indices,"generated_at_utc"] = generated
        for position, feature in enumerate(feature_records):
            feature["native_generated_at"] = generated
            for component in ("full","neural","raw_neural"):
                for mode_index, mode in enumerate(MODES):
                    feature[f"p_{component}_{mode}"] = float(predictions[component][position,mode_index])
        # A real ordinary-adapter control per observed BO tests unchanged all-mode math.
        for bo in sorted(set(best_of.tolist())):
            position = next(i for i,value in enumerate(best_of) if value == bo)
            request,target,provenance = queries[position]
            raw_one = {key[4:]:value[position] for key,value in arrays.items() if key.startswith("raw.")}
            hist_one = {key[5:]:value[position] for key,value in arrays.items() if key.startswith("hist.")}
            encoded = encode_snapshot({"target":target,"raw":raw_one,"hist":hist_one})
            ordinary = adapter.predict(json.loads(_json(request)), snapshot_value=encoded)
            errors = {mode:abs(ordinary["diagnostics"]["mode_probabilities"][mode]-float(predictions["full"][position,j]))
                for j,mode in enumerate(MODES)}
            if max(errors.values()) > 1e-6:
                raise ValueError("Batch native probabilities differ from ordinary adapter")
            swapped_raw = {key:value[::-1].copy()[None,...] for key,value in raw_one.items()}
            swapped_hist = {key:value[::-1].copy()[None,...] for key,value in hist_one.items()}
            swapped = a0_data_focus.predict_models(adapter._backend,adapter._shared,adapter._models,
                adapter._calibration,swapped_raw,swapped_hist,np.asarray([bo],dtype=np.int64))
            side_errors = {component:float(np.max(np.abs(np.asarray(predictions[component])[position]
                +np.asarray(swapped[component])[0]-1.0))) for component in predictions}
            if max(side_errors.values()) > 2e-6:
                raise ValueError("Native side-symmetry control failed")
            cold = a0_data_focus.predict_models(adapter._backend,adapter._shared,adapter._models,
                adapter._calibration,{key:np.zeros_like(value) for key,value in swapped_raw.items()},
                {key:np.zeros_like(value) for key,value in swapped_hist.items()},np.asarray([bo],dtype=np.int64))
            cold_error = max(float(np.max(np.abs(value-.5))) for value in cold.values())
            if cold_error > 2e-6:
                raise ValueError("Native neutral software control failed")
            parity.append({"row_id":npz_ids[position],"best_of":bo,"absolute_errors":errors,
                "public_api":"NativeC0Adapter.predict(snapshot_value=encoded_eight_arrays)",
                "side_symmetry_errors":side_errors,"neutral_max_error":cold_error,
                "software_controls_not_forecast_fallback":True})
    if {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in arrays.items()} != before:
        raise ValueError("Native batch inference mutated staged inputs")
    fresh = fresh.set_index("row_id",drop=False).loc[prior.row_id].reset_index(drop=True)
    input_columns = [name for name in prior.columns if name.startswith(("prior_","p_full_","p_neural_","p_raw_neural_"))
        or name in {"p_C0","native_admitted","native_absence","native_absence_reasons","roster_a","roster_b",
                    "competition_context","overlapping_prior_player_ids","generated_at_utc","roster_policy","feature_source_max_day",
                    "forecast_eligible","forecast_absence","target_native_effective_release_day","target_effective_label_release_day"}]
    for name in input_columns:
        fresh[name] = prior[name].tolist()
    # Fresh quote support changes actions only, never label maturity or cash denominator.
    for name in ("row_id","financial_eligible","fit_eligible","y","phase","scenario_decision_at"):
        if not prior[name].equals(fresh[name]):
            raise ValueError("Freshness diagnostic changed frozen opportunity/label membership")
    feature_columns = ["row_id","request","target","prior_source_a","prior_source_b",
        "feature_source_max_day","cutoff_day","history_membership_id","history_series_consumed",
        "tensor_sha256","native_history_provenance",*ORDER_FIELDS,"native_generated_at",
        "forecast_eligible","forecast_absence","target_native_effective_release_day","target_effective_label_release_day",
        *(f"p_{component}_{mode}" for component in ("full","neural","raw_neural") for mode in MODES)]
    features = pd.DataFrame(feature_records,columns=feature_columns)
    memberships = {"schema":"c0_input_aligned_history_membership_v1","raw_history_sha256":raw_sha,
        "release_map_sha256":release_sha,"series":history_series,"cutoffs":membership_by_cutoff,
        "history_audit":prepared["history_audit"],"release_audit":prepared["release_audit"]}
    _preflight(paths)
    _atomic(paths["04_feature"]/"native_snapshots.npz", lambda path:
        _write_npz(path,{"row_id":np.asarray(npz_ids,dtype=str),**arrays}))
    _atomic(paths["04_feature"]/"feature_rows.parquet", lambda path:features.to_parquet(path,index=False))
    _save_json(paths["04_feature"]/"history_membership.json",memberships)
    for name, frame in (("prior_panel",prior),("actual_lineup_panel",actual),
            ("fresh300s_prior_panel",fresh),("native_control_panel",control)):
        _atomic(paths["05_model_input"]/(name+".parquet"),lambda path,frame=frame:frame.to_parquet(path,index=False))
    outputs = {}
    for stage,names in OUTPUTS.items():
        for name in names:
            if name == "preparation_receipt.json":
                continue
            path = paths[stage]/name
            outputs[path.relative_to(PROJECT_ROOT).as_posix()] = {"sha256":_sha(path),"bytes":path.stat().st_size}
    receipt = {"schema":"c0_input_aligned_preparation_v1","run_id":run_id,
        "bound_protocol_run_id":protocol["run_id"],"protocol_sha256":_canonical_sha(protocol),
        "evidence_class":"EXPOSED_HISTORICAL_DEVELOPMENT_ONLY" if smoke_limit is None else "SOFTWARE_SMOKE_NOT_FORMAL",
        "formal_fit_eligible":smoke_limit is None,"real_head_fit_count":0,"base_fit_count":0,
        "preparation_started_at_actual_utc":started,"preparation_completed_at_actual_utc":_now(),
        "input_sha256":{key:reference["sha256"] for key,reference in protocol["sources"].items()},
        "code_sha256":protocol["code_sha256"],"outputs":outputs,"native_identity":identity,
        "mode":"full","all_modes":list(MODES),"counts":_counts(prior),"fresh300s_counts":_counts(fresh),
        "actual_lineup_counts":_counts(actual),
        "forecast_eligible_count":int(prior.forecast_eligible.sum()),
        "forecast_absence_counts":dict(Counter(prior.forecast_absence.dropna())),
        "native_control_count":len(control),"provider_denominator":len(prior),
        "native_available_count":len(features),"native_absence_counts":dict(Counter(prior.native_absence.dropna())),
        "roster_context_counts":dict(Counter(prior.competition_context.dropna())),
        "roster_source_event_counts":{side:len({source["source_event_id"] for source in
            prior[f"prior_source_{side}"] if source is not None}) for side in ("a","b")},
        "available_best_of_counts":dict(Counter(int(value) for value in prior.loc[prior.native_admitted,"best_of"])),
        "native_history_source":{"raw_history_path":str(raw_path),"raw_history_sha256":raw_sha,
            "release_map_path":prepared["release_path"],"release_map_sha256":release_sha,
            "history_series_total":len(prepared["series"]),"history_series_consumed":prepared["cursor"]},
        "archival_coverage":archival_coverage,"fresh300s_coverage":fresh_coverage,
        "array_inventory":{key:{"shape":list(value.shape),"dtype":str(value.dtype),"sha256":before[key]}
            for key,value in arrays.items()},"npz_row_ids":npz_ids,
        "full_mode_parity_evidence":parity,"input_arrays_unmodified":True,
        "history_load_count":1,"history_state_count":int(prepared["state"] is not None),
        "source_membership_policy":"strict_effective_release_day_before_UTC_decision_day",
        "forecast_eligibility_policy":"both_effective_target_label_and_corrected_native_release_not_before_UTC_decision_day",
        "smoke_selection_policy":None if smoke_limit is None else "input_only_prior_supported_phase_and_BO_stratification_plus_unavailable",
        "smoke_selected_row_ids":None if smoke_limit is None else prior.row_id.tolist(),
        "limitations":["Retrospective observed roster/context scenario; not official announcement or authentic G0.",
            "24h historical quote scope and provider source clocks do not certify current all-state300s admission.",
            "Actual lineup diagnostic controls are saved historical predictions, never prior-input fallback.",
            "All historical phases are development-exposed; actual computation clocks are retained."]}
    _save_json(paths["05_model_input"]/"preparation_receipt.json",receipt)
    return receipt


def _write_npz(path,payload):
    with path.open("wb") as stream:
        np.savez_compressed(stream,**payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--research-root",type=Path,required=True)
    parser.add_argument("--smoke-limit",type=int)
    arguments = parser.parse_args()
    _prepare_worker(json.loads(arguments.worker.read_text()),arguments.research_root,arguments.smoke_limit)
