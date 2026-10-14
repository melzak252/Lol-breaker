#!/usr/bin/env python3
"""Inventory local fresh-data evidence and fixed-policy bounded-return diagnostics.

No fitting, network, database access, executable ROI claim or promotion. Run with
.venv/bin/python and a new --output-dir. --candidates accepts a JSON object with
"candidates": [{"name": "raw_sports", "path": "forecasts.csv", "column":
"raw_sports", "phase": "open", "bookmaker": "STS", "years": [2026]}]. Paths
are relative to the config directory. Optional "columns" maps canonical metadata
names to source names; optional "artifact_paths" lists frozen fit/model files.
One entry per name/phase; no label-based choice of columns, phase, year or book.
--observations optionally inventories the prerequisite schema recorded in protocol.
Use --frozen-protocol prior-run/protocol.json to check observations collected after
that prior lock. Without it, rows are checked against this run's fresh lock and
cannot retroactively become prospective. The prior record remains unqualified.
Local timestamps/hashes are checked assertions, not independent certification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.corrected_history_evaluation import frozen_hashes, sha256
from scripts.experiment_exp081_market_calibration import checked_probabilities, code_hashes, selected_uncertainty, write_json
from scripts.experiment_betting_calibration import MIN_EV, TAX, policy
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS

CANDIDATES = ("raw_sports", "benter", "conditional", "conditional_payoff", "outcome_student", "market_student", "benter_student")
BASE = ("golgg_match_id", "date", "y_true", "phase", "year", "odds_a", "odds_b")
TIMES = ("feature_history_max_at", "source_available_at", "data_cutoff_at", "captured_at", "predicted_at", "quote_at", "quote_observed_at", "decision_at", "match_start_at", "match_end_at", "outcome_available_at", "settled_at", "outcome_observed_at")
EVIDENCE = ("model", "features", "forecast", "quote", "match_end", "outcome")
OBSERVATION_FIELDS = ("golgg_match_id", "candidate", "phase", "bookmaker", "bookmaker_a", "bookmaker_b", "team_a_id", "team_b_id", "market", "feature_version", "protocol_sha256", "p_a", "odds_a", "odds_b", "y_true", *TIMES, *(field for kind in EVIDENCE for field in (kind + "_path", kind + "_sha256")))
KNOWN_REPORTS = (
    "data/artifacts/exp089-prospective-confirmation/run01/data_readiness.json",
    "data/artifacts/canonical79-prospective-v1/run01/local_tournament_readiness.json",
    "data/artifacts/canonical79-prospective-v1/run01/raw_history_readiness.json",
    "data/artifacts/model-redesign-v1/source-v2/manifest.json",
    "data/oddspapi_lol_2026_model_audit/summary.json",
)
QUOTE_PATH = "data/oddspapi_lol_2026_model_audit/selected_pre_match_quotes.csv"
SOURCE_CONTRACTS = ("scripts/confirm_oddsfree_student.py", "scripts/export_prospective_features.py", "scripts/prospective_sports_features.py")
BLOCKERS = (
    "No independently verified pre-outcome policy/model/forecast commitment: local hashes and timestamps are not trusted publication evidence.",
    "No independently certified feature-source/roster availability or compatibility with the frozen candidate model.",
    "No independently certified same-book executable quote, exact identity/orientation, series market and settlement/end-time provenance chain.",
    "Untouched cohort membership and absence of prior outcome inspection are not established by a later calendar date or user-supplied flags.",
    "IID fixed-policy selected payoffs are an explicit diagnostic assumption, not established for temporally dependent esports observations.",
)


def local_path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def csv_frame(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"missing or duplicate CSV header: {path}")
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("timestamp must carry explicit timezone")
    return stamp.tz_convert("UTC")


def inventory():
    """Bounded discovery: metadata/headers only, never load model or outcome values."""
    records, inputs = [], {}
    for name in KNOWN_REPORTS:
        path = ROOT / name
        record = {"path": name, "exists": path.is_file()}
        if path.is_file():
            inputs[str(path)] = sha256(path)
            value = json.loads(path.read_text())
            record["sha256"] = inputs[str(path)]
            for key in ("decision", "status", "observed_at", "source_available_at", "extracted_at", "availability_caveat", "comparison_classification", "comparison_limitation", "temporal_quote_policy", "selected_pre_match_quotes", "real_forecasts_captured", "eligible_for_frozen_confirmation"):
                if key in value:
                    record[key] = value[key]
            if "audit" in value:
                record["prior_audit_summary"] = value["audit"].get("summary")
                record["prior_audit_files"] = value["audit"].get("files", [])
        records.append(record)
    quotes = ROOT / QUOTE_PATH
    quote_record = {"path": QUOTE_PATH, "exists": quotes.is_file(), "eligible_live_rows": 0}
    if quotes.is_file():
        inputs[str(quotes)] = sha256(quotes)
        with quotes.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            count, valid_before, books, first, last = 0, 0, {}, None, None
            for row in reader:
                count += 1
                book = row.get("bookmaker", "")
                books[book] = books.get(book, 0) + 1
                try:
                    start, quote = timestamp(row["match_start_at"]), timestamp(row["quote_at"])
                    valid_before += int(quote < start)
                    first = start if first is None else min(first, start)
                    last = start if last is None else max(last, start)
                except (ValueError, KeyError, TypeError):
                    pass
        quote_record.update(sha256=inputs[str(quotes)], rows=count, by_bookmaker=books,
                            quote_strictly_before_start=valid_before, fields=fields,
                            first_start=None if first is None else first.isoformat(),
                            last_start=None if last is None else last.isoformat(),
                            missing_prerequisite_fields=sorted(set(OBSERVATION_FIELDS) - set(fields)),
                            classification="historical_quote_observations_not_fresh_forecasts",
                            blockers=["Retrospective model_prob_a_proxy and already inspected winner_a are not frozen decision-time forecasts.", "Quote timestamp does not establish prediction/feature availability, untouched outcomes, end time or settlement."])
    records.append(quote_record)
    # Narrow known research artifact tree; only small manifest/protocol metadata.
    bundles = []
    artifact_root = ROOT / "data/artifacts"
    if artifact_root.is_dir():
        for directory in sorted(artifact_root.iterdir()):
            if not directory.is_dir() or not any(word in directory.name for word in ("prospective", "confirmation", "feature-export")):
                continue
            for path in sorted(directory.rglob("*.json")):
                if path.name not in {"manifest.json", "protocol.json", "export.json"}:
                    continue
                inputs[str(path)] = sha256(path)
                value = json.loads(path.read_text())
                bundles.append({"path": str(path.relative_to(ROOT)), "sha256": inputs[str(path)],
                                **{key: value[key] for key in ("kind", "version", "locked_at", "predicted_at", "source_available_at", "row_count", "status", "eligible_for_frozen_confirmation", "limitations") if key in value}})
    forecast_bundles = [row for row in bundles if row.get("kind") == "prospective_local_series_forecasts"]
    return {"scope": "bounded local metadata inventory; no DB, network, mtime inference or new forecast capture", "records": records,
            "prospective_bundles": bundles, "local_forecast_bundle_count": len(forecast_bundles),
            "eligible_live_rows": 0, "freshness_warning": "Neither later-than-2025 nor later-than-a-prior-cutoff means uninspected or point-in-time certified."}, inputs


def load_candidates(config):
    if config is None:
        return [], {}
    value = json.loads(config.read_text())
    if not isinstance(value, dict) or set(value) != {"candidates"} or not isinstance(value["candidates"], list):
        raise ValueError("candidates config requires exactly a candidates list")
    entries, seen, inputs = [], set(), {str(config): sha256(config)}
    for record in value["candidates"]:
        allowed = {"name", "path", "column", "phase", "bookmaker", "columns", "years", "artifact_paths"}
        if not isinstance(record, dict) or set(record) - allowed or not {"name", "path", "column", "phase", "bookmaker"} <= set(record):
            raise ValueError("invalid candidate specification")
        name, phase = record["name"], record["phase"]
        if name not in CANDIDATES or phase not in ("open", "close") or (name, phase) in seen:
            raise ValueError("unknown or duplicate candidate/phase; never pool duplicate streams")
        if not isinstance(record["bookmaker"], str) or not record["bookmaker"].strip():
            raise ValueError("one predeclared bookmaker is required")
        columns = record.get("columns", {})
        if not isinstance(columns, dict) or set(columns) - set(BASE) - {"bookmaker", "bookmaker_a", "bookmaker_b"} or any(not isinstance(v, str) for v in columns.values()) or len(set(columns.values())) != len(columns):
            raise ValueError("columns must be a one-to-one canonical-to-source metadata mapping")
        years = record.get("years", [2026])
        if not isinstance(years, list) or not years or any(type(year) is not int for year in years) or len(years) != len(set(years)):
            raise ValueError("years must be predeclared distinct integers")
        path = local_path(record["path"], config.parent)
        artifacts = [local_path(item, config.parent) for item in record.get("artifact_paths", [])]
        for item in (path, *artifacts):
            inputs[str(item)] = sha256(item)
        seen.add((name, phase))
        entries.append({**record, "path": str(path), "years": years, "artifact_paths": [str(item) for item in artifacts]})
    return entries, inputs


def diagnostic(entry, cache):
    path = Path(entry["path"])
    if path not in cache:
        cache[path] = csv_frame(path)
    source = cache[path]
    mapping = {source_name: canonical for canonical, source_name in entry.get("columns", {}).items()}
    frame = source.rename(columns=mapping)
    required = set(BASE) | {entry["column"]}
    if not frame.columns.is_unique or not required <= set(frame.columns):
        raise ValueError(f"{entry['name']}: duplicate or missing forecast columns {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame.phase.eq(entry["phase"])].copy()
    years = pd.to_numeric(frame.year, errors="raise")
    frame = frame.loc[years.isin(entry["years"])].copy()
    ids = frame.golgg_match_id
    if ids.eq("").any() or ids.str.strip().ne(ids).any() or ids.duplicated().any():
        raise ValueError("duplicate or invalid event ID within candidate/phase")
    dates = pd.to_datetime(frame.date, errors="raise")
    if dates.isna().any() or not np.array_equal(dates.dt.year, pd.to_numeric(frame.year)):
        raise ValueError("date/year mismatch")
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    for field in ("y_true", "odds_a", "odds_b", entry["column"]):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    y = frame.y_true.to_numpy(float)
    if not np.isin(y, [0., 1.]).all():
        raise ValueError("diagnostics require binary observed outcomes")
    p = checked_probabilities(frame[entry["column"]].to_numpy(float), entry["name"])
    oa, ob = frame.odds_a.to_numpy(float), frame.odds_b.to_numpy(float)
    chosen, side_a = policy(p, oa, ob)
    same_book = np.ones(len(frame), dtype=bool)
    for field in ("bookmaker", "bookmaker_a", "bookmaker_b"):
        if field in frame:
            same_book &= frame[field].eq(entry["bookmaker"]).to_numpy(dtype=bool)
    paired = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1) & same_book
    chosen &= paired
    offered = np.where(side_a, oa, ob)
    selected = chosen & (offered <= 5.)
    selected_y, selected_p = np.where(side_a, y, 1-y), np.where(side_a, p, 1-p)
    payoff = (1-TAX)*offered*selected_y - 1
    expected = (1-TAX)*offered*selected_p - 1
    n = int(selected.sum())
    mean = float(payoff[selected].mean()) if n else None
    radius = 4.4 * math.sqrt(math.log(len(CANDIDATES)/.05)/(2*n)) if n else None
    lower = max(-1., mean-radius) if n else None
    uncertainty_frame = frame.loc[paired].reset_index(drop=True)
    bootstrap = selected_uncertainty(uncertainty_frame, selected[paired], np.zeros(int(paired.sum())), payoff[paired])
    bootstrap.pop("probability_optimism", None)
    if "payoff_optimism" in bootstrap:
        bootstrap["selected_mean_payoff"] = bootstrap.pop("payoff_optimism")
    result = {"candidate": entry["name"], "phase": entry["phase"], "bookmaker": entry["bookmaker"], "years": entry["years"],
              "rows": len(frame), "finite_same_book_paired_rows": int(paired.sum()), "invalid_pair_rows": int((~paired).sum()),
              "uncapped_selected_count": int(chosen.sum()), "excluded_selected_odds_above_5": int((chosen & ~selected).sum()),
              "selected_count": n, "mean_unit_payoff": mean, "hoeffding_radius": radius, "hoeffding_lower_bound": lower,
              "status": "diagnostic_only" if n else "undefined_empty_selection", "diagnostic_lower_bound_above_zero": bool(n and lower > 0),
              "statistical_certificate_valid": False, "eligible_live_rows": 0, "promotion": False,
              "bookmaker_evidence": "CSV checked against declared book where present; config declaration alone is not provenance",
              "monthly_bootstrap_diagnostic": bootstrap}
    output = frame[list(BASE)].copy()
    output[entry["name"]] = p
    output["candidate"] = entry["name"]
    output["bookmaker"] = entry["bookmaker"]
    output["valid_pair"] = paired
    output["selected"] = selected
    output["selected_side"] = np.where(selected, np.where(side_a, "a", "b"), "abstain")
    output["selected_odds"] = np.where(selected, offered, np.nan)
    output["estimated_unit_payoff"] = np.where(selected, expected, np.nan)
    output["unit_payoff"] = np.where(selected, payoff, np.nan)
    return result, output


def observation_gate(path, lock, protocol_hash, now):
    if path is None:
        return {"rows": 0, "structurally_complete_rows": 0, "eligible_live_rows": 0, "status": "no_observations_supplied", "reason_counts": {"no_fresh_observation_bundle": 1}}, {}, []
    frame = csv_frame(path)
    inputs = {str(path): sha256(path)}
    missing = sorted(set(OBSERVATION_FIELDS) - set(frame.columns))
    if missing:
        return {"rows": len(frame), "structurally_complete_rows": 0, "eligible_live_rows": 0, "status": "missing_prerequisite_columns", "missing_columns": missing, "reason_counts": {"missing_prerequisite_columns": len(frame)}}, inputs, []
    counts, rows, valid_count = {}, [], 0
    duplicate = frame.duplicated(["golgg_match_id", "candidate", "phase"], keep=False)
    for index, row in frame.iterrows():
        reasons = []
        if duplicate.loc[index]:
            reasons.append("duplicate_candidate_phase_event")
        if row.candidate not in CANDIDATES or row.phase not in ("open", "close") or row.market != "series_match_winner":
            reasons.append("invalid_candidate_phase_or_target")
        if not row.golgg_match_id or row.team_a_id == row.team_b_id or not row.team_a_id or not row.team_b_id:
            reasons.append("invalid_exact_identity")
        if not row.bookmaker or not row.bookmaker == row.bookmaker_a == row.bookmaker_b:
            reasons.append("not_same_book_pair")
        if not row.feature_version or row.protocol_sha256 != protocol_hash:
            reasons.append("feature_version_or_protocol_binding_missing")
        try:
            p, oa, ob, y = (float(row[field]) for field in ("p_a", "odds_a", "odds_b", "y_true"))
            if not all(math.isfinite(x) for x in (p, oa, ob, y)) or not 0 <= p <= 1 or min(oa, ob) <= 1 or y not in (0, 1):
                reasons.append("invalid_probability_pair_or_outcome")
        except (ValueError, TypeError):
            reasons.append("invalid_probability_pair_or_outcome")
        try:
            t = {field: timestamp(row[field]) for field in TIMES}
            if not (lock <= t["captured_at"] <= t["predicted_at"] <= t["decision_at"] < t["match_start_at"] < t["match_end_at"] <= t["outcome_available_at"] <= t["settled_at"] <= t["outcome_observed_at"] <= now):
                reasons.append("forecast_or_outcome_timing_contract_failed")
            if not (t["feature_history_max_at"] <= t["source_available_at"] <= t["captured_at"]):
                reasons.append("feature_availability_contract_failed")
            if t["data_cutoff_at"] != t["captured_at"]:
                reasons.append("data_cutoff_differs_from_capture")
            if not (lock <= t["predicted_at"] <= t["quote_at"] <= t["quote_observed_at"] <= t["decision_at"]):
                reasons.append("quote_availability_contract_failed")
            if t["feature_history_max_at"].date() >= t["captured_at"].date():
                reasons.append("date_only_history_not_strictly_prior_day")
        except (ValueError, TypeError, OverflowError):
            reasons.append("missing_or_naive_timestamp")
        for kind in EVIDENCE:
            try:
                evidence_path = local_path(row[kind + "_path"], path.parent)
                if not row[kind + "_path"] or not evidence_path.is_file():
                    raise ValueError("missing evidence bytes")
                if str(evidence_path) not in inputs:
                    inputs[str(evidence_path)] = sha256(evidence_path)
                if inputs[str(evidence_path)] != row[kind + "_sha256"]:
                    raise ValueError("evidence digest mismatch")
            except (OSError, ValueError):
                reasons.append(kind + "_evidence_missing_or_hash_mismatch")
        valid_count += int(not reasons)
        for reason in sorted(set(reasons)):
            counts[reason] = counts.get(reason, 0) + 1
        rows.append({"golgg_match_id": row.golgg_match_id, "candidate": row.candidate, "phase": row.phase,
                     "structural_checks_passed": not reasons, "eligible_live": 0, "reasons": ";".join(sorted(set(reasons)))})
    return {"rows": len(frame), "structurally_complete_rows": valid_count, "eligible_live_rows": 0,
            "status": "blocked_independent_provenance_and_dependence_unverified", "reason_counts": counts,
            "qualification_blockers": list(BLOCKERS), "structural_checks_are_not_certification": True}, inputs, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--frozen-protocol", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (TAX, MIN_EV) != (.12, .05):
        raise RuntimeError("shared fixed policy changed")
    config = args.candidates.resolve() if args.candidates else None
    entries, inputs = load_candidates(config)
    discovery, discovered_inputs = inventory()
    inputs.update(discovered_inputs)
    for name in SOURCE_CONTRACTS:
        inputs[str(ROOT / name)] = sha256(ROOT / name)
    observation_path = args.observations.resolve() if args.observations else None
    if observation_path:
        inputs[str(observation_path)] = sha256(observation_path)
    code, frozen = code_hashes(), frozen_hashes()
    locked = datetime.now(timezone.utc)
    specs = {name: {"tax": TAX, "maximum_side_ev_strictly_above": MIN_EV, "exact_ties": "abstain",
                    "offered_selected_odds_cap": 5., "cap_applies_only_to_bounded_risk_audit": True,
                    "cap_order": "choose maximum-EV side first; abstain if selected odds exceed cap; no alternate side",
                    "same_book_pair": "finite decimal odds_a and odds_b strictly above 1", "stake": "one unit"} for name in CANDIDATES}
    candidate_artifact_hashes = {
        path: inputs[path] for entry in entries for path in entry["artifact_paths"]
    }
    observation_lock, observation_protocol_hash = timestamp(locked), None
    prior_binding = None
    if args.frozen_protocol:
        prior_path = args.frozen_protocol.resolve()
        prior_hash = sha256(prior_path)
        seal_path = prior_path.with_suffix(".sha256")
        if seal_path.read_text().strip() != prior_hash:
            raise ValueError("prior frozen protocol hash mismatch")
        prior = json.loads(prior_path.read_text())
        if (prior.get("version") != "exp081-policy-certification-audit-v1"
                or prior.get("policies") != specs or prior.get("promotion") is not False
                or prior.get("frozen_hashes") != frozen or prior.get("code_hashes") != code
                or prior.get("candidates") != entries
                or prior.get("candidate_artifact_hashes") != candidate_artifact_hashes):
            raise ValueError("prior frozen candidate specification, fit/model artifact or executable source changed")
        if any(not entry["artifact_paths"] for entry in entries):
            raise ValueError("prior protocol reuse requires explicit frozen fit/model artifacts for every supplied candidate")
        observation_lock = timestamp(prior["locked_at"])
        if observation_lock > timestamp(locked):
            raise ValueError("prior protocol lock is in the future")
        observation_protocol_hash = prior_hash
        inputs[str(prior_path)], inputs[str(seal_path)] = prior_hash, sha256(seal_path)
        prior_binding = {"path": str(prior_path), "sha256": prior_hash, "locked_at": prior["locked_at"],
                         "status": "hash_checked_local_record_not_independent_commitment"}
    protocol = {"version": "exp081-policy-certification-audit-v1", "locked_at": locked.isoformat(), "promotion": False,
                "observation_protocol": prior_binding,
                "status": "fresh_local_policy_record_not_external_timestamp_certificate", "candidates": entries, "policies": specs,
                "candidate_artifact_hashes": candidate_artifact_hashes,
                "primary_phase": "open", "close": "separate sensitivity; never pool events or pick better phase after outcomes",
                "default_years": [2026], "hyperparameter_or_policy_selection": False,
                "input_hashes": inputs.copy(), "code_hashes": code, "frozen_hashes": frozen,
                "hoeffding": {"assumption": "IID selected unit payoffs under each genuinely fixed pre-outcome policy; fixed sample/horizon, no optional stopping",
                              "range": [-1., 3.4], "delta": .05, "K": 7,
                              "lower_bound": "max(-1, sample_mean - 4.4*sqrt(log(7/0.05)/(2*n_selected)))",
                              "empty_selection": "undefined, not accepted", "family": "seven predeclared policies within primary open; close is a separate sensitivity, not an extra jointly covered family",
                              "warning": "Not valid certification when freshness/PIT/dependence or fixed-policy requirements fail; no policy accepted is legitimate."},
                "bootstrap": {"resamples": 5000, "seed": 82, "unit": "calendar month", "include_empty_selection_months": True, "diagnostic_only": True},
                "prerequisite_schema": {"observations_csv": list(OBSERVATION_FIELDS),
                    "legacy_raw_sports_exact": sorted(_REQUIRED_BASE_FIELDS | {"best_of", "golgg_match_id"}),
                    "legacy_metadata_exact": ["golgg_match_id", "match_start_at", "feature_history_max_at"],
                    "legacy_outcomes_exact": ["golgg_match_id", "match_start_at", "y_true", "settled_at"],
                    "evidence_paths": "relative to observations CSV; hashes prove current bytes only, not timestamp authority or semantic provenance"},
                "existing_contract": {"sources": list(SOURCE_CONTRACTS),
                    "capture": "protocol.locked_at <= captured_at <= predicted_at < match_start_at; feature_history_max_at <= captured_at; local source_available_at/data_cutoff_at equal actual capture",
                    "settlement": "legacy scorer checks match_start_at <= settled_at <= observed_at, exact forecast IDs/start and binary y_true; this alone does NOT prove actual match completion",
                    "archive": "sealed protocol/model/source/environment hashes, immutable raw/metadata/forecast bytes, exact prediction recomputation; EXP089 API is for its two models, not an EXP081 certificate",
                    "features": "date-only history strictly prior source calendar day; pre-event locally observed roster/bracket evidence; corrected prospective feature version is explicitly NOT verified frozen-model-compatible"},
                "additional_performance_contract": [
                    "Frozen candidate implementation/model/feature contract and event inclusion rule must be committed before outcomes, with independent timestamp evidence; do not infer this from flags.",
                    "predicted_at denotes the frozen odds-free sports/student prediction; decision_at denotes the later market-aware forecast/selection computation, never silently the same provenance event.",
                    "data_cutoff_at == captured_at <= predicted_at <= quote_at <= quote_observed_at <= decision_at < match_start_at < match_end_at <= outcome_available_at <= settled_at <= outcome_observed_at <= audit time",
                    "captured_at is the existing canonical cutoff/source capture, explicitly bound to data_cutoff_at; feature_history_max_at <= source_available_at <= captured_at; strictly earlier source calendar day for date-only history",
                    "protocol lock <= predicted_at <= quote_at <= quote_observed_at <= decision_at; independent same-book active/executable paired series quote and exact side identity evidence",
                    "Actual end time, outcome publication/source and settlement record linked to frozen forecast IDs; no start-time proxy for completion",
                    "Independent audit of unseen cohort/outcomes and temporal dependence; local boolean flags or hashes alone cannot satisfy these requirements."],
                "readiness_blockers": list(BLOCKERS)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", protocol)
    protocol_hash = sha256(args.output_dir / "protocol.json")
    with (args.output_dir / "protocol.sha256").open("x") as handle:
        handle.write(protocol_hash + "\n")
    try:
        gate, observation_inputs, observation_rows = observation_gate(
            observation_path, observation_lock, observation_protocol_hash or protocol_hash,
            timestamp(datetime.now(timezone.utc)))
        inputs.update(observation_inputs)
        summaries, forecasts, cache = [], [], {}
        for entry in entries:
            summary, frame = diagnostic(entry, cache)
            summaries.append(summary)
            forecasts.append(frame)
        result = {"status": "blocked_no_certified_fresh_cohort", "protocol_sha256": protocol_hash,
                  "promotion": False, "eligible_live_rows": 0, "statistical_certificate_valid": False,
                  "accepted_policies": [], "readiness": gate, "readiness_blockers": list(BLOCKERS),
                  "statistical_diagnostics": summaries, "missing_candidate_streams": [
                      {"candidate": name, "phase": phase} for phase in ("open", "close") for name in CANDIDATES
                      if not any(entry["name"] == name and entry["phase"] == phase for entry in entries)],
                  "implementable_gate": "Local schema, exact IDs, disjoint streams, time inequalities and referenced byte hashes are checked; independent provenance and IID/freshness qualification is intentionally not inferred.",
                  "new_observations_available": "See bounded inventory; historical retrospective forecasts and quotes are never relabeled fresh."}
        write_json(args.output_dir / "inventory.json", discovery)
        write_json(args.output_dir / "summary.json", result)
        combined = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(columns=[*BASE, *CANDIDATES, "candidate", "selected", "unit_payoff"])
        combined.to_csv(args.output_dir / "oos_predictions.csv", index=False)
        pd.DataFrame(observation_rows, columns=["golgg_match_id", "candidate", "phase", "structural_checks_passed", "eligible_live", "reasons"]).to_csv(args.output_dir / "observation_readiness.csv", index=False)
        for path, digest in {**inputs, **code}.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError(f"input or code changed during audit: {path}")
        if frozen_hashes() != frozen:
            raise RuntimeError("frozen models changed during audit")
        write_json(args.output_dir / "manifest.json", {"input_hashes": inputs, "code_hashes": code, "frozen_hashes": frozen,
                   "output_hashes": {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}})
        print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir), "diagnostic_streams": len(summaries), "accepted_policies": [], "promotion": False}))
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "promotion": False})
        raise


if __name__ == "__main__":
    main()
