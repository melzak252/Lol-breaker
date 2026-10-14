"""Numerical integration of sealed historical tournament scenario inputs.

This module never fits a predictor, regenerates native inputs or reruns the closed
study. Exactness concerns the retained conditional law, not sports accuracy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from src.analysis.tournament_metrics import _actual_score_law, _compiled_pair_laws, score_research_event
from src.models.tournament_bracket_simulator import simulate_tournament_bracket
from src.models.tournament_formats import required_best_ofs
from src.models.tournament_improvement_runner import save_json
from src.models.tournament_native import _digest, native_model_eligibility
from src.models.tournament_research_cohort import validate_research_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDITIONS = ("worlds-2023", "worlds-2024", "msi-2023", "msi-2024")
MODELS = ("native_c0_unchanged", "origin_eligible_annual_a0", "prior_calibrated_glicko_format", "uniform")
METHODS = ("monte_carlo", "hybrid", "exact")
DECISION = "HISTORICAL_NUMERICAL_INTEGRATION_ONLY"
SCHEMA = "historical_tournament_numerical_integration_v1"
TARGETS = ("champion_prob", "final_prob", "advance_prob", "placement_prob",
           "swiss_record_prob", "joint_final_prob", "joint_advance_prob")
RUNTIME_CODE = (
    "src/models/tournament_numerical_benchmark.py", "src/models/tournament_bracket_simulator.py",
    "src/models/tournament_formats.py", "src/analysis/tournament_metrics.py",
    "src/models/frozen_tournament.py",
    "src/analysis/tournament_numerics.py", "src/models/tournament_scorelines.py",
    "src/models/tournament_native.py", "src/models/tournament_research_cohort.py",
    "src/models/tournament_improvement_runner.py", "src/analysis/ev_asof_replay.py",
    "betting_app/core/models/native_c0.py", "scripts/run_tournament_distribution_benchmark.py",
)
RULES = {
    "worlds-2023": {"research_reference": 22, "sections": "6.3.2–6.3.4",
        "url": "https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/1068e920b7b4826b2c34a22ec602646ded96cb3f.pdf",
        "verified_rules": "Round1 regional separation; later Swiss rounds permit rematches; 3-0 versus 3-2 quarterfinals on opposite halves."},
    "worlds-2024": {"research_reference": 23, "sections": "6.3.3–6.3.4",
        "url": "https://cmsassets.rgpub.io/sanity/files/dsfx7636/news/f69130a70849462c75fbd834f740f170963b173e.pdf?accountingTag=lol_esports",
        "verified_rules": "Swiss no-rematch restriction with discretionary waiver; drawn teams shift to next available legal slot."},
    "msi-2023": {"research_reference": 24, "sections": "6.2–6.3",
        "url": "https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/96b920d226d09d3c875711aeaac0214975636d56.pdf",
        "verified_rules": "Eight-team BO5 double elimination, one grand final without reset."},
    "msi-2024": {"research_reference": 25, "sections": "6.1–6.3",
        "url": "https://cmsassets.rgpub.io/sanity/files/dsfx7636/news_live/c5eb4dfed0dd2574a94a145c6ac9a23818cbf8b8.pdf",
        "verified_rules": "BO5 double elimination, one final; regional separation through upper rounds1/2 and lower round1; draw after May5."},
}


def _component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id must be a single safe nonempty path component")
    return value


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite source JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def _selection(values, allowed, label):
    selected = tuple(allowed if values is None else values)
    if not selected or len(set(selected)) != len(selected) or any(x not in allowed for x in selected):
        raise ValueError(f"{label} must contain unique supported values: {allowed}")
    return selected


def _clock(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Cutoff must be timezone-aware")
    return result.astimezone(timezone.utc)


def _bind(path, expected, bindings):
    path = Path(path)
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Expected source digest must be canonical SHA-256: {path}")
    key = str(path.resolve())
    if key in bindings and bindings[key] == expected:
        return
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"Sealed source bytes changed: {path}")
    bindings[str(path.resolve())] = actual


def _validate_pairs(event, payload, bindings):
    """Complete pair/BO and retained native-origin proof, without inference."""
    table = payload["pair_table"]
    if table.get("probability_unit") != "series_win" or payload.get("team_strength_draws") is not None:
        raise ValueError("Only retained fixed series-win laws are admitted; no residual arms")
    formats = required_best_ofs(event["spec"])
    order = {team: i for i, team in enumerate(event["teams"])}
    expected = {(a, b, bo) for a, b in combinations(event["teams"], 2) for bo in formats}
    probabilities = {}
    native_kind = {"native_c0_unchanged": "c0", "origin_eligible_annual_a0": "a0"}.get(payload["model"])
    if native_kind:
        contract = table["artifact_contract"]
        eligibility = native_model_eligibility(event, native_kind, contract)
        if not eligibility["available"]:
            raise ValueError(f"Retained native artifact no longer eligible: {eligibility['reason']}")
        for key, path in contract["artifact_paths"].items():
            _bind(path, contract["artifact_sha256"][key], bindings)
        provenance = table["provenance"]
        if (_clock(provenance["cutoff"]) != _clock(event["cutoff"]) or
                provenance["roster_sha256"] != _digest(event["rosters"]) or
                provenance["pair_table_sha256"] != _digest(table["rows"])):
            raise ValueError("Native pair table cutoff/roster/row binding differs")
    elif payload["model"] == "prior_calibrated_glicko_format":
        provenance = table["provenance"]
        calibration = provenance["configuration"]["calibration"]
        if (_clock(provenance["cutoff"]) != _clock(event["cutoff"]) or
                calibration["model_year"] != event["year"] or
                calibration["cal_end_exclusive"] > event["cutoff"][:10]):
            raise ValueError("Retained Glicko calibration origin or cutoff differs")
        for format_calibration in calibration["formats"].values():
            label_day = format_calibration.get("max_label_day")
            if label_day is not None and _clock(label_day) >= _clock(event["cutoff"]):
                raise ValueError("Retained Glicko calibration uses post-origin labels")
        source = provenance["source"]
        _bind(source["path"], source["sha256"], bindings)
    for row in table["rows"]:
        a, b, bo, p = row["A"], row["B"], row["BO"], row["p"]
        if (a not in order or b not in order or a == b or type(bo) is not int or
                isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("Malformed retained pair probability")
        key = (a, b, bo) if order[a] < order[b] else (b, a, bo)
        if key in probabilities:
            raise ValueError("Duplicate retained pair/BO cell")
        probabilities[key] = p if order[a] < order[b] else 1 - p
        if native_kind:
            proof = row["provenance"]
            target = proof["target"]
            if (target["team1_id"] != a or target["team2_id"] != b or target["best_of"] != bo or
                    _clock(target["decision_at"]) != _clock(event["cutoff"]) or
                    _clock(target["effective_decision_at"]) != _clock(event["cutoff"]) or
                    target["cutoff_day"] != event["cutoff"][:10] or
                    target["competition_context"] != event["event_title"] or
                    target.get("feature_available") is not True):
                raise ValueError("Retained native target identity or cutoff differs")
            feature_day = target["feature_source_max_day"]
            if feature_day is not None and feature_day >= event["cutoff"][:10]:
                raise ValueError("Retained native inputs use unreleased feature history")
            # Hypothetical pairs have no actual target ID or result to release.
            # Never import a target-specific v3 exemption into the v4 runtime.
            if any(target.get(k) is not None for k in ("match_id", "golgg_match_id", "result", "result_day", "label_release_day")):
                raise ValueError("Retained pair target is not an outcome-blind hypothetical target")
            for side, team in (("a", a), ("b", b)):
                roster = sorted(event["rosters"][team])
                for field in (f"roster_{side}", f"tensor_player_order_{side}",
                              f"raw_tensor_player_order_{side}", f"temporal_tensor_player_order_{side}"):
                    if target.get(field) != roster:
                        raise ValueError("Retained native tensor player order differs")
            if proof["artifact_contract"] != table["artifact_contract"] or proof.get("probability_unit") != "series_win":
                raise ValueError("Native row artifact contract differs from table")
    if set(probabilities) != expected:
        raise ValueError("Retained pair table is incomplete for entrant pairs and required BO formats")
    forecast = payload["forecast"]
    if forecast.get("conditional_score_distributions") is None and forecast.get("series_score_distributions") is None:
        raise ValueError("Retained supplied score law is missing")
    compiled = _compiled_pair_laws(forecast["pair_scoreline_laws"])
    if set(compiled) != set(probabilities):
        raise ValueError("Compiled score laws do not cover the complete pair table")
    for key, p in probabilities.items():
        actual = _actual_score_law(None, forecast, p, key[2])
        saved_p, saved_pmf, _ = compiled[key]
        if abs(saved_p - p) > 1e-10 or any(abs(actual[s] - saved_pmf[s]) > 1e-10 for s in actual):
            raise ValueError("Saved score law differs from the supplied engine law or winner marginal")
    return probabilities


def _load_source(source_run_id, editions, models):
    source_run_id = _component(source_run_id)
    source05 = PROJECT_ROOT / "data/05_model_input/tournament_improvement" / source_run_id
    source08 = PROJECT_ROOT / "data/08_reporting/tournament_improvement" / source_run_id
    protocol_path = source05 / "protocol.json"
    protocol, receipt = _json(protocol_path), _json(source08 / "receipt.json")
    seal = _json(source08 / "sealed_verification.json")
    if (protocol["run_id"] != source_run_id or receipt["run_id"] != source_run_id or
            seal["run_id"] != source_run_id or receipt["status"] != "COMPLETED" or
            receipt.get("smoke") or seal["status"] != "PASS"):
        raise ValueError("Numerical inputs require a completed sealed non-smoke source study")
    bindings = {}
    _bind(protocol_path, receipt["protocol_sha256"], bindings)
    for name in ("receipt.json", "sealed_verification.json", "sources.json", "coverage.json"):
        path = source08 / name
        if name != "sealed_verification.json":
            expected = (seal["evidence_sha256"].get(str(path.relative_to(PROJECT_ROOT)))
                        or receipt["output_sha256"].get(str(path)))
            if expected is None:
                raise ValueError(f"Source receipt lacks a byte binding: {path}")
            _bind(path, expected, bindings)
        else:
            bindings[str(path)] = _sha(path)
    manifest_path = Path(protocol["cohort_path"])
    _bind(manifest_path, protocol["bindings"][str(manifest_path)], bindings)
    manifest = _json(manifest_path)
    coverage = _json(source08 / "coverage.json")
    cells = {(c["event_id"], c["model"]): c for c in coverage["cells"]}
    if len(cells) != len(coverage["cells"]):
        raise ValueError("Source coverage has duplicate event/model cells")
    sources = _json(source08 / "sources.json")
    if sources["bindings"] != protocol["bindings"] or sources["artifact_contracts"] != protocol["artifact_contracts"]:
        raise ValueError("Source inventory and sealed protocol disagree")
    events, payloads, absences = {}, {}, {}
    for edition in editions:
        event = manifest["events"][edition]
        validate_research_event(event)
        input_path, label_path = Path(event["input_path"]), Path(event["labels_path"])
        _bind(input_path, event["input_sha256"], bindings)
        _bind(label_path, event["labels_sha256"], bindings)
        frozen_input = _json(input_path)
        for field in ("id", "edition_id", "cutoff", "year", "teams", "rosters", "roster_evidence", "spec", "source_provenance", "scope"):
            if frozen_input[field] != event[field]:
                raise ValueError(f"Manifest and frozen input differ: {edition}/{field}")
        events[edition] = event
        for model in models:
            cell = cells[edition, model]
            eligible = event["model_eligibility"][model]
            if cell["available"] != eligible["available"]:
                raise ValueError("Saved eligibility and coverage disagree")
            if not cell["available"]:
                if not cell.get("reason") or cell.get("forecast_path"):
                    raise ValueError("Unavailable saved cell requires reason and no forecast")
                absences[edition, model] = cell["reason"]
                continue
            path = Path(cell["forecast_path"])
            _bind(path, receipt["output_sha256"][str(path)], bindings)
            payload = _json(path)
            if (payload["event_id"] != edition or payload["edition_id"] != edition or
                    payload["model"] != model or payload["event"] != event):
                raise ValueError("Saved forecast identity or event differs")
            probabilities = _validate_pairs(event, payload, bindings)
            payloads[edition, model] = (payload, probabilities, path)
    return events, payloads, absences, bindings, protocol, receipt


def _runtime_bindings():
    paths = [PROJECT_ROOT / name for name in RUNTIME_CODE]
    # Optional exact backend modules are part of the executed implementation.
    paths += list((PROJECT_ROOT / "src/models").glob("tournament*exact*.py"))
    return {str(path): _sha(path) for path in paths}


def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def _compare(left, right, reference):
    rows = {}
    for target in TARGETS:
        a, b = _flatten(left.get(target)), _flatten(right.get(target))
        if not a and not b:
            rows[target] = {"available": False, "reason": "Target absent in this event"}
            continue
        keys = set(a) | set(b)
        # Missing observed joint categories are zero integration mass, not a
        # declaration that the underlying event is structurally impossible.
        differences = {k: a.get(k, 0.) - b.get(k, 0.) for k in sorted(keys)}
        rows[target] = {"available": True, "categories": len(keys),
                        "max_absolute_difference": max(abs(x) for x in differences.values()),
                        "root_mean_square_difference": math.sqrt(math.fsum(x*x for x in differences.values()) / len(keys)),
                        "l1_difference": math.fsum(abs(x) for x in differences.values()),
                        "signed_differences": differences}
    return {"reference": reference, "scope": "integration_of_identical_retained_scenario_law",
            "coupling": "same_seed_not_certified_common_random_numbers",
            "predictive_improvement_claim": False, "targets": rows}


def _execute(run_id, source_run_id, editions, models, simulations, seed, methods, persist):
    editions = _selection(editions, EDITIONS, "editions")
    models = _selection(models, MODELS, "models")
    methods = _selection(methods, METHODS, "methods")
    if type(simulations) is not int or simulations < 1 or type(seed) is not int or seed < 0:
        raise ValueError("simulations must be positive and seed a nonnegative integer")
    paths = None
    if persist:
        run_id = _component(run_id)
        paths = {s: PROJECT_ROOT / "data" / s / "tournament_numerical" / run_id
                 for s in ("05_model_input", "07_model_output", "08_reporting")}
        if any(path.exists() for path in paths.values()):
            raise FileExistsError("Numerical stage already exists; choose a new run_id")
    start = time.perf_counter()
    events, payloads, absences, source_bindings, source_protocol, source_receipt = _load_source(source_run_id, editions, models)
    implementation = _runtime_bindings()
    from src.analysis.ev_asof_replay import REPLAY_CONTRACT_VERSION
    if REPLAY_CONTRACT_VERSION != "a0_input_repair_asof_v4":
        raise ValueError("Numerical reuse requires the current v4 target-release guard")
    research_path = PROJECT_ROOT / "reports/reliable_tournament_simulation_research_20261006.md"
    research_bytes = research_path.read_bytes()
    research_sha256 = hashlib.sha256(research_bytes).hexdigest()
    reviewed_research_path = paths["05_model_input"] / "reviewed_research.md" if persist else research_path
    manifest_path = PROJECT_ROOT / "conf/base/tournament_manifest.json"
    unchanged_manifest = _sha(manifest_path)
    runtime_history = PROJECT_ROOT / "src/analysis/ev_asof_replay.py"
    transition = {"source_protocol_bindings_are_runtime_certification": False,
                  "historical_source_sha256": source_protocol["bindings"].get(str(runtime_history)),
                  "current_source_sha256": implementation[str(runtime_history)],
                  "current_replay_contract": REPLAY_CONTRACT_VERSION,
                  "policy": "Retained hypothetical pair tables have no actual target ID or released target label; identities, prior feature boundaries and artifact eligibility revalidated. No new native inference or target-release exemption.",
                  "authenticated_bytes_lineage": True, "authenticated_pre_event_publication": False}
    input_manifest = {"schema": SCHEMA, "run_id": run_id, "source_run_id": source_run_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "editions": editions, "models": models,
        "methods": methods, "simulations": simulations, "seed": seed,
        "stopping_rule": "fixed_n_predeclared", "probability_targets": TARGETS,
        "comparison": "MC versus exact MSI; MC versus hybrid Worlds (method disagreement, not exact error)",
        "exact_reference_editions": [e for e in editions if e.startswith("msi-")],
        "scope": "retained_historical_conditional_scenario_inputs",
        "native_input_inference": False, "predictive_fits": 0, "residual_arms": [],
        "source_bindings": {p: digest for p, digest in source_bindings.items()
                            if p not in {event["labels_path"] for event in events.values()}},
        "implementation_bindings": implementation,
        "current_source_transition": transition,
        "canonical_tournament_manifest": {"path": str(manifest_path), "sha256": unchanged_manifest, "modified": False},
        "research_report": {"path": str(reviewed_research_path), "sha256": research_sha256,
            "binding": "exclusive_reviewed_snapshot" if persist else "smoke_repository_source",
            "original_repository_path": str(research_path), "original_repository_sha256": research_sha256},
        "events": {e: {"edition_id": e, "cutoff": event["cutoff"], "spec": event["spec"],
                         "teams": event["teams"], "rosters": event["rosters"], "scope": event["scope"],
                         "roster_basis": event["roster_basis"], "roster_evidence": event["roster_evidence"],
                         "source_provenance": event["source_provenance"]} for e, event in events.items()},
        "rule_source_ledger": {e: {**RULES[e], "organizer_draw_law_verified": False,
            "engine_draw_law": "existing minimum-cost/uniform-feasible-matching scenario, not certified Riot lottery",
            "known_opening_draw_scope": events[e]["source_provenance"]["draw_basis"]} for e in editions}}
    label_manifest = {e: {"path": events[e]["labels_path"], "sha256": events[e]["labels_sha256"]} for e in editions}
    if persist:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=False)
        with reviewed_research_path.open("xb") as stream:
            stream.write(research_bytes)
        pair_inputs = {}
        for (edition, model), (retained, probabilities, source_path) in payloads.items():
            input_path = paths["05_model_input"] / "pairs" / f"{edition}__{model}.json"
            supplied = retained["forecast"]
            save_json(input_path, {"edition_id": edition, "model": model,
                "source_forecast_sha256": source_bindings[str(source_path.resolve())],
                "probability_unit": "series_win",
                "rows": [{"team_a": a, "team_b": b, "best_of": bo, "p": p}
                         for (a, b, bo), p in probabilities.items()],
                "conditional_score_distributions": supplied["conditional_score_distributions"],
                "series_score_distributions": supplied["series_score_distributions"],
                "score_law_metadata": supplied["score_law_metadata"]})
            pair_inputs[f"{edition}/{model}"] = {"path": str(input_path), "sha256": _sha(input_path)}
        input_manifest["forecasting_pair_inputs"] = pair_inputs
        save_json(paths["05_model_input"] / "input_manifest.json", input_manifest)
        save_json(paths["05_model_input"] / "observed_labels_manifest.json", label_manifest)
    rows, comparisons = [], []
    for edition in editions:
        event = events[edition]
        labels = _json(event["labels_path"])
        for model in models:
            if (edition, model) in absences:
                rows.extend({"edition_id": edition, "model": model, "method": method,
                             "available": False, "reason": absences[edition, model]} for method in methods)
                continue
            retained, probabilities, source_path = payloads[edition, model]
            saved = retained["forecast"]
            requested = list(methods)
            if edition.startswith("msi-") and "exact" not in requested:
                requested.append("exact")
            forecasts = {}
            for method in requested:
                if method == "exact" and edition.startswith("worlds-"):
                    rows.append({"edition_id": edition, "model": model, "method": method,
                                 "available": False, "reason": "EXACT_ADAPTIVE_SWISS_NOT_SUPPORTED"})
                    continue
                reused_reference_from = None
                if (method == "exact" and method not in methods and
                        forecasts.get("hybrid", {}).get("integration", {}).get("kind") == "exact"):
                    forecast = forecasts["hybrid"]
                    elapsed = 0.0
                    reused_reference_from = "hybrid"
                else:
                    t0 = time.perf_counter()
                    forecast = simulate_tournament_bracket(copy.deepcopy(event["spec"]), probabilities,
                        scorelines=saved["series_score_distributions"],
                        conditional_score_distributions=saved["conditional_score_distributions"],
                        simulations=simulations, seed=seed, integration=method)
                    elapsed = time.perf_counter() - t0
                metrics = score_research_event(event, forecast, labels, pair_table=probabilities)
                forecasts[method] = forecast
                row = {"edition_id": edition, "model": model, "method": method, "available": True,
                       "reason": None, "requested_cell": method in methods, "wall_seconds": elapsed,
                       "reused_reference_from": reused_reference_from,
                       "additional_integration_evaluations": 0 if reused_reference_from else 1,
                       "integration": forecast["integration"], "metrics": metrics,
                       "numerical_uncertainty": forecast.get("numerical_uncertainty"),
                       "scope": event["scope"], "native_input_inference": False}
                if persist:
                    path = paths["07_model_output"] / f"{edition}__{model}__{method}.json"
                    save_json(path, {"schema": SCHEMA, "edition_id": edition, "model": model,
                        "method": method, "source_forecast_sha256": source_bindings[str(source_path.resolve())],
                        "reused_reference_from": reused_reference_from,
                        "scoreline_model": retained["scoreline_model"],
                        "supplied_score_law_metadata": saved["score_law_metadata"], "forecast": forecast})
                    row["forecast_path"] = str(path)
                rows.append(row)
            if "monte_carlo" in forecasts:
                reference = "exact" if edition.startswith("msi-") else "hybrid"
                if reference in forecasts:
                    comparison = _compare(forecasts["monte_carlo"], forecasts[reference], reference)
                    comparison.update(edition_id=edition, model=model,
                        exact_error=reference == "exact",
                        interpretation="arithmetic integration error" if reference == "exact" else "finite-sample method disagreement; neither is exact Swiss truth")
                    comparisons.append(comparison)
    for path, digest in source_bindings.items():
        _bind(path, digest, {})
    for path, digest in implementation.items():
        _bind(path, digest, {})
    if _sha(manifest_path) != unchanged_manifest:
        raise ValueError("Canonical tournament manifest changed during numerical execution")
    decision = {"decision": DECISION, "adoption": False, "promotion": False,
                "predictive_improvement_claim": False, "independent_prospective_editions": 0,
                "reason": "Numerical methods integrate identical retained historical scenario inputs; four requested editions do not establish predictive superiority."}
    report = {"schema": SCHEMA, "run_id": run_id, "source_run_id": source_run_id,
              "simulations": simulations, "seed": seed, "methods": methods,
              "requested_cells": len(editions) * len(models) * len(methods), "cells": rows,
              "comparisons": comparisons, "decision": decision,
              "elapsed_seconds": time.perf_counter() - start, "current_source_transition": transition,
              "source_decision_unchanged": source_receipt["decision"]}
    if persist:
        _bind(reviewed_research_path, research_sha256, {})
        save_json(paths["08_reporting"] / "metrics.json", report)
        save_json(paths["08_reporting"] / "decision.json", decision)
        outputs = {str(p): _sha(p) for stage in paths.values() for p in stage.rglob("*") if p.is_file()}
        save_json(paths["08_reporting"] / "receipt.json", {"schema": SCHEMA, "status": "COMPLETED",
            "run_id": run_id, "source_run_id": source_run_id, "decision": DECISION,
            "source_nonmutation": True, "manifest_nonmutation": True, "source_bindings": source_bindings,
            "implementation_bindings": implementation, "output_sha256": outputs,
            "native_input_inference": False, "predictive_fits": 0,
            "requested_cells": report["requested_cells"],
            "available_requested_cells": sum(r["available"] and r.get("requested_cell", True) for r in rows),
            "current_source_transition": transition})
    return report


def run_numerical_benchmark(run_id, *, source_run_id="tournament_c0_20261006_01", editions=None,
                            models=None, simulations=10000, seed=20261008,
                            methods=("monte_carlo", "hybrid")) -> dict:
    """Freeze and execute a new exclusive numerical run; retain unavailable cells."""
    return _execute(run_id, source_run_id, editions, models, simulations, seed, methods, True)


def numerical_smoke(*, simulations=8, seed=20261008, source_run_id="tournament_c0_20261006_01") -> dict:
    """Exercise all requested cells under retained laws without writing artifacts."""
    return _execute("numerical_smoke", source_run_id, EDITIONS, MODELS, simulations, seed,
                    ("monte_carlo", "hybrid"), False)
