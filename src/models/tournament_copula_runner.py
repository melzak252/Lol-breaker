"""Exclusive, source-bound development study of a marginal-preserving copula.

The active goal is not completed by this experiment. Frozen starting lineups
are conditional scenarios; day-only labels are not publication timestamps.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, roots_hermitenorm

from src.analysis.bootstrap import block_ratio_interval
from src.analysis.tournament_metrics import score_research_event
from src.analysis.tournament_numerics import summarize_tournament_numerics
from src.models import tournament_improvement_runner as previous
from src.models import tournament_numerical_benchmark as numerical
from src.models.tournament_bracket_simulator import simulate_tournament_bracket, _pair_scoreline_laws
from src.models.tournament_research_cohort import build_observed_labels, validate_research_event
from src.models.tournament_scorelines import ConditionalScorelineModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "marginal_preserving_tournament_copula_protocol_v1"
DESIGN_PATH = Path("data/05_model_input/tournament_copula/c0_tournament_copula_20261007_01/design.json")
SOURCE_RUN = "tournament_c0_20261006_01"
NUMERICAL_MANIFEST = Path("data/05_model_input/tournament_numerical/tournament_numerical_20261007_01/input_manifest.json")
CANDIDATE = "native_c0_shared_player_copula"
CONTROLS = ("native_c0_unchanged", CANDIDATE, "prior_calibrated_glicko_format", "uniform")
BASE_MODELS = (CONTROLS[0], CONTROLS[2], CONTROLS[3])
PATH_METRIC = "normalized_joint_sports_path_nll"
SCORE_METRICS = ("champion_brier", "champion_log_loss", "finalists_brier", "joint_finalists_brier",
                 "joint_finalists_log_loss", "advance_brier", "joint_qualification_brier",
                 "joint_qualification_log_loss", "placement_rps", "swiss_record_rps",
                 "series_length_rps", "scoreline_log_loss")
METRICS = (PATH_METRIC, *SCORE_METRICS)
GUARDS = ("champion_brier", "champion_log_loss", "advance_brier", "placement_rps",
          "joint_finalists_brier", "joint_finalists_log_loss", "joint_qualification_brier",
          "joint_qualification_log_loss")
CODE = tuple(dict.fromkeys((*previous.CODE, *numerical.RUNTIME_CODE,
    "src/models/tournament_copula.py", "src/models/tournament_copula_runner.py",
    "src/analysis/tournament_numerics.py", "src/analysis/metrics.py")))


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return numerical._json(path)


def _paths(run_id):
    numerical._component(run_id)
    return {stage: PROJECT_ROOT / "data" / stage / "tournament_copula" / run_id
            for stage in ("05_model_input", "06_models", "07_model_output", "08_reporting")}


def _runtime():
    return {"python": sys.version, "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "pandas", "scikit-learn", "pyarrow")}}


def _locator(path, root):
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return "project:" + str(path.relative_to(PROJECT_ROOT))
    if path.is_relative_to(root):
        return "research:" + str(path.relative_to(root))
    raise ValueError(f"Study source is outside configured project/research roots: {path}")


def _resolve(locator, root=None):
    if locator.startswith("project:"):
        return PROJECT_ROOT / locator.removeprefix("project:")
    if locator.startswith("research:") and root is not None:
        return root / locator.removeprefix("research:")
    raise ValueError(f"Invalid source locator: {locator}")


def _relocate(value, old_project, old_root, root):
    """Old absolute paths are lineage only, never a research-root resolver."""
    if isinstance(value, dict):
        return {_relocate(k, old_project, old_root, root): _relocate(v, old_project, old_root, root)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_relocate(v, old_project, old_root, root) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        if path.is_relative_to(old_project):
            return str(PROJECT_ROOT / path.relative_to(old_project))
        if path.is_relative_to(old_root):
            return str(root / path.relative_to(old_root))
    return value


def _admit_sources(root):
    """Revalidate retained tables under current eligibility, not old runtime pins."""
    source05 = PROJECT_ROOT / "data/05_model_input/tournament_improvement" / SOURCE_RUN
    source08 = PROJECT_ROOT / "data/08_reporting/tournament_improvement" / SOURCE_RUN
    protocol_path = source05 / "protocol.json"
    source = _json(protocol_path)
    old_project = Path(source["paths"]["05_model_input"]).parents[3]
    old_root = Path(source["research_root"])
    relocate = lambda v: _relocate(v, old_project, old_root, root)
    receipt, seal = _json(source08 / "receipt.json"), _json(source08 / "sealed_verification.json")
    if receipt.get("status") != "COMPLETED" or receipt.get("smoke") or seal.get("status") != "PASS":
        raise ValueError("Copula reuse requires a completed sealed non-smoke source study")
    bindings = {}
    numerical._bind(protocol_path, receipt["protocol_sha256"], bindings)
    for name in ("receipt.json", "sealed_verification.json", "coverage.json", "sources.json"):
        path = source08 / name
        if name == "sealed_verification.json":
            bindings[str(path.resolve())] = _sha(path)
        else:
            digest = seal["evidence_sha256"].get(str(path.relative_to(PROJECT_ROOT)))
            if digest is None:
                digest = relocate(receipt["output_sha256"]).get(str(path))
            numerical._bind(path, digest, bindings)
    original_manifest_path = source["cohort_path"]
    manifest_path = Path(relocate(original_manifest_path))
    numerical._bind(manifest_path, source["bindings"][original_manifest_path], bindings)
    manifest = _json(manifest_path)
    coverage = _json(source08 / "coverage.json")
    cells = {(c["event_id"], c["model"]): c for c in coverage["cells"]}
    if len(cells) != len(coverage["cells"]):
        raise ValueError("Source coverage contains duplicate event/model cells")
    inventory = _json(source08 / "sources.json")
    if inventory["bindings"] != source["bindings"] or inventory["artifact_contracts"] != source["artifact_contracts"]:
        raise ValueError("Source protocol and sealed inventory differ")
    events, labels, tables, absences = {}, {}, {}, {}
    for event_id, original in manifest["events"].items():
        event = relocate(original)
        events[event_id] = event
        if not event.get("input_path"):
            for model in CONTROLS:
                base = CONTROLS[0] if model == CANDIDATE else model
                reason = event["model_eligibility"][base].get("reason")
                if not reason:
                    raise ValueError("Absent source event requires a prerequisite reason")
                absences[event_id, model] = reason
            continue
        validate_research_event(event)
        numerical._bind(event["input_path"], event["input_sha256"], bindings)
        numerical._bind(event["labels_path"], event["labels_sha256"], bindings)
        frozen_input = relocate(_json(event["input_path"]))
        for field in ("id", "edition_id", "cutoff", "year", "teams", "rosters", "roster_evidence",
                      "spec", "source_provenance", "scope"):
            if frozen_input[field] != event[field]:
                raise ValueError(f"Manifest/frozen input differ: {event_id}/{field}")
        observed = relocate(_json(event["labels_path"]))
        rebuilt = build_observed_labels(event_id, event["edition_id"], observed["matches"], spec=event["spec"])
        if any(observed.get(key) != value for key, value in rebuilt.items()):
            raise ValueError(f"Observed labels differ from fully consumed legal route: {event_id}")
        expected_matches = 40 if event_id.startswith("worlds-") else 14 if event_id.startswith("msi-") else 5
        if len(observed["matches"]) != expected_matches:
            raise ValueError(f"Incomplete observed sports response path: {event_id}")
        placement_source = observed.get("official_placement_source")
        if placement_source:
            numerical._bind(placement_source["path"], placement_source["sha256"], bindings)
        labels[event_id] = observed
        for model in BASE_MODELS:
            cell = cells[event_id, model]
            if cell["available"] != event["model_eligibility"][model]["available"]:
                raise ValueError("Source coverage/eligibility differ")
            if not cell["available"]:
                if not cell.get("reason") or cell.get("forecast_path"):
                    raise ValueError("Unavailable source model needs reason and no forecast")
                absences[event_id, model] = cell["reason"]
                if model == CONTROLS[0]:
                    absences[event_id, CANDIDATE] = cell["reason"]
                continue
            original_path = cell["forecast_path"]
            path = Path(relocate(original_path))
            numerical._bind(path, receipt["output_sha256"][original_path], bindings)
            payload = relocate(_json(path))
            if payload["event"] != event or payload["model"] != model or payload["event_id"] != event_id:
                raise ValueError("Retained forecast/event identity differs")
            probabilities = numerical._validate_pairs(event, payload, bindings)
            tables[event_id, model] = (payload, probabilities, path)
    return events, labels, tables, absences, bindings, source


def _check_chronology(events, labels, partitions):
    requested = [event for part in ("TRAIN", "CAL", "ASSESS") for event in partitions[part]]
    if len(requested) != 8 or len(set(requested)) != 8 or set(requested) != set(events):
        raise ValueError("Frozen study must retain exactly all eight requested editions")
    if len(partitions["TRAIN"]) != 4 or len(partitions["CAL"]) != 2 or len(partitions["ASSESS"]) != 2:
        raise ValueError("Frozen chronological partition sizes differ")
    for earlier, later in (("TRAIN", "CAL"), ("CAL", "ASSESS")):
        boundary = min(events[e]["cutoff"][:10] for e in partitions[later] if events[e].get("cutoff"))
        for event in partitions[earlier]:
            if event not in labels:
                raise ValueError(f"Mandatory {earlier} complete path unavailable: {event}")
            for match in labels[event]["matches"]:
                # Conservative next-day release proxy for day-only observations.
                release = pd.Timestamp(match["date"][:10]) + pd.Timedelta(days=1)
                if release >= pd.Timestamp(boundary):
                    raise ValueError(f"{earlier} label release proxy crosses {later} origin: {event}")


def freeze_copula_study(run_id, *, research_root=None, design_path=None):
    """Freeze current bytes and all 32 requested cells, without running any fit."""
    paths = _paths(run_id)
    if any(paths[stage].exists() for stage in paths if stage != "05_model_input"):
        raise FileExistsError("Copula execution stage already exists; choose a new run_id")
    input_dir = paths["05_model_input"]
    if input_dir.exists() and any(not p.is_file() or p.name not in {"design.json", "saved_goal.json"}
                                  for p in input_dir.iterdir()):
        raise FileExistsError("Copula input namespace already occupied")
    root = previous._root(research_root)
    design_source = Path(design_path) if design_path is not None else PROJECT_ROOT / DESIGN_PATH
    if not design_source.is_absolute():
        design_source = PROJECT_ROOT / design_source
    design = _json(design_source)
    authoritative = _json(PROJECT_ROOT / DESIGN_PATH)
    if design != authoritative:
        raise ValueError("Study design differs from the authoritative frozen recipe")
    goal_source = PROJECT_ROOT / design["goal_path"]
    goal = _json(goal_source)
    events, labels, tables, absences, bindings, source = _admit_sources(root)
    _check_chronology(events, labels, design["partitions"])
    canonical = previous._canonical_frame(source["design"], root)
    numeric_path = PROJECT_ROOT / NUMERICAL_MANIFEST
    numeric = _json(numeric_path)
    if numeric["source_run_id"] != SOURCE_RUN:
        raise ValueError("Numerical source transition points to a different study")
    for cell, contract in numeric["forecasting_pair_inputs"].items():
        event_id, model = cell.split("/")
        if model not in BASE_MODELS:
            continue
        # Reuse the portable path suffix, not its historical workstation prefix.
        path = PROJECT_ROOT / "data/05_model_input/tournament_numerical" / numeric["run_id"] / "pairs" / Path(contract["path"]).name
        numerical._bind(path, contract["sha256"], bindings)
        supplied = _json(path)
        rows = {(r["team_a"], r["team_b"], r["best_of"]): r["p"] for r in supplied["rows"]}
        if rows != tables[event_id, model][1]:
            raise ValueError("Numerical first-four native pair inputs differ")
    for path in (design_source, goal_source, numeric_path, PROJECT_ROOT / source["design"]["base_forecasts"],
                 PROJECT_ROOT / "conf/base/research_benchmark.json", PROJECT_ROOT / "conf/base/tournament_manifest.json"):
        bindings[str(path.resolve())] = _sha(path)
    for relative in CODE:
        path = PROJECT_ROOT / relative
        bindings[str(path.resolve())] = _sha(path)
    bank = root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    for path in bank.rglob("*"):
        if path.is_file():
            bindings[str(path.resolve())] = _sha(path)
    labels_path = root / "nonlinear-history-20260912/data/03_primary/aligned_scoreline_labels.parquet"
    bindings[str(labels_path.resolve())] = _sha(labels_path)
    for tree in (root / previous.replay.CAUSAL_ARCHIVE_RELATIVE,
                 root / "nonlinear-history-20260912/code", root / "a0-phase-walkforward-20260914/code"):
        for path in tree.rglob("*.py"):
            bindings[str(path.resolve())] = _sha(path)
    # The source rule/roster evidence is an input too, not merely text copied
    # into a scenario. Bind its sealed non-code evidence under current roots.
    old_project = Path(source["paths"]["05_model_input"]).parents[3]
    old_root = Path(source["research_root"])
    for original_path, digest in source["bindings"].items():
        path = Path(_relocate(original_path, old_project, old_root, root))
        if path.is_relative_to(PROJECT_ROOT / "data/artifacts") or path == PROJECT_ROOT / "conf/base/tournament_formats.json":
            numerical._bind(path, digest, bindings)
    from src.analysis.ev_asof_replay import REPLAY_CONTRACT_VERSION
    if REPLAY_CONTRACT_VERSION != numeric["current_source_transition"]["current_replay_contract"]:
        raise ValueError("Current source transition requires explicit review")
    cells = []
    for part in ("TRAIN", "CAL", "ASSESS"):
        for event_id in design["partitions"][part]:
            for model in CONTROLS:
                base = CONTROLS[0] if model == CANDIDATE else model
                item = tables.get((event_id, base))
                cells.append({"event_id": event_id, "partition": part, "model": model,
                    "available": item is not None, "reason": absences.get((event_id, model)),
                    "source_forecast": _locator(item[2], root) if item else None})
    input_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("design.json", design), ("saved_goal.json", goal)):
        path = input_dir / name
        if path.exists():
            if _json(path) != payload:
                raise ValueError(f"Existing prefit receipt differs: {name}")
        else:
            previous.save_json(path, payload)
        bindings[str(path.resolve())] = _sha(path)
    runtime = _runtime()
    runtime_path = input_dir / "runtime.json"
    previous.save_json(runtime_path, runtime)
    bindings[str(runtime_path.resolve())] = _sha(runtime_path)
    transition = {**numeric["current_source_transition"],
        "current_source_sha256": _sha(PROJECT_ROOT / "src/analysis/ev_asof_replay.py"),
        "source_protocol_bindings_are_runtime_certification": False}
    protocol = {"schema": SCHEMA, "run_id": run_id, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": design, "goal": goal, "runtime": runtime,
        "research_root_local_provenance": str(root),
        "bindings": {_locator(path, root): digest for path, digest in bindings.items()},
        "code_sha256": {relative: _sha(PROJECT_ROOT / relative) for relative in CODE},
        "cells": cells, "events": events, "canonical_design": source["design"],
        "canonical_rows": len(canonical), "source_transition": transition,
        "source_protocol": _locator(PROJECT_ROOT / "data/05_model_input/tournament_improvement" / SOURCE_RUN / "protocol.json", root),
        "scenario": "development_exposed_frozen_starting_roster_conditional_scenario",
        "roster_limit": "Actual per-match substitutions unrecorded; retain five starting players throughout each complete path",
        "label_time_limit": "Day-only observations; conservative next-day label release proxy, not authenticated publication",
        "path_scope": design["recipe"]["path_scope"], "base_fits": 0,
        "requested_dependence_fits": 1, "goal_complete": False}
    protocol_path = input_dir / "protocol.json"
    previous.save_json(protocol_path, protocol)
    previous.save_json(input_dir / "protocol_binding.json",
                       {"schema": SCHEMA, "protocol_sha256": _sha(protocol_path)})
    return protocol_path


def _verify(protocol_path, research_root=None):
    protocol = _json(protocol_path)
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported copula protocol schema")
    # Project sources can be checked before resolving an optional external root.
    root = None
    for locator, digest in protocol["bindings"].items():
        if locator.startswith("research:") and root is None:
            root = previous._root(research_root)
        path = _resolve(locator, root)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"Frozen copula binding changed: {locator}")
    anchor_path = Path(protocol_path).with_name("protocol_binding.json")
    if not anchor_path.is_file() or _json(anchor_path).get("protocol_sha256") != _sha(protocol_path):
        raise ValueError("Frozen copula protocol binding changed")
    root = root or previous._root(research_root)
    if protocol["runtime"] != _runtime():
        raise ValueError("Frozen copula runtime changed")
    if protocol["design"] != _json(PROJECT_ROOT / DESIGN_PATH):
        raise ValueError("Frozen copula recipe changed")
    if _json(protocol_path) != protocol:
        raise ValueError("Protocol changed during verification")
    return protocol, root


def _marginal_guard(frame, rho, *, require_canonical=True):
    if not math.isfinite(rho) or not 0 <= rho <= .5:
        raise ValueError("Copula rho must lie in [0,.5]")
    counts = frame.groupby("model_year").size().to_dict()
    if require_canonical and (len(frame) != 11550 or counts != {2024: 4642, 2025: 3805, 2026: 3103}):
        raise ValueError("Canonical guard requires all 11550 native rows (4642/3805/3103)")
    p = frame.p_c0.to_numpy(float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid canonical native margin")
    nodes, weights = roots_hermitenorm(128)
    weights /= math.sqrt(2 * math.pi)
    error = 0.
    for values in np.array_split(np.unique(p[(p > 0) & (p < 1)]), 32):
        if len(values):
            integrated = ndtr((ndtri(values)[:, None] + math.sqrt(rho) * nodes) /
                              math.sqrt(1 - rho)) @ weights
            error = max(error, float(np.max(np.abs(integrated - values))))
    # Independent normal contrast and residual have total variance rho+(1-rho)=1.
    # Consequently the prior Bernoulli margin is exactly p, including endpoints;
    # LL/Brier deltas are analytically zero, not sample-mean or copied-output tests.
    return {"passed": error < 1e-10, "rows": len(frame), "annual_membership": counts,
        "rho": rho, "analytic_identity": "Var(sqrt(rho)*Z+sqrt(1-rho)*epsilon)=1; Phi(Phi_inverse(p))=p",
        "quadrature_nodes": 128, "quadrature_max_absolute_error": error,
        "certainties_preserved": True, "log_loss_delta": 0., "brier_delta": 0.,
        "canonical_accuracy_improvement_claim": False,
        "coverage_complete": not require_canonical or len(frame) == 11550}


def _comparison(rows, requested, partition, reference, metric):
    selected = {(r["event_id"], r["model"]): r for r in rows if r["partition"] == partition}
    deltas, weights, editions, nonfinite, support, absent_targets = [], [], [], [], [], []
    for event in requested:
        pair = [selected.get((event, model)) for model in (CANDIDATE, reference)]
        if any(row is None or not row.get("eligible") for row in pair):
            support.append(f"{event}: missing eligible {CANDIDATE}/{reference}")
            continue
        values = [row["metrics"].get(metric) for row in pair]
        counts = [row["target_counts"].get(metric, 0) for row in pair]
        if all(value is None for value in values) and counts == [0, 0] and all(
                row["unavailable_reasons"].get(metric) for row in pair):
            absent_targets.append(event)
            continue
        if any(value is None for value in values) or counts[0] != counts[1] or counts[0] <= 0:
            support.append(f"{event}/{metric}: incomplete common target support")
            continue
        editions.append(pair[0]["edition_id"])
        if pair[0]["edition_id"] != pair[1]["edition_id"]:
            raise ValueError("Paired edition identities differ")
        if not all(math.isfinite(value) for value in values):
            nonfinite.append(event)
            continue
        deltas.append(values[0] - values[1])
        weights.append(counts[0])
    if len(set(editions)) != len(editions):
        raise ValueError("Copula study expects one complete event per edition")
    result = {"partition": partition, "reference": reference, "metric": metric,
        "eligible_editions": len(editions), "edition_ids": editions, "nonfinite_editions": nonfinite,
        "support_reasons": support, "absent_target_events": absent_targets,
        "mean_delta": None, "ci_lower": None, "ci_upper": None,
        "interval_reason": None, "bootstraps": 5000, "interval_unit": "whole_edition"}
    if nonfinite:
        result["interval_reason"] = "Nonfinite available target loss; no finite-intersection substitution"
    elif not deltas:
        result["interval_reason"] = "No common eligible target editions"
    else:
        result["mean_delta"] = float(np.mean(deltas))
        if support:
            result["interval_reason"] = "Incomplete requested common support"
        elif len(deltas) < 2:
            result["interval_reason"] = "Fewer than two independent editions; no degenerate confidence interval"
        else:
            interval = block_ratio_interval(deltas, np.ones(len(deltas)), editions, draws=5000, seed=20261007)
            result.update(ci_lower=interval["ci_lower"], ci_upper=interval["ci_upper"], bootstrap=interval)
            result["target_weighted_sensitivity"] = block_ratio_interval(
                np.asarray(deltas) * weights, weights, editions, draws=5000, seed=20261007)
    return result


def _decision(rows, marginal_guard, partitions, *, numerical_passed):
    comparisons = [_comparison(rows, partitions[part], part, reference, metric)
        for part in ("CAL", "ASSESS") for reference in (CONTROLS[0], CONTROLS[2])
        for metric in sorted({m for row in rows for m in row.get("metrics", {})})]
    cal_reasons, primary_reasons, gate_reasons = [], [], []
    for c in comparisons:
        if c["partition"] != "CAL" or c["reference"] != CONTROLS[0] or c["metric"] not in (PATH_METRIC, *GUARDS):
            continue
        if len(c["absent_target_events"]) == len(partitions["CAL"]):
            continue
        cal_reasons.extend(c["support_reasons"])
        delta = c["mean_delta"]
        if delta is None or (delta >= 0 if c["metric"] == PATH_METRIC else delta > 0):
            cal_reasons.append(f"{c['metric']}: CAL strict response improvement/nonworsening guard failed")
    if not marginal_guard.get("passed"):
        cal_reasons.append("Canonical mathematical marginal guard failed")
    if not numerical_passed:
        cal_reasons.append("Fixed-rho higher-power numerical verification failed")
    by_event = defaultdict(dict)
    for row in rows:
        if row["partition"] == "ASSESS":
            by_event[row["event_id"]][row["model"]] = row
    eligible = []
    for event in partitions["ASSESS"]:
        models = by_event[event]
        missing = [m for m in CONTROLS if m not in models or not models[m].get("eligible")]
        if missing:
            primary_reasons.append(f"{event}: missing mandatory models {', '.join(missing)}")
        elif all(models[m]["metrics"].get(PATH_METRIC) is not None and
                 math.isfinite(models[m]["metrics"][PATH_METRIC]) for m in CONTROLS):
            eligible.append(models[CANDIDATE]["edition_id"])
        else:
            primary_reasons.append(f"{event}: nonfinite or absent primary response score")
    n = len(set(eligible))
    if n < 8:
        primary_reasons.append(f"Only {n} independent complete primary editions; minimum eight required")
    for c in comparisons:
        if c["partition"] != "ASSESS" or c["metric"] not in (PATH_METRIC, *GUARDS):
            continue
        if len(c["absent_target_events"]) == len(partitions["ASSESS"]):
            continue
        primary_reasons.extend(c["support_reasons"])
        if c["metric"] == PATH_METRIC:
            if c["ci_upper"] is None or c["ci_upper"] >= 0:
                gate_reasons.append(f"{c['reference']}: strictly negative primary response upper bound unavailable")
        elif c["mean_delta"] is None or c["mean_delta"] > 0:
            gate_reasons.append(f"{c['reference']}/{c['metric']}: primary point guard failed")
    for row in rows:
        if row.get("eligible") and any(row.get(key) is not True for key in
                                      ("structural_checks_passed", "chronology_checks_passed")):
            gate_reasons.append(f"{row['event_id']}/{row['model']}: structural/chronology checks failed")
    cal_passed = not cal_reasons
    retained = cal_passed and not primary_reasons and not gate_reasons
    return {"status": "COMPLETED_ARITHMETIC_AND_MODEL_EXECUTION", "adopted": False,
        "decision": "NOT_ADOPTED_INSUFFICIENT_TOURNAMENT_EVIDENCE" if primary_reasons else
                    "HISTORICAL_DEVELOPMENT_CANDIDATE_RETAINED" if retained else "NOT_ADOPTED",
        "retained_development_only": retained, "calibration_gate": {"passed": cal_passed, "reasons": sorted(set(cal_reasons))},
        "eligible_primary_editions": n, "minimum_primary_editions": 8,
        "primary_support_reasons": list(dict.fromkeys(primary_reasons)), "gate_reasons": sorted(set(gate_reasons)),
        "comparisons": comparisons, "goal_complete": False, "promotion_claim": False,
        "great_or_calibrated_claim": False, "all_years_development_exposed": True,
        "training_scores_are_assessment": False, "future_authentic_evidence_required": True}


def _path_event(event, labels, probabilities):
    return {"event_id": event["id"], "cutoff": event["cutoff"], "probability_cutoff": event["cutoff"],
            "rosters": event["rosters"], "probabilities": probabilities,
            "matches": labels["matches"], "spec": event["spec"]}


def _diagnostic_nll(diagnostic):
    value = diagnostic.get("normalized_nll")
    if value is not None:
        return float(value)
    if diagnostic.get("log_probability_status") == "negative_infinity" or diagnostic.get("reason"):
        return math.inf
    raise ValueError("Path diagnostic has neither a normalized score nor an explicit impossibility reason")


def run_copula_study(protocol_path, *, research_root=None, smoke=False):
    """Execute one frozen TRAIN fit, fixed-rho CAL/ASSESS and matched controls.

    Smoke uses a distinct namespace, seeds and small budgets; it is never formal
    evidence. No canonical inference or frozen base retraining occurs.
    """
    protocol, root = _verify(protocol_path, research_root=research_root)
    run_id = protocol["run_id"] + "__smoke" if smoke else protocol["run_id"]
    paths = _paths(run_id)
    if any(paths[stage].exists() for stage in ("06_models", "07_model_output", "08_reporting")):
        raise FileExistsError("Copula execution stage already exists")
    start = time.perf_counter()
    events, labels, tables, absences, bindings, source = _admit_sources(root)
    design, partitions = protocol["design"], protocol["design"]["partitions"]
    _check_chronology(events, labels, partitions)
    # Re-admission may only consume already-frozen sources, never newly found cells.
    for path, digest in bindings.items():
        if protocol["bindings"].get(_locator(path, root)) != digest:
            raise ValueError(f"Source not bound before fit: {path}")
    current_cells = {(e, m): (e, CONTROLS[0] if m == CANDIDATE else m) in tables
                     for e in events for m in CONTROLS}
    if current_cells != {(c["event_id"], c["model"]): c["available"] for c in protocol["cells"]}:
        raise ValueError("Source availability changed since freeze")
    frame = previous._canonical_frame(source["design"], root)
    for stage in ("06_models", "07_model_output", "08_reporting"):
        paths[stage].mkdir(parents=True, exist_ok=False)
    report_dir, output_dir = paths["08_reporting"], paths["07_model_output"]
    timings, fits, rows, path_diagnostics, numeric_failures = {}, 0, [], [], []
    receipt = {"run_id": run_id, "protocol_run_id": protocol["run_id"], "smoke": smoke,
        "formal_evidence": not smoke, "status": "RUNNING", "base_fits": 0,
        "protocol_sha256": _sha(protocol_path), "code_sha256": protocol["code_sha256"],
        "runtime": protocol["runtime"], "goal_complete": False}
    try:
        from src.models.tournament_copula import fit_tournament_copula, tournament_copula_path_likelihood
        fit_settings = design["recipe"]["fit_integration"]
        verify_settings = design["recipe"]["verification_integration"]
        simulation = design["simulation"]
        n = 64 if smoke else simulation["fixed_complete_paths"]
        path_seed = simulation["seed"] + (100000 if smoke else 0)
        latent_seed = simulation["latent_seed"] + (100000 if smoke else 0)
        scorelaw = ConditionalScorelineModel(beta_momentum=.35)
        conditional = scorelaw.conditional_distributions()
        train = [_path_event(events[e], labels[e], tables[e, CONTROLS[0]][1]) for e in partitions["TRAIN"]]
        fit_start = time.perf_counter()
        model = fit_tournament_copula(train, max_rho=design["recipe"]["rho_bounds"][1],
            sobol_power=5 if smoke else fit_settings["scrambled_sobol_power"],
            replicates=2 if smoke else fit_settings["replicates"],
            seed=fit_settings["seed"] + (100000 if smoke else 0),
            training_cutoff=min(events[e]["cutoff"] for e in partitions["CAL"]))
        fits += 1
        timings["fit_seconds"] = time.perf_counter() - fit_start
        previous.save_json(paths["06_models"] / "copula.json", {"model": model.to_dict(),
            "smoke": smoke, "formal_evidence": not smoke, "training_editions": partitions["TRAIN"],
            "dependence_fits": fits, "base_fits": 0})
        guard = _marginal_guard(frame, model.rho)
        from src.analysis.model_benchmark import compute_benchmark_summary
        guard["native_baseline_benchmark_by_year"] = {
            str(year): asdict(compute_benchmark_summary(group.y.to_numpy(int),
                group.p_c0.to_numpy(float), probability_epsilon=None))
            for year, group in frame.groupby("model_year")}
        guard["benchmark_scope"] = "Current unclipped baseline adapter; any calibration regression is descriptive, not a new predictor fit"
        previous.save_json(report_dir / "canonical_marginal_guard.json", guard)
        coverage = []
        for part in ("TRAIN", "CAL", "ASSESS"):
            for event_id in partitions[part]:
                event = events[event_id]
                for control in CONTROLS:
                    base = CONTROLS[0] if control == CANDIDATE else control
                    cell_start = time.perf_counter()
                    if (event_id, base) not in tables:
                        reason = absences[event_id, control]
                        row = {"event_id": event_id, "edition_id": event["edition_id"], "partition": part,
                            "model": control, "eligible": False, "metrics": dict.fromkeys(METRICS),
                            "target_counts": dict.fromkeys(METRICS, 0),
                            "unavailable_reasons": dict.fromkeys(METRICS, reason)}
                        rows.append(row)
                        coverage.append({"event_id": event_id, "model": control, "available": False, "reason": reason})
                        continue
                    retained, probabilities, source_path = tables[event_id, base]
                    kwargs = {"conditional_score_distributions": conditional,
                              "simulations": n, "seed": path_seed, "integration": "monte_carlo"}
                    sampler_provenance = None
                    if control == CANDIDATE:
                        draws = model.sample(event["rosters"], n, latent_seed)
                        kwargs.update(team_strength_draws=draws, copula_rho=model.rho)
                        sampler_provenance = {"kind": "iid_prior_gaussian_player_streams", "algorithm": "PCG64",
                            "seed": latent_seed, "seed_derivation": "SHA256(explicit root seed, player identity)",
                            "player_identities": sorted(p for roster in event["rosters"].values() for p in roster),
                            "frozen_rosters": event["rosters"], "distribution": "standard_normal_team_states",
                            "learned_player_variance_or_transfer_benefit": False, "model": model.to_dict()}
                    forecast = simulate_tournament_bracket(event["spec"], probabilities, **kwargs)
                    if control == CANDIDATE:
                        # Generic supplied vectors are not iid. Only this actual Gaussian
                        # model generator certifies the new independent complete paths.
                        forecast["integration"]["latent_sampling"] = "iid_prior_gaussian_player_streams"
                        forecast["integration"]["latent_sampling_provenance"] = sampler_provenance
                        forecast["integration"].update(iid=True,
                            effective_sampling_unit="independent_complete_path",
                            fixed_n=n, stopping_rule="fixed_n")
                        forecast["numerical_uncertainty"] = summarize_tournament_numerics(forecast)
                        forecast["finite_latent_conditional_pair_scoreline_laws"] = forecast["pair_scoreline_laws"]
                        # Analytic prior margins and original-p winner bins are unchanged.
                        # Preserve finite-vector mixture separately, never call its sample
                        # mean the exact marginal or use it as the native series guard.
                        forecast["pair_scoreline_laws"] = _pair_scoreline_laws(
                            probabilities, None, conditional, None)
                        forecast["pair_scoreline_laws"]["marginal_scope"] = "analytic_iid_gaussian_prior_not_empirical_latent_mean"
                    observed = labels[event_id]
                    scored = score_research_event(event, forecast, observed, pair_table=retained["pair_table"])
                    rho = model.rho if control == CANDIDATE else 0.
                    diag = tournament_copula_path_likelihood(_path_event(event, observed, probabilities), rho,
                        sobol_power=6 if smoke else verify_settings["scrambled_sobol_power"],
                        replicates=2 if smoke else verify_settings["replicates"],
                        seed=verify_settings["seed"] + (100000 if smoke else 0))
                    nll = _diagnostic_nll(diag)
                    se = diag.get("normalized_nll_standard_error")
                    passed = math.isfinite(nll) and se is not None and math.isfinite(se) and se <= verify_settings["maximum_normalized_log_likelihood_standard_error"]
                    if not passed:
                        numeric_failures.append(f"{event_id}/{control}: fixed-rho path integration failed precision/impossibility gate")
                    path_diagnostics.append({"event_id": event_id, "model": control, "partition": part,
                        "rho": rho, "diagnostic": diag, "numerical_passed": passed,
                        "scope": design["recipe"]["path_scope"]})
                    metrics = {metric: scored[metric] for metric in SCORE_METRICS}
                    metrics[PATH_METRIC] = nll
                    counts = {**scored["target_counts"], PATH_METRIC: len(observed["matches"])}
                    row = {"event_id": event_id, "edition_id": event["edition_id"], "partition": part,
                        "year": event["year"], "model": control, "eligible": True,
                        "metrics": metrics, "target_counts": counts,
                        "unavailable_reasons": scored["unavailable_reasons"],
                        "actual_joint_numerics": scored.get("actual_joint_numerics"),
                        "structural_checks_passed": True, "chronology_checks_passed": True,
                        "assessment_evidence": part == "ASSESS" and not smoke,
                        "roster_basis": event.get("roster_basis"), "simulations": n}
                    if not math.isfinite(nll):
                        row["unavailable_reasons"][PATH_METRIC] = diag.get("reason", "Impossible sports response path")
                    rows.append(row)
                    output_path = output_dir / f"{event_id}__{control}.json"
                    previous.save_json(output_path, {"event_id": event_id, "model": control,
                        "event": event, "pair_table": retained["pair_table"], "forecast": forecast,
                        "source_forecast": _locator(source_path, root),
                        "scoreline_model": {"kind": "fixed_markov", "beta_momentum": .35},
                        "latent_provenance": sampler_provenance, "path_diagnostic": diag,
                        "score": row, "formal_evidence": not smoke})
                    coverage.append({"event_id": event_id, "model": control, "available": True,
                        "reason": None, "forecast_path": str(output_path)})
                    timings[f"{event_id}/{control}"] = time.perf_counter() - cell_start
        decision = _decision(rows, guard, partitions, numerical_passed=not numeric_failures)
        decision.update(smoke=smoke, formal_evidence=not smoke, numerical_failure_reasons=numeric_failures,
                        dependence_fits=fits, rho=model.rho)
        if smoke:
            decision.update(status="NONFORMAL_SMOKE_COMPLETED", decision="NONFORMAL_SMOKE_NO_MODEL_DECISION",
                            retained_development_only=False)
        denominators = {part: {metric: {control: {
            "requested_events": len(partitions[part]),
            "eligible_events": sum(r["eligible"] and r["metrics"].get(metric) is not None for r in rows
                                   if r["partition"] == part and r["model"] == control),
            "finite_events": sum(r["eligible"] and r["metrics"].get(metric) is not None and
                                 math.isfinite(r["metrics"][metric]) for r in rows
                                 if r["partition"] == part and r["model"] == control),
            "target_count": sum(r["target_counts"].get(metric, 0) for r in rows
                                if r["partition"] == part and r["model"] == control)}
            for control in CONTROLS} for metric in METRICS} for part in ("TRAIN", "CAL", "ASSESS")}
        previous.save_json(report_dir / "coverage.json", {"cells": coverage, "requested_cells": 32,
                                                         "metric_denominators": denominators})
        previous.save_json(report_dir / "scores.json", {"rows": rows, "path_diagnostics": path_diagnostics,
            "train_scope": "fit diagnostics only, never assessment", "path_scope": design["recipe"]["path_scope"]})
        csv_rows = [{"event_id": r["event_id"], "edition_id": r["edition_id"], "model": r["model"],
                     "partition": r["partition"], "eligible": r["eligible"], **r["metrics"]} for r in rows]
        pd.DataFrame(csv_rows).to_csv(report_dir / "edition_scores.csv", index=False, mode="x")
        previous.save_json(report_dir / "decision.json", decision)
        receipt.update(status="COMPLETED", dependence_fits=fits, rho=model.rho,
            simulations=n, path_seed=path_seed, latent_seed=latent_seed, timings_seconds=timings,
            numerical_failure_reasons=numeric_failures, elapsed_seconds=time.perf_counter() - start,
            nonfinite_reasons={f"{r['event_id']}/{r['model']}/{metric}":
                r["unavailable_reasons"].get(metric, "Nonfinite available score")
                for r in rows for metric, value in r["metrics"].items()
                if value is not None and not math.isfinite(value)},
            outputs_sha256={str(path): _sha(path) for stage in ("06_models", "07_model_output", "08_reporting")
                            for path in paths[stage].rglob("*") if path.is_file()})
        previous.save_json(report_dir / "receipt.json", receipt)
        return {"run_id": run_id, "protocol_path": str(protocol_path), "decision": decision,
                "paths": {stage: str(path) for stage, path in paths.items()},
                "receipt_path": str(report_dir / "receipt.json")}
    except Exception as exc:
        receipt.update(status="FAILED", dependence_fits=fits, timings_seconds=timings,
            elapsed_seconds=time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
        previous.save_json(report_dir / "receipt.json", receipt)
        raise
