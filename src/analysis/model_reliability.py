"""Locked reliability-study orchestration over the existing benchmark primitives.

Historical eligibility is distinct from prospective confirmation and from input
availability. No fitting, probability repair, market input, or promotion occurs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.a0_data_focus import _summary
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.research_benchmark import attach_candidate, losses, safe, validate_frame
from src.models.a0_data_focus import MODES, PROJECT_ROOT, resolve_research_root
from src.models.a0_native_refit import _sha256, _write_json_exclusive
from src.models.competition_tiers import classify_competition

CRITICAL = ("bo3", "bo5", "major", "source_unseen_player")
CONFIDENCE_EDGES = (.5, .6, .7, .8, .9, 1.)
NEUTRAL_TOLERANCE = 1e-12
HISTORY_COVERAGE_THRESHOLD = .9


def calibration_gate(interval, config):
    lower, upper = interval.get("ci_lower"), interval.get("ci_upper")
    if (interval["n"] < config["calibration_min_rows"]
            or interval["blocks"] < config["calibration_min_months"]
            or lower is None or upper is None
            or not np.isfinite([lower, upper]).all()):
        return "INSUFFICIENT_EVIDENCE"
    tolerance = config["calibration_gap_tolerance"]
    if lower >= -tolerance and upper <= tolerance:
        return "SUPPORTED_IN_DEVELOPMENT"
    return "FAILED_CALIBRATION_TOLERANCE"


def _sources():
    paths = ("src/analysis/model_reliability.py", "scripts/run_model_benchmark.py",
             "src/analysis/a0_data_focus.py", "src/analysis/model_benchmark.py",
             "src/analysis/production_market_benchmark.py", "src/analysis/research_benchmark.py",
             "src/models/competition_tiers.py")
    return {path: _sha256(PROJECT_ROOT / path) for path in paths}


def freeze_evaluation(run_id, family_protocol, family_predictions, family_completion,
                      *, family_profiles, research_root=None, output_name="evaluation_protocol.json"):
    """Freeze assessment sources/criteria before any study outcomes are scored."""
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be one path component")
    if Path(output_name).name != output_name or output_name in ("", ".", ".."):
        raise ValueError("protocol output name must be a file name")
    root = resolve_research_root(research_root)
    inputs = PROJECT_ROOT / "data/05_model_input/model_reliability" / run_id
    neural_path = inputs / "neural_protocol.json"
    neural = json.loads(neural_path.read_text())
    if neural["run_id"] != run_id or neural["schema"] != "a0_reliability_annual_v1":
        raise ValueError("neural protocol identity differs")
    family_protocol = Path(family_protocol).resolve()
    if _sha256(root / neural["canonical_path"]) != neural["canonical_sha256"]:
        raise ValueError("canonical data changed")
    protocol = {
        "schema": "model_reliability_evaluation_v1", "run_id": run_id,
        "scope": "EXPOSED_HISTORICAL_DEVELOPMENT_NOT_PROSPECTIVE",
        "neural_protocol": str(neural_path.relative_to(PROJECT_ROOT)),
        "neural_protocol_sha256": _sha256(neural_path),
        "neural_completion_sha256": _sha256(PROJECT_ROOT / "data/08_reporting/model_reliability" / run_id / "neural_completed.json"),
        "family_protocol": str(family_protocol.relative_to(PROJECT_ROOT)),
        "family_protocol_sha256": _sha256(family_protocol),
        "family_predictions": str(Path(family_predictions).resolve().relative_to(PROJECT_ROOT)),
        "family_completion": str(Path(family_completion).resolve().relative_to(PROJECT_ROOT)),
        "family_completion_sha256": _sha256(Path(family_completion).resolve()),
        "family_profiles": family_profiles,
        "canonical_path": neural["canonical_path"], "canonical_sha256": neural["canonical_sha256"],
        "evaluation": neural["evaluation"], "source_sha256": _sources(),
        "critical_profiles": list(CRITICAL), "guard_population": "all_2024_2026",
        "confidence_edges": list(CONFIDENCE_EDGES),
        "neutral_tolerance": NEUTRAL_TOLERANCE,
        "history_coverage_threshold": HISTORY_COVERAGE_THRESHOLD,
        "diagnostic_context_modes": list(MODES),
        "confidence_profiles": "paired=fixed annual A0; calibration=own favorite, exclude numerical neutral",
        "edition_sensitivity": "source tournament name plus calendar year, NOT verified independent editions",
        "eligibility": "primary improvement plus all noninferiority guards; never automatic adoption",
        "calibration_scope": "full-context overall and critical-profile gate; other context modes and confidence bins retain separate diagnostic statuses",
    }
    output = inputs / output_name
    _write_json_exclusive(output, protocol)
    return {"protocol": str(output), "sha256": _sha256(output)}


def _checked_json(path, expected_hash=None):
    path = Path(path)
    if expected_hash is not None and _sha256(path) != expected_hash:
        raise ValueError(f"receipt checksum differs: {path}")
    return json.loads(path.read_text())


def _check_artifacts(receipt):
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("completion receipt must enumerate immutable artifacts")
    for relative, expected in artifacts.items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"artifact checksum differs: {relative}")


def _check_family_receipt(receipt, protocol):
    frozen = _checked_json(PROJECT_ROOT / protocol["family_protocol"], protocol["family_protocol_sha256"])
    if (receipt.get("schema") != "family_reliability_execution_receipt_v1"
            or receipt.get("protocol_sha256") != protocol["family_protocol_sha256"]
            or receipt.get("run_id") != frozen["run_id"]
            or receipt.get("rows") != 11550
            or receipt.get("sources_sha256") != frozen["sources_sha256"]
            or receipt.get("inputs_sha256") != frozen["inputs_sha256"]):
        raise ValueError("complete matching family execution required")
    paths, hashes = receipt["output_paths"], receipt["outputs_sha256"]
    if not paths or set(paths) != set(hashes):
        raise ValueError("family output receipt must bind every artifact")
    if Path(paths["predictions.parquet"]).resolve() != (PROJECT_ROOT / protocol["family_predictions"]).resolve():
        raise ValueError("selected family predictions are not bound by the receipt")
    for name, path in paths.items():
        if _sha256(path) != hashes[name]:
            raise ValueError(f"family artifact checksum differs: {name}")
    if set(receipt["input_paths"]) != set(frozen["inputs_sha256"]):
        raise ValueError("family input receipt must bind every input")
    for name, path in receipt["input_paths"].items():
        if _sha256(path) != frozen["inputs_sha256"][name]:
            raise ValueError(f"family input checksum differs: {name}")
    for relative, expected in frozen["sources_sha256"].items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"family source checksum differs: {relative}")


def _align(base, candidate, probability, *, online_updates=False):
    attach_candidate(base, candidate, probability, online_updates=online_updates)
    indexed = candidate.assign(golgg_match_id=candidate.golgg_match_id.astype(str)).set_index("golgg_match_id")
    indexed = indexed.loc[base.golgg_match_id.astype(str)]
    for key in ("roster_a", "roster_b"):
        if key in candidate and [tuple(map(str, row)) for row in base[key]] != [tuple(map(str, row)) for row in indexed[key]]:
            raise ValueError(f"candidate roster orientation differs: {key}")
    return indexed


def _load_neural(base, protocol):
    run_id = protocol["run_id"]
    report_root = PROJECT_ROOT / "data/08_reporting/model_reliability" / run_id
    completed = _checked_json(report_root / "neural_completed.json", protocol["neural_completion_sha256"])
    if (completed["status"] != "TRAINING_COMPLETE_NOT_SCORED"
            or completed["protocol_sha256"] != protocol["neural_protocol_sha256"]
            or set(completed["origins"]) != {"2024", "2025", "2026"}):
        raise ValueError("complete matching annual neural training required")
    neural = _checked_json(PROJECT_ROOT / protocol["neural_protocol"], protocol["neural_protocol_sha256"])
    panels, seed_panels, receipt_hashes = {}, {}, {}
    for year in (2024, 2025, 2026):
        origin = completed["origins"][str(year)]
        receipt = _checked_json(PROJECT_ROOT / origin["completed_path"], origin["completed_sha256"])
        if receipt["contract"]["protocol_sha256"] != protocol["neural_protocol_sha256"]:
            raise ValueError("annual artifact belongs to another protocol")
        _check_artifacts(receipt)
        receipt_hashes[str(year)] = origin["completed_sha256"]
        output = PROJECT_ROOT / "data/07_model_output/model_reliability" / run_id / "neural" / str(year)
        for variant in ("control", "supervised_18"):
            candidate = pd.read_parquet(output / f"{variant}.parquet")
            panels.setdefault(variant, []).append(candidate)
            logit_path = output / f"{variant}_logits.npz"
            if year == 2026:
                reference_run = neural["reference_2026"]["run_id"]
                reference_receipt = _checked_json(
                    PROJECT_ROOT / "data/08_reporting/a0_data_focus" / reference_run / "completed.json",
                    neural["reference_2026"]["completed_sha256"])
                logit_path = PROJECT_ROOT / "data/07_model_output/a0_data_focus" / reference_run / f"{variant}_logits.npz"
                if _sha256(logit_path) != reference_receipt["artifacts"][str(logit_path.relative_to(PROJECT_ROOT))]:
                    raise ValueError("reference seed probabilities changed")
            with np.load(logit_path) as bundle:
                seeds = bundle["raw_seed_probability"][-len(candidate):]
            for mode_index, mode in enumerate(MODES):
                np.testing.assert_allclose(seeds[:, mode_index].mean(1), candidate[f"p_raw_neural_{mode}"], atol=1e-12, rtol=0)
                for seed_index, seed in enumerate(neural["training"]["seeds"]):
                    seed_panels.setdefault((variant, mode, seed), []).append(pd.DataFrame({
                        "golgg_match_id": candidate.golgg_match_id.astype(str), "p": seeds[:, mode_index, seed_index]}))
    probability_columns = []
    for variant, frames in panels.items():
        candidate = _align(base, pd.concat(frames, ignore_index=True), "p_full_full")
        for kind in ("full", "neural", "raw_neural"):
            for mode in MODES:
                column = f"{variant}__{kind}__{mode}"
                base[column] = candidate[f"p_{kind}_{mode}"].to_numpy(float)
                probability_columns.append(column)
        for column in ("history_stat_coverage", "csd_xpd_coverage"):
            values = candidate[column].to_numpy(float)
            if variant == "control":
                base[column] = values
            else:
                np.testing.assert_allclose(base[column], values, atol=0, rtol=0, equal_nan=True)
        for kind in ("neural", "raw_neural"):
            for mode in MODES:
                values = candidate[f"nll_{kind}_{mode}"].to_numpy(float)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("invalid scoreline NLL")
                base[f"{variant}__nll__{kind}__{mode}"] = values
    seeds = {}
    for key, frames in seed_panels.items():
        values = pd.concat(frames, ignore_index=True).set_index("golgg_match_id")
        seeds[key] = values.loc[base.golgg_match_id, "p"].to_numpy(float)
    return probability_columns, seeds, receipt_hashes


def _contrast(frame, candidate, reference, config):
    y = frame.y.to_numpy(int)
    a, b = frame[candidate].to_numpy(float), frame[reference].to_numpy(float)
    months = frame.date.astype(str).str[:7]
    proxy = frame.tournament_name.fillna("UNKNOWN").astype(str) + "|" + frame.date.astype(str).str[:4]
    ll = losses(y, a) - losses(y, b)
    kwargs = {"draws": config["resamples"], "seed": config["seed"]}
    return {"candidate": candidate, "reference": reference,
            "log_loss_monthly": paired_cluster_interval(ll, months, **kwargs),
            "brier_monthly": paired_cluster_interval((a-y)**2 - (b-y)**2, months, **kwargs),
            "log_loss_tournament_year_proxy": paired_cluster_interval(ll, proxy, **kwargs)}


def _profile_masks(frame, family_profiles):
    identities = [classify_competition(name, pd.Timestamp(day).date())
                  for name, day in zip(frame.tournament_name, frame.date)]
    frame["profile_tier"] = [item.tier.value for item in identities]
    frame["profile_family"] = [item.family for item in identities]
    frame["profile_scope"] = [item.scope.value for item in identities]
    masks = {"all": np.ones(len(frame), bool),
             "replication_2024_2025": frame.date.astype(str).str[:4].isin(["2024", "2025"]),
             "major": frame.profile_tier.eq("major"),
             "source_unseen_player": frame.unknown_players.gt(0),
             "all_players_seen": frame.unknown_players.eq(0),
             "new_organization": frame.new_organization,
             "established_organization": ~frame.new_organization,
             "w20_absent": frame.w20_min_gate.eq(0),
             "w20_present": frame.w20_min_gate.gt(0),
             "history_stat_low": frame.history_stat_coverage.lt(HISTORY_COVERAGE_THRESHOLD),
             "history_stat_high": frame.history_stat_coverage.ge(HISTORY_COVERAGE_THRESHOLD),
             "history_stat_unknown": frame.history_stat_coverage.isna(),
             "csd_xpd_absent": frame.csd_xpd_coverage.eq(0),
             "csd_xpd_present": frame.csd_xpd_coverage.gt(0),
             "csd_xpd_unknown": frame.csd_xpd_coverage.isna()}
    for bo in (1, 3, 5):
        masks[f"bo{bo}"] = frame.best_of.eq(bo)
    for year in (2024, 2025, 2026):
        masks[f"year_{year}"] = frame.date.astype(str).str[:4].eq(str(year))
    for field in ("profile_tier", "profile_family", "profile_scope", "experience", "coplay", "disagreement", *family_profiles):
        if field not in frame:
            raise ValueError(f"prespecified profile is missing: {field}")
        values = frame[field].fillna("UNKNOWN").astype(str)
        for value in sorted(values.unique()):
            masks[f"{field}={value}"] = values.eq(value)
    q = np.maximum(frame.p_a0_full, 1-frame.p_a0_full)
    for lower, upper in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
        masks[f"fixed_a0_confidence_{lower:.1f}_{upper:.1f}"] = (q >= lower) & (q < upper) & (q-.5 > NEUTRAL_TOLERANCE)
    masks["fixed_a0_neutral"] = q-.5 <= NEUTRAL_TOLERANCE
    return {name: np.asarray(mask, bool) for name, mask in masks.items()}


def _calibration(frame, column, config, mask=None):
    p = frame[column].to_numpy(float)
    neutral = np.abs(p-.5) <= NEUTRAL_TOLERANCE
    selected = ~neutral
    if mask is not None:
        selected &= mask
    confidence = np.maximum(p[selected], 1-p[selected])
    won = np.where(p[selected] > .5, frame.y.to_numpy(int)[selected], 1-frame.y.to_numpy(int)[selected])
    interval = paired_cluster_interval(confidence-won, frame.date.astype(str).str[:7].to_numpy()[selected],
                                       draws=config["resamples"], seed=config["seed"])
    return {"n": len(confidence), "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "favorite_win_rate": float(won.mean()) if len(won) else None,
            "neutral_excluded": int((neutral & (np.ones(len(p), bool) if mask is None else mask)).sum()),
            "gap_monthly": interval, "status": calibration_gate(interval, config)}


def _improvement(contrast, config):
    ll, brier = contrast["log_loss_monthly"], contrast["brier_monthly"]
    return (ll["ci_upper"] is not None and np.isfinite(ll["ci_upper"])
            and ll["ci_upper"] < config["ll_interval_upper_below"]
            and brier["delta"] <= config["brier_delta_at_most"])


def _guard(contrast, config):
    interval = contrast["log_loss_monthly"]
    if (interval["blocks"] < config["guard_min_months"] or interval["ci_upper"] is None
            or not np.isfinite(interval["ci_upper"])):
        return "INSUFFICIENT_EVIDENCE"
    return "PASS" if interval["ci_upper"] < config["guard_ll_margin"] else "NONINFERIORITY_NOT_ESTABLISHED"


def evaluate_study(protocol_path, output_dir, research_root=None):
    """Generate complete coverage, profiles and nonpromotional eligibility decisions."""
    protocol_path, output_dir = Path(protocol_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol = _checked_json(protocol_path)
    if protocol["schema"] != "model_reliability_evaluation_v1" or protocol["source_sha256"] != _sources():
        raise ValueError("evaluation source/protocol identity differs")
    config = protocol["evaluation"]
    if config["resamples"] < 5000 or config["probability_epsilon"] is not None:
        raise ValueError("locked unclipped monthly evaluation is required")
    root = resolve_research_root(research_root)
    canonical_path = root / protocol["canonical_path"]
    if _sha256(canonical_path) != protocol["canonical_sha256"]:
        raise ValueError("canonical population changed")
    frame = pd.read_parquet(canonical_path).rename(columns={f"p_{mode}": f"p_a0_{mode}" for mode in MODES})
    frame.golgg_match_id = frame.golgg_match_id.astype(str)
    if len(frame) != 11550 or frame.golgg_match_id.duplicated().any():
        raise ValueError("the complete unique 11,550-row population is required")
    probabilities = [f"p_a0_{mode}" for mode in MODES]
    neural_columns, seeds, neural_receipts = _load_neural(frame, protocol)
    probabilities.extend(neural_columns)
    family_receipt = _checked_json(PROJECT_ROOT / protocol["family_completion"], protocol["family_completion_sha256"])
    _check_family_receipt(family_receipt, protocol)
    family = pd.read_parquet(PROJECT_ROOT / protocol["family_predictions"])
    aligned = _align(frame, family, "p_family", online_updates=True)
    for column in ("p_player", "p_family", "p_player_raw", "p_family_raw"):
        frame[column] = aligned[column].to_numpy(float)
        probabilities.append(column)
    for column in protocol["family_profiles"]:
        if column in frame:
            raise ValueError(f"rating profile would overwrite canonical evidence: {column}")
        frame[column] = aligned[column].to_numpy()
    validate_frame(frame, probabilities)
    masks = _profile_masks(frame, protocol["family_profiles"])
    headline = {"a0": "p_a0_full", "control": "control__full__full",
                "supervised_18": "supervised_18__full__full", "player": "p_player", "family": "p_family"}
    calibration_models = dict(headline)
    for mode in MODES:
        if mode != "full":
            calibration_models[f"a0__{mode}"] = f"p_a0_{mode}"
            for variant in ("control", "supervised_18"):
                calibration_models[f"{variant}__{mode}"] = f"{variant}__full__{mode}"
    metrics, contrasts, calibrations, seed_metrics, scoreline = [], [], [], [], []
    contrast_index, calibration_index = {}, {}
    comparisons = {"neural": (headline["supervised_18"], headline["control"]),
                   "family": (headline["family"], headline["player"]),
                   "neural_vs_a0": (headline["supervised_18"], headline["a0"]),
                   "family_vs_a0": (headline["family"], headline["a0"])}
    for profile, mask in masks.items():
        selected = frame.loc[mask]
        columns = probabilities if profile in ("all", "replication_2024_2025", "year_2024", "year_2025", "year_2026") else list(calibration_models.values())
        for column in columns:
            metrics.append({"profile": profile, "column": column, **(_summary(selected, column) if len(selected) else {"n_samples": 0})})
        for name, (candidate, reference) in comparisons.items():
            value = _contrast(selected, candidate, reference, config)
            contrasts.append({"profile": profile, "comparison": name, **value})
            contrast_index[profile, name] = value
        for name, column in calibration_models.items():
            value = _calibration(frame, column, config, mask)
            calibrations.append({"profile": profile, "model": name, "column": column, **value})
            calibration_index[profile, name] = value
    for name, column in calibration_models.items():
        q = np.maximum(frame[column], 1-frame[column])
        for lower, upper in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
            mask = (q >= lower) & (q < upper) & (q-.5 > NEUTRAL_TOLERANCE)
            calibrations.append({"profile": f"own_confidence_{lower:.1f}_{upper:.1f}", "model": name,
                                 "column": column, **_calibration(frame, column, config, mask.to_numpy())})
    for profile in ("all", "replication_2024_2025", "year_2024", "year_2025", "year_2026"):
        selected = frame.loc[masks[profile]].copy()
        for (variant, mode, seed), values in seeds.items():
            selected["seed_probability"] = values[masks[profile]]
            seed_metrics.append({"profile": profile, "variant": variant, "mode": mode, "seed": seed,
                                 **_summary(selected, "seed_probability")})
        for column in frame.columns:
            if "__nll__" in column:
                scoreline.append({"profile": profile, "column": column, "n": len(selected), "mean_nll": selected[column].mean()})
    gates = {}
    for name, candidate in (("neural", "supervised_18"), ("family", "family")):
        primary_profile = "replication_2024_2025" if name == "neural" else "all"
        primary = contrast_index[primary_profile, name]
        improvement = _improvement(primary, config)
        full_nonworse = contrast_index["all", name]["log_loss_monthly"]["delta"] <= config["full_population_ll_delta_at_most"]
        guards = {profile: _guard(contrast_index[profile, name], config) for profile in CRITICAL}
        eligible = improvement and (full_nonworse or name == "family") and all(value == "PASS" for value in guards.values())
        support = {profile: calibration_index[profile, candidate]["status"] for profile in ("all", *CRITICAL)}
        gates[name] = {"primary_profile": primary_profile, "primary_improvement": bool(improvement),
                       "full_population_nonworse": bool(full_nonworse), "noninferiority_guards": guards,
                       "historically_eligible_for_forward": bool(eligible), "calibration_support": support,
                       "full_context_calibration_supported_in_development": all(value == "SUPPORTED_IN_DEVELOPMENT" for value in support.values()),
                       "prospective_confirmation": "NOT_RUN", "production_promotion": "NOT_AUTHORIZED"}
    result = {"status": "DEVELOPMENT_ONLY", "run_id": protocol["run_id"], "n": len(frame),
              "coverage": {column: {"available": int(frame[column].notna().sum()), "missing": int(frame[column].isna().sum())} for column in probabilities},
              "year_counts": frame.date.astype(str).str[:4].value_counts().sort_index().to_dict(),
              "protocol_sha256": _sha256(protocol_path), "neural_receipts": neural_receipts,
              "family_completion_sha256": _sha256(PROJECT_ROOT / protocol["family_completion"]),
              "gates": gates, "primary_contrasts": {name: contrast_index[gate["primary_profile"], name] for name, gate in gates.items()},
              "conditional_a0_family_integration": "REQUIRED_NEXT" if gates["family"]["primary_improvement"] else "NOT_TRIGGERED_GATE_FAILED",
              "blocking_unit": "calendar month; sensitivity uses tournament-name/year proxies, not verified editions",
              "probability_policy": "strict interior probabilities; unclipped LL and Brier everywhere",
              "limitations": ["Every year is exposed historical development", "Actual historical first-map rosters are a scenario, not authenticated announcements",
                              "Native CAL fits calibration and assembly on the same rows; no cross-fitting claim",
                              "Calibration intervals are aggregate support, not per-match confidence intervals", "No deployment, financial certification or completed prospective study"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output_dir / "joined.parquet", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(output_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(scoreline).to_csv(output_dir / "scoreline_nll.csv", index=False)
    _write_json_exclusive(output_dir / "contrasts.json", safe(contrasts))
    _write_json_exclusive(output_dir / "calibration.json", safe(calibrations))
    _write_json_exclusive(output_dir / "evaluation.json", safe(result))
    result["artifacts"] = {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()}
    _write_json_exclusive(output_dir / "completed.json", safe(result))
    return safe(result)
