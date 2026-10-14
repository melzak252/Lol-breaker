#!/usr/bin/env python3
"""Fit annual EXP039/EXP081 research derivatives on all eligible recovered history.

Offline only. A single strictly prior-day replay supplies both native original46
and canonical79 inputs. No odds or tournament inventory determines eligibility.
The checked-in frozen artifacts are hash-checked, never fitted or overwritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import os
import shutil
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import importlib.metadata
from itertools import groupby
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.corrected_history_evaluation import (
    build_exp039_features, chronological_fold_masks, code_hashes,
    exp039_probability, frozen_hashes, reliability_bins,
)
from scripts.export_prospective_features import iter_matches
from scripts.prospective_sports_features import (
    ChronologicalFeatureState, FEATURE_CONTRACT, FEATURE_VERSION, _prepare_history,
)
from scripts.train_and_tune_siamese_series import (
    build_model_artifact, build_training_features, fit_platt_scaling,
    train_single_member, write_json_exclusive,
)
from scripts.build_siamese_research_dataset import sha256
from scripts.benchmark_siamese_architectures import monthly_bootstrap, probability_metrics
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.symmetric_series import _independent_series_probability
from src.models.team_order import swap_orientation
from src.models.tournament_prediction import json_digest
from src.utils import golgg_schema as schema
from src.utils.module_loading import load_module_from_path

VERSION = "full-historical-annual-priorday-v1"
EXP039 = "exp039-priorday-original46-sym-slope-annual-v1"
SEEDS = [20260906 + 101 * i for i in range(5)]
NATIVE_FIELDS = ("player_tm_sigma_avg1", "player_tm_sigma_avg2")
OUTPUT_COLUMNS = [
    "golgg_match_id", "date", "team1_id", "team2_id", "team1_name", "team2_name",
    "best_of", "y_true", "p_exp039", "p_exp081", "feature_history_max_at", "fold_year",
]
BENCHMARK_VERSION = "matched-base-recipes-priorday-annual-v1"
STRONG_SEEDS = [20260906 + 17 * i for i in range(5)]
BENCHMARK_MODELS = {
    "p_exp039": "exp039-native46-matched-priorday-annual-v1",
    "p_linear79": "linear79-matched-priorday-annual-v1",
    "p_exp081": "exp081-weak-matched-priorday-annual-v1",
    "p_exp081_strong": "exp081-report-informed-membercal-matched-priorday-annual-v1",
    "p_exp081_strong_ensemble": "exp081-report-informed-ensemblecal-matched-priorday-annual-v1",
    "p_glicko": "player-glicko-once-bon-matched-priorday-annual-v1",
    "p_glicko_format": "player-glicko-once-bon-formatcal-matched-priorday-annual-v1",
}

def exp081_model_version(gamma: float) -> str:
    """Return the immutable research version for a finite focal-loss gamma."""
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("EXP081 focal-loss gamma must be finite and non-negative")
    label = f"{gamma:g}"
    family = "bce" if gamma == 0 else "focal"
    return f"exp081-priorday-canonical79-{family}-g{label}-annual-v1"


def exp081_loss_name(gamma: float) -> str:
    """Return the loss metadata corresponding to the validated gamma."""
    exp081_model_version(gamma)
    return "binary_cross_entropy" if gamma == 0 else f"focal_loss_gamma_{gamma:g}"


def emit(output: Path, stage: str, **values) -> None:
    record = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
    line = json.dumps(record, allow_nan=False)
    with (output / "run.jsonl").open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def implementation_hashes() -> dict:
    paths = [Path(__file__), ROOT / "scripts/prospective_sports_features.py",
             ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py",
             ROOT / "src/models/team_order.py", ROOT / "src/models/symmetric_series.py",
             ROOT / "src/models/tournament_prediction.py", ROOT / "src/analysis/probability_metrics.py",
             ROOT / "scripts/export_prospective_features.py", ROOT / "src/utils/golgg_schema.py"]
    return {**code_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def replay_implementation_hashes() -> dict:
    """Identify replay code independently of model fitting/reporting changes."""
    functions = (replay, iter_matches)
    hashes = {fn.__name__: hashlib.sha256(
        ast.dump(ast.parse(inspect.getsource(fn)), include_attributes=False).encode()
    ).hexdigest() for fn in functions}
    paths = [ROOT / "scripts/prospective_sports_features.py",
             ROOT / "scripts/build_siamese_research_dataset.py",
             ROOT / "scripts/05_ratingi_baseline/03_generate_ratings.py",
             ROOT / "scripts/06_metamodel/06i_best_metamodel_config_search.py",
             ROOT / "scripts/export_prospective_features.py", ROOT / "src/utils/module_loading.py",
             ROOT / "src/models/symmetric_series.py", ROOT / "src/utils/golgg_schema.py",
             *sorted((ROOT / "src/ratings").glob("*.py"))]
    return {**hashes, **{str(path.relative_to(ROOT)): sha256(path) for path in paths}}


def reuse_replay(source: Path, previous: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    """Reuse only hash-verified, explicitly audited rows, never a trusted prefix."""
    audit_path = previous / "audit.json"
    audit = json.loads(audit_path.read_text())
    expected_source = {"path": str(source.resolve()), "sha256": sha256(source)}
    if (audit.get("status") != "complete"
            or audit.get("source", {}).get("sha256") != expected_source["sha256"]
            or audit.get("feature_contract") != FEATURE_CONTRACT
            or audit.get("feature_contract_sha256") != json_digest(FEATURE_CONTRACT)
            or audit.get("replay_code_sha256") != replay_implementation_hashes()
            or audit.get("temporal_violations") != 0
            or audit.get("source_transitions_rejected_or_skipped") != 0
            or audit.get("availability_certified") is not False):
        raise ValueError("replay reuse requires matching audited source, feature contract and replay code provenance")
    files = ("snapshots.csv", "target_exclusions.csv", "source_inventory.csv")
    for name in files:
        if audit.get("outputs", {}).get(name) != sha256(previous / name):
            raise ValueError(f"replay reuse content hash mismatch: {name}")
    frame = pd.read_csv(previous / "snapshots.csv", dtype={
        "golgg_match_id": str, "team1_id": str, "team2_id": str, "team_a": str, "team_b": str})
    if (len(frame) != audit["snapshots"] or frame.golgg_match_id.duplicated().any()
            or not (frame.history_last_source_day < frame.date).all()
            or (pd.to_datetime(frame.feature_history_max_at, utc=True) > pd.to_datetime(frame.date, utc=True)).any()):
        raise ValueError("reused replay rows violate accounting or chronology")
    directory = output / "replay"
    directory.mkdir(exist_ok=False)
    for name in files:
        with (previous / name).open("rb") as src, (directory / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(directory / name) != audit["outputs"][name]:
            raise ValueError(f"reused replay changed during copy: {name}")
    audit = {**audit, "reuse": {"audit_path": str(audit_path.resolve()), "audit_sha256": sha256(audit_path)},
             "source": expected_source}
    write_json_exclusive(directory / "audit.json", audit)
    return frame, audit


def replay(source: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    directory = output / "replay"
    directory.mkdir(exist_ok=False)
    inventory = {}

    def records():
        for raw in iter_matches(source):
            identifier = str(raw["match_id"])
            if identifier in inventory:
                raise ValueError(f"duplicate source match ID {identifier}")
            # Name labels are resolved by exact game IDs, never scores or winners.
            a, b = schema.team1_id(raw), schema.team2_id(raw)
            inventory[identifier] = {
                "golgg_match_id": identifier, "date": raw["date"],
                "team1_id": a, "team2_id": b,
                "team1_name": schema.team1_name(raw), "team2_name": schema.team2_name(raw),
                "tournament": schema.match_tournament(raw), "best_of": schema.best_of(raw),
                "games": len(raw["games"]),
            }
            yield raw

    started = time.monotonic()
    try:
        series, source_audit = _prepare_history(records(), date.max)
    except Exception as exc:
        write_json_exclusive(directory / "failure.json", {
            "stage": "strict_source_validation", "reason": str(exc),
            "source_rows_seen": len(inventory), "transitions_silently_skipped": 0,
        })
        raise
    source_frame = pd.DataFrame(inventory.values())
    source_frame.to_csv(directory / "source_inventory.csv", index=False, mode="x")
    emit(output, "source_validated", series=len(series), maps=sum(len(s.games) for s in series),
         years=source_frame.date.str[:4].value_counts().sort_index().to_dict())
    state, position, rows, exclusions = ChronologicalFeatureState(), 0, [], []
    targets = sorted(series, key=lambda item: (item.start_day, item.order))
    previous_year = None
    for day, grouped in groupby(targets, key=lambda item: item.start_day):
        while position < len(series) and series[position].games[-1].day < day:
            state.consume(series[position])
            position += 1
        if day.year != previous_year:
            emit(output, "replay_year", year=day.year, snapshots=len(rows), released_series=position)
            previous_year = day.year
        for item in grouped:
            meta = inventory[item.identifier]
            reason = None
            if item.target is None:
                reason = "state_only_nonbinary_or_initial_advantage"
            teams = sorted((item.team1, item.team2))
            if reason is None and any(team not in state.latest_rosters for team in teams):
                reason = "no_prior_observed_roster"
            if reason is None:
                rosters = {team: state.latest_rosters[team][1] for team in teams}
                try:
                    row = state.pairs(rosters, day, [item.best_of]).iloc[0].to_dict()
                except ValueError as exc:
                    reason = str(exc)
            if reason is not None:
                exclusions.append({**meta, "reason": reason})
                continue
            # TM has no cutoff-time decay. Match RatingManager's native aggregate
            # without computing all six rating families a second time.
            tm = state.manager.systems["tm"]
            for side, team in enumerate(teams, 1):
                roster = rosters[team]
                row[f"player_tm_sigma_avg{side}"] = sum(tm.get_player_rating(p).sigma for p in roster) / len(roster)
                row[f"roster{side}_ids"] = json.dumps(roster)
                row[f"roster{side}_last_observed_day"] = state.latest_rosters[team][0][0].isoformat()
                if state.latest_rosters[team][0][0] >= day:
                    raise ValueError("target/same-day roster leaked into pre-match snapshot")
            names = {meta["team1_id"]: meta["team1_name"], meta["team2_id"]: meta["team2_name"]}
            if set(names) != set(teams) or not all(names.values()):
                raise ValueError(f"{item.identifier}: missing exact ID-aligned display names")
            if state.last_day >= day or not np.isfinite([row[n] for n in NATIVE_FIELDS]).all():
                raise ValueError("native feature state is nonfinite or not strictly prior-day")
            row.update(
                golgg_match_id=item.identifier, date=day.isoformat(),
                result_day=item.games[-1].day.isoformat(),
                team1_id=teams[0], team2_id=teams[1], team1_name=names[teams[0]], team2_name=names[teams[1]],
                tournament=meta["tournament"], BoN=item.best_of,
                y_true=item.target if item.team1 == teams[0] else 1-item.target,
                history_last_source_day=state.last_day.isoformat(),
            )
            rows.append(row)
    # Validate the final day's transitions too, although no later target uses them.
    while position < len(series):
        state.consume(series[position])
        position += 1
    frame = pd.DataFrame(rows)
    excluded = pd.DataFrame(exclusions)
    if len(frame) + len(excluded) != len(series) or frame.golgg_match_id.duplicated().any():
        raise ValueError("replay failed full-source target accounting")
    if not (frame.history_last_source_day < frame.date).all():
        raise ValueError("source-day temporal integrity violation")
    if (pd.to_datetime(frame.feature_history_max_at, utc=True) > pd.to_datetime(frame.date, utc=True)).any():
        raise ValueError("exclusive history upper bound is after target midnight")
    frame.to_csv(directory / "snapshots.csv", index=False, mode="x")
    excluded.to_csv(directory / "target_exclusions.csv", index=False, mode="x")
    audit = {
        **source_audit, "status": "complete", "feature_version": FEATURE_VERSION,
        "source": {"path": str(source.resolve()), "sha256": sha256(source)},
        "feature_contract_sha256": json_digest(FEATURE_CONTRACT),
        "replay_code_sha256": replay_implementation_hashes(),
        "feature_contract": FEATURE_CONTRACT, "roster_policy": "previous_observed_roster_proxy",
        "native_tm_sigma": "native RatingManager-equivalent TM means at identical prior-day state and roster; no imputation",
        "source_date_min": source_frame.date.min(), "source_date_max": source_frame.date.max(),
        "source_year_counts": source_frame.date.str[:4].value_counts().sort_index().to_dict(),
        "snapshot_year_counts": frame.date.str[:4].value_counts().sort_index().to_dict(),
        "target_exclusion_counts": dict(Counter(excluded.reason)),
        "snapshots": len(frame), "target_exclusions": len(excluded),
        "history_series_consumed": position, "history_maps_consumed": sum(len(s.games) for s in series),
        "source_transitions_rejected_or_skipped": 0,
        "state_only_series": sum(s.target is None for s in series),
        "drawn_series": sum(s.is_draw for s in series),
        "temporal_violations": 0, "availability_certified": False,
        "history_timestamp_semantics": "exclusive next-midnight UTC technical bound of final source day, not certified publication time",
        "runtime_seconds": time.monotonic() - started,
        "outputs": {n: sha256(directory / n) for n in ("snapshots.csv", "target_exclusions.csv", "source_inventory.csv")},
    }
    write_json_exclusive(directory / "audit.json", audit)
    emit(output, "replay_complete", snapshots=len(frame), excluded=len(excluded), seconds=audit["runtime_seconds"])
    return frame, audit


def scores(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    bins = reliability_bins(y, p)
    return {
        "n": len(y), "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == y))),
        "reliability_bins": bins,
    }


def annual_artifact_metadata(protocol: dict, fold: dict, fold_dir: Path) -> dict:
    """Common semantic provenance; eligibility is technical, not availability."""
    return {
        "feature_version": FEATURE_VERSION,
        "feature_algebra_version": "ratings-w20-symmetric-series-v1",
        "feature_contract": FEATURE_CONTRACT,
        "feature_contract_sha256": json_digest(FEATURE_CONTRACT),
        "eligible_from": fold["origin"] + "T00:00:00+00:00",
        "training_end": fold["fit"]["result_day_max"],
        "calibration_end": fold["calibration"]["result_day_max"],
        "prediction_semantics": "direct_series_probability",
        "availability_certified": False,
        "roster_policy": "previous_observed_roster_proxy",
        "temporal_limitations": protocol["limitations"],
        "provenance": {
            "fold": fold, "source": protocol["source"],
            "code_sha256": protocol["code_sha256"],
            "recipe_sha256": sha256(fold_dir / "recipe.json"),
            "replay_audit_sha256": sha256(fold_dir.parent.parent / "replay/audit.json"),
            "feature_columns_sha256": sha256(fold_dir.parent.parent / "feature_columns.json"),
        },
    }


def rating_series_probabilities(frame: pd.DataFrame, *, reverse: bool = False) -> np.ndarray:
    """Player Glicko map probability gets exactly one shared BoN projection."""
    p = frame.player_gl.to_numpy(float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("rating control requires explicit finite player_gl map probabilities")
    if not frame.best_of.isin([1, 3, 5]).all():
        raise ValueError("rating control supports only explicit Bo1/3/5")
    if reverse:
        p = 1 - p
    return np.array([_independent_series_probability(float(value), int(bo))
                     for value, bo in zip(p, frame.best_of)])


def fit_benchmark_controls(frame, masks, scaled, names, scales, result, fold, fold_dir, protocol):
    """Additional recipes consume the very same prepared matrices and fold masks."""
    fit, cal, test = masks
    xt, xc, xe = scaled
    y = frame.y_true.to_numpy(float)
    semantic = annual_artifact_metadata(protocol, fold, fold_dir)
    symmetry, evidence = {}, {}
    linear = LogisticRegression(C=0.1, fit_intercept=False, max_iter=2000,
                                solver="lbfgs", random_state=42)
    linear.fit(xt, y[fit])
    linear_slope = fit_platt_scaling(linear.decision_function(xc), y[cal])
    z = linear_slope * linear.decision_function(xe)
    result["p_linear79"] = np.clip(expit(z), 1e-12, 1-1e-12)
    symmetry["linear79"] = float(np.max(abs(result.p_linear79.to_numpy()
        + expit(linear_slope * linear.decision_function(-xe)) - 1)))
    linear_artifact = {**semantic, "model_version": protocol["models"]["p_linear79"],
        "model": linear, "features": names, "scales": scales,
        "calibration_slope": linear_slope, "training": protocol["config"]["linear79"],
        "series_projection_count": 0}
    with (fold_dir / "linear79.joblib").open("xb") as stream:
        joblib.dump(linear_artifact, stream)
    evidence["linear79"] = {"calibration_slope": linear_slope,
                            "solver_iterations": linear.n_iter_.tolist()}
    cfg = protocol["config"]["exp081_strong"]
    members = []
    for seed in cfg["seeds"]:
        member = train_single_member(xt, y[fit], xc, y[cal], epochs=cfg["epochs"],
            batch_size=cfg["batch_size"], lr=cfg["lr"], gamma=cfg["gamma"],
            d_h1=cfg["d_h1"], d_h2=cfg["d_h2"], seed=seed,
            weight_decay=cfg["weight_decay"], bootstrap_fraction=cfg["bootstrap_fraction"])
        members.append(member)
        emit(fold_dir.parent.parent, "strong_member_fitted", year=fold["year"], seed=seed, adam_steps=member.t)
    member_slopes = [float(m.platt_slope) for m in members]
    cal_z = np.stack([m.forward_anti_symmetric(xc)[0].ravel() * m.platt_slope for m in members]).mean(axis=0)
    test_logits = np.stack([m.forward_anti_symmetric(xe)[0].ravel() * m.platt_slope for m in members])
    reverse_z = np.stack([m.forward_anti_symmetric(-xe)[0].ravel() * m.platt_slope for m in members]).mean(axis=0)
    ensemble_slope = fit_platt_scaling(cal_z, y[cal])
    for suffix, additional_slope in (("strong", 1.0), ("strong_ensemble", ensemble_slope)):
        column = f"p_exp081_{suffix}"
        z = test_logits.mean(axis=0) * additional_slope
        result[column] = np.clip(expit(z), 1e-12, 1-1e-12)
        result[f"exp081_{suffix}_z"] = z
        result[f"exp081_{suffix}_sigma_z"] = test_logits.std(axis=0) * additional_slope
        symmetry[f"exp081_{suffix}"] = float(np.max(abs(result[column].to_numpy()
            + expit(reverse_z * additional_slope) - 1)))
        artifact = build_model_artifact(members, names, scales)
        artifact.update(semantic, model_name=f"EXP081-Report-Informed-{suffix}-Research",
            model_version=protocol["models"][column], training=cfg,
            ensemble_calibration={
                "method": "individual member slopes, then separately declared positive ensemble slope embedded multiplicatively"
                          if suffix == "strong_ensemble" else "individual member slopes; no additional ensemble calibration",
                "slope": additional_slope, "member_slopes_before_ensemble": member_slopes,
                "calibration_end": fold["calibration"]["result_day_max"],
                "calibration_rows_sha256": fold["calibration"]["row_ids_sha256"],
                "source_sha256": sha256(fold_dir.parent.parent / "replay/snapshots.csv"),
            })
        artifact["architecture"]["loss"] = exp081_loss_name(cfg["gamma"])
        for item in artifact["ensemble"]:
            item["platt_slope"] *= additional_slope
        write_json_exclusive(fold_dir / f"exp081_{suffix}.json", artifact)
    evidence["exp081_strong"] = {"member_slopes": member_slopes, "ensemble_slope": ensemble_slope,
        "seeds": cfg["seeds"], "adam_steps": [m.t for m in members],
        "parameter_l2_norms": [float(np.sqrt(sum(np.square(p).sum() for p in m.params))) for m in members]}
    raw_cal = rating_series_probabilities(frame.loc[cal])
    raw_test = rating_series_probabilities(frame.loc[test])
    raw_reverse = rating_series_probabilities(frame.loc[test], reverse=True)
    result["p_glicko"] = raw_test
    calibrated, reverse_calibrated = raw_test.copy(), raw_reverse.copy()
    slopes = {}
    for bo in (1, 3, 5):
        mask = frame.loc[cal, "best_of"].to_numpy() == bo
        target = frame.loc[test, "best_of"].to_numpy() == bo
        if not mask.any() or len(np.unique(y[cal][mask])) != 2:
            raise ValueError(f"{fold['year']}: Bo{bo} requires both outcomes on earlier calibration rows")
        slopes[str(bo)] = fit_platt_scaling(logit(raw_cal[mask]), y[cal][mask])
        calibrated[target] = expit(slopes[str(bo)] * logit(raw_test[target]))
        reverse_calibrated[target] = expit(slopes[str(bo)] * logit(raw_reverse[target]))
    result["p_glicko_format"] = np.clip(calibrated, 1e-12, 1-1e-12)
    symmetry["glicko"] = float(np.max(abs(raw_test + raw_reverse - 1)))
    symmetry["glicko_format"] = float(np.max(abs(result.p_glicko_format.to_numpy() + reverse_calibrated - 1)))
    for column in ("p_glicko", "p_glicko_format"):
        write_json_exclusive(fold_dir / f"{column[2:]}.json", {
            **semantic, "model_version": protocol["models"][column],
            "map_probability_field": "player_gl", "series_projection_count": 1,
            "map_clip": [0.001, 0.999], "calibration_slopes": slopes if column.endswith("format") else None,
            "calibration": "positive zero-intercept series-logit slope per BoN, previous full calendar year"
                           if column.endswith("format") else "none",
        })
    evidence["glicko_format"] = {"slopes": slopes}
    for column, version in protocol["models"].items():
        result[f"{column[2:]}_model_version"] = version
    if max(symmetry.values()) > 1e-8:
        raise ValueError("matched control side symmetry failed")
    fold["symmetry_max_error"].update(symmetry)
    fold["benchmark_fit_evidence"] = evidence


def benchmark_slices(frame: pd.DataFrame) -> tuple[dict, dict]:
    """Only supported observed slices; no invented roster counts or league labels."""
    masks = {"full_history": np.ones(len(frame), dtype=bool), "2024_plus": frame.date >= "2024-01-01"}
    for column in ("best_of", "fold_year", "competition_tier", "competition_family"):
        if column in frame:
            for value in sorted(frame[column].dropna().unique()):
                masks[f"{column}={value}"] = frame[column] == value
    p = frame.p_exp039
    masks.update(confidence_exp039_heavy_favorite=p >= .75,
                 confidence_exp039_moderate_favorite=(p >= .60) & (p < .75),
                 confidence_exp039_tossup=p.between(.45, .55), confidence_exp039_underdog=p <= .40)
    if {"days_since_last_1", "days_since_last_2"} <= set(frame):
        masks["rest_either_ge30days"] = frame[["days_since_last_1", "days_since_last_2"]].max(axis=1) >= 30
    if {"player_gl", "team_gl"} <= set(frame):
        disagreement = abs(frame.player_gl - frame.team_gl)
        masks["glicko_player_team_disagreement_le008"] = disagreement <= .08
        masks["glicko_player_team_disagreement_gt015"] = disagreement > .15
    for name in ("glicko_rd", "trueskill_sigma"):
        flag = f"{name}_high_earlier_q75"
        if flag in frame:
            masks[f"{name}_high_train_q75"] = frame[flag].astype(bool)
            masks[f"{name}_below_train_q75"] = ~frame[flag].astype(bool)
    if "roster_min_prior_series" in frame:
        prior = frame.roster_min_prior_series
        masks.update(rookie_no_prior_series=prior == 0, rookie_1_to4=prior.between(1, 4),
                     roster_prior5_to19=prior.between(5, 19), roster_prior20plus=prior >= 20,
                     roster_all_prior_ge10=prior >= 10, roster_any_prior_lt10=prior < 10)
    unsupported = {name: "not present in audited replay; no inference or imputation" for name in
                   ("roster_min_prior_series", "competition_tier", "competition_family") if name not in frame}
    return masks, unsupported


def score_benchmark(result: pd.DataFrame, output: Path, protocol: dict) -> dict:
    models = list(protocol["models"])
    masks, unsupported = benchmark_slices(result)
    metrics, paired, reliability = [], [], []
    for label, mask in masks.items():
        part = result.loc[mask]
        for model in models:
            metrics.append({"cohort": label, "model": model, **probability_metrics(part.y_true, part[model])})
            if len(part):
                reliability.extend({"cohort": label, "model": model, **item}
                                   for item in reliability_bins(part.y_true, part[model]))
        for candidate, reference in protocol["comparisons"]:
            item = {"cohort": label, "candidate": candidate, "reference": reference, "n": len(part),
                    "direction": "candidate minus reference; negative favors candidate", "seed": 82}
            if pd.to_datetime(part.date).dt.to_period("M").nunique() < 2:
                paired.append({**item, "status": "unavailable_fewer_than_two_months"})
                continue
            y = part.y_true.to_numpy(float)
            for metric, delta in (
                ("log_loss", binary_log_loss_vector(y, part[candidate]) - binary_log_loss_vector(y, part[reference])),
                ("brier", (part[candidate].to_numpy()-y)**2 - (part[reference].to_numpy()-y)**2),
            ):
                paired.append({**item, "metric": metric, "status": "nominal_descriptive_retrospective",
                               **monthly_bootstrap(delta, part.date, 5000)})
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False, mode="x")
    pd.DataFrame(paired).to_csv(output / "paired_monthly.csv", index=False, mode="x")
    pd.DataFrame(reliability).to_csv(output / "reliability.csv", index=False, mode="x")
    report = {"version": BENCHMARK_VERSION, "status": "retrospective_research_not_promotion",
        "overall": {r["model"]: r for r in metrics if r["cohort"] == "full_history"},
        "2024_plus": {r["model"]: r for r in metrics if r["cohort"] == "2024_plus"},
        "unsupported_slices": unsupported, "metrics": metrics, "paired_monthly": paired,
        "multiplicity": "nominal descriptive intervals; no selection, superiority gate or promotion"}
    write_json_exclusive(output / "benchmark_summary.json", report)
    return report


def fit_folds(frame: pd.DataFrame, output: Path, protocol: dict) -> pd.DataFrame:
    x39, names39, rank = build_exp039_features(frame)
    x81, names81 = build_training_features(frame)
    metadata = json.loads((ROOT / "docs/assets/final_symmetric_calibrated_market_comparison/sym_cal_lr_elasticnet_w20_binomial_metadata.json").read_text())
    if names39 != metadata["features"] or x81.shape[1] != 79:
        raise ValueError("original46/canonical79 reference schema mismatch")
    write_json_exclusive(output / "feature_columns.json", {"exp039": names39, "exp081": names81})
    feature_evidence = {
        "feature_contract_sha256": json_digest(FEATURE_CONTRACT),
        "columns_sha256": sha256(output / "feature_columns.json"),
        "native46_matrix_sha256": hashlib.sha256(np.ascontiguousarray(x39[names39].to_numpy(float)).tobytes()).hexdigest(),
        "canonical79_matrix_sha256": hashlib.sha256(np.ascontiguousarray(x81).tobytes()).hexdigest(),
        "snapshot_sha256": sha256(output / "replay/snapshots.csv"),
        "code_sha256": protocol["code_sha256"],
    }
    write_json_exclusive(output / "fit_input_manifest.json", feature_evidence)
    helper = load_module_from_path(ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py", "full_historical_lr")
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=False)
    y = frame.y_true.to_numpy(float)
    blocks, folds = [], []
    for year in range(2020, 2027):
        started = time.monotonic()
        fold_dir = models_dir / str(year)
        fold_dir.mkdir(exist_ok=False)
        fit, cal = chronological_fold_masks(frame, origin=f"{year}-01-01T00:00:00+00:00",
                                             calibration_start=f"{year-1}-01-01", fit_start=frame.date.min())
        test = ((frame.date >= f"{year}-01-01") & (frame.date < f"{year+1}-01-01")).to_numpy()
        if any(not mask.any() for mask in (fit, cal, test)) or any(len(np.unique(y[m])) != 2 for m in (fit, cal)):
            raise ValueError(f"{year}: missing source prerequisites for annual fit/calibration/test")
        if np.any(fit & cal) or np.any((fit | cal) & test):
            raise ValueError("annual partitions overlap")
        memberships, fold = {}, {"year": year, "origin": f"{year}-01-01", "calibration_start": f"{year-1}-01-01"}
        for label, mask in (("fit", fit), ("calibration", cal), ("test", test)):
            part = frame.loc[mask]
            memberships[label] = part.golgg_match_id.tolist()
            fold[label] = {"n": len(part), "date_min": part.date.min(), "date_max": part.date.max(),
                           "result_day_max": part.result_day.max(), "row_ids_sha256": json_digest(memberships[label])}
        if fold["fit"]["result_day_max"] >= fold["calibration_start"] or fold["calibration"]["result_day_max"] >= fold["origin"]:
            raise ValueError("fit/calibration labels were unavailable at annual boundary")
        write_json_exclusive(fold_dir / "row_memberships.json", memberships)
        write_json_exclusive(fold_dir / "recipe.json", {**fold, "config": protocol["config"], "models": protocol["models"],
                                                        "replay_audit_sha256": sha256(output / "replay/audit.json"),
                                                        "input_manifest_sha256": sha256(output / "fit_input_manifest.json"),
                                                        "selection_partition": None, "early_stopping_partition": None,
                                                        "selection_policy": "fixed recipe and epoch count; no row-driven selection"})
        emit(output, "fold_started", year=year, fit=fold["fit"], calibration=fold["calibration"], test=fold["test"])
        model = helper.build_logistic_regression()
        model.fit(x39.loc[fit, names39], y[fit])
        raw_cal39 = exp039_probability(model, x39.loc[cal], names39, rank)
        slope39 = fit_platt_scaling(logit(raw_cal39), y[cal])
        raw39 = exp039_probability(model, x39.loc[test], names39, rank)
        p39 = np.clip(expit(slope39 * logit(raw39)), 1e-12, 1 - 1e-12)
        flipped = swap_orientation(x39.loc[test], names39, rank, np.ones(test.sum(), dtype=bool))
        reverse39 = np.clip(expit(slope39 * logit(exp039_probability(model, flipped, names39, rank))), 1e-12, 1 - 1e-12)
        with (fold_dir / "exp039.joblib").open("xb") as stream:
            joblib.dump({**annual_artifact_metadata(protocol, fold, fold_dir),
                         "model_version": protocol["models"]["p_exp039"], "pipeline": model, "features": names39,
                         "input_semantics": "native46; requires same-state player_tm_sigma_avg1/2",
                         "training": protocol["config"]["exp039"], "series_projection_count": 0,
                         "rank_probability_features": rank, "calibration_slope": slope39,
                         "symmetry": "symmetrize probabilities before zero-intercept positive Platt slope",
                         "fold": fold}, stream)
        fold["exp039_fit_evidence"] = {
            "solver_iterations": model.named_steps["model"].n_iter_.tolist(),
            "coefficient_l2_norm": float(np.linalg.norm(model.named_steps["model"].coef_)),
            "scaler_fit_rows": int(model.named_steps["scaler"].n_samples_seen_),
            "calibration_slope": slope39,
        }
        scales = x81[fit].std(axis=0)
        scales[scales == 0] = 1
        scaled_fit, scaled_cal, scaled_test = x81[fit] / scales, x81[cal] / scales, x81[test] / scales
        members, fit_evidence = [], []
        for index, seed in enumerate(SEEDS):
            member_started = time.monotonic()
            member = train_single_member(scaled_fit, y[fit], scaled_cal, y[cal],
                                         epochs=protocol["config"]["exp081"]["epochs"], seed=seed,
                                         batch_size=64, lr=0.003, gamma=protocol["config"]["exp081"]["gamma"], d_h1=32, d_h2=16,
                                         weight_decay=protocol["config"]["exp081"]["weight_decay"],
                                         bootstrap_fraction=protocol["config"]["exp081"]["bootstrap_fraction"])
            members.append(member)
            evidence = {"member": index, "seed": seed, "adam_steps": member.t,
                        "parameter_l2_norm": float(np.sqrt(sum(np.square(p).sum() for p in member.params))),
                        "seconds": time.monotonic() - member_started}
            fit_evidence.append(evidence)
            emit(output, "exp081_member_fitted", year=year, **evidence)
        raw_cal81 = np.stack([m.forward_anti_symmetric(scaled_cal)[0].ravel() for m in members]).mean(axis=0)
        slope81 = fit_platt_scaling(raw_cal81, y[cal])
        for member in members:
            member.platt_slope = slope81
        logits = np.stack([m.forward_anti_symmetric(scaled_test)[0].ravel() * slope81 for m in members])
        reverse_logits = np.stack([m.forward_anti_symmetric(-scaled_test)[0].ravel() * slope81 for m in members])
        z, sigma = logits.mean(axis=0), logits.std(axis=0, ddof=0)
        p81 = np.clip(expit(z), 1e-12, 1 - 1e-12)
        reverse81 = np.clip(expit(reverse_logits.mean(axis=0)), 1e-12, 1 - 1e-12)
        symmetry = {"exp039": float(np.max(abs(p39 + reverse39 - 1))), "exp081": float(np.max(abs(p81 + reverse81 - 1)))}
        if max(symmetry.values()) > 1e-8:
            raise ValueError("annual model side symmetry failed")
        fold.update(exp081_fit_evidence=fit_evidence, exp081_calibration_slope=slope81, symmetry_max_error=symmetry)
        artifact = build_model_artifact(members, names81, scales)
        artifact.update(annual_artifact_metadata(protocol, fold, fold_dir),
                        model_name="EXP081-Prior-Day-Annual-Research", model_version=protocol["models"]["p_exp081"],
                        training=protocol["config"]["exp081"],
                        ensemble_calibration={"method": "positive slope on mean raw ensemble logit, embedded identically in members",
                            "slope": slope81, "calibration_end": fold["calibration"]["result_day_max"],
                            "calibration_rows_sha256": fold["calibration"]["row_ids_sha256"],
                            "source_sha256": feature_evidence["snapshot_sha256"]})
        artifact["architecture"]["loss"] = exp081_loss_name(protocol["config"]["exp081"]["gamma"])
        write_json_exclusive(fold_dir / "exp081.json", artifact)
        result = frame.loc[test].copy()
        result["p_exp039"], result["p_exp081"], result["fold_year"] = p39, p81, year
        result["exp039_model_version"], result["exp081_model_version"] = protocol["models"]["p_exp039"], protocol["models"]["p_exp081"]
        result["exp081_sigma_z"], result["exp081_z"] = sigma, z
        result["p_exp039_uncalibrated_symmetric"] = raw39
        if protocol.get("benchmark"):
            fit_benchmark_controls(frame, (fit, cal, test), (scaled_fit, scaled_cal, scaled_test),
                                   names81, scales, result, fold, fold_dir, protocol)
            fold["diagnostic_thresholds"] = {}
            for name, columns in {
                "glicko_rd": ["team_gl_rd1", "team_gl_rd2", "player_gl_rd_avg1", "player_gl_rd_avg2"],
                "trueskill_sigma": ["team_ts_sigma1", "team_ts_sigma2", "player_ts_sigma_avg1", "player_ts_sigma_avg2"],
            }.items():
                values = frame[columns].mean(axis=1)
                threshold = float(values.loc[fit].quantile(.75))
                result[f"{name}_high_earlier_q75"] = values.loc[test].to_numpy() >= threshold
                fold["diagnostic_thresholds"][name] = {"train_q75": threshold, "fit_rows_sha256": fold["fit"]["row_ids_sha256"]}
        result = result[OUTPUT_COLUMNS + [c for c in result if c not in OUTPUT_COLUMNS]]
        result.to_csv(fold_dir / "predictions.csv", index=False, mode="x")
        fold["metrics"] = {"exp039": scores(y[test], p39), "exp081": scores(y[test], p81),
                           "coinflip": scores(y[test], np.full(test.sum(), 0.5)),
                           "fit_prevalence": scores(y[test], np.full(test.sum(), y[fit].mean()))}
        fold["calibration_metrics"] = {"exp039": scores(y[cal], np.clip(expit(slope39 * logit(raw_cal39)), 1e-12, 1 - 1e-12)),
                                       "exp081": scores(y[cal], np.clip(expit(slope81 * raw_cal81), 1e-12, 1 - 1e-12))}
        if protocol.get("benchmark"):
            fold["benchmark_metrics"] = {column: probability_metrics(y[test], result[column]) for column in protocol["models"]}
        fold["runtime_seconds"] = time.monotonic() - started
        fold["artifacts"] = {p.name: sha256(p) for p in sorted(fold_dir.iterdir()) if p.is_file()}
        write_json_exclusive(fold_dir / "completed.json", {"status": "complete", **fold})
        blocks.append(result)
        folds.append(fold)
        emit(output, "fold_complete", year=year, n=len(result), seconds=fold["runtime_seconds"],
             exp039_logloss=fold["metrics"]["exp039"]["log_loss"], exp081_logloss=fold["metrics"]["exp081"]["log_loss"])
    result = pd.concat(blocks, ignore_index=True).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    result.to_csv(output / "predictions.csv", index=False, mode="x")
    write_json_exclusive(output / "folds.json", {"folds": folds})
    write_json_exclusive(output / "model_metrics.json", {
        "overall": {"exp039": scores(result.y_true, result.p_exp039), "exp081": scores(result.y_true, result.p_exp081)},
        "by_year": {str(f["year"]): f["metrics"] for f in folds},
        "by_best_of": {str(bo): {"exp039": scores(p.y_true, p.p_exp039), "exp081": scores(p.y_true, p.p_exp081)}
                       for bo, p in result.groupby("best_of")},
    })
    if protocol.get("benchmark"):
        score_benchmark(result, output, protocol)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/artifacts/full-historical-predictions-20260909")
    parser.add_argument("--exp081-gamma", type=float, default=1.0,
                        help="Fixed focal-loss gamma; 0 is BCE. Creates a separately versioned research derivative.")
    parser.add_argument("--benchmark", action="store_true", help="Fixed matched base-recipe retrospective benchmark; never promotion")
    parser.add_argument("--reuse-replay", type=Path, help="Audited replay directory with matching source, contract and code hashes")
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    exp081_version = exp081_model_version(args.exp081_gamma)
    if args.benchmark and args.exp081_gamma != 1.0:
        raise ValueError("matched benchmark fixes gamma=1; gamma ablations use the legacy non-benchmark mode")
    if args.benchmark and not any(arg == "--output" or arg.startswith("--output=") for arg in sys.argv):
        raise ValueError("matched benchmark requires an explicit new --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("protocol.json", "replay", "models", "predictions.csv")):
        raise ValueError("refusing to overwrite an existing full historical model run")
    started = time.monotonic()
    random.seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    protocol = {
        "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "code_sha256": implementation_hashes(), "frozen_artifacts": frozen_hashes(),
        "models": {"p_exp039": EXP039, "p_exp081": exp081_version},
        "config": {
            "years": list(range(2020, 2027)), "fold": "fit all history completed before previous Jan1; calibrate previous calendar year; test current year",
            "selection": "fixed existing recipes; no tuning, checkpoint selection or transformations learned from test",
            "exp039": {"features": 46, "C": 0.03297234640536737, "penalty": "elasticnet", "l1_ratio": 0.9439657999531195,
                       "solver": "saga", "max_iter": 5000, "seed": 42, "preprocessing": "train-only median imputer and StandardScaler",
                       "calibration": "positive zero-intercept slope fitted to symmetric calibration probabilities",
                       "derivative_changes": "strict prior-day previous-roster history; all-source rather than odds-filtered population; annual partitions; slope-only previous-year calibration instead of frozen intercept OOF Platt"},
            "exp081": {"features": 79, "epochs": 35, "batch_size": 64, "lr": 0.003, "gamma": args.exp081_gamma, "d_h1": 32, "d_h2": 16,
                       "weight_decay": 0.0001, "bootstrap_fraction": 1.0,
                       "regularization_objective": "mean focal loss + weight_decay/2 * sum(weight matrices squared); biases excluded",
                       "bootstrap": "seeded RandomState replacement draw max(1,floor(N*fraction)); fixed sample across epochs",
                       "seeds": SEEDS, "members": 5, "preprocessing": "train-only standard deviation without centering; zero scale becomes1",
                       "calibration": "positive slope on mean raw ensemble logit; same slope embedded in all members",
                       "selection": "fixed epochs; no early stopping or test-driven selection"},
        },
        "feature_contract": FEATURE_CONTRACT,
        "feature_contract_sha256": json_digest(FEATURE_CONTRACT),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "trueskill", "openskill", "glicko2")},
                        "blas_threads": 1, "cpu_count": os.cpu_count(), "processor": platform.processor(),
                        "thread_environment": {name: os.environ.get(name) for name in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                        "installed_distributions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}},
        "limitations": ["retrospective annual reconstructions, not contemporaneously captured forecasts",
                        "source release timestamps, roster announcements and global collection completeness are uncertified",
                        "recovered source retains 309 unresolved discovered series outside the supplied projection",
                        "fixed legacy recipe hyperparameters were selected in later research; this is not historically untouched model selection",
                        "frozen EXP039 is unchanged and not reused backward; these are explicitly named research derivatives"],
        "production_mutation": False,
    }
    protocol["benchmark"] = args.benchmark
    protocol["reuse_replay"] = str(args.reuse_replay.resolve()) if args.reuse_replay else None
    if args.benchmark:
        protocol.update(version=BENCHMARK_VERSION, models=BENCHMARK_MODELS,
            question="Does the report-informed stronger recipe compete on identical complete prior-roster annual rows?",
            status="predeclared_retrospective_research_not_original_reproduction_or_promotion",
            evaluation={"primary_metric": "series log_loss", "cohorts": ["2020..2026 full history", "2024+"],
                        "bootstrap": "paired monthly blocks, 5000 replicates, seed82; all eligible sports rows, no market cohort",
                        "slices": "year, BoN, native46 confidence, rest, Glicko disagreement, train-q75 rating uncertainty; roster/tier/family only when present",
                        "calibration_diagnostics": "10 equal-width bins; unconstrained slope/intercept diagnostic only, never applied to predictions",
                        "brier_decomposition": "binned reliability-resolution+uncertainty with explicit binning residual",
                        "inference": "nominal descriptive intervals; no candidate selected or promoted"},
            comparisons=[(name, "p_exp039") for name in BENCHMARK_MODELS if name != "p_exp039"] + [
                ("p_exp081_strong", "p_exp081"), ("p_exp081_strong_ensemble", "p_exp081_strong"),
                ("p_glicko_format", "p_glicko"), ("p_exp081_strong", "p_linear79")])
        protocol["config"]["linear79"] = {
            "features": 79, "C": .1, "fit_intercept": False, "solver": "lbfgs", "max_iter": 2000,
            "seed": 42, "tol": 1e-4, "preprocessing": "same train-only std and no centering as Siamese",
            "calibration": "prior-year positive zero-intercept slope"}
        protocol["config"]["exp081_strong"] = {
            **protocol["config"]["exp081"], "weight_decay": .01, "bootstrap_fraction": .8,
            "seeds": STRONG_SEEDS,
            "calibration": "individual member positive slopes on previous full year; sigmoid(mean calibrated logits)",
            "ensemble_comparison": "separate additional positive slope on mean member-calibrated logit, same previous full year; embedded multiplicatively",
            "limitations": "report-informed joint L2/bagging/seeds/calibration derivative; original epochs/checkpoint/L2 normalization unknown"}
        protocol["config"]["glicko"] = {"input": "player_gl map probability", "map_clip": [.001, .999],
            "projection": "shared independent-series Bo1/3/5 conversion exactly once",
            "calibrated_control": "separate positive zero-intercept series-logit slope per format; all previous-year rows grouped by BoN",
            "missing_format_policy": "fail closed if any supported format lacks both calibration outcomes"}
        manifest = ROOT / "data/artifacts/corrected-historical-reruns-20260908/reported-recipe081/manifest.json"
        protocol["report_informed_recipe_source"] = {"path": str(manifest), "sha256": sha256(manifest)}
    with threadpool_limits(limits=1):
        protocol["environment"]["threadpools_during_fit"] = threadpool_info()
    write_json_exclusive(args.output / "protocol.json", protocol)
    try:
        with threadpool_limits(limits=1):
            frame, audit = (reuse_replay(args.source, args.reuse_replay.resolve(), args.output)
                            if args.reuse_replay else replay(args.source, args.output))
            result = fit_folds(frame, args.output, protocol)
        expected = frame.loc[(frame.date >= "2020-01-01") & (frame.date < "2027-01-01"), "golgg_match_id"]
        if result.golgg_match_id.duplicated().any() or set(expected) != set(result.golgg_match_id):
            raise ValueError("predictions do not cover every eligible 2020..2026 source target")
        if not np.isfinite(result[list(protocol["models"])].to_numpy()).all():
            raise ValueError("nonfinite exported probabilities")
        if sha256(args.source) != protocol["source"]["sha256"] or frozen_hashes() != protocol["frozen_artifacts"]:
            raise ValueError("source/frozen artifacts changed during experiment")
        if implementation_hashes() != protocol["code_sha256"]:
            raise ValueError("experiment implementation changed while executing")
        complete = {"status": "complete", "source_series": audit["raw_series"], "predicted_series": len(result),
                    "predicted_date_min": result.date.min(), "predicted_date_max": result.date.max(),
                    "all_eligible_2020_plus_predicted": True, "temporal_violations": 0,
                    "frozen_artifacts_unchanged": True, "source_unchanged": True,
                    "runtime_seconds": time.monotonic() - started,
                    "outputs": {n: sha256(args.output / n) for n in ("predictions.csv", "protocol.json", "feature_columns.json", "folds.json", "model_metrics.json", "replay/audit.json")}}
        if args.benchmark:
            complete["outputs"].update({name: sha256(args.output / name) for name in
                ("fit_input_manifest.json", "benchmark_summary.json", "metrics.csv", "paired_monthly.csv", "reliability.csv")})
        write_json_exclusive(args.output / "completed.json", complete)
        emit(args.output, "complete", **complete)
    except Exception as exc:
        write_json_exclusive(args.output / "failure.json", {"status": "failed", "reason": str(exc),
                             "runtime_seconds": time.monotonic() - started, "completed_folds_preserved": True})
        raise


if __name__ == "__main__":
    main()
