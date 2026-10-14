#!/usr/bin/env python3
"""EXP081 research-only OPEN Benter distillation and sports residual diagnostics.

No production model is loaded for inference or changed. All three students retain
all eligible sports outcomes; only the two auxiliary arms use 2025H1 teachers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_siamese_architectures import probability_metrics
from scripts.corrected_history_evaluation import frozen_hashes, sha256
from scripts.experiment_exp081_market_calibration import (
    EPS, METADATA, MIN_EV, RESAMPLES, SEED, TAX, checked_probabilities,
    code_hashes, design, load_aligned, paired_uncertainty, phase_frame, policy,
    selected_uncertainty, write_json,
)
from scripts.experiment_oddsfree_distillation import predict_artifact, stages, teacher_masks
from scripts.oddsfree_distillation import fit_auxiliary_student
from scripts.train_and_tune_siamese_series import build_training_features, fit_platt_scaling
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS

VERSION = "exp081-open-benter-sports-distillation-v1"
STUDENTS = ("outcome_student", "market_student", "benter_student")
MODELS = ("raw_sports", "benter", *STUDENTS)
IDENTITIES = {key: "string" for key in ("golgg_match_id", "team1_id", "team2_id")}
GROUP_FIELDS = ("best_of", "competition_tier", "roster_min_prior_series")
LIMITATIONS = [
    "Previously inspected retrospective history; 2026 is chronological, not an untouched holdout.",
    "Calendar-day pre-result replay is not certified point-in-time availability.",
    "First-map rosters are observed from games; pre-match announcement timestamps unavailable.",
    "Sports source ingestion, fixture/format publication, prediction and STS quote timestamps unavailable.",
    "Collection completeness and rejected source transitions remain unverified.",
    "Monthly bootstrap intervals are diagnostics, not guarantees; no multiplicity correction or refitting.",
    "Residual subgroup associations are descriptive, not causal or feature-selection evidence.",
    "OPEN Benter and opening market are training labels only; student inference is sports-only.",
    "SERIES outputs must not be expanded again as independent map probabilities.",
    "No promotion, staking, production adoption, or live eligibility is authorized.",
]
FEATURE_AUDIT = {
    "builder": "scripts.train_and_tune_siamese_series.build_training_features -> src.models.symmetric_series.build_feature_mapping",
    "source": "corrected039081 replay only; no align_legacy_context or legacy snapshot reuse",
    "sports_inputs": "six team/player rating families and uncertainties; weak/peak player ratings; rest-day differences; strictly earlier-date W20 team statistics; explicit best_of",
    "market_exclusion": "explicit _REQUIRED_BASE_FIELDS plus best_of projection; no odds, bookmaker, market, outcomes, identities, tournament, tier or roster-prior count enter inference",
    "format": "Existing rating-logit Bo3/Bo5 interactions and independent-series amplification can express format-associated residuals; no new features added.",
    "competition_tier": "Existing tournament-name/date taxonomy is diagnostic metadata, not a canonical79 feature. Rating strengths, uncertainties and W20 team performance may covary with competition tier; association is not causal.",
    "rookie_prior": "roster_min_prior_series is the minimum among the ten observed first-map roster players of their distinct prior series appearances in the corrected source, across all map participants. Counts update only after every snapshot on a date. Existing split: <10 rookie, >=10 stable; it is not team history length or debut age.",
    "rookie_explanation": "Player rating uncertainties, weakest Elo/peak Glicko, team/player logits and W20 team history may capture parts of low-prior-roster uncertainty. The prior-count diagnostic is not an input and first-map roster availability is uncertified.",
    "unavailable_timestamps": ["pre-match roster announcement", "fixture/format publication", "sports source ingestion", "exact event start", "prediction creation", "opening quote observation/availability"],
    "diagnostic_feature_summaries": ["rating_consensus_logit", "team_glicko_rd_diff", "player_glicko_rd_diff", "player_ts_sigma_diff", "w20_win_rate_diff", "w20_gd15_diff", "rest_days_diff"],
}


def paired_scores(frame, model, reference):
    y = frame.y_true.to_numpy(float)
    p = np.clip(frame[model].to_numpy(float), EPS, 1 - EPS)
    q = np.clip(frame[reference].to_numpy(float), EPS, 1 - EPS)
    return {
        "n": len(frame), "direction": "model minus reference; negative favors model",
        "log_loss": paired_uncertainty(binary_log_loss_vector(y, p) - binary_log_loss_vector(y, q), frame.date),
        "brier": paired_uncertainty((p - y) ** 2 - (q - y) ** 2, frame.date),
    }


def selected_residuals(frame, model, selected, side_a):
    p, y = frame[model].to_numpy(float), frame.y_true.to_numpy(float)
    pp, won = np.where(side_a, p, 1 - p), np.where(side_a, y, 1 - y)
    price = np.where(side_a, frame.odds_a, frame.odds_b)
    probability = pp - won
    payoff = (1 - TAX) * price * probability
    result = {
        "eligible_n": len(frame), "selected_n": int(selected.sum()),
        "uncertainty": selected_uncertainty(frame, selected, probability, payoff),
    }
    if selected.any():
        result.update(
            mean_probability=float(pp[selected].mean()), win_fraction=float(won[selected].mean()),
            probability_optimism=float(probability[selected].mean()),
            odds_weighted_return_optimism=float(payoff[selected].mean()),
            estimated_unit_payoff=float(((1 - TAX) * price[selected] * pp[selected] - 1).mean()),
            retrospective_realized_unit_payoff=float(((1 - TAX) * price[selected] * won[selected] - 1).mean()),
        )
    rows = frame[[*METADATA, "phase", "year", "odds_a", "odds_b"]].copy()
    rows["model"], rows["selected"], rows["side_a"] = model, selected, side_a
    rows["selected_probability"], rows["selected_outcome"] = pp, won
    rows["probability_residual"], rows["payoff_residual"] = probability, payoff
    return result, rows


def evaluate(frame, models, *, priced):
    result, decisions = {}, []
    if priced:
        fixed_selected, fixed_side = policy(frame.raw_sports, frame.odds_a, frame.odds_b)
    for model in models:
        p, y = frame[model].to_numpy(float), frame.y_true.to_numpy(float)
        record = {
            "proper_scores": probability_metrics(y, np.clip(p, EPS, 1 - EPS)),
            "original_side_probability_residual": paired_uncertainty(p - y, frame.date),
            "paired": {reference: paired_scores(frame, model, reference)
                       for reference in ("outcome_student", "raw_sports", "benter")
                       if reference in models and reference != model},
        }
        if priced:
            own_selected, own_side = policy(p, frame.odds_a, frame.odds_b)
            for selection, mask, side in (
                ("fixed_original_raw_sports", fixed_selected, fixed_side),
                ("own_max_ev_side", own_selected, own_side),
            ):
                record[selection], rows = selected_residuals(frame, model, mask, side)
                rows["selection_policy"] = selection
                decisions.append(rows)
        result[model] = record
    return result, decisions


def group_masks(frame):
    # Existing definitions only; no search for cutpoints or outcome-based groups.
    masks = {f"format_bo{bo}": frame.best_of.eq(bo).to_numpy() for bo in (1, 3, 5)}
    masks.update({f"competition_tier_{tier}": frame.competition_tier.eq(tier).to_numpy()
                  for tier in sorted(frame.competition_tier.unique())})
    masks["roster_rookie_prior_lt10"] = frame.roster_min_prior_series.lt(10).to_numpy()
    masks["roster_stable_prior_ge10"] = frame.roster_min_prior_series.ge(10).to_numpy()
    return masks


def residual_groups(full, common, x_test, feature_names):
    records = []
    diagnostic_indices = [feature_names.index(name) for name in FEATURE_AUDIT["diagnostic_feature_summaries"]]
    for cohort_name, frame, models in (("full_sports_2026", full, ("raw_sports", *STUDENTS)),
                                        ("open_common_2026", common, MODELS)):
        for group, mask in group_masks(frame).items():
            subset = frame.loc[mask].reset_index(drop=True)
            if subset.empty:
                records.append({"cohort": cohort_name, "group": group, "n": 0})
                continue
            metrics, _ = evaluate(subset, models, priced=cohort_name == "open_common_2026")
            positions = full.index[full.golgg_match_id.isin(subset.golgg_match_id)].to_numpy()
            means = np.mean(np.abs(x_test[np.ix_(positions, diagnostic_indices)]), axis=0)
            records.append({
                "cohort": cohort_name, "group": group, "n": len(subset), "models": metrics,
                "mean_absolute_existing_feature": dict(zip(FEATURE_AUDIT["diagnostic_feature_summaries"], map(float, means))),
            })
    return records


def load_teacher(args, opening, inputs, frozen):
    teacher_protocol = json.loads((args.teacher_dir / "protocol.json").read_text())
    teacher_summary = json.loads((args.teacher_dir / "summary.json").read_text())
    if teacher_summary.get("status") != "complete_exploratory" or not teacher_summary.get("immutable_hashes_verified"):
        raise ValueError("teacher run did not complete immutable-input verification")
    for path in (args.snapshots, args.predictions, args.odds):
        if teacher_protocol["input_hashes"].get(str(path)) != inputs[str(path)]:
            raise ValueError(f"teacher input ancestry mismatch: {path}")
    if teacher_protocol["frozen_hashes"] != frozen:
        raise ValueError("teacher frozen-model ancestry mismatch")
    all_rows = pd.read_csv(args.teacher_dir / "oos_predictions.csv", dtype=IDENTITIES)
    if all_rows.duplicated(["phase", "golgg_match_id"]).any():
        raise ValueError("duplicate teacher event within phase")
    teacher = all_rows.loc[all_rows.phase.eq("open")].copy()
    expected = opening.loc[opening.date.ge("2025-01-01") & opening.date.lt("2027-01-01")]
    if set(teacher.golgg_match_id) != set(expected.golgg_match_id):
        raise ValueError("teacher OPEN population differs from immutable phase-specific input")
    joined = expected.merge(teacher, on="golgg_match_id", suffixes=("", "__teacher"), validate="one_to_one")
    for field in METADATA[1:]:
        if not joined[field].eq(joined[field + "__teacher"]).all():
            raise ValueError(f"teacher label/time/side invariant failed: {field}")
    for field in ("sports", "market", "odds_a", "odds_b"):
        np.testing.assert_allclose(joined[field], joined[field + "__teacher"], atol=1e-12, rtol=0)
    if not teacher.year.eq(pd.to_datetime(teacher.date).dt.year).all():
        raise ValueError("teacher year/date mismatch")
    fits = json.loads((args.teacher_dir / "fits.json").read_text())
    ancestry = []
    for year in (2025, 2026):
        records = [fit for fit in fits if fit["phase"] == "open" and fit["model"] == "benter" and fit["test_year"] == year]
        if len(records) != 1:
            raise ValueError(f"expected one OPEN Benter fit for {year}")
        fit = records[0]
        train = opening.loc[opening.date.lt(f"{year}-01-01")]
        test = teacher.loc[teacher.year.eq(year)]
        counts = {str(k): int(v) for k, v in train.date.str[:4].value_counts().sort_index().items()}
        if (fit["train_year_counts"] != counts or (year == 2025 and set(counts) != {"2024"})
                or fit["train_n"] != len(train) or fit["test_n"] != len(test)
                or fit["train_date_min"] != train.date.min() or fit["train_date_max"] != train.date.max()
                or fit["test_date_min"] != test.date.min() or fit["test_date_max"] != test.date.max()
                or fit["train_positive_n"] != int(train.y_true.sum())
                or fit["test_positive_n"] != int(test.y_true.sum())
                or fit["intercept"] != 0 or not fit["optimization"]["success"]
                or not train.date.max() < test.date.min()):
            raise ValueError(f"invalid chronological teacher ancestry for {year}")
        reconstructed = expit(design(test, ("sports", "market")) @ np.array([fit["coefficients"][key] for key in ("sports", "market")]))
        np.testing.assert_allclose(reconstructed, checked_probabilities(test.benter, "benter"), atol=1e-12, rtol=0)
        ancestry.append({**fit, "forecast_reconstruction_verified": True,
                         "use": "2025H1 auxiliary labels only" if year == 2025 else "2026 contextual comparator only"})
    return teacher, ancestry


def run(args):
    for name in ("snapshots", "predictions", "odds", "teacher_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    paths = [args.snapshots, args.snapshots.with_name("audit.json"), args.predictions, args.odds]
    paths += [args.teacher_dir / name for name in ("oos_predictions.csv", "fits.json", "summary.json", "protocol.json")]
    inputs = {str(path): sha256(path) for path in paths}
    code, frozen = code_hashes(), frozen_hashes()
    if (TAX, MIN_EV, SEED, RESAMPLES) != (.12, .05, 82, 5000):
        raise RuntimeError("shared policy/bootstrap differs from fixed contract")
    created = datetime.now(timezone.utc)
    run_version = VERSION + "-" + created.strftime("%Y%m%dT%H%M%S%fZ")
    protocol = {
        "experiment": "exp081-separate-benter-sports-distillation", "version": run_version,
        "created_utc": created.isoformat(), "status": "exploratory_pre_run_protocol",
        "input_hashes": inputs, "code_hashes": code, "frozen_hashes": frozen,
        "question": "Can sports-only canonical79 students learn useful information from chronological OPEN Benter labels without market inference inputs?",
        "models": list(MODELS), "students": list(STUDENTS), "contextual_controls": ["raw_sports", "benter"],
        "fit": "all corrected sports outcomes 2020-01-01 <= date < 2025-07-01",
        "calibration": "all corrected sports outcomes 2025-07-01 <= date < 2026-01-01; positive slope-only",
        "test": "all corrected sports outcomes in 2026; OPEN priced common sample separately",
        "teacher": "existing OPEN Benter OOF 2025 trained only 2024; only 2025H1 can supervise fitting",
        "auxiliary_targets": "market and Benter share EXACT same teacher-covered fitting rows; all other targets remain NaN, including earlier years; no market substitution",
        "weights": {"outcome_student": 0.0, "market_student": .3, "benter_student": .3},
        "loss": "mean_all BCE(y,p) + weight*mean_covered BCE(q,p) + ||coef||^2/(2*C*N_all)",
        "c": .1, "intercept": 0, "scale_centering": False,
        "hyperparameter_selection": False, "feature_audit": FEATURE_AUDIT,
        "policy": {"tax": TAX, "ev_strictly_greater_than": MIN_EV, "ties": "abstain", "one_side": "maximum EV", "pair": "finite same-book STS OPEN prices >1"},
        "selection_estimands": ["fixed original raw-sports selection and side", "own maximum-EV selection and side"],
        "bootstrap": {"unit": "calendar month", "resamples": RESAMPLES, "seed": SEED, "refit": False, "empty_selection_months": "included", "diagnostic_only": True},
        "diagnostics": "fixed format Bo1/3/5, existing competition tiers, existing rookie prior <10; no feature engineering or selection from outcomes",
        "eligible_live_rows": 0, "eligibility_live": 0, "promotion_approved": False,
        "limitations": LIMITATIONS, "argv": sys.argv,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "protocol.json", protocol)
    try:
        aligned, odds, alignment = load_aligned(args)
        opening, opening_audit = phase_frame(aligned, odds, "open")
        teacher, ancestry = load_teacher(args, opening, inputs, frozen)
        replay_audit = json.loads(args.snapshots.with_name("audit.json").read_text())
        if (replay_audit.get("legacy_feature_reuse") is not False
                or replay_audit.get("ratings_version") != "corrected-history-six-family-daybatch-v1"
                or replay_audit["outputs"]["snapshots.csv"] != inputs[str(args.snapshots)]):
            raise ValueError("uncertified corrected replay source/hash")
        full = pd.read_csv(args.snapshots, dtype=IDENTITIES, low_memory=False)
        if full[list(METADATA)].isna().any().any() or not full.golgg_match_id.is_unique:
            raise ValueError("missing metadata or duplicate sports events")
        parsed = pd.to_datetime(full.date, format="%Y-%m-%d", errors="raise")
        if (not parsed.dt.strftime("%Y-%m-%d").eq(full.date).all() or not full.y_true.isin([0, 1]).all()
                or full.team1_id.eq(full.team2_id).any() or not full.best_of.isin([1, 3, 5]).all()):
            raise ValueError("invalid sports date/label/identity/format")
        if (full[list(GROUP_FIELDS)].isna().any().any()
                or not np.isfinite(full.roster_min_prior_series).all()
                or (full.roster_min_prior_series < 0).any()):
            raise ValueError("invalid existing sports subgroup metadata")
        full = full.loc[full.date.ge("2020-01-01") & full.date.lt("2027-01-01")].sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
        masks = teacher_masks(full.date, 2026)
        stage_records = stages(full.date, masks)
        train, calibration, test = (masks[key] for key in ("train", "calibration", "prediction"))
        if not np.all(np.stack(list(masks.values())).sum(axis=0) == 1):
            raise ValueError("sports stages overlap or omit outcomes")
        if not (full.date[train].max() < full.date[calibration].min() < full.date[test].min() and full.date[calibration].max() < full.date[test].min()):
            raise ValueError("nonchronological sports stages")
        raw_fields = sorted(_REQUIRED_BASE_FIELDS | {"best_of"})
        x, names = build_training_features(full[raw_fields])
        if x.shape != (len(full), 79) or not np.isfinite(x).all():
            raise ValueError("canonical79 contract failed")
        if "BoN" in full and not full.BoN.eq(full.best_of).all():
            raise ValueError("explicit sports series formats disagree")
        merged = full[[*METADATA]].merge(teacher[["golgg_match_id", "benter", "market"]], on="golgg_match_id", how="left", validate="one_to_one")
        covered = train & full.date.ge("2025-01-01").to_numpy() & np.isfinite(merged.benter)
        benter_target = np.where(covered, merged.benter, np.nan)
        market_target = np.where(covered, merged.market, np.nan)
        if not covered.any() or not np.array_equal(np.isfinite(benter_target), np.isfinite(market_target)):
            raise ValueError("auxiliary arms do not share nonempty identical coverage")
        target_rows = full[[*METADATA]].copy()
        target_rows["stage"] = np.select([train, calibration, test], ["train", "calibration", "test"], default="invalid")
        target_rows["auxiliary_eligible"] = covered
        target_rows["market_target"], target_rows["benter_target"] = market_target, benter_target
        target_rows.to_csv(args.output_dir / "training_targets_and_stages.csv", index=False)
        scaler = StandardScaler(with_mean=False).fit(x[train])
        scaled_train, scaled_cal, scaled_test = (scaler.transform(x[mask]) for mask in (train, calibration, test))
        y = full.y_true.to_numpy(float)
        predictions = full.loc[test, [*METADATA, "competition_tier", "roster_min_prior_series"]].reset_index(drop=True)
        predictions["phase"], predictions["year"] = "open", 2026
        predictions["eligibility_live"] = 0
        predictions = predictions.merge(aligned[["golgg_match_id", "sports"]].rename(columns={"sports": "raw_sports"}), on="golgg_match_id", how="left", validate="one_to_one")
        if predictions.raw_sports.isna().any():
            raise ValueError("raw sports reference does not cover the full 2026 sports cohort")
        fits = []
        for model, target, weight in (("outcome_student", benter_target, 0.), ("market_student", market_target, .3), ("benter_student", benter_target, .3)):
            coef, optimization = fit_auxiliary_student(scaled_train, y[train], target[train], weight, c=.1)
            slope = fit_platt_scaling(scaled_cal @ coef, y[calibration])
            p = checked_probabilities(expit(slope * (scaled_test @ coef)), model)
            artifact = {
                "version": run_version + "-" + model, "model": model, "status": "experimental_not_qualified",
                "target": "series", "year": 2026, "feature_names": names, "raw_input_fields": raw_fields,
                "scale": scaler.scale_, "coef": coef, "slope": slope, "intercept": 0., "with_mean": False,
                "weight": weight, "c": .1, "stages": stage_records,
                "teacher_fit_years": [2024], "auxiliary_date_min": str(full.date[covered].min()), "auxiliary_date_max": str(full.date[covered].max()),
                "requires_odds_at_inference": False, "eligible_live_rows": 0,
                "input_hashes": inputs, "code_hashes": code, "limitations": LIMITATIONS,
            }
            path = args.output_dir / f"2026-{model}.joblib"
            joblib.dump(artifact, path)
            inferred = predict_artifact(joblib.load(path), full.loc[test, raw_fields])
            np.testing.assert_allclose(inferred, p, atol=1e-12, rtol=0)
            symmetry = float(np.max(np.abs(p + expit(slope * ((-scaled_test) @ coef)) - 1)))
            if symmetry > 1e-9:
                raise ValueError("student side symmetry failed")
            predictions[model] = p
            fits.append({"model": model, "version": artifact["version"], "weight": weight, "c": .1,
                         "outcome_train_n": int(train.sum()), "covered_train_n": int(covered.sum()),
                         "stages": stage_records, "optimization": optimization, "calibration_slope": slope,
                         "serialized_sports_only_reconstruction_verified": True, "symmetry_max_error": symmetry,
                         "artifact": path.name, "artifact_sha256": sha256(path)})
        predictions = predictions.merge(teacher[["golgg_match_id", "odds_a", "odds_b", "benter", "market"]], on="golgg_match_id", how="left", validate="one_to_one")
        common = predictions.loc[predictions.benter.notna()].reset_index(drop=True)
        if common.empty:
            raise ValueError("empty common OPEN 2026 cohort")
        prices = common[["odds_a", "odds_b"]].to_numpy(float)
        if not np.isfinite(prices).all() or not (prices > 1).all():
            raise ValueError("invalid common OPEN price pair")
        full_metrics, _ = evaluate(predictions, ("raw_sports", *STUDENTS), priced=False)
        common_metrics, decision_rows = evaluate(common, MODELS, priced=True)
        groups = residual_groups(predictions, common, x[test], names)
        predictions.to_csv(args.output_dir / "full_sports_predictions.csv", index=False)
        common.to_csv(args.output_dir / "oos_predictions.csv", index=False)
        pd.concat(decision_rows, ignore_index=True).to_csv(args.output_dir / "selected_diagnostics.csv", index=False)
        write_json(args.output_dir / "fits.json", fits)
        write_json(args.output_dir / "teacher_ancestry.json", ancestry)
        write_json(args.output_dir / "feature_residuals.json", {"definitions": FEATURE_AUDIT, "groups": groups})
        write_json(args.output_dir / "alignment.json", {"sports_reference": alignment, "open": opening_audit,
                   "sports_stages": stage_records, "all_outcome_rows_retained": True,
                   "teacher_fit_2024_only": True, "auxiliary_2025H1_only": True,
                   "same_covered_teacher_rows": True, "teacher_missing_never_filled": True,
                   "auxiliary_n": int(covered.sum()), "full_test_n": len(predictions), "common_test_n": len(common)})
        for path, expected_hash in {**inputs, **code}.items():
            if sha256(Path(path)) != expected_hash:
                raise RuntimeError(f"immutable input/code changed: {path}")
        if frozen_hashes() != frozen:
            raise RuntimeError("frozen model hashes changed")
        summary = {"status": "complete_exploratory", "version": run_version,
                   "eligible_live_rows": 0, "eligibility_live": 0, "promotion_approved": False,
                   "immutable_hashes_verified": True, "full_sports_2026": full_metrics,
                   "open_common_2026": common_metrics, "full_test_n": len(predictions), "common_test_n": len(common),
                   "limitations": LIMITATIONS + replay_audit["promotion_blockers"]}
        write_json(args.output_dir / "summary.json", summary)
        write_json(args.output_dir / "manifest.json", {"input_hashes": inputs, "code_hashes": code,
                   "frozen_hashes": frozen, "output_hashes": {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}})
        print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir), "full_test_n": len(predictions), "common_test_n": len(common), "eligible_live_rows": 0}))
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "eligible_live_rows": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/artifacts/corrected039081-20260908/replay/snapshots.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/artifacts/corrected-historical-reruns-20260908/comparison/predictions.csv")
    parser.add_argument("--odds", type=Path, default=ROOT / "data/artifacts/exp081-full-audit-20260908/identity-only-v3/odds.csv")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "data/artifacts/exp081-market-calibration-20260908/run01")
    parser.add_argument("--output-dir", type=Path, required=True, help="Exclusive new research artifact directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
