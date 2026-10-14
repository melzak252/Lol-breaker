#!/usr/bin/env python3
"""Offline inference of the exact frozen EXP039 artifacts on archived real series.

No fitting, databases, network, feature imputation, or output-level binomial transform.
Historical reconstruction eligibility is separate from prospective certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.calibration import logit
from src.models.team_order import swap_orientation, symmetrize_binary_probabilities
from src.utils.module_loading import load_module_from_path

MODEL = "EXP039-exact-frozen-sym-then-platt-series"
ARTIFACT_COMMIT = "c2e91fa1489b92d581d9c0ed81bace437bc2ab3a"
ARTIFACT_DAY = "2026-06-03"
REPLAY = ROOT / "data/artifacts/corrected039081-20260908/replay"
METADATA = ROOT / "docs/assets/final_symmetric_calibrated_market_comparison/sym_cal_lr_elasticnet_w20_binomial_metadata.json"
FROZEN = {kind: ROOT / f"betting_app/models/sym_cal_lr_elasticnet_w20_binomial_{kind}.joblib" for kind in ("pipeline", "calibrator")}
POLICY = "frozen_weights__refreshed_day_start_features__observed_first_map_roster__retrospective"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def training_evidence(pipeline: object, hashes: dict[str, str]) -> dict:
    predictions_path = ROOT / "data/golgg_y_predicts.csv"
    odds_path = ROOT / "data/odds.csv"
    original = pd.read_csv(predictions_path, dtype={"golgg_match_id": str})
    odds = pd.read_csv(odds_path, usecols=["golgg_match_id"], dtype=str).drop_duplicates()
    joined = original.merge(odds, on="golgg_match_id", validate="one_to_one")
    joined = joined.loc[joined.date >= "2020-01-01"]
    names = list(pipeline.feature_names_in_[:20])
    scaler = pipeline.named_steps["scaler"]
    imputer = pipeline.named_steps["imputer"]
    history = {}
    for kind, path in FROZEN.items():
        data = subprocess.run(["git", "show", f"{ARTIFACT_COMMIT}:{path.relative_to(ROOT)}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
        history[kind] = hashlib.sha256(data).hexdigest()
    if history != hashes:
        raise ValueError("frozen bytes differ from the historical June 3 commit")
    commit = subprocess.run(["git", "show", "-s", "--format=%H%n%aI%n%cI%n%s", ARTIFACT_COMMIT],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    historical_code = {}
    for path in ("betting_app/scripts/train_thesis_model.py", "scripts/05_ratingi_baseline/03_generate_ratings.py"):
        data = subprocess.run(["git", "show", f"{ARTIFACT_COMMIT}:{path}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
        historical_code[path] = hashlib.sha256(data).hexdigest()
    return {
        "metadata": json.loads(METADATA.read_text()),
        "metadata_sha256": sha256(METADATA),
        "git_commit": commit, "historical_artifact_sha256": history,
        "historical_training_code_sha256": historical_code,
        "historical_code_evidence": "June3 training script lines329-390: expanding2021+ OOF Platt then all-data final pipeline; June3 rating generator line147: y_true is majority-of-map series winner",
        "local_training_join": {"n": len(joined), "date_min": joined.date.min(), "date_max": joined.date.max(),
                               "calibration_oof_n": int((joined.date >= "2021-01-01").sum()),
                               "calibration_oof_date_min": joined.loc[joined.date >= "2021-01-01", "date"].min(),
                               "calibration_oof_date_max": joined.date.max()},
        "frozen_scaler_n_samples_seen": int(scaler.n_samples_seen_),
        "first20_rating_feature_statistics_agreement": {
            "max_abs_mean_difference": float(np.max(abs(joined[names].mean().to_numpy() - scaler.mean_[:20]))),
            "max_abs_variance_difference": float(np.max(abs(joined[names].var(ddof=0).to_numpy() - scaler.var_[:20]))),
            "max_abs_imputer_median_difference": float(np.max(abs(joined[names].median().to_numpy() - imputer.statistics_[:20]))),
        },
        "training_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (predictions_path, odds_path)},
        "cutoff_interpretation": "2026-03-22 is a strongly supported reconstructed data cutoff, not an immutable training-row/calibration manifest certification; byte-identical artifacts exist in local Git dated 2026-06-03",
        "fit_recipe": "final pipeline fits ALL clean odds-mapped 2020+ series; final Platt fits symmetric OOF predictions from expanding 1000-series folds starting 2021-01-01; no fitting in this run",
        "missing_evidence": ["hash-bound complete original training rows and OOF/calibration rows", "independently certified artifact deployment/publication timestamp", "timestamped source/roster availability"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=REPLAY / "snapshots.csv")
    parser.add_argument("--audit", type=Path, default=REPLAY / "audit.json")
    parser.add_argument("--outcomes", type=Path, help="Aligned EXP081 series_rows.csv for canonical refreshed feature inputs")
    parser.add_argument("--native-supplement", type=Path, help="Hash-audited native TM sigma fields omitted from canonical79 export")
    parser.add_argument("--output", type=Path, default=ROOT / "data/artifacts/real-series-model-comparison-20260909/exp039/corrected-replay")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {kind: sha256(path) for kind, path in FROZEN.items()}
    model_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    # These exact local, versioned files are the only deserialized pickle objects.
    pipeline = joblib.load(FROZEN["pipeline"])
    calibrator = joblib.load(FROZEN["calibrator"])
    names = list(pipeline.feature_names_in_)
    metadata = json.loads(METADATA.read_text())
    if names != metadata["features"] or len(names) != 46:
        raise ValueError("frozen input order disagrees with original metadata")
    evidence = training_evidence(pipeline, hashes)
    audit = json.loads(args.audit.read_text())
    policy = audit.get("information_policy")
    if policy is None:
        if args.audit.resolve() != (REPLAY / "audit.json").resolve():
            raise ValueError("alternate snapshots require an explicit information_policy in audit")
        policy = POLICY
    snapshots_hash = sha256(args.snapshots)
    expected = audit.get("outputs", {}).get(args.snapshots.name)
    if expected is None or snapshots_hash != expected:
        raise ValueError("snapshot bytes must match the supplied replay audit")
    identities = {"golgg_match_id": str, "team1_id": str, "team2_id": str, "series_id": str, "team_a": str, "team_b": str}
    frame = pd.read_csv(args.snapshots, dtype=identities)
    if args.native_supplement is not None:
        supplement_hash = sha256(args.native_supplement)
        if audit.get("outputs", {}).get(args.native_supplement.name) != supplement_hash:
            raise ValueError("native supplemental bytes must match replay audit")
        native = pd.read_csv(args.native_supplement, dtype=identities)
        keys = ["series_id", "team_a", "team_b", "forecast_cutoff"]
        columns = ["player_tm_sigma_avg1", "player_tm_sigma_avg2"]
        if any(column in frame for column in columns):
            raise ValueError("native supplement must not overwrite existing snapshot values")
        frame = frame.merge(native[keys + columns], on=keys, how="left", validate="one_to_one")
        if frame[columns].isna().any().any():
            raise ValueError("missing native TM sigma for exact prior-roster snapshot")
    if args.outcomes is not None:
        outcome_universe = pd.read_csv(args.outcomes, dtype=identities)
        outcome_universe = outcome_universe.loc[outcome_universe.information_policy == "refreshed_prior_day"]
        outcomes = outcome_universe.loc[outcome_universe.probability_a.notna()]
        keys = ["series_id", "phase_id", "golgg_match_id", "source_date", "forecast_cutoff", "team_a", "team_b", "best_of"]
        before = len(frame)
        frame = frame.merge(outcomes[keys + ["y_true", "tournament"]], on=keys, how="left", validate="one_to_one")
        if len(frame) != before or frame.y_true.isna().any():
            raise ValueError("canonical features lack exact ordered-side/date/format outcome alignment")
        frame["date"], frame["BoN"] = frame.source_date, frame.best_of
        frame["team1_id"], frame["team2_id"] = frame.team_a, frame.team_b
        frame["team1_name"], frame["team2_name"] = frame.team_a, frame.team_b
        max_at = pd.to_datetime(frame.feature_history_max_at, utc=True)
        cutoffs = pd.to_datetime(frame.forecast_cutoff, utc=True)
        if max_at.isna().any() or cutoffs.isna().any() or (max_at > cutoffs).any():
            raise ValueError("canonical features contain information released after cutoff")
    if frame.golgg_match_id.duplicated().any() or not frame.y_true.isin([0, 1]).all():
        raise ValueError("series IDs or binary aligned outcomes invalid")
    if not frame.BoN.isin([1, 3, 5]).all() or not frame.BoN.eq(frame.best_of).all():
        raise ValueError("unsupported or inconsistent series format")
    helper = load_module_from_path(ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py", "frozen039_binomial_helpers")
    enriched, generated = helper.add_binomial_features(frame)
    if generated != names[-6:] or not np.isfinite(enriched[names].to_numpy(float)).all():
        raise ValueError("complete finite original46 inputs required; no proxy/default features allowed")
    swapped = swap_orientation(enriched, names, helper.RANK_PROB_FEATURES, np.ones(len(frame), dtype=bool))
    raw = np.clip(pipeline.predict_proba(enriched[names])[:, 1], .001, .999)
    reverse = np.clip(pipeline.predict_proba(swapped[names])[:, 1], .001, .999)
    symmetric = symmetrize_binary_probabilities(raw, reverse)
    probability = np.clip(calibrator.predict_proba(logit(symmetric))[:, 1], .001, .999)
    reverse_calibrated = np.clip(calibrator.predict_proba(logit(1 - symmetric))[:, 1], .001, .999)
    if not np.isfinite(probability).all() or not ((probability > 0) & (probability < 1)).all():
        raise ValueError("invalid actual model output")
    result = frame[["golgg_match_id", "date", "tournament", "team1_id", "team2_id", "team1_name", "team2_name", "best_of", "y_true"]].rename(columns={
        "golgg_match_id": "series_id", "date": "source_date", "team1_id": "team_a_id", "team2_id": "team_b_id", "team1_name": "team_a", "team2_name": "team_b"})
    result["source_series_id"] = frame.golgg_match_id
    if "series_id" in frame:
        result["series_id"] = frame.series_id
    result["phase_id"] = frame["phase_id"] if "phase_id" in frame else None
    result["forecast_cutoff"] = frame["forecast_cutoff"] if "forecast_cutoff" in frame else result.source_date
    if "feature_history_max_at" in frame:
        result["feature_history_max_at"] = frame["feature_history_max_at"]
    result["forecast_cutoff_precision"] = "source calendar day start; timezone unverified"
    result["model_identifier"] = MODEL
    result["model_sha256"] = model_hash
    result["pipeline_sha256"] = hashes["pipeline"]
    result["calibrator_sha256"] = hashes["calibrator"]
    result["probability_a"] = probability
    result["probability_b"] = 1 - probability
    result["raw_probability_a"] = raw
    result["raw_swapped_probability_b"] = reverse
    result["symmetric_probability_a"] = symmetric
    result["reverse_run_calibrated_probability_b"] = reverse_calibrated
    result["calibration_symmetry_error"] = abs(probability + reverse_calibrated - 1)
    result["information_policy"] = policy
    result["input_snapshot_sha256"] = snapshots_hash
    result["training_cutoff_reconstructed"] = evidence["local_training_join"]["date_max"]
    result["artifact_available_local_git_day"] = ARTIFACT_DAY
    result["eligible_historical_reconstructed"] = result.source_date > ARTIFACT_DAY
    result["eligible_prospective_certified"] = False
    result["eligible"] = result.eligible_historical_reconstructed
    result["status"] = np.where(result.eligible, "historical_reconstructed_post_artifact", "diagnostic_pre_artifact")
    result["reason"] = np.where(result.eligible,
        "byte-identical local Git artifact predates origin; retrospective day-batched features, not certified source availability or untouched holdout",
        "origin does not follow documented artifact existence; frozen fit/calibration overlaps much inspected history")
    result["feature_state_mode"] = "refreshed_day_start"
    result["model_weight_mode"] = "frozen_exact_artifacts"
    result["log_loss"] = -(result.y_true * np.log(probability) + (1 - result.y_true) * np.log1p(-probability))
    result.to_csv(args.output / "predictions.csv", index=False)
    result.loc[result.eligible].to_csv(args.output / "historically_eligible_predictions.csv", index=False)
    result.iloc[:0].to_csv(args.output / "prospectively_certified_predictions.csv", index=False)
    exclusions = []
    if args.outcomes is not None:
        missing = outcome_universe.loc[~outcome_universe.series_id.isin(result.series_id)]
        fields = ["series_id", "phase_id", "golgg_match_id", "source_date", "forecast_cutoff", "tournament", "team_a", "team_b", "best_of", "y_true"]
        for row in missing.to_dict("records"):
            record = {key: None if pd.isna(row[key]) else row[key] for key in fields}
            record.update(probability_a=None, model_identifier=MODEL, model_sha256=model_hash,
                          information_policy=policy, eligible=False, status="excluded_no_inference",
                          reason="no_compatible_refreshed_prior_observed_roster_snapshot_for_frozen039",
                          upstream_availability_reason=row["reason"])
            exclusions.append(record)
    for filename, reason in (("feature_excluded_series.json", "missing_compatible_complete_w20_snapshot"), ("rejected_series.json", "source_structural_rejection")):
        path = args.snapshots.parent / filename
        if path.exists():
            if audit.get("outputs", {}).get(filename) != sha256(path):
                raise ValueError(f"exclusion ledger checksum mismatch: {filename}")
            for row in json.loads(path.read_text())["rows"]:
                source = row.get("series", row)
                exclusions.append({"series_id": str(source["match_id"]), "source_date": source.get("date"),
                                   "forecast_cutoff": source.get("date"), "tournament": source.get("tournament"), "phase_id": None,
                                   "team_a": None, "team_b": None, "best_of": source.get("best_of"),
                                   "probability_a": None, "y_true": None, "model_identifier": MODEL, "model_sha256": model_hash,
                                   "information_policy": policy, "eligible": False, "status": "excluded_no_inference",
                                   "reason": row.get("reason", reason), "exclusion_class": reason})
    write_json(args.output / "exclusions.json", {"n": len(exclusions), "rows": exclusions})
    result.groupby(["tournament", "status"], dropna=False).agg(n=("series_id", "size"), log_loss=("log_loss", "mean")).reset_index().to_csv(args.output / "tournament_diagnostics.csv", index=False)
    summary = {
        "model": MODEL, "artifact_sha256": hashes, "model_sha256": model_hash, "feature_order": names,
        "inference_rows": len(result), "historically_eligible_rows": int(result.eligible.sum()), "prospectively_certified_rows": 0,
        "excluded_no_inference_rows": len(exclusions), "date_min": result.source_date.min(), "date_max": result.source_date.max(),
        "contract": {"target": "series winner in source team1 orientation", "binomial": "six INPUT map-rating probabilities converted for actual BoN; model already trained against series y_true; NO output-level binomial conversion",
                     "symmetry_calibration_order": "clip raw original/swapped; average original and complemented swapped; frozen intercept-bearing Platt(logit); clip .001..999",
                     "calibrator_slope": float(calibrator.coef_[0, 0]), "calibrator_intercept": float(calibrator.intercept_[0]),
                     "calibrated_reverse_max_sum_error": float(result.calibration_symmetry_error.max()),
                     "reverse_alignment": "probability_b=1-probability_a from ONE source orientation; reverse-run output is diagnostic, not a second coherent marginal",
                     "runtime_conflict": "registry labels EXP039 map; original training and DB backtest establish SERIES target; applying another binomial expansion would double-convert",
                     "information_policy": policy, "frozen_tournament_feature_state": False},
        "training_evidence": evidence,
        "feature_provenance": {"snapshots": str(args.snapshots), "sha256": snapshots_hash, "audit": str(args.audit), "audit_sha256": sha256(args.audit),
                               "outcomes": str(args.outcomes) if args.outcomes else None,
                               "outcomes_sha256": sha256(args.outcomes) if args.outcomes else None,
                               "native_supplement": str(args.native_supplement) if args.native_supplement else None,
                               "native_supplement_sha256": sha256(args.native_supplement) if args.native_supplement else None,
                               "ratings_version": audit.get("ratings_version"), "same_day_policy": audit.get("same_day_policy"),
                               "limitations": audit.get("promotion_blockers"),
                               "roster_policy": audit.get("roster_policy", "observed first-map roster"),
                               "distribution_difference": "chronological corrected six-family replay and complete-only W20 under the recorded roster policy; original frozen training used legacy rating CSV, sequential series updates and rolling defaults; equal feature semantics/order are not identical training distribution"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), ROOT / "src/models/team_order.py", ROOT / "src/models/calibration.py", ROOT / "scripts/06_metamodel/06aa_w20_binomial_lr_vs_extratrees_bootstrap.py", ROOT / "betting_app/scripts/train_thesis_model.py", ROOT / "scripts/05_ratingi_baseline/03_generate_ratings.py"]},
        "output_schema": list(result.columns), "interpretation": "all inspected development history; no model promotion, trust, prospective performance, or betting claim",
    }
    if hashes != {kind: sha256(path) for kind, path in FROZEN.items()}:
        raise ValueError("frozen artifacts changed during inference")
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("inference_rows", "historically_eligible_rows", "prospectively_certified_rows", "excluded_no_inference_rows")}))
    print(args.output)


if __name__ == "__main__":
    main()
