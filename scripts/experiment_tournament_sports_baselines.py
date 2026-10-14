#!/usr/bin/env python3
"""Fixed odds-free modern-series benchmark; local retrospective research only.

Run with --output-dir NEW. Never loads a database, changes a frozen model, selects
on test outcomes, or treats reconstructed source history as point-in-time certified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_siamese_architectures import monthly_bootstrap, probability_metrics, temporal_masks
from scripts.build_siamese_research_dataset import sha256
from scripts.corrected_history_evaluation import reliability_bins
from scripts.experiment_rating_stack import fit_ridge, ridge_logits, ridge_metadata
from scripts.train_and_tune_siamese_series import build_training_features, fit_platt_scaling, write_json_exclusive
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.symmetric_series import _independent_series_probability

VERSION = "oddsfree-tournament-sports-baselines-v1"
REPLAY = ROOT / "data/artifacts/corrected039081-20260908/replay"
RERUN = ROOT / "data/artifacts/corrected-historical-reruns-20260908"
EVALUATION = REPLAY.parent / "evaluation-complete"
IDENTITY = ["golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y_true"]
IDS = {name: str for name in IDENTITY[:3]}
SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")
ARCHIVED = {
    "exp039": (EVALUATION, "exp039-corrected-history-research-v1"),
    "recipe__reported_regularization_bagging": (RERUN / "reported-recipe081", "p__reported_regularization_bagging"),
    "architecture__linear79": (RERUN / "architecture-original-protocol", "linear79"),
    "annual__annual090_uniform_refit": (RERUN / "annual090/run01", "p__annual090_uniform_refit"),
    "annual__annual090_recency365": (RERUN / "annual090/run01", "p__annual090_recency365"),
}
UNCERTAINTY = ("team_gl_rd1", "team_gl_rd2", "player_gl_rd_avg1", "player_gl_rd_avg2",
               "team_ts_sigma1", "team_ts_sigma2", "player_ts_sigma_avg1", "player_ts_sigma_avg2")


def load_json(path):
    return json.loads(Path(path).read_text())


def open_probability(values):
    return np.clip(values, 1e-12, 1 - 1e-12)


def checked_frame(path, columns):
    frame = pd.read_csv(path, dtype=IDS, usecols=columns)
    if frame.empty or frame.golgg_match_id.isna().any() or not frame.golgg_match_id.is_unique:
        raise ValueError(f"{path}: empty, missing or duplicate series IDs")
    if frame[IDENTITY].isna().any().any():
        raise ValueError(f"{path}: missing identity/date/format/label")
    dates = pd.to_datetime(frame.date, errors="raise")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError(f"{path}: expected explicit whole dates")
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    if not frame.best_of.isin([1, 3, 5]).all() or not frame.y_true.isin([0, 1]).all():
        raise ValueError(f"{path}: invalid binary series format/label")
    if frame.team1_id.eq(frame.team2_id).any():
        raise ValueError(f"{path}: identical opposing teams")
    return frame.sort_values(["date", "golgg_match_id"]).set_index("golgg_match_id", drop=False)


def align_exact(reference, source, label):
    if set(reference.index) != set(source.index):
        raise ValueError(f"{label}: full modern cohort IDs differ; never silently intersect")
    source = source.loc[reference.index]
    for column in IDENTITY:
        if not np.array_equal(reference[column].to_numpy(), source[column].to_numpy()):
            raise ValueError(f"{label}: {column} differs on exact series IDs")
    return source


def stage_bounds(frame, mask):
    part = frame.loc[mask]
    if part.empty:
        raise ValueError("empty fixed chronological stage")
    return {"n": len(part), "min": part.date.min(), "max": part.date.max()}


def check_stages(stages, year, frame, *, semester=False):
    """Compare archived cutoffs and counts against the actual immutable rows."""
    masks = dict(zip(("train", "calibration", "test"), temporal_masks(frame.date, year)))
    if semester:
        dates = pd.to_datetime(frame.date)
        masks["calibration"] = masks["calibration"] & (dates >= f"{year-1}-07-01").to_numpy()
    for name, mask in masks.items():
        actual = stage_bounds(frame, mask)
        recorded = stages[name]
        normalized = {"n": recorded["n"],
                      "min": pd.Timestamp(recorded.get("min", recorded.get("date_min"))).strftime("%Y-%m-%d"),
                      "max": pd.Timestamp(recorded.get("max", recorded.get("date_max"))).strftime("%Y-%m-%d")}
        if actual != normalized:
            raise ValueError(f"{year}/{name}: archived stage differs: {normalized} != {actual}")
    if not stages["train"].get("max", stages["train"].get("date_max")) < stages["calibration"].get("min", stages["calibration"].get("date_min")):
        raise ValueError("training must finish before calibration")
    return {name: stage_bounds(frame, mask) for name, mask in masks.items()}


def operational_audit(snapshot_columns):
    """Document the actual source contract, not the obsolete backfill label."""
    missing = {
        "regional_history": ["competition_family_a", "competition_family_b", "competition_tier_a", "competition_tier_b",
                             "regional_location_mean", "regional_location_variance", "regional_glicko_probability"],
        "roster_reconstruction": ["player_ids_1", "player_ids_2", "per_player_rating_value", "per_player_rd",
                                  "per_player_sigma", "per_player_major_games_played", "per_player_last_match_at"],
        "availability": ["source_available_at", "roster_announced_at", "prediction_at", "model_available_at"],
        "historical_deployment": ["active_model_version_at_prediction", "earlier_fitted_model_artifact_sha256",
                                   "earlier_calibrator_artifact_sha256", "rating_state_version_at_prediction"],
    }
    return {
        "status": "blocked_exact_operational_chronological_replay",
        "default_current_model": "EXP081 Siamese, registry default (environment overrides not replayable historically)",
        "current_contract": "canonical79 -> standardized anti-symmetric member logits -> each member's stored positive Platt slope -> mean calibrated logits -> sigmoid; output already FULL SERIES",
        "calibration": "Frozen fitted member slopes; no extra binomial expansion. Hybrid temperature0.80 and market blending are separate and excluded.",
        "70_20_10_contract": "EXP078 linear_symmetric exception fallback ONLY: renormalize available 0.70 player consensus + 0.20 team consensus + 0.10 W20; clip map probability[.01,.99], then one binomial series projection. No fitted fallback calibrator.",
        "backfill_contract_mismatch": "backfill_operational_predictions advertises70/20/10 but invokes the current unified wrapper with no canonical.best_of and no canonical79 raw state, then calls series_probability. The wrapper returns result.prob_a (SERIES). Missing canonical.best_of resolves to1 in the current snapshot adapter, so this is NOT proof of double-expanding the requested BoN. The payload, model ancestry and unit documentation are obsolete; the constructed start-minus1minute prediction timestamp is not an observed historical forecast.",
        "snapshot_ratings": "Corrected snapshots use unadjusted six-family RatingManager probabilities, not operational family-calibrated regional Glicko/location-adjusted probabilities or individual major-rookie caps.",
        "missing_fields_by_requirement": {group: [field for field in fields if field not in snapshot_columns] for group, fields in missing.items()},
        "missing_artifacts": "Per-prediction historically deployed model/calibrator, frozen earlier regional posterior state and timestamped roster/source archives; today's frozen artifacts do not prove earlier availability.",
        "w20": "All six W20 formula inputs exist as rolling fields; W20 is not the blocker. Its canonical operational formula is1.25*win_diff+.035*kills_diff-.03*deaths_diff+.00018*gd15_diff+.00008*dpm_diff+.07*towers_diff then sigmoid (runtime uses truthy fallbacks).",
        "replacement_policy": "No approximation or simple average is emitted under an operational model name.",
        "source_symbols": ["betting_app/core/models/registry.py:get_active_model", "betting_app/core/models/engine.py:PredictionEngine.predict_from_features",
                           "betting_app/services/upcoming_inference_service.py:predict_probability_from_features",
                           "betting_app/services/upcoming_inference_service.py:load_roster_player_ratings",
                           "betting_app/scripts/backfill_operational_predictions.py:_raw_probabilities",
                           "src/models/siamese_series.py:SiameseSeriesModel"],
    }


def archive_evidence(frame, modern, output, source_hashes):
    """Reuse exact stored predictions and inspect their earlier-fit ancestry."""
    comparison_path = RERUN / "comparison/predictions.csv"
    available = pd.read_csv(comparison_path, nrows=0).columns
    mandatory = list(ARCHIVED)[:3]
    missing = [name for name in mandatory if name not in available]
    if missing:
        raise ValueError(f"required frozen comparison columns missing: {missing}")
    selected = [name for name in ARCHIVED if name in available]
    comparison = align_exact(modern, checked_frame(comparison_path, IDENTITY + selected), "shared OOF")
    source_hashes[str(comparison_path.relative_to(ROOT))] = sha256(comparison_path)
    evidence, blocked = {}, {}
    summaries = {}
    years = sorted(pd.to_datetime(modern.date).dt.year.unique())
    for name in ARCHIVED:
        directory, column = ARCHIVED[name]
        if name not in selected:
            blocked[name] = "named earlier-fit control absent from shared OOF; not retrained or substituted"
            continue
        path = directory / "predictions.csv"
        source = align_exact(modern, checked_frame(path, IDENTITY + [column]), name)
        if not np.allclose(source[column], comparison[name], atol=1e-12, rtol=0):
            raise ValueError(f"{name}: shared OOF predictions differ from originating artifact")
        p = comparison[name].to_numpy(float)
        if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise ValueError(f"{name}: invalid archived series probabilities")
        summary_path = directory / "summary.json"
        if directory not in summaries:
            summaries[directory] = load_json(summary_path)
        summary = summaries[directory]
        snapshot_digest = source_hashes[str((REPLAY / "snapshots.csv").relative_to(ROOT))]
        if name == "exp039" or name.startswith("architecture__"):
            archived_snapshot_digest = summary["dataset_audit"]["outputs"]["snapshots.csv"]
        elif name.startswith("recipe__"):
            archived_snapshot_digest = summary["sources"][str((REPLAY / "snapshots.csv").relative_to(ROOT))]
        else:
            archived_snapshot_digest = summary["source"]["sha256"]
        if archived_snapshot_digest != snapshot_digest:
            raise ValueError(f"{name}: originating snapshot ancestry differs")
        expected_prediction_digest = summary.get("predictions_sha256", summary.get("provenance", {}).get("predictions_sha256"))
        if expected_prediction_digest and expected_prediction_digest != sha256(path):
            raise ValueError(f"{name}: originating prediction digest mismatch")
        sources = [path, summary_path]
        folds = []
        for year in years:
            if name == "exp039":
                archived = next(f for f in summary["folds"] if f["year"] == year)
                stages = {"train": archived["fit"], "calibration": archived["calibration"], "test": archived["test"]}
                artifact = directory / f"exp039-corrected-history-research-v1-{year}.joblib"
                if sha256(artifact) != summary["provenance"]["output_artifacts_sha256"][artifact.name]:
                    raise ValueError(f"{name}/{year}: fitted artifact digest mismatch")
                sources.append(artifact)
            elif name.startswith("recipe__"):
                fold_path = directory / f"{year}-calibration.json"
                archived = load_json(fold_path)
                stages = archived["stages"]
                artifact = directory / f"{year}-reported_regularization_bagging.json"
                if sha256(artifact) != archived["models"]["reported_regularization_bagging"]["artifact_sha256"]:
                    raise ValueError(f"{name}/{year}: fitted artifact digest mismatch")
                sources.extend([fold_path, artifact, directory / "manifest.json"])
            elif name.startswith("architecture__"):
                archived = next(f for f in summary["protocol"]["folds"] if f["year"] == year)
                stages = {stage: {"n": archived[f"{stage}_n"], "min": archived.get(f"{stage}_min", frame.date.min()),
                                  "max": archived[f"{stage}_max"]} for stage in ("train", "calibration", "test")}
            else:
                fold_path = directory / f"{year}-fold.json"
                archived = load_json(fold_path)
                stages = archived["stages"]
                candidate = name.removeprefix("annual__annual090_")
                artifact = directory / f"{year}-annual090-{candidate}.joblib"
                # Metadata inspection of trusted local research artifacts; no refitting.
                payload = joblib.load(artifact)
                meta = payload["metadata"]
                if meta["candidate"] != candidate or meta["year"] != year or meta["selection"] is not None:
                    raise ValueError(f"{name}/{year}: not the fixed non-selected annual candidate")
                if not pd.Timestamp(meta["train_max_date"]) < pd.Timestamp(meta["fit_before"]) <= pd.Timestamp(meta["forecast_start"]):
                    raise ValueError(f"{name}/{year}: not trained strictly before forecast")
                if not pd.Timestamp(meta["calibration_max_date"]) < pd.Timestamp(meta["forecast_start"]):
                    raise ValueError(f"{name}/{year}: calibration crosses forecast")
                if candidate == "recency365":
                    weights = archived["fits"][candidate]["weights"]
                    if weights["half_life_days"] != 365 or not np.isclose(weights["total_mass"], stages["train"]["n"]):
                        raise ValueError(f"{name}/{year}: recency half-life or training mass differs")
                    if pd.Timestamp(weights["latest_training_date"]) >= pd.Timestamp(weights["fit_before"]):
                        raise ValueError(f"{name}/{year}: recency control consumes later outcomes")
                sources.extend([fold_path, artifact, directory / "protocol.json"])
            checked = check_stages(stages, int(year), frame, semester=name.startswith("annual__"))
            folds.append({"year": int(year), "stages": checked, "earlier_fit_and_calibration_verified": True})
        for path in sources:
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)
        output[name] = p
        evidence[name] = {"prediction_source": str(directory.relative_to(ROOT) / "predictions.csv"),
                          "source_column": column, "folds": folds, "retrained": False,
                          "snapshot_ancestry_sha256_verified": archived_snapshot_digest,
                          "claim": "fixed archived corrected-history research comparison, not a frozen original operational reproduction"}
    for directory, summary in summaries.items():
        # Preserve the ancestors' provenance fields without importing their market diagnostics.
        evidence[str(directory.relative_to(ROOT))] = {key: summary[key] for key in
            ("sources", "source", "code_sha256", "predictions_sha256", "limitations") if key in summary}
    return evidence, blocked


def composition_columns(names):
    return {
        "ridge_team_only": [i for i, name in enumerate(names) if name.startswith("team_")],
        "ridge_player_only": [i for i, name in enumerate(names) if name.startswith("player_")],
        "ridge_consensus_only": [i for i, name in enumerate(names) if name.startswith("rating_consensus_logit")],
    }


def research_probabilities(frame):
    """Explicit research averages are not the operational regional consensus."""
    team = frame[[f"team_{system}" for system in SYSTEMS]].mean(axis=1).to_numpy()
    player = frame[[f"player_{system}" for system in SYSTEMS]].mean(axis=1).to_numpy()
    map_models = {"team_elo": frame.team_elo.to_numpy(), "player_gl": frame.player_gl.to_numpy(),
                  "team_mean6": team, "player_mean6": player, "consensus_mean12": (team + player) / 2}
    result = {name: open_probability(np.fromiter(
        (_independent_series_probability(float(p), int(bo)) for p, bo in zip(values, frame.best_of)),
        dtype=float, count=len(frame))) for name, values in map_models.items()}
    return result, abs(team - player)


def fit_compositions(frame, modern_mask, predictions, output, x, names):
    subsets = composition_columns(names)
    raw, disagreement = research_probabilities(frame)
    predictions["player_team_disagreement"] = disagreement[modern_mask]
    uncertainty = pd.DataFrame({
        "glicko_rd": frame[list(UNCERTAINTY[:4])].mean(axis=1),
        "trueskill_sigma": frame[list(UNCERTAINTY[4:])].mean(axis=1),
    })
    for name in uncertainty:
        predictions[f"{name}_mean"] = uncertainty.loc[modern_mask, name].to_numpy()
        predictions[f"{name}_high_earlier_q75"] = False
    predictions["coinflip"] = 0.5
    for name, values in raw.items():
        predictions[f"{name}_series_raw"] = values[modern_mask]
        predictions[f"{name}_series_calibrated"] = np.nan
    for name in subsets:
        predictions[name] = np.nan
    folds, membership = [], []
    for year in sorted(pd.to_datetime(predictions.date).dt.year.unique()):
        train, cal, test = temporal_masks(frame.date, int(year))
        stages = {stage: stage_bounds(frame, mask) for stage, mask in (("train", train), ("calibration", cal), ("test", test))}
        if not stages["train"]["max"] < stages["calibration"]["min"] <= stages["calibration"]["max"] < stages["test"]["min"]:
            raise ValueError("chronological train/calibration/test stages overlap")
        y = frame.y_true.to_numpy(float)
        if any(len(np.unique(y[mask])) != 2 for mask in (train, cal)):
            raise ValueError(f"{year}: fitting/calibration requires both outcomes")
        protocol = {"year": int(year), "stages": stages, "feature_subsets": {key: [names[i] for i in indices] for key, indices in subsets.items()},
                    "selection": "none; C=.1, zero intercept, lbfgs seed83 max_iter10000 tol1e-8, TRAIN-only StandardScaler(with_mean=False), prior-year positive slope Platt"}
        write_json_exclusive(output / f"{year}-fit-protocol.json", protocol)
        for stage, mask in (("train", train), ("calibration", cal), ("test", test)):
            part = frame.loc[mask, ["golgg_match_id", "date"]].copy()
            part["fold"], part["stage"] = int(year), stage
            membership.append(part)
        target = test[modern_mask]
        fold = {**protocol, "models": {}, "diagnostic_thresholds": {}}
        for name, values in raw.items():
            slope = float(fit_platt_scaling(logit(values[cal]), y[cal]))
            predictions.loc[target, f"{name}_series_calibrated"] = open_probability(expit(slope * logit(values[test])))
            fold["models"][f"{name}_series_calibrated"] = {"slope": slope, "fit": "fixed rating/history state; only calibration fit in prior year"}
        for name, indices in subsets.items():
            selected = x[:, indices]
            model = fit_ridge(selected[train], y[train], [names[i] for i in indices])
            slope = float(fit_platt_scaling(ridge_logits(model, selected[cal]), y[cal]))
            p = open_probability(expit(slope * ridge_logits(model, selected[test])))
            predictions.loc[target, name] = p
            artifact = {"version": f"{VERSION}-{year}-{name}", "model": model, "slope": slope,
                        "indices": indices, "canonical_feature_names": names, "protocol": protocol,
                        "target": "full series probability; do not project again"}
            with (output / f"{year}-{name}.joblib").open("xb") as handle:
                joblib.dump(artifact, handle)
            fold["models"][name] = {**ridge_metadata(model), "slope": slope}
        for name in uncertainty:
            threshold = float(uncertainty.loc[train, name].quantile(.75))
            predictions.loc[target, f"{name}_high_earlier_q75"] = uncertainty.loc[test, name].to_numpy() >= threshold
            fold["diagnostic_thresholds"][name] = {"train_q75": threshold, "units": "native rating-system units; never pooled across systems"}
        write_json_exclusive(output / f"{year}-fit.json", fold)
        folds.append(fold)
    pd.concat(membership, ignore_index=True).to_csv(output / "fit_membership.csv", index=False, mode="x")
    return folds, ["coinflip", *[f"{name}_series_{suffix}" for name in raw for suffix in ("raw", "calibrated")], *subsets]


def slice_masks(frame):
    roster = frame.roster_min_prior_series
    reference = frame.exp039
    masks = {"overall": np.ones(len(frame), dtype=bool),
             "rookie_no_prior_series": roster == 0, "rookie_1_to4": roster.between(1, 4),
             "roster_prior5_to19": roster.between(5, 19), "roster_prior20plus": roster >= 20,
             "roster_all_prior_ge10": roster >= 10, "roster_any_prior_lt10": roster < 10,
             "disagreement_le008": frame.player_team_disagreement <= .08,
             "disagreement_gt015": frame.player_team_disagreement > .15,
             "tier1": frame.competition_tier.isin(["major", "international"]),
             "confidence_exp039_heavy_favorite": reference >= .75,
             "confidence_exp039_moderate_favorite": (reference >= .60) & (reference < .75),
             "confidence_exp039_tossup": reference.between(.45, .55),
             "confidence_exp039_underdog": reference <= .40,
             "rest_either_ge30days": frame[["days_since_last_1", "days_since_last_2"]].max(axis=1) >= 30}
    for field in ("competition_tier", "best_of"):
        for value in sorted(frame[field].unique()):
            masks[f"{field}={value}"] = frame[field] == value
    for year in sorted(pd.to_datetime(frame.date).dt.year.unique()):
        masks[f"year={year}"] = frame.date.str.startswith(str(year))
    for field in ("glicko_rd", "trueskill_sigma"):
        high = frame[f"{field}_high_earlier_q75"].astype(bool)
        masks[f"{field}_high_train_q75"] = high
        masks[f"{field}_below_train_q75"] = ~high
    return masks


def score_comparisons(predictions, models, output):
    rows, paired, reliability = [], [], []
    pairs = [(name, "exp039") for name in models if name != "exp039"]
    pairs.extend([("ridge_team_only", "ridge_player_only"), ("ridge_consensus_only", "ridge_player_only"),
                  ("team_mean6_series_calibrated", "team_mean6_series_raw"),
                  ("player_mean6_series_calibrated", "player_mean6_series_raw"),
                  ("consensus_mean12_series_calibrated", "consensus_mean12_series_raw")])
    if all(name in models for name in ("annual__annual090_recency365", "annual__annual090_uniform_refit")):
        pairs.append(("annual__annual090_recency365", "annual__annual090_uniform_refit"))
    for label, mask in slice_masks(predictions).items():
        part = predictions.loc[mask]
        for name in models:
            result = probability_metrics(part.y_true, part[name])
            rows.append({"cohort": label, "model": name, **result})
            if len(part):
                reliability.extend({"cohort": label, "model": name, **entry} for entry in reliability_bins(part.y_true, part[name]))
        for candidate, reference in pairs:
            item = {"cohort": label, "candidate": candidate, "reference": reference, "n": len(part),
                    "direction": "candidate minus reference; negative loss favors candidate", "seed": 82}
            if pd.to_datetime(part.date).dt.to_period("M").nunique() < 2:
                paired.append({**item, "status": "unavailable_fewer_than_two_months"})
                continue
            y = part.y_true.to_numpy(float)
            for metric, delta in (("log_loss", binary_log_loss_vector(y, part[candidate]) - binary_log_loss_vector(y, part[reference])),
                                  ("brier", (part[candidate].to_numpy() - y)**2 - (part[reference].to_numpy() - y)**2)):
                paired.append({**item, "metric": metric, "status": "nominal_paired_monthly_ci", **monthly_bootstrap(delta, part.date, 5000)})
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False, mode="x")
    pd.DataFrame(paired).to_csv(output / "paired_monthly.csv", index=False, mode="x")
    pd.DataFrame(reliability).to_csv(output / "reliability.csv", index=False, mode="x")
    return [row for row in rows if row["cohort"] == "overall"], [row for row in paired if row["cohort"] == "overall"]


def hypothesis_evidence(paired):
    """Classify fixed nominal comparisons without selecting/promoting a winner."""
    result = []
    for row in paired:
        if row.get("metric") != "log_loss":
            continue
        if row["ci95_high"] < 0:
            status = "nominal_interval_favors_candidate; retrospective_not_confirmation"
        elif row["ci95_low"] > 0:
            status = "nominal_interval_favors_reference; candidate_improvement_unsupported"
        else:
            status = "direction_unresolved; improvement_unsupported"
        result.append({"candidate": row["candidate"], "reference": row["reference"],
                       "status": status, "log_loss_delta": row["mean"],
                       "nominal_ci95": [row["ci95_low"], row["ci95_high"]],
                       "multiplicity_adjusted": False})
    return result


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "version": VERSION, "locked_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Odds-free chronological full modern-series benchmark and narrow sports composition diagnostics",
        "target": "P(team1 wins FULL SERIES), canonical sides; never binomially expand stored series forecasts",
        "cohort": "Every2024+ row in immutable corrected snapshots; all tiers and Bo1/3/5; no bookmaker join, market-selected subset or outcome-dependent exclusion. Source W20 exclusions remain inherited and reported.",
        "input_paths": {"snapshots": str(REPLAY / "snapshots.csv"), "oof": str(RERUN / "comparison/predictions.csv")},
        "model_definitions": {
            "exp039": "Frozen comparison COLUMN: archived annual corrected-history46-feature EXP039 research refits with prior-year slope calibration; NOT the original frozen operational artifact",
            "recipe__reported_regularization_bagging": "Archived corrected report-informed EXP08179->32->16 focal gamma1 five-member, weight_decay.01/bootstrap.8,35epochs; member slopes prior year; not current operational and not original-fit provenance certified",
            "architecture__linear79": "Reuse earlier-trained canonical79 zero-intercept linear architecture control, prior-calendar-year calibration",
            "annual__annual090_uniform_refit": "Fixed earlier-trained canonical79 ridge C.1, uniform TRAIN mass; priorH2 calibration; no SELECT winner",
            "annual__annual090_recency365": "Same fixed annual ridge, exponential365day half-life weights normalized to TRAIN N, same priorH2 calibration; tests recency weighting, not full team-strength dynamics",
            "coinflip": "Fixed P=.5 outcome-only null, ties half-credit in accuracy",
            "team_elo/player_gl_series_raw": "Established replay outcome-updated rating probabilities projected to series ONCE using canonical helper",
            "team_mean6/player_mean6/consensus_mean12_series_raw": "Explicit equal arithmetic mean of unadjusted map rating probabilities, then ONE canonical series projection; research only, not operational consensus",
            "*_series_calibrated": "Raw research controls with positive zero-intercept Platt slope fitted on previous calendar year only",
            "ridge_team_only/ridge_player_only": "Canonical79 feature subset starts team_/player_ respectively, including own raw mean/uncertainty differences and BoN transforms; NO W20, rest or mixed consensus; fixed ridge C.1; TRAIN-only scaling; prior-year slope calibration",
            "ridge_consensus_only": "Only canonical rating_consensus_logit and Bo3/Bo5 interactions (mean of twelve LOGITS), fixed ridge and prior-year slope; distinct from arithmetic-mean-probability research control",
        },
        "new_fit_stages": "TRAIN dates[2020-01-01,Y-1-01-01), CAL[Y-1-01-01,Y-01-01), TEST[Y-01-01,Y+1-01-01). No retrain of available earlier-fit models; no test tuning; new version per fold.",
        "seeds": {"new_ridge": 83, "monthly_bootstrap": 82, "new_slope_fits": "deterministic"},
        "slices": {"tier": "all exact snapshot labels plus major/international Tier1 union", "BoN": [1, 3, 5],
                   "roster_min_prior_series": ["0", "1-4", "5-19", ">=20", "all>=10", "any<10"],
                   "disagreement": "absolute mean6 player minus mean6 team: agreement<=.08 and severe>.15 map probability",
                   "confidence": "Fixed exp039 reference SERIES P(A) buckets >=.75, [.60,.75), [.45,.55], <=.40; identical event membership across candidates, not market-odds tiers",
                   "rest": "either team>=30days",
                   "uncertainty": "mean of team/player side Glicko RD separately from TrueSkill sigma; high>=earlier TRAIN q75 per fold; units never pooled", "years": "every observed modern calendar year"},
        "paired": "5000 paired calendar-month block resamples, seed82, match-weighted loss difference; all models vs exp039 plus fixed composition/calibration/recency pairs, overall and all slices; nominal95%, no winner selection or multiplicity-adjusted claims",
        "metrics": "LogLoss, Brier +10bin reliability/resolution/uncertainty/binning residual, ECE10/MCE10, diagnostic calibration slope/intercept, AUC, accuracy (ties half-credit); diagnostic coefficients are not applied to predictions",
        "hypotheses": ["Team/player/consensus composition changes predictive score", "Earlier-only calibration changes proper scores", "Earlier-fit recency weighting changes proper scores", "Experience/disagreement/rating uncertainty stratify forecast errors (association only)"],
        "eligibility_live": 0, "source_timing": "uncertified retrospective day-batched replay; observed first-map rosters, no certified announcement/collection times",
        "selection": "Predeclared for this run, not an untouched preregistration: historical cohorts already inspected; all comparisons diagnostic and no production promotion",
        "forbidden": ["market features", "market teacher labels", "market/Benter students", "bookmaker cohort filters", "live DB", "network", "production changes"],
        "forecast_metadata_columns": IDENTITY + ["tournament", "competition_tier", "roster_min_prior_series", "eligibility_live"],
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}},
    }
    write_json_exclusive(output / "protocol.json", protocol)
    source_hashes = {}
    code_paths = [Path(__file__), ROOT / "scripts/benchmark_siamese_architectures.py", ROOT / "scripts/experiment_rating_stack.py",
                  ROOT / "scripts/train_and_tune_siamese_series.py", ROOT / "scripts/corrected_history_evaluation.py",
                  ROOT / "src/models/symmetric_series.py", ROOT / "src/models/siamese_series.py", ROOT / "src/analysis/probability_metrics.py",
                  ROOT / "betting_app/core/models/engine.py", ROOT / "betting_app/core/models/registry.py",
                  ROOT / "betting_app/services/upcoming_inference_service.py", ROOT / "betting_app/scripts/backfill_operational_predictions.py"]
    for path in [*code_paths, REPLAY / "audit.json", REPLAY / "snapshots.csv"]:
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
    audit = load_json(REPLAY / "audit.json")
    if sha256(REPLAY / "snapshots.csv") != audit["outputs"]["snapshots.csv"]:
        raise ValueError("immutable replay snapshot checksum differs from audit")
    # Explicit sports allowlist: no accidental market column enters fit or scoring.
    from src.models.symmetric_series import _REQUIRED_BASE_FIELDS
    columns = list(dict.fromkeys(IDENTITY + ["BoN", "tournament", "competition_tier", "roster_min_prior_series"] + sorted(_REQUIRED_BASE_FIELDS)))
    frame = checked_frame(REPLAY / "snapshots.csv", columns)
    frame = frame.loc[frame.date >= "2020-01-01"].reset_index(drop=True)
    numeric = list(_REQUIRED_BASE_FIELDS) + ["roster_min_prior_series"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all() or (frame.roster_min_prior_series < 0).any():
        raise ValueError("incomplete sports inputs/experience; no imputation or silent cohort reduction")
    if frame[["tournament", "competition_tier"]].isna().any().any():
        raise ValueError("missing event or tier identity")
    modern_mask = frame.date.ge("2024-01-01").to_numpy()
    modern = frame.loc[modern_mask].set_index("golgg_match_id", drop=False)
    if len(modern) != audit["modern_2024_rows"]:
        raise ValueError("full modern snapshot cohort differs from immutable replay audit")
    predictions = modern[IDENTITY + ["tournament", "competition_tier", "roster_min_prior_series", "days_since_last_1", "days_since_last_2"]].copy()
    predictions["eligibility_live"] = 0
    operational = operational_audit(frame.columns)
    write_json_exclusive(output / "operational_replay_audit.json", operational)
    ancestry, blocked = archive_evidence(frame, modern, predictions, source_hashes)
    blocked["exact_current_operational_replay"] = operational["status"]
    blocked["original_frozen_exp039_chronological_replay"] = "The shared exp039 column is a corrected research refit. Original frozen fit/calibrator historical cutoffs and deployment availability are not established; not substituted or scored as honest earlier-fit operational replay."
    x, names = build_training_features(frame)
    write_json_exclusive(output / "ancestry.json", {"models": ancestry, "blocked": blocked, "source_sha256": source_hashes,
        "replay_audit": {key: audit[key] for key in ("temporal_status", "promotion_blockers", "exclusions", "source", "collection", "same_day_policy")}})
    write_json_exclusive(output / "feature_protocol.json", {"canonical79": names, "ablations": {key: [names[i] for i in values] for key, values in composition_columns(names).items()}, "source_sha256": source_hashes})
    folds, new_models = fit_compositions(frame, modern_mask, predictions, output, x, names)
    models = [name for name in ARCHIVED if name in predictions] + new_models
    if not np.isfinite(predictions[models].to_numpy(float)).all():
        raise ValueError("a fixed model left unforecast modern rows")
    predictions = predictions.reset_index(drop=True)
    predictions.to_csv(output / "predictions.csv", index=False, mode="x")
    overall, paired = score_comparisons(predictions, models, output)
    roster_audit = {
        "canonical_input_structure": "79 engineered scalar match features, not ten per-player identity streams. Player probabilities/minimum Elo/maximum Glicko/mean RD and sigma are already roster aggregates; team ratings summarize overlapping results.",
        "consensus_redundancy": "rating_consensus_logit and its BoN interactions deterministically average12 existing rating logits. Series amplification is another transform of the same input, not independent evidence.",
        "double_counting_claim": "Repeated correlated representations exist; their inclusion in a jointly trained predictor is not itself probabilistic double-counting. This study does not multiply team/player likelihoods or treat ten ratings as independent observations.",
        "diagnostic_correlation": {"mean6_team_player": float(np.corrcoef(frame.loc[modern_mask, [f'team_{s}' for s in SYSTEMS]].mean(axis=1), frame.loc[modern_mask, [f'player_{s}' for s in SYSTEMS]].mean(axis=1))[0, 1])},
        "roster_min_prior_series": "Existing retrospective minimum per-player prior series over actual first-map roster, NOT count of major-level games, roster continuity, or timestamped announced lineup.",
        "causal_limit": "Composition ablations test predictive representations, not causal player influence; changes include feature count/regularization geometry. No independent-player likelihood or causal roster claim is justified.",
    }
    write_json_exclusive(output / "roster_input_audit.json", roster_audit)
    summary = {"version": VERSION, "status": "completed_retrospective_research", "n": len(predictions),
        "date_min": predictions.date.min(), "date_max": predictions.date.max(), "models": models, "overall": overall,
        "paired_overall": paired, "blocked": blocked, "new_folds": folds,
        "fixed_hypothesis_evidence": hypothesis_evidence(paired),
        "unsupported_hypotheses": ["Exact operational70/20/10 or currently deployed EXP081 historical superiority: faithful earlier operational inputs/artifacts unavailable",
            "Live point-in-time eligibility or untouched external confirmation: source timing missing and history already inspected",
            "Team/player causal attribution or independent evidence: aggregate inputs share historical outcomes",
            "Roster continuity/transfer/fatigue mechanism: no individual longitudinal roster or workload fields",
            "Rating RD/sigma or ensemble member spread provides calibrated probability coverage: diagnostic proxies only",
            "Recency weighting establishes optimal strength evolution or a dynamic state process: narrow same-ridge half-life comparison only",
            "Tournament champion/podium calibration: this runner scores series, not event resampling or simulated tournament paths"],
        "claims_policy": "Nominal loss CIs and slices are descriptive; no automatic winner/promotion, no unsupported hypothesis upgraded from favorable test scores.",
        "coverage": {"full_snapshot_modern_cohort": len(predictions), "bookmaker_filter_used": False,
                     "inherited_source_exclusions": audit["exclusions"], "source_collection_complete": audit["collection"]["complete"], "eligibility_live_n": 0},
        "roster_input_audit": roster_audit, "protocol_sha256": sha256(output / "protocol.json")}
    write_json_exclusive(output / "summary.json", summary)
    for relative, digest in source_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"immutable source changed during benchmark: {relative}")
    write_json_exclusive(output / "manifest.json", {"version": VERSION, "source_sha256": source_hashes,
        "outputs": {str(path.relative_to(output)): sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
        "models": models, "eligibility_live": 0, "output_directory_exclusive": True})
    print(json.dumps({"status": summary["status"], "rows": len(predictions), "models": models, "blocked": blocked, "output_dir": str(output)}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="NEW exclusive research output directory")
    run(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
