#!/usr/bin/env python3
"""Evaluate and train a Sym-Cal candidate with shared competition adjustments.

The experiment keeps every legacy rating system's domestic state unchanged.
For a cross-league match, the family/tier posterior learned by the corrected
Glicko-2 engine is converted to log-odds and applied to Elo, TrueSkill,
OpenSkill, Plackett-Luce, and Thurstone-Mosteller probabilities.  Glicko inputs
come directly from the corrected engine.  Evaluation is paired, chronological,
order-symmetric, and expanding-Platt calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.probability_metrics import binary_log_loss_vector, calculate_ece
from src.ratings.competition_adjustment import CompetitionAdjustment, adjust_probability
from src.ratings.family_calibrated_glicko2 import FamilyCalibratedGlicko2, RatingEvent

MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
MODEL_VERSION = "multirating-family-v1"
RATINGS_VERSION = "player-multirating-family-v1"
UPDATE_INTERVAL = 1_000
BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 42
DIAGNOSTIC_START = pd.Timestamp("2024-01-01")
RATING_PROBABILITIES = ("player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm")
ADJUSTED_LEGACY_PROBABILITIES = tuple(name for name in RATING_PROBABILITIES if name != "player_gl")


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment_modules() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    candidate = load_module(
        "exp076_candidate_rating",
        PROJECT_ROOT / "scripts/05_ratingi_baseline/05j_evaluate_calibrated_glicko2.py",
    )
    base = load_module(
        "exp076_w20_base",
        PROJECT_ROOT / "scripts/06_metamodel/06ab_w20_binomial_all_models_bootstrap.py",
    )
    symmetry = load_module(
        "exp076_symmetry",
        PROJECT_ROOT / "scripts/06_metamodel/06ag_team_order_sensitivity_analysis.py",
    )
    calibration = load_module(
        "exp076_calibration",
        PROJECT_ROOT / "scripts/06_metamodel/06ah_calibration_symmetry_diagnostic.py",
    )
    return candidate, base, symmetry, calibration


def replay_shared_adjustments(
    records: Sequence[Any], candidate_module: ModuleType
) -> pd.DataFrame:
    """Replay corrected Glicko state and export pre-match shared offsets."""

    engine = FamilyCalibratedGlicko2()
    domestic: dict[str, Any] = {}
    output: list[dict[str, Any]] = []
    for _, grouped in groupby(records, key=lambda record: record.match_date):
        period = tuple(grouped)
        events: list[RatingEvent] = []
        for match in period:
            affiliation_a, affiliation_b = candidate_module._event_affiliations(match, domestic)
            events.append(
                RatingEvent(
                    event_id=match.match_id,
                    event_date=match.match_date,
                    team_a_id=match.team_1_id,
                    team_b_id=match.team_2_id,
                    players_a=match.players_1,
                    players_b=match.players_2,
                    family_a=affiliation_a.family if affiliation_a else candidate_module.UNKNOWN,
                    family_b=affiliation_b.family if affiliation_b else candidate_module.UNKNOWN,
                    tier_a=affiliation_a.tier if affiliation_a else candidate_module.UNKNOWN,
                    tier_b=affiliation_b.tier if affiliation_b else candidate_module.UNKNOWN,
                    scores=match.scores,
                )
            )

        ordered = engine._validated_events(events)
        player_states, family_states, tier_states = engine._frozen_period_states(
            period[0].match_date, ordered
        )
        probabilities = engine.process_period(events)
        for match, event in zip(period, events, strict=True):
            ranking_affiliations_known = (
                event.family_a != candidate_module.UNKNOWN
                and event.family_b != candidate_module.UNKNOWN
            )
            adjustment_active = (
                match.competition.scope.value == "cross_league"
                and ranking_affiliations_known
            )
            location = (
                engine._location_difference(event, family_states, tier_states)
                if adjustment_active
                else CompetitionAdjustment(mean=0.0, variance=0.0)
            )

            def side_features(
                players: Sequence[str], family: str, tier: str
            ) -> tuple[float, float]:
                if not ranking_affiliations_known:
                    mean = 0.0
                    variance = engine._unknown_location().variance
                else:
                    family_state = family_states[family]
                    tier_state = tier_states[tier]
                    mean = family_state.mean + tier_state.mean
                    variance = family_state.variance + tier_state.variance
                ratings = [player_states[player].rating + mean for player in players]
                deviations = [
                    math.sqrt(player_states[player].rd**2 + variance)
                    for player in players
                ]
                return max(ratings), float(np.mean(deviations))

            max_a, rd_a = side_features(event.players_a, event.family_a, event.tier_a)
            max_b, rd_b = side_features(event.players_b, event.family_b, event.tier_b)
            output.append(
                {
                    "golgg_match_id": match.match_id,
                    "candidate_player_gl": probabilities[match.match_id],
                    "candidate_player_gl_max1": max_a,
                    "candidate_player_gl_max2": max_b,
                    "candidate_player_gl_rd_avg1": rd_a,
                    "candidate_player_gl_rd_avg2": rd_b,
                    "competition_adjustment_mean": location.mean,
                    "competition_adjustment_variance": location.variance,
                    "competition_scope": match.competition.scope.value,
                }
            )
        candidate_module._apply_domestic_affiliations(domestic, period)
    return pd.DataFrame(output)


def safe_match_ids(records: Sequence[Any]) -> set[str]:
    """Exclude every date with repeated participants from both model variants."""

    player_dates = Counter(
        (record.match_date, player)
        for record in records
        for player in (*record.players_1, *record.players_2)
    )
    team_dates = Counter(
        (record.match_date, team)
        for record in records
        for team in (record.team_1_id, record.team_2_id)
    )
    return {
        record.match_id
        for record in records
        if all(
            player_dates[(record.match_date, player)] == 1
            for player in (*record.players_1, *record.players_2)
        )
        and all(
            team_dates[(record.match_date, team)] == 1
            for team in (record.team_1_id, record.team_2_id)
        )
    }


def prepare_frames(
    data_dir: Path,
    records: Sequence[Any],
    shared: pd.DataFrame,
    base_module: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    helper = base_module.load_helper_module()
    helper.PROJECT_ROOT = data_dir.parent
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(base_module.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner")
    data = data.merge(shared, on="golgg_match_id", how="inner")
    data = data.sort_values("date").reset_index(drop=True)
    data, binomial_features = base_module.add_binomial_features(data)
    features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES + binomial_features
    data = data.dropna(subset=features + ["y_true"]).copy()
    data = data[data["date"] >= pd.Timestamp("2020-01-01")]
    safe_ids = safe_match_ids(records)
    before_safety = len(data)
    data = data[data["golgg_match_id"].astype(str).isin(safe_ids)].reset_index(drop=True)

    legacy = data.copy()
    candidate = data.copy()
    candidate["player_gl"] = candidate["candidate_player_gl"]
    for name in ADJUSTED_LEGACY_PROBABILITIES:
        candidate[name] = [
            adjust_probability(
                probability,
                CompetitionAdjustment(mean=mean, variance=variance),
            )
            for probability, mean, variance in zip(
                candidate[name],
                candidate["competition_adjustment_mean"],
                candidate["competition_adjustment_variance"],
                strict=True,
            )
        ]
    for target, source in (
        ("player_gl_max1", "candidate_player_gl_max1"),
        ("player_gl_max2", "candidate_player_gl_max2"),
        ("player_gl_rd_avg1", "candidate_player_gl_rd_avg1"),
        ("player_gl_rd_avg2", "candidate_player_gl_rd_avg2"),
    ):
        candidate[target] = candidate[source]
    best_of = candidate["BoN"].fillna(1).astype(int).to_numpy()
    for name in RATING_PROBABILITIES:
        candidate[f"{name}_binom_series"] = base_module.series_probability(
            candidate[name].to_numpy(dtype=float), best_of
        )
    return legacy, candidate, features, {
        "rows_before_same_day_exclusion": before_safety,
        "rows_after_same_day_exclusion": len(data),
        "excluded_repeated_participant_date": before_safety - len(data),
    }


def final_prediction_stream(
    data: pd.DataFrame,
    features: list[str],
    base_module: ModuleType,
    symmetry_module: ModuleType,
    calibration_module: ModuleType,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_all = symmetry_module.walk_forward_original_and_swapped_test(
        base_module, data, features
    )
    raw = raw_all[raw_all["variant"] == "Order-symmetrized prediction"].copy()
    calibrated = calibration_module.expanding_calibrate_variant(
        raw, "Order-symmetrized prediction"
    )
    final = calibrated[
        (calibrated["calibration"] == "platt_expanding")
        & calibrated["calibrator_available"].astype(bool)
        & (calibrated["date"] >= DIAGNOSTIC_START)
    ].copy()
    return raw, final


def probability_metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    probability = frame[column].to_numpy(dtype=float)
    return {
        "n": len(frame),
        "log_loss": float(log_loss(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "auc": float(roc_auc_score(y_true, probability)),
        "ece": float(calculate_ece(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, probability >= 0.5)),
    }


def bootstrap_delta(frame: pd.DataFrame, delta_column: str) -> dict[str, Any]:
    monthly = frame.groupby("month")[delta_column].agg(delta_sum="sum", n="size")
    sums = monthly["delta_sum"].to_numpy(dtype=float)
    counts = monthly["n"].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    samples = np.empty(BOOTSTRAP_REPETITIONS)
    for index in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(monthly), size=len(monthly))
        samples[index] = sums[selected].sum() / counts[selected].sum()
    return {
        "observed": float(frame[delta_column].mean()),
        "ci_lower_95": float(np.quantile(samples, 0.025)),
        "ci_upper_95": float(np.quantile(samples, 0.975)),
    }


def evaluate_pair(legacy: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    paired = legacy[["golgg_match_id", "date", "y_true", "y_prob"]].rename(
        columns={"y_prob": "legacy_probability"}
    ).merge(
        candidate[["golgg_match_id", "date", "y_true", "y_prob"]].rename(
            columns={"y_prob": "candidate_probability"}
        ),
        on=["golgg_match_id", "date", "y_true"],
        validate="one_to_one",
    )
    paired["month"] = pd.to_datetime(paired["date"]).dt.to_period("M").astype(str)
    paired["delta_log_loss"] = binary_log_loss_vector(
        paired["y_true"].to_numpy(), paired["candidate_probability"].to_numpy()
    ) - binary_log_loss_vector(
        paired["y_true"].to_numpy(), paired["legacy_probability"].to_numpy()
    )
    return paired, {
        "legacy": probability_metrics(paired, "legacy_probability"),
        "candidate": probability_metrics(paired, "candidate_probability"),
        "paired_log_loss_delta": bootstrap_delta(paired, "delta_log_loss"),
    }


def train_candidate_artifact(
    frame: pd.DataFrame,
    features: list[str],
    raw_predictions: pd.DataFrame,
    artifact_dir: Path,
    metadata: dict[str, Any],
) -> None:
    if artifact_dir.exists():
        raise FileExistsError(f"immutable candidate directory already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True)
    pipeline = load_experiment_modules()[1].build_logistic_regression()
    pipeline.fit(frame[features], frame["y_true"].astype(int))
    raw = raw_predictions.sort_values(["date", "golgg_match_id"])
    clipped = np.clip(raw["y_prob"].to_numpy(dtype=float), 0.001, 0.999)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=RANDOM_SEED,
    )
    calibrator.fit(logits, raw["y_true"].astype(int))
    joblib.dump(pipeline, artifact_dir / "pipeline.joblib")
    joblib.dump(calibrator, artifact_dir / "calibrator.joblib")
    dataset_hash = hashlib.sha256(
        frame[["golgg_match_id", "date", "y_true", *features]]
        .sort_values(["date", "golgg_match_id"])
        .to_json(orient="records", date_format="iso", double_precision=10)
        .encode("utf-8")
    ).hexdigest()
    training_data = frame[["golgg_match_id", "date", "y_true", *features]].copy()
    dataset_path = artifact_dir / "train_dataset.parquet"
    try:
        training_data.to_parquet(dataset_path, index=False)
    except (ImportError, ModuleNotFoundError):
        dataset_path = artifact_dir / "train_dataset.csv"
        training_data.to_csv(dataset_path, index=False)
    metadata.update(
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "ratings_version": RATINGS_VERSION,
            "feature_count": len(features),
            "training_rows": len(frame),
            "dataset_hash": dataset_hash,
            "dataset_path": dataset_path.name,
            "feature_names": features,
            "status": "candidate",
            "operational": False,
        }
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/experiments/exp076_multirating_family_symcal",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "betting_app/models/ml"
            / MODEL_NAME
            / MODEL_VERSION
        ),
    )
    args = parser.parse_args()
    if args.artifact_dir.exists():
        raise FileExistsError(
            f"immutable candidate directory already exists: {args.artifact_dir}"
        )
    candidate_module, base_module, symmetry_module, calibration_module = (
        load_experiment_modules()
    )
    records, load_counts = candidate_module.load_matches(args.data_dir / "golgg_matches.json")
    shared = replay_shared_adjustments(records, candidate_module)
    legacy_frame, candidate_frame, features, frame_counts = prepare_frames(
        args.data_dir, records, shared, base_module
    )
    legacy_raw, legacy_final = final_prediction_stream(
        legacy_frame, features, base_module, symmetry_module, calibration_module
    )
    candidate_raw, candidate_final = final_prediction_stream(
        candidate_frame, features, base_module, symmetry_module, calibration_module
    )
    paired, evaluation = evaluate_pair(legacy_final, candidate_final)
    scope_by_id = shared.set_index("golgg_match_id")["competition_scope"]
    paired["competition_scope"] = paired["golgg_match_id"].map(scope_by_id)
    evaluation["cohorts"] = {}
    for scope, group in paired.groupby("competition_scope"):
        evaluation["cohorts"][scope] = {
            "legacy": probability_metrics(group, "legacy_probability"),
            "candidate": probability_metrics(group, "candidate_probability"),
            "paired_log_loss_delta": bootstrap_delta(group, "delta_log_loss"),
        }
    summary = {
        "experiment": "EXP-076",
        "contract": {
            "shared_offset_source": "player-glicko2-family-v1",
            "adjusted_probabilities": list(ADJUSTED_LEGACY_PROBABILITIES),
            "corrected_glicko_probability": "player_gl",
            "domestic_behavior": "identity; shared adjustment activates only for known cross-league affiliations",
            "evaluation_start": DIAGNOSTIC_START.date().isoformat(),
            "update_interval": UPDATE_INTERVAL,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        },
        "load_counts": load_counts,
        "frame_counts": frame_counts,
        "evaluation": evaluation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    train_candidate_artifact(
        candidate_frame,
        features,
        candidate_raw,
        args.artifact_dir,
        {"evaluation": evaluation, "contract": summary["contract"]},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
