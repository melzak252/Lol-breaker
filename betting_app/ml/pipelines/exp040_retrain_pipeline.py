"""Retrain pipeline for the EXP-040 model family (Markov Series + Venn-Abers Conformal Calibration).

Key properties:
1. Hierarchical Markov Series Simulator replaces the naive binomial series
   approximation and averages unknown game-one priority assignments.
2. Temperature calibration and Venn-Abers fitting consume chronological
   out-of-fold scores; their metrics are not presented as held-out Venn-Abers
   performance.
3. Registers the artifact as ``candidate``. EXP-040 has no compatible
   upcoming-feature producer yet, so it must not be marked ``shadow`` and
   scheduled for inference before that source contract exists.

Never modifies or overwrites frozen exp-039 artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

from betting_app.core.config import PROJECT_ROOT
from betting_app.ml.calibration.candidate_calibration import (
    TemperatureScalingCalibrator,
    brier_score_decomposition,
    expected_calibration_error,
)
from betting_app.ml.calibration.venn_abers import VennAbersCalibrator
from betting_app.ml.features.candidate_features import (
    compute_series_side_priority,
    compute_side_advantage,
)
from betting_app.ml.models.markov_series import MarkovSeriesSimulator
from collections import defaultdict, deque
from betting_app.ml.registry.repository import (
    EvaluationRunRecord,
    ModelVersionRecord,
    record_evaluation_run,
    register_model_version,
)
from betting_app.scripts.backtest_exp039_db_market import load_backtest_matches, load_best_of_map
from betting_app.scripts.rebuild_ratings import RATING_SYSTEM_PARAMS, game_score_for_match_team1
from betting_app.scripts.rebuild_w20_features import (
    average_history,
    load_all_games_grouped,
    load_all_player_stats_grouped,
    update_team_history,
)
from betting_app.scripts.train_thesis_model import build_logistic_regression
from betting_app.services.thesis_inference_service import (
    ALL_FEATURES,
    BINOMIAL_FEATURES,
    EPSILON,
    OPTUNA_BASE_FEATURES,
    RANK_PROB_FEATURES,
    _logit,
    _swap_feature_vector,
    _symmetrize,
)
from src.ratings.manager import RatingManager

logger = logging.getLogger(__name__)

MODEL_NAME = "Hierarchical-Markov-VennAbers-EXP040"
FEATURE_VERSION = "exp040-markov-va-v1"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts" / "exp040"


@dataclass(frozen=True)
class Exp040RetrainConfig:
    model_name: str = MODEL_NAME
    model_version: str | None = None
    min_date: str = "2020-01-01"
    limit: int | None = None
    initial_train_before: str = "2021-01-01"
    update_interval: int = 1000
    artifact_root: str = str(DEFAULT_ARTIFACT_ROOT)
    register_model: bool = True
    status_on_success: str = "candidate"
    min_shadow_log_loss: float = 0.62
    min_shadow_auc: float = 0.72


@dataclass(frozen=True)
class Exp040RetrainResult:
    model_name: str
    model_version: str
    artifact_dir: str
    pipeline_path: str
    calibrator_path: str
    metadata_path: str
    dataset_path: str
    dataset_hash: str
    n_matches: int
    n_features: int
    registered_status: str
    metrics: dict[str, Any]


def _default_model_version() -> str:
    return datetime.now(UTC).strftime("exp040-candidate-%Y%m%d-%H%M%S")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _dataset_hash(frame: pd.DataFrame) -> str:
    columns = ["golgg_match_id", "date", "y_true", *ALL_FEATURES]
    payload = frame[columns].sort_values(["date", "golgg_match_id"]).to_json(
        orient="records", date_format="iso", double_precision=10
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidate_dataset(min_date: str, limit: int | None = None) -> pd.DataFrame:
    """Build chronological dataset augmenting base features with Markov Series simulation."""
    matches = load_backtest_matches(after_date=min_date, limit=limit)
    if not matches:
        raise RuntimeError(f"No GOL.GG matches found on or after {min_date}")

    best_of_map = load_best_of_map()
    games_by_match = load_all_games_grouped()
    player_stats_by_game_side = load_all_player_stats_grouped()

    manager = RatingManager(RATING_SYSTEM_PARAMS)
    markov_sim = MarkovSeriesSimulator()
    team_history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=20))
    team_match_ids: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=20))
    rows: list[dict[str, Any]] = []

    for match in tqdm(matches, desc="Building EXP-040 candidate dataset"):
        if not match.players1 or not match.players2 or not match.games:
            continue
        manager.update_before_match(match.team1_id, match.team2_id, match.players1, match.players2, match.match_date)
        ratings = manager.predict_match(match.team1_id, match.team2_id, match.players1, match.players2)
        t1_hist = average_history(team_history.get(match.team1_id))
        t2_hist = average_history(team_history.get(match.team2_id))

        features: dict[str, float] = {f: float(ratings.get(f, 0.5)) for f in OPTUNA_BASE_FEATURES}
        for stat in ["win_rate", "kills", "deaths", "gd15", "dpm", "vspm", "towers", "nashors", "gold", "duration"]:
            features[f"t1_rolling_{stat}"] = float(t1_hist[stat])
            features[f"t2_rolling_{stat}"] = float(t2_hist[stat])

        best_of = best_of_map.get(match.match_id, 1)
        # Historical rows lack a trustworthy Game-1 side-selection label.
        # Averaging both assignments preserves team-order symmetry.
        for i, rank_feat in enumerate(RANK_PROB_FEATURES):
            p_neutral = float(features[rank_feat])
            binom_feat = BINOMIAL_FEATURES[i]
            with_priority = markov_sim.predict_series_proba(
                p_neutral_a=p_neutral,
                team_a_has_game1_priority=True,
                best_of=int(best_of),
                blue_side_bonus=0.22,
            )
            without_priority = markov_sim.predict_series_proba(
                p_neutral_a=p_neutral,
                team_a_has_game1_priority=False,
                best_of=int(best_of),
                blue_side_bonus=0.22,
            )
            features[binom_feat] = (with_priority + without_priority) / 2.0

        scores = [game_score_for_match_team1(match, g) for g in match.games]
        y_true = int(sum(scores) > len(scores) / 2)
        rows.append({
            "golgg_match_id": str(match.match_id),
            "date": pd.Timestamp(match.match_date),
            "team1_id": match.team1_id,
            "team2_id": match.team2_id,
            "team1_name": match.team1_name,
            "team2_name": match.team2_name,
            "best_of": int(best_of),
            "n_games": int(len(scores)),
            "y_true": y_true,
            **features,
        })

        for game in match.games:
            score_1 = game_score_for_match_team1(match, game)
            manager.update_after_game(match.team1_id, match.team2_id, match.players1, match.players2, score_1, 1 - score_1)
        manager.update_after_match(match.team1_id, match.team2_id, match.players1, match.players2, scores)

        for game in games_by_match.get(str(match.match_id), []):
            update_team_history(team_history, team_match_ids, match.team1_id, match.match_id, game, player_stats_by_game_side)
            update_team_history(team_history, team_match_ids, match.team2_id, match.match_id, game, player_stats_by_game_side)

    frame = pd.DataFrame(rows).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    return frame.dropna(subset=[*ALL_FEATURES, "y_true"]).reset_index(drop=True)


def train_oof_and_final(
    frame: pd.DataFrame,
    *,
    initial_train_before: str,
    update_interval: int,
) -> tuple[Any, TemperatureScalingCalibrator, VennAbersCalibrator, dict[str, Any]]:
    """Walk-forward training with strictly nested out-of-fold calibration."""
    oof_predictions: list[float] = []
    oof_actuals: list[int] = []

    test_start_idx = int(np.argmax(pd.to_datetime(frame["date"]) >= pd.to_datetime(initial_train_before)))
    if test_start_idx == 0 and pd.to_datetime(frame["date"].iloc[0]) >= pd.to_datetime(initial_train_before):
        test_start_idx = min(len(frame) // 4, 1000)

    for step in range(test_start_idx, len(frame), update_interval):
        train_df = frame.iloc[:step]
        chunk = frame.iloc[step : min(step + update_interval, len(frame))]
        if chunk.empty or len(train_df["y_true"].unique()) < 2:
            continue

        model = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000)
        model.fit(train_df[ALL_FEATURES], train_df["y_true"].astype(int))

        original = np.clip(model.predict_proba(chunk[ALL_FEATURES])[:, 1], EPSILON, 1.0 - EPSILON)
        swapped = np.vstack([_swap_feature_vector(row.reshape(1, -1))[0] for row in chunk[ALL_FEATURES].to_numpy(dtype=float)])
        swapped_prob = np.clip(model.predict_proba(swapped)[:, 1], EPSILON, 1.0 - EPSILON)
        p_sym = np.array([_symmetrize(o, s) for o, s in zip(original, swapped_prob)], dtype=float)

        y = chunk["y_true"].astype(int).to_numpy()
        oof_predictions.extend(p_sym.tolist())
        oof_actuals.extend(y.tolist())

    oof_p = np.array(oof_predictions, dtype=float)
    oof_y = np.array(oof_actuals, dtype=int)

    # Nested calibration fitted on OOF walk-forward predictions
    z_oof = _logit(np.clip(oof_p, 1e-4, 1.0 - 1e-4))
    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(z_oof, oof_y)
    calibrated_oof = calibrator.transform(z_oof)

    final_pipeline = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000)
    final_pipeline.fit(frame[ALL_FEATURES], frame["y_true"].astype(int))

    # Fit the Venn-Abers layer only on chronological OOF scores. It is an
    # inference component, not an in-sample performance claim.
    venn_abers = VennAbersCalibrator().fit(calibrated_oof, oof_y)

    def _eval(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        return {
            "n": int(len(y)),
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
            "ece": float(expected_calibration_error(y, p, n_bins=10)),
            "brier_decomp": brier_score_decomposition(y, p),
        }

    metrics = {
        "oof_uncalibrated": _eval(oof_y, oof_p),
        "oof_temperature_calibrated": _eval(oof_y, calibrated_oof),
        "calibrator_temperature": float(calibrator.temperature_),
        "venn_abers_calibration_examples": int(len(oof_y)),
        "dataset_size": int(len(frame)),
        "feature_count": len(ALL_FEATURES),
    }

    return final_pipeline, calibrator, venn_abers, metrics


def run_exp040_retrain(config: Exp040RetrainConfig | None = None) -> Exp040RetrainResult:
    cfg = config or Exp040RetrainConfig()
    version = cfg.model_version or _default_model_version()
    started = datetime.now(UTC).isoformat(timespec="seconds")

    logger.info("Starting EXP-040 candidate retrain for %s (%s)", cfg.model_name, version)
    frame = build_candidate_dataset(min_date=cfg.min_date, limit=cfg.limit)
    dataset_hash = _dataset_hash(frame)

    pipeline, calibrator, venn_abers, metrics = train_oof_and_final(
        frame,
        initial_train_before=cfg.initial_train_before,
        update_interval=cfg.update_interval,
    )

    artifact_dir = Path(cfg.artifact_root) / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = artifact_dir / "pipeline.joblib"
    calibrator_path = artifact_dir / "calibrator.joblib"
    metadata_path = artifact_dir / "metadata.json"
    dataset_path = artifact_dir / "training_dataset.parquet"

    joblib.dump(
        {
            "estimator": pipeline,
            "feature_names": ALL_FEATURES,
            "temperature_calibrator": calibrator,
            "venn_abers_calibrator": venn_abers,
        },
        pipeline_path,
    )
    # Retain the temperature artifact for inspection; registry inference uses
    # the self-contained pipeline bundle above.
    joblib.dump(calibrator, calibrator_path)
    try:
        frame.to_parquet(dataset_path, index=False)
    except Exception:
        dataset_path = artifact_dir / "training_dataset.csv"
        frame.to_csv(dataset_path, index=False)

    oof_cal = metrics["oof_temperature_calibrated"]
    status = (
        cfg.status_on_success
        if oof_cal["log_loss"] <= cfg.min_shadow_log_loss and oof_cal["auc"] >= cfg.min_shadow_auc
        else "candidate"
    )

    metadata = {
        "model_name": cfg.model_name,
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_start_at": started,
        "training_examples": int(len(frame)),
        "feature_names": ALL_FEATURES,
        "dataset_hash": dataset_hash,
        "dataset_path": str(dataset_path),
        "pipeline_path": str(pipeline_path),
        "calibrator_path": str(calibrator_path),
        "git_commit": _git_commit(),
        "config": asdict(cfg),
        "metrics": metrics,
        "notes": "EXP-040 candidate artifact with Markov Series simulation, chronological OOF temperature calibration, and Venn-Abers inference bounds; frozen exp-039 is preserved.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if cfg.register_model:
        register_model_version(ModelVersionRecord(
            model_name=cfg.model_name,
            model_version=version,
            status=status,
            artifact_path=str(pipeline_path),
            feature_version=FEATURE_VERSION,
            training_start_at=started,
            training_end_at=datetime.now(UTC).isoformat(timespec="seconds"),
            dataset_hash=dataset_hash,
            git_commit=metadata["git_commit"],
            metrics=metrics,
            notes="EXP-040 candidate with hierarchical Markov simulation and calibrated Venn-Abers bounds; not inference-eligible until an upcoming-feature producer is added.",
        ))
        record_evaluation_run(EvaluationRunRecord(
            model_name=cfg.model_name,
            model_version=version,
            run_type="exp040_candidate_retrain",
            status="completed",
            config=asdict(cfg),
            metrics=metrics,
            notes="EXP-040 candidate retrain",
        ))

    return Exp040RetrainResult(
        model_name=cfg.model_name,
        model_version=version,
        artifact_dir=str(artifact_dir),
        pipeline_path=str(pipeline_path),
        calibrator_path=str(calibrator_path),
        metadata_path=str(metadata_path),
        dataset_path=str(dataset_path),
        dataset_hash=dataset_hash,
        n_matches=int(len(frame)),
        n_features=len(ALL_FEATURES),
        registered_status=status,
        metrics=metrics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain EXP-040 candidate model with Markov simulation and nested calibration")
    parser.add_argument("--model-version")
    parser.add_argument("--min-date", default="2020-01-01")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--initial-train-before", default="2021-01-01")
    parser.add_argument("--update-interval", type=int, default=1000)
    parser.add_argument("--no-register", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    config = Exp040RetrainConfig(
        model_version=args.model_version,
        min_date=args.min_date,
        limit=args.limit,
        initial_train_before=args.initial_train_before,
        update_interval=args.update_interval,
        register_model=not args.no_register,
    )
    result = run_exp040_retrain(config)
    print("\nEXP-040 Retrain Successful:")
    print(f"  Model Name:         {result.model_name}")
    print(f"  Model Version:      {result.model_version}")
    print(f"  Registered Status:  {result.registered_status}")
    print(f"  Artifact Directory: {result.artifact_dir}")
    print(f"  OOF Log Loss:       {result.metrics['oof_temperature_calibrated']['log_loss']:.4f}")
    print(f"  OOF ECE:            {result.metrics['oof_temperature_calibrated']['ece']:.4f}")
    print(f"  Calibrator Temp T:  {result.metrics['calibrator_temperature']:.3f}")


if __name__ == "__main__":
    main()
