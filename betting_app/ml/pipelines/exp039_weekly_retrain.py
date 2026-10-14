"""Weekly retraining for the EXP-039 model family.

This pipeline intentionally does **not** overwrite the frozen thesis artifact
``Sym-Cal LR-ElasticNet-W20-Binomial/exp-039``.  It retrains the same model
family (46 EXP-039 features, order symmetry, Platt calibration) from the live
GOL.GG/rating database and registers a separate immutable version, e.g.
``exp039-weekly-20260730-031500``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

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
    _series_probability,
    _swap_feature_vector,
    _symmetrize,
)
from src.ratings.manager import RatingManager


MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
DEFAULT_ARTIFACT_ROOT = Path("betting_app/models/ml")
FEATURE_VERSION = "exp039-db-retrain-v1"


@dataclass(frozen=True)
class Exp039RetrainConfig:
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
class Exp039RetrainResult:
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
    return datetime.now(UTC).strftime("exp039-weekly-%Y%m%d-%H%M%S")


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


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), EPSILON, 1.0 - EPSILON)
    y = np.asarray(y, dtype=int)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "mean_prob": float(np.mean(p)),
        "target_rate": float(np.mean(y)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = None
    return out


def build_exp039_training_frame(min_date: str = "2020-01-01", limit: int | None = None) -> pd.DataFrame:
    """Materialize leakage-safe EXP-039 training rows from the DB.

    Ratings and W20 features are predicted/read before each historical match,
    then updated after match games. This mirrors the DB backtest and upcoming
    inference feature schema, but saves raw feature columns for retraining.
    """
    matches = load_backtest_matches(limit=limit, after_date=min_date)
    best_of_map = load_best_of_map()
    games_by_match = load_all_games_grouped()
    player_stats_by_game_side = load_all_player_stats_grouped()

    manager = RatingManager(RATING_SYSTEM_PARAMS)
    team_history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=20))
    team_match_ids: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=20))
    rows: list[dict[str, Any]] = []

    for match in tqdm(matches, desc="Build EXP-039 retrain dataset"):
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
        series_probs = _series_probability(np.array([features[f] for f in RANK_PROB_FEATURES], dtype=float), best_of)
        for i, feature in enumerate(BINOMIAL_FEATURES):
            features[feature] = float(series_probs[i])

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


def train_oof_and_final(frame: pd.DataFrame, *, initial_train_before: str, update_interval: int) -> tuple[Any, Any, dict[str, Any]]:
    initial_cutoff = pd.Timestamp(initial_train_before)
    train_df = frame[frame["date"] < initial_cutoff].copy()
    test_pool = frame[frame["date"] >= initial_cutoff].copy()
    if len(train_df) < 100 or len(test_pool) < 20:
        raise ValueError(f"Not enough data for walk-forward: train={len(train_df)} test={len(test_pool)}")

    oof_probs: list[np.ndarray] = []
    oof_true: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(test_pool), update_interval), desc="EXP-039 retrain walk-forward"):
        chunk = test_pool.iloc[start : start + update_interval].copy()
        model = build_logistic_regression()
        model.fit(train_df[ALL_FEATURES], train_df["y_true"].astype(int))

        original = np.clip(model.predict_proba(chunk[ALL_FEATURES])[:, 1], EPSILON, 1.0 - EPSILON)
        swapped = np.vstack([_swap_feature_vector(row.reshape(1, -1))[0] for row in chunk[ALL_FEATURES].to_numpy(dtype=float)])
        swapped_prob = np.clip(model.predict_proba(swapped)[:, 1], EPSILON, 1.0 - EPSILON)
        p_sym = np.array([_symmetrize(o, s) for o, s in zip(original, swapped_prob)], dtype=float)
        y = chunk["y_true"].astype(int).to_numpy()
        oof_probs.append(p_sym)
        oof_true.append(y)
        fold_metrics.append({
            "start_date": str(chunk["date"].min().date()),
            "end_date": str(chunk["date"].max().date()),
            **_safe_metrics(y, p_sym),
        })
        train_df = pd.concat([train_df, chunk], ignore_index=True)

    oof_p = np.concatenate(oof_probs)
    oof_y = np.concatenate(oof_true)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(_logit(oof_p), oof_y)
    oof_cal = np.clip(calibrator.predict_proba(_logit(oof_p))[:, 1], EPSILON, 1.0 - EPSILON)

    final_pipeline = build_logistic_regression()
    final_pipeline.fit(frame[ALL_FEATURES], frame["y_true"].astype(int))

    metrics = {
        "oof_uncalibrated": _safe_metrics(oof_y, oof_p),
        "oof_calibrated": _safe_metrics(oof_y, oof_cal),
        "folds": fold_metrics,
        "initial_train_before": initial_train_before,
        "update_interval": int(update_interval),
    }
    return final_pipeline, calibrator, metrics


def run_exp039_weekly_retrain(config: Exp039RetrainConfig | None = None) -> Exp039RetrainResult:
    cfg = config or Exp039RetrainConfig()
    version = cfg.model_version or _default_model_version()
    started = datetime.now(UTC).isoformat(timespec="seconds")

    frame = build_exp039_training_frame(min_date=cfg.min_date, limit=cfg.limit)
    if len(frame) == 0:
        raise ValueError("EXP-039 retrain dataset is empty")
    pipeline, calibrator, metrics = train_oof_and_final(
        frame,
        initial_train_before=cfg.initial_train_before,
        update_interval=cfg.update_interval,
    )
    dataset_hash = _dataset_hash(frame)
    metrics.update({
        "dataset_size": int(len(frame)),
        "feature_count": len(ALL_FEATURES),
        "dataset_hash": dataset_hash,
    })

    artifact_dir = Path(cfg.artifact_root) / cfg.model_name / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = artifact_dir / "pipeline.joblib"
    calibrator_path = artifact_dir / "calibrator.joblib"
    metadata_path = artifact_dir / "metadata.json"
    dataset_path = artifact_dir / "train_dataset.parquet"
    csv_dataset_path = artifact_dir / "train_dataset.csv"
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(calibrator, calibrator_path)
    try:
        frame.to_parquet(dataset_path, index=False)
    except Exception:
        dataset_path = csv_dataset_path
        frame.to_csv(dataset_path, index=False)

    status = cfg.status_on_success
    gate_passed = True
    gate_reasons: list[str] = []
    if status == "shadow":
        oof = metrics["oof_calibrated"]
        if float(oof["log_loss"]) > cfg.min_shadow_log_loss:
            gate_passed = False
            gate_reasons.append(f"oof_log_loss {oof['log_loss']:.6f} > {cfg.min_shadow_log_loss}")
        if oof.get("auc") is None or float(oof["auc"]) < cfg.min_shadow_auc:
            gate_passed = False
            gate_reasons.append(f"oof_auc {oof.get('auc')} < {cfg.min_shadow_auc}")
        status = "shadow" if gate_passed else "candidate"
    metrics["shadow_quality_gate"] = {
        "requested_status": cfg.status_on_success,
        "registered_status": status,
        "passed": gate_passed,
        "reasons": gate_reasons,
        "thresholds": {"min_shadow_log_loss": cfg.min_shadow_log_loss, "min_shadow_auc": cfg.min_shadow_auc},
    }

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
        "notes": "Retrained EXP-039 family artifact; frozen exp-039 artifact is not overwritten.",
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
            notes="EXP-039 family weekly retrain; candidate unless explicitly promoted after evaluation.",
        ))
        record_evaluation_run(EvaluationRunRecord(
            model_name=cfg.model_name,
            model_version=version,
            run_type="exp039_weekly_retrain",
            status="completed",
            config=asdict(cfg),
            metrics=metrics,
            notes="EXP-039 family weekly retrain",
        ))

    return Exp039RetrainResult(
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
    parser = argparse.ArgumentParser(description="Retrain the EXP-039 model family from DB history")
    parser.add_argument("--model-version")
    parser.add_argument("--min-date", default="2020-01-01")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--initial-train-before", default="2021-01-01")
    parser.add_argument("--update-interval", type=int, default=1000)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--status-on-success", default="candidate", choices=["candidate", "shadow"])
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_exp039_weekly_retrain(Exp039RetrainConfig(
        model_version=args.model_version,
        min_date=args.min_date,
        limit=args.limit,
        initial_train_before=args.initial_train_before,
        update_interval=args.update_interval,
        artifact_root=args.artifact_root,
        register_model=not args.no_register,
        status_on_success=args.status_on_success,
    ))
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
