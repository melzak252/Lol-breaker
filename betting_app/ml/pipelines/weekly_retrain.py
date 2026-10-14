"""Weekly retraining pipeline for production candidate models.

This is intentionally a plain-Python pipeline: it can be executed directly in
Docker today and later wrapped as Kedro nodes without changing the core logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from betting_app.core.db import get_session
from betting_app.ml.registry.repository import EvaluationRunRecord, ModelVersionRecord, record_evaluation_run, register_model_version
from betting_app.ml.training.artifacts import DEFAULT_ARTIFACT_ROOT, compute_training_dataset_hash, train_and_save_model
from betting_app.ml.training.candidates import default_candidate_specs
from betting_app.ml.training.features import load_training_dataset
from betting_app.ml.training.types import CandidateEvaluation, ModelCandidateSpec, TrainedModelArtifact
from betting_app.ml.training.walk_forward import evaluate_candidate_walk_forward, select_best_evaluation


@dataclass(frozen=True)
class WeeklyRetrainConfig:
    model_name: str | None = None
    model_version: str | None = None
    feature_version: str | None = None
    ratings_version: str | None = None
    days_back: int | None = None
    min_features: int = 5
    min_train_size: int = 80
    test_size: int = 20
    step_size: int | None = None
    train_window_days: int | None = None
    artifact_root: str = str(DEFAULT_ARTIFACT_ROOT)
    register_model: bool = True
    status_on_success: str = "candidate"
    min_shadow_dataset_size: int = 0
    max_shadow_log_loss: float = 1.0
    min_shadow_accuracy: float = 0.0
    run_type: str = "weekly_retrain"


@dataclass(frozen=True)
class WeeklyRetrainResult:
    artifact: TrainedModelArtifact
    best_evaluation: CandidateEvaluation
    all_evaluations: list[CandidateEvaluation]
    evaluation_run_id: str
    dataset_size: int
    feature_count: int
    registered_status: str


def _shadow_quality_gate(metrics: dict[str, Any], cfg: WeeklyRetrainConfig) -> tuple[bool, list[str]]:
    """Conservative guardrail before a weekly retrain becomes a shadow model.

    Shadow models are picked up by scheduled inference.  A weak retrain should
    still be saved and registered for inspection, but only as ``candidate`` so
    it cannot pollute shadow predictions.
    """
    failures: list[str] = []
    dataset_size = int(metrics.get("dataset_size") or 0)
    best = metrics.get("best_candidate") or {}
    mean_log_loss = best.get("mean_log_loss")
    mean_accuracy = best.get("mean_accuracy")

    if dataset_size < cfg.min_shadow_dataset_size:
        failures.append(f"dataset_size {dataset_size} < {cfg.min_shadow_dataset_size}")
    if mean_log_loss is None:
        failures.append("mean_log_loss missing")
    elif float(mean_log_loss) > cfg.max_shadow_log_loss:
        failures.append(f"mean_log_loss {mean_log_loss} > {cfg.max_shadow_log_loss}")
    if mean_accuracy is None:
        failures.append("mean_accuracy missing")
    elif float(mean_accuracy) < cfg.min_shadow_accuracy:
        failures.append(f"mean_accuracy {mean_accuracy} < {cfg.min_shadow_accuracy}")

    return not failures, failures


def _default_model_version() -> str:
    return datetime.now(UTC).strftime("weekly-%Y%m%d-%H%M%S")


def _evaluation_to_metrics(evaluation: CandidateEvaluation) -> dict[str, Any]:
    return {
        "candidate_name": evaluation.candidate.name,
        "estimator_type": evaluation.candidate.estimator_type,
        "candidate_params": evaluation.candidate.params,
        "folds": len(evaluation.folds),
        "mean_log_loss": round(evaluation.mean_log_loss, 6),
        "mean_brier": round(evaluation.mean_brier, 6),
        "mean_accuracy": round(evaluation.mean_accuracy, 6),
        "fold_results": [asdict(fold) for fold in evaluation.folds],
    }


def run_weekly_retrain_pipeline(
    config: WeeklyRetrainConfig | None = None,
    *,
    candidate_specs: list[ModelCandidateSpec] | None = None,
    session: Session | None = None,
) -> WeeklyRetrainResult:
    """Train candidate models, pick best walk-forward result and save artifact."""
    cfg = config or WeeklyRetrainConfig()
    if not cfg.model_name:
        raise ValueError("model_name must be provided explicitly")
    model_version = cfg.model_version or _default_model_version()
    own_session = session is None
    sess = session or get_session()
    try:
        dataset = load_training_dataset(
            feature_version=cfg.feature_version,
            ratings_version=cfg.ratings_version,
            days_back=cfg.days_back,
            min_features=cfg.min_features,
        )
        if dataset.size == 0:
            raise ValueError("Training dataset is empty")

        specs = candidate_specs or default_candidate_specs()
        evaluations = [
            evaluate_candidate_walk_forward(
                dataset,
                spec,
                min_train_size=cfg.min_train_size,
                test_size=cfg.test_size,
                step_size=cfg.step_size,
                train_window_days=cfg.train_window_days,
            )
            for spec in specs
        ]
        best = select_best_evaluation(evaluations)
        metrics: dict[str, Any] = {
            "dataset_size": dataset.size,
            "feature_count": len(dataset.feature_names),
            "dataset_hash": compute_training_dataset_hash(dataset, dataset.feature_names),
            "best_candidate": _evaluation_to_metrics(best),
            "candidates": [_evaluation_to_metrics(ev) for ev in evaluations],
        }

        artifact = train_and_save_model(
            dataset,
            best.candidate,
            model_name=cfg.model_name,
            model_version=model_version,
            metrics=metrics,
            artifact_root=Path(cfg.artifact_root),
        )
        metrics = artifact.metrics
        requested_status = cfg.status_on_success
        registered_status = requested_status
        gate_reasons: list[str] = []
        if requested_status == "shadow":
            gate_passed, gate_reasons = _shadow_quality_gate(metrics, cfg)
            registered_status = "shadow" if gate_passed else "candidate"
            metrics["shadow_quality_gate"] = {
                "passed": gate_passed,
                "requested_status": requested_status,
                "registered_status": registered_status,
                "reasons": gate_reasons,
                "thresholds": {
                    "min_shadow_dataset_size": cfg.min_shadow_dataset_size,
                    "max_shadow_log_loss": cfg.max_shadow_log_loss,
                    "min_shadow_accuracy": cfg.min_shadow_accuracy,
                },
            }
            metadata_path = Path(artifact.metadata_path)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["metrics"] = metrics
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        if cfg.register_model:
            register_model_version(
                ModelVersionRecord(
                    model_name=cfg.model_name,
                    model_version=model_version,
                    status=registered_status,
                    artifact_path=artifact.artifact_path,
                    feature_version=cfg.feature_version,
                    dataset_hash=str(metrics["dataset_hash"]),
                    metrics=metrics,
                    notes=(
                        "Auto-registered by weekly retraining pipeline"
                        if not gate_reasons
                        else "Auto-registered as candidate; shadow quality gate failed: " + "; ".join(gate_reasons)
                    ),
                ),
                session=sess,
            )

        run = record_evaluation_run(
            EvaluationRunRecord(
                model_name=cfg.model_name,
                model_version=model_version,
                run_type=cfg.run_type,
                status="completed",
                config=asdict(cfg),
                metrics=metrics,
                notes="Weekly retraining candidate selection",
            ),
            session=sess,
        )
        return WeeklyRetrainResult(
            artifact=artifact,
            best_evaluation=best,
            all_evaluations=evaluations,
            evaluation_run_id=run.id,
            dataset_size=dataset.size,
            feature_count=len(dataset.feature_names),
            registered_status=registered_status,
        )
    finally:
        if own_session:
            sess.close()
