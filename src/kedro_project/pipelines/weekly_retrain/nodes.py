"""Kedro nodes wrapping the weekly retraining pipeline.

Each node delegates to the existing plain-Python pipeline in
``betting_app.ml.pipelines.weekly_retrain``, keeping all business logic
in one place while gaining Kedro's configuration, logging, and
reproducibility benefits.
"""

from __future__ import annotations

from typing import Any

from betting_app.ml.pipelines.weekly_retrain import (
    WeeklyRetrainConfig,
    WeeklyRetrainResult,
    run_weekly_retrain_pipeline,
)


def run_weekly_retrain(params: dict[str, Any]) -> dict[str, Any]:
    """Run the weekly retraining pipeline with Kedro-provided parameters.

    Parameters
    ----------
    params : dict
        Parameters from ``conf/base/parameters.yml`` under the
        ``weekly_retrain`` key.

    Returns
    -------
    dict
        Summary metrics of the retraining run.
    """
    config = WeeklyRetrainConfig(
        model_name=params.get("model_name", "Operational-Retrained-Tabular"),
        min_features=params.get("min_features", 5),
        min_train_size=params.get("min_train_size", 80),
        test_size=params.get("test_size", 20),
        step_size=params.get("step_size"),
        train_window_days=params.get("train_window_days"),
        register_model=params.get("register_model", True),
        status_on_success=params.get("status_on_success", "candidate"),
        min_shadow_dataset_size=params.get("min_shadow_dataset_size", 0),
        max_shadow_log_loss=params.get("max_shadow_log_loss", 1.0),
        min_shadow_accuracy=params.get("min_shadow_accuracy", 0.0),
        run_type=params.get("run_type", "weekly_retrain"),
    )
    result: WeeklyRetrainResult = run_weekly_retrain_pipeline(config)
    return {
        "model_version": result.artifact.model_version,
        "dataset_size": result.dataset_size,
        "feature_count": result.feature_count,
        "registered_status": result.registered_status,
        "best_candidate": result.best_evaluation.candidate.name,
        "mean_log_loss": result.best_evaluation.mean_log_loss,
        "mean_brier": result.best_evaluation.mean_brier,
        "mean_accuracy": result.best_evaluation.mean_accuracy,
        "evaluation_run_id": result.evaluation_run_id,
    }
