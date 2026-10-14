"""CLI for the weekly retraining pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from betting_app.ml.pipelines.weekly_retrain import WeeklyRetrainConfig, run_weekly_retrain_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and register an explicitly named tabular candidate")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version")
    parser.add_argument("--feature-version")
    parser.add_argument("--ratings-version")
    parser.add_argument("--days-back", type=int)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=80)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--artifact-root", default="betting_app/models/ml")
    parser.add_argument("--status-on-success", default="candidate", choices=["candidate", "shadow"])
    parser.add_argument("--min-shadow-dataset-size", type=int, default=500)
    parser.add_argument("--max-shadow-log-loss", type=float, default=0.70)
    parser.add_argument("--min-shadow-accuracy", type=float, default=0.52)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = WeeklyRetrainConfig(
        model_name=args.model_name,
        model_version=args.model_version,
        feature_version=args.feature_version,
        ratings_version=args.ratings_version,
        days_back=args.days_back,
        min_features=args.min_features,
        min_train_size=args.min_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        artifact_root=args.artifact_root,
        register_model=not args.no_register,
        status_on_success=args.status_on_success,
        min_shadow_dataset_size=args.min_shadow_dataset_size,
        max_shadow_log_loss=args.max_shadow_log_loss,
        min_shadow_accuracy=args.min_shadow_accuracy,
    )
    result = run_weekly_retrain_pipeline(config)
    payload = {
        "model_name": result.artifact.model_name,
        "model_version": result.artifact.model_version,
        "artifact_path": result.artifact.artifact_path,
        "metadata_path": result.artifact.metadata_path,
        "dataset_size": result.dataset_size,
        "feature_count": result.feature_count,
        "best_candidate": result.best_evaluation.candidate.name,
        "mean_log_loss": round(result.best_evaluation.mean_log_loss, 6),
        "mean_brier": round(result.best_evaluation.mean_brier, 6),
        "mean_accuracy": round(result.best_evaluation.mean_accuracy, 6),
        "candidate_count": len(result.all_evaluations),
        "evaluation_run_id": result.evaluation_run_id,
        "registered_status": result.registered_status,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Model: {payload['model_name']} / {payload['model_version']}")
        print(f"Best candidate: {payload['best_candidate']}")
        print(f"Dataset: {payload['dataset_size']} examples, {payload['feature_count']} features")
        print(f"LogLoss: {payload['mean_log_loss']} | Brier: {payload['mean_brier']} | Acc: {payload['mean_accuracy']}")
        print(f"Artifact: {Path(payload['artifact_path'])}")
        print(f"Registered status: {payload['registered_status']}")
        print(f"Evaluation run: {payload['evaluation_run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
