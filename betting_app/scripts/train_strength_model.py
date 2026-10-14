"""Train EXP-047 strength model on EXP-046 leakage-safe dataset.

This is an offline training/evaluation CLI.  It does not change production
prediction paths; it writes a versioned model artifact under
``betting_app/models/ml`` for review and later promotion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from betting_app.core.db import init_db
from betting_app.ml.training.strength_dataset import StrengthDatasetConfig, build_strength_dataset_from_db
from betting_app.ml.training.strength_model import (
    StrengthModelConfig,
    save_strength_artifacts,
    train_strength_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EXP-047 leakage-safe strength model")
    parser.add_argument("--min-date", default="2015-01-01", help="Earliest GOL.GG match date to use")
    parser.add_argument("--max-date", default=None, help="Latest GOL.GG match date to use")
    parser.add_argument("--min-prior-matches", type=int, default=3, help="Minimum prior matches per team before emitting a row")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional raw GOL.GG row limit for smoke tests")
    parser.add_argument("--initial-train-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--min-fold-train-size", type=int, default=500)
    parser.add_argument("--logistic-c", type=float, default=0.20)
    parser.add_argument("--l1-ratio", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--model-name", default="Strength-Calibrated-LR")
    parser.add_argument("--model-version", default="exp-047")
    parser.add_argument("--artifact-root", default="betting_app/models/ml")
    parser.add_argument("--no-order-augmentation", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Run evaluation without writing artifacts")
    parser.add_argument("--json-output", default=None, help="Optional path for metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()

    dataset_config = StrengthDatasetConfig(
        min_date=args.min_date,
        max_date=args.max_date,
        min_prior_matches=args.min_prior_matches,
        limit_rows=args.limit_rows,
    )
    model_config = StrengthModelConfig(
        model_name=args.model_name,
        model_version=args.model_version,
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_fold_train_size=args.min_fold_train_size,
        logistic_c=args.logistic_c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        use_order_augmentation=not args.no_order_augmentation,
        calibrate=not args.no_calibration,
    )

    dataset = build_strength_dataset_from_db(dataset_config)
    training = train_strength_model(dataset, model_config)

    artifact_path: str | None = None
    if not args.no_save:
        artifact_path = str(
            save_strength_artifacts(
                dataset=dataset,
                training=training,
                config=model_config,
                artifact_root=Path(args.artifact_root),
            )
        )

    payload: dict[str, Any] = {
        "experiment_ids": ["EXP-046", "EXP-047"],
        "dataset": dataset.metadata,
        "model_config": asdict(model_config),
        "metrics": training.metrics,
        "folds": [asdict(f) for f in training.folds],
        "artifact_path": artifact_path,
    }
    output = json.dumps(payload, indent=2, default=str)
    print(output)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
