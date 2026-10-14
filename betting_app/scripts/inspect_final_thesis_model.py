"""Inspect availability of final EXP-039 thesis model artefacts.

The thesis result is Sym-Cal LR-ElasticNet-W20-Binomial.  The repo currently
contains scripts/experiment logs, but not necessarily a serialized sklearn model
and calibration artefact.  This script records that fact in model_artifacts so
the UI/ops layer can distinguish "operational fallback model" from "exact thesis
model loaded".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from betting_app.core.config import PROJECT_ROOT
from betting_app.core.db import init_db, transaction


MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
MODEL_VERSION = "exp-039"
EXPECTED_DIR = PROJECT_ROOT / "docs" / "assets" / "final_symmetric_calibrated_market_comparison"
EXPECTED_FILES = [
    "final_symmetric_calibrated_market_common_sample.csv",
    "final_symmetric_calibrated_market_metrics.csv",
]
SERIALIZED_CANDIDATES = [
    "sym_cal_lr_elasticnet_w20_binomial.joblib",
    "sym_cal_lr_elasticnet_w20_binomial.pkl",
    "final_model.joblib",
    "final_model.pkl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="Upsert model_artifacts status row")
    args = parser.parse_args()

    init_db()
    report = inspect_artifacts()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.register:
        register_report(report)
        print("Registered final thesis model artefact status in model_artifacts")


def inspect_artifacts() -> dict:
    existing_files = [name for name in EXPECTED_FILES if (EXPECTED_DIR / name).exists()]
    serialized = [name for name in SERIALIZED_CANDIDATES if (EXPECTED_DIR / name).exists()]
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "artifact_dir": str(EXPECTED_DIR),
        "artifact_dir_exists": EXPECTED_DIR.exists(),
        "expected_files_present": existing_files,
        "serialized_model_candidates_present": serialized,
        "exact_inference_ready": bool(serialized),
        "status": "active" if serialized else "missing_serialized_model",
        "note": (
            "Exact EXP-039 inference needs serialized estimator + calibration/symmetry artefacts. "
            "Until then app uses Operational-PlayerTeamRatings-W20 / hybrid fallback."
        ),
    }


def register_report(report: dict) -> None:
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO model_artifacts(
                model_name, model_version, artifact_path, feature_schema_json,
                model_params_json, training_cutoff_at, metrics_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, model_version) DO UPDATE SET
                artifact_path = excluded.artifact_path,
                feature_schema_json = excluded.feature_schema_json,
                model_params_json = excluded.model_params_json,
                metrics_json = excluded.metrics_json,
                status = excluded.status
            """,
            (
                MODEL_NAME,
                MODEL_VERSION,
                str(EXPECTED_DIR),
                json.dumps({"expected_files": EXPECTED_FILES, "serialized_candidates": SERIALIZED_CANDIDATES}),
                json.dumps({"estimator": "LogisticRegression ElasticNet", "postprocess": ["order_symmetry", "Platt expanding"]}),
                "2026-05-24",
                json.dumps(report, ensure_ascii=False),
                report["status"],
            ),
        )


if __name__ == "__main__":
    main()
