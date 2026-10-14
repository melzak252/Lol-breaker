"""Run inference for registered shadow/production ML models.

This module intentionally writes to the existing `canonical_predictions` table
instead of introducing a parallel prediction table. Existing match-detail and
prediction-history endpoints already read that table, so shadow model outputs
become visible without replacing the current thesis/hybrid production flow.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from betting_app.core.db import get_session
from betting_app.ml.inference.types import InferencePrediction, RegisteredModelArtifact, ShadowInferenceResult
from betting_app.ml.registry import ensure_registry_tables, list_model_versions
from betting_app.ml.training.features import flatten_numeric_features


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_loads(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def _resolve_artifact_path(path: str) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return Path.cwd() / artifact_path


def load_registered_model_artifacts(
    *,
    statuses: Iterable[str] = ("shadow",),
    model_name: str | None = None,
    model_version: str | None = None,
    session: Session | None = None,
) -> list[RegisteredModelArtifact]:
    """Load registry rows that are eligible for inference."""
    status_set = set(statuses)
    own_session = session is None
    sess = session or get_session()
    try:
        ensure_registry_tables(sess)
        rows: list[dict[str, Any]] = []
        for status in status_set:
            rows.extend(list_model_versions(status=status, session=sess))
        out: list[RegisteredModelArtifact] = []
        for row in rows:
            if model_name and row.get("model_name") != model_name:
                continue
            if model_version and row.get("model_version") != model_version:
                continue
            artifact_path = row.get("artifact_path")
            if not artifact_path:
                continue
            out.append(
                RegisteredModelArtifact(
                    model_name=str(row["model_name"]),
                    model_version=str(row["model_version"]),
                    status=str(row["status"]),
                    artifact_path=str(artifact_path),
                    feature_version=row.get("feature_version"),
                    metrics=row.get("metrics") or {},
                )
            )
        return out
    finally:
        if own_session:
            sess.close()


def _load_model_bundle(
    artifact_path: str,
) -> tuple[Any, list[str], Any | None, Any | None]:
    """Load a registry artifact and optional calibration components.

    Legacy artifacts contain only an estimator and feature names. EXP-040 adds
    temperature and Venn-Abers calibrators without changing that base contract.
    """

    path = _resolve_artifact_path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "estimator" in bundle and "feature_names" in bundle:
        return (
            bundle["estimator"],
            list(bundle["feature_names"]),
            bundle.get("temperature_calibrator"),
            bundle.get("venn_abers_calibrator"),
        )
    raise ValueError(f"Unsupported model artifact format: {path}")


def _latest_feature_rows(
    *,
    feature_version: str | None = None,
    limit: int | None = None,
    session: Session,
) -> list[dict[str, Any]]:
    where = ["cm.status = 'upcoming'", "umf.features_json IS NOT NULL"]
    latest_where = ["features_json IS NOT NULL"]
    params: dict[str, Any] = {}
    if feature_version:
        where.append("umf.feature_version = :feature_version")
        latest_where.append("feature_version = :feature_version")
        params["feature_version"] = feature_version
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT :limit"
        params["limit"] = int(limit)
    rows = session.execute(text(f"""
        SELECT umf.canonical_match_id,
               umf.feature_version,
               umf.ratings_version,
               umf.data_cutoff_at,
               umf.features_json
        FROM upcoming_match_features umf
        JOIN (
            SELECT canonical_match_id, MAX(id) AS latest_id
            FROM upcoming_match_features
            WHERE {' AND '.join(latest_where)}
            GROUP BY canonical_match_id
        ) latest ON latest.latest_id = umf.id
        JOIN canonical_matches cm ON cm.id = umf.canonical_match_id
        WHERE {' AND '.join(where)}
        ORDER BY cm.start_time_normalized ASC, cm.id ASC
        {limit_sql}
    """), params).mappings().all()
    return [dict(row) for row in rows]


def _matrix_row(features_json: Any, feature_names: list[str]) -> tuple[np.ndarray, int]:
    payload = _json_loads(features_json)
    features = flatten_numeric_features(payload)
    missing = sum(1 for name in feature_names if name not in features)
    values = [features.get(name, np.nan) for name in feature_names]
    return np.asarray([values], dtype=float), missing


def predict_feature_row(
    model: RegisteredModelArtifact,
    estimator: Any,
    feature_names: list[str],
    row: dict[str, Any],
    *,
    temperature_calibrator: Any | None = None,
    venn_abers_calibrator: Any | None = None,
) -> InferencePrediction:
    x, missing_count = _matrix_row(row.get("features_json"), feature_names)
    prob_a = float(estimator.predict_proba(x)[0, 1])
    prob_a = min(max(prob_a, 1e-6), 1.0 - 1e-6)
    diagnostics: dict[str, Any] = {
        "source": "ml_registry",
        "registry_status": model.status,
        "artifact_path": model.artifact_path,
        "feature_count": len(feature_names),
        "missing_feature_count": missing_count,
    }
    if temperature_calibrator is not None:
        logit = np.log(prob_a / (1.0 - prob_a))
        prob_a = float(temperature_calibrator.transform(np.asarray([logit]))[0])
        diagnostics["temperature_calibrated"] = True
    if venn_abers_calibrator is not None:
        intervals = venn_abers_calibrator.predict_intervals(np.asarray([prob_a]))
        prob_a = float(np.asarray(intervals.p).item())
        lower_a = float(np.asarray(intervals.p_lower).item())
        upper_a = float(np.asarray(intervals.p_upper).item())
        diagnostics["conformal"] = {
            "method": "venn_abers",
            "p_lower_a": lower_a,
            "p_upper_a": upper_a,
            "uncertainty": upper_a - lower_a,
        }
    prob_a = min(max(prob_a, 1e-6), 1.0 - 1e-6)
    prob_b = 1.0 - prob_a
    return InferencePrediction(
        canonical_match_id=int(row["canonical_match_id"]),
        model_name=model.model_name,
        model_version=model.model_version,
        prob_a=prob_a,
        prob_b=prob_b,
        features_version=row.get("feature_version"),
        ratings_version=row.get("ratings_version"),
        data_cutoff_at=str(row.get("data_cutoff_at")) if row.get("data_cutoff_at") is not None else None,
        diagnostics=diagnostics,
    )


def write_prediction(prediction: InferencePrediction, *, session: Session) -> None:
    """Replace active prediction for same match/model/version with a new active row."""
    session.execute(text("""
        UPDATE canonical_predictions
        SET prediction_status = 'stale'
        WHERE canonical_match_id = :mid
          AND model_name = :model_name
          AND model_version = :model_version
          AND prediction_status = 'active'
    """), {
        "mid": prediction.canonical_match_id,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
    })
    session.execute(text("""
        INSERT INTO canonical_predictions (
            canonical_match_id, model_artifact_id, model_name, model_version,
            predicted_at, prob_a, prob_b, prediction_status,
            features_version, ratings_version, data_cutoff_at, diagnostics_json
        ) VALUES (
            :mid, NULL, :model_name, :model_version,
            :predicted_at, :prob_a, :prob_b, 'active',
            :features_version, :ratings_version, :data_cutoff_at, :diagnostics_json
        )
    """), {
        "mid": prediction.canonical_match_id,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
        "predicted_at": _now_iso(),
        "prob_a": prediction.prob_a,
        "prob_b": prediction.prob_b,
        "features_version": prediction.features_version,
        "ratings_version": prediction.ratings_version,
        "data_cutoff_at": prediction.data_cutoff_at,
        "diagnostics_json": json.dumps(prediction.diagnostics, sort_keys=True),
    })


def run_registry_shadow_inference(
    *,
    statuses: Iterable[str] = ("shadow",),
    model_name: str | None = None,
    model_version: str | None = None,
    limit: int | None = None,
    session: Session | None = None,
) -> ShadowInferenceResult:
    """Generate active `canonical_predictions` rows for registry models."""
    own_session = session is None
    sess = session or get_session()
    try:
        models = load_registered_model_artifacts(
            statuses=statuses,
            model_name=model_name,
            model_version=model_version,
            session=sess,
        )
        predictions_written = 0
        feature_rows_seen = 0
        loaded_models = 0
        model_versions: list[str] = []
        for model in models:
            (
                estimator,
                feature_names,
                temperature_calibrator,
                venn_abers_calibrator,
            ) = _load_model_bundle(model.artifact_path)
            loaded_models += 1
            model_versions.append(f"{model.model_name}:{model.model_version}")
            rows = _latest_feature_rows(
                feature_version=model.feature_version,
                limit=limit,
                session=sess,
            )
            feature_rows_seen += len(rows)
            for row in rows:
                prediction = predict_feature_row(
                    model,
                    estimator,
                    feature_names,
                    row,
                    temperature_calibrator=temperature_calibrator,
                    venn_abers_calibrator=venn_abers_calibrator,
                )
                write_prediction(prediction, session=sess)
                predictions_written += 1
        sess.commit()
        return ShadowInferenceResult(
            models_seen=len(models),
            models_loaded=loaded_models,
            feature_rows_seen=feature_rows_seen,
            predictions_written=predictions_written,
            model_versions=model_versions,
        )
    except Exception:
        sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()
