"""Generic persistence helpers for predictions and their model identities."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from betting_app.core.db import query_df, transaction
from betting_app.core.models.registry import get_active_model


MODEL_NAME = get_active_model().name
MODEL_VERSION = get_active_model().version


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO string."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def add_prediction(
    match_id: int,
    prob_a: float,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
    features_version: str | None = None,
    ratings_version: str | None = None,
    data_cutoff_at: str | None = None,
) -> int:
    """Store a probability prediction for an upcoming match."""

    if not 0 <= prob_a <= 1:
        raise ValueError("prob_a must be in [0, 1]")
    with transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions(
                match_id, model_name, model_version, predicted_at, prob_a, prob_b,
                features_version, ratings_version, data_cutoff_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                model_name,
                model_version,
                utc_now_iso(),
                float(prob_a),
                float(1.0 - prob_a),
                features_version,
                ratings_version,
                data_cutoff_at,
            ),
        )
        return int(cursor.lastrowid)


def latest_predictions() -> pd.DataFrame:
    """Return latest prediction per match."""

    return query_df(
        """
        SELECT p.*
        FROM predictions p
        JOIN (
            SELECT match_id, MAX(predicted_at) AS predicted_at
            FROM predictions
            WHERE status = 'active'
            GROUP BY match_id
        ) latest ON latest.match_id = p.match_id AND latest.predicted_at = p.predicted_at
        """
    )


def add_canonical_prediction(
    canonical_match_id: int,
    prob_a: float,
    model_name: str,
    model_version: str,
    model_artifact_id: int | None = None,
    features_version: str | None = None,
    ratings_version: str | None = None,
    data_cutoff_at: str | None = None,
    diagnostics_json: str | None = None,
) -> int:
    """Store a model probability for a cross-bookmaker canonical match."""

    if not 0 <= prob_a <= 1:
        raise ValueError("prob_a must be in [0, 1]")
    with transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO canonical_predictions(
                canonical_match_id, model_artifact_id, model_name, model_version,
                predicted_at, prob_a, prob_b, features_version, ratings_version,
                data_cutoff_at, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_match_id,
                model_artifact_id,
                model_name,
                model_version,
                utc_now_iso(),
                float(prob_a),
                float(1.0 - prob_a),
                features_version,
                ratings_version,
                data_cutoff_at,
                diagnostics_json,
            ),
        )
        return int(cursor.lastrowid)


def latest_canonical_predictions() -> pd.DataFrame:
    """Return latest active prediction per canonical match and model."""

    return query_df(
        """
        SELECT p.*
        FROM canonical_predictions p
        JOIN (
            SELECT canonical_match_id, model_name, model_version, MAX(predicted_at) AS predicted_at
            FROM canonical_predictions
            WHERE prediction_status = 'active'
            GROUP BY canonical_match_id, model_name, model_version
        ) latest
          ON latest.canonical_match_id = p.canonical_match_id
         AND latest.model_name = p.model_name
         AND latest.model_version = p.model_version
         AND latest.predicted_at = p.predicted_at
        """
    )
