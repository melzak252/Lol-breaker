"""Feature extraction and dataset loading for regular model retraining."""

from __future__ import annotations

import json
import math
from typing import Any

from betting_app.core.db import query_df
from betting_app.ml.training.types import TrainingDataset, TrainingExample


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def flatten_numeric_features(payload: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested JSON into numeric features.

    Lists are intentionally skipped to keep the first production retraining slice
    stable and schema-light. Aggregated roster/rating/team statistics already
    exist as dict fields in `features_json`.
    """
    features: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            safe_key = str(key).replace(".", "_").replace(" ", "_")
            name = f"{prefix}.{safe_key}" if prefix else safe_key
            if _is_number(value):
                features[name] = float(value)
            elif isinstance(value, dict):
                features.update(flatten_numeric_features(value, name))
    return features


def parse_features_json(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    try:
        return flatten_numeric_features(json.loads(raw))
    except (TypeError, json.JSONDecodeError):
        return {}


def load_training_dataset(
    *,
    feature_version: str | None = None,
    ratings_version: str | None = None,
    days_back: int | None = None,
    min_features: int = 5,
) -> TrainingDataset:
    """Load supervised examples from finished matches with stored feature JSON."""
    where = [
        "cm.status IN ('finished', 'completed')",
        "cm.winner_side IN ('team_a', 'team_b', 'a', 'b')",
        "umf.features_json IS NOT NULL",
    ]
    params: dict[str, Any] = {}
    if feature_version:
        where.append("umf.feature_version = :feature_version")
        params["feature_version"] = feature_version
    if ratings_version:
        where.append("umf.ratings_version = :ratings_version")
        params["ratings_version"] = ratings_version
    if days_back is not None:
        where.append("REPLACE(cm.start_time_normalized, 'T', ' ') >= REPLACE(:min_start, 'T', ' ')")
        # Keep dependency-free date calculation in SQL layer for portability.
        where.append("cm.start_time_normalized IS NOT NULL")
        from datetime import UTC, datetime, timedelta

        params["min_start"] = (datetime.now(UTC) - timedelta(days=days_back)).isoformat(timespec="seconds")

    df = query_df(
        f"""
        SELECT cm.id AS canonical_match_id,
               cm.start_time_normalized,
               cm.winner_side,
               umf.features_json
        FROM upcoming_match_features umf
        JOIN canonical_matches cm ON cm.id = umf.canonical_match_id
        WHERE {' AND '.join(where)}
        ORDER BY cm.start_time_normalized ASC, cm.id ASC
        """,
        params,
    )

    examples: list[TrainingExample] = []
    feature_names: set[str] = set()
    for row in df.to_dict("records"):
        feats = parse_features_json(row.get("features_json"))
        if len(feats) < min_features:
            continue
        raw_side = row.get("winner_side")
        target = 1 if raw_side in ("team_a", "a") else 0
        occurred_at = str(row.get("start_time_normalized") or "")
        example = TrainingExample(
            canonical_match_id=int(row["canonical_match_id"]),
            occurred_at=occurred_at,
            target=target,
            features=feats,
        )
        examples.append(example)
        feature_names.update(feats.keys())

    return TrainingDataset(examples=examples, feature_names=sorted(feature_names))
