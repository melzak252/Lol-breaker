"""Types for registry-backed model inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RegisteredModelArtifact:
    model_name: str
    model_version: str
    status: str
    artifact_path: str
    feature_version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferencePrediction:
    canonical_match_id: int
    model_name: str
    model_version: str
    prob_a: float
    prob_b: float
    features_version: str | None
    ratings_version: str | None
    data_cutoff_at: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowInferenceResult:
    models_seen: int
    models_loaded: int
    feature_rows_seen: int
    predictions_written: int
    model_versions: list[str]
