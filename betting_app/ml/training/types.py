"""Typed objects for production retraining workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingExample:
    canonical_match_id: int
    occurred_at: str
    target: int
    features: dict[str, float]


@dataclass(frozen=True)
class TrainingDataset:
    examples: list[TrainingExample]
    feature_names: list[str]

    @property
    def size(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class ModelCandidateSpec:
    name: str
    estimator_type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    train_size: int
    test_size: int
    test_start_at: str
    test_end_at: str
    log_loss: float
    brier: float
    accuracy: float


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: ModelCandidateSpec
    folds: list[FoldResult]
    mean_log_loss: float
    mean_brier: float
    mean_accuracy: float


@dataclass(frozen=True)
class TrainedModelArtifact:
    model_name: str
    model_version: str
    artifact_path: str
    metadata_path: str
    dataset_path: str
    feature_names_path: str
    dataset_metadata_path: str
    feature_names: list[str]
    metrics: dict[str, Any]
