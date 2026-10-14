"""Artifact persistence for retrained models."""

from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from betting_app.ml.training.candidates import build_estimator
from betting_app.ml.training.types import ModelCandidateSpec, TrainedModelArtifact, TrainingDataset
from betting_app.ml.training.walk_forward import dataset_to_matrix

DEFAULT_ARTIFACT_ROOT = Path("betting_app/models/ml")


def _dataset_records(dataset: TrainingDataset, feature_names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in dataset.examples:
        records.append(
            {
                "canonical_match_id": example.canonical_match_id,
                "occurred_at": example.occurred_at,
                "target": example.target,
                "features": {name: example.features.get(name) for name in feature_names},
            }
        )
    return records


def compute_training_dataset_hash(dataset: TrainingDataset, feature_names: list[str] | None = None) -> str:
    """Hash the exact examples/features used for training.

    This is intentionally based on the materialized model-input dataset, not only
    on source row ids, so DB corrections or feature changes produce a new hash.
    """
    names = feature_names or dataset.feature_names
    payload = json.dumps(_dataset_records(dataset, names), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_training_dataset_snapshot(
    dataset: TrainingDataset,
    *,
    artifact_dir: Path,
    feature_names: list[str],
) -> tuple[Path, Path, Path, str]:
    """Persist an immutable training dataset snapshot next to the model."""
    dataset_path = artifact_dir / "train_dataset.jsonl"
    feature_names_path = artifact_dir / "feature_names.json"
    dataset_metadata_path = artifact_dir / "dataset_metadata.json"
    records = _dataset_records(dataset, feature_names)
    dataset_hash = compute_training_dataset_hash(dataset, feature_names)

    with dataset_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            f.write("\n")

    feature_names_path.write_text(json.dumps(feature_names, indent=2, sort_keys=True), encoding="utf-8")
    dataset_metadata = {
        "dataset_hash": dataset_hash,
        "format": "jsonl",
        "rows": dataset.size,
        "feature_count": len(feature_names),
        "target": "canonical_matches.winner_side: team_a/a -> 1, team_b/b -> 0",
        "created_at": datetime.now(UTC).isoformat(),
    }
    dataset_metadata_path.write_text(json.dumps(dataset_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dataset_path, feature_names_path, dataset_metadata_path, dataset_hash


def train_and_save_model(
    dataset: TrainingDataset,
    candidate: ModelCandidateSpec,
    *,
    model_name: str,
    model_version: str,
    metrics: dict[str, Any],
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> TrainedModelArtifact:
    artifact_dir = artifact_root / model_name / model_version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"

    x, y, feature_names = dataset_to_matrix(dataset)
    dataset_path, feature_names_path, dataset_metadata_path, dataset_hash = save_training_dataset_snapshot(
        dataset,
        artifact_dir=artifact_dir,
        feature_names=feature_names,
    )
    estimator = build_estimator(candidate)
    estimator.fit(x, y)
    joblib.dump({"estimator": estimator, "feature_names": feature_names}, model_path)

    final_metrics = {**metrics, "dataset_hash": dataset_hash}

    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "candidate": {"name": candidate.name, "estimator_type": candidate.estimator_type, "params": candidate.params},
        "trained_at": datetime.now(UTC).isoformat(),
        "training_examples": dataset.size,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "dataset_hash": dataset_hash,
        "dataset_path": str(dataset_path),
        "feature_names_path": str(feature_names_path),
        "dataset_metadata_path": str(dataset_metadata_path),
        "metrics": final_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return TrainedModelArtifact(
        model_name=model_name,
        model_version=model_version,
        artifact_path=str(model_path),
        metadata_path=str(metadata_path),
        dataset_path=str(dataset_path),
        feature_names_path=str(feature_names_path),
        dataset_metadata_path=str(dataset_metadata_path),
        feature_names=feature_names,
        metrics=final_metrics,
    )
