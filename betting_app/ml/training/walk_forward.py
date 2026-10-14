"""Walk-forward validation for candidate models."""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean

import numpy as np

from betting_app.ml.metrics import accuracy_from_prob, binary_log_loss, brier_score
from betting_app.ml.training.candidates import build_estimator
from betting_app.ml.training.types import CandidateEvaluation, FoldResult, ModelCandidateSpec, TrainingDataset


def dataset_to_matrix(dataset: TrainingDataset, feature_names: list[str] | None = None):
    names = feature_names or dataset.feature_names
    x = np.array([[ex.features.get(name, np.nan) for name in names] for ex in dataset.examples], dtype=float)
    y = np.array([ex.target for ex in dataset.examples], dtype=int)
    return x, y, names


def _fold_slices(n: int, *, min_train_size: int, test_size: int, step_size: int):
    start = min_train_size
    while start < n:
        end = min(start + test_size, n)
        if end <= start:
            break
        yield slice(0, start), slice(start, end)
        start += step_size


def _parse_occurred_at(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _apply_train_window(
    dataset: TrainingDataset,
    train_slice: slice,
    test_slice: slice,
    *,
    train_window_days: int | None,
) -> list[int]:
    train_indices = list(range(train_slice.start or 0, train_slice.stop or 0))
    if train_window_days is None:
        return train_indices
    if train_window_days <= 0:
        raise ValueError("train_window_days must be positive when provided")

    test_start = _parse_occurred_at(dataset.examples[test_slice][0].occurred_at)
    if test_start is None:
        return train_indices
    cutoff = test_start - timedelta(days=train_window_days)
    filtered: list[int] = []
    for row_idx in train_indices:
        occurred_at = _parse_occurred_at(dataset.examples[row_idx].occurred_at)
        if occurred_at is None or occurred_at >= cutoff:
            filtered.append(row_idx)
    return filtered


def evaluate_candidate_walk_forward(
    dataset: TrainingDataset,
    candidate: ModelCandidateSpec,
    *,
    min_train_size: int = 80,
    test_size: int = 20,
    step_size: int | None = None,
    train_window_days: int | None = None,
) -> CandidateEvaluation:
    if dataset.size < min_train_size + 1:
        raise ValueError(f"Not enough examples for walk-forward validation: {dataset.size}")
    step = step_size or test_size
    x, y, feature_names = dataset_to_matrix(dataset)
    folds: list[FoldResult] = []
    for idx, (train_slice, test_slice) in enumerate(_fold_slices(len(y), min_train_size=min_train_size, test_size=test_size, step_size=step), start=1):
        train_indices = _apply_train_window(dataset, train_slice, test_slice, train_window_days=train_window_days)
        if len(train_indices) < min_train_size:
            continue
        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_slice], y[test_slice]
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            continue
        estimator = build_estimator(candidate)
        estimator.fit(x_train, y_train)
        prob = estimator.predict_proba(x_test)[:, 1]
        examples = dataset.examples[test_slice]
        folds.append(FoldResult(
            fold_index=idx,
            train_size=len(y_train),
            test_size=len(y_test),
            test_start_at=examples[0].occurred_at,
            test_end_at=examples[-1].occurred_at,
            log_loss=binary_log_loss(y_test.tolist(), prob.tolist()),
            brier=brier_score(y_test.tolist(), prob.tolist()),
            accuracy=accuracy_from_prob(y_test.tolist(), prob.tolist()),
        ))
    if not folds:
        raise ValueError("No valid walk-forward folds; check dataset size and class balance")
    return CandidateEvaluation(
        candidate=candidate,
        folds=folds,
        mean_log_loss=mean(f.log_loss for f in folds),
        mean_brier=mean(f.brier for f in folds),
        mean_accuracy=mean(f.accuracy for f in folds),
    )


def select_best_evaluation(evaluations: list[CandidateEvaluation]) -> CandidateEvaluation:
    if not evaluations:
        raise ValueError("No candidate evaluations available")
    return sorted(evaluations, key=lambda ev: (ev.mean_log_loss, ev.mean_brier, -ev.mean_accuracy))[0]
