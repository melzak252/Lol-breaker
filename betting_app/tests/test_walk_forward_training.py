from __future__ import annotations

from betting_app.ml.training.types import ModelCandidateSpec, TrainingDataset, TrainingExample
from betting_app.ml.training.walk_forward import evaluate_candidate_walk_forward


def _example(idx: int, occurred_at: str, target: int) -> TrainingExample:
    signal = 1.0 if target == 1 else -1.0
    return TrainingExample(
        canonical_match_id=10_000 + idx,
        occurred_at=occurred_at,
        target=target,
        features={"signal": signal, "idx": float(idx)},
    )


def test_walk_forward_train_window_uses_recent_history_only() -> None:
    dataset = TrainingDataset(
        examples=[
            _example(0, "2023-01-01T12:00:00+00:00", 0),
            _example(1, "2023-02-01T12:00:00+00:00", 1),
            _example(2, "2023-03-01T12:00:00+00:00", 0),
            _example(3, "2023-04-01T12:00:00+00:00", 1),
            _example(4, "2025-01-01T12:00:00+00:00", 0),
            _example(5, "2025-02-01T12:00:00+00:00", 1),
            _example(6, "2025-03-01T12:00:00+00:00", 0),
            _example(7, "2025-04-01T12:00:00+00:00", 1),
            _example(8, "2026-01-01T12:00:00+00:00", 0),
            _example(9, "2026-02-01T12:00:00+00:00", 1),
        ],
        feature_names=["signal", "idx"],
    )
    candidate = ModelCandidateSpec(
        name="logreg_test",
        estimator_type="logistic_regression",
        params={"C": 1.0, "max_iter": 200},
    )

    full_history = evaluate_candidate_walk_forward(
        dataset,
        candidate,
        min_train_size=4,
        test_size=2,
        step_size=2,
    )
    rolling = evaluate_candidate_walk_forward(
        dataset,
        candidate,
        min_train_size=4,
        test_size=2,
        step_size=2,
        train_window_days=400,
    )

    assert [fold.train_size for fold in full_history.folds] == [4, 6, 8]
    assert len(rolling.folds) == 1
    assert rolling.folds[0].train_size == 4
    assert rolling.folds[0].test_start_at == "2026-01-01T12:00:00+00:00"
