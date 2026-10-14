"""Unit and integration tests for the EXP-040 candidate retrain pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betting_app.ml.pipelines.exp040_retrain_pipeline import (
    ALL_FEATURES,
    Exp040RetrainConfig,
    _dataset_hash,
    train_oof_and_final,
)


def _generate_synthetic_training_frame(n_samples: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d").tolist()
    y_true = np.random.binomial(1, 0.53, size=n_samples)

    data = {
        "golgg_match_id": np.arange(1000, 1000 + n_samples),
        "date": dates,
        "best_of": np.random.choice([1, 3, 5], size=n_samples),
        "y_true": y_true,
    }

    # Generate feature columns
    for feat in ALL_FEATURES:
        data[feat] = np.random.normal(0.0, 1.0, size=n_samples)

    return pd.DataFrame(data)


def test_exp040_config_defaults() -> None:
    cfg = Exp040RetrainConfig()
    assert cfg.model_name == "Hierarchical-Markov-VennAbers-EXP040"
    assert cfg.status_on_success == "candidate"
    assert cfg.min_shadow_log_loss == 0.62
    assert cfg.min_shadow_auc == 0.72


def test_dataset_hash_stability() -> None:
    frame = _generate_synthetic_training_frame(50)
    hash_1 = _dataset_hash(frame)
    hash_2 = _dataset_hash(frame)
    assert hash_1 == hash_2
    assert len(hash_1) == 64  # SHA-256


def test_train_oof_and_final_synthetic() -> None:
    frame = _generate_synthetic_training_frame(120)
    pipeline, calibrator, venn_abers, metrics = train_oof_and_final(
        frame,
        initial_train_before="2021-02-01",
        update_interval=20,
    )

    assert pipeline is not None
    assert calibrator is not None
    assert hasattr(calibrator, "temperature_")
    assert calibrator.temperature_ > 0.0
    assert venn_abers.is_fitted_

    assert "oof_uncalibrated" in metrics
    assert "oof_temperature_calibrated" in metrics
    assert "calibrator_temperature" in metrics

    oof_cal = metrics["oof_temperature_calibrated"]
    assert 0.0 <= oof_cal["log_loss"] <= 2.0
    assert 0.0 <= oof_cal["brier"] <= 1.0
    assert 0.0 <= oof_cal["accuracy"] <= 1.0
    assert 0.0 <= oof_cal["ece"] <= 1.0
