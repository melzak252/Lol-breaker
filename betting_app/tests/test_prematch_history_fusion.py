"""Observable contracts for joint history/summary architecture experiments."""
from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.models.prematch_history_fusion import HistoryFusionModel


def learning_batch(n=96):
    rng = np.random.default_rng(291)
    y = np.arange(n) % 2
    history = np.zeros((n, 2, 5, 16, 3), dtype=np.float32)
    history[:, 0, :, :3, 0] = (2 * y - 1)[:, None, None]
    history[:, 1, :, :3, 0] = -(2 * y - 1)[:, None, None]
    history[..., 1:] = rng.normal(0, 0.1, history[..., 1:].shape)
    mask = np.zeros(history.shape[:-1], dtype=bool)
    mask[..., :3] = True
    return {
        "players": np.zeros((n, 2, 5, 3), dtype=np.float32),
        "teams": np.zeros((n, 2, 2), dtype=np.float32),
        "context": np.tile([1., 0., 0.], (n, 1)),
        "baseline_series": np.full((n, 2), .5),
        "history": history,
        "history_mask": mask,
    }, y


def reversed_batch(batch):
    return {key: (1 - value if key == "baseline_series" else value if key == "context" else value[:, ::-1]) for key, value in batch.items()}


@pytest.mark.parametrize("architecture", ["input", "representation"])
def test_histories_distinguish_outcomes_with_identical_static_features(tmp_path, architecture):
    batch, y = learning_batch()
    model = HistoryFusionModel({"architecture": architecture, "hidden": 16, "epochs": 80, "dropout": 0.0}, seed=127).fit(batch, y)
    p = model.predict_proba(batch)
    assert np.mean((p >= .5) == y) > .9
    assert p[y == 1].mean() - p[y == 0].mean() > .1
    np.testing.assert_allclose(p + model.predict_proba(reversed_batch(batch)), 1., atol=1e-7, rtol=0)
    contaminated = {key: value.copy() for key, value in batch.items()}
    contaminated["history"][~contaminated["history_mask"]] = np.nan
    np.testing.assert_allclose(model.predict_proba(contaminated), p, atol=0, rtol=0)
    cold = {key: value[:2].copy() for key, value in batch.items()}
    cold["history_mask"][:] = False
    cold["history"][:] = np.nan
    np.testing.assert_allclose(model.predict_proba(cold), .5, atol=1e-12, rtol=0)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(batch), p)


def test_visible_nonfinite_history_is_not_silently_treated_as_padding():
    batch, y = learning_batch(12)
    batch["history"][0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        HistoryFusionModel({"architecture": "representation", "epochs": 1}, seed=9).fit(batch, y)


def test_omitted_history_is_not_silently_replaced_with_cold_start():
    batch, y = learning_batch(12)
    del batch["history"]
    del batch["history_mask"]
    with pytest.raises(ValueError):
        HistoryFusionModel({"architecture": "representation", "epochs": 1}, seed=9).fit(batch, y)
