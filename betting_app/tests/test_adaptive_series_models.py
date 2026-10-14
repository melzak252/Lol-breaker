"""Observable contracts for EXP-090 temporal fitting and offset predictions."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from scripts.adaptive_series_models import (
    fit_prequential_slope,
    fit_residual_booster,
    fit_ridge,
    recency_weights,
)


def test_recency_retains_history_without_changing_regularization_mass():
    weights, audit = recency_weights(
        pd.to_datetime(["2022-01-01", "2023-01-01"]), "2023-01-02"
    )
    np.testing.assert_allclose(weights[1] / weights[0], 2.0)
    np.testing.assert_allclose(weights.sum(), 2.0)
    assert 1 < audit["effective_sample_size"] < 2
    with pytest.raises(ValueError, match="before"):
        recency_weights(pd.to_datetime(["2023-01-02"]), "2023-01-02")


def test_residual_booster_includes_base_offset_in_training_and_inference():
    # Each sign has an exact 75/25 outcome split. The pretrained logit is
    # already right: omitting LightGBM's init_score learns it twice (~.90).
    x = np.repeat(np.array([[-1.0], [1.0]]), 400, axis=0)
    y = np.concatenate((np.tile([0, 0, 0, 1], 100), np.tile([1, 1, 1, 0], 100)))
    base, _ = fit_ridge(x, y, c=1e8)
    model, _ = fit_residual_booster(
        base, x, y, x, y, seed=90, max_rounds=10, min_child_samples=20
    )
    p = expit(model.logit(np.array([[-1.0], [1.0]])))
    np.testing.assert_allclose(p, [0.25, 0.75], atol=1e-5)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)


def test_residual_prediction_is_symmetric_on_unseen_feature_vectors():
    rng = np.random.default_rng(90)
    x = rng.normal(size=(500, 3))
    y = rng.binomial(1, expit(x[:, 0] + x[:, 1] * x[:, 2] ** 2))
    base, _ = fit_ridge(x[:350], y[:350])
    model, _ = fit_residual_booster(
        base,
        x[:350],
        y[:350],
        x[350:],
        y[350:],
        seed=90,
        max_rounds=10,
        min_child_samples=20,
    )
    unseen = rng.normal(size=(17, 3))
    np.testing.assert_allclose(
        expit(model.logit(unseen)) + expit(model.logit(-unseen)), 1.0, atol=1e-12
    )


def _oof_rows():
    return pd.DataFrame(
        {
            "golgg_match_id": range(9),
            "date": pd.to_datetime(
                ["2023-07-02"] * 4 + ["2023-10-02"] * 4 + ["2024-01-02"]
            ),
            "base_fit_before": pd.to_datetime(
                ["2023-07-01"] * 4 + ["2023-10-01"] * 4 + ["2024-01-01"]
            ),
            "base_train_max_date": pd.to_datetime(
                ["2023-06-30"] * 4 + ["2023-09-30"] * 4 + ["2023-12-31"]
            ),
            "raw_logit": [1, 1, 1, 1, -1, -1, -1, -1, 4],
            "y_true": [1, 1, 1, 0, 0, 0, 0, 1, 1],
        }
    )


def test_prequential_calibration_ignores_future_outcomes():
    rows = _oof_rows()
    slope, audit = fit_prequential_slope(rows, "2024-01-01")
    rows.loc[8, ["y_true", "raw_logit"]] = [0, -100]
    altered_slope, _ = fit_prequential_slope(rows, "2024-01-01")
    np.testing.assert_allclose(slope, np.log(3), rtol=1e-5)
    assert slope == altered_slope
    assert audit["n"] == 8


def test_prequential_calibration_rejects_in_sample_past_predictions():
    rows = _oof_rows()
    rows.loc[0, "base_train_max_date"] = rows.loc[0, "date"]
    with pytest.raises(ValueError, match="out.of.sample"):
        fit_prequential_slope(rows, "2024-01-01")
