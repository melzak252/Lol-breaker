"""Nontrivial invariants of the retrospective paired architecture experiment."""

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_siamese_architectures import (
    augment_disagreement,
    monthly_bootstrap,
    probability_metrics,
    temporal_masks,
)


def test_disagreement_interactions_swap_without_changing_mean_bias():
    names = [
        f"{scope}_{s}_logit"
        for s in ("elo", "gl", "ts", "os", "pl", "tm")
        for scope in ("team", "player")
    ]
    values = np.random.default_rng(4).normal(size=(9, len(names)))
    np.testing.assert_allclose(
        augment_disagreement(-values, names),
        -augment_disagreement(values, names),
        atol=1e-14,
    )
    assert np.any(augment_disagreement(values, names)[:, -5:] != 0)


def test_monthly_bootstrap_weights_rows_not_equal_month_means():
    dates = pd.Series(["2024-01-01"] * 9 + ["2024-02-01"])
    delta = np.array([-1.0] * 9 + [1.0])
    result = monthly_bootstrap(delta, dates, repetitions=5000)
    assert result["mean"] == pytest.approx(-0.8)
    assert result["ci95_low"] == pytest.approx(-1)
    assert result["ci95_high"] == pytest.approx(1)
    with pytest.raises(ValueError, match="months"):
        monthly_bootstrap(delta, pd.Series(["2024-01-01"] * 10))


def test_calendar_split_keeps_calibration_and_evaluation_disjoint():
    dates = pd.Series(
        pd.to_datetime(
            ["2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2025-01-01"]
        )
    )
    train, calibration, test = temporal_masks(dates, 2024)
    assert np.flatnonzero(train).tolist() == [0]
    assert np.flatnonzero(calibration).tolist() == [1, 2]
    assert np.flatnonzero(test).tolist() == [3]


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5764096037593363, 0.531943521682863, 0.3826836556593029, 0.6526254862407531],
        [0.2, 0.8, 0.7, 0.3],
        [0.5, 0.5, 0.2, 0.8],
        [0.5, 0.5, 0.5, 0.5],
    ],
)
def test_unidentifiable_calibration_preserves_proper_scores(probabilities):
    labels = np.array([1, 0, 0, 1])
    p = np.array(probabilities)
    result = probability_metrics(labels, p)
    assert result["calibration_slope"] is None
    assert result["calibration_intercept"] is None
    assert result["log_loss"] == pytest.approx(
        -np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))
    )
    assert result["brier"] == pytest.approx(np.mean((p - labels) ** 2))


def test_overlapping_outcomes_have_finite_calibration_coefficients():
    result = probability_metrics(
        [0, 0, 0, 1, 0, 1, 1, 1], [0.25] * 4 + [0.75] * 4
    )
    assert result["calibration_intercept"] == pytest.approx(0, abs=1e-6)
    assert result["calibration_slope"] == pytest.approx(1, abs=1e-6)
