"""Observable uncertainty and causal-resampling contracts, independent of archives."""

import numpy as np
import pandas as pd
import pytest

from src.models.a0_uncertainty import (
    annual_membership,
    bootstrap_training_indices,
    summarize_members,
)


def test_uncertainty_decomposes_outcome_variance_and_swaps_quantiles():
    members = np.array([[0.2, 0.4, 0.8], [0.5, 0.5, 0.5]])
    forward = summarize_members(members)
    reverse = summarize_members(1.0 - members)
    np.testing.assert_allclose(forward["p"], [1.4 / 3.0, 0.5])
    np.testing.assert_allclose(forward["p"] + reverse["p"], 1.0)
    np.testing.assert_allclose(
        forward["aleatoric_variance"] + forward["epistemic_variance"],
        forward["p"] * (1.0 - forward["p"]),
    )
    np.testing.assert_allclose(forward["p_std"], reverse["p_std"])
    np.testing.assert_allclose(forward["p_q10"], 1.0 - reverse["p_q90"])
    np.testing.assert_allclose(forward["mutual_information"], reverse["mutual_information"], atol=1e-15)
    assert forward["epistemic_variance"][1] == 0.0
    assert forward["total_variance"][1] == 0.25
    assert forward["mutual_information"][1] == pytest.approx(0.0, abs=1e-15)
    assert forward["mutual_information"][0] > 0.0


def test_uncertainty_rejects_invalid_probabilities_instead_of_clipping():
    with pytest.raises(ValueError):
        summarize_members(np.array([[0.2, np.nan, 0.8]]))
    with pytest.raises(ValueError):
        summarize_members(np.array([[0.2, 1.0, 0.8]]))


def test_annual_membership_requires_released_labels_not_only_match_dates():
    frame = pd.DataFrame({
        "golgg_match_id": ["closed_train", "late_train", "closed_cal", "late_cal", "test"],
        "date": ["2022-12-30", "2022-12-31", "2023-05-01", "2023-12-31", "2024-01-01"],
        "result_day": ["2022-12-31", "2023-01-01", "2023-05-01", "2024-01-01", "2024-01-02"],
    })
    train, calibration, test = annual_membership(frame, 2024)
    np.testing.assert_array_equal(train, [0])
    np.testing.assert_array_equal(calibration, [2])
    np.testing.assert_array_equal(test, [4])


def test_bootstrap_keeps_week_multiplicities_and_ignores_future_rows():
    frame = pd.DataFrame({
        "golgg_match_id": [str(i) for i in range(8)],
        "date": ["2022-01-03", "2022-01-09", "2022-01-10", "2022-01-16",
                 "2022-01-17", "2022-01-23", "2022-01-24", "2025-01-01"],
        "y": [0, 1, 1, 0, 0, 1, 1, 0],
    })
    eligible = np.arange(7)
    sample, receipt = bootstrap_training_indices(frame, eligible, seed=0)
    changed = frame.copy()
    changed.loc[7, ["date", "y"]] = ["2010-01-01", 1]
    replay, replay_receipt = bootstrap_training_indices(changed, eligible, seed=0)
    np.testing.assert_array_equal(sample, replay)
    assert receipt == replay_receipt
    assert set(sample).issubset(set(eligible))
    multiplicities = np.bincount(sample, minlength=len(frame))
    assert multiplicities[0] == multiplicities[1]
    assert multiplicities[2] == multiplicities[3]
    assert multiplicities[4] == multiplicities[5]
    assert multiplicities[7] == 0
    assert np.any(multiplicities[:7] == 0)
    assert np.any(multiplicities[:7] > 1)
    assert np.all(np.diff(sample) >= 0)
