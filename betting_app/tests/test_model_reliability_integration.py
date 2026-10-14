"""Frozen reliability integration's numerical and membership boundaries."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.model_reliability_integration import (
    _calibrate_rating_cal,
    blend_series_probabilities,
    fit_alpha,
)


def test_fit_alpha_compares_both_endpoints_when_rating_is_better():
    alpha = fit_alpha(np.full(8, 0.55), np.full(8, 0.9), np.ones(8, dtype=int))

    assert alpha == 1.0


def test_fit_alpha_compares_both_endpoints_when_native_is_better():
    alpha = fit_alpha(np.full(8, 0.85), np.full(8, 0.1), np.ones(8, dtype=int))

    assert alpha == 0.0


def test_logit_blend_is_symmetric_under_swapping_sides():
    base = np.array([0.12, 0.71, 0.94])
    rating = np.array([0.36, 0.83, 0.59])

    forward = blend_series_probabilities(base, rating, 0.37)
    swapped = blend_series_probabilities(1 - base, 1 - rating, 0.37)

    np.testing.assert_allclose(forward + swapped, 1.0, rtol=0, atol=2e-16)


def test_rating_cal_uses_complete_native_subset_of_prior_year():
    rows, calibrators = [], {}
    for year in (2024, 2025, 2026):
        for month in ("03", "09"):
            rows.append({
                "golgg_match_id": f"{year}-{month}", "team1_id": "A", "team2_id": "B",
                "date": f"{year-1}-{month}-15", "result_day": f"{year-1}-{month}-15",
                "best_of": 3, "y": 1, "roster_a": ["a1", "a2", "a3", "a4", "a5"],
                "roster_b": ["b1", "b2", "b3", "b4", "b5"],
                "p_player_raw": .64, "p_family_raw": .64,
                "calibration_for_year": year, "calibration_origin": f"{year}-01-01",
                "state_receipt_sha256": "1"*64, "roster_receipt_sha256": "2"*64,
            })
        calibrators[str(year)] = {
            "origin": f"{year}-01-01", "calibration_year": year-1,
            "calibration_end": f"{year-1}-09-15", "calibration_rows": 2,
            "calibration_ids_sha256": "3"*64,
            "player": {"temperature": 2.0}, "family": {"temperature": .5},
        }
    prior_year = pd.DataFrame(rows)
    native = prior_year.loc[prior_year.golgg_match_id.str.endswith("-09")].copy()
    native["origin"] = native.calibration_for_year
    result, _ = _calibrate_rating_cal(native, prior_year, {"calibrators": calibrators})
    assert result.golgg_match_id.tolist() == native.golgg_match_id.tolist()
    np.testing.assert_allclose(result.p_player, 4/7, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.p_family, 256/337, rtol=0, atol=1e-15)
    missing = prior_year.loc[prior_year.golgg_match_id.ne("2025-09")]
    with pytest.raises(ValueError, match="membership"):
        _calibrate_rating_cal(native, missing, {"calibrators": calibrators})
