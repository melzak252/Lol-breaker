"""Unit tests for Venn-Abers conformal calibration and risk gating.

Tests cover:
1. VennAbersCalibrator:
   - Finite-sample interval bounds [p0, p1] in [0, 1] with p0 <= p <= p1.
   - Epistemic uncertainty width (p1 - p0) shrinkage with calibration sample size.
   - Strict binary symmetry:
       p0(team_a) == 1.0 - p1(team_b)
       p(team_a) + p(team_b) == 1.0
       uncertainty(team_a) == uncertainty(team_b)
   - Exact mathematical equivalence to naive augmented IsotonicRegression.
   - Significant reduction of Expected Calibration Error (ECE) on miscalibrated models.
   - Scikit-learn API compatibility (fit, predict_proba, transform, predict).
   - Named tuple unpacking and property access.
   - Robustness to extreme logits, ties, single samples, and homogeneous labels.
2. ConformalRiskGater:
   - Conservative lower EV formula: ev_lower = p_lower * (odds * (1.0 - tax_rate)) - 1.0.
   - Gating logic: rejection of negative conservative EV.
   - Gating logic: rejection of excessive uncertainty (interval width > max_uncertainty).
   - Approval of actionable bets meeting both criteria.
   - Handling of invalid odds (<= 1.0), NaNs, and scalar vs vector inputs.
   - Custom tax rates and thresholds.
3. End-to-end integration:
   - Full pipeline connecting model scoring, Venn-Abers calibration, and conformal risk gating.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from betting_app.ml.calibration.candidate_calibration import expected_calibration_error
from betting_app.ml.calibration.venn_abers import (
    ConformalRiskGater,
    VennAbersCalibrator,
    VennAbersIntervals,
)


# ===========================================================================
# 1. VennAbersCalibrator Tests
# ===========================================================================


class TestVennAbersCalibrator:
    @pytest.fixture
    def synthetic_calibration_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic calibration data with overconfident probabilities."""
        np.random.seed(42)
        n = 1000
        true_logits = np.random.normal(0.0, 1.2, size=n)
        true_probs = expit(true_logits)
        y = np.random.binomial(1, true_probs)
        # Inflated overconfident model scores
        uncal_scores = expit(true_logits * 2.0)
        return uncal_scores, y

    def test_fit_returns_self_and_sets_fitted_state(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify scikit-learn API convention that fit returns self and sets fitted state."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator()
        assert not calibrator.is_fitted_

        ret = calibrator.fit(scores, y)
        assert ret is calibrator
        assert calibrator.is_fitted_
        assert calibrator.unique_scores_ is not None
        assert calibrator.F0_ is not None
        assert calibrator.F1_ext_ is not None

    def test_interval_bounds_strictly_within_zero_one(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify p0, p1, and p are always within [0, 1] across the entire test spectrum."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        test_scores = np.linspace(-0.5, 1.5, 200)
        res = calibrator.predict_intervals(test_scores)

        assert isinstance(res, VennAbersIntervals)
        assert np.all(res.p0 >= 0.0)
        assert np.all(res.p0 <= 1.0)
        assert np.all(res.p1 >= 0.0)
        assert np.all(res.p1 <= 1.0)
        assert np.all(res.p >= 0.0)
        assert np.all(res.p <= 1.0)

    def test_lower_bound_less_than_or_equal_to_upper_bound(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify p0 <= p1 everywhere, so epistemic uncertainty is non-negative."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        test_scores = np.linspace(0.0, 1.0, 150)
        res = calibrator.predict_intervals(test_scores)

        assert np.all(res.p0 <= res.p1)
        assert np.all(res.uncertainty >= 0.0)
        assert np.all(res.uncertainty_width >= 0.0)

    def test_point_prediction_lies_within_bounds(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify p0 <= p <= p1 everywhere for point prediction p = p1 / (1 - p0 + p1)."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        test_scores = np.linspace(0.05, 0.95, 100)
        res = calibrator.predict_intervals(test_scores)

        # Allow 1e-15 tolerance for floating-point boundaries
        assert np.all(res.p >= res.p0 - 1e-15)
        assert np.all(res.p <= res.p1 + 1e-15)

    def test_exact_binary_symmetry_under_role_inversion(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify exact binary symmetry:
            p0(team_a) == 1.0 - p1(team_b)
            p(team_a) + p(team_b) == 1.0
            uncertainty(team_a) == uncertainty(team_b)
        where team_b score is (1.0 - team_a score).
        """
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator(symmetric=True).fit(scores, y)

        team_a_scores = np.linspace(0.0, 1.0, 101)
        team_b_scores = 1.0 - team_a_scores

        res_a = calibrator.predict_intervals(team_a_scores)
        res_b = calibrator.predict_intervals(team_b_scores)

        # 1. Lower bound of A equals (1 - upper bound of B)
        np.testing.assert_allclose(res_a.p0, 1.0 - res_b.p1, atol=1e-15)

        # 2. Upper bound of A equals (1 - lower bound of B)
        np.testing.assert_allclose(res_a.p1, 1.0 - res_b.p0, atol=1e-15)

        # 3. Point probabilities strictly sum to 1.0
        np.testing.assert_allclose(res_a.p + res_b.p, 1.0, atol=1e-15)

        # 4. Uncertainty width is identical
        np.testing.assert_allclose(res_a.uncertainty, res_b.uncertainty, atol=1e-15)

    def test_exact_equivalence_to_naive_augmented_isotonic_regression(self) -> None:
        """Verify fast O(log K) inference exactly reproduces naive IsotonicRegression on augmented sets."""
        np.random.seed(123)
        cal_s = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
        cal_y = np.array([0, 0, 1, 0, 1, 0, 1, 1])

        # Symmetrize calibration set for exact comparison with symmetric=True
        s_sym = np.concatenate([cal_s, 1.0 - cal_s])
        y_sym = np.concatenate([cal_y, 1 - cal_y])

        test_points = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95])

        # Naive augmented IsotonicRegression
        p0_naive = []
        p1_naive = []
        for ts in test_points:
            iso0 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso0.fit(np.append(s_sym, ts), np.append(y_sym, 0))
            p0_naive.append(iso0.predict([ts])[0])

            iso1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso1.fit(np.append(s_sym, ts), np.append(y_sym, 1))
            p1_naive.append(iso1.predict([ts])[0])

        p0_naive_arr = np.array(p0_naive)
        p1_naive_arr = np.array(p1_naive)

        calibrator = VennAbersCalibrator(symmetric=True).fit(cal_s, cal_y)
        res = calibrator.predict_intervals(test_points)

        np.testing.assert_allclose(res.p0, p0_naive_arr, atol=1e-12)
        np.testing.assert_allclose(res.p1, p1_naive_arr, atol=1e-12)

    def test_calibration_error_reduction(self) -> None:
        """Verify Venn-Abers calibration significantly reduces Expected Calibration Error (ECE)."""
        np.random.seed(999)
        n_total = 6000
        true_logits = np.random.normal(0.0, 1.3, size=n_total)
        true_probs = expit(true_logits)
        y = np.random.binomial(1, true_probs)

        # Severely overconfident uncalibrated predictions
        uncal_probs = expit(true_logits * 2.2)

        cal_s, test_s = uncal_probs[:3000], uncal_probs[3000:]
        cal_y, test_y = y[:3000], y[3000:]

        uncal_ece = expected_calibration_error(test_y, test_s, n_bins=10)

        calibrator = VennAbersCalibrator().fit(cal_s, cal_y)
        calibrated_probs = calibrator.predict_proba(test_s)[:, 1]
        cal_ece = expected_calibration_error(test_y, calibrated_probs, n_bins=10)

        # Calibrated ECE must be significantly lower than uncalibrated ECE (< 0.025 and >50% reduction)
        assert cal_ece < uncal_ece
        assert cal_ece < 0.025
        assert (uncal_ece - cal_ece) / uncal_ece > 0.50

    def test_uncertainty_shrinks_with_larger_calibration_dataset(self) -> None:
        """Verify finite-sample interval width (epistemic uncertainty) decreases with sample size."""
        np.random.seed(42)

        def make_data(n: int) -> tuple[np.ndarray, np.ndarray]:
            logits = np.random.normal(0.0, 1.0, size=n)
            probs = expit(logits)
            y = np.random.binomial(1, probs)
            return probs, y

        test_points = np.linspace(0.1, 0.9, 50)

        small_s, small_y = make_data(100)
        large_s, large_y = make_data(5000)

        cal_small = VennAbersCalibrator().fit(small_s, small_y)
        cal_large = VennAbersCalibrator().fit(large_s, large_y)

        unc_small = np.mean(cal_small.predict_intervals(test_points).uncertainty)
        unc_large = np.mean(cal_large.predict_intervals(test_points).uncertainty)

        assert unc_large < unc_small

    def test_scikit_learn_api_predict_proba_and_predict(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify predict_proba returns (N, 2) summing to 1.0 and predict returns {0, 1}."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        test_scores = np.array([0.1, 0.4, 0.6, 0.9])
        probas = calibrator.predict_proba(test_scores)

        assert probas.shape == (4, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-15)
        np.testing.assert_allclose(probas[:, 0] + probas[:, 1], 1.0, atol=1e-15)

        preds = calibrator.predict(test_scores, threshold=0.5)
        assert preds.shape == (4,)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds[0] == 0  # low score -> 0
        assert preds[3] == 1  # high score -> 1

    def test_transform_method(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify transform returns 1D array of calibrated point probabilities."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        test_scores = np.array([0.2, 0.8])
        transformed = calibrator.transform(test_scores)

        assert transformed.shape == (2,)
        assert 0.0 <= transformed[0] < transformed[1] <= 1.0
        np.testing.assert_allclose(transformed, calibrator.predict_proba(test_scores)[:, 1], atol=1e-15)

    def test_named_tuple_properties_and_unpacking(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify VennAbersIntervals can be unpacked as a 3-tuple and accessed via properties."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        test_scores = np.array([0.3, 0.7])
        res = calibrator.predict_intervals(test_scores)

        # 3-tuple unpacking
        p0, p1, p = res
        np.testing.assert_array_equal(p0, res.p0)
        np.testing.assert_array_equal(p1, res.p1)
        np.testing.assert_array_equal(p, res.p)

        # Aliases
        np.testing.assert_array_equal(res.p_lower, res.p0)
        np.testing.assert_array_equal(res.p_upper, res.p1)
        np.testing.assert_array_equal(res.p_point, res.p)
        np.testing.assert_array_equal(res.uncertainty, res.p1 - res.p0)
        np.testing.assert_array_equal(res.uncertainty_width, res.uncertainty)

    def test_paired_team_scores_input_shape(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify support for 2D inputs of shape (N, 2) representing [P(B), P(A)]."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        # Paired inputs [P(B), P(A)]
        p_a = np.array([0.3, 0.7, 0.85])
        p_b = 1.0 - p_a
        paired = np.column_stack([p_b, p_a])

        res = calibrator.predict_intervals(paired)
        res_single = calibrator.predict_intervals(p_a)

        np.testing.assert_allclose(res.p, res_single.p, atol=1e-15)
        np.testing.assert_allclose(res.p0, res_single.p0, atol=1e-15)
        np.testing.assert_allclose(res.p1, res_single.p1, atol=1e-15)

    def test_logits_outside_zero_one_automatically_converted(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify scores outside [0, 1] (such as raw log-odds / logits) are converted via expit."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        logits = np.array([-2.5, 0.0, 2.5])
        probs = expit(logits)

        res_logits = calibrator.predict_intervals(logits)
        res_probs = calibrator.predict_intervals(probs)

        np.testing.assert_allclose(res_logits.p, res_probs.p, atol=1e-15)
        np.testing.assert_allclose(res_logits.p0, res_probs.p0, atol=1e-15)
        np.testing.assert_allclose(res_logits.p1, res_probs.p1, atol=1e-15)

    def test_scalar_test_input_handling(
        self, synthetic_calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify passing a single scalar score returns a scalar VennAbersIntervals."""
        scores, y = synthetic_calibration_data
        calibrator = VennAbersCalibrator().fit(scores, y)

        res = calibrator.predict_intervals(0.65)
        assert float(res.p0) <= float(res.p) <= float(res.p1)
        assert float(res.uncertainty) == float(res.p1) - float(res.p0)

    def test_input_validation_errors(self) -> None:
        """Verify proper error handling for invalid or un-fitted states."""
        calibrator = VennAbersCalibrator()

        # Not fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.predict_intervals([0.5])

        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.predict_proba([0.5])

        # None inputs
        with pytest.raises(ValueError, match="must not be None"):
            calibrator.fit(None, [0, 1])

        with pytest.raises(ValueError, match="must not be None"):
            calibrator.fit([0.5, 0.6], None)

        # Empty inputs
        with pytest.raises(ValueError, match="must not be empty"):
            calibrator.fit([], [])

        # Mismatched lengths
        with pytest.raises(ValueError, match="Length mismatch"):
            calibrator.fit([0.3, 0.4], [0])

        # Non-binary labels
        with pytest.raises(ValueError, match="must be binary"):
            calibrator.fit([0.2, 0.5, 0.8], [0, 1, 2])

        # NaNs in inputs
        with pytest.raises(ValueError, match="NaN or infinite"):
            calibrator.fit([0.2, np.nan], [0, 1])

        calibrator.fit([0.2, 0.7], [0, 1])
        with pytest.raises(ValueError, match="NaN or infinite"):
            calibrator.predict_intervals([0.5, np.nan])


# ===========================================================================
# 2. ConformalRiskGater Tests
# ===========================================================================


class TestConformalRiskGater:
    def test_ev_lower_exact_formula(self) -> None:
        """Verify ev_lower = p_lower * (odds * (1.0 - tax_rate)) - 1.0."""
        gater = ConformalRiskGater(tax_rate=0.12, min_ev=0.0, max_uncertainty=0.08)

        # Example: p_lower = 0.60, odds = 2.0
        # effective_odds = 2.0 * 0.88 = 1.76
        # ev_lower = 0.60 * 1.76 - 1.0 = 1.056 - 1.0 = 0.056
        ev = gater.compute_ev_lower(0.60, 2.0)
        assert math.isclose(float(ev), 0.056, abs_tol=1e-9)

        # Vectorized
        p_low = np.array([0.50, 0.60, 0.70])
        odds = np.array([2.0, 2.0, 2.0])
        evs = gater.compute_ev_lower(p_low, odds)
        expected = p_low * (2.0 * 0.88) - 1.0
        np.testing.assert_allclose(evs, expected, atol=1e-12)

    def test_filter_bets_approves_actionable_bet(self) -> None:
        """Verify bet is approved when ev_lower >= min_ev and uncertainty <= max_uncertainty."""
        gater = ConformalRiskGater(tax_rate=0.12, min_ev=0.0, max_uncertainty=0.08)

        # p_lower = 0.60, p_upper = 0.64 (uncertainty = 0.04 <= 0.08)
        # odds = 2.0 -> ev_lower = 0.60 * 1.76 - 1.0 = 0.056 >= 0.0
        res = gater.filter_bets(0.60, 0.64, 2.0)

        assert res["is_actionable"] is True
        assert res["ev_lower"] > 0.0
        assert res["uncertainty"] == pytest.approx(0.04)
        assert res["uncertainty_width"] == pytest.approx(0.04)

    def test_filter_bets_rejects_negative_ev_lower(self) -> None:
        """Verify bet is rejected when even pessimistic bound has negative EV."""
        gater = ConformalRiskGater(tax_rate=0.12, min_ev=0.0, max_uncertainty=0.08)

        # p_lower = 0.50, p_upper = 0.54 (uncertainty = 0.04 <= 0.08)
        # odds = 2.0 -> ev_lower = 0.50 * 1.76 - 1.0 = 0.88 - 1.0 = -0.12 < 0.0
        res = gater.filter_bets(0.50, 0.54, 2.0)

        assert res["is_actionable"] is False
        assert res["ev_lower"] < 0.0

    def test_filter_bets_rejects_excessive_uncertainty_even_with_positive_ev(self) -> None:
        """Verify bet is rejected when interval width exceeds max_uncertainty despite positive EV."""
        gater = ConformalRiskGater(tax_rate=0.12, min_ev=0.0, max_uncertainty=0.08)

        # p_lower = 0.60, p_upper = 0.72 -> uncertainty = 0.12 > 0.08 (exceeds threshold!)
        # odds = 2.0 -> ev_lower = 0.056 > 0.0 (positive EV, but too uncertain)
        res = gater.filter_bets(0.60, 0.72, 2.0)

        assert res["is_actionable"] is False
        assert res["uncertainty"] > gater.max_uncertainty

    def test_filter_bets_vectorized_mask(self) -> None:
        """Verify filtering an array of bets returns a proper boolean mask and matched outputs."""
        gater = ConformalRiskGater(tax_rate=0.12, min_ev=0.0, max_uncertainty=0.08)

        p_lower = np.array([0.60, 0.50, 0.60, 0.70])
        p_upper = np.array([0.64, 0.55, 0.75, 0.74])
        odds = np.array([2.0, 2.0, 2.0, 1.8])

        # Bet 0: p_lower=0.60, unc=0.04 (<=0.08), ev=0.056 (>=0) -> ACTIONABLE
        # Bet 1: p_lower=0.50, unc=0.05 (<=0.08), ev=-0.12 (<0)  -> REJECT (negative EV)
        # Bet 2: p_lower=0.60, unc=0.15 (>0.08),  ev=0.056 (>=0) -> REJECT (too uncertain)
        # Bet 3: p_lower=0.70, unc=0.04 (<=0.08), ev=0.70 * (1.8*0.88) - 1.0 = 0.1088 (>=0) -> ACTIONABLE

        res = gater.filter_bets(p_lower, p_upper, odds)
        mask = res["is_actionable"]

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert list(mask) == [True, False, False, True]
        assert len(res["ev_lower"]) == 4
        assert len(res["uncertainty"]) == 4

    def test_filter_bets_rejects_invalid_odds(self) -> None:
        """Verify odds <= 1.0 are rejected as non-actionable."""
        gater = ConformalRiskGater()
        res_zero = gater.filter_bets(0.60, 0.64, 0.9)
        res_one = gater.filter_bets(0.60, 0.64, 1.0)

        assert res_zero["is_actionable"] is False
        assert res_one["is_actionable"] is False

    def test_filter_bets_rejects_nans_and_invalid_bounds(self) -> None:
        """Verify NaNs or inverted bounds (p_lower > p_upper) are rejected gracefully."""
        gater = ConformalRiskGater()

        # Inverted bounds
        res_inv = gater.filter_bets(0.70, 0.60, 2.0)
        assert res_inv["is_actionable"] is False

        # Out of bounds
        res_oob = gater.filter_bets(-0.1, 0.5, 2.0)
        assert res_oob["is_actionable"] is False

        # NaN in p_lower
        res_nan = gater.filter_bets(np.nan, 0.60, 2.0)
        assert res_nan["is_actionable"] is False

    def test_custom_tax_rates_and_min_ev(self) -> None:
        """Verify custom tax rates (e.g. 0% zero-tax exchange) and higher min_ev thresholds."""
        # Zero-tax market (e.g. Pinnacle or betting exchange)
        gater_no_tax = ConformalRiskGater(tax_rate=0.0, min_ev=0.05, max_uncertainty=0.08)

        # p_lower = 0.51, odds = 2.0
        # ev_lower = 0.51 * 2.0 - 1.0 = 0.02 < 0.05 min_ev -> REJECT
        res = gater_no_tax.filter_bets(0.51, 0.55, 2.0)
        assert res["is_actionable"] is False
        assert res["ev_lower"] == pytest.approx(0.02)

        # p_lower = 0.54, odds = 2.0
        # ev_lower = 0.54 * 2.0 - 1.0 = 0.08 >= 0.05 min_ev -> APPROVE
        res2 = gater_no_tax.filter_bets(0.54, 0.58, 2.0)
        assert res2["is_actionable"] is True
        assert res2["ev_lower"] == pytest.approx(0.08)

    def test_gater_parameter_validation(self) -> None:
        """Verify parameter validation during ConformalRiskGater initialization."""
        with pytest.raises(ValueError, match="tax_rate"):
            ConformalRiskGater(tax_rate=-0.05)

        with pytest.raises(ValueError, match="tax_rate"):
            ConformalRiskGater(tax_rate=1.0)

        with pytest.raises(ValueError, match="max_uncertainty"):
            ConformalRiskGater(max_uncertainty=0.0)

        with pytest.raises(ValueError, match="max_uncertainty"):
            ConformalRiskGater(max_uncertainty=-0.05)


# ===========================================================================
# 3. End-to-End Pipeline Integration Test
# ===========================================================================


class TestEndToEndPipelineIntegration:
    def test_conformal_calibration_and_risk_gating_pipeline(self) -> None:
        """Verify complete pipeline: training data -> calibrator -> intervals -> risk gating."""
        np.random.seed(777)
        n_cal = 2000
        n_test = 20

        # Simulate base model predictions and true outcomes
        true_logits = np.random.normal(0.0, 1.2, size=n_cal)
        y_cal = np.random.binomial(1, expit(true_logits))
        scores_cal = expit(true_logits * 1.8)  # overconfident

        # Fit calibrator
        calibrator = VennAbersCalibrator(symmetric=True).fit(scores_cal, y_cal)

        # New test match predictions
        test_scores = np.random.uniform(0.3, 0.8, size=n_test)
        # Market offered odds
        market_odds = np.random.uniform(1.6, 2.5, size=n_test)

        # 1. Obtain Venn-Abers calibrated intervals
        intervals = calibrator.predict_intervals(test_scores)
        assert len(intervals.p0) == n_test
        assert len(intervals.p1) == n_test
        assert len(intervals.p) == n_test

        # 2. Filter bets through ConformalRiskGater
        gater = ConformalRiskGater(tax_rate=0.12, min_ev=0.02, max_uncertainty=0.08)
        decisions = gater.filter_bets(intervals.p_lower, intervals.p_upper, market_odds)

        actionable_mask = decisions["is_actionable"]
        assert isinstance(actionable_mask, np.ndarray)

        # Verify invariant on every actionable bet
        for i, is_act in enumerate(actionable_mask):
            if is_act:
                # Conservative EV must meet min_ev
                assert decisions["ev_lower"][i] >= 0.02
                # Uncertainty must be strictly within threshold
                assert decisions["uncertainty"][i] <= 0.08
                # Bounds must be valid
                assert 0.0 <= intervals.p0[i] <= intervals.p1[i] <= 1.0
                assert market_odds[i] > 1.0
