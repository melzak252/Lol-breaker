"""Unit tests for EXP-040 candidate probability calibration components.

Validates:
1. TemperatureScalingCalibrator:
   - Scalar temperature optimization on synthetic overconfident logits.
   - Flattening of overconfident probabilities when T > 1.0.
   - Sharpening of underconfident probabilities when T < 1.0.
   - Strict binary symmetry: P(team_a) + P(team_b) == 1.0.
   - Multi-format input shapes (1D, 2D 1-col, 2D 2-col).
   - Probability bounds in [0, 1] and numerical stability with extreme logits.
   - Scikit-learn API compatibility (fit returns self, transform, predict, predict_proba).
2. BetaCalibrator:
   - Two-parameter ("ab") and three-parameter ("abm") Beta calibration (Kull et al., 2017).
   - Strict monotonicity: higher input scores strictly yield higher or equal calibrated probabilities.
   - Preservation of tails without excessive shrinkage or distortion.
   - Exact side-symmetry: P(s) + P(1 - s) == 1.0 when symmetric=True.
   - Identity mapping recovery on well-calibrated inputs.
   - Support for 'am' and 'a' variants.
   - Automatic sigmoid conversion for inputs outside [0, 1].
3. UncertaintyGatedCalibrator:
   - Zero shrinkage when rating uncertainty and market discrepancy are below thresholds.
   - Bayesian shrinkage towards market probability when discrepancy > threshold (0.20).
   - Bayesian shrinkage towards 0.5 base rate when average sigma > 2.5 and market is absent.
   - Monotonic increase of shrinkage weight with both uncertainty (sigma) and market discrepancy.
   - Exact boundary sharpness at thresholds.
   - Max shrinkage capping and custom base rates.
   - Strict binary side-symmetry under role inversion.
   - Vectorized matrix, tuple, and dictionary input unpacking.
4. Expected Calibration Error (ECE) and Brier Score Decomposition:
   - ECE computation, empty array handling, zero ECE on perfectly calibrated data.
   - Murphy (1973) Brier score decomposition (Reliability, Resolution, Uncertainty).
   - Range validation in [0, 1] and boundary condition stability.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from scipy.special import expit, logit

from betting_app.ml.calibration.candidate_calibration import (
    BetaCalibrator,
    TemperatureScalingCalibrator,
    UncertaintyGatedCalibrator,
    brier_score_decomposition,
    expected_calibration_error,
)


# ===========================================================================
# 1. TemperatureScalingCalibrator Tests
# ===========================================================================


class TestTemperatureScalingCalibrator:
    def test_temperature_optimization_recovers_synthetic_inflation(self) -> None:
        """When an overconfident model has inflated logits, T > 1.0 is recovered."""
        np.random.seed(42)
        n_samples = 3000
        true_logits = np.random.normal(0.0, 1.2, size=n_samples)
        true_probs = expit(true_logits)
        y_true = np.random.binomial(1, true_probs)

        # Model is overconfident by a factor of 2.2
        inflation_factor = 2.2
        model_logits = inflation_factor * true_logits

        calibrator = TemperatureScalingCalibrator(initial_temperature=1.0)
        calibrator.fit(model_logits, y_true)

        # Fitted temperature should closely match the ground-truth inflation factor
        assert calibrator.temperature_ > 1.8
        assert calibrator.temperature_ < 2.6
        assert math.isclose(calibrator.temperature_, inflation_factor, rel_tol=0.20)

    def test_flattens_overconfident_probabilities_when_t_gt_1(self) -> None:
        """When T > 1.0, extreme probabilities are pulled towards 0.5."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=2.0)
        calibrator.temperature_ = 2.0

        raw_logit = 3.0  # expit(3.0) ~ 0.9526
        p_raw = expit(raw_logit)

        p_cal = calibrator.transform(np.array([raw_logit]))[0]

        # 0.9526 flattened to expit(1.5) ~ 0.8176
        assert p_cal < p_raw
        assert p_cal > 0.5
        assert math.isclose(p_cal, expit(1.5), abs_tol=1e-6)

        # Negative logit: -3.0 -> expit(-3.0) ~ 0.0474 flattened to expit(-1.5) ~ 0.1824
        p_neg_raw = expit(-raw_logit)
        p_neg_cal = calibrator.transform(np.array([-raw_logit]))[0]
        assert p_neg_cal > p_neg_raw
        assert p_neg_cal < 0.5
        assert math.isclose(p_neg_cal, expit(-1.5), abs_tol=1e-6)

    def test_sharpens_underconfident_probabilities_when_t_lt_1(self) -> None:
        """When T < 1.0, probabilities are pushed away from 0.5."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=0.5)
        calibrator.temperature_ = 0.5

        raw_logit = 1.0  # expit(1.0) ~ 0.7311
        p_cal = calibrator.transform(np.array([raw_logit]))[0]

        # expit(1.0 / 0.5) = expit(2.0) ~ 0.8808 > 0.7311
        assert p_cal > expit(raw_logit)
        assert math.isclose(p_cal, expit(2.0), abs_tol=1e-6)

    def test_exact_binary_symmetry_opposing_logits(self) -> None:
        """P(team_a wins) + P(team_b wins) == 1.0 for opposing logits z and -z."""
        calibrator = TemperatureScalingCalibrator()
        calibrator.temperature_ = 1.85

        logits_a = np.linspace(-5.0, 5.0, 101)
        logits_b = -logits_a

        p_a = calibrator.transform(logits_a)
        p_b = calibrator.transform(logits_b)

        # Exact binary symmetry for all points
        assert np.allclose(p_a + p_b, 1.0, atol=1e-15)
        # Neutral point is exactly 0.5
        assert math.isclose(calibrator.transform(np.array([0.0]))[0], 0.5, abs_tol=1e-15)

    def test_paired_2d_logits_sum_to_one(self) -> None:
        """2D paired logits [z_0, z_1] return 2D array strictly summing to 1.0 per row."""
        calibrator = TemperatureScalingCalibrator()
        calibrator.temperature_ = 2.4

        paired_logits = np.array([
            [-1.5, 1.5],
            [2.0, -2.0],
            [0.0, 0.0],
            [3.5, -1.0],
        ])

        transformed = calibrator.transform(paired_logits)
        assert transformed.shape == (4, 2)
        assert np.allclose(transformed.sum(axis=1), 1.0, atol=1e-15)
        assert np.all((transformed >= 0.0) & (transformed <= 1.0))

    def test_2d_single_column_input_preserves_shape(self) -> None:
        """2D (N, 1) logits return 2D (N, 1) probabilities."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=1.5)
        logits_2d = np.array([[-1.0], [0.0], [2.0]])
        p_2d = calibrator.transform(logits_2d)
        assert p_2d.shape == (3, 1)
        assert np.all((p_2d >= 0.0) & (p_2d <= 1.0))

    def test_sklearn_api_predict_and_predict_proba(self) -> None:
        """Verify standard predict and predict_proba contracts."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=1.5)
        calibrator.temperature_ = 1.5

        logits = np.array([-2.0, 0.0, 2.0])
        probas = calibrator.predict_proba(logits)

        assert probas.shape == (3, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-15)
        assert probas[0, 1] < 0.5
        assert math.isclose(probas[1, 1], 0.5, abs_tol=1e-15)
        assert probas[2, 1] > 0.5

        preds = calibrator.predict(logits, threshold=0.5)
        assert np.array_equal(preds, np.array([0, 1, 1]))

    def test_fit_returns_self(self) -> None:
        """Scikit-learn convention: fit returns self."""
        calibrator = TemperatureScalingCalibrator()
        returned = calibrator.fit(np.array([1.0, -1.0]), np.array([1, 0]))
        assert returned is calibrator

    def test_probability_bounds_strictly_in_unit_interval(self) -> None:
        """Random logits over wide dynamic range produce probabilities strictly in [0, 1]."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=2.0)
        np.random.seed(10)
        random_logits = np.random.uniform(-50.0, 50.0, size=500)
        probs = calibrator.transform(random_logits)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.all(np.isfinite(probs))

    def test_numerical_stability_extreme_logits(self) -> None:
        """Extreme logits (+-100) do not produce NaN or overflow."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=2.0)
        calibrator.temperature_ = 2.0

        extreme_logits = np.array([-500.0, -100.0, 0.0, 100.0, 500.0])
        p_cal = calibrator.transform(extreme_logits)

        assert np.all(np.isfinite(p_cal))
        assert p_cal[0] >= 0.0
        assert p_cal[-1] <= 1.0
        assert np.all(p_cal >= 0.0) and np.all(p_cal <= 1.0)

    def test_invalid_input_validations(self) -> None:
        """Invalid inputs to fit and constructor raise appropriate exceptions."""
        with pytest.raises(ValueError, match="initial_temperature must be > 0"):
            TemperatureScalingCalibrator(initial_temperature=-1.0)

        calibrator = TemperatureScalingCalibrator()
        with pytest.raises(ValueError, match="empty logits"):
            calibrator.fit(np.array([]), np.array([]))

        with pytest.raises(ValueError, match="Mismatch"):
            calibrator.fit(np.array([1.0, 2.0]), np.array([1.0]))

        with pytest.raises(ValueError, match="binary labels"):
            calibrator.fit(np.array([1.0, 2.0]), np.array([0.0, 2.0]))

        with pytest.raises(ValueError, match="NaN or infinite"):
            calibrator.fit(np.array([1.0, np.nan]), np.array([0.0, 1.0]))

    def test_single_class_fallback(self) -> None:
        """When validation data has only one class, calibrator gracefully retains default."""
        calibrator = TemperatureScalingCalibrator(initial_temperature=1.25)
        calibrator.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
        assert calibrator.temperature_ == 1.25


# ===========================================================================
# 2. BetaCalibrator Tests
# ===========================================================================


class TestBetaCalibrator:
    def test_two_parameter_fit_monotonicity(self) -> None:
        """Two-parameter ('ab') Beta calibration maintains strict monotonicity."""
        np.random.seed(42)
        n = 1500
        raw_p = np.random.uniform(0.01, 0.99, size=n)
        distorted_logit = 1.6 * logit(raw_p)
        y_true = np.random.binomial(1, expit(distorted_logit))

        bc = BetaCalibrator(parameters="ab", symmetric=True)
        bc.fit(raw_p, y_true)

        # Monotonicity test: evaluation on fine grid
        grid = np.linspace(0.001, 0.999, 1000)
        cal_grid = bc.transform(grid)

        diffs = np.diff(cal_grid)
        assert np.all(diffs >= -1e-12), "Calibrated probabilities must be monotonically non-decreasing"

    def test_three_parameter_fit_and_transforms(self) -> None:
        """Three-parameter ('abm') Beta calibration fits slope and intercept."""
        np.random.seed(123)
        n = 1500
        raw_p = np.random.uniform(0.02, 0.98, size=n)
        y_true = np.random.binomial(1, expit(1.4 * logit(raw_p) + 0.2))

        bc = BetaCalibrator(parameters="abm", symmetric=False)
        bc.fit(raw_p, y_true)

        assert bc.a_ > 0.0
        assert bc.b_ > 0.0

        p_eval = bc.transform(np.array([0.1, 0.5, 0.9]))
        assert len(p_eval) == 3
        assert np.all((p_eval >= 0.0) & (p_eval <= 1.0))
        assert p_eval[0] < p_eval[1] < p_eval[2]

    def test_all_parameter_variants_fit(self) -> None:
        """Verify 'ab', 'abm', 'am', and 'a' variants fit and predict successfully."""
        np.random.seed(7)
        s = np.random.uniform(0.1, 0.9, size=200)
        y = np.random.binomial(1, s)

        for variant in ("ab", "abm", "am", "a"):
            bc = BetaCalibrator(parameters=variant, symmetric=False)  # type: ignore[arg-type]
            returned = bc.fit(s, y)
            assert returned is bc
            p = bc.transform(s)
            assert len(p) == len(s)
            assert np.all((p >= 0.0) & (p <= 1.0))

    def test_auto_conversion_of_logit_inputs(self) -> None:
        """Inputs containing values outside [0, 1] are treated as logits and sigmoid converted."""
        bc = BetaCalibrator(parameters="ab")
        raw_logits = np.array([-3.0, -1.0, 0.0, 1.5, 4.0])
        y = np.array([0, 0, 1, 1, 1])

        bc.fit(raw_logits, y)
        p_cal = bc.transform(raw_logits)
        assert len(p_cal) == 5
        assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))
        assert np.all(np.diff(p_cal) >= 0.0)

    def test_2d_single_column_input_preserves_shape(self) -> None:
        """2D (N, 1) probability input preserves (N, 1) shape."""
        bc = BetaCalibrator()
        probs_2d = np.array([[0.2], [0.5], [0.8]])
        p_2d = bc.transform(probs_2d)
        assert p_2d.shape == (3, 1)
        assert np.all((p_2d >= 0.0) & (p_2d <= 1.0))

    def test_strict_side_symmetry_with_symmetric_true(self) -> None:
        """P_cal(s) + P_cal(1 - s) == 1.0 for all s in [0, 1]."""
        bc = BetaCalibrator(parameters="ab", symmetric=True)
        # Assign asymmetric parameters to verify the symmetrizer restores exact symmetry
        bc.a_ = 1.8
        bc.b_ = 1.3
        bc.c_ = 0.25

        scores = np.linspace(0.01, 0.99, 50)
        p_dir = bc.transform(scores)
        p_rev = bc.transform(1.0 - scores)

        assert np.allclose(p_dir + p_rev, 1.0, atol=1e-14)
        assert math.isclose(bc.transform(np.array([0.5]))[0], 0.5, abs_tol=1e-14)

    def test_prevention_of_tail_distortion_and_collapse(self) -> None:
        """Extreme scores are preserved near 0 and 1 without collapsing to flat constant."""
        bc = BetaCalibrator(parameters="ab", symmetric=True)
        bc.a_ = 1.5
        bc.b_ = 1.5
        bc.c_ = 0.0

        p_low = bc.transform(np.array([0.001]))[0]
        p_high = bc.transform(np.array([0.999]))[0]

        assert p_low < 0.01
        assert p_high > 0.99
        assert p_low >= 0.0
        assert p_high <= 1.0

    def test_identity_mapping_recovery(self) -> None:
        """On already calibrated scores, Beta calibration recovers near-identity map."""
        np.random.seed(42)
        n = 3000
        well_cal_p = np.random.uniform(0.05, 0.95, size=n)
        y_true = np.random.binomial(1, well_cal_p)

        bc = BetaCalibrator(parameters="ab", symmetric=True)
        bc.fit(well_cal_p, y_true)

        # On well-calibrated data, a and b should be close to 1.0
        assert math.isclose(bc.a_, 1.0, rel_tol=0.25)
        assert math.isclose(bc.b_, 1.0, rel_tol=0.25)

        test_points = np.array([0.2, 0.4, 0.6, 0.8])
        calibrated = bc.transform(test_points)
        assert np.allclose(calibrated, test_points, atol=0.08)

    def test_beta_calibrator_api_predict_proba(self) -> None:
        """predict_proba returns 2-column format summing to 1.0."""
        bc = BetaCalibrator(parameters="ab")
        probs = np.array([0.1, 0.3, 0.7, 0.9])
        probas = bc.predict_proba(probs)

        assert probas.shape == (4, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-15)

    def test_invalid_parameters_raise_error(self) -> None:
        """Unsupported parameter variant raises ValueError."""
        with pytest.raises(ValueError, match="parameters must be one of"):
            BetaCalibrator(parameters="invalid_option")  # type: ignore[arg-type]


# ===========================================================================
# 3. UncertaintyGatedCalibrator Tests
# ===========================================================================


class TestUncertaintyGatedCalibrator:
    def test_low_uncertainty_and_discrepancy_applies_zero_shrinkage(self) -> None:
        """When sigma <= 2.5 and discrepancy <= 0.20, shrinkage weight is 0.0."""
        calibrator = UncertaintyGatedCalibrator(
            discrepancy_threshold=0.20,
            sigma_threshold=2.5,
        )

        p_model = 0.65
        sigma_a = 1.5
        sigma_b = 1.8  # avg_sigma = 1.65 <= 2.5
        p_market = 0.70  # discrepancy = 0.05 <= 0.20

        weight = calibrator.compute_shrinkage_weight(p_model, sigma_a, sigma_b, p_market)
        p_final = calibrator.calibrate(p_model, sigma_a, sigma_b, p_market)

        assert weight == 0.0
        assert math.isclose(p_final, p_model, abs_tol=1e-15)

    def test_high_uncertainty_triggers_bayesian_shrinkage(self) -> None:
        """When average sigma > 2.5, shrinkage towards market consensus activates."""
        calibrator = UncertaintyGatedCalibrator(
            discrepancy_threshold=0.20,
            sigma_threshold=2.5,
        )

        p_model = 0.70
        sigma_a = 3.5
        sigma_b = 3.5  # avg_sigma = 3.5 > 2.5
        p_market = 0.60  # discrepancy = 0.10 <= 0.20

        weight = calibrator.compute_shrinkage_weight(p_model, sigma_a, sigma_b, p_market)
        p_final = calibrator.calibrate(p_model, sigma_a, sigma_b, p_market)

        assert weight > 0.0
        assert p_market < p_final < p_model

    def test_high_discrepancy_triggers_bayesian_shrinkage(self) -> None:
        """When discrepancy > 0.20, shrinkage activates even under low uncertainty."""
        calibrator = UncertaintyGatedCalibrator(
            discrepancy_threshold=0.20,
            sigma_threshold=2.5,
        )

        p_model = 0.85
        sigma_a = 1.0
        sigma_b = 1.0  # avg_sigma = 1.0 <= 2.5
        p_market = 0.55  # discrepancy = 0.30 > 0.20

        weight = calibrator.compute_shrinkage_weight(p_model, sigma_a, sigma_b, p_market)
        p_final = calibrator.calibrate(p_model, sigma_a, sigma_b, p_market)

        assert weight > 0.0
        assert p_market < p_final < p_model

    def test_monotonic_increase_with_uncertainty(self) -> None:
        """Shrinkage weight increases monotonically as team uncertainty increases."""
        calibrator = UncertaintyGatedCalibrator(sigma_threshold=2.5)

        sigmas = [2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
        weights = [
            calibrator.compute_shrinkage_weight(0.60, s, s, p_market=0.62)
            for s in sigmas
        ]

        # First two should be 0.0 (at or below threshold)
        assert weights[0] == 0.0
        assert weights[1] == 0.0
        # Subsequent weights strictly increase
        for w1, w2 in zip(weights[1:-1], weights[2:], strict=True):
            assert w2 > w1

    def test_monotonic_increase_with_market_discrepancy(self) -> None:
        """Shrinkage weight increases monotonically as discrepancy increases."""
        calibrator = UncertaintyGatedCalibrator(discrepancy_threshold=0.20)

        discrepancies = [0.05, 0.15, 0.20, 0.25, 0.35, 0.50]
        weights = [
            calibrator.compute_shrinkage_weight(0.50 + d, 1.5, 1.5, p_market=0.50)
            for d in discrepancies
        ]

        assert weights[0] == 0.0
        assert weights[1] == 0.0
        assert weights[2] == 0.0
        for w1, w2 in zip(weights[2:-1], weights[3:], strict=True):
            assert w2 > w1

    def test_threshold_boundary_sharpness(self) -> None:
        """Verify exact behavior at and immediately above thresholds."""
        calibrator = UncertaintyGatedCalibrator(
            discrepancy_threshold=0.20,
            sigma_threshold=2.5,
        )

        w_at_sigma = calibrator.compute_shrinkage_weight(0.60, 2.5, 2.5, p_market=0.60)
        assert w_at_sigma == 0.0

        w_at_disc = calibrator.compute_shrinkage_weight(0.70, 1.0, 1.0, p_market=0.50)
        assert w_at_disc == 0.0

        w_above_sigma = calibrator.compute_shrinkage_weight(0.60, 2.51, 2.51, p_market=0.60)
        assert w_above_sigma > 0.0

        w_above_disc = calibrator.compute_shrinkage_weight(0.71, 1.0, 1.0, p_market=0.50)
        assert w_above_disc > 0.0

    def test_max_shrinkage_capping(self) -> None:
        """max_shrinkage caps the maximum shrinkage weight."""
        calibrator = UncertaintyGatedCalibrator(max_shrinkage=0.40)
        w = calibrator.compute_shrinkage_weight(0.99, 10.0, 10.0, p_market=0.01)
        assert w == 0.40

        p_final = calibrator.calibrate(0.99, 10.0, 10.0, p_market=0.01)
        assert math.isclose(p_final, 0.598, abs_tol=1e-6)

    def test_shrinkage_towards_base_rate_when_market_absent(self) -> None:
        """When market odds are absent (None/NaN), shrinkage is towards 0.5 base rate."""
        calibrator = UncertaintyGatedCalibrator(sigma_threshold=2.5, base_rate=0.5)

        p_model = 0.80
        sigma_a = 4.0
        sigma_b = 4.0  # avg_sigma = 4.0 > 2.5

        p_final_none = calibrator.calibrate(p_model, sigma_a, sigma_b, p_market=None)
        p_final_nan = calibrator.calibrate(p_model, sigma_a, sigma_b, p_market=float("nan"))

        assert 0.5 < p_final_none < p_model
        assert math.isclose(p_final_none, p_final_nan, abs_tol=1e-12)

    def test_custom_base_rate(self) -> None:
        """When market is absent, shrinks towards custom base rate."""
        calibrator = UncertaintyGatedCalibrator(sigma_threshold=2.0, base_rate=0.55)
        p_final = calibrator.calibrate(0.85, 4.0, 4.0, p_market=None)
        assert 0.55 < p_final < 0.85

    def test_fit_returns_self(self) -> None:
        """Scikit-learn style fit returns self."""
        calibrator = UncertaintyGatedCalibrator()
        assert calibrator.fit() is calibrator

    def test_probability_bounds_at_extremes(self) -> None:
        """Extreme probability inputs (0.0, 1.0) stay bounded in [0, 1]."""
        calibrator = UncertaintyGatedCalibrator()
        p0 = calibrator.calibrate(0.0, 3.0, 3.0, p_market=0.2)
        p1 = calibrator.calibrate(1.0, 3.0, 3.0, p_market=0.8)
        assert 0.0 <= p0 <= 1.0
        assert 0.0 <= p1 <= 1.0

    def test_exact_side_symmetry(self) -> None:
        """Role inversion preserves strict binary symmetry: P(A wins) + P(B wins) == 1.0."""
        calibrator = UncertaintyGatedCalibrator()

        cases = [
            (0.75, 3.2, 2.8, 0.60),
            (0.85, 1.2, 1.5, 0.55),
            (0.50, 4.0, 4.0, 0.50),
            (0.90, 3.8, 4.2, None),
            (0.30, 2.0, 2.0, 0.35),
        ]

        for p_m, s_a, s_b, p_mkt in cases:
            p_final_a = calibrator.calibrate(p_m, s_a, s_b, p_mkt)

            p_m_b = 1.0 - p_m
            s_a_b = s_b
            s_b_b = s_a
            p_mkt_b = (1.0 - p_mkt) if p_mkt is not None else None
            p_final_b = calibrator.calibrate(p_m_b, s_a_b, s_b_b, p_mkt_b)

            assert math.isclose(p_final_a + p_final_b, 1.0, abs_tol=1e-14), (
                f"Symmetry failed for case {(p_m, s_a, s_b, p_mkt)}: "
                f"{p_final_a} + {p_final_b} = {p_final_a + p_final_b}"
            )

    def test_vectorized_and_unpacked_transform(self) -> None:
        """Verify 2D matrix and dictionary input transformation."""
        calibrator = UncertaintyGatedCalibrator()

        matrix = np.array([
            [0.65, 1.0, 1.0, 0.65],  # low uncertainty & low disc -> no shrinkage
            [0.85, 3.5, 3.5, 0.55],  # high uncertainty & high disc -> shrinkage
        ])

        transformed = calibrator.transform(matrix)
        assert len(transformed) == 2
        assert math.isclose(transformed[0], 0.65, abs_tol=1e-6)
        assert transformed[1] < 0.85

        # Dict unpacking
        d = {"p_model": 0.70, "sigma_a": 3.0, "sigma_b": 3.0, "p_market": 0.50}
        p_dict = calibrator.transform(d)
        assert isinstance(p_dict, np.ndarray)


# ===========================================================================
# 4. ECE and Brier Score Decomposition Tests
# ===========================================================================


class TestCalibrationMetrics:
    def test_ece_perfect_calibration(self) -> None:
        """Perfect predictions have an ECE of exactly 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece == 0.0

    def test_ece_bounded_and_consistent(self) -> None:
        """ECE is bounded in [0, 1] and reflects miscalibration magnitude."""
        np.random.seed(42)
        p_well = np.random.uniform(0.1, 0.9, size=2000)
        y_well = np.random.binomial(1, p_well)
        ece_well = expected_calibration_error(y_well, p_well, n_bins=10)

        p_bad = 1.0 - p_well
        ece_bad = expected_calibration_error(y_well, p_bad, n_bins=10)

        assert 0.0 <= ece_well <= 0.05
        assert ece_bad > 0.30

    def test_ece_bins_variation(self) -> None:
        """ECE computes correctly with different bin numbers."""
        np.random.seed(42)
        y = np.random.binomial(1, 0.5, size=500)
        p = np.random.uniform(0.0, 1.0, size=500)

        ece_5 = expected_calibration_error(y, p, n_bins=5)
        ece_20 = expected_calibration_error(y, p, n_bins=20)

        assert 0.0 <= ece_5 <= 1.0
        assert 0.0 <= ece_20 <= 1.0

    def test_ece_empty_arrays_and_errors(self) -> None:
        """ECE gracefully handles empty arrays and validates invalid inputs."""
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

        with pytest.raises(ValueError, match="Length mismatch"):
            expected_calibration_error(np.array([1, 0]), np.array([0.5]))

        with pytest.raises(ValueError, match="positive integer"):
            expected_calibration_error(np.array([1]), np.array([0.5]), n_bins=0)

    def test_brier_score_decomposition_perfect_predictions(self) -> None:
        """Perfect forecasts have 0 Brier score and 0 Reliability error."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        decomp = brier_score_decomposition(y_true, y_prob, n_bins=10)

        assert decomp["brier_score"] == 0.0
        assert decomp["reliability"] == 0.0
        assert decomp["uncertainty"] == 0.25
        assert decomp["resolution"] == 0.25
        # Murphy identity: REL - RES + UNC == Brier
        assert math.isclose(
            decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"],
            decomp["brier_score"],
            abs_tol=1e-12,
        )

    def test_brier_score_decomposition_murphy_identity(self) -> None:
        """On arbitrary forecasts, REL - RES + UNC approximates empirical Brier score."""
        np.random.seed(99)
        y_true = np.random.binomial(1, 0.35, size=2000)
        y_prob = np.random.uniform(0.1, 0.8, size=2000)

        decomp = brier_score_decomposition(y_true, y_prob, n_bins=20)

        rel = decomp["reliability"]
        res = decomp["resolution"]
        unc = decomp["uncertainty"]
        brier = decomp["brier_score"]

        # Base rate uncertainty: p * (1 - p)
        mean_y = float(np.mean(y_true))
        assert math.isclose(unc, mean_y * (1.0 - mean_y), abs_tol=1e-12)

        # The difference between (REL - RES + UNC) and Brier is bounded by within-bin variance
        reconstructed_brier = rel - res + unc
        assert abs(reconstructed_brier - brier) < 0.005

    def test_brier_score_decomposition_empty_handling(self) -> None:
        """Empty input returns NaNs in dictionary."""
        decomp = brier_score_decomposition(np.array([]), np.array([]))
        assert math.isnan(decomp["brier_score"])
        assert math.isnan(decomp["reliability"])
        assert math.isnan(decomp["resolution"])
        assert math.isnan(decomp["uncertainty"])

    def test_brier_decomp_all_zeros_or_ones(self) -> None:
        """Base rate uncertainty is 0.0 when outcomes are homogeneous."""
        y_zeros = np.zeros(100)
        p = np.full(100, 0.2)
        decomp_zeros = brier_score_decomposition(y_zeros, p)
        assert decomp_zeros["uncertainty"] == 0.0
        assert decomp_zeros["brier_score"] > 0.0
