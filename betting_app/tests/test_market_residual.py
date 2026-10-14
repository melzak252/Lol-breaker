"""Unit tests for MarketResidualModel and ResidualEdgeDetector.

Validates:
1. MarketResidualModel:
   - Supervised residual learning (y_true - p_market).
   - Strict antisymmetry: predict_residual(-X) == -predict_residual(X) for Ridge and LightGBM.
   - Bounded clamping: residual predictions strictly inside [-max_residual, max_residual].
   - Convergence to market: when Delta_X = 0, predict_residual is 0.0 and predict_proba == p_market.
   - Calibrated probabilities: bounded within [min_prob, max_prob] ([0.01, 0.99]).
   - Strict binary symmetry under team inversion: P(Team A) + P(Team B) == 1.0.
   - Both 1D and 2D probability formats ([P(Team B), P(Team A)]).
   - Binary prediction thresholding.
   - Robust input validation and edge cases (NaNs, Infs, mismatched shapes, invalid labels).
   - Differential feature engineering with disagreement terms.
2. ResidualEdgeDetector:
   - Bookmaker overround margin calculation from decimal odds.
   - Net expected edge calculation with market margin and turnover tax.
   - Actionable signal generation for Team A and Team B (via inverted residual).
   - Dead-band non-actionable filtering when residual is within bookmaker spread.
   - Friction filtering where turnover tax eliminates marginal edges.
   - Batch edge detection.
   - Input validation and non-negative parameter constraints.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from betting_app.ml.models.market_residual import (
    EdgeSignal,
    MarketResidualModel,
    ResidualEdgeDetector,
    compute_differential_features,
)


# ===========================================================================
# 1. MarketResidualModel Tests
# ===========================================================================


class TestMarketResidualModel:
    """Tests for the MarketResidualModel architecture and invariants."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic dataset where positive feature implies true outcome > market."""
        np.random.seed(42)
        n_samples = 400
        n_features = 4
        # Differential features
        X = np.random.normal(0.0, 1.0, size=(n_samples, n_features))
        # Unbiased base market probability around 0.50
        p_market = np.clip(np.random.normal(0.50, 0.15, size=n_samples), 0.10, 0.90)
        # Market underweights feature 0: true win prob has additional positive weight on feature 0
        true_prob = np.clip(p_market + 0.10 * np.tanh(X[:, 0]), 0.02, 0.98)
        y_true = (np.random.uniform(0, 1, size=n_samples) < true_prob).astype(float)
        return X, y_true, p_market

    def test_fit_and_basic_residual_prediction(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Model fits and predicts residual direction matching feature weight."""
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(estimator_type="ridge", alpha=1.0, max_residual=0.15)
        model.fit(X, y_true, p_market)

        assert model.is_fitted_
        assert model.n_features_in_ == X.shape[1]

        # Samples with strong positive X[:, 0] should produce positive predicted residual
        strong_pos = np.array([[3.0, 0.0, 0.0, 0.0]])
        strong_neg = np.array([[-3.0, 0.0, 0.0, 0.0]])

        res_pos = model.predict_residual(strong_pos)[0]
        res_neg = model.predict_residual(strong_neg)[0]

        assert res_pos > 0.0
        assert res_neg < 0.0

    def test_strict_antisymmetry_ridge(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Strict antisymmetry: predict_residual(-X) == -predict_residual(X) for Ridge."""
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(estimator_type="ridge", alpha=1.0, max_residual=0.15)
        model.fit(X, y_true, p_market)

        np.random.seed(123)
        test_X = np.random.normal(0.0, 2.0, size=(50, X.shape[1]))

        res_pos = model.predict_residual(test_X)
        res_neg = model.predict_residual(-test_X)

        # Numerical equality to floating point precision
        assert np.allclose(res_neg, -res_pos, atol=1e-12)
        assert np.allclose(res_pos + res_neg, 0.0, atol=1e-12)

    def test_strict_antisymmetry_lightgbm(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Strict antisymmetry: predict_residual(-X) == -predict_residual(X) for LightGBM."""
        pytest.importorskip("lightgbm")
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(
            estimator_type="lightgbm",
            l2_reg=2.0,
            max_residual=0.15,
            lgb_params={"n_estimators": 25, "min_child_samples": 5},
        )
        model.fit(X, y_true, p_market)

        np.random.seed(456)
        test_X = np.random.normal(0.0, 2.0, size=(50, X.shape[1]))

        res_pos = model.predict_residual(test_X)
        res_neg = model.predict_residual(-test_X)

        assert np.allclose(res_neg, -res_pos, atol=1e-12)
        assert np.allclose(res_pos + res_neg, 0.0, atol=1e-12)

    def test_bounded_clamping(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Residual predictions are strictly clamped to [-max_residual, max_residual]."""
        X, y_true, p_market = synthetic_data
        max_bound = 0.12
        model = MarketResidualModel(estimator_type="ridge", alpha=0.01, max_residual=max_bound)
        model.fit(X, y_true, p_market)

        # Extreme features that would otherwise produce large unconstrained linear predictions
        extreme_X = np.array([
            [1000.0, 500.0, -200.0, 800.0],
            [-1000.0, -500.0, 200.0, -800.0],
            [50.0, 20.0, -10.0, 30.0],
        ])

        residuals = model.predict_residual(extreme_X)
        assert np.all(residuals >= -max_bound)
        assert np.all(residuals <= max_bound)
        assert math.isclose(residuals[0], max_bound, rel_tol=1e-9)
        assert math.isclose(residuals[1], -max_bound, rel_tol=1e-9)

    def test_convergence_to_market_when_x_is_zero(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """When Delta_X = 0 (identical teams), residual is 0.0 and probability converges to market."""
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(estimator_type="ridge", alpha=1.0)
        model.fit(X, y_true, p_market)

        zero_X = np.zeros((10, X.shape[1]))
        res_zero = model.predict_residual(zero_X)

        # Residual must be exactly 0.0
        assert np.allclose(res_zero, 0.0, atol=1e-14)

        # Calibrated probability must equal p_market for any valid market probabilities
        test_p_market = np.array([0.15, 0.30, 0.50, 0.70, 0.85, 0.50, 0.42, 0.61, 0.20, 0.80])
        p_final = model.predict_proba(zero_X, test_p_market, return_2d=False)

        assert np.allclose(p_final, test_p_market, atol=1e-14)

    def test_probability_output_constraints(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Calibrated probabilities are strictly bounded within [min_prob, max_prob]."""
        X, y_true, p_market = synthetic_data
        min_p = 0.02
        max_p = 0.98
        model = MarketResidualModel(min_prob=min_p, max_prob=max_p, max_residual=0.15)
        model.fit(X, y_true, p_market)

        # Near boundary market probabilities with aligned residuals
        test_X = np.array([
            [100.0, 0.0, 0.0, 0.0],
            [-100.0, 0.0, 0.0, 0.0],
        ])
        near_boundary_market = np.array([0.95, 0.05])

        p_final = model.predict_proba(test_X, near_boundary_market)
        assert np.all(p_final >= min_p)
        assert np.all(p_final <= max_p)
        assert math.isclose(p_final[0], max_p)
        assert math.isclose(p_final[1], min_p)

    def test_strict_binary_symmetry_under_team_inversion(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Binary symmetry: P(A wins) + P(B wins) == 1.0 under role inversion."""
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(estimator_type="ridge", alpha=1.0)
        model.fit(X, y_true, p_market)

        np.random.seed(999)
        test_X = np.random.normal(0.0, 1.5, size=(60, X.shape[1]))
        test_p_market = np.random.uniform(0.10, 0.90, size=60)

        # Team A probability with features X and market p
        p_team_a = model.predict_proba(test_X, test_p_market, return_2d=False)

        # Team B probability with inverted features -X and market (1 - p)
        p_team_b = model.predict_proba(-test_X, 1.0 - test_p_market, return_2d=False)

        # Sum must be identically 1.0 across all matches
        total = p_team_a + p_team_b
        assert np.allclose(total, 1.0, atol=1e-14)

    def test_2d_probability_format_and_convenience_methods(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """predict_proba(return_2d=True) and predict_proba_2d return shape (N, 2) summing to 1."""
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(estimator_type="ridge", alpha=1.0)
        model.fit(X, y_true, p_market)

        test_X = X[:10]
        test_pm = p_market[:10]

        p_1d = model.predict_proba(test_X, test_pm, return_2d=False)
        p_2d = model.predict_proba(test_X, test_pm, return_2d=True)
        p_2d_conv = model.predict_proba_2d(test_X, test_pm)

        assert p_1d.shape == (10,)
        assert p_2d.shape == (10, 2)
        assert np.allclose(p_2d, p_2d_conv)

        # Column 0 is P(Team B) = 1 - P(Team A), Column 1 is P(Team A)
        assert np.allclose(p_2d[:, 1], p_1d)
        assert np.allclose(p_2d[:, 0], 1.0 - p_1d)
        assert np.allclose(p_2d.sum(axis=1), 1.0, atol=1e-14)

        # Binary prediction method
        preds = model.predict(test_X, test_pm, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1})
        assert np.array_equal(preds, (p_1d >= 0.5).astype(int))

    def test_fit_with_sample_weights(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Fitting with sample weights runs correctly and respects weights."""
        X, y_true, p_market = synthetic_data
        weights = np.ones(len(X))
        weights[:50] = 5.0

        model = MarketResidualModel(estimator_type="ridge", alpha=1.0)
        model.fit(X, y_true, p_market, sample_weight=weights)

        assert model.is_fitted_
        preds = model.predict_residual(X[:10])
        assert len(preds) == 10

    def test_scalar_and_broadcasting_p_market(
        self, synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """predict_proba supports scalar float p_market broadcasting across all rows."""
        X, y_true, p_market = synthetic_data
        model = MarketResidualModel(estimator_type="ridge").fit(X, y_true, p_market)

        test_X = X[:5]
        p_scalar = model.predict_proba(test_X, p_market=0.50)
        assert len(p_scalar) == 5
        p_array = model.predict_proba(test_X, p_market=np.full(5, 0.50))
        assert np.allclose(p_scalar, p_array)

    def test_error_handling_and_input_validation(self) -> None:
        """Model raises clear errors on invalid inputs and unfitted state."""
        model = MarketResidualModel()

        # NotFittedError before fit
        with pytest.raises(NotFittedError, match="not fitted yet"):
            model.predict_residual(np.array([[1.0, 2.0]]))

        with pytest.raises(NotFittedError, match="not fitted yet"):
            model.predict_proba(np.array([[1.0, 2.0]]), 0.5)

        # Invalid constructor params
        with pytest.raises(ValueError, match="max_residual must be positive"):
            MarketResidualModel(max_residual=0.0)
        with pytest.raises(ValueError, match="Invalid probability bounds"):
            MarketResidualModel(min_prob=0.8, max_prob=0.2)
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            MarketResidualModel(alpha=-1.0)
        with pytest.raises(ValueError, match="l2_reg must be non-negative"):
            MarketResidualModel(l2_reg=-0.5)

        # Empty inputs
        with pytest.raises(ValueError, match="must contain at least one sample"):
            model.fit(np.empty((0, 2)), np.array([]), np.array([]))

        # NaNs and Infs in X
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            model.fit(
                np.array([[1.0, np.nan]]),
                np.array([1.0]),
                np.array([0.5]),
            )

        # Length mismatch in fit
        with pytest.raises(ValueError, match="Length mismatch"):
            model.fit(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([1.0]),
                np.array([0.5, 0.6]),
            )

        # Invalid labels y_true
        with pytest.raises(ValueError, match="must lie within \\[0, 1\\]"):
            model.fit(
                np.array([[1.0, 2.0]]),
                np.array([1.5]),
                np.array([0.5]),
            )

        # Invalid p_market
        with pytest.raises(ValueError, match="must lie strictly within \\(0, 1\\)"):
            model.fit(
                np.array([[1.0, 2.0]]),
                np.array([1.0]),
                np.array([1.0]),
            )

        # Dimension mismatch at prediction time
        model.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 0.0]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            model.predict_residual(np.array([[1.0, 2.0, 3.0]]))

        # Unsupported estimator type
        with pytest.raises(ValueError, match="Unsupported estimator_type"):
            m_bad = MarketResidualModel(estimator_type="unsupported")
            m_bad.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 0.0]), np.array([0.5, 0.5]))


# ===========================================================================
# 2. Differential Features Helper Tests
# ===========================================================================


class TestDifferentialFeatures:
    """Tests for compute_differential_features helper."""

    def test_feature_subtraction_and_antisymmetry(self) -> None:
        """Delta_X = X_A - X_B and inverting teams negates Delta_X."""
        a = np.array([[10.0, 5.0], [2.0, 8.0]])
        b = np.array([[4.0, 7.0], [3.0, 1.0]])

        diff_ab = compute_differential_features(a, b)
        diff_ba = compute_differential_features(b, a)

        expected = np.array([[6.0, -2.0], [-1.0, 7.0]])
        assert np.allclose(diff_ab, expected)
        assert np.allclose(diff_ba, -expected)
        assert np.allclose(diff_ab + diff_ba, 0.0)

    def test_disagreement_feature_inclusion(self) -> None:
        """Disagreement vector is appended and inverts under role swap."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[0.5, 1.0]])
        disag = np.array([0.15])

        res = compute_differential_features(a, b, disagreement=disag)
        assert res.shape == (1, 3)
        assert np.allclose(res, np.array([[0.5, 1.0, 0.15]]))

    def test_differential_shape_mismatch(self) -> None:
        """Shape mismatch between team features raises ValueError."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_differential_features(a, b)


# ===========================================================================
# 3. ResidualEdgeDetector Tests
# ===========================================================================


class TestResidualEdgeDetector:
    """Tests for the ResidualEdgeDetector transaction friction and signal logic."""

    def test_margin_from_odds(self) -> None:
        """Correct overround margin calculation from decimal odds."""
        # Fair even money odds 2.00 / 2.00 -> 0% margin
        m_zero = ResidualEdgeDetector.margin_from_odds(2.0, 2.0)
        assert math.isclose(m_zero, 0.0, abs_tol=1e-12)

        # Standard 1.90 / 1.90 odds -> (1/1.9 + 1/1.9) - 1 = 2/1.9 - 1 ~= 0.05263
        m_standard = ResidualEdgeDetector.margin_from_odds(1.90, 1.90)
        assert math.isclose(m_standard, 2.0 / 1.90 - 1.0, rel_tol=1e-9)

        # Vectorized odds
        odds_a = np.array([1.90, 1.50, 2.50])
        odds_b = np.array([1.90, 2.60, 1.55])
        margins = ResidualEdgeDetector.margin_from_odds(odds_a, odds_b)
        assert len(margins) == 3
        assert np.all(margins > 0.0)

        # Invalid odds <= 1.0
        with pytest.raises(ValueError, match="strictly greater than 1.0"):
            ResidualEdgeDetector.margin_from_odds(1.0, 2.0)

    def test_compute_expected_edge_formula(self) -> None:
        """Formula: expected_edge = predicted_residual - (market_margin / 2.0) - turnover_tax."""
        detector = ResidualEdgeDetector(turnover_tax=0.0, default_market_margin=0.06)

        # predicted_residual = 0.08, margin = 0.06 -> half margin = 0.03 -> edge = 0.05
        edge = detector.compute_expected_edge(predicted_residual=0.08, market_margin=0.06)
        assert math.isclose(edge, 0.05, rel_tol=1e-9)

        # With turnover tax = 0.02 -> edge = 0.08 - 0.03 - 0.02 = 0.03
        edge_tax = detector.compute_expected_edge(
            predicted_residual=0.08, market_margin=0.06, turnover_tax=0.02
        )
        assert math.isclose(edge_tax, 0.03, rel_tol=1e-9)

    def test_actionable_signal_team_a(self) -> None:
        """Positive residual exceeding friction generates actionable signal on Team A."""
        detector = ResidualEdgeDetector(min_edge=0.02, turnover_tax=0.0, default_market_margin=0.04)

        # residual = +0.07, margin = 0.04 -> half vig = 0.02 -> net edge A = +0.05 >= 0.02
        signal = detector.detect_edge(predicted_residual=0.07, market_margin=0.04)

        assert isinstance(signal, EdgeSignal)
        assert signal.is_actionable is True
        assert signal.recommended_side == "team_a"
        assert math.isclose(signal.recommended_edge, 0.05, rel_tol=1e-9)
        assert math.isclose(signal.expected_edge_a, 0.05, rel_tol=1e-9)
        assert math.isclose(signal.expected_edge_b, -0.09, rel_tol=1e-9)

    def test_actionable_signal_team_b(self) -> None:
        """Negative residual on Team A implies positive residual on Team B, signaling Team B."""
        detector = ResidualEdgeDetector(min_edge=0.02, turnover_tax=0.0, default_market_margin=0.04)

        # residual on A = -0.07 -> residual on B = +0.07 -> net edge B = +0.05 >= 0.02
        signal = detector.detect_edge(predicted_residual=-0.07, market_margin=0.04)

        assert signal.is_actionable is True
        assert signal.recommended_side == "team_b"
        assert math.isclose(signal.recommended_edge, 0.05, rel_tol=1e-9)
        assert math.isclose(signal.expected_edge_a, -0.09, rel_tol=1e-9)
        assert math.isclose(signal.expected_edge_b, 0.05, rel_tol=1e-9)

    def test_dead_band_non_actionable(self) -> None:
        """Small residuals swallowed by vig do not trigger an actionable signal."""
        detector = ResidualEdgeDetector(min_edge=0.02, turnover_tax=0.0, default_market_margin=0.05)

        # residual = +0.02, margin = 0.05 -> half vig = 0.025 -> edge A = -0.005 < 0.02
        # edge B = -0.02 - 0.025 = -0.045 < 0.02
        signal = detector.detect_edge(predicted_residual=0.02, market_margin=0.05)

        assert signal.is_actionable is False
        assert signal.recommended_side is None
        assert signal.recommended_edge == 0.0

    def test_turnover_tax_filters_marginal_edges(self) -> None:
        """Turnover tax reduces edge below threshold, correctly filtering out unviable bets."""
        # Without tax: residual 0.05, margin 0.04 -> edge = 0.03 >= 0.02 (actionable)
        detector_no_tax = ResidualEdgeDetector(min_edge=0.02, turnover_tax=0.0)
        sig_no_tax = detector_no_tax.detect_edge(predicted_residual=0.05, market_margin=0.04)
        assert sig_no_tax.is_actionable is True

        # With turnover tax 0.02 -> edge = 0.05 - 0.02 - 0.02 = 0.01 < 0.02 (not actionable)
        detector_tax = ResidualEdgeDetector(min_edge=0.02, turnover_tax=0.02)
        sig_tax = detector_tax.detect_edge(predicted_residual=0.05, market_margin=0.04)
        assert sig_tax.is_actionable is False
        assert sig_tax.recommended_side is None

    def test_detect_edge_from_odds(self) -> None:
        """Detect edge computes margin from odds when market_margin is omitted."""
        detector = ResidualEdgeDetector(min_edge=0.02)
        # Odds 1.90 / 1.90 has margin ~0.05263. Half margin ~0.02632
        # Residual +0.08 -> edge = 0.08 - 0.02632 = 0.05368 >= 0.02
        signal = detector.detect_edge(predicted_residual=0.08, odds_a=1.90, odds_b=1.90)

        assert signal.is_actionable is True
        assert signal.recommended_side == "team_a"
        assert math.isclose(signal.market_margin, 2.0 / 1.90 - 1.0, rel_tol=1e-9)

    def test_batch_edge_detection(self) -> None:
        """Batch detection processes an array of residuals and margins into signals."""
        detector = ResidualEdgeDetector(min_edge=0.02, default_market_margin=0.04)
        residuals = np.array([0.08, -0.09, 0.01, 0.00, 0.12])
        margins = np.array([0.04, 0.04, 0.05, 0.04, 0.06])

        signals = detector.detect_edges_batch(residuals, margins)
        assert len(signals) == 5

        assert signals[0].is_actionable is True and signals[0].recommended_side == "team_a"
        assert signals[1].is_actionable is True and signals[1].recommended_side == "team_b"
        assert signals[2].is_actionable is False and signals[2].recommended_side is None
        assert signals[3].is_actionable is False and signals[3].recommended_side is None
        assert signals[4].is_actionable is True and signals[4].recommended_side == "team_a"

    def test_detector_invalid_parameters(self) -> None:
        """Negative thresholds in ResidualEdgeDetector raise ValueError."""
        with pytest.raises(ValueError, match="min_edge must be non-negative"):
            ResidualEdgeDetector(min_edge=-0.01)
        with pytest.raises(ValueError, match="turnover_tax must be non-negative"):
            ResidualEdgeDetector(turnover_tax=-0.05)
        with pytest.raises(ValueError, match="default_market_margin must be non-negative"):
            ResidualEdgeDetector(default_market_margin=-0.02)
