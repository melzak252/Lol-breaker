"""Market Residual Learning Model and Edge Detector for Sports Betting.

Instead of predicting raw binary match outcomes y directly from scratch, the
Market Residual Model predicts the residual market inefficiency:
    target_residual = y_true - p_market

Key Architectural Guarantees:
1. Strict Antisymmetry:
   Flipping team order (B vs A) inverts differential features (Delta_X -> -Delta_X),
   and the model guarantees:
       predict_residual(-X) == -predict_residual(X)
2. Market Convergence at Equivalence:
   When feature differential Delta_X = 0 (two identical teams), predict_residual(0) == 0.0,
   so predict_proba converges exactly to p_market.
3. Bounded Clamping:
   Predicted residuals are strictly clamped to [-max_residual, max_residual]
   (e.g. +/- 0.15) to prevent overshooting bookmaker odds.
4. Calibrated Probability & Binary Symmetry:
   p_final = np.clip(p_market + predicted_residual, min_prob, max_prob)
   Guarantees P(A wins) + P(B wins) == 1.0 under team inversion.
5. Transaction Friction & Edge Detection:
   ResidualEdgeDetector evaluates expected edge after bookmaker vig and turnover tax:
       expected_edge = predicted_residual - (market_margin / 2.0) - turnover_tax
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    _HAS_LIGHTGBM = False


def _ensure_2d_float(arr: Any, name: str) -> np.ndarray:
    """Validate and convert input to a 2D float64 numpy array."""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    np_arr = np.asarray(arr, dtype=float)
    if np_arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if np_arr.ndim == 1:
        np_arr = np_arr.reshape(-1, 1)
    elif np_arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {np_arr.shape}")
    if np.isnan(np_arr).any() or np.isinf(np_arr).any():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np_arr


def _ensure_1d_float(arr: Any, name: str) -> np.ndarray:
    """Validate and convert input to a 1D float64 numpy array."""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    np_arr = np.asarray(arr, dtype=float).ravel()
    if np_arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if np.isnan(np_arr).any() or np.isinf(np_arr).any():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np_arr


def compute_differential_features(
    features_a: np.ndarray,
    features_b: np.ndarray,
    disagreement: np.ndarray | None = None,
) -> np.ndarray:
    """Compute differential features Delta_X = X_A - X_B.

    Parameters
    ----------
    features_a : np.ndarray
        Feature matrix for Team A of shape (n_samples, n_features).
    features_b : np.ndarray
        Feature matrix for Team B of shape (n_samples, n_features).
    disagreement : np.ndarray, optional
        Optional model-market disagreement vector (e.g. p_base - p_market)
        of shape (n_samples,) or (n_samples, 1). Appended to differentials.

    Returns
    -------
    np.ndarray
        Differential feature matrix of shape (n_samples, n_features (+ 1)).
    """
    x_a = _ensure_2d_float(features_a, "features_a")
    x_b = _ensure_2d_float(features_b, "features_b")
    if x_a.shape != x_b.shape:
        raise ValueError(f"Shape mismatch between features_a {x_a.shape} and features_b {x_b.shape}")

    delta_x = x_a - x_b
    if disagreement is not None:
        disag = _ensure_2d_float(disagreement, "disagreement")
        if len(disag) != len(delta_x):
            raise ValueError(
                f"Length mismatch: disagreement has {len(disag)} rows, features have {len(delta_x)}"
            )
        delta_x = np.hstack([delta_x, disag])

    return delta_x


@dataclass(frozen=True)
class EdgeSignal:
    """Actionable value signal evaluated against market margin and friction."""

    predicted_residual: float
    market_margin: float
    expected_edge_a: float
    expected_edge_b: float
    is_actionable: bool
    recommended_side: str | None  # "team_a", "team_b", or None
    recommended_edge: float
    turnover_tax: float = 0.0


class MarketResidualModel(BaseEstimator, RegressorMixin):
    """Sports betting residual learning model for market inefficiency extraction.

    Instead of predicting raw binary outcome y directly from scratch, this model
    predicts the market residual:
        target_residual = y_true - p_market

    Parameters
    ----------
    estimator_type : {"ridge", "lightgbm"} or custom regressor, default="ridge"
        Core regression estimator. If "ridge", uses L2 regularized Ridge without intercept.
        If "lightgbm", uses LGBMRegressor with L2 regularization.
    alpha : float, default=1.0
        L2 regularization strength for Ridge regression.
    l2_reg : float, default=1.0
        L2 regularization strength (reg_lambda) for LightGBM.
    max_residual : float, default=0.15
        Maximum absolute residual clamping bound (+/- max_residual). Prevents overshooting.
    min_prob : float, default=0.01
        Lower probability bound for clipped calibrated probability.
    max_prob : float, default=0.99
        Upper probability bound for clipped calibrated probability.
    symmetrize_training : bool, default=True
        If True, augments training data with negated pairs (-X, -target_residual)
        to enforce balanced learning around zero.
    antisymmetric_eval : bool, default=True
        If True, enforces strict mathematical antisymmetry at prediction time via
        canonical odd decomposition: r(X) = 0.5 * (g(X) - g(-X)).
    random_state : int, default=42
        Seed for reproducible random state in base estimators.
    lgb_params : dict, optional
        Additional parameters passed to LGBMRegressor if estimator_type="lightgbm".
    """

    def __init__(
        self,
        estimator_type: str | Any = "ridge",
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        max_residual: float = 0.15,
        min_prob: float = 0.01,
        max_prob: float = 0.99,
        symmetrize_training: bool = True,
        antisymmetric_eval: bool = True,
        random_state: int = 42,
        lgb_params: dict[str, Any] | None = None,
    ) -> None:
        if max_residual <= 0.0:
            raise ValueError(f"max_residual must be positive, got {max_residual}")
        if not (0.0 <= min_prob < max_prob <= 1.0):
            raise ValueError(f"Invalid probability bounds: min_prob={min_prob}, max_prob={max_prob}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if l2_reg < 0.0:
            raise ValueError(f"l2_reg must be non-negative, got {l2_reg}")

        self.estimator_type = estimator_type
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.max_residual = max_residual
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.symmetrize_training = symmetrize_training
        self.antisymmetric_eval = antisymmetric_eval
        self.random_state = random_state
        self.lgb_params = lgb_params or {}

        self.model_: Any = None
        self.n_features_in_: int = 0
        self.is_fitted_: bool = False

    def _build_estimator(self) -> Any:
        """Instantiate the configured base regression estimator."""
        if isinstance(self.estimator_type, str):
            est_name = self.estimator_type.lower()
            if est_name == "ridge":
                return Ridge(
                    alpha=self.alpha,
                    fit_intercept=False,
                    random_state=self.random_state,
                )
            elif est_name in {"lightgbm", "lgbm"}:
                if not _HAS_LIGHTGBM:
                    raise ImportError(
                        "lightgbm is not installed. Install lightgbm or use estimator_type='ridge'."
                    )
                params: dict[str, Any] = {
                    "reg_alpha": 0.0,
                    "reg_lambda": self.l2_reg,
                    "random_state": self.random_state,
                    "verbosity": -1,
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                }
                params.update(self.lgb_params)
                return lgb.LGBMRegressor(**params)
            else:
                raise ValueError(f"Unsupported estimator_type: {self.estimator_type}")
        else:
            # Custom regressor instance
            from sklearn.base import clone
            return clone(self.estimator_type)

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        p_market: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> MarketResidualModel:
        """Fit the residual estimator on differential features and market error.

        Parameters
        ----------
        X : np.ndarray
            Differential feature matrix Delta_X of shape (n_samples, n_features).
        y_true : np.ndarray
            Binary ground-truth labels in {0, 1} of shape (n_samples,).
        p_market : np.ndarray
            Market implied probabilities in (0, 1) of shape (n_samples,).
        sample_weight : np.ndarray, optional
            Sample weights of shape (n_samples,).

        Returns
        -------
        MarketResidualModel
            Fitted instance.
        """
        x_arr = _ensure_2d_float(X, "X")
        y_arr = _ensure_1d_float(y_true, "y_true")
        pm_arr = _ensure_1d_float(p_market, "p_market")

        n_samples = len(x_arr)
        if len(y_arr) != n_samples or len(pm_arr) != n_samples:
            raise ValueError(
                f"Length mismatch: X has {n_samples} samples, y_true has {len(y_arr)}, "
                f"p_market has {len(pm_arr)}"
            )

        if not np.all((y_arr >= 0.0) & (y_arr <= 1.0)):
            raise ValueError("All y_true values must lie within [0, 1]")
        if not np.all((pm_arr > 0.0) & (pm_arr < 1.0)):
            raise ValueError("All p_market values must lie strictly within (0, 1)")

        # Target residual: actual outcome minus market implied probability
        target_residual = y_arr - pm_arr

        if self.symmetrize_training:
            # Augment dataset with negated features and negated residuals
            # Team (B, A) inversion has features -Delta_X and residual -(y - p_market)
            x_fit = np.vstack([x_arr, -x_arr])
            target_fit = np.concatenate([target_residual, -target_residual])
            if sample_weight is not None:
                sw_arr = _ensure_1d_float(sample_weight, "sample_weight")
                if len(sw_arr) != n_samples:
                    raise ValueError("sample_weight length mismatch")
                sw_fit = np.concatenate([sw_arr, sw_arr])
            else:
                sw_fit = None
        else:
            x_fit = x_arr
            target_fit = target_residual
            sw_fit = _ensure_1d_float(sample_weight, "sample_weight") if sample_weight is not None else None

        estimator = self._build_estimator()
        if sw_fit is not None:
            estimator.fit(x_fit, target_fit, sample_weight=sw_fit)
        else:
            estimator.fit(x_fit, target_fit)

        self.model_ = estimator
        self.n_features_in_ = x_arr.shape[1]
        self.is_fitted_ = True
        return self

    def predict_residual(self, X: np.ndarray) -> np.ndarray:
        """Predict clamped market residual for differential features Delta_X.

        Guarantees strict antisymmetry: predict_residual(-X) == -predict_residual(X),
        and predict_residual(0) == 0.0.

        Parameters
        ----------
        X : np.ndarray
            Differential features of shape (n_samples, n_features) or (n_features,).

        Returns
        -------
        np.ndarray
            1D array of predicted residuals clamped to [-max_residual, max_residual].
        """
        if not self.is_fitted_ or self.model_ is None:
            raise NotFittedError("MarketResidualModel is not fitted yet. Call fit() first.")

        x_arr = _ensure_2d_float(X, "X")
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: X has {x_arr.shape[1]} features, "
                f"expected {self.n_features_in_}"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if self.antisymmetric_eval:
                # Canonical odd function projection: r(X) = 0.5 * (g(X) - g(-X))
                # Mathematically guarantees r(-X) = -r(X) and r(0) = 0.0 for ANY base regressor
                pred_pos = self.model_.predict(x_arr)
                pred_neg = self.model_.predict(-x_arr)
                raw_residual = 0.5 * (pred_pos - pred_neg)
            else:
                raw_residual = self.model_.predict(x_arr)
        raw_residual = np.asarray(raw_residual, dtype=float).ravel()
        # Clamping to symmetric bounds [-max_residual, max_residual] preserves antisymmetry
        clamped = np.clip(raw_residual, -self.max_residual, self.max_residual)
        return clamped

    def predict_proba(
        self,
        X: np.ndarray,
        p_market: np.ndarray | float,
        return_2d: bool = False,
    ) -> np.ndarray:
        """Compute final calibrated match outcome probability.

        p_final = np.clip(p_market + predicted_residual, min_prob, max_prob)

        Under team order inversion (Delta_X -> -Delta_X, p_market -> 1 - p_market):
            p_final(A) + p_final(B) == 1.0 strictly holds.

        Parameters
        ----------
        X : np.ndarray
            Differential features of shape (n_samples, n_features).
        p_market : np.ndarray or float
            Market implied probability (or probabilities) for Team A in (0, 1).
        return_2d : bool, default=False
            If True, returns a 2D array of shape (n_samples, 2) where column 0
            is P(Team B wins) and column 1 is P(Team A wins).
            If False, returns a 1D array of shape (n_samples,) representing P(Team A wins).

        Returns
        -------
        np.ndarray
            Calibrated probability array.
        """
        residual = self.predict_residual(X)
        n_samples = len(residual)

        pm_arr = np.asarray(p_market, dtype=float)
        if pm_arr.ndim == 0:
            p_m = np.full(n_samples, float(pm_arr), dtype=float)
        else:
            p_m = _ensure_1d_float(pm_arr, "p_market")
            if len(p_m) != n_samples:
                raise ValueError(
                    f"Length mismatch: X has {n_samples} samples, p_market has {len(p_m)}"
                )

        if not np.all((p_m > 0.0) & (p_m < 1.0)):
            raise ValueError("All p_market values must lie strictly within (0, 1)")

        p_final = np.clip(p_m + residual, self.min_prob, self.max_prob)

        if return_2d:
            return np.column_stack([1.0 - p_final, p_final])
        return p_final

    def predict_proba_2d(
        self,
        X: np.ndarray,
        p_market: np.ndarray | float,
    ) -> np.ndarray:
        """Return 2D probabilities [P(Team B wins), P(Team A wins)] summing to 1.0."""
        return self.predict_proba(X, p_market, return_2d=True)

    def predict(
        self,
        X: np.ndarray,
        p_market: np.ndarray | float,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary win prediction (1 if p_final >= threshold else 0)."""
        probs = self.predict_proba(X, p_market, return_2d=False)
        return (probs >= threshold).astype(int)


class ResidualEdgeDetector:
    """Actionable value signal generator comparing predicted residual against transaction friction.

    Evaluates expected edge after deducting bookmaker overround (market margin) and turnover tax:
        expected_edge = predicted_residual - (market_margin / 2.0) - turnover_tax

    Parameters
    ----------
    min_edge : float, default=0.02
        Minimum expected edge threshold required to trigger an actionable signal (e.g. 2%).
    turnover_tax : float, default=0.0
        Turnover tax or exchange commission rate (e.g. 0.05 or 0.12).
    default_market_margin : float, default=0.05
        Default bookmaker overround used if not supplied explicitly (e.g. 5% vig).
    """

    def __init__(
        self,
        min_edge: float = 0.02,
        turnover_tax: float = 0.0,
        default_market_margin: float = 0.05,
    ) -> None:
        if min_edge < 0.0:
            raise ValueError(f"min_edge must be non-negative, got {min_edge}")
        if turnover_tax < 0.0:
            raise ValueError(f"turnover_tax must be non-negative, got {turnover_tax}")
        if default_market_margin < 0.0:
            raise ValueError(f"default_market_margin must be non-negative, got {default_market_margin}")

        self.min_edge = min_edge
        self.turnover_tax = turnover_tax
        self.default_market_margin = default_market_margin

    @staticmethod
    def margin_from_odds(
        odds_a: np.ndarray | float,
        odds_b: np.ndarray | float,
    ) -> np.ndarray | float:
        """Calculate market overround margin from decimal odds: (1/O_A + 1/O_B) - 1.0.

        Parameters
        ----------
        odds_a : np.ndarray or float
            Decimal odds for Team A (> 1.0).
        odds_b : np.ndarray or float
            Decimal odds for Team B (> 1.0).

        Returns
        -------
        np.ndarray or float
            Implied bookmaker margin.
        """
        o_a = np.asarray(odds_a, dtype=float)
        o_b = np.asarray(odds_b, dtype=float)
        if np.any(o_a <= 1.0) or np.any(o_b <= 1.0):
            raise ValueError("Decimal odds must be strictly greater than 1.0")

        margin = (1.0 / o_a) + (1.0 / o_b) - 1.0
        if margin.ndim == 0:
            return float(margin)
        return margin

    def compute_expected_edge(
        self,
        predicted_residual: np.ndarray | float,
        market_margin: np.ndarray | float | None = None,
        turnover_tax: float | None = None,
    ) -> np.ndarray | float:
        """Compute net expected edge = predicted_residual - (market_margin / 2.0) - turnover_tax.

        Parameters
        ----------
        predicted_residual : np.ndarray or float
            Predicted residual from MarketResidualModel.
        market_margin : np.ndarray or float, optional
            Bookmaker overround. If None, uses default_market_margin.
        turnover_tax : float, optional
            Turnover tax rate. If None, uses self.turnover_tax.

        Returns
        -------
        np.ndarray or float
            Net expected edge after transaction friction.
        """
        res = np.asarray(predicted_residual, dtype=float)
        m = (
            np.asarray(self.default_market_margin, dtype=float)
            if market_margin is None
            else np.asarray(market_margin, dtype=float)
        )
        tax = (
            float(self.turnover_tax)
            if turnover_tax is None
            else float(turnover_tax)
        )

        edge = res - (m / 2.0) - tax
        if edge.ndim == 0:
            return float(edge)
        return edge

    def detect_edge(
        self,
        predicted_residual: float,
        market_margin: float | None = None,
        odds_a: float | None = None,
        odds_b: float | None = None,
        turnover_tax: float | None = None,
    ) -> EdgeSignal:
        """Detect actionable edge for a single match prediction.

        Evaluates Team A (residual = r) and Team B (residual = -r).

        Parameters
        ----------
        predicted_residual : float
            Predicted residual on Team A.
        market_margin : float, optional
            Overround margin. If None and odds are provided, computed from odds.
        odds_a : float, optional
            Decimal odds for Team A.
        odds_b : float, optional
            Decimal odds for Team B.
        turnover_tax : float, optional
            Custom turnover tax. Defaults to self.turnover_tax.

        Returns
        -------
        EdgeSignal
            Evaluated edge signal dataclass.
        """
        r = float(predicted_residual)
        tax = float(self.turnover_tax if turnover_tax is None else turnover_tax)

        if market_margin is not None:
            m = float(market_margin)
        elif odds_a is not None and odds_b is not None:
            m = float(self.margin_from_odds(odds_a, odds_b))
        else:
            m = float(self.default_market_margin)

        edge_a = r - (m / 2.0) - tax
        edge_b = -r - (m / 2.0) - tax

        is_a_actionable = edge_a >= self.min_edge
        is_b_actionable = edge_b >= self.min_edge

        if is_a_actionable and edge_a > edge_b:
            rec_side = "team_a"
            rec_edge = edge_a
            actionable = True
        elif is_b_actionable:
            rec_side = "team_b"
            rec_edge = edge_b
            actionable = True
        else:
            rec_side = None
            rec_edge = 0.0
            actionable = False

        return EdgeSignal(
            predicted_residual=r,
            market_margin=m,
            expected_edge_a=edge_a,
            expected_edge_b=edge_b,
            is_actionable=actionable,
            recommended_side=rec_side,
            recommended_edge=rec_edge,
            turnover_tax=tax,
        )

    def detect_edges_batch(
        self,
        predicted_residuals: np.ndarray,
        market_margins: np.ndarray | float | None = None,
        turnover_tax: float | None = None,
    ) -> list[EdgeSignal]:
        """Detect actionable edges across a batch of match predictions."""
        residuals = _ensure_1d_float(predicted_residuals, "predicted_residuals")
        n = len(residuals)

        if market_margins is None:
            margins = np.full(n, self.default_market_margin, dtype=float)
        else:
            margins_arr = np.asarray(market_margins, dtype=float)
            if margins_arr.ndim == 0:
                margins = np.full(n, float(margins_arr), dtype=float)
            else:
                margins = _ensure_1d_float(margins_arr, "market_margins")
                if len(margins) != n:
                    raise ValueError(f"Length mismatch: {n} residuals vs {len(margins)} margins")

        return [
            self.detect_edge(
                predicted_residual=residuals[i],
                market_margin=margins[i],
                turnover_tax=turnover_tax,
            )
            for i in range(n)
        ]
