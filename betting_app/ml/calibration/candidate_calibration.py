"""Calibration algorithms and metrics for match outcome candidate models (EXP-040).

This module implements:
1. TemperatureScalingCalibrator: Post-hoc logit temperature scaling optimizing scalar
   temperature T > 0 via binary cross-entropy (Negative Log-Likelihood) while preserving
   strict side-symmetry P(A) + P(B) == 1.0.
2. BetaCalibrator: Two-parameter and three-parameter Beta calibration for binary
   classifiers (Kull et al., 2017), preventing tail distortion and excessive shrinkage
   while maintaining monotonicity.
3. UncertaintyGatedCalibrator: Bayesian uncertainty shrinkage combining model probabilities
   with team rating variance and external market consensus, shrinking towards market odds
   or a 0.5 base rate when uncertainty or discrepancy is elevated.
4. expected_calibration_error: Equal-width expected calibration error (ECE).
5. brier_score_decomposition: Murphy (1973) decomposition of the Brier score into
   reliability, resolution, and uncertainty components.
"""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

EPSILON: float = 1e-12


def _ensure_1d_float(arr: Any, name: str = "array") -> np.ndarray:
    """Convert input to a contiguous 1D float array with sanity checks."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim > 1:
        if a.shape[1] == 1:
            return a.ravel()
        if a.ndim == 2 and a.shape[1] == 2:
            # Multi-column binary: return difference column 1 - column 0 (logit for class 1)
            return (a[:, 1] - a[:, 0]).ravel()
        raise ValueError(f"{name} must be 1D or 2D with 1 or 2 columns, got shape {a.shape}")
    return a.ravel()


class TemperatureScalingCalibrator:
    """Temperature scaling calibrator for binary classification logits.

    Fits a scalar temperature parameter T > 0 on out-of-fold validation logits
    by minimizing the Negative Log-Likelihood (binary cross-entropy):

        P(y = 1 | z) = sigma(z / T) = 1 / (1 + exp(-z / T))

    Properties:
    - When T > 1.0, overconfident probabilities are flattened towards 0.5.
    - When T < 1.0, underconfident probabilities are sharpened away from 0.5.
    - Preserves strict binary side-symmetry:
        P(team_a wins) + P(team_b wins) == sigma(z / T) + sigma(-z / T) == 1.0.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 1e-3,
        max_temperature: float = 50.0,
        max_iter: int = 1000,
        tol: float = 1e-7,
    ) -> None:
        if initial_temperature <= 0:
            raise ValueError(f"initial_temperature must be > 0, got {initial_temperature}")
        self.initial_temperature = float(initial_temperature)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.max_iter = max_iter
        self.tol = tol
        self.temperature_: float = float(initial_temperature)

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> TemperatureScalingCalibrator:
        """Fit optimal scalar temperature T by minimizing NLL on validation logits.

        Parameters
        ----------
        logits : np.ndarray
            1D array of uncalibrated logits z = log(p / (1 - p)), or 2D array of logits.
        y_true : np.ndarray
            1D array of binary ground-truth labels in {0, 1}.

        Returns
        -------
        self : TemperatureScalingCalibrator
        """
        z = _ensure_1d_float(logits, "logits")
        y = _ensure_1d_float(y_true, "y_true")

        if len(z) == 0:
            raise ValueError("Cannot fit TemperatureScalingCalibrator on empty logits.")
        if len(z) != len(y):
            raise ValueError(f"Mismatch between logits length ({len(z)}) and y_true length ({len(y)}).")
        if not np.all(np.isfinite(z)):
            raise ValueError("Logits contain NaN or infinite values.")

        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError("y_true must contain binary labels in {0, 1}.")
        if len(unique_labels) < 2:
            # Only one class present in calibration fold; fallback to initial temperature
            self.temperature_ = self.initial_temperature
            return self

        # Objective function: Negative Log-Likelihood with stable logaddexp
        def nll_objective(t_arr: np.ndarray) -> tuple[float, np.ndarray]:
            t = float(t_arr[0])
            scaled_z = z / t
            # - [y * log(sigma(u)) + (1-y) * log(1-sigma(u))]
            # = y * logaddexp(0, -u) + (1-y) * logaddexp(0, u)
            loss = float(np.mean(y * np.logaddexp(0.0, -scaled_z) + (1.0 - y) * np.logaddexp(0.0, scaled_z)))

            # Analytical gradient w.r.t T:
            # d/dT [ loss ] = (1 / N) * sum( (sigma(u) - y) * (-z / T^2) )
            p_pred = expit(scaled_z)
            grad_t = float(np.mean((p_pred - y) * (-z / (t * t))))
            return loss, np.array([grad_t], dtype=float)

        res = minimize(
            nll_objective,
            x0=np.array([self.initial_temperature], dtype=float),
            jac=True,
            bounds=[(self.min_temperature, self.max_temperature)],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.temperature_ = float(res.x[0])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate logits to probabilities P(y = 1) in [0, 1].

        If input is 1D with shape (N,), returns 1D array of probabilities (N,).
        If input is 2D with shape (N, 1), returns 2D array (N, 1).
        If input is 2D with shape (N, 2), returns 2D array (N, 2) where column 0
        is P(y = 0) and column 1 is P(y = 1), strictly summing to 1.0.

        Parameters
        ----------
        logits : np.ndarray
            Logits to calibrate.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        arr = np.asarray(logits, dtype=float)
        t = self.temperature_

        if arr.ndim == 2 and arr.shape[1] == 2:
            # Paired team logits [z_0, z_1] (e.g. [z_b, z_a])
            z_diff = arr[:, 1] - arr[:, 0]
            p1 = expit(z_diff / t)
            p0 = expit(-z_diff / t)
            # Exact symmetry: p0 + p1 = 1.0
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

        z = _ensure_1d_float(arr, "logits")
        p = expit(z / t)
        p = np.clip(p, 0.0, 1.0)

        if arr.ndim == 2 and arr.shape[1] == 1:
            return p.reshape(-1, 1)
        return p

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return 2D probabilities [P(y = 0), P(y = 1)]."""
        arr = np.asarray(logits, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return self.transform(arr)
        p = self.transform(arr).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions thresholded at 0.5."""
        p = self.transform(logits)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        return (p.ravel() >= threshold).astype(int)


class BetaCalibrator:
    """Beta calibration for binary classifiers (Kull et al., 2017).

    Maps uncalibrated predicted probabilities s in (0, 1) to calibrated probabilities
    using a generalized linear model derived from Beta score distributions:

        logit(p_cal) = a * ln(s) - b * ln(1 - s) + c

    Parameters:
    - "ab"  : Two-parameter version with a >= 0, b >= 0, and c = 0.0.
              Monotonic, includes the identity function at a = 1, b = 1.
    - "abm" : Three-parameter version fitting a >= 0, b >= 0, and intercept c in R.
    - "am"  : Standard logistic calibration (Platt scaling) with a = b >= 0 and intercept c.
    - "a"   : One-parameter scaling with a = b >= 0 and c = 0.0.

    When `symmetric=True`, predictions are guaranteed to satisfy side-symmetry:
        P_cal(s) + P_cal(1 - s) == 1.0 for all s in [0, 1].
    """

    def __init__(
        self,
        parameters: Literal["ab", "abm", "am", "a"] = "ab",
        symmetric: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
    ) -> None:
        valid_params = {"ab", "abm", "am", "a"}
        if parameters not in valid_params:
            raise ValueError(f"parameters must be one of {valid_params}, got '{parameters}'")
        self.parameters = parameters
        self.symmetric = symmetric
        self.max_iter = max_iter
        self.tol = tol

        self.a_: float = 1.0
        self.b_: float = 1.0
        self.c_: float = 0.0

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> BetaCalibrator:
        """Fit Beta calibration parameters on validation probabilities and true outcomes.

        Parameters
        ----------
        probs : np.ndarray
            Predicted probabilities s in [0, 1] (or 1D array). If values are outside
            [0, 1], they are treated as logits and converted via sigmoid.
        y_true : np.ndarray
            Binary ground-truth labels in {0, 1}.

        Returns
        -------
        self : BetaCalibrator
        """
        raw_s = _ensure_1d_float(probs, "probs")
        y = _ensure_1d_float(y_true, "y_true")

        if len(raw_s) == 0:
            raise ValueError("Cannot fit BetaCalibrator on empty inputs.")
        if len(raw_s) != len(y):
            raise ValueError(f"Mismatch between probs length ({len(raw_s)}) and y_true length ({len(y)}).")
        if not np.all(np.isfinite(raw_s)):
            raise ValueError("Input probabilities contain NaN or infinite values.")

        # Convert logits to probabilities if input values fall outside [0, 1]
        if np.any(raw_s < 0.0) or np.any(raw_s > 1.0):
            s = expit(raw_s)
        else:
            s = raw_s

        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError("y_true must contain binary labels in {0, 1}.")
        if len(unique_labels) < 2:
            self.a_ = 1.0
            self.b_ = 1.0
            self.c_ = 0.0
            return self

        # Symmetrize training sample pairs if requested
        if self.symmetric:
            s_fit = np.concatenate([s, 1.0 - s])
            y_fit = np.concatenate([y, 1.0 - y])
        else:
            s_fit = s
            y_fit = y

        # Numerically safe clipping
        s_safe = np.clip(s_fit, EPSILON, 1.0 - EPSILON)
        ln_s = np.log(s_safe)
        ln_1_minus_s = np.log(1.0 - s_safe)

        def nll_objective(p_vec: np.ndarray) -> tuple[float, np.ndarray]:
            if self.parameters == "ab":
                a, b = float(p_vec[0]), float(p_vec[1])
                c = 0.0
            elif self.parameters == "abm":
                a, b, c = float(p_vec[0]), float(p_vec[1]), float(p_vec[2])
            elif self.parameters == "am":
                a = b = float(p_vec[0])
                c = float(p_vec[1])
            elif self.parameters == "a":
                a = b = float(p_vec[0])
                c = 0.0
            else:
                raise ValueError(f"Unknown parameters: {self.parameters}")

            z = a * ln_s - b * ln_1_minus_s + c
            loss = float(np.mean(y_fit * np.logaddexp(0.0, -z) + (1.0 - y_fit) * np.logaddexp(0.0, z)))

            # Gradients
            p_pred = expit(z)
            diff = p_pred - y_fit
            d_loss_dz = diff

            if self.parameters == "ab":
                grad_a = float(np.mean(d_loss_dz * ln_s))
                grad_b = float(np.mean(d_loss_dz * (-ln_1_minus_s)))
                return loss, np.array([grad_a, grad_b], dtype=float)
            elif self.parameters == "abm":
                grad_a = float(np.mean(d_loss_dz * ln_s))
                grad_b = float(np.mean(d_loss_dz * (-ln_1_minus_s)))
                grad_c = float(np.mean(d_loss_dz))
                return loss, np.array([grad_a, grad_b, grad_c], dtype=float)
            elif self.parameters == "am":
                grad_a = float(np.mean(d_loss_dz * (ln_s - ln_1_minus_s)))
                grad_c = float(np.mean(d_loss_dz))
                return loss, np.array([grad_a, grad_c], dtype=float)
            else:  # "a"
                grad_a = float(np.mean(d_loss_dz * (ln_s - ln_1_minus_s)))
                return loss, np.array([grad_a], dtype=float)

        # Bounds: non-negativity a >= 1e-4, b >= 1e-4 guarantees monotonicity
        if self.parameters == "ab":
            x0 = np.array([1.0, 1.0], dtype=float)
            bounds = [(1e-4, 50.0), (1e-4, 50.0)]
        elif self.parameters == "abm":
            x0 = np.array([1.0, 1.0, 0.0], dtype=float)
            bounds = [(1e-4, 50.0), (1e-4, 50.0), (-20.0, 20.0)]
        elif self.parameters == "am":
            x0 = np.array([1.0, 0.0], dtype=float)
            bounds = [(1e-4, 50.0), (-20.0, 20.0)]
        else:  # "a"
            x0 = np.array([1.0], dtype=float)
            bounds = [(1e-4, 50.0)]

        res = minimize(
            nll_objective,
            x0=x0,
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        if self.parameters == "ab":
            self.a_, self.b_ = float(res.x[0]), float(res.x[1])
            self.c_ = 0.0
        elif self.parameters == "abm":
            self.a_, self.b_, self.c_ = float(res.x[0]), float(res.x[1]), float(res.x[2])
        elif self.parameters == "am":
            self.a_ = self.b_ = float(res.x[0])
            self.c_ = float(res.x[1])
        else:
            self.a_ = self.b_ = float(res.x[0])
            self.c_ = 0.0

        return self

    def _raw_predict(self, s: np.ndarray) -> np.ndarray:
        """Apply raw Beta calibration mapping."""
        s_safe = np.clip(s, EPSILON, 1.0 - EPSILON)
        z = self.a_ * np.log(s_safe) - self.b_ * np.log(1.0 - s_safe) + self.c_
        return expit(z)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities to calibrated probabilities in [0, 1].

        If `symmetric=True`, computes the symmetrized probability:
            P_sym(s) = 0.5 * (P_cal(s) + 1.0 - P_cal(1 - s))
        which guarantees exact binary symmetry P(A) + P(B) == 1.0.
        """
        arr = np.asarray(probs, dtype=float)
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            s = expit(arr)
        else:
            s = arr

        s_1d = _ensure_1d_float(s, "probs")
        p_raw = self._raw_predict(s_1d)

        if self.symmetric:
            p_rev = self._raw_predict(1.0 - s_1d)
            p_out = 0.5 * (p_raw + (1.0 - p_rev))
        else:
            p_out = p_raw

        p_out = np.clip(p_out, 0.0, 1.0)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return p_out.reshape(-1, 1)
        return p_out

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Return 2D probabilities [P(y = 0), P(y = 1)]."""
        p = self.transform(probs).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions thresholded at 0.5."""
        p = self.transform(probs).ravel()
        return (p >= threshold).astype(int)


class UncertaintyGatedCalibrator:
    """Uncertainty-gated Bayesian shrinkage calibrator.

    Combines the raw model probability with rating uncertainty (sigma_a, sigma_b)
    and optional external market consensus (e.g. Pinnacle no-vig probability).

    Gating Logic:
    - Discrepancy: absolute difference between model probability and market probability.
      If market probability is absent (None or NaN), market discrepancy is 0.0 and
      shrinkage is governed solely by rating uncertainty towards a 0.5 base rate.
    - If discrepancy > discrepancy_threshold (default 0.20) OR average sigma > sigma_threshold (default 2.5),
      applies Bayesian shrinkage:
          P_final = (1 - shrinkage_weight) * P_model + shrinkage_weight * P_target
      where P_target is P_market when available, else base_rate (0.5).
    - Shrinkage weight increases monotonically with both uncertainty and market discrepancy.
    - Strictly preserves binary side-symmetry:
        P_final(A wins against B) + P_final(B wins against A) == 1.0.
    """

    def __init__(
        self,
        discrepancy_threshold: float = 0.20,
        sigma_threshold: float = 2.5,
        sigma_scale: float = 2.5,
        discrepancy_scale: float = 0.30,
        max_shrinkage: float = 1.0,
        base_rate: float = 0.5,
    ) -> None:
        if discrepancy_threshold < 0:
            raise ValueError(f"discrepancy_threshold must be >= 0, got {discrepancy_threshold}")
        if sigma_threshold < 0:
            raise ValueError(f"sigma_threshold must be >= 0, got {sigma_threshold}")
        if sigma_scale <= 0:
            raise ValueError(f"sigma_scale must be > 0, got {sigma_scale}")
        if discrepancy_scale <= 0:
            raise ValueError(f"discrepancy_scale must be > 0, got {discrepancy_scale}")
        if not (0.0 <= max_shrinkage <= 1.0):
            raise ValueError(f"max_shrinkage must be in [0, 1], got {max_shrinkage}")
        if not (0.0 <= base_rate <= 1.0):
            raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")

        self.discrepancy_threshold = float(discrepancy_threshold)
        self.sigma_threshold = float(sigma_threshold)
        self.sigma_scale = float(sigma_scale)
        self.discrepancy_scale = float(discrepancy_scale)
        self.max_shrinkage = float(max_shrinkage)
        self.base_rate = float(base_rate)

    def fit(self, X: Any = None, y: Any = None) -> UncertaintyGatedCalibrator:
        """Scikit-learn compatible fit (stateless/rule-based, returns self)."""
        return self

    def compute_shrinkage_weight(
        self,
        p_model: float | np.ndarray,
        sigma_a: float | np.ndarray,
        sigma_b: float | np.ndarray,
        p_market: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Compute the scalar or element-wise shrinkage weight in [0, max_shrinkage].

        Parameters
        ----------
        p_model : float or np.ndarray
            Model probability for Team A winning.
        sigma_a : float or np.ndarray
            Rating standard deviation for Team A.
        sigma_b : float or np.ndarray
            Rating standard deviation for Team B.
        p_market : float or np.ndarray, optional
            External market consensus probability for Team A.

        Returns
        -------
        float or np.ndarray
            Shrinkage weight in [0, max_shrinkage].
        """
        pm = np.asarray(p_model, dtype=float)
        sa = np.asarray(sigma_a, dtype=float)
        sb = np.asarray(sigma_b, dtype=float)

        avg_sigma = (sa + sb) / 2.0
        excess_sigma = np.maximum(0.0, avg_sigma - self.sigma_threshold)
        w_sigma = excess_sigma / (excess_sigma + self.sigma_scale)

        if p_market is not None:
            pmkt = np.asarray(p_market, dtype=float)
            # Valid market mask
            valid_mkt = np.isfinite(pmkt)
            discrepancy = np.where(valid_mkt, np.abs(pm - pmkt), 0.0)
        else:
            discrepancy = np.zeros_like(pm)

        excess_disc = np.maximum(0.0, discrepancy - self.discrepancy_threshold)
        w_disc = excess_disc / (excess_disc + self.discrepancy_scale)

        # Independent combination ensuring monotonic increase in both dimensions
        w = 1.0 - (1.0 - w_sigma) * (1.0 - w_disc)
        w = np.clip(w, 0.0, self.max_shrinkage)

        if pm.ndim == 0 and sa.ndim == 0 and sb.ndim == 0:
            return float(w)
        return w

    def calibrate(
        self,
        p_model: float | np.ndarray,
        sigma_a: float | np.ndarray,
        sigma_b: float | np.ndarray,
        p_market: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Apply uncertainty-gated Bayesian shrinkage to model probabilities.

        P_final = (1 - shrinkage_weight) * P_model + shrinkage_weight * P_target

        Parameters
        ----------
        p_model : float or np.ndarray
            Model probability for Team A.
        sigma_a : float or np.ndarray
            Team A rating uncertainty standard deviation.
        sigma_b : float or np.ndarray
            Team B rating uncertainty standard deviation.
        p_market : float or np.ndarray, optional
            Market consensus probability for Team A.

        Returns
        -------
        float or np.ndarray
            Calibrated probability strictly bounded in [0, 1].
        """
        pm = np.asarray(p_model, dtype=float)
        w = self.compute_shrinkage_weight(pm, sigma_a, sigma_b, p_market)

        if p_market is not None:
            pmkt = np.asarray(p_market, dtype=float)
            target = np.where(np.isfinite(pmkt), pmkt, self.base_rate)
        else:
            target = np.full_like(pm, self.base_rate)

        p_final = (1.0 - w) * pm + w * target
        p_final = np.clip(p_final, 0.0, 1.0)

        if pm.ndim == 0:
            return float(p_final)
        return p_final

    def transform(
        self,
        p_model: Any,
        sigma_a: Any = None,
        sigma_b: Any = None,
        p_market: Any = None,
    ) -> np.ndarray:
        """Calibrate input array or unpacked data structure."""
        if sigma_a is None and sigma_b is None:
            # Attempt to unpack from 2D array or dict
            if isinstance(p_model, dict):
                res = self.calibrate(
                    p_model["p_model"],
                    p_model["sigma_a"],
                    p_model["sigma_b"],
                    p_model.get("p_market"),
                )
                return np.atleast_1d(np.asarray(res, dtype=float))
            arr = np.asarray(p_model, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                pm = arr[:, 0]
                sa = arr[:, 1]
                sb = arr[:, 2]
                pmkt = arr[:, 3] if arr.shape[1] >= 4 else None
                res = self.calibrate(pm, sa, sb, pmkt)
                return np.atleast_1d(np.asarray(res, dtype=float))
            raise ValueError(
                "When sigma_a/sigma_b are not provided, p_model must be a 2D array "
                "with columns [p_model, sigma_a, sigma_b, (p_market)] or a dict."
            )

        res = self.calibrate(p_model, sigma_a, sigma_b, p_market)
        return np.atleast_1d(np.asarray(res, dtype=float))

    def predict_proba(
        self,
        p_model: Any,
        sigma_a: Any = None,
        sigma_b: Any = None,
        p_market: Any = None,
    ) -> np.ndarray:
        """Return 2D probabilities [P(y = 0), P(y = 1)]."""
        p = self.transform(p_model, sigma_a, sigma_b, p_market).ravel()
        return np.column_stack([1.0 - p, p])


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute equal-width Expected Calibration Error (ECE).

    ECE measures the discrepancy between predicted probability and true outcome
    frequency by binning predictions into `n_bins` equal intervals in [0, 1]:

        ECE = sum_{k=1}^K (|B_k| / N) * | mean_{i in B_k}(y_i) - mean_{i in B_k}(p_i) |

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels in {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    n_bins : int, default=10
        Number of equal-width bins.

    Returns
    -------
    float
        ECE score in [0, 1]. Returns 0.0 if inputs are empty.
    """
    yt = _ensure_1d_float(y_true, "y_true")
    yp = _ensure_1d_float(y_prob, "y_prob")

    if len(yt) == 0:
        return 0.0
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_prob has {len(yp)}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    yp = np.clip(yp, 0.0, 1.0)
    total = len(yt)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Bin indices from 0 to n_bins - 1
    bin_indices = np.digitize(yp, bin_edges[1:-1])

    ece = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(yt[mask]))
            bin_conf = float(np.mean(yp[mask]))
            ece += (count / total) * abs(bin_acc - bin_conf)

    return float(np.clip(ece, 0.0, 1.0))


def brier_score_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Murphy (1973) partition of the Brier score.

    Brier score decomposes into three interpretable components:
        Brier Score ≈ Reliability - Resolution + Uncertainty

    - Uncertainty: Var(Y) = c * (1 - c), base rate difficulty independent of the forecast.
    - Reliability: sum (|B_k| / N) * (p_bar_k - o_bar_k)^2, calibration error (lower is better, 0 is perfect).
    - Resolution: sum (|B_k| / N) * (o_bar_k - c)^2, ability to discriminate different outcomes (higher is better).

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels in {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities in [0, 1].
    n_bins : int, default=10
        Number of bins used for the partition.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'brier_score', 'reliability', 'resolution', 'uncertainty'.
    """
    yt = _ensure_1d_float(y_true, "y_true")
    yp = _ensure_1d_float(y_prob, "y_prob")

    if len(yt) == 0:
        return {
            "brier_score": float("nan"),
            "brier": float("nan"),
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
        }
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_prob has {len(yp)}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    yp = np.clip(yp, 0.0, 1.0)
    total = len(yt)
    brier = float(np.mean((yp - yt) ** 2))
    base_rate = float(np.mean(yt))
    uncertainty = float(base_rate * (1.0 - base_rate))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(yp, bin_edges[1:-1])

    rel = 0.0
    res = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        count = int(np.sum(mask))
        if count > 0:
            bin_conf = float(np.mean(yp[mask]))
            bin_acc = float(np.mean(yt[mask]))
            rel += count * ((bin_conf - bin_acc) ** 2)
            res += count * ((bin_acc - base_rate) ** 2)

    reliability = float(rel / total)
    resolution = float(res / total)

    return {
        "brier_score": brier,
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }
