"""Bounded, A0-anchored correction for sports-series probabilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class AnchoredSportsCorrection:
    """Fit BO-conditional logit corrections while keeping A0 as the anchor."""

    SCHEMA = "a0_anchored_sports_correction_v1"
    MODEL_ID = "anchored_sports_correction"
    BEST_OF = (1, 3, 5)
    _N_FEATURES = 3
    _DELTA_BOUNDS = (-0.75, 1.0)
    _BETA_BOUNDS = (0.0, 1.0)

    def __init__(self, include_specialist: bool, penalty: float, knot_probability: float = 0.8):
        if not isinstance(include_specialist, (bool, np.bool_)):
            raise ValueError("include_specialist must be boolean")
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        if not np.isfinite(knot_probability) or not 0.5 < knot_probability < 1:
            raise ValueError("knot_probability must be finite and in (0.5, 1)")
        self.include_specialist = bool(include_specialist)
        self.penalty = float(penalty)
        self.knot_probability = float(knot_probability)
        self.coefficients = np.zeros((len(self.BEST_OF), self._N_FEATURES), dtype=float)
        self.fit_diagnostics: dict[str, Any] = {
            "status": "unfitted", "n": 0, "objective": None,
            "success": None, "gradient_norm": None, "iterations": 0,
            "coefficient_bounds": self._bounds_description(),
        }

    @staticmethod
    def _vector(values, name: str, *, allow_none: bool = False):
        if values is None and allow_none:
            return None
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        try:
            array = array.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @classmethod
    def _validate_inputs(cls, p_a0, p_specialist, best_of, y=None, *, require_nonempty=False):
        anchor = cls._vector(p_a0, "p_a0")
        specialist = cls._vector(p_specialist, "p_specialist", allow_none=True)
        bo = cls._vector(best_of, "best_of")
        targets = None if y is None else cls._vector(y, "y")
        vectors = (specialist, bo) + (() if targets is None else (targets,))
        if any(item is not None and item.shape != anchor.shape for item in vectors):
            raise ValueError("p_a0, p_specialist, best_of, and y must have matching shapes")
        if require_nonempty and anchor.size == 0:
            raise ValueError("fit requires at least one observation")
        if np.any((anchor <= 0.0) | (anchor >= 1.0)):
            raise ValueError("p_a0 must be strictly between zero and one")
        if specialist is not None and np.any((specialist <= 0.0) | (specialist >= 1.0)):
            raise ValueError("p_specialist must be strictly between zero and one")
        if np.any(~np.isin(bo, cls.BEST_OF)):
            raise ValueError("best_of values must be one of 1, 3, or 5")
        bo = bo.astype(np.int8, copy=False)
        if targets is not None and np.any((targets != 0.0) & (targets != 1.0)):
            raise ValueError("y must contain only binary 0/1 outcomes")
        return anchor, specialist, bo, targets

    @staticmethod
    def _logit(probability):
        return np.log(probability) - np.log1p(-probability)

    def _feature_matrix(self, p_a0, p_specialist):
        z0 = self._logit(p_a0)
        knot = self._logit(np.asarray([self.knot_probability]))[0]
        lo = np.sign(z0) * np.minimum(np.abs(z0), knot)
        hi = z0 - lo
        if self.include_specialist:
            return np.column_stack((lo, hi, self._logit(p_specialist) - z0))
        return np.column_stack((lo, hi, np.zeros_like(z0)))

    def _bounds_description(self):
        return {
            "delta_lo": list(self._DELTA_BOUNDS),
            "delta_hi": list(self._DELTA_BOUNDS),
            "beta": list(self._BETA_BOUNDS) if self.include_specialist else [0.0, 0.0],
        }

    def fit(self, p_a0, p_specialist, best_of, y):
        anchor, specialist, bo, targets = self._validate_inputs(
            p_a0, p_specialist, best_of, y, require_nonempty=True
        )
        if self.include_specialist and specialist is None:
            raise ValueError("p_specialist is required when include_specialist=True")
        X = self._feature_matrix(anchor, specialist)
        row_ids = np.searchsorted(self.BEST_OF, bo)
        active = np.zeros((anchor.size, len(self.BEST_OF) * self._N_FEATURES), dtype=float)
        active[np.arange(anchor.size), row_ids * self._N_FEATURES] = X[:, 0]
        active[np.arange(anchor.size), row_ids * self._N_FEATURES + 1] = X[:, 1]
        if self.include_specialist:
            active[np.arange(anchor.size), row_ids * self._N_FEATURES + 2] = X[:, 2]
        target = targets.astype(float, copy=False)
        anchor_logits = self._logit(anchor)

        # scipy uses bounds in flattened BO-major [delta_lo, delta_hi, beta] order.
        bounds = []
        for _ in self.BEST_OF:
            bounds.extend((self._DELTA_BOUNDS, self._DELTA_BOUNDS,
                           self._BETA_BOUNDS if self.include_specialist else (0.0, 0.0)))

        def objective_gradient(flat):
            logits = anchor_logits + active @ flat
            loss = np.mean(np.logaddexp(0.0, logits) - target * logits)
            objective = loss + 0.5 * self.penalty * np.dot(flat, flat)
            gradient = active.T @ (expit(logits) - target) / anchor.size + self.penalty * flat
            return float(objective), gradient

        result = minimize(objective_gradient, self.coefficients.ravel().copy(), jac=True,
                          method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 2000, "maxls": 50})
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"anchored correction optimizer did not converge: {result.message}")
        self.coefficients = result.x.reshape(len(self.BEST_OF), self._N_FEATURES).copy()
        gradient = np.asarray(result.jac, dtype=float)
        lower = np.asarray([bound[0] for bound in bounds])
        upper = np.asarray([bound[1] for bound in bounds])
        projected_gradient = gradient.copy()
        projected_gradient[(result.x <= lower + 1e-8) & (gradient > 0)] = 0.0
        projected_gradient[(result.x >= upper - 1e-8) & (gradient < 0)] = 0.0
        self.fit_diagnostics = {
            "status": "converged", "n": int(anchor.size), "objective": float(result.fun),
            "success": bool(result.success), "message": str(result.message),
            "gradient_norm": float(np.linalg.norm(gradient, ord=np.inf)),
            "projected_gradient_norm": float(np.linalg.norm(projected_gradient, ord=np.inf)),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "coefficients": self.coefficients.tolist(),
            "coefficient_bounds": self._bounds_description(),
        }
        return self

    def predict_proba(self, p_a0, p_specialist, best_of) -> np.ndarray:
        anchor, specialist, bo, _ = self._validate_inputs(p_a0, p_specialist, best_of)
        if self.include_specialist and specialist is None:
            raise ValueError("p_specialist is required when include_specialist=True")
        if anchor.size == 0 or not np.any(self.coefficients):
            return anchor.copy()
        X = self._feature_matrix(anchor, specialist)
        row_ids = np.searchsorted(self.BEST_OF, bo)
        logits = self._logit(anchor) + np.einsum("ij,ij->i", X, self.coefficients[row_ids])
        probabilities = expit(logits)
        identity_rows = ~np.any(self.coefficients[row_ids], axis=1)
        probabilities[identity_rows] = anchor[identity_rows]
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
            raise ValueError("correction produced a probability outside the strict interior")
        return probabilities

    def to_dict(self) -> dict[str, Any]:
        diagnostics = dict(self.fit_diagnostics)
        return {
            "schema": self.SCHEMA,
            "model_id": self.MODEL_ID,
            "include_specialist": self.include_specialist,
            "penalty": self.penalty,
            "knot_probability": self.knot_probability,
            "best_of": list(self.BEST_OF),
            "coefficient_names": ["delta_lo", "delta_hi", "beta"],
            "coefficients": self.coefficients.tolist(),
            "fit_diagnostics": diagnostics,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "AnchoredSportsCorrection":
        if not isinstance(state, dict) or state.get("schema") != cls.SCHEMA or state.get("model_id") != cls.MODEL_ID:
            raise ValueError("incompatible correction artifact schema or model identity")
        if state.get("best_of") != list(cls.BEST_OF) or state.get("coefficient_names") != ["delta_lo", "delta_hi", "beta"]:
            raise ValueError("incompatible correction artifact dimensions")
        try:
            model = cls(state["include_specialist"], state["penalty"], state["knot_probability"])
            coefficients = np.asarray(state["coefficients"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid correction artifact parameters") from exc
        if coefficients.shape != (len(cls.BEST_OF), cls._N_FEATURES) or not np.all(np.isfinite(coefficients)):
            raise ValueError("correction artifact has invalid coefficient dimensions or values")
        for row in coefficients:
            if np.any((row[:2] < cls._DELTA_BOUNDS[0]) | (row[:2] > cls._DELTA_BOUNDS[1])):
                raise ValueError("correction artifact delta coefficient outside bounds")
            beta_bounds = cls._BETA_BOUNDS if model.include_specialist else (0.0, 0.0)
            if not beta_bounds[0] <= row[2] <= beta_bounds[1]:
                raise ValueError("correction artifact beta coefficient outside bounds")
        model.coefficients = coefficients.copy()
        diagnostics = state.get("fit_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("correction artifact diagnostics must be an object")
        model.fit_diagnostics = dict(diagnostics)
        return model
