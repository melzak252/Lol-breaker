"""C0-anchored symmetric calibration with context-specific main effects."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit


class ContextSlopeCalibration:
    """Fit a symmetric positive slope shared with context-dependent effects."""

    SCHEMA = "c0_context_slope_calibration_v1"
    ANCHOR_IDENTITY = "native12__full__full"
    CONTEXT_NAMES = (
        "low_statistics",
        "novel_roster_support",
        "high_sports_disagreement",
    )
    SLOPE_BOUNDS = (0.05, 20.0)
    ALLOWED_PENALTIES = (1.0, 10.0, 100.0)
    FEASIBILITY_MARGIN = 1e-8

    def __init__(self, kind: str = "pooled", penalty: float = 100.0):
        if kind not in {"pooled", "independent", "global"}:
            raise ValueError("kind must be pooled, independent, or global")
        if isinstance(penalty, (bool, np.bool_)) or not np.isfinite(penalty) or float(penalty) not in self.ALLOWED_PENALTIES:
            raise ValueError("penalty must be one of 1, 10, or 100")
        self.kind = kind
        self.penalty = float(penalty)
        self.parameters: dict[str, float] | None = None
        self.fit_diagnostics: dict[str, Any] | None = None

    @classmethod
    def _inputs(cls, p, contexts, y=None):
        probabilities = np.asarray(p, dtype=float)
        context = np.asarray(contexts, dtype=float)
        if probabilities.ndim != 1 or context.shape != (len(probabilities), 3):
            raise ValueError("probabilities and contexts must be aligned vector and N-by-3 matrix")
        if not len(probabilities) or not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
            raise ValueError("probabilities must be finite and strictly inside (0, 1)")
        if np.isinf(context).any() or not np.isin(context[np.isfinite(context)], [0., 1.]).all():
            raise ValueError("contexts must contain only 0, 1, or NaN")
        target = None
        if y is not None:
            target = np.asarray(y, dtype=float)
            if target.shape != probabilities.shape or not np.isfinite(target).all() or not np.isin(target, [0., 1.]).all():
                raise ValueError("outcomes must be an aligned finite binary vector")
        return probabilities, context, target

    def _design(self, contexts):
        # Missing dimensions are zero in the slope formula, and remain visible
        # through fit diagnostics rather than being reinterpreted as another effect.
        x = np.where(np.isnan(contexts), 0.0, contexts)
        design = np.column_stack((1.0 - x.sum(axis=1), x))
        if self.kind == "global":
            return np.ones((len(x), 1), dtype=float)
        return design

    def _constraints(self):
        if self.kind == "global":
            matrix = np.ones((1, 1), dtype=float)
        else:
            states = np.asarray([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float)
            matrix = np.column_stack((1.0 - states.sum(axis=1), states))
        lower = self.SLOPE_BOUNDS[0] + self.FEASIBILITY_MARGIN
        upper = self.SLOPE_BOUNDS[1] - self.FEASIBILITY_MARGIN
        return LinearConstraint(matrix, lower, upper)


    def fit(self, p, y, contexts):
        probabilities, context, target = self._inputs(p, contexts, y)
        design = self._design(context)
        logits = np.log(probabilities) - np.log1p(-probabilities)
        features = design * logits[:, None]
        count_params = 1 if self.kind == "global" else 4
        if self.kind == "pooled":
            center = np.asarray([1.0, 1.0, 1.0, 1.0])
            # 0.5 * lambda * sum((a_j-a0)^2) + 0.5 * (a0-1)^2
            penalty_matrix = np.zeros((4, 4), dtype=float)
            for j in range(1, 4):
                vector = np.zeros(4)
                vector[0], vector[j] = -1.0, 1.0
                penalty_matrix += self.penalty * np.outer(vector, vector)
            penalty_matrix[0, 0] += 1.0
        elif self.kind == "independent":
            center = np.ones(4, dtype=float)
            penalty_matrix = np.diag([1.0, self.penalty, self.penalty, self.penalty])
        else:
            center = np.ones(1, dtype=float)
            penalty_matrix = np.ones((1, 1), dtype=float)

        def objective(parameters):
            z = features @ parameters
            # Stable negative Bernoulli log-likelihood summed over rows.
            nll = np.logaddexp(0.0, z) - target * z
            delta = parameters - center
            penalty_value = 0.5 * float(delta @ penalty_matrix @ delta)
            return float(np.sum(nll) + penalty_value)

        def gradient(parameters):
            z = features @ parameters
            residual = expit(z) - target
            return features.T @ residual + penalty_matrix @ (parameters - center)
        initial = np.ones(count_params, dtype=float)
        result = minimize(
            objective, initial, jac=gradient, method="SLSQP", constraints=(self._constraints(),),
            bounds=[(
                self.SLOPE_BOUNDS[0] + self.FEASIBILITY_MARGIN,
                self.SLOPE_BOUNDS[1] - self.FEASIBILITY_MARGIN,
            )] * count_params,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )
        if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise RuntimeError(f"Context calibration optimizer failed: {result.message}")
        values = np.asarray(result.x, dtype=float)
        if self.kind == "global":
            slopes = values.copy()
            names = ("a0",)
        else:
            slopes = np.asarray([row @ values for row in self._constraints().A])
            names = ("a0", "a_low_statistics", "a_novel_roster_support", "a_high_sports_disagreement")
        if np.any(slopes < self.SLOPE_BOUNDS[0]) or np.any(slopes > self.SLOPE_BOUNDS[1]):
            raise RuntimeError("Optimizer returned a calibration outside its slope constraints")
        self.parameters = {name: float(value) for name, value in zip(names, values)}
        fitted_q = self._predict(probabilities, context)
        zfit = np.log(fitted_q) - np.log1p(-fitted_q)
        log_likelihood = float(np.sum(target * zfit - np.logaddexp(0.0, zfit)))
        unknown = np.isnan(context).sum(axis=0)
        penalty_value = float(0.5 * (values - center) @ penalty_matrix @ (values - center))
        self.fit_diagnostics = {
            "status": "CONVERGED",
            "n_rows": int(len(probabilities)),
            "objective": float(result.fun),
            "log_likelihood": log_likelihood,
            "penalty_value": penalty_value,
            "optimizer_success": bool(result.success),
            "iterations": int(result.nit),
            "unknown_context_counts": {name: int(count) for name, count in zip(self.CONTEXT_NAMES, unknown)},
        }
        return self

    def _predict(self, probabilities, context):
        if self.parameters is None:
            raise ValueError("calibration model is not fitted")
        if self.kind == "global":
            slopes = np.full(len(probabilities), self.parameters["a0"])
        else:
            vector = np.asarray([self.parameters[name] for name in (
                "a0", "a_low_statistics", "a_novel_roster_support", "a_high_sports_disagreement"
            )])
            slopes = self._design(context) @ vector
        logits = np.log(probabilities) - np.log1p(-probabilities)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            calibrated = expit(slopes * logits)
        if not np.isfinite(calibrated).all() or np.any((calibrated <= 0.0) | (calibrated >= 1.0)):
            raise ValueError("calibration saturated outside the strict probability interior")
        return calibrated

    def predict_proba(self, p, contexts):
        probabilities, context, _ = self._inputs(p, contexts)
        return self._predict(probabilities, context)

    def to_dict(self):
        if self.parameters is None or self.fit_diagnostics is None:
            raise ValueError("cannot serialize an unfitted calibration model")
        return {
            "schema": self.SCHEMA,
            "anchor_identity": self.ANCHOR_IDENTITY,
            "context_names": list(self.CONTEXT_NAMES),
            "kind": self.kind,
            "penalty": self.penalty,
            "parameters": dict(self.parameters),
            "fit_diagnostics": dict(self.fit_diagnostics),
        }

    @classmethod
    def from_dict(cls, value):
        expected = {"schema", "anchor_identity", "context_names", "kind", "penalty", "parameters", "fit_diagnostics"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Malformed context calibration artifact fields")
        if value["schema"] != cls.SCHEMA or value["anchor_identity"] != cls.ANCHOR_IDENTITY:
            raise ValueError("Incompatible context calibration schema or anchor")
        if value["context_names"] != list(cls.CONTEXT_NAMES):
            raise ValueError("Incompatible calibration context names")
        model = cls(value["kind"], value["penalty"])
        parameters = value["parameters"]
        keys = {"a0"} if model.kind == "global" else {
            "a0", "a_low_statistics", "a_novel_roster_support", "a_high_sports_disagreement"
        }
        if not isinstance(parameters, dict) or set(parameters) != keys:
            raise ValueError("Malformed context calibration parameters")
        parsed = np.asarray(list(parameters.values()), dtype=float)
        if not np.isfinite(parsed).all() or np.any((parsed < cls.SLOPE_BOUNDS[0]) | (parsed > cls.SLOPE_BOUNDS[1])):
            raise ValueError("Calibration parameters violate slope bounds")
        if model.kind != "global":
            states = np.asarray([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float)
            params = np.asarray([parameters[k] for k in (
                "a0", "a_low_statistics", "a_novel_roster_support", "a_high_sports_disagreement"
            )])
            slopes = np.column_stack((1.0 - states.sum(axis=1), states)) @ params
            if np.any((slopes < cls.SLOPE_BOUNDS[0]) | (slopes > cls.SLOPE_BOUNDS[1])):
                raise ValueError("Calibration artifact violates one of eight context slope bounds")
        diagnostics = value["fit_diagnostics"]
        required_diagnostics = {
            "status", "n_rows", "objective", "log_likelihood", "penalty_value",
            "optimizer_success", "iterations", "unknown_context_counts",
        }
        if not isinstance(diagnostics, dict) or set(diagnostics) != required_diagnostics:
            raise ValueError("Malformed calibration fit diagnostics")
        if diagnostics["status"] != "CONVERGED" or diagnostics["optimizer_success"] is not True:
            raise ValueError("Calibration artifact does not record a successful fit")
        if isinstance(diagnostics["n_rows"], bool) or not isinstance(diagnostics["n_rows"], int) or diagnostics["n_rows"] <= 0:
            raise ValueError("Invalid calibration fit row count")
        if isinstance(diagnostics["iterations"], bool) or not isinstance(diagnostics["iterations"], int) or diagnostics["iterations"] < 0:
            raise ValueError("Invalid calibration optimizer iteration count")
        if any(not np.isfinite(diagnostics[key]) for key in ("objective", "log_likelihood", "penalty_value")):
            raise ValueError("Nonfinite calibration fit diagnostics")
        counts = diagnostics["unknown_context_counts"]
        if not isinstance(counts, dict) or set(counts) != set(cls.CONTEXT_NAMES) or any(
            isinstance(counts[name], bool) or not isinstance(counts[name], int) or counts[name] < 0
            for name in cls.CONTEXT_NAMES
        ):
            raise ValueError("Malformed unknown-context fit counts")
        model.parameters = {key: float(parameters[key]) for key in keys}
        model.fit_diagnostics = dict(diagnostics)
        return model
