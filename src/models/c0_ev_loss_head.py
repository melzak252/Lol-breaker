"""Fixed, side-symmetric EV-loss heads for historical development only.

Return-MSE through sigmoid is nonconvex. A successful deterministic single
L-BFGS-B run is a local solution, never a global-optimum certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit

from src.models.c0_betting_price_head import (
    _FEATURES, _NATIVE_IDENTITY, _TRANSFORM, _contract, _is_numeric,
    _numeric_column, _numeric_vector, _regularization, price_features,
)
from src.models.ev_approach_models import C0_EPSILON, _fit_boxed

_SCHEMA = "c0_historical_ev_loss_head_v1"
_TAX = .12
_NET = .88
_GRID = (.01, .1, 1.)
_OBJECTIVES = {
    "likelihood": "mean_unclipped_logit_nll_plus_lambda_over_2_centered_squared_l2",
    "return_mse": "mean_raw_two_sided_return_mse_plus_lambda_over_2_centered_squared_l2",
}


def _objective_name(objective: Any) -> str:
    if not isinstance(objective, str) or objective not in _OBJECTIVES:
        raise ValueError("Unknown EV-loss objective")
    return objective


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    try:
        original = np.asarray(value, dtype=object)
        if original.ndim != ndim or not all(_is_numeric(v) for v in original.flat):
            raise ValueError(f"Invalid numeric {name} shape or values")
        result = np.asarray(value, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric {name}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {name}")
    return result


def _binary(y: np.ndarray) -> None:
    if not np.isin(y, (0., 1.)).all():
        raise ValueError("y must contain binary winner labels")


def _weights(odds_a: np.ndarray, odds_b: np.ndarray) -> np.ndarray:
    if (odds_a <= 1).any() or (odds_b <= 1).any():
        raise ValueError("Decimal prices must be finite and greater than 1")
    with np.errstate(over="ignore", invalid="ignore"):
        weight = .5 * _NET ** 2 * (odds_a ** 2 + odds_b ** 2)
    if not np.isfinite(weight).all():
        raise ValueError("Nonfinite derived return-MSE weight")
    return weight


def _frame_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _numeric_column(frame, "y")
    _binary(y)
    oa, ob = _numeric_column(frame, "odds_a"), _numeric_column(frame, "odds_b")
    _weights(oa, ob)
    return y, oa, ob


def return_mse(frame: pd.DataFrame, probability: Any) -> np.ndarray:
    """Per-row two-sided net-return squared error; weights are not normalized."""
    y, oa, ob = _frame_targets(frame)
    p = _array(probability, "probability", 1)
    if p.shape != y.shape or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probability shape or domain")
    result = _weights(oa, ob) * (p - y) ** 2
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite return-MSE")
    return result


def ev_loss_objective(theta: Any, x: Any, y: Any, odds_a: Any, odds_b: Any,
                      center: Any, regularization: Any, objective: str) -> tuple[float, np.ndarray]:
    """Raw training loss and analytic gradient, with centered lambda/2 L2."""
    objective = _objective_name(objective)
    regularization = _regularization(regularization)
    theta, center = _array(theta, "theta", 1), _array(center, "center", 1)
    x, y = _array(x, "x", 2), _array(y, "y", 1)
    oa, ob = _array(odds_a, "odds_a", 1), _array(odds_b, "odds_b", 1)
    if (not len(y) or not len(theta) or x.shape != (len(y), len(theta))
            or center.shape != theta.shape or oa.shape != y.shape or ob.shape != y.shape):
        raise ValueError("Invalid EV-loss objective shape or empty population")
    _binary(y)
    weight = _weights(oa, ob)  # Also validate the common price contract for NLL.
    with np.errstate(over="ignore", invalid="ignore"):
        eta = x @ theta
        if not np.isfinite(eta).all():
            raise ValueError("Nonfinite training logit")
        p = expit(eta)
        if objective == "likelihood":
            data_loss = np.logaddexp(0., eta) - y * eta
            residual = p - y
        else:
            data_loss = weight * (p - y) ** 2
            residual = 2 * weight * (p - y) * p * (1 - p)
        delta = theta - center
        value = float(np.mean(data_loss) + .5 * regularization * np.sum(delta ** 2))
        gradient = x.T @ residual / len(y) + regularization * delta
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite EV-loss objective or gradient")
    return value, gradient


@dataclass(frozen=True)
class C0EVLossHead:
    recipe_id: str
    objective: str
    coefficients: tuple[float, ...]
    regularization: float
    optimizer: dict[str, Any]
    bindings: dict[str, Any]

    def __post_init__(self) -> None:
        center, bounds = _contract(self.recipe_id)
        _objective_name(self.objective)
        theta = _numeric_vector(self.coefficients, len(center), "coefficients")
        regularization = _regularization(self.regularization)
        if any((lo is not None and t < lo) or (hi is not None and t > hi)
               for t, (lo, hi) in zip(theta, bounds)):
            raise ValueError("Coefficient bounds violated")
        m = self.optimizer
        if (not isinstance(m, dict) or m.get("success") is not True
                or m.get("optimizer") != "L-BFGS-B" or not isinstance(self.bindings, dict)):
            raise ValueError("Invalid optimizer/head lineage")
        if _numeric_vector(m.get("parameters"), len(theta), "optimizer parameters") != theta:
            raise ValueError("Optimizer parameters differ from coefficients")
        for key in ("objective_value", "data_loss", "raw_optimizer_nll", "raw_optimizer_return_mse",
                    "projected_gradient_inf_norm"):
            if not _is_numeric(m.get(key)) or not np.isfinite(m[key]) or m[key] < 0:
                raise ValueError(f"Invalid optimizer {key}")
        for key in ("status", "iterations", "function_evaluations"):
            if (isinstance(m.get(key), (bool, np.bool_))
                    or not isinstance(m.get(key), (int, np.integer)) or m[key] < 0):
                raise ValueError(f"Invalid optimizer {key}")
        if m["status"] != 0 or not isinstance(m.get("message"), str):
            raise ValueError("Invalid optimizer convergence")
        expected_geometry = "nonconvex" if self.objective == "return_mse" else "convex"
        if (m.get("objective") != self.objective or m.get("loss_geometry") != expected_geometry
                or m.get("optimality_claim") != "local_solver_convergence_only"
                or m.get("initialization") != center
                or m.get("options") != {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50}):
            raise ValueError("Invalid optimizer objective contract")
        _numeric_vector(m["initialization"], len(center), "optimizer initialization")
        if any(not _is_numeric(v) for v in m["options"].values()):
            raise ValueError("Invalid numeric optimizer options")
        raw_key = "raw_optimizer_nll" if self.objective == "likelihood" else "raw_optimizer_return_mse"
        if m["data_loss"] != m[raw_key]:
            raise ValueError("Optimizer data loss differs from objective")
        with np.errstate(over="ignore", invalid="ignore"):
            expected = m["data_loss"] + .5 * regularization * np.sum((np.asarray(theta) - center) ** 2)
        if not np.isfinite(expected) or not np.isclose(m["objective_value"], expected, rtol=1e-12, atol=1e-12):
            raise ValueError("Optimizer objective differs from declared penalty")

    def predict_details(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        _weights(_numeric_column(frame, "odds_a"), _numeric_column(frame, "odds_b"))
        with np.errstate(over="ignore", invalid="ignore"):
            eta = price_features(frame, self.recipe_id) @ np.asarray(self.coefficients)
        if not np.isfinite(eta).all():
            raise ValueError("Nonfinite EV-loss head logit")
        raw = expit(eta)
        p = np.clip(raw, C0_EPSILON, 1 - C0_EPSILON)
        input_clipped = np.zeros(len(frame), dtype=bool)
        for column in (("q", "p_C0") if self.recipe_id == "H-price" else ("q",)):
            values = frame[column].to_numpy(dtype=float)
            input_clipped |= (values < C0_EPSILON) | (values > 1 - C0_EPSILON)
        return {"logit": eta, "raw_probability": raw, "probability": p,
                "clipped": raw != p, "input_clipped": input_clipped}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_details(frame)["probability"]

    def to_dict(self) -> dict[str, Any]:
        center, bounds = _contract(self.recipe_id)
        return {"schema": _SCHEMA, "recipe_id": self.recipe_id, "objective": self.objective,
                "objective_formula": _OBJECTIVES[self.objective], "tax_rate": _TAX,
                "net_multiplier": _NET, "coefficients": list(self.coefficients),
                "regularization": float(self.regularization), "epsilon": C0_EPSILON,
                "transform": _TRANSFORM, "features": list(_FEATURES[self.recipe_id]),
                "knot_probability": .8, "native_identity": _NATIVE_IDENTITY, "mode": "full",
                "center": center, "bounds": [list(b) for b in bounds],
                "scope": "historical_development_only", "optimizer": deepcopy(self.optimizer),
                "bindings": deepcopy(self.bindings)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "C0EVLossHead":
        if not isinstance(value, Mapping):
            raise ValueError("Invalid EV-loss head artifact")
        recipe, objective = value.get("recipe_id"), _objective_name(value.get("objective"))
        center, bounds = _contract(recipe)
        expected = {"schema": _SCHEMA, "objective_formula": _OBJECTIVES[objective],
                    "tax_rate": _TAX, "net_multiplier": _NET, "epsilon": C0_EPSILON,
                    "transform": _TRANSFORM, "features": list(_FEATURES[recipe]),
                    "knot_probability": .8, "native_identity": _NATIVE_IDENTITY, "mode": "full",
                    "center": center, "bounds": [list(b) for b in bounds],
                    "scope": "historical_development_only"}
        keys = set(expected) | {"recipe_id", "objective", "coefficients", "regularization", "optimizer", "bindings"}
        if set(value) != keys or any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("Wrong historical EV-loss head contract")
        for key in ("tax_rate", "net_multiplier", "epsilon", "knot_probability"):
            if not _is_numeric(value[key]):
                raise ValueError("Invalid numeric transform contract")
        _numeric_vector(value["center"], len(center), "center")
        for bound in value["bounds"]:
            if any(v is not None and not _is_numeric(v) for v in bound):
                raise ValueError("Invalid numeric bounds")
        return cls(recipe, objective, _numeric_vector(value["coefficients"], len(center), "coefficients"),
                   _regularization(value["regularization"]), deepcopy(value["optimizer"]), deepcopy(value["bindings"]))


def fit_ev_loss_head(recipe: str, objective: str, train: pd.DataFrame, *, regularization: float) -> C0EVLossHead:
    objective, regularization = _objective_name(objective), _regularization(regularization)
    center_values, bounds = _contract(recipe)
    x = price_features(train, recipe)
    y, oa, ob = _frame_targets(train)
    if not len(y) or len(np.unique(y)) != 2 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("INSUFFICIENT_FIT_SUPPORT")
    bindings = train.attrs.get("c0_bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("Invalid model source bindings")
    center = np.asarray(center_values)
    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return ev_loss_objective(theta, x, y, oa, ob, center, regularization, objective)
    theta, metadata = _fit_boxed(lambda t: loss(t)[0], lambda t: loss(t)[1], center, bounds)
    eta, p = x @ theta, expit(x @ theta)
    nll = float(np.mean(np.logaddexp(0., eta) - y * eta))
    mse = float(np.mean(_weights(oa, ob) * (p - y) ** 2))
    _, gradient = loss(theta)
    projected = gradient.copy()
    for i, (lower, upper) in enumerate(bounds):
        if lower is not None and theta[i] <= lower:
            projected[i] = min(projected[i], 0.)
        if upper is not None and theta[i] >= upper:
            projected[i] = max(projected[i], 0.)
    metadata.update({"objective": objective, "data_loss": nll if objective == "likelihood" else mse,
                     "raw_optimizer_nll": nll, "raw_optimizer_return_mse": mse,
                     "loss_geometry": "nonconvex" if objective == "return_mse" else "convex",
                     "optimality_claim": "local_solver_convergence_only", "initialization": center_values,
                     "projected_gradient_inf_norm": float(np.max(np.abs(projected))),
                     "options": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50}})
    return C0EVLossHead(recipe, objective, tuple(float(t) for t in theta), regularization,
                        metadata, deepcopy(bindings))


def select_ev_loss_head(candidates: Any, cal: pd.DataFrame) -> tuple[C0EVLossHead, dict[str, Any]]:
    if (not isinstance(candidates, (list, tuple)) or len(candidates) != len(_GRID)
            or not all(isinstance(h, C0EVLossHead) for h in candidates)
            or {h.regularization for h in candidates} != set(_GRID)
            or len({h.recipe_id for h in candidates}) != 1
            or len({h.objective for h in candidates}) != 1):
        raise ValueError("Invalid matched EV-loss candidate grid")
    y, _, _ = _frame_targets(cal)
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("INSUFFICIENT_FIT_SUPPORT")
    scores = [float(np.mean(return_mse(cal, h.predict(cal)))) for h in candidates]
    best = min(scores)
    eligible = [i for i, score in enumerate(scores) if score <= best + 1e-12]
    selected_index = max(eligible, key=lambda i: candidates[i].regularization)
    head = candidates[selected_index]
    return head, {"criterion": "return_mse", "lambda": head.regularization,
                  "return_mse": scores[selected_index], "tie_tolerance": 1e-12,
                  "tie_break": "largest_lambda", "candidates": [
                      {"lambda": h.regularization, "return_mse": score}
                      for h, score in zip(candidates, scores)]}
