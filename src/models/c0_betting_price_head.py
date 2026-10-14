"""Side-symmetric price-aware heads for exposed historical development only.

These heads neither modify native C0 nor authorize production/forward use.
The caller binds execution prices to its admitted book set and supplies lineage
through ``frame.attrs['c0_bindings']``; a fitted head does not authenticate it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.models.ev_approach_models import (
    C0_EPSILON,
    C0_NATIVE_VERSION,
    _c0_probability,
    _finite_numeric,
    _fit_boxed,
    _labels,
    _regularized_loss,
)

_SCHEMA = "c0_historical_price_head_v1"
_TRANSFORM = "odd_logit_fixed_knot_0.8_plus_signed_log_execution_price_ratio"
_OBJECTIVE = "mean_unclipped_logit_nll_plus_lambda_over_2_centered_squared_l2"
_NATIVE_IDENTITY = f"Causal-C0:{C0_NATIVE_VERSION}"
_KNOT = float(logit(.8))
_FEATURES = {
    "M-price": ("inner_market_logit", "outer_market_logit", "log_odds_b_over_odds_a"),
    "H-price": ("inner_market_logit", "outer_market_logit", "log_odds_b_over_odds_a", "native_logit"),
}


def _contract(recipe: str) -> tuple[list[float], list[tuple[float | None, float | None]]]:
    if not isinstance(recipe, str) or recipe not in _FEATURES:
        raise ValueError("Unknown historical price recipe")
    center = [1., 1., 0.]
    bounds = [(0., None), (0., None), (None, None)]
    if recipe == "H-price":
        center.append(0.)
        bounds.append((0., None))
    return center, bounds


def _is_numeric(value: Any) -> bool:
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)))


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame or not frame[column].map(_is_numeric).all():
        raise ValueError(f"Missing, boolean or nonnumeric {column}")
    return _finite_numeric(frame, column)


def _regularization(value: Any) -> float:
    if not _is_numeric(value) or not np.isfinite(value) or value < 0:
        raise ValueError("Invalid regularization")
    return float(value)


def _numeric_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or not all(_is_numeric(v) and np.isfinite(v) for v in value)):
        raise ValueError(f"Invalid {name}")
    return tuple(float(v) for v in value)


def price_features(frame: pd.DataFrame, recipe: str) -> np.ndarray:
    """Odd features; M-price deliberately does not read native probabilities.

    The signed price feature is computed as a difference of logs to avoid
    overflowing a ratio of otherwise finite decimal prices.
    """
    _contract(recipe)
    m = logit(_c0_probability(frame, "q"))
    odds_a = _numeric_column(frame, "odds_a")
    odds_b = _numeric_column(frame, "odds_b")
    if (odds_a <= 1).any() or (odds_b <= 1).any():
        raise ValueError("Decimal prices must be finite and greater than 1")
    features = [np.sign(m) * np.minimum(np.abs(m), _KNOT),
                np.sign(m) * np.maximum(np.abs(m) - _KNOT, 0.),
                np.log(odds_b) - np.log(odds_a)]
    if recipe == "H-price":
        features.append(logit(_c0_probability(frame, "p_C0")))
    return np.column_stack(features)


@dataclass(frozen=True)
class C0PriceHead:
    recipe_id: str
    coefficients: tuple[float, ...]
    regularization: float
    optimizer: dict[str, Any]
    bindings: dict[str, Any]

    def __post_init__(self) -> None:
        center, bounds = _contract(self.recipe_id)
        theta = _numeric_vector(self.coefficients, len(center), "coefficients")
        for coefficient, (lower, upper) in zip(theta, bounds):
            if ((lower is not None and coefficient < lower)
                    or (upper is not None and coefficient > upper)):
                raise ValueError("Coefficient bounds violated")
        _regularization(self.regularization)
        metadata = self.optimizer
        if (not isinstance(metadata, dict) or metadata.get("success") is not True
                or metadata.get("optimizer") != "L-BFGS-B"
                or not isinstance(self.bindings, dict)):
            raise ValueError("Invalid optimizer/head lineage")
        if _numeric_vector(metadata.get("parameters"), len(theta), "optimizer parameters") != theta:
            raise ValueError("Optimizer parameters differ from coefficients")
        for key in ("objective_value", "raw_optimizer_nll"):
            value = metadata.get(key)
            if not _is_numeric(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid optimizer {key}")
        for key in ("status", "iterations", "function_evaluations"):
            value = metadata.get(key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"Invalid optimizer {key}")
        if metadata["status"] != 0 or not isinstance(metadata.get("message"), str):
            raise ValueError("Invalid optimizer convergence")
        expected = metadata["raw_optimizer_nll"] + .5 * float(self.regularization) * float(
            np.sum((np.asarray(theta) - center) ** 2))
        if not np.isclose(metadata["objective_value"], expected, rtol=1e-12, atol=1e-12):
            raise ValueError("Optimizer objective differs from declared penalty")

    def predict_details(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        eta = price_features(frame, self.recipe_id) @ np.asarray(self.coefficients)
        if not np.isfinite(eta).all():
            raise ValueError("Nonfinite price-head logit")
        raw = expit(eta)
        probability = np.clip(raw, C0_EPSILON, 1 - C0_EPSILON)
        input_clipped = np.zeros(len(frame), dtype=bool)
        for column in (("q", "p_C0") if self.recipe_id == "H-price" else ("q",)):
            values = frame[column].to_numpy(dtype=float)
            input_clipped |= (values < C0_EPSILON) | (values > 1 - C0_EPSILON)
        return {"logit": eta, "raw_probability": raw, "probability": probability,
                "clipped": raw != probability, "input_clipped": input_clipped}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_details(frame)["probability"]

    def to_dict(self) -> dict[str, Any]:
        center, bounds = _contract(self.recipe_id)
        return {"schema": _SCHEMA, "recipe_id": self.recipe_id,
                "coefficients": list(self.coefficients), "regularization": float(self.regularization),
                "epsilon": C0_EPSILON, "transform": _TRANSFORM,
                "features": list(_FEATURES[self.recipe_id]), "knot_probability": .8,
                "native_identity": _NATIVE_IDENTITY, "mode": "full",
                "objective": _OBJECTIVE, "center": center,
                "bounds": [list(bound) for bound in bounds],
                "scope": "historical_development_only",
                "optimizer": deepcopy(self.optimizer), "bindings": deepcopy(self.bindings)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "C0PriceHead":
        if not isinstance(value, Mapping):
            raise ValueError("Invalid price-head artifact")
        recipe = value.get("recipe_id")
        center, bounds = _contract(recipe)
        expected = {"schema": _SCHEMA, "epsilon": C0_EPSILON, "transform": _TRANSFORM,
                    "features": list(_FEATURES[recipe]), "knot_probability": .8,
                    "native_identity": _NATIVE_IDENTITY, "mode": "full",
                    "objective": _OBJECTIVE, "center": center,
                    "bounds": [list(bound) for bound in bounds],
                    "scope": "historical_development_only"}
        keys = set(expected) | {"recipe_id", "coefficients", "regularization", "optimizer", "bindings"}
        if set(value) != keys or any(value.get(key) != item for key, item in expected.items()):
            raise ValueError("Wrong historical price-head contract")
        # Equality alone would accept booleans as numeric penalty centers/bounds.
        if (_numeric_vector(value["center"], len(center), "center") != tuple(center)
                or not _is_numeric(value["epsilon"]) or not _is_numeric(value["knot_probability"])):
            raise ValueError("Invalid numeric transform contract")
        for declared, bound in zip(value["bounds"], bounds):
            if any(v is not None and not _is_numeric(v) for v in declared):
                raise ValueError("Invalid numeric bounds")
        theta = _numeric_vector(value["coefficients"], len(center), "coefficients")
        return cls(recipe, theta, _regularization(value["regularization"]),
                   deepcopy(value["optimizer"]), deepcopy(value["bindings"]))


def fit_price_head(recipe: str, train: pd.DataFrame, *, regularization: float) -> C0PriceHead:
    """Fit uniform stable Bernoulli NLL + lambda/2 centered squared L2.

    Fits start deterministically at the declared center. No calibration refit,
    outcome-selected feature, intercept, optimizer retry or fallback is used.
    """
    regularization = _regularization(regularization)
    center_values, bounds = _contract(recipe)
    x = price_features(train, recipe)
    _numeric_column(train, "y")
    y = _labels(train)
    if not len(y) or len(np.unique(y)) != 2 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("INSUFFICIENT_FIT_SUPPORT")
    bindings = train.attrs.get("c0_bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("Invalid model source bindings")
    center = np.asarray(center_values)

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return _regularized_loss(theta, x, y, center, regularization, "logloss")

    theta, metadata = _fit_boxed(lambda t: loss(t)[0], lambda t: loss(t)[1], center, bounds)
    eta = x @ theta
    metadata["raw_optimizer_nll"] = float(np.mean(np.logaddexp(0., eta) - y * eta))
    return C0PriceHead(recipe, tuple(float(v) for v in theta), regularization,
                       metadata, deepcopy(bindings))
