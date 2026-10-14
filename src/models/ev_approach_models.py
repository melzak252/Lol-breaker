"""Low-capacity, side-symmetric heads for the frozen EV alternatives experiment.

The recipes consume only decision-time fields during prediction.  Outcome and
closing-market fields are targets used by ``fit_head``/``validation_loss`` only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

_EPS = 1e-12
_FOCAL_GAMMA = 2.0
_SMOOTH_EPSILON = 0.1
_CLASS_WEIGHT = 2.0
_KNOT = float(logit(0.8))
_KELLY_FRACTION = 0.25
_KELLY_CAP = 0.025
_ACTION_TIE_TOLERANCE = 1e-12

# Values are intentionally descriptive: callers should use keys as stable IDs
# and fitted metadata for the actual objective/score identity.
APPROACHES: dict[str, str] = {
    "sports_cal": "positive-slope sports-only calibration",
    "market_cal": "positive-slope market-only calibration",
    "market_shape": "market-only odd monotone two-slope market shape",
    "joint_logloss": "regularized joint winner log loss",
    "joint_brier": "regularized joint winner Brier loss",
    "return_mse": "symmetric executable-return squared error",
    "market_residual": "market-offset residual Brier objective",
    "class_weighted": "favorite-oriented weighted log loss",
    "class_weighted_corrected": "weighted log loss with posterior inversion",
    "focal": "gamma-two binary focal loss",
    "focal_corrected": "gamma-two focal loss with posterior recovery",
    "label_smoothing": "epsilon-0.1 uniform label smoothing",
    "label_smoothing_corrected": "smoothed loss with posterior desmoothing",
    "ranking": "within-match pairwise winner-over-loser ranking margin",
    "ranking_calibrated": "pairwise ranker with later held-out positive-slope calibration",
    "close_target": "released closing-probability target regression",
    "clv_target": "released logit closing-line-movement target regression",
    "direct_profit": "deterministic quarter-Kelly action policy fit to realized net profit",
    "kelly_utility": "capped quarter-Kelly realized log-wealth objective",
}

_QUOTE_REQUIRED = frozenset({
    "joint_logloss", "joint_brier", "return_mse",
    "market_residual", "close_target", "clv_target", "direct_profit", "kelly_utility",
})

# Required fitted-recipe columns. The winner label is needed for every fit;
# objective-specific prices and released closing targets are included here.
_PROBABILITY_INPUTS: dict[str, frozenset[str]] = {
    method: (frozenset({"p_s"}) if method == "sports_cal" else
             frozenset({"q"}) if method in {"market_cal", "market_shape"} else
             frozenset({"p_s", "q"}))
    for method in APPROACHES
}
APPROACH_INPUTS: dict[str, frozenset[str]] = {
    method: frozenset(
        {"y"} | set(_PROBABILITY_INPUTS[method]) |
        ({"odds_a", "odds_b"} if method in _QUOTE_REQUIRED else set()) |
        ({"close_q"} if method in {"close_target", "clv_target"} else set())
    )
    for method in APPROACHES
}



def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Missing required model input column {column!r}")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite numeric values")
    return values


def _probability_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _finite_numeric(frame, column)
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{column} must lie in [0, 1]")
    return np.clip(values, _EPS, 1.0 - _EPS)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    y = _finite_numeric(frame, "y")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must contain binary winner labels")
    return y


def _quotes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    odds_a = _finite_numeric(frame, "odds_a")
    odds_b = _finite_numeric(frame, "odds_b")
    if (odds_a <= 1.0).any() or (odds_b <= 1.0).any():
        raise ValueError("Decimal quote odds/prices must be finite and greater than 1.0")
    return odds_a, odds_b


def _features(frame: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the method's odd feature matrix and current market logit."""
    if method == "sports_cal":
        z_s = logit(_probability_column(frame, "p_s"))
        return z_s[:, None], np.zeros(len(frame), dtype=float)
    z_q = logit(_probability_column(frame, "q"))
    if method == "market_cal":
        return z_q[:, None], z_q
    if method == "market_shape":
        magnitude = np.abs(z_q)
        direction = np.sign(z_q)
        inner = direction * np.minimum(magnitude, _KNOT)
        outer = direction * np.maximum(magnitude - _KNOT, 0.0)
        return np.column_stack((inner, outer)), z_q
    z_s = logit(_probability_column(frame, "p_s"))
    if method == "market_residual":
        return z_s[:, None], z_q
    return np.column_stack((z_s, z_q)), z_q


def _row_ids(frame: pd.DataFrame) -> list[str]:
    column = next((name for name in ("row_id", "golgg_match_id") if name in frame), None)
    values = frame[column].tolist() if column else frame.index.tolist()
    return [str(value) for value in values]


def _optimizer_metadata(result: Any, optimizer: str) -> dict[str, Any]:
    if not result.success or not np.isfinite(np.asarray(result.x, dtype=float)).all():
        raise RuntimeError(f"{optimizer} failed: {result.message}")
    return {
        "optimizer": optimizer,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "objective_value": float(result.fun),
        "parameters": np.asarray(result.x, dtype=float).tolist(),
    }


def _fit_boxed(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
) -> tuple[np.ndarray, dict[str, Any]]:
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    details = _optimizer_metadata(result, "L-BFGS-B")
    return np.asarray(result.x, dtype=float), details


def _regularized_loss(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    center: np.ndarray,
    regularization: float,
    kind: str,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    eta = x @ theta
    probability = expit(eta)
    if kind == "logloss":
        data_loss = np.logaddexp(0.0, eta) - y * eta
        residual = probability - y
        if sample_weight is not None:
            data_loss = data_loss * sample_weight
            residual = residual * sample_weight
        loss = float(np.mean(data_loss))
        grad = x.T @ residual / len(y)
    elif kind == "brier":
        squared = (probability - y) ** 2
        residual = 2.0 * (probability - y) * probability * (1.0 - probability)
        if sample_weight is not None:
            squared = squared * sample_weight
            residual = residual * sample_weight
        loss = float(np.mean(squared))
        grad = x.T @ residual / len(y)
    elif kind == "return_mse":
        if sample_weight is None:
            raise ValueError("return_mse requires executable-price sample weights")
        loss = float(np.mean(sample_weight * (probability - y) ** 2))
        grad = x.T @ (2.0 * sample_weight * (probability - y) * probability * (1.0 - probability)) / len(y)
    elif kind == "focal":
        p = np.clip(probability, _EPS, 1.0 - _EPS)
        gamma = _FOCAL_GAMMA
        loss_rows = -y * (1.0 - p) ** gamma * np.log(p) - (1.0 - y) * p**gamma * np.log1p(-p)
        derivative_p = (
            y * (gamma * (1.0 - p) ** (gamma - 1.0) * np.log(p) - (1.0 - p) ** gamma / p)
            + (1.0 - y) * (-gamma * p ** (gamma - 1.0) * np.log1p(-p) + p**gamma / (1.0 - p))
        )
        loss = float(np.mean(loss_rows))
        derivative_eta = np.where(
            (probability > _EPS) & (probability < 1.0 - _EPS),
            probability * (1.0 - probability),
            0.0,
        )
        grad = x.T @ (derivative_p * derivative_eta) / len(y)
    else:
        raise ValueError(f"Unknown differentiable loss {kind!r}")
    delta = theta - center
    penalty = regularization
    loss += 0.5 * penalty * float(delta @ delta)
    grad = grad + penalty * delta
    return loss, grad


def _weighted_favorite_targets(q: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    favorite_orientation = np.where(q > 0.5, 1.0, np.where(q < 0.5, -1.0, 0.0))
    favorite_y = np.where(favorite_orientation > 0, y, np.where(favorite_orientation < 0, 1.0 - y, y))
    # No favorite exists at an exact neutral quote; equal weights preserve A/B symmetry.
    weights = np.where(favorite_orientation == 0.0, 1.0,
                       np.where(favorite_y == 1.0, _CLASS_WEIGHT, 1.0))
    return favorite_orientation, favorite_y, weights


def _focal_posterior(raw: np.ndarray) -> np.ndarray:
    """Analytically invert the population gamma=2 focal optimum."""
    p = np.clip(raw, _EPS, 1.0 - _EPS)
    gamma = _FOCAL_GAMMA
    a = gamma * (1.0 - p) ** (gamma - 1.0) * (-np.log(p)) + (1.0 - p) ** gamma / p
    b = gamma * p ** (gamma - 1.0) * (-np.log1p(-p)) + p**gamma / (1.0 - p)
    return b / (a + b)


def _invert_favorite_weight(raw_favorite: np.ndarray) -> np.ndarray:
    # Weighted log loss with positive-class weight w has optimum w*p/(w*p+1-p).
    return raw_favorite / (_CLASS_WEIGHT - (_CLASS_WEIGHT - 1.0) * raw_favorite)


def _close_probability_target(frame: pd.DataFrame) -> np.ndarray:
    if "close_q" not in frame:
        raise ValueError("Released within-book close_q is required; executable close odds are not no-vig targets")
    close_q = pd.to_numeric(frame["close_q"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(close_q).all() or ((close_q < 0.0) | (close_q > 1.0)).any():
        raise ValueError("Released close_q target must be finite and lie in [0, 1]")
    return np.clip(close_q, _EPS, 1.0 - _EPS)


def _quarter_kelly_fractions(p: np.ndarray, odds_a: np.ndarray, odds_b: np.ndarray,
                             tax_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Quarter-Kelly fractions for each side under after-tax decimal payouts."""
    after_tax = 1.0 - tax_rate
    net_win_a = after_tax * odds_a - 1.0
    net_win_b = after_tax * odds_b - 1.0
    fraction_a = np.zeros(len(p), dtype=float)
    fraction_b = np.zeros(len(p), dtype=float)
    valid_a = net_win_a > 0.0
    valid_b = net_win_b > 0.0
    fraction_a[valid_a] = _KELLY_FRACTION * np.maximum(
        (after_tax * odds_a[valid_a] * p[valid_a] - 1.0) / net_win_a[valid_a], 0.0
    )
    fraction_b[valid_b] = _KELLY_FRACTION * np.maximum(
        (after_tax * odds_b[valid_b] * (1.0 - p[valid_b]) - 1.0) / net_win_b[valid_b], 0.0
    )
    np.minimum(fraction_a, _KELLY_CAP, out=fraction_a)
    np.minimum(fraction_b, _KELLY_CAP, out=fraction_b)
    return fraction_a, fraction_b


def _decision_allocations(p: np.ndarray, odds_a: np.ndarray, odds_b: np.ndarray,
                          tax_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Choose one strictly better positive side, otherwise abstain."""
    fraction_a, fraction_b = _quarter_kelly_fractions(p, odds_a, odds_b, tax_rate)
    difference = fraction_a - fraction_b
    choose_a = (difference > _ACTION_TIE_TOLERANCE) & (fraction_a > 0.0)
    choose_b = (difference < -_ACTION_TIE_TOLERANCE) & (fraction_b > 0.0)
    side = np.full(len(p), None, dtype=object)
    side[choose_a] = "A"
    side[choose_b] = "B"
    fraction = np.where(choose_a, fraction_a, np.where(choose_b, fraction_b, 0.0))
    return side, fraction


def _action_objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                      odds_a: np.ndarray, odds_b: np.ndarray, tax_rate: float,
                      regularization: float, center: np.ndarray, utility: str) -> float:
    p = expit(x @ theta)
    side, fraction = _decision_allocations(p, odds_a, odds_b, tax_rate)
    after_tax = 1.0 - tax_rate
    realized_a = np.where(y == 1.0, after_tax * odds_a - 1.0, -1.0)
    realized_b = np.where(y == 0.0, after_tax * odds_b - 1.0, -1.0)
    realized_return = np.where(side == "A", realized_a, np.where(side == "B", realized_b, 0.0))
    if utility == "profit":
        per_row = fraction * realized_return
    elif utility == "log_wealth":
        per_row = np.log1p(fraction * realized_return)
    else:
        raise ValueError(f"Unknown action utility {utility!r}")
    delta = theta - center
    penalty = regularization
    return -float(np.mean(per_row)) + 0.5 * penalty * float(delta @ delta)


def _fit_action_objective(x: np.ndarray, y: np.ndarray, odds_a: np.ndarray, odds_b: np.ndarray,
                          tax_rate: float, regularization: float, utility: str
                          ) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-start Powell fit avoids treating a flat no-bet region as a gradient."""
    center = np.array([0.0, 1.0])
    starts = [center, np.array([1.0, 0.0]), np.array([1.0, 1.0]),
              np.array([4.0, 0.25]), np.array([0.25, 4.0]), np.array([4.0, 4.0])]
    bounds = [(0.0, None), (0.0, None)]
    objective = lambda theta: _action_objective(
        theta, x, y, odds_a, odds_b, tax_rate, regularization, center, utility
    )
    outcomes: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for start in starts:
        result = minimize(
            objective, start.copy(), method="Powell", bounds=bounds,
            options={"maxiter": 1000, "xtol": 1e-7, "ftol": 1e-10},
        )
        diagnostics = _optimizer_metadata(result, "Powell bounded multi-start")
        diagnostics["start"] = start.tolist()
        outcomes.append((float(result.fun), np.asarray(result.x, dtype=float), diagnostics))
    best_value, best_parameters, best_diagnostics = min(outcomes, key=lambda result: result[0])
    flat = max(result[0] for result in outcomes) - min(result[0] for result in outcomes) <= 1e-12
    return best_parameters, {
        "optimizer": "Powell bounded multi-start",
        "selected": best_diagnostics,
        "runs": [result[2] for result in outcomes],
        "selected_objective_value": best_value,
        "flat_across_starts": bool(flat),
        "flat_region_policy": "retain the best regularized solution among deterministic starts",
    }


@dataclass
class FittedApproach:
    method: str
    coefficients: np.ndarray
    metadata: dict[str, Any]
    regularization: float
    tax_rate: float
    market_offset: bool = False
    calibration_slope: float | None = None

    def _check_predict_inputs(self, frame: pd.DataFrame) -> None:
        if self.method in _QUOTE_REQUIRED:
            _quotes(frame)

    def _favorite_probability(self, frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
        q = _probability_column(frame, "q")
        orientation = np.where(q > 0.5, 1.0, np.where(q < 0.5, -1.0, 0.0))
        favorite_raw = np.where(orientation > 0.0, raw,
                                np.where(orientation < 0.0, 1.0 - raw, raw))
        if self.method == "class_weighted_corrected":
            corrected_favorite = _invert_favorite_weight(favorite_raw)
            favorite = np.where(orientation == 0.0, raw, corrected_favorite)
        else:
            favorite = favorite_raw
        return np.where(orientation < 0.0, 1.0 - favorite, favorite)

    def decision_actions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the one-side-or-abstain stake action; never a win probability."""
        if self.method not in {"direct_profit", "kelly_utility"}:
            raise ValueError("decision_actions is defined only for direct_profit and kelly_utility")
        self._check_predict_inputs(frame)
        x, _ = _features(frame, self.method)
        p = expit(x @ self.coefficients)
        odds_a, odds_b = _quotes(frame)
        side, fraction = _decision_allocations(p, odds_a, odds_b, self.tax_rate)
        return pd.DataFrame(
            {"action_side": side, "action_fraction": fraction},
            index=frame.index,
        )

    def _decision_score(self, frame: pd.DataFrame) -> np.ndarray:
        actions = self.decision_actions(frame)
        direction = np.where(actions["action_side"].eq("A"), 1.0,
                             np.where(actions["action_side"].eq("B"), -1.0, 0.0))
        return 0.5 + 0.5 * direction * actions["action_fraction"].to_numpy(dtype=float) / _KELLY_CAP

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a win probability, target forecast, ranking score, or decision score."""
        if self.method in {"direct_profit", "kelly_utility"}:
            return self._decision_score(frame)
        self._check_predict_inputs(frame)
        x, offset = _features(frame, self.method)
        if self.method == "ranking":
            return x @ self.coefficients
        if self.method == "ranking_calibrated":
            if self.calibration_slope is None:
                raise RuntimeError("ranking_calibrated model has no held-out calibration slope")
            return expit(self.calibration_slope * (x @ self.coefficients))
        if self.method == "clv_target":
            return expit(offset + x @ self.coefficients)
        if self.method == "close_target":
            return expit(x @ self.coefficients)
        raw = expit(offset + x @ self.coefficients if self.market_offset else x @ self.coefficients)
        if self.method in {"class_weighted", "class_weighted_corrected"}:
            return self._favorite_probability(frame, raw)
        if self.method == "focal_corrected":
            return _focal_posterior(raw)
        if self.method == "label_smoothing_corrected":
            return np.clip((raw - _SMOOTH_EPSILON / 2.0) / (1.0 - _SMOOTH_EPSILON), 0.0, 1.0)
        return raw

    def validation_loss(self, frame: pd.DataFrame) -> float:
        """Score the emitted prediction under its declared observable metric."""
        y = _labels(frame)
        if self.method == "ranking":
            x, _ = _features(frame, self.method)
            margin = 2.0 * (2.0 * y - 1.0) * (x @ self.coefficients)
            return float(np.mean(np.logaddexp(0.0, -margin)))
        if self.method in {"direct_profit", "kelly_utility"}:
            actions = self.decision_actions(frame)
            odds_a, odds_b = _quotes(frame)
            after_tax = 1.0 - self.tax_rate
            returns_a = np.where(y == 1.0, after_tax * odds_a - 1.0, -1.0)
            returns_b = np.where(y == 0.0, after_tax * odds_b - 1.0, -1.0)
            realized_return = np.where(actions["action_side"].eq("A"), returns_a,
                                       np.where(actions["action_side"].eq("B"), returns_b, 0.0))
            fraction = actions["action_fraction"].to_numpy(dtype=float)
            if self.method == "direct_profit":
                return -float(np.mean(fraction * realized_return))
            return -float(np.mean(np.log1p(fraction * realized_return)))
        p = np.clip(self.predict(frame), _EPS, 1.0 - _EPS)
        return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log1p(-p))))



def _base_metadata(method: str, train: pd.DataFrame, regularization: float,
                   score_kind: str, objective: str, selection_metric: str,
                   transforms: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "score_kind": score_kind,
        "objective": objective,
        "selection_metric": selection_metric,
        "regularization": float(regularization),
        "regularization_convention": "frozen mean-loss scale: mean(loss) + lambda/2 * ||parameter-center||^2, with no division by sample count",
        "required_probability_inputs": sorted(_PROBABILITY_INPUTS[method]),
        "native_sports_signal_used": "p_s" in _PROBABILITY_INPUTS[method],
        "transformations": transforms,
        "fit_partition": {
            "fit_ids": _row_ids(train),
            "calibration_ids": [],
            "calibration_design": "no separate calibration partition; fitted parameters use fit_ids only",
        },
    }


def _ranker_fit(x: np.ndarray, y: np.ndarray, regularization: float) -> tuple[np.ndarray, dict[str, Any]]:
    center = np.array([0.0, 1.0])
    signed_x = 2.0 * (2.0 * y - 1.0)[:, None] * x

    penalty = regularization
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        margins = signed_x @ theta
        losses = np.logaddexp(0.0, -margins)
        residual = -expit(-margins)
        delta = theta - center
        return (float(np.mean(losses) + 0.5 * penalty * (delta @ delta)),
                signed_x.T @ residual / len(y) + penalty * delta)

    coef, diagnostics = _fit_boxed(
        lambda theta: objective(theta)[0], lambda theta: objective(theta)[1], center.copy(),
        [(0.0, None), (0.0, None)],
    )
    diagnostics["pairwise_comparisons"] = int(len(y))
    diagnostics["pairwise_definition"] = "each match's observed winner outranks its loser; margin=2*winner_signed_score"
    return coef, diagnostics
def _effective_timing(train: pd.DataFrame, actual: str, scenario: str) -> pd.Series | None:
    """Use actual times, or explicitly supplied scenario times, never inferred ones."""
    if actual not in train and scenario not in train:
        return None
    actual_values = (pd.to_datetime(train[actual], utc=True, errors="raise")
                     if actual in train else pd.Series(pd.NaT, index=train.index, dtype="datetime64[ns, UTC]"))
    scenario_values = (pd.to_datetime(train[scenario], utc=True, errors="raise")
                       if scenario in train else pd.Series(pd.NaT, index=train.index, dtype="datetime64[ns, UTC]"))
    return actual_values.where(actual_values.notna(), scenario_values)


def _heldout_order(train: pd.DataFrame) -> tuple[np.ndarray, pd.Series | None]:
    decision = _effective_timing(train, "decision_at", "scenario_decision_at")
    if decision is None:
        return np.arange(len(train), dtype=int), None
    if decision.isna().any():
        raise ValueError("ranking_calibrated requires actual or explicit scenario decision times")
    order_frame = pd.DataFrame({"time": decision, "position": np.arange(len(train))})
    order = order_frame.sort_values(["time", "position"], kind="mergesort")["position"].to_numpy(dtype=int)
    return order, decision.reset_index(drop=True)



def _fit_positive_slope_calibration(scores: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, Any]]:
    if len(scores) != len(y) or len(y) == 0:
        raise ValueError("Positive-slope calibration requires non-empty, equal-length score and label arrays")
    result = minimize_scalar(
        lambda slope: float(np.mean(np.logaddexp(0.0, slope * scores) - y * slope * scores)),
        bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"Positive-slope calibration optimization failed: {result.message}")
    return float(result.x), {
        "optimizer": "bounded positive-slope scalar log-loss",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "objective_value": float(result.fun),
        "parameters": [float(result.x)],
        "slope_bounds": [0.05, 20.0],
    }


def fit_head(method: str, train: pd.DataFrame, regularization: float = 0.1,
             tax_rate: float = 0.12) -> FittedApproach:
    """Fit one frozen low-capacity recipe on caller-supplied past TRAIN rows."""
    if method not in APPROACHES:
        raise ValueError(f"Unknown EV approach method {method!r}")
    if not isinstance(regularization, (int, float, np.integer, np.floating)) or not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and nonnegative")
    if not np.isfinite(tax_rate) or tax_rate < 0.0 or tax_rate >= 1.0:
        raise ValueError("tax_rate must be finite and in [0, 1)")
    if train.empty:
        raise ValueError("Cannot fit an EV approach on an empty TRAIN partition")
    y = _labels(train)
    if method in _QUOTE_REQUIRED:
        odds_a, odds_b = _quotes(train)
    else:
        odds_a = odds_b = None
    x, offset = _features(train, method)
    reg = float(regularization)
    bounds_nonnegative = [(0.0, None)] * x.shape[1]
    centers: dict[str, np.ndarray] = {
        "sports_cal": np.array([1.0]),
        "market_cal": np.array([1.0]),
        "market_shape": np.array([1.0, 1.0]),
        "market_residual": np.array([0.0]),
        "ranking": np.array([0.0, 1.0]),
        "ranking_calibrated": np.array([0.0, 1.0]),
    }
    center = centers.get(method, np.array([0.0, 1.0]))
    market_offset = method == "market_residual"
    diagnostics: dict[str, Any]
    calibration_slope: float | None = None

    if method in {"ranking", "ranking_calibrated"}:
        if method == "ranking":
            coef, diagnostics = _ranker_fit(x, y, reg)
            fit_ids = _row_ids(train)
            cal_ids: list[str] = []
            score_kind = "ranking_score"
            objective = "mean softplus(-2*(2*y-1)*score) + lambda/2*||beta-(0,1)||^2"
            selection_metric = "within-match pairwise logistic ranking loss"
        else:
            order, decision_times = _heldout_order(train)
            cal_count = max(1, int(np.ceil(0.20 * len(order))))
            split = len(order) - cal_count
            if split < 1:
                raise ValueError("ranking_calibrated needs at least one ranker TRAIN row before its later CAL rows")
            rank_positions, cal_positions = order[:split], order[split:]
            calibration = train.iloc[cal_positions]
            if decision_times is not None:
                release_times = _effective_timing(train, "label_available_at", "scenario_label_available_at")
                if release_times is None:
                    raise ValueError("ranking_calibrated requires actual or explicit scenario label release times")
                first_cal_decision = decision_times.iloc[cal_positions].min()
                rank_positions = np.asarray(
                    [pos for pos in rank_positions
                     if pd.notna(release_times.iloc[pos]) and release_times.iloc[pos] < first_cal_decision],
                    dtype=int,
                )
                if len(rank_positions) == 0:
                    raise ValueError("ranking_calibrated has no ranker TRAIN labels released before first CAL decision")
            rank_train = train.iloc[rank_positions]
            rank_x, _ = _features(rank_train, method)
            rank_y = _labels(rank_train)
            coef, rank_diagnostics = _ranker_fit(rank_x, rank_y, reg)
            cal_x, _ = _features(calibration, method)
            cal_y = _labels(calibration)
            calibration_slope, calibration_diagnostics = _fit_positive_slope_calibration(cal_x @ coef, cal_y)
            diagnostics = {"ranker": rank_diagnostics, "calibration": calibration_diagnostics,
                           "ranker_refit_after_calibration": False}
            fit_ids = _row_ids(rank_train)
            cal_ids = _row_ids(calibration)
            score_kind = "probability"
            objective = "within-match pairwise logistic ranking margin; later held-out positive-slope log-loss calibration"
            selection_metric = "winner log loss of the complete fixed-ranker plus later-TRAIN-calibrator recipe"
        metadata = _base_metadata(method, train, reg, score_kind, objective, selection_metric,
                                  {"features": ["logit(p_s)", "logit(q)"], "intercept": "none",
                                   "ranker_calibration": "positive slope only; no A/B-specific intercept"})
        metadata["fit_partition"] = {
            "fit_ids": fit_ids,
            "calibration_ids": cal_ids,
            "ordering": ("actual/scenario decision time then stable input order"
                         if method == "ranking_calibrated" and decision_times is not None
                         else "caller-provided row order; timestamp chronology unavailable"),
            "calibration_design": (
                "later 20% held out; ranker TRAIN labels released before first CAL decision; "
                "calibrator fit on frozen ranker scores"
                if method == "ranking_calibrated" and decision_times is not None
                else "later 20% positional holdout; timestamp and label-release chronology unavailable"
                if method == "ranking_calibrated" else "none"
            ),
        }
        metadata["optimizer"] = diagnostics
        metadata["coefficients"] = coef.tolist()
        if calibration_slope is not None:
            metadata["calibration_slope"] = calibration_slope
        return FittedApproach(method, coef, metadata, reg, float(tax_rate), False, calibration_slope)

    if method in {"sports_cal", "market_cal"}:
        x_fit = x[:, :1]
        center_fit = np.array([1.0])
        objective = lambda theta: _regularized_loss(
            theta, x_fit, y, center_fit, reg, "logloss"
        )
        coef, diagnostics = _fit_boxed(
            lambda theta: objective(theta)[0],
            lambda theta: objective(theta)[1],
            center_fit.copy(),
            [(0.05, 20.0)],
        )
        input_name = "logit(p_s)" if method == "sports_cal" else "logit(q)"
        metadata = _base_metadata(
            method, train, reg, "probability",
            f"mean Bernoulli log loss of sigmoid(a*{input_name}) + lambda/2*(a-1)^2",
            "winner log loss",
            {"features": [input_name], "intercept": "none", "slope_bounds": [0.05, 20.0],
             "regularization_center": [1.0],
             "regularization_convention": "mean(loss) + lambda/2 * ||slope-1||^2"},
        )
        metadata["optimizer"] = diagnostics
        metadata["coefficients"] = coef.tolist()
        return FittedApproach(method, coef, metadata, reg, float(tax_rate))
    if method == "market_shape":
        kind = "logloss"
        center = centers[method]
        bounds = [(0.0, None)] * len(center)
        objective_name = "mean winner log loss + lambda/2*||beta-(1,1)||^2"
        transform = {"features": ["sign(z_q)*min(abs(z_q), logit(0.8))",
                                  "sign(z_q)*max(abs(z_q)-logit(0.8), 0)"],
                     "intercept": "none", "slope_constraints": "nonnegative inner and outer market slopes",
                     "market_knot": _KNOT, "regularization_center": [1.0, 1.0]}
        objective = lambda theta: _regularized_loss(theta, x, y, center, reg, "logloss")
        transforms_extra: dict[str, Any] = {}
        coef, diagnostics = _fit_boxed(
            lambda theta: objective(theta)[0],
            lambda theta: objective(theta)[1],
            center.copy(),
            bounds,
        )
        metadata = _base_metadata(method, train, reg, "probability", objective_name,
                                  "winner log loss",
                                  {**transform, **transforms_extra})
        metadata["optimizer"] = diagnostics
        metadata["coefficients"] = coef.tolist()
        return FittedApproach(method, coef, metadata, reg, float(tax_rate))
    elif method in {"joint_logloss", "joint_brier", "return_mse", "market_residual",
                    "class_weighted", "class_weighted_corrected", "focal", "focal_corrected",
                    "label_smoothing", "label_smoothing_corrected"}:
        if method in {"joint_logloss", "class_weighted", "class_weighted_corrected", "focal", "focal_corrected",
                      "label_smoothing", "label_smoothing_corrected"}:
            kind = "logloss"
        elif method in {"joint_brier", "market_residual"}:
            kind = "brier"
        else:
            kind = "return_mse"
        bounds = bounds_nonnegative
        transform = {"features": ["logit(p_s)", "logit(q)"], "intercept": "none",
                     "slope_constraints": "nonnegative", "market_offset": "logit(q)" if market_offset else None}
        objective_name = {
            "joint_logloss": "mean Bernoulli log loss + lambda/2*||beta-(0,1)||^2",
            "joint_brier": "mean squared winner-probability error + lambda/2*||beta-(0,1)||^2",
            "return_mse": "mean(w_i*(p_hat-y)^2) + lambda/2*||beta-(0,1)||^2, w_i=(odds_a^2+odds_b^2)/mean(odds_a^2+odds_b^2)",
            "market_residual": "mean squared ((y-q) - (p_hat-q)) + lambda/2*alpha^2 with logit(q) offset",
            "class_weighted": "favorite-oriented weighted Bernoulli log loss; favorite-win weight=2, other=1; neutral quote weights=1; plus lambda/2 identity shrinkage",
            "class_weighted_corrected": "same favorite-oriented weighted log loss and shrinkage; inverse odds-to-posterior map at nonneutral q",
            "focal": "mean binary focal loss gamma=2 plus lambda/2 identity shrinkage",
            "focal_corrected": "same gamma=2 focal loss and shrinkage; analytical binary population posterior recovery",
            "label_smoothing": "mean BCE against 0.9*y + 0.05 plus lambda/2 identity shrinkage",
            "label_smoothing_corrected": "same epsilon=0.1 smoothed BCE and shrinkage; inverse population smoothing map",
        }[method]
        if method in {"class_weighted", "class_weighted_corrected"}:
            q = _probability_column(train, "q")
            orientation, favorite_y, sample_weight = _weighted_favorite_targets(q, y)
            training_orientation = np.where(orientation == 0.0, 1.0, orientation)
            x_fit = x * training_orientation[:, None]
            objective = lambda theta: _regularized_loss(theta, x_fit, favorite_y, center, reg, "logloss", sample_weight)
            transforms_extra = {"favorite_orientation": "q>0.5 => A favorite; q<0.5 => B; exact tie un-oriented",
                                "favorite_win_weight": _CLASS_WEIGHT,
                                "posterior_correction": "p_favorite = p_weighted/(2-p_weighted)" if method.endswith("corrected") else None}
        else:
            y_fit = (1.0 - _SMOOTH_EPSILON) * y + _SMOOTH_EPSILON / 2.0 if method.startswith("label_smoothing") else y
            if method.startswith("label_smoothing"):
                objective = lambda theta: _regularized_loss(theta, x, y_fit, center, reg, "logloss")
                transforms_extra = {"smoothed_target": "0.9*y + 0.05",
                                    "posterior_correction": "clip((p_smoothed-0.05)/0.9, 0, 1)" if method.endswith("corrected") else None}
            elif method in {"focal", "focal_corrected"}:
                objective = lambda theta: _regularized_loss(theta, x, y, center, reg, "focal")
                transforms_extra = {"gamma": _FOCAL_GAMMA,
                                    "posterior_correction": "B(p)/(A(p)+B(p)); binary focal population inverse" if method.endswith("corrected") else None}
            elif method == "return_mse":
                weights = ((1.0 - float(tax_rate)) * odds_a) ** 2 + ((1.0 - float(tax_rate)) * odds_b) ** 2
                weights = weights / float(np.mean(weights))
                objective = lambda theta: _regularized_loss(theta, x, y, center, reg, "return_mse", weights)
                transforms_extra = {"side_symmetric_return": "mean weighted Brier loss over both quoted sides; normalized weight=(odds_a^2+odds_b^2)/mean(odds_a^2+odds_b^2), equivalent to after-tax odds because the common factor cancels",
                                    "tax_rate": float(tax_rate)}
            elif method == "market_residual":
                residual_x = x[:, 0]
                def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
                    p = expit(offset + residual_x * theta[0])
                    residual = p - y
                    penalty = reg
                    return (float(np.mean(residual**2) + 0.5 * penalty * theta[0] ** 2),
                            np.array([2.0 * np.mean(residual * p * (1.0 - p) * residual_x)
                                      + penalty * theta[0]]))
                transforms_extra = {"market_reference": "fixed logit(q) offset",
                                    "residual": "alpha*logit(p_s)",
                                    "target_identity": "y-q versus p_hat-q"}
            else:
                objective = lambda theta: _regularized_loss(theta, x, y, center, reg, kind)
                transforms_extra = {}
        center_fit = np.zeros_like(center) if method in {"market_residual"} else center
        def loss(theta: np.ndarray) -> float:
            return objective(theta)[0]
        def gradient(theta: np.ndarray) -> np.ndarray:
            return objective(theta)[1]
        coef, diagnostics = _fit_boxed(loss, gradient, center_fit.copy(), bounds)
        score_kind = "probability"
        if method in {"class_weighted", "class_weighted_corrected"}:
            score_kind = "tilted_probability_score" if method == "class_weighted" else "probability"
        elif method in {"focal", "focal_corrected"}:
            score_kind = "focal_score" if method == "focal" else "probability"
        elif method in {"label_smoothing", "label_smoothing_corrected"}:
            score_kind = "smoothed_probability_score" if method == "label_smoothing" else "probability"
        metadata = _base_metadata(method, train, reg, score_kind, objective_name,
                                  "winner log loss of emitted prediction" if method not in {
                                      "market_residual"} else "winner log loss",
                                  {**transform, **transforms_extra})
        metadata["optimizer"] = diagnostics
        metadata["coefficients"] = coef.tolist()
        if method in {"class_weighted", "focal", "label_smoothing"}:
            metadata["score_identity"] = {
                "class_weighted": "bounded favorite-oriented tilted forecast score; not a true posterior",
                "focal": "bounded raw gamma-two focal forecast score; not a true posterior",
                "label_smoothing": "bounded smoothed-target forecast score; not a true posterior",
            }[method]
        return FittedApproach(method, coef, metadata, reg, float(tax_rate), market_offset)


    if method in {"close_target", "clv_target"}:
        close_q = _close_probability_target(train)
        if method == "close_target":
            target = close_q
            target_name = "released within-book close_q"
            objective_name = "mean squared closing-probability error + lambda/2*||beta-(0,1)||^2"
            center = np.array([0.0, 1.0])
            bounds = bounds_nonnegative
            def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
                p = expit(x @ theta)
                residual = p - target
                delta = theta - center
                return (float(np.mean(residual**2) + 0.5 * reg * (delta @ delta)),
                        x.T @ (2.0 * residual * p * (1.0 - p)) / len(target) + reg * delta)
            selection_metric = "winner log loss of emitted closing-probability forecast"
            score_kind = "probability"
            transformations = {"features": ["logit(p_s)", "logit(q)"], "target": target_name,
                               "prediction": "sigmoid(x beta), estimated released closing no-vig probability",
                               "regularization_center": [0.0, 1.0], "current_quote_required": True}
            fitted_offset = False
        else:
            current_q = _probability_column(train, "q")
            target = logit(close_q) - logit(current_q)
            center = np.zeros(x.shape[1], dtype=float)
            bounds = [(None, None)] * x.shape[1]
            objective_name = "mean squared logit(close_q)-logit(q) error + lambda/2*||beta||^2"
            def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
                residual = x @ theta - target
                return (float(np.mean(residual**2) + 0.5 * reg * (theta @ theta)),
                        2.0 * x.T @ residual / len(target) + reg * theta)
            selection_metric = "winner log loss of emitted closing-probability forecast"
            score_kind = "probability"
            transformations = {"features": ["logit(p_s)", "logit(q)"], "target": "logit(close_q)-logit(q)",
                               "prediction": "sigmoid(logit(current_q)+x beta)",
                               "regularization_center": [0.0, 0.0],
                               "current_quote_required": True}
            fitted_offset = False
        coef, diagnostics = _fit_boxed(
            lambda theta: objective(theta)[0], lambda theta: objective(theta)[1], center.copy(), bounds
        )
        metadata = _base_metadata(method, train, reg, score_kind, objective_name, selection_metric, transformations)
        metadata["optimizer"] = diagnostics
        metadata["coefficients"] = coef.tolist()
        metadata["future_close_used_at_prediction"] = False
        return FittedApproach(method, coef, metadata, reg, float(tax_rate), fitted_offset)

    if method in {"direct_profit", "kelly_utility"}:
        utility = "profit" if method == "direct_profit" else "log_wealth"
        coef, diagnostics = _fit_action_objective(
            x, y, odds_a, odds_b, float(tax_rate), reg, utility
        )
        objective_name = (
            "minimize negative mean realized net PnL plus lambda/2*||beta-(0,1)||^2 under deterministic capped quarter-Kelly actions"
            if method == "direct_profit" else
            "minimize negative mean realized log wealth plus lambda/2*||beta-(0,1)||^2 under deterministic capped quarter-Kelly actions"
        )
        selection_metric = (
            "negative realized PnL under the same deterministic action policy"
            if method == "direct_profit" else
            "negative realized log wealth under the same deterministic action policy"
        )
        metadata = _base_metadata(
            method, train, reg, "decision_score", objective_name, selection_metric,
            {"features": ["logit(p_s)", "logit(q)"], "intercept": "none",
             "action_policy": "compute each side f=clip(0.25*(after_tax_odds*p_side-1)/(after_tax_odds-1),0,0.025) when denominator>0; choose the strictly larger positive f; differences within 1e-12 or both zero abstain",
             "free_capital_fraction_cap": _KELLY_CAP,
             "action_payoffs": "after-tax win return=(1-tax)*decimal_odds-1; loss return=-1; abstain return=0",
             "prediction_contract": "0.5 +/- 0.5*(action_fraction/0.025) by chosen side; bounded decision score, not a probability",
             "action_api": "decision_actions(frame) returns action_side and action_fraction (share of free capital)"},
        )
        metadata["optimizer"] = diagnostics
        metadata["coefficients"] = coef.tolist()
        return FittedApproach(method, coef, metadata, reg, float(tax_rate))

    raise AssertionError(f"Unimplemented registered approach {method!r}")

# Native-C0 is a separately versioned experiment; legacy epsilon and recipes
# above deliberately remain unchanged.
C0_RECIPES = ("S", "S-cal", "M", "M-cal", "M-shape", "Fixed50", "H")
C0_EPSILON = 1e-6
C0_NATIVE_VERSION = "c0-native-2026-w32-e12-v1"


def _c0_probability(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame or frame[name].map(lambda v: isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int,float,np.integer,np.floating))).any():
        raise ValueError(f"Missing, boolean or nonnumeric {name}")
    values = _finite_numeric(frame, name)
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} outside [0,1]")
    return np.clip(values, C0_EPSILON, 1 - C0_EPSILON)


def c0_features(frame: pd.DataFrame, recipe: str) -> np.ndarray:
    if recipe not in C0_RECIPES:
        raise ValueError("Unknown C0 recipe")
    z = logit(_c0_probability(frame, "p_C0")) if recipe in {"S", "S-cal", "Fixed50", "H"} else None
    m = logit(_c0_probability(frame, "q")) if recipe in {"M", "M-cal", "M-shape", "Fixed50", "H"} else None
    if recipe in {"S", "S-cal"}:
        return z[:, None]
    if recipe in {"M", "M-cal"}:
        return m[:, None]
    if recipe == "M-shape":
        return np.column_stack((np.sign(m) * np.minimum(abs(m), _KNOT),
                                np.sign(m) * np.maximum(abs(m) - _KNOT, 0)))
    return np.column_stack((z, m))


@dataclass(frozen=True)
class C0MarketHead:
    recipe_id: str
    coefficients: tuple[float, ...]
    regularization: float
    optimizer: dict[str, Any]
    bindings: dict[str, Any]

    def predict_details(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        eta = c0_features(frame, self.recipe_id) @ np.asarray(self.coefficients)
        raw = expit(eta)
        scored = np.clip(raw, C0_EPSILON, 1 - C0_EPSILON)
        input_clipped = np.zeros(len(frame), dtype=bool)
        required = ({"p_C0"} if self.recipe_id in {"S","S-cal"} else {"q"}
                    if self.recipe_id in {"M","M-cal","M-shape"} else {"p_C0","q"})
        for column in required:
            values = frame[column].to_numpy(float)
            input_clipped |= (values < C0_EPSILON) | (values > 1-C0_EPSILON)
        return {"logit": eta, "raw_probability": raw, "probability": scored,
                "clipped": raw != scored, "input_clipped": input_clipped}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_details(frame)["probability"]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "c0_market_head_v1", "recipe_id": self.recipe_id,
                "coefficients": list(self.coefficients), "regularization": self.regularization,
                "epsilon": C0_EPSILON, "transform": "odd_logit_fixed_knot_0.8",
                "native_identity": f"Causal-C0:{C0_NATIVE_VERSION}", "mode": "full",
                "optimizer": self.optimizer, "bindings": self.bindings,
                "objective": "mean_unclipped_logit_nll_plus_lambda_over_2_centered_squared_l2",
                "center": ([0.,1.] if self.recipe_id == "H" else list(self.coefficients)
                           if self.recipe_id in {"S","M","Fixed50"} else [1.] * len(self.coefficients))}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "C0MarketHead":
        if (value.get("schema") != "c0_market_head_v1" or value.get("mode") != "full"
                or value.get("native_identity") != f"Causal-C0:{C0_NATIVE_VERSION}"
                or value.get("epsilon") != C0_EPSILON
                or value.get("transform") != "odd_logit_fixed_knot_0.8"
                or value.get("objective") != "mean_unclipped_logit_nll_plus_lambda_over_2_centered_squared_l2"):
            raise ValueError("Wrong native head contract")
        recipe = value["recipe_id"]
        expected_center = ([0.,1.] if recipe == "H" else [.5,.5] if recipe == "Fixed50" else
                           [1.,1.] if recipe == "M-shape" else [1.])
        if value.get("center") != expected_center:
            raise ValueError("Wrong coefficient penalty center")
        if (not isinstance(value.get("coefficients"), list) or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) for v in value["coefficients"])):
            raise ValueError("Nonnumeric coefficients")
        theta = tuple(float(v) for v in value["coefficients"])
        expected = 2 if recipe in {"M-shape", "Fixed50", "H"} else 1
        if recipe not in C0_RECIPES or len(theta) != expected or not np.isfinite(theta).all():
            raise ValueError("Invalid coefficients")
        if min(theta) < 0 or (recipe in {"S-cal", "M-cal"} and not .05 <= theta[0] <= 20):
            raise ValueError("Coefficient bounds violated")
        fixed = {"S": (1.,), "M": (1.,), "Fixed50": (.5, .5)}
        if recipe in fixed and theta != fixed[recipe]:
            raise ValueError("Changed fixed recipe")
        regularization = value["regularization"]
        if (isinstance(regularization, bool) or not isinstance(regularization, (int,float))
                or not np.isfinite(regularization) or regularization < 0
                or value.get("optimizer", {}).get("success") is not True
                or not isinstance(value.get("bindings"), dict)):
            raise ValueError("Invalid optimizer/head lineage")
        return cls(recipe, theta, float(value["regularization"]), value["optimizer"], value["bindings"])


def fit_c0_market_head(recipe_id: str, train: pd.DataFrame, *, regularization: float) -> C0MarketHead:
    if (isinstance(regularization, (bool,np.bool_))
            or not isinstance(regularization,(int,float,np.integer,np.floating))
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError("Invalid regularization")
    x = c0_features(train, recipe_id)
    bindings = dict(train.attrs.get("c0_bindings", {}))
    fixed = {"S": (1.,), "M": (1.,), "Fixed50": (.5, .5)}
    if recipe_id in fixed:
        return C0MarketHead(recipe_id, fixed[recipe_id], 0., {"optimizer": "none", "success": True}, bindings)
    if "y" not in train or train["y"].map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ValueError("Missing or boolean winner labels")
    y = _labels(train)
    if len(y) == 0 or len(np.unique(y)) != 2 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("INSUFFICIENT_FIT_SUPPORT")
    center = np.array([0., 1.] if recipe_id == "H" else [1.] * x.shape[1])
    bounds = [(.05, 20.)] if recipe_id in {"S-cal", "M-cal"} else [(0., None)] * x.shape[1]
    def loss(theta):
        return _regularized_loss(theta, x, y, center, regularization, "logloss")
    theta, metadata = _fit_boxed(lambda t: loss(t)[0], lambda t: loss(t)[1], center, bounds)
    eta = x @ theta
    metadata["raw_optimizer_nll"] = float(np.mean(np.logaddexp(0, eta) - y * eta))
    return C0MarketHead(recipe_id, tuple(theta), regularization, metadata, bindings)


def select_c0_market_head(candidates: list[C0MarketHead], cal: pd.DataFrame):
    """Unpenalized clipped CAL LL; largest lambda within the frozen tolerance."""
    if len(candidates) != 3 or sorted(h.regularization for h in candidates) != [.01,.1,1.]:
        raise ValueError("Exactly three declared lambda candidates required")
    if len({h.recipe_id for h in candidates}) != 1 or any(h.bindings != candidates[0].bindings for h in candidates):
        raise ValueError("Candidate family/lineage mismatch")
    y = _labels(cal)
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError("INSUFFICIENT_FIT_SUPPORT")
    scored = []
    for head in candidates:
        p = head.predict(cal)
        ll = float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))
        scored.append((ll,head))
    best = min(v[0] for v in scored)
    ll, chosen = max((v for v in scored if v[0] <= best+1e-12),key=lambda v:v[1].regularization)
    return chosen, {"lambda":chosen.regularization,"LL":ll,
                    "candidates":[{"lambda":h.regularization,"LL":score} for score,h in scored]}
