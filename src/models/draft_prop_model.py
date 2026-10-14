"""Fixed Stage2 TRAIN mean heads with CAL-only reflected-mixture dispersion.

The frozen length model is retained solely for distribution API compatibility;
this module fits neither length nor any assessment-dependent parameters.
"""
from __future__ import annotations

from numbers import Real
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import nbinom

from src.models.prop_distribution import DistributionMixture, DistributionPrediction


_DEFAULT_CONFIG = {
    "num_leaves": 15, "n_estimators": 400, "learning_rate": .05,
    "min_child_samples": 100, "n_jobs": 4, "verbosity": -1,
    "deterministic": True, "force_col_wise": True,
}
_ALPHA_BOUNDS = (1e-6, 100.)
_SIGMA_BOUNDS = (1e-4, 5.)
_SLOT = re.compile(r"(d_(?:champion|player_logcount|role_logcount|role_frequency|kills_mean|duration_mean))_([ab])_([0-4])\Z")


def _names(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{name} must contain column names")
    result = list(value)
    if len(set(result)) != len(result) or (not result and not allow_empty):
        raise ValueError(f"{name} must be unique" + (" and nonempty" if not allow_empty else ""))
    return result


def _configuration(config: dict) -> dict:
    if not isinstance(config, dict) or config.keys() - _DEFAULT_CONFIG.keys():
        raise ValueError("config must use only the frozen LightGBM keys")
    result = _DEFAULT_CONFIG | config
    for key in ("deterministic", "force_col_wise"):
        if result[key] is not True:
            raise ValueError(f"{key} must remain True")
    if isinstance(result["verbosity"], bool) or result["verbosity"] != -1:
        raise ValueError("verbosity must remain -1")
    for key in ("num_leaves", "n_estimators", "min_child_samples", "n_jobs"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value != int(value) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(value)
    if result["num_leaves"] < 2:
        raise ValueError("num_leaves must be at least 2")
    rate = result["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, Real) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    result["learning_rate"] = float(rate)
    return result


def _schema(feature_columns, categorical_features, side_swap, reference_features):
    features = _names(feature_columns, "feature_columns")
    categories = _names(categorical_features, "categorical_features", allow_empty=True)
    references = _names(reference_features, "reference_features")
    if any(not name.startswith(("x_", "p_", "d_")) for name in features):
        raise ValueError("feature_columns must contain only x_, p_ or d_ predictors")
    if set(categories) - set(features) or set(categories) != {name for name in features if name.startswith("d_champion_")}:
        raise ValueError("categorical_features must name every and only champion predictor")
    if not isinstance(side_swap, dict):
        raise ValueError("side_swap must be a column mapping")
    draft = {name for name in features if name.startswith("d_")}
    if set(side_swap) != draft or any(not isinstance(v, str) or v not in draft for v in side_swap.values()):
        raise ValueError("side_swap must cover exactly the draft feature columns")
    for name in draft:
        match = _SLOT.fullmatch(name)
        if match is None:
            raise ValueError(f"invalid draft role-slot feature: {name}")
        prefix, team, role = match.groups()
        other = f"{prefix}_{'b' if team == 'a' else 'a'}_{role}"
        if side_swap[name] != other or side_swap.get(other) != name:
            raise ValueError("side_swap must be an involution preserving feature kind and played role")
    if set(references) - set(features) or any(not name.startswith("x_") for name in references):
        raise ValueError("reference_features must be unchanged symmetric base x_ features")
    return features, categories, dict(side_swap), references


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("input must be a DataFrame with unique columns")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    if any(not pd.api.types.is_numeric_dtype(frame[name].dtype) or pd.api.types.is_bool_dtype(frame[name].dtype) for name in columns):
        raise ValueError("features must have numeric dtypes with integer-coded categorical identities")
    values = frame[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("features cannot contain infinity; numeric NaN cold history is allowed")
    return values


def _best_of(frame: pd.DataFrame) -> np.ndarray:
    if "best_of" not in frame:
        raise ValueError("missing best_of column")
    values = frame["best_of"].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isin(values, (1, 3, 5)).all():
        raise ValueError("best_of must contain only 1, 3 or 5")
    return values.astype(np.int64)


def _labels(frame: pd.DataFrame, bo: np.ndarray) -> dict[str, np.ndarray]:
    labels = {}
    for task, column in (("kills", "kills"), ("duration", "duration_minutes"), ("length", "maps_played")):
        if column not in frame:
            raise ValueError(f"missing {column} label column")
        values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"{column} labels cannot be infinite")
        observed = ~np.isnan(values)
        y = values[observed]
        if task == "kills" and np.any((y < 0) | (y != np.floor(y))):
            raise ValueError("kills labels must be nonnegative integers or NaN")
        if task == "duration" and np.any(y <= 0):
            raise ValueError("duration labels must be positive minutes or NaN")
        if task == "length" and np.any((y != np.floor(y)) | (y < bo[observed] // 2 + 1) | (y > bo[observed])):
            raise ValueError("maps_played labels are outside legal best_of support")
        labels[task] = values
    return labels


def _category_codes(values: np.ndarray, indices: list[int]) -> np.ndarray:
    codes = values[:, indices]
    if not np.isfinite(codes).all() or np.any((codes != np.floor(codes)) | (codes < -1) | (codes >= np.iinfo(np.int32).max)):
        raise ValueError("champion category codes must be -1 or nonnegative int32 codes")
    observed = np.unique(codes[codes >= 0]).astype(np.int64)
    return observed


def _minimum(objective, bounds: tuple[float, float], name: str) -> tuple[float, dict]:
    log_bounds = tuple(np.log(bounds))
    result = minimize_scalar(objective, bounds=log_bounds, method="bounded", options={"xatol": 1e-8, "maxiter": 500})
    if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError(f"{name} CAL optimizer failed: {result.message}")
    candidates = [(float(result.x), float(result.fun))] + [(x, float(objective(x))) for x in log_bounds]
    parameter, loss = min(candidates, key=lambda item: item[1])
    if not np.isfinite(loss):
        raise RuntimeError(f"{name} CAL optimizer produced nonfinite NLL")
    value = float(np.exp(parameter))
    return value, {"status": "success", "objective": loss, "nfev": int(result.nfev),
                   "bounds": list(bounds), "boundary": parameter in log_bounds, "value": value}


class DraftPropModel:
    """Directly joblib-serializable fixed mean heads and scalar dispersions."""

    def __init__(self, *, feature_columns, categorical_features, side_swap,
                 length_reference, reference_features, config, seed):
        self.feature_columns = list(feature_columns)
        self.categorical_features = list(categorical_features)
        self.side_swap = dict(side_swap)
        self.length_reference = length_reference
        self.reference_features = list(reference_features)
        self.config = dict(config)
        self.seed = seed
        self.n_features_in_ = len(self.feature_columns)
        self.categorical_indices = [self.feature_columns.index(name) for name in self.categorical_features]
        self.reflection_indices = [self.feature_columns.index(self.side_swap.get(name, name)) for name in self.feature_columns]
        self.champion_codes = np.empty(0, dtype=np.int64)
        self.kills_model: Any = None
        self.duration_model: Any = None
        self.kills_alpha: float | None = None
        self.duration_sigma: float | None = None
        self.fit_receipt: dict = {}

    def _means(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kills_model is None or self.duration_model is None:
            raise RuntimeError("TRAIN mean heads are not fitted")
        if len(values) == 0:
            return np.empty((2, 0)), np.empty((2, 0))
        oriented = np.concatenate((values, values[:, self.reflection_indices]), axis=0) if self.side_swap else values
        mu = np.asarray(self.kills_model.booster_.predict(oriented, num_threads=self.config["n_jobs"]), dtype=np.float64)
        logmu = np.asarray(self.duration_model.booster_.predict(oriented, num_threads=self.config["n_jobs"]), dtype=np.float64)
        if self.side_swap:
            mu, logmu = mu.reshape(2, len(values)), logmu.reshape(2, len(values))
        else:
            mu, logmu = np.stack((mu, mu)), np.stack((logmu, logmu))
        with np.errstate(over="ignore", under="ignore"):
            duration_mean = np.exp(logmu)
        if not np.isfinite(mu).all() or np.any(mu <= 0) or not np.isfinite(logmu).all() or not np.isfinite(duration_mean).all() or np.any(duration_mean <= 0):
            raise ValueError("mean heads must produce finite positive legal marginals without clipping")
        return mu, logmu

    def _length_probs(self, frame: pd.DataFrame, bo: np.ndarray) -> np.ndarray:
        reference = self.length_reference.predict(_matrix(frame, self.reference_features), bo)
        if not isinstance(reference, (DistributionPrediction, DistributionMixture)) or not np.array_equal(reference.best_of, bo):
            raise ValueError("frozen length prediction must preserve best_of row support")
        if isinstance(reference, DistributionMixture):
            return sum(w * component.length_probs for w, component in zip(reference.weights, reference.components) if w > 0)
        return reference.length_probs.copy()

    def predict(self, frame: pd.DataFrame) -> DistributionMixture:
        values = _matrix(frame, self.feature_columns)
        _category_codes(values, self.categorical_indices)
        bo = _best_of(frame)
        if self.kills_alpha is None or self.duration_sigma is None:
            raise RuntimeError("CAL dispersions are not fitted")
        mu, logmu = self._means(values)
        probs = self._length_probs(frame, bo)
        components = tuple(DistributionPrediction(mu[i], np.full(len(bo), self.kills_alpha),
                                                 logmu[i], np.full(len(bo), self.duration_sigma),
                                                 probs.copy(), bo) for i in range(2))
        return DistributionMixture(components, np.array([.5, .5]))


def fit_draft_prop_model(train: pd.DataFrame, cal: pd.DataFrame, *,
                         feature_columns: list[str], categorical_features: list[str],
                         side_swap: dict[str, str], length_reference: object,
                         reference_features: list[str], config: dict,
                         seed: int = 20261007) -> DraftPropModel:
    """Fit exactly two TRAIN regressors and two CAL global dispersion scalars.

    Both controls duplicate TRAIN rows in two orientations with half weights.
    Missing labels are masked per task; missing numeric history stays NaN and
    categorical unknowns remain -1, LightGBM's missing categorical sentinel.
    """
    features, categories, swap, references = _schema(feature_columns, categorical_features, side_swap, reference_features)
    recipe = _configuration(config)
    if isinstance(seed, bool) or not isinstance(seed, Real) or not np.isfinite(seed) or seed != int(seed) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("seed must be a nonnegative int32 integer")
    if not callable(getattr(length_reference, "predict", None)):
        raise ValueError("length_reference must expose genuine frozen predict(X, best_of)")
    width = getattr(length_reference, "n_features_in_", len(references))
    if width != len(references):
        raise ValueError("reference feature width differs from frozen model schema")
    X, X_cal = _matrix(train, features), _matrix(cal, features)
    _matrix(train, references)
    _matrix(cal, references)
    bo, bo_cal = _best_of(train), _best_of(cal)
    labels, cal_labels = _labels(train, bo), _labels(cal, bo_cal)
    for task in ("kills", "duration"):
        if not np.isfinite(labels[task]).any() or not np.isfinite(cal_labels[task]).any():
            raise ValueError(f"TRAIN and CAL each require observed {task} label support")
    if not np.any(labels["kills"][np.isfinite(labels["kills"])] > 0):
        raise ValueError("TRAIN kills require positive Poisson mean support")
    model = DraftPropModel(feature_columns=features, categorical_features=categories, side_swap=swap,
                           length_reference=length_reference, reference_features=references,
                           config=recipe, seed=int(seed))
    model.champion_codes = _category_codes(X, model.categorical_indices)
    _category_codes(X_cal, model.categorical_indices)
    # Validate genuine frozen length support before undertaking mean fits.
    length_probs = model._length_probs(cal, bo_cal)
    DistributionPrediction(np.ones(len(cal)), np.ones(len(cal)), np.zeros(len(cal)),
                           np.ones(len(cal)), length_probs, bo_cal)
    reflected = np.concatenate((X, X[:, model.reflection_indices]), axis=0)
    from lightgbm import LGBMRegressor
    params = recipe | {"random_state": int(seed)}
    for task, objective, attribute in (("kills", "poisson", "kills_model"), ("duration", "regression", "duration_model")):
        valid = np.isfinite(labels[task])
        rows = np.tile(valid, 2)
        target = labels[task][valid]
        if task == "duration":
            target = np.log(target)
        target = np.tile(target, 2)
        estimator = LGBMRegressor(objective=objective, **params)
        estimator.fit(reflected[rows], target, sample_weight=np.full(len(target), .5),
                      categorical_feature=model.categorical_indices)
        if estimator.booster_.current_iteration() < 1:
            raise RuntimeError(f"{task} LightGBM performed no boosting iterations")
        setattr(model, attribute, estimator)
    mu, logmu = model._means(X_cal)
    count_valid = np.isfinite(cal_labels["kills"])
    count_y, count_mu = cal_labels["kills"][count_valid], mu[:, count_valid]
    def count_loss(logalpha):
        alpha = np.exp(logalpha)
        logs = nbinom.logpmf(count_y[None, :], 1. / alpha, 1. / (1. + alpha * count_mu))
        return float(-(logsumexp(logs, axis=0) - np.log(2.)).mean())
    model.kills_alpha, count_status = _minimum(count_loss, _ALPHA_BOUNDS, "NB2 alpha")
    duration_valid = np.isfinite(cal_labels["duration"])
    logy = np.log(cal_labels["duration"][duration_valid])
    residual = logy[None, :] - logmu[:, duration_valid]
    def duration_loss(logsigma):
        sigma = np.exp(logsigma)
        logs = -logy[None, :] - logsigma - .5 * np.log(2. * np.pi) - .5 * (residual / sigma) ** 2
        return float(-(logsumexp(logs, axis=0) - np.log(2.)).mean())
    model.duration_sigma, duration_status = _minimum(duration_loss, _SIGMA_BOUNDS, "lognormal sigma")
    model.fit_receipt = {
        "family": "draft_prop_reflected_lightgbm", "seed": int(seed), "config": recipe.copy(),
        "train_rows": len(train), "cal_rows": len(cal), "augmented_train_rows": 2 * len(train),
        "orientation_weights": [.5, .5], "mean_regressors": 2, "dispersion_fits": 2,
        "length_fits": 0, "mean_partition": "TRAIN", "dispersion_partition": "CAL",
        "means_frozen_for_calibration": True, "preprocessing_partition": "none",
        "task_support": {task: {"train": int(np.isfinite(y).sum()), "cal": int(np.isfinite(cal_labels[task]).sum())}
                         for task, y in labels.items()},
        "schema": {"feature_columns": features.copy(), "categorical_features": categories.copy(),
                   "categorical_indices": model.categorical_indices.copy(), "side_swap": swap.copy(),
                   "reference_features": references.copy(), "observed_train_champion_codes": model.champion_codes.tolist()},
        "mean_fits": {task: {"partition": "TRAIN", "objective": estimator.objective,
                             "label_rows": int(np.isfinite(labels[task]).sum()),
                             "orientation_rows": 2 * int(np.isfinite(labels[task]).sum()),
                             "total_sample_weight": int(np.isfinite(labels[task]).sum()),
                             "boosting_iterations": int(estimator.booster_.current_iteration())}
                      for task, estimator in (("kills", model.kills_model), ("duration", model.duration_model))},
        "calibration": {"partition": "CAL", "means_frozen": True, "exact_orientation_mixture": True,
                        "kills": count_status | {"parameter": "alpha", "n": int(count_valid.sum())},
                        "duration": duration_status | {"parameter": "sigma", "n": int(duration_valid.sum())}},
        "parameter_support": {"alpha": list(_ALPHA_BOUNDS), "sigma": list(_SIGMA_BOUNDS), "mean_clipping": False},
        "length": {"source": "unchanged_frozen_reference", "purpose": "API_compatibility_only", "fits": 0},
    }
    return model
