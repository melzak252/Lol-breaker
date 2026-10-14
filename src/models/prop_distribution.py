"""Standalone sports distributions, TRAIN-fitted means and CAL-only calibration.

Counts use NB2 (variance = mu + alpha * mu**2); duration is lognormal in
minutes. Length columns encode extra maps beyond wins-needed, not absolute
map counts. This module has no data/artifact/C0 dependencies.

Frozen configuration keys: pooled ``smoothing``; glm ``ridge``,
``length_ridge``, ``max_iter``; tree ``num_leaves`` (15/31), ``n_estimators``
(400), ``learning_rate`` (.05), ``min_child_samples`` (100), ``n_jobs`` (4);
neural ``hidden`` (64/128), ``epochs`` (12/24), ``batch_size`` (512), ``lr``
(.001), ``weight_decay`` (.0001). Unknown keys reject. Models and checkpoint
bundles are directly joblib-serializable; no serialization-specific reload API
is needed. A 24-epoch neural fit also exposes its independently calibrated
12-epoch bundle at ``checkpoints[12]``.

Positive support safeguards are applied to model parameters, never to losses
or observations. Calibration requires observed CAL counts and durations;
unidentifiable length temperature is explicitly recorded. Missing TRAIN task
support and optimizer failures raise rather than silently fitting another family.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr
from scipy.stats import nbinom
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
import torch
from torch import nn


# These bounds define the parametric support, not observation/loss clipping.
_LOG_MU_BOUNDS = (-12.0, 12.0)
_LOG_ALPHA_BOUNDS = (-12.0, 5.0)
_LOG_SIGMA_BOUNDS = (-7.0, 2.0)


def _vector(value: Any, n: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    return result


def _best_of(value: Any, n: int) -> np.ndarray:
    result = _vector(value, n, "best_of")
    if not np.isin(result, (1, 3, 5)).all():
        raise ValueError("best_of must contain only 1, 3 or 5")
    return result.astype(np.int64)


def _features(value: Any, width: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError("X must be a [N,D] matrix with at least one feature")
    if width is not None and result.shape[1] != width:
        raise ValueError("feature width differs from TRAIN")
    if np.isinf(result).any():
        raise ValueError("features must not contain infinity (NaN is allowed)")
    return result


def _labels(kills: Any, duration: Any, maps: Any, bo: np.ndarray) -> dict[str, np.ndarray]:
    labels = {"kills": _vector(kills, len(bo), "kills"),
              "duration": _vector(duration, len(bo), "duration"),
              "length": _vector(maps, len(bo), "maps")}
    for name, values in labels.items():
        if np.isinf(values).any():
            raise ValueError(f"{name} labels cannot be infinite")
        valid = ~np.isnan(values)
        observed = values[valid]
        if name == "kills" and np.any((observed < 0) | (observed != np.floor(observed))):
            raise ValueError("kills labels must be nonnegative integers or NaN")
        if name == "duration" and np.any(observed <= 0):
            raise ValueError("duration labels must be positive minutes or NaN")
        if name == "length":
            minimum = bo[valid] // 2 + 1
            if np.any((observed != np.floor(observed)) | (observed < minimum) | (observed > bo[valid])):
                raise ValueError("maps label is outside legal best_of support")
    return labels


def _lines(line: Any, n: int) -> np.ndarray:
    result = np.asarray(line, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(n, float(result))
    if result.shape != (n,) or np.isnan(result).any():
        raise ValueError("line must be scalar or [N], without NaN")
    return result


@dataclass
class DistributionPrediction:
    kills_mu: np.ndarray
    kills_alpha: np.ndarray
    duration_log_mu: np.ndarray
    duration_log_sigma: np.ndarray
    length_probs: np.ndarray
    best_of: np.ndarray

    def __post_init__(self) -> None:
        raw_bo = np.asarray(self.best_of)
        if raw_bo.ndim != 1:
            raise ValueError("best_of must be one-dimensional")
        self.best_of = _best_of(raw_bo, len(raw_bo))
        for name in ("kills_mu", "kills_alpha", "duration_log_mu", "duration_log_sigma"):
            values = _vector(getattr(self, name), len(raw_bo), name)
            if not np.isfinite(values).all() or (name != "duration_log_mu" and np.any(values <= 0)):
                raise ValueError(f"{name} must be finite" + (" and positive" if name != "duration_log_mu" else ""))
            setattr(self, name, values)
        self.length_probs = np.asarray(self.length_probs, dtype=np.float64)
        if self.length_probs.shape != (len(raw_bo), 3):
            raise ValueError("length_probs must have shape [N,3]")
        probs = self.length_probs
        if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1., rtol=1e-7, atol=1e-9):
            raise ValueError("length_probs must be nonnegative finite normalized probabilities")
        legal = np.arange(3)[None, :] <= (self.best_of // 2)[:, None]
        if np.any(probs[~legal] != 0):
            raise ValueError("length_probs puts mass outside legal best_of support")

    def kills_over(self, line: Any) -> np.ndarray:
        r = 1. / self.kills_alpha
        return nbinom.sf(np.floor(_lines(line, len(self.best_of))), r, 1. / (1. + self.kills_alpha * self.kills_mu))

    def duration_over(self, line: Any) -> np.ndarray:
        threshold = _lines(line, len(self.best_of))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(threshold) - self.duration_log_mu) / self.duration_log_sigma
        return np.where(threshold <= 0., 1., ndtr(-z))

    def maps_over(self, line: Any) -> np.ndarray:
        support = (self.best_of // 2 + 1)[:, None] + np.arange(3)
        return (self.length_probs * (support > _lines(line, len(self.best_of))[:, None])).sum(axis=1)


@dataclass
class DistributionMixture:
    """Exact mixture of marginals, not an average of distribution parameters."""
    components: tuple[DistributionPrediction, ...] | list[DistributionPrediction]
    weights: np.ndarray | None = None
    best_of: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.components = tuple(self.components)
        if not self.components or not all(isinstance(c, DistributionPrediction) for c in self.components):
            raise ValueError("mixture needs at least one DistributionPrediction")
        self.best_of = self.components[0].best_of
        if any(not np.array_equal(c.best_of, self.best_of) for c in self.components):
            raise ValueError("mixture components must have identical best_of rows")
        weights = np.ones(len(self.components)) if self.weights is None else np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (len(self.components),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("mixture weights must be finite nonnegative with positive sum")
        scaled = weights / weights.max()
        self.weights = scaled / scaled.sum()

    def _over(self, method: str, line: Any) -> np.ndarray:
        return sum(w * getattr(c, method)(line) for w, c in zip(self.weights, self.components) if w > 0)

    def kills_over(self, line: Any) -> np.ndarray:
        return self._over("kills_over", line)

    def duration_over(self, line: Any) -> np.ndarray:
        return self._over("duration_over", line)

    def maps_over(self, line: Any) -> np.ndarray:
        return self._over("maps_over", line)


def distribution_nll(pred: DistributionPrediction | DistributionMixture, kills: Any, duration: Any, maps: Any) -> dict[str, np.ndarray]:
    """Per-row proper NLL; NaN exclusively marks missing task labels.

    An impossible valid length event has infinite loss (never epsilon-clipped).
    Lognormal logpdf includes the minutes-scale Jacobian.
    """
    labels = _labels(kills, duration, maps, pred.best_of)
    if isinstance(pred, DistributionMixture):
        components = [distribution_nll(c, kills, duration, maps) for c in pred.components]
        with np.errstate(divide="ignore"):
            logweights = np.log(pred.weights)
        return {task: -logsumexp(np.stack([-c[task] for c in components]) + logweights[:, None], axis=0)
                for task in labels}
    result = {task: np.full(len(pred.best_of), np.nan) for task in labels}
    valid = ~np.isnan(labels["kills"])
    alpha, mu = pred.kills_alpha[valid], pred.kills_mu[valid]
    result["kills"][valid] = -nbinom.logpmf(labels["kills"][valid], 1. / alpha, 1. / (1. + alpha * mu))
    valid = ~np.isnan(labels["duration"])
    y = labels["duration"][valid]
    sigma = pred.duration_log_sigma[valid]
    result["duration"][valid] = np.log(y) + np.log(sigma) + .5 * np.log(2. * np.pi) + .5 * ((np.log(y) - pred.duration_log_mu[valid]) / sigma) ** 2
    valid = ~np.isnan(labels["length"])
    indices = (labels["length"][valid] - (pred.best_of[valid] // 2 + 1)).astype(int)
    with np.errstate(divide="ignore"):
        result["length"][valid] = -np.log(pred.length_probs[valid, indices])
    return result


@dataclass
class _TrainPreprocessor:
    mean: np.ndarray
    scale: np.ndarray
    observed: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> _TrainPreprocessor:
        present = ~np.isnan(X)
        counts = present.sum(axis=0)
        mean = np.divide(np.where(present, X, 0.).sum(axis=0), counts, out=np.zeros(X.shape[1]), where=counts > 0)
        filled = np.where(present, X, mean)
        scale = np.sqrt(np.mean((filled - mean) ** 2, axis=0))
        return cls(mean, np.where(scale > 1e-8, scale, 1.), counts > 0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        present = ~np.isnan(X) & self.observed
        result = np.concatenate((np.where(present, (X - self.mean) / self.scale, 0.), (~present).astype(float)), axis=1)
        if not np.isfinite(result).all():
            raise ValueError("TRAIN-preprocessed features are not finite")
        return result


class _SummaryMLP(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 7))

    def forward(self, X: torch.Tensor, bo: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = self.network(X)
        logmu = raw[:, 0].clamp(*_LOG_MU_BOUNDS)
        logalpha = raw[:, 1].clamp(*_LOG_ALPHA_BOUNDS)
        logduration = raw[:, 2].clamp(*_LOG_MU_BOUNDS)
        logsigma = raw[:, 3].clamp(*_LOG_SIGMA_BOUNDS)
        legal = torch.arange(3, device=bo.device)[None, :] <= (bo // 2)[:, None]
        logits = raw[:, 4:].masked_fill(~legal, -torch.inf)
        return logmu, logalpha, logduration, logsigma, logits


def _configuration(family: str, config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "pooled": {"smoothing": 1.},
        "glm": {"ridge": 1., "length_ridge": 1., "max_iter": 1000},
        "tree": {"num_leaves": 15, "n_estimators": 400, "learning_rate": .05, "min_child_samples": 100, "n_jobs": 4},
        "neural": {"hidden": 64, "epochs": 12, "batch_size": 512, "lr": .001, "weight_decay": .0001},
    }
    if family not in defaults:
        raise ValueError("family must be pooled, glm, tree or neural")
    supplied = {} if config is None else dict(config)
    if supplied.keys() - defaults[family].keys():
        raise ValueError(f"unknown {family} config keys: {sorted(supplied.keys() - defaults[family].keys())}")
    result = defaults[family] | supplied
    if any(not np.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("config values must be finite and positive")
    integer_keys = {"max_iter", "num_leaves", "n_estimators", "min_child_samples", "n_jobs", "hidden", "epochs", "batch_size"}
    if any(v != int(v) for k, v in result.items() if k in integer_keys):
        raise ValueError("integer config keys require integers")
    result = {k: int(v) if k in integer_keys else float(v) for k, v in result.items()}
    return result


def _checked_fit(estimator: Any, X: np.ndarray, y: np.ndarray, name: str) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(X, y)
    except ConvergenceWarning as exc:
        raise RuntimeError(f"{name} optimizer failed: {exc}") from exc
    return estimator


def _temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        logits = np.log(probs) / temperature
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


class DistributionModel:
    """Reusable native bundle; all mutable fit state is serializable."""
    def __init__(self, family: str, config: dict[str, Any], seed: int, width: int):
        self.family, self.config, self.seed, self.n_features_in_ = family, config, seed, width
        self.preprocessor: _TrainPreprocessor | None = None
        self.kills_model: Any = None
        self.duration_model: Any = None
        self.length_models: dict[int, Any] = {}
        self.network: _SummaryMLP | None = None
        self.kills_alpha = 1.
        self.duration_sigma = 1.
        self.alpha_factor = 1.
        self.sigma_factor = 1.
        self.length_temperature = 1.
        self.checkpoints: dict[int, DistributionModel] = {}
        self.fit_receipt: dict[str, Any] = {}

    def _raw_predict(self, X: np.ndarray, bo: np.ndarray) -> DistributionPrediction:
        n = len(X)
        if n == 0:
            empty = np.empty(0)
            return DistributionPrediction(empty, empty, empty, empty, np.empty((0, 3)), bo)
        values = self.preprocessor.transform(X) if self.preprocessor is not None else X
        if self.family == "neural":
            if self.network is None:
                raise RuntimeError("neural network is not fitted")
            for best in (3, 5):
                if np.any(bo == best) and not self.fit_receipt["length_train_classes"][str(best)]:
                    raise ValueError(f"no TRAIN length support for BO{best}")
            values = np.concatenate((values, bo[:, None] / 5.), axis=1)
            chunks = []
            self.network.eval()
            with torch.no_grad():
                for start in range(0, n, 4096):
                    outputs = self.network(torch.as_tensor(values[start:start + 4096], dtype=torch.float32), torch.as_tensor(bo[start:start + 4096]))
                    chunks.append([out.numpy() for out in outputs[:4]] + [torch.softmax(outputs[4].double(), dim=1).numpy()])
            combined = [np.concatenate([chunk[i] for chunk in chunks]) for i in range(5)]
            return DistributionPrediction(np.exp(combined[0]), np.exp(combined[1]), combined[2], np.exp(combined[3]), combined[4], bo)
        if self.family == "pooled":
            mu = np.full(n, self.kills_model)
            logduration = np.full(n, self.duration_model)
        else:
            mu = self.kills_model.predict(values)
            logduration = self.duration_model.predict(values)
        # Regressors also share explicitly declared positive parameter support.
        mu = np.exp(np.clip(np.log(np.maximum(mu, np.exp(_LOG_MU_BOUNDS[0]))), *_LOG_MU_BOUNDS))
        logduration = np.clip(logduration, *_LOG_MU_BOUNDS)
        probs = np.zeros((n, 3))
        probs[bo == 1, 0] = 1.
        for best in (3, 5):
            rows = bo == best
            if not rows.any():
                continue
            if best not in self.length_models:
                raise ValueError(f"no TRAIN length support for BO{best}")
            model = self.length_models[best]
            if self.family == "pooled":
                probs[rows, :len(model)] = model
            else:
                predicted = model.predict_proba(values[rows])
                probs[np.ix_(rows, model.classes_.astype(int))] = predicted
        return DistributionPrediction(mu, np.ones(n), logduration, np.ones(n), probs, bo)

    def predict(self, X: Any, best_of: Any) -> DistributionPrediction:
        X = _features(X, self.n_features_in_)
        bo = _best_of(best_of, len(X))
        raw = self._raw_predict(X, bo)
        return DistributionPrediction(raw.kills_mu, raw.kills_alpha * self.alpha_factor if self.family == "neural" else np.full(len(X), self.kills_alpha),
                                      raw.duration_log_mu, raw.duration_log_sigma * self.sigma_factor if self.family == "neural" else np.full(len(X), self.duration_sigma),
                                      _temperature(raw.length_probs, self.length_temperature), bo)


def _bounded_minimum(objective: Any, bounds: tuple[float, float], name: str) -> tuple[float, dict[str, Any]]:
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-8, "maxiter": 500})
    candidates = [(float(result.x), float(result.fun)), (bounds[0], float(objective(bounds[0]))), (bounds[1], float(objective(bounds[1])))]
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"{name} calibration optimizer failed: {result.message}")
    parameter, loss = min(candidates, key=lambda item: item[1])
    return parameter, {"status": "success", "objective": loss, "nfev": int(result.nfev), "bounds": list(bounds), "boundary": parameter in bounds}


def _calibrate(model: DistributionModel, X: np.ndarray, bo: np.ndarray, labels: dict[str, np.ndarray]) -> None:
    raw = model._raw_predict(X, bo)
    receipt: dict[str, Any] = {"partition": "CAL", "means_frozen": True}
    for task in ("kills", "duration"):
        if not np.isfinite(labels[task]).any():
            raise ValueError(f"CAL has no valid {task} labels for dispersion calibration")
    valid = np.isfinite(labels["kills"])
    alpha_base = raw.kills_alpha[valid] if model.family == "neural" else np.ones(valid.sum())
    mu, y = raw.kills_mu[valid], labels["kills"][valid]
    factor_bounds = (.05, 20.) if model.family == "neural" else (1e-6, 100.)
    def count_loss(logfactor: float) -> float:
        alpha = alpha_base * np.exp(logfactor)
        return float(-nbinom.logpmf(y, 1. / alpha, 1. / (1. + alpha * mu)).mean())
    value, status = _bounded_minimum(count_loss, tuple(np.log(factor_bounds)), "NB dispersion")
    alpha = float(np.exp(value))
    if model.family == "neural":
        model.alpha_factor = alpha
    else:
        model.kills_alpha = alpha
    receipt["kills"] = status | {"parameter": "alpha_factor" if model.family == "neural" else "alpha", "value": alpha, "n": int(valid.sum())}
    valid = np.isfinite(labels["duration"])
    residual = np.log(labels["duration"][valid]) - raw.duration_log_mu[valid]
    sigma_base = raw.duration_log_sigma[valid] if model.family == "neural" else np.ones(valid.sum())
    sigma_bounds = (.05, 20.) if model.family == "neural" else (1e-4, 5.)
    def duration_loss(logfactor: float) -> float:
        sigma = sigma_base * np.exp(logfactor)
        return float((np.log(labels["duration"][valid]) + np.log(sigma) + .5 * np.log(2. * np.pi) + .5 * (residual / sigma) ** 2).mean())
    value, status = _bounded_minimum(duration_loss, tuple(np.log(sigma_bounds)), "lognormal dispersion")
    sigma = float(np.exp(value))
    if model.family == "neural":
        model.sigma_factor = sigma
    else:
        model.duration_sigma = sigma
    receipt["duration"] = status | {"parameter": "sigma_factor" if model.family == "neural" else "sigma", "value": sigma, "n": int(valid.sum())}
    valid = np.isfinite(labels["length"]) & (bo != 1)
    if valid.any():
        targets = (labels["length"][valid] - (bo[valid] // 2 + 1)).astype(int)
        probabilities = raw.length_probs[valid]
        if np.any(probabilities[np.arange(len(targets)), targets] == 0):
            raise ValueError("CAL length event has zero TRAIN-model support; no silent probability smoothing")
        with np.errstate(divide="ignore"):
            logprobs = np.log(probabilities)
        def length_loss(temperature: float) -> float:
            logits = logprobs / temperature
            return float((logsumexp(logits, axis=1) - logits[np.arange(len(targets)), targets]).mean())
        value, status = _bounded_minimum(length_loss, (.5, 2.), "length temperature")
        model.length_temperature = value
        receipt["length"] = status | {"temperature": value, "n": int(valid.sum())}
    else:
        receipt["length"] = {"status": "not_identifiable", "reason": "no non-BO1 CAL labels", "temperature": 1., "n": 0}
    model.fit_receipt["calibration"] = receipt


def _fit_neural(model: DistributionModel, X: np.ndarray, bo: np.ndarray, labels: dict[str, np.ndarray]) -> None:
    assert model.preprocessor is not None
    features = torch.as_tensor(model.preprocessor.transform(X), dtype=torch.float32)
    # BO is an explicit summary covariate as well as the categorical support mask.
    features = torch.cat((features, torch.as_tensor(bo[:, None] / 5., dtype=torch.float32)), dim=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model.seed)
        network = _SummaryMLP(features.shape[1], model.config["hidden"])
    model.network = network
    optimizer = torch.optim.AdamW(network.parameters(), lr=model.config["lr"], weight_decay=model.config["weight_decay"])
    generator = torch.Generator().manual_seed(model.seed)
    bo_tensor = torch.as_tensor(bo)
    targets = {task: torch.as_tensor(y, dtype=torch.float64) for task, y in labels.items()}
    updates = 0
    epoch_losses = []
    for epoch in range(1, model.config["epochs"] + 1):
        network.train()
        total_loss, contributing_batches = 0., 0
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), model.config["batch_size"]):
            indices = order[start:start + model.config["batch_size"]]
            outputs = network(features[indices], bo_tensor[indices])
            logmu, logalpha, logduration, logsigma = (out.double() for out in outputs[:4])
            losses = []
            valid = torch.isfinite(targets["kills"][indices])
            if valid.any():
                y = targets["kills"][indices][valid]
                r = torch.exp(-logalpha[valid])
                logratio = logmu[valid] + logalpha[valid]
                lognormalizer = torch.nn.functional.softplus(logratio)
                logpmf = torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.) - r * lognormalizer + y * (logratio - lognormalizer)
                losses.append(-logpmf.mean())
            valid = torch.isfinite(targets["duration"][indices])
            if valid.any():
                logy = torch.log(targets["duration"][indices][valid])
                losses.append((logy + logsigma[valid] + .5 * np.log(2. * np.pi) + .5 * ((logy - logduration[valid]) * torch.exp(-logsigma[valid])) ** 2).mean())
            valid = torch.isfinite(targets["length"][indices]) & (bo_tensor[indices] != 1)
            if valid.any():
                target = (targets["length"][indices][valid] - (bo_tensor[indices][valid] // 2 + 1)).long()
                losses.append(torch.nn.functional.cross_entropy(outputs[4][valid].double(), target))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"neural optimizer nonfinite loss at epoch{epoch}, update{updates}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                raise RuntimeError(f"neural optimizer nonfinite gradient at epoch{epoch}, update{updates}")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in network.parameters()):
                raise RuntimeError(f"neural optimizer nonfinite parameters at epoch{epoch}, update{updates}")
            updates += 1
            total_loss += float(loss.detach())
            contributing_batches += 1
        if contributing_batches == 0:
            raise RuntimeError("neural TRAIN has no observed tasks")
        epoch_losses.append(total_loss / contributing_batches)
        if epoch == 12 and model.config["epochs"] == 24:
            snapshot = copy.deepcopy(model)
            snapshot.config = model.config | {"epochs": 12}
            snapshot.fit_receipt = copy.deepcopy(model.fit_receipt)
            snapshot.fit_receipt.update(config=snapshot.config.copy(), updates=updates, epochs_completed=12,
                                        optimization={"status": "fixed_epochs_completed", "epoch_mean_batch_nll": epoch_losses.copy()}, checkpoint_epochs=[])
            model.checkpoints[12] = snapshot
    model.fit_receipt.update(updates=updates, epochs_completed=model.config["epochs"], checkpoint_epochs=sorted(model.checkpoints),
                             optimization={"status": "fixed_epochs_completed", "epoch_mean_batch_nll": epoch_losses})


def fit_distribution_model(family: str, X: Any, best_of: Any, kills: Any, duration: Any, maps: Any, *,
                           X_cal: Any, best_of_cal: Any, kills_cal: Any, duration_cal: Any, maps_cal: Any,
                           seed: int = 20261007, config: dict[str, Any] | None = None) -> DistributionModel:
    """Fit frozen TRAIN means/conditional dispersion, then CAL-only calibration.

    GLM uses PoissonRegressor(alpha=ridge), Ridge(alpha=ridge), and per-BO
    LogisticRegression(C=1/length_ridge). Trees use Poisson/regression/multiclass
    LightGBM. Unsupported non-deterministic BO length strata raise when scored;
    classifiers require at least two observed classes (no substituted baseline).
    ``fit_receipt`` records actual optimizer updates, task masks, support,
    preprocessing partition, parameter safeguards and independent calibration.
    """
    config = _configuration(family, config)
    if int(seed) != seed:
        raise ValueError("seed must be an integer")
    X = _features(X)
    if len(X) == 0:
        raise ValueError("TRAIN cannot be empty")
    bo = _best_of(best_of, len(X))
    labels = _labels(kills, duration, maps, bo)
    X_cal = _features(X_cal, X.shape[1])
    bo_cal = _best_of(best_of_cal, len(X_cal))
    cal_labels = _labels(kills_cal, duration_cal, maps_cal, bo_cal)
    for task in ("kills", "duration"):
        if not np.isfinite(labels[task]).any():
            raise ValueError(f"TRAIN has no valid {task} labels")
        if not np.isfinite(cal_labels[task]).any():
            raise ValueError(f"CAL has no valid {task} labels")
    model = DistributionModel(family, config, int(seed), X.shape[1])
    model.fit_receipt = {"family": family, "seed": int(seed), "config": config.copy(), "updates": 0,
                         "train_rows": len(X), "cal_rows": len(X_cal), "preprocessing_partition": "TRAIN" if family in ("glm", "neural") else "none",
                         "task_support": {task: {"train": int(np.isfinite(y).sum()), "cal": int(np.isfinite(cal_labels[task]).sum())} for task, y in labels.items()},
                         "length_train_classes": {}, "optimization": {"status": "not_started"},
                         "parameter_support": {"log_mu": list(_LOG_MU_BOUNDS), "log_alpha_neural": list(_LOG_ALPHA_BOUNDS), "log_sigma_neural": list(_LOG_SIGMA_BOUNDS)},
                         "checkpoint_epochs": []}
    for best in (1, 3, 5):
        observed = (bo == best) & np.isfinite(labels["length"])
        model.fit_receipt["length_train_classes"][str(best)] = np.unique(labels["length"][observed]).astype(int).tolist()
    if family in ("glm", "neural"):
        model.preprocessor = _TrainPreprocessor.fit(X)
    values = model.preprocessor.transform(X) if model.preprocessor is not None else X
    if family == "neural":
        _fit_neural(model, X, bo, labels)
    else:
        kill_rows, duration_rows = np.isfinite(labels["kills"]), np.isfinite(labels["duration"])
        if family == "pooled":
            model.kills_model = float(np.mean(labels["kills"][kill_rows]))
            model.duration_model = float(np.mean(np.log(labels["duration"][duration_rows])))
        elif family == "glm":
            model.kills_model = _checked_fit(PoissonRegressor(alpha=config["ridge"], max_iter=config["max_iter"]), values[kill_rows], labels["kills"][kill_rows], "PoissonRegressor")
            model.duration_model = _checked_fit(Ridge(alpha=config["ridge"]), values[duration_rows], np.log(labels["duration"][duration_rows]), "Ridge")
        else:
            from lightgbm import LGBMClassifier, LGBMRegressor
            params = config | {"random_state": int(seed), "verbosity": -1, "deterministic": True, "force_col_wise": True}
            model.kills_model = _checked_fit(LGBMRegressor(objective="poisson", **params), values[kill_rows], labels["kills"][kill_rows], "LightGBM Poisson")
            model.duration_model = _checked_fit(LGBMRegressor(objective="regression", **params), values[duration_rows], np.log(labels["duration"][duration_rows]), "LightGBM duration")
        for best in (3, 5):
            rows = (bo == best) & np.isfinite(labels["length"])
            if not rows.any():
                continue
            targets = (labels["length"][rows] - (best // 2 + 1)).astype(int)
            if family == "pooled":
                counts = np.bincount(targets, minlength=best // 2 + 1) + config["smoothing"]
                model.length_models[best] = counts / counts.sum()
            else:
                if len(np.unique(targets)) < 2:
                    raise ValueError(f"{family} BO{best} TRAIN lengths require at least two observed classes")
                if family == "glm":
                    estimator = LogisticRegression(C=1. / config["length_ridge"], max_iter=config["max_iter"], random_state=int(seed))
                else:
                    estimator = LGBMClassifier(objective="multiclass", num_class=len(np.unique(targets)), **params)
                model.length_models[best] = _checked_fit(estimator, values[rows], targets, f"{family} BO{best} lengths")
        if family == "pooled":
            model.fit_receipt["optimization"] = {"status": "analytic_TRAIN_estimates"}
        elif family == "glm":
            iterations = {"kills": int(model.kills_model.n_iter_), "duration": "closed_form",
                          "length": {str(best): np.asarray(est.n_iter_).astype(int).tolist() for best, est in model.length_models.items()}}
            model.fit_receipt["updates"] = iterations["kills"] + sum(sum(v) for v in iterations["length"].values())
            model.fit_receipt["optimization"] = {"status": "success", "iterations": iterations}
        else:
            iterations = {"kills": int(model.kills_model.booster_.current_iteration()), "duration": int(model.duration_model.booster_.current_iteration()),
                          "length": {str(best): int(est.booster_.current_iteration()) for best, est in model.length_models.items()}}
            model.fit_receipt["updates"] = iterations["kills"] + iterations["duration"] + sum(iterations["length"].values())
            model.fit_receipt["optimization"] = {"status": "success", "boosting_iterations": iterations}
    _calibrate(model, X_cal, bo_cal, cal_labels)
    for snapshot in model.checkpoints.values():
        _calibrate(snapshot, X_cal, bo_cal, cal_labels)
    return model
