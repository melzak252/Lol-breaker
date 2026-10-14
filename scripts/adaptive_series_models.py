"""EXP-090 odds-free full-series models; no data loading or market inputs.

All feature columns are odd under opponent exchange. Calibration and temporal
partitions belong to the runner; these inference objects retain no outcomes.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.train_and_tune_siamese_series import _training_matrix, fit_platt_scaling
from src.models.redesign_series import _labels, _log_loss, _matrix, _open_probability


@dataclass
class LinearScore:
    scale: np.ndarray
    coefficient: np.ndarray

    def logit(self, x: np.ndarray) -> np.ndarray:
        x = _matrix(x, "prediction data", len(self.scale))
        return (x / self.scale) @ self.coefficient


@dataclass
class ResidualScore:
    base: LinearScore
    booster: object
    weight: float = 1.0

    def logit(self, x: np.ndarray) -> np.ndarray:
        x = _matrix(x, "prediction data", len(self.base.scale))
        correction = self.booster.predict(x, raw_score=True, num_threads=1)
        reverse = self.booster.predict(-x, raw_score=True, num_threads=1)
        return self.base.logit(x) + self.weight * 0.5 * (correction - reverse)


def deployed_logits(score, x: np.ndarray) -> np.ndarray:
    """Calibrate precisely the bounded probability emitted at inference."""
    return logit(_open_probability(expit(score.logit(x))))


def recency_weights(dates, fit_before, *, half_life_days: float = 365.0):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    cutoff = pd.Timestamp(fit_before)
    if not len(dates) or dates.isna().any() or pd.isna(cutoff):
        raise ValueError("training dates and cutoff must be complete")
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("half life must be finite and positive")
    if not (dates < cutoff).all():
        raise ValueError(
            "every weighted training outcome must be before the fit cutoff"
        )
    age = np.asarray((cutoff - dates).total_seconds()) / 86400.0
    weights = np.exp2(-age / half_life_days)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("recency weights underflowed or are invalid")
    weights *= len(weights) / weights.sum()
    return weights, {
        "half_life_days": float(half_life_days),
        "n": len(weights),
        "total_mass": float(weights.sum()),
        "effective_sample_size": float(weights.sum() ** 2 / (weights @ weights)),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "fit_before": cutoff.isoformat(),
        "latest_training_date": dates.max().isoformat(),
    }


def fit_ridge(x, y, *, sample_weight=None, c: float = 0.1):
    x = _training_matrix(x, "training data")
    y = _labels(y, "training labels", len(x))
    if not np.isfinite(c) or c <= 0:
        raise ValueError("ridge C must be finite and positive")
    if len(np.unique(y)) != 2:
        raise ValueError("training labels must contain both classes")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if (
            sample_weight.shape != y.shape
            or not np.isfinite(sample_weight).all()
            or np.any(sample_weight <= 0)
        ):
            raise ValueError(
                "sample weights must be finite, positive and match training rows"
            )
        if not np.isclose(sample_weight.sum(), len(x), rtol=1e-10):
            raise ValueError(
                "sample weights must preserve training mass and ridge penalty strength"
            )
    # Match the frozen C=.1 control's unweighted TRAIN-only scale exactly.
    # Recency changes the outcome objective, not the feature basis or penalty.
    scaler = StandardScaler(with_mean=False)
    transformed = scaler.fit_transform(x)
    linear = LogisticRegression(
        C=c,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-8,
        random_state=90,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        linear.fit(transformed, y, sample_weight=sample_weight)
    model = LinearScore(scaler.scale_, linear.coef_[0])
    return model, {
        "c": float(c),
        "optimizer_iterations": int(linear.n_iter_[0]),
        "n": len(x),
        "scale_fit": "unweighted_train_only",
    }


def _offset_stop_loss(y, prediction):
    n = len(y) // 2
    if not n or len(y) != 2 * n or not np.array_equal(y[n:], 1 - y[:n]):
        raise ValueError("offset stopping requires aligned original/swapped outcomes")
    # LightGBM supplies probabilities INCLUDING eval_init_score here. Its
    # serialized booster supplies only the learned residual at inference.
    z = logit(_open_probability(prediction))
    p = expit(0.5 * (z[:n] - z[n:]))
    return "offset_projected_log_loss", _log_loss(y[:n], p), False


def fit_residual_booster(
    base,
    x_train,
    y_train,
    x_stop,
    y_stop,
    *,
    seed: int,
    max_rounds: int = 500,
    min_child_samples: int = 200,
):
    import lightgbm as lgb

    x_train = _matrix(x_train, "training data", len(base.scale))
    x_stop = _matrix(x_stop, "stopping data", len(base.scale))
    y_train = _labels(y_train, "training labels", len(x_train))
    y_stop = _labels(y_stop, "stopping labels", len(x_stop))
    if max_rounds < 1 or min_child_samples < 1:
        raise ValueError("boosting rounds and leaf population must be positive")
    train_offset, stop_offset = base.logit(x_train), base.logit(x_stop)
    params = {
        "objective": "binary",
        "metric": "None",
        "n_estimators": max_rounds,
        "max_depth": 2,
        "num_leaves": 4,
        "min_child_samples": min_child_samples,
        "learning_rate": 0.03,
        "reg_lambda": 100.0,
        "max_bin": 63,
        "random_state": seed,
        "n_jobs": 1,
        "device_type": "cpu",
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    fitted = lgb.LGBMClassifier(**params)
    fitted.fit(
        np.concatenate((x_train, -x_train)),
        np.concatenate((y_train, 1 - y_train)),
        sample_weight=np.full(2 * len(x_train), 0.5),
        init_score=np.concatenate((train_offset, -train_offset)),
        eval_set=[
            (np.concatenate((x_stop, -x_stop)), np.concatenate((y_stop, 1 - y_stop)))
        ],
        eval_init_score=[np.concatenate((stop_offset, -stop_offset))],
        eval_metric=_offset_stop_loss,
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
    )
    if fitted.best_iteration_ <= 0:
        raise ValueError("residual booster did not select a stopping iteration")
    booster = lgb.Booster(
        model_str=fitted.booster_.model_to_string(num_iteration=fitted.best_iteration_)
    )
    return ResidualScore(base, booster), {
        "parameters": params,
        "best_iteration": int(fitted.best_iteration_),
        "stop_log_loss": _log_loss(
            y_stop, expit(ResidualScore(base, booster).logit(x_stop))
        ),
        "split_count": int(booster.feature_importance(importance_type="split").sum()),
        "offset": "frozen_full_training_ridge; in-sample TRAIN residuals, independent STOP/SELECT/CAL/TEST",
        "projection": "base_logit + weight * (residual(x)-residual(-x))/2",
    }


def fit_prequential_slope(predictions: pd.DataFrame, forecast_start):
    """Fit prior-six-month calibration from genuinely earlier model predictions.

    The new base may refit on all past outcomes, including these calibration
    dates. The calibration logits themselves MUST originate from earlier
    quarterly bases that had not observed the respective outcome.
    """
    start = pd.Timestamp(forecast_start)
    if (
        pd.isna(start)
        or start.day != 1
        or start.month not in (1, 4, 7, 10)
        or start != start.normalize()
    ):
        raise ValueError("forecast start must be an exact calendar quarter boundary")
    dates = pd.to_datetime(predictions["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("prediction dates must be complete")
    window_start = start - pd.DateOffset(months=6)
    past = predictions.loc[(dates >= window_start) & (dates < start)].copy()
    if (
        past.empty
        or past.golgg_match_id.isna().any()
        or past.golgg_match_id.duplicated().any()
    ):
        raise ValueError(
            "prequential calibration requires unique nonempty prior series"
        )
    past_dates = pd.to_datetime(past.date)
    fit_before = pd.to_datetime(past.base_fit_before, errors="raise")
    train_max = pd.to_datetime(past.base_train_max_date, errors="raise")
    expected_boundary = past_dates.dt.to_period("Q").dt.start_time
    if (
        fit_before.isna().any()
        or train_max.isna().any()
        or not (
            (train_max < fit_before)
            & (fit_before <= past_dates)
            & (fit_before == expected_boundary)
        ).all()
    ):
        raise ValueError(
            "calibration logits must be out-of-sample quarterly predictions"
        )
    expected_quarters = pd.period_range(
        window_start, start - pd.Timedelta(days=1), freq="Q"
    )
    if set(past_dates.dt.to_period("Q")) != set(expected_quarters):
        raise ValueError("both prior calibration quarters must be represented")
    y = _labels(past.y_true.to_numpy(), "calibration labels", len(past))
    z = past.raw_logit.to_numpy(float)
    if not np.isfinite(z).all():
        raise ValueError("calibration logits must be finite")
    slope = fit_platt_scaling(logit(_open_probability(expit(z))), y)
    return slope, {
        "n": len(past),
        "slope": slope,
        "forecast_start": start.isoformat(),
        "window_start": window_start.isoformat(),
        "latest_outcome_date": past_dates.max().isoformat(),
        "source_quarters": [str(q) for q in expected_quarters],
        "contract": "prior out-of-sample logits; final base refit includes past calibration outcomes",
    }
