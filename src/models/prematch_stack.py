"""Side-symmetric learned connector for out-of-time prematch specialists."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.prematch_features import PLAYER_FEATURE_NAMES

_PROBABILITY_CLIP = 1e-6
_RANDOM_SEED = 20260927
def _expert_probabilities(probabilities: Any) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("probabilities must have shape [N, 3] with N > 0")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("expert probabilities must be finite and within [0, 1]")
    return np.clip(values, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)


def _context_values(context: Any, rows: int) -> np.ndarray:
    values = np.asarray(context, dtype=np.float64)
    if values.shape != (rows, 4):
        raise ValueError("context must have shape [N, 4]")
    if not np.isfinite(values).all():
        raise ValueError("context must contain only finite values")
    return values


def _connector_features(probabilities: np.ndarray, context: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    logits = np.log(probabilities) - np.log1p(-probabilities)
    normalized_context = (context - mean) / scale
    return np.concatenate((logits, (logits[:, :, None] * normalized_context[:, None, :]).reshape(len(logits), 12)), axis=1)


class SpecialistStack:
    """Regularized connector: history, LightGBM, then player Gaussian-BT."""

    def __init__(self, C: float = 0.1) -> None:
        try:
            regularization = float(C)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("C must be a finite positive regularization parameter") from exc
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("C must be a finite positive regularization parameter")
        self.C = regularization
        self.context_mean_: np.ndarray | None = None
        self.context_scale_: np.ndarray | None = None
        self.model_: LogisticRegression | None = None

    def fit(
        self,
        probabilities: Any,
        context: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> SpecialistStack:
        experts = _expert_probabilities(probabilities)
        covariates = _context_values(context, len(experts))
        labels = np.asarray(y)
        if labels.shape != (len(experts),) or not np.isin(labels, (0, 1)).all():
            raise ValueError("y must be a length-N vector of binary labels")
        labels = labels.astype(np.int64, copy=False)
        if sample_weight is None:
            weights = None
            class_mass = np.bincount(labels, minlength=2).astype(np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (len(experts),) or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("sample_weight must be a finite nonnegative length-N vector")
            class_mass = np.bincount(labels, weights=weights, minlength=2)
        if (class_mass <= 0).any():
            raise ValueError("training labels must contain both classes with positive weight")

        mean = covariates.mean(axis=0)
        scale = covariates.std(axis=0)
        scale[scale == 0] = 1.0
        features = _connector_features(experts, covariates, mean, scale)
        model = LogisticRegression(
            C=self.C,
            fit_intercept=False,
            solver="lbfgs",
            random_state=_RANDOM_SEED,
            max_iter=1000,
        )
        model.fit(features, labels, sample_weight=weights)
        self.context_mean_ = mean
        self.context_scale_ = scale
        self.model_ = model
        return self

    def predict_proba(self, probabilities: Any, context: Any) -> np.ndarray:
        if self.model_ is None or self.context_mean_ is None or self.context_scale_ is None:
            raise ValueError("SpecialistStack must be fitted before prediction")
        experts = _expert_probabilities(probabilities)
        covariates = _context_values(context, len(experts))
        features = _connector_features(experts, covariates, self.context_mean_, self.context_scale_)
        positive_column = int(np.flatnonzero(self.model_.classes_ == 1)[0])
        result = np.asarray(self.model_.predict_proba(features)[:, positive_column], dtype=np.float64)
        return np.clip(result, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)


def stack_context(batch: Any, probabilities: Any) -> np.ndarray:
    """Return [N,4]: min log support, max log inactivity, mean slow-rating SD, expert-probability SD.

    Player aggregations span both sides and all five roles. Debutants have zero
    support and undefined inactivity: take the maximum observed inactivity,
    or zero when nobody has appeared. Minimum support retains the cold-start
    signal. Other missing or infinite context fields are invalid.
    """
    try:
        player_values = batch["players"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("batch must contain players[N, 2, 5, F]") from exc
    players = np.asarray(player_values, dtype=np.float64)
    expected_features = len(PLAYER_FEATURE_NAMES)
    if players.ndim != 4 or players.shape[1:] != (2, 5, expected_features):
        raise ValueError(f"players must have shape [N, 2, 5, {expected_features}]")
    experts = _expert_probabilities(probabilities)
    if len(players) != len(experts):
        raise ValueError("players and probabilities must have the same row count")

    support_idx = PLAYER_FEATURE_NAMES.index("log_appearance_count")
    inactivity_idx = PLAYER_FEATURE_NAMES.index("log_days_since_last_appearance")
    rating_sd_idx = PLAYER_FEATURE_NAMES.index("slow_rating_sd")
    support = players[..., support_idx]
    inactivity = players[..., inactivity_idx]
    rating_sd = players[..., rating_sd_idx]
    if not (np.isfinite(support).all() and np.isfinite(rating_sd).all()):
        raise ValueError("player support and rating uncertainty must be finite")
    missing_inactivity = np.isnan(inactivity)
    if np.isinf(inactivity).any() or (missing_inactivity & (support != 0)).any():
        raise ValueError("Only players without prior appearances may have undefined inactivity")
    return np.column_stack((
        support.min(axis=(1, 2)),
        np.where(missing_inactivity, 0.0, inactivity).max(axis=(1, 2)),
        rating_sd.mean(axis=(1, 2)),
        experts.std(axis=1),
    )).astype(np.float64, copy=False)


def _iso_dates(values: Any, field: str) -> pd.DatetimeIndex:
    parsed: list[date] = []
    try:
        for value in values:
            if isinstance(value, str):
                parsed.append(date.fromisoformat(value))
            elif isinstance(value, datetime):
                parsed.append(value.date())
            elif isinstance(value, date):
                parsed.append(value)
            else:
                parsed.append(pd.Timestamp(value).date())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must contain valid ISO dates") from exc
    if any(pd.isna(value) for value in parsed):
        raise ValueError(f"{field} must not contain missing dates")
    return pd.DatetimeIndex(parsed)


def stack_training_indices(frame: pd.DataFrame, origin: str) -> np.ndarray:
    """Return rows available strictly before an evaluation origin, rejecting leakage."""
    required = ("match_id", "date", "release_day", "model_origin", "training_cutoff")
    if not isinstance(frame, pd.DataFrame) or any(name not in frame.columns for name in required):
        raise ValueError(f"frame must contain columns: {', '.join(required)}")
    match_ids = frame["match_id"]
    if match_ids.isna().any() or match_ids.duplicated().any():
        raise ValueError("match_id values must be present and unique")

    dates = {name: _iso_dates(frame[name].tolist(), name) for name in required[1:]}
    origin_date = _iso_dates([origin], "origin")[0]
    target_date = dates["date"]
    model_origin = dates["model_origin"]
    training_cutoff = dates["training_cutoff"]
    release_day = dates["release_day"]
    if ((training_cutoff >= model_origin) | (model_origin > target_date)).any():
        raise ValueError("purported forecast chronology must satisfy training_cutoff < model_origin <= date")
    if "calibration_last_label_release" in frame.columns:
        calibration_release = _iso_dates(frame["calibration_last_label_release"].tolist(), "calibration_last_label_release")
        if (calibration_release >= model_origin).any():
            raise ValueError("calibration labels must be released strictly before model_origin")
    return np.asarray((target_date < origin_date) & (release_day < origin_date), dtype=bool)
