"""TRAIN-only bootstrap learning with STOP-only ensemble checkpoint selection.

Inputs must already be all-odd features: swapping the series sides negates every
column. STOP calibration selects a checkpoint, never the returned forecast's
calibration. The caller separately fits final aggregate calibration on CAL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from scripts.train_and_tune_siamese_series import (
    NumPySiameseMLP,
    _labels,
    _training_matrix,
    fit_platt_scaling,
    focal_loss_and_grad,
)


class _InferenceMember(NumPySiameseMLP):
    """Reuse the shared forward pass, retaining weights but no Adam or fit state."""

    def __init__(self, model: NumPySiameseMLP) -> None:
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = (
            parameter.copy() for parameter in model.params
        )


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real features")
    return _training_matrix(values, name)


def _binary_labels(values: np.ndarray, rows: int) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("labels must contain real binary values")
    return _labels(values, rows)


def _mean_logits(models: Sequence[NumPySiameseMLP], x: np.ndarray) -> np.ndarray:
    total = np.zeros(len(x), dtype=np.float64)
    for model in models:
        total += model.forward_anti_symmetric(x)[0][:, 0]
    total /= len(models)
    if not np.isfinite(total).all():
        raise ValueError("ensemble produced non-finite logits")
    return total


@dataclass
class EarlyStoppedSiameseScorer:
    """Picklable raw mean-logit forecast; scale and member weights are fit state."""

    scale: np.ndarray
    members: tuple[_InferenceMember, ...]

    def logit(self, raw_x: np.ndarray) -> np.ndarray:
        x = _matrix(raw_x, "inference data")
        if x.shape[1] != len(self.scale):
            raise ValueError("inference feature dimensions differ from TRAIN")
        scaled = x / self.scale
        if not np.isfinite(scaled).all():
            raise ValueError("scaled inference data must be finite")
        return _mean_logits(self.members, scaled)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value: float, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (value <= 0.0 if positive else value < 0.0)
    ):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")
    return float(value)


def fit_early_stopped_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_stop: np.ndarray,
    y_stop: np.ndarray,
    *,
    gamma: float,
    seed: int,
    members: int = 5,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 0.001,
) -> tuple[EarlyStoppedSiameseScorer, dict[str, Any]]:
    """Select a complete ensemble by provisionally calibrated STOP LogLoss.

    Each member has one fixed TRAIN bootstrap and independently seeded epoch
    shuffles. An epoch advances all members before scoring the mean raw logit.
    Strictly lower STOP loss wins; ties retain the earlier checkpoint. Patience
    counts consecutive non-improving epochs. No STOP slope is stored in scorer.
    """
    train = _matrix(x_train, "TRAIN data")
    stop = _matrix(x_stop, "STOP data")
    if train.shape[1] != stop.shape[1]:
        raise ValueError("TRAIN and STOP feature dimensions differ")
    labels_train = _binary_labels(y_train, len(train))
    labels_stop = _binary_labels(y_stop, len(stop))
    gamma = _real(gamma, "gamma")
    lr = _real(lr, "lr", positive=True)
    weight_decay = _real(weight_decay, "weight_decay")
    seed = _integer(seed, "seed", minimum=0)
    if seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    members = _integer(members, "members")
    max_epochs = _integer(max_epochs, "max_epochs")
    patience = _integer(patience, "patience")
    batch_size = _integer(batch_size, "batch_size")

    scaler = StandardScaler(with_mean=False)
    train = scaler.fit_transform(train)
    stop = scaler.transform(stop)
    if not np.isfinite(train).all() or not np.isfinite(stop).all():
        raise ValueError("scaled TRAIN and STOP data must be finite")
    if not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0.0).any():
        raise ValueError("TRAIN feature scales must be finite and positive")

    seed_rng = np.random.RandomState(seed)
    member_seeds = [int(value) for value in seed_rng.randint(0, 2**32, size=members)]
    rngs = [np.random.RandomState(member_seed) for member_seed in member_seeds]
    bootstraps = [rng.choice(len(train), size=len(train), replace=True) for rng in rngs]
    models = [
        NumPySiameseMLP(d_in=train.shape[1], seed=member_seed)
        for member_seed in member_seeds
    ]
    history: list[dict[str, int | float]] = []
    best_members: tuple[_InferenceMember, ...] = ()
    best_loss = float("inf")
    best_slope = 1.0
    best_epoch = 0
    stale_epochs = 0
    eps = np.finfo(np.float64).eps

    for epoch in range(1, max_epochs + 1):
        for model, rng, bootstrap in zip(models, rngs, bootstraps, strict=True):
            # Shuffle indices, not copies of each member's full training matrix.
            order = rng.permutation(len(train))
            for start in range(0, len(train), batch_size):
                indices = bootstrap[order[start : start + batch_size]]
                logits, cache = model.forward_anti_symmetric(train[indices])
                loss, grad_z = focal_loss_and_grad(
                    labels_train[indices], logits, gamma=gamma
                )
                if not np.isfinite(loss) or not np.isfinite(grad_z).all():
                    raise ValueError(
                        "training produced non-finite focal loss or gradients"
                    )
                grads = model.backward(grad_z, cache, weight_decay=weight_decay)
                if any(not np.isfinite(gradient).all() for gradient in grads):
                    raise ValueError("training produced non-finite parameter gradients")
                model.step_adam(grads, lr=lr)
                if any(not np.isfinite(parameter).all() for parameter in model.params):
                    raise ValueError("training produced non-finite member parameters")

        raw_logits = _mean_logits(models, stop)
        forecast_logits = logit(np.clip(expit(raw_logits), eps, 1.0 - eps))
        slope = fit_platt_scaling(forecast_logits, labels_stop)
        stop_loss = float(
            log_loss(labels_stop, expit(slope * forecast_logits), labels=[0, 1])
        )
        if not np.isfinite(stop_loss):
            raise ValueError("STOP selection produced non-finite LogLoss")
        history.append(
            {"epoch": epoch, "stop_log_loss": stop_loss, "stop_slope": slope}
        )
        if stop_loss < best_loss:
            best_loss, best_slope, best_epoch = stop_loss, slope, epoch
            best_members = tuple(_InferenceMember(model) for model in models)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    scorer = EarlyStoppedSiameseScorer(scale=scaler.scale_, members=best_members)
    audit: dict[str, Any] = {
        "seed": seed,
        "member_seeds": member_seeds,
        "members": members,
        "gamma": gamma,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_rows": len(train),
        "stop_rows": len(stop),
        "n_features": train.shape[1],
        "hidden_dimensions": [models[0].d_h1, models[0].d_h2],
        "scaler_fit_partition": "TRAIN",
        "scaler_with_mean": False,
        "bootstrap": "one fixed TRAIN sample with replacement per member",
        "member_details": [
            {
                "seed": member_seed,
                "bootstrap_unique_rows": int(np.unique(bootstrap).size),
                "chosen_optimizer_steps": best_epoch
                * ((len(train) + batch_size - 1) // batch_size),
            }
            for member_seed, bootstrap in zip(member_seeds, bootstraps, strict=True)
        ],
        "checkpoint_metric": "binary_log_loss_of_STOP_slope_calibrated_mean_logit_forecast",
        "calibration_input": "logit(clip(expit(mean_raw_logit), machine_eps, 1-machine_eps))",
        "stop_slope_purpose": "checkpoint selection only; final CAL slope is caller-owned",
        "tie_break": "earliest epoch",
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "stopped_early": len(history) < max_epochs,
        "best_stop_log_loss": best_loss,
        "best_stop_slope": best_slope,
        "history": history,
    }
    return scorer, audit
