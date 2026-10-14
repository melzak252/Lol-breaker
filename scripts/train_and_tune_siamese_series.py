#!/usr/bin/env python3
"""Train explicitly experimental Siamese ensembles from complete canonical snapshots.

This script implements:
1. Pure NumPy training with vectorised Adam optimizer (100% portable, no PyTorch needed).
2. Guaranteed machine anti-symmetry: z_sym(x) = 0.5 * (f(x) - f(-x)).
3. Focal Loss with configurable focusing parameter gamma.
4. Bootstrap bagging ensemble (K members) with epistemic uncertainty estimation.
5. Post-hoc Platt scaling / temperature calibration on validation split.
6. Export a new, non-promoted artifact compatible with the runtime loader.
7. Optional development-only hyperparameter grid (--tune).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.symmetric_series import (
    FEATURE_VERSION,
    _REQUIRED_BASE_FIELDS,
    build_feature_mapping,
)

EXPERIMENT_VERSION = "experimental-canonical79-siamese-v1"
TEMPORAL_LIMITATIONS = (
    "Dates partition supplied rows only; this script does not establish point-in-time "
    "availability, rating-state provenance, series grouping, or rolling-window leakage freedom. "
    "Validation metrics reuse calibration rows and are not unbiased model-selection evidence. "
    "No superiority, operational qualification, or active-model promotion is implied."
)


def parse_args() -> argparse.Namespace:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = PROJECT_ROOT / "data/artifacts" / f"{EXPERIMENT_VERSION}-{run_id}"
    parser = argparse.ArgumentParser(
        description="Train and tune Symmetrized Siamese Series neural ensemble."
    )
    parser.add_argument(
        "--ratings-data",
        type=Path,
        default=PROJECT_ROOT / "data/golgg_y_predicts.csv",
        help="CSV of canonical raw snapshots with explicit best_of, date and binary label",
    )
    parser.add_argument(
        "--rolling-data",
        type=Path,
        default=PROJECT_ROOT / "data/golgg_rolling_stats.csv",
        help="Path to golgg_rolling_stats.csv with W20 features",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=output_dir / "model.json",
        help="New experimental artifact path; existing files are never overwritten",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=output_dir / "summary.json",
        help="New training summary path; existing files are never overwritten",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2025-01-01",
        help="Calibration/test boundary (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--val-date",
        type=str,
        default="2024-01-01",
        help="Train/calibration boundary (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of training epochs per ensemble member",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
        help="Focal Loss focusing parameter gamma (0.0 = standard BCE)",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=5,
        help="Number of bootstrap ensemble members (K)",
    )
    parser.add_argument(
        "--d-h1",
        type=int,
        default=32,
        help="Hidden layer 1 dimension",
    )
    parser.add_argument(
        "--d-h2",
        type=int,
        default=16,
        help="Hidden layer 2 dimension",
    )
    parser.add_argument(
        "--risk-kappa",
        type=float,
        default=0.75,
        help="Risk adjustment factor kappa for P_low = sigma(z - kappa * sigma_z)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Development grid using calibration rows for selection (not unbiased evidence)",
    )
    return parser.parse_args()


def merge_training_snapshots(
    ratings: pd.DataFrame, rolling: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join raw snapshots one-to-one without changing ratings side order or dropping rows."""
    key = "golgg_match_id"
    for name, frame in (("ratings", ratings), ("rolling", rolling)):
        if frame is None:
            continue
        if not frame.columns.is_unique:
            raise ValueError(f"{name} has duplicate columns")
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name} requires unique, non-null {key}")
    if rolling is None:
        return ratings.copy()
    overlap = set(ratings.columns) & set(rolling.columns) - {key}
    # Rename the right-hand columns explicitly so arbitrary input suffixes cannot collide.
    renamed = {column: f"__rolling_{column}" for column in rolling if column != key}
    if (
        any(column in ratings for column in renamed.values())
        or "__merge_status" in ratings
    ):
        raise ValueError("input uses reserved merge columns")
    merged = ratings.merge(
        rolling.rename(columns=renamed),
        on=key,
        how="left",
        sort=False,
        validate="one_to_one",
        indicator="__merge_status",
    )
    if not merged["__merge_status"].eq("both").all():
        missing = merged.loc[merged["__merge_status"] != "both", key].tolist()
        raise ValueError(f"missing rolling snapshots for {key}: {missing[:10]}")
    checked = overlap & (
        _REQUIRED_BASE_FIELDS
        | {
            "team1_id",
            "team2_id",
            "team1_name",
            "team2_name",
            "date",
            "best_of",
            "BoN",
            "y_true",
            "y_team1",
        }
    )
    for column in sorted(checked):
        left, right = merged[column], merged[renamed[column]]
        if column == "date":
            left = pd.to_datetime(left, errors="raise", utc=True)
            right = pd.to_datetime(right, errors="raise", utc=True)
        if left.isna().any() or right.isna().any() or not left.eq(right).all():
            raise ValueError(
                f"ratings/rolling orientation or value mismatch in {column}"
            )
    for column in rolling:
        if column != key and column not in ratings:
            merged[column] = merged[renamed[column]]
    return merged.drop(columns=[*renamed.values(), "__merge_status"])


def build_training_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the runtime's canonical 79 features, exclusively from explicit raw inputs."""
    if not frame.columns.is_unique:
        raise ValueError("snapshot has duplicate columns")
    required = _REQUIRED_BASE_FIELDS | {"best_of"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing required raw snapshot columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("snapshot dataset is empty")
    names: list[str] = []
    matrix = None
    columns = sorted(required | ({"BoN"} & set(frame.columns)))
    rows = frame[columns].itertuples(index=False, name=None)
    for position, raw_values in enumerate(rows):
        row = dict(zip(columns, raw_values))
        best_of = row["best_of"]
        if (
            pd.isna(best_of)
            or isinstance(best_of, (bool, np.bool_))
            or best_of not in (1, 3, 5)
        ):
            raise ValueError(f"row {position}: explicit best_of must be 1, 3 or 5")
        if "BoN" in row and (pd.isna(row["BoN"]) or row["BoN"] != best_of):
            raise ValueError(f"row {position}: BoN conflicts with explicit best_of")
        try:
            values = build_feature_mapping(row, best_of=int(best_of))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {position}: {exc}") from exc
        if matrix is None:
            names = list(values)
            matrix = np.empty((len(frame), len(names)), dtype=np.float64)
        matrix[position] = [values[name] for name in names]
    return matrix, names


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector or (N, 1) column")
    return array


def _labels(values: np.ndarray, rows: int) -> np.ndarray:
    labels = _vector(values, "labels")
    if len(labels) != rows or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("labels must contain exactly one binary value per row")
    return labels


def _training_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a nonempty finite feature matrix")
    return matrix


class NumPySiameseMLP:
    """A two-hidden-layer MLP with exact machine anti-symmetry and Adam optimizer."""

    def __init__(
        self,
        d_in: int,
        d_h1: int = 32,
        d_h2: int = 16,
        seed: int = 42,
    ) -> None:
        if any(
            not isinstance(size, (int, np.integer)) or size < 1
            for size in (d_in, d_h1, d_h2)
        ):
            raise ValueError("network dimensions must be positive integers")
        self.d_in = d_in
        self.d_h1 = d_h1
        self.d_h2 = d_h2
        rng = np.random.RandomState(seed)

        # He / Kaiming normal initialization
        self.W1 = rng.randn(d_in, d_h1) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_h1)
        self.W2 = rng.randn(d_h1, d_h2) * np.sqrt(2.0 / d_h1)
        self.b2 = np.zeros(d_h2)
        self.W3 = rng.randn(d_h2, 1) * np.sqrt(2.0 / d_h2)
        self.b3 = np.zeros(1)

        self.platt_slope: float = 1.0

        # Adam optimizer state
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward_anti_symmetric(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward pass enforcing z_sym = 0.5 * (f(X) - f(-X))."""
        # Positive branch: +X
        z1_pos = X @ self.W1 + self.b1
        h1_pos = np.maximum(0.0, z1_pos)
        z2_pos = h1_pos @ self.W2 + self.b2
        h2_pos = np.maximum(0.0, z2_pos)
        out_pos = h2_pos @ self.W3 + self.b3

        # Negative branch: -X
        z1_neg = (-X) @ self.W1 + self.b1
        h1_neg = np.maximum(0.0, z1_neg)
        z2_neg = h1_neg @ self.W2 + self.b2
        h2_neg = np.maximum(0.0, z2_neg)
        out_neg = h2_neg @ self.W3 + self.b3

        z_sym = 0.5 * (out_pos - out_neg)

        cache = {
            "X": X,
            "z1_pos": z1_pos,
            "h1_pos": h1_pos,
            "z2_pos": z2_pos,
            "h2_pos": h2_pos,
            "z1_neg": z1_neg,
            "h1_neg": h1_neg,
            "z2_neg": z2_neg,
            "h2_neg": h2_neg,
        }
        return z_sym, cache

    def backward(
        self,
        grad_z: np.ndarray,
        cache: dict[str, np.ndarray],
        weight_decay: float = 1e-4,
    ) -> list[np.ndarray]:
        """Analytical backpropagation through symmetric difference."""
        X = cache["X"]
        h1_pos, h2_pos = cache["h1_pos"], cache["h2_pos"]
        h1_neg, h2_neg = cache["h1_neg"], cache["h2_neg"]
        z1_pos, z2_pos = cache["z1_pos"], cache["z2_pos"]
        z1_neg, z2_neg = cache["z1_neg"], cache["z2_neg"]

        N = len(X)
        d_out_pos = 0.5 * grad_z
        d_out_neg = -0.5 * grad_z

        # Branch gradients for W3, b3
        dW3 = (h2_pos.T @ d_out_pos + h2_neg.T @ d_out_neg) / N + weight_decay * self.W3
        db3 = (np.sum(d_out_pos, axis=0) + np.sum(d_out_neg, axis=0)) / N

        # Layer 2
        dh2_pos = d_out_pos @ self.W3.T
        dh2_neg = d_out_neg @ self.W3.T

        dz2_pos = dh2_pos * (z2_pos > 0)
        dz2_neg = dh2_neg * (z2_neg > 0)

        dW2 = (h1_pos.T @ dz2_pos + h1_neg.T @ dz2_neg) / N + weight_decay * self.W2
        db2 = (np.sum(dz2_pos, axis=0) + np.sum(dz2_neg, axis=0)) / N

        # Layer 1
        dh1_pos = dz2_pos @ self.W2.T
        dh1_neg = dz2_neg @ self.W2.T

        dz1_pos = dh1_pos * (z1_pos > 0)
        dz1_neg = dh1_neg * (z1_neg > 0)

        dW1 = (X.T @ dz1_pos - X.T @ dz1_neg) / N + weight_decay * self.W1
        db1 = (np.sum(dz1_pos, axis=0) + np.sum(dz1_neg, axis=0)) / N

        return [dW1, db1, dW2, db2, dW3, db3]

    def step_adam(
        self,
        grads: list[np.ndarray],
        lr: float = 0.003,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Update weights using Adam optimizer."""
        self.t += 1
        params = self.params
        for i in range(len(params)):
            g = grads[i]
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * (g**2)

            m_hat = self.m[i] / (1.0 - beta1**self.t)
            v_hat = self.v[i] / (1.0 - beta2**self.t)

            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def focal_loss_and_grad(
    y_true: np.ndarray,
    z_raw: np.ndarray,
    gamma: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Stable focal loss and per-row derivative; backward performs mean reduction."""
    z = _vector(z_raw, "logits")
    y = _labels(y_true, len(z))
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("focal gamma must be finite and nonnegative")
    signed = (2.0 * y - 1.0) * z
    log_p_t = -np.logaddexp(0.0, -signed)
    p_t = expit(signed)
    one_minus_p_t = expit(-signed)
    modulation = one_minus_p_t**gamma
    loss = float(np.mean(-modulation * log_p_t))
    # Includes the derivative of the focal modulating factor, not just weighted BCE.
    gradient = (2.0 * y - 1.0) * modulation * (gamma * p_t * log_p_t - one_minus_p_t)
    return loss, gradient.reshape(np.asarray(z_raw).shape)


def fit_platt_scaling(z_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit a positive zero-intercept slope using only the provided calibration rows.

    Optimize log-slope over a broad numerical range after normalizing logits, rather
    than imposing the old [0.5, 1.5] grid. Boundary optima (separable or anti-predictive
    samples) approach infinite or zero temperature; bounds keep artifacts finite.
    """
    z = _vector(z_val, "calibration logits")
    y = _labels(y_val, len(z))
    magnitude = float(np.max(np.abs(z)))
    if magnitude == 0.0:
        return 1.0
    normalized = z / magnitude
    signed = (2.0 * y - 1.0) * normalized

    def objective(log_slope: float) -> float:
        return float(np.mean(np.logaddexp(0.0, -signed * np.exp(log_slope))))

    result = minimize_scalar(
        objective,
        bounds=(-20.0, 20.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("positive-slope calibration optimization failed")
    slope = float(np.exp(result.x) / magnitude)
    if not np.isfinite(slope) or slope <= 0.0:
        raise ValueError("calibration produced a non-finite or non-positive slope")
    return slope


def train_single_member(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.003,
    gamma: float = 1.0,
    d_h1: int = 32,
    d_h2: int = 16,
    seed: int = 42,
    weight_decay: float = 1e-4,
    bootstrap_fraction: float = 1.0,
) -> NumPySiameseMLP:
    """Train with seeded replacement sampling and member calibration.

    Defaults retain the historical full-size bootstrap and L2 gradient. The
    objective adds weight_decay / 2 * sum(W**2), excluding all biases.
    """
    X_train = _training_matrix(X_train, "training data")
    X_val = _training_matrix(X_val, "calibration data")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("training and calibration feature dimensions differ")
    y_train = _labels(y_train, len(X_train))
    y_val = _labels(y_val, len(X_val))
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch_size and learning rate must be positive")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("focal gamma must be finite and nonnegative")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not np.isfinite(bootstrap_fraction) or not 0 < bootstrap_fraction <= 1:
        raise ValueError("bootstrap_fraction must be finite and in (0, 1]")
    rng = np.random.RandomState(seed)
    n_train = len(X_train)
    n_boot = max(1, int(n_train * bootstrap_fraction))
    boot_idx = rng.choice(n_train, size=n_boot, replace=True)
    X_boot = X_train[boot_idx]
    y_boot = y_train[boot_idx]

    model = NumPySiameseMLP(d_in=X_train.shape[1], d_h1=d_h1, d_h2=d_h2, seed=seed)

    for epoch in range(epochs):
        perm = rng.permutation(n_boot)
        for i in range(0, n_boot, batch_size):
            batch_idx = perm[i : i + batch_size]
            X_b = X_boot[batch_idx]
            y_b = y_boot[batch_idx]

            z_b, cache = model.forward_anti_symmetric(X_b)
            loss, grad_z = focal_loss_and_grad(y_b, z_b, gamma=gamma)
            grads = model.backward(grad_z, cache, weight_decay=weight_decay)
            model.step_adam(grads, lr=lr)

    # Fit Platt scaling slope on validation set
    z_val, _ = model.forward_anti_symmetric(X_val)
    model.platt_slope = fit_platt_scaling(z_val, y_val)
    return model


def evaluate_ensemble(
    members: list[NumPySiameseMLP],
    X: np.ndarray,
    y: np.ndarray,
    risk_kappa: float = 0.75,
) -> dict[str, float | None]:
    """Evaluate mean predictions; undefined cohort metrics are null, never fabricated."""
    X = _training_matrix(X, "evaluation data")
    y = _labels(y, len(X))
    if not members:
        raise ValueError("ensemble must contain at least one member")
    if not np.isfinite(risk_kappa) or risk_kappa < 0:
        raise ValueError("risk_kappa must be finite and nonnegative")
    logits = np.stack(
        [m.forward_anti_symmetric(X)[0][:, 0] * m.platt_slope for m in members]
    )
    if not np.isfinite(logits).all():
        raise ValueError("ensemble produced non-finite logits")
    z_mean = np.mean(logits, axis=0)
    z_std = np.std(logits, axis=0, ddof=0)
    probabilities = expit(z_mean)
    coin_mask = (probabilities >= 0.45) & (probabilities <= 0.55)
    losses = np.logaddexp(0.0, -(2.0 * y - 1.0) * z_mean)
    return {
        "log_loss": float(np.mean(losses)),
        "brier": float(brier_score_loss(y, probabilities)),
        "roc_auc": (
            float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None
        ),
        "accuracy": float(accuracy_score(y, (probabilities >= 0.5).astype(int))),
        "coin_flip_log_loss": (
            float(np.mean(losses[coin_mask])) if coin_mask.any() else None
        ),
        "mean_sigma_z": float(np.mean(z_std)),
        "mean_p_low_a": float(np.mean(expit(z_mean - risk_kappa * z_std))),
        "mean_p_low_b": float(np.mean(expit(-z_mean - risk_kappa * z_std))),
    }


def build_model_artifact(
    members: list[NumPySiameseMLP],
    feature_names: list[str],
    scales: np.ndarray,
    *,
    risk_kappa: float = 0.75,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize the exact runtime schema without impersonating a frozen experiment."""
    canonical_names = list(
        build_feature_mapping({name: 0.5 for name in _REQUIRED_BASE_FIELDS}, best_of=1)
    )
    if list(feature_names) != canonical_names:
        raise ValueError("export requires canonical feature order")
    scales = _vector(scales, "scales")
    if len(scales) != len(feature_names) or np.any(scales <= 0):
        raise ValueError("export requires one positive scale per feature")
    if not members:
        raise ValueError("cannot export an empty ensemble")
    if not np.isfinite(risk_kappa) or risk_kappa < 0:
        raise ValueError("risk_kappa must be finite and nonnegative")
    for member in members:
        if member.d_in != len(feature_names):
            raise ValueError("ensemble input dimension differs from canonical features")
        if not all(np.isfinite(parameter).all() for parameter in member.params):
            raise ValueError("cannot export non-finite model weights")
        if not np.isfinite(member.platt_slope) or member.platt_slope <= 0:
            raise ValueError("cannot export non-positive or non-finite calibration")
    return {
        "model_name": "Experimental-Canonical79-Siamese-Series",
        "model_version": EXPERIMENT_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "experimental_not_qualified",
        "prediction_semantics": "direct_series_probability",
        "series_projection_count": 0,
        "ensemble_aggregation": "sigmoid(mean(member_platt_slope * antisymmetric_logit))",
        "temporal_limitations": TEMPORAL_LIMITATIONS,
        "provenance": provenance or {},
        "architecture": {
            "family": "siamese_mlp_bagged",
            "d_in": len(feature_names),
            "d_h1": members[0].d_h1,
            "d_h2": members[0].d_h2,
            "activation": "relu",
            "symmetry": "z(x) = 0.5 * (f(x) - f(-x)); no centering",
            "risk_kappa": risk_kappa,
            "n_members": len(members),
            "uncertainty_ddof": 0,
        },
        "scaler": {
            "feature_names": list(feature_names),
            "means": np.zeros(len(feature_names)).tolist(),
            "scales": scales.tolist(),
        },
        "ensemble": [
            {
                "w1": member.W1.tolist(),
                "b1": member.b1.tolist(),
                "w2": member.W2.tolist(),
                "b2": member.b2.tolist(),
                "w3": member.W3.tolist(),
                "b3": float(member.b3[0]),
                "platt_slope": float(member.platt_slope),
            }
            for member in members
        ],
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Fail closed on collisions, including a concurrent writer or existing symlink."""
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")


def _source_metadata(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def main() -> int:
    args = parse_args()
    print(f"=== {EXPERIMENT_VERSION}: diagnostic training only ===")
    print(TEMPORAL_LIMITATIONS)
    if args.n_members < 1:
        raise ValueError("n_members must be positive")
    outputs = (
        [args.output_report] if args.tune else [args.output_model, args.output_report]
    )
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("model and report output paths must be distinct")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    ratings = pd.read_csv(args.ratings_data)
    sources = {"ratings": _source_metadata(args.ratings_data)}
    rolling = None
    missing_raw = (_REQUIRED_BASE_FIELDS | {"best_of"}) - set(ratings.columns)
    if missing_raw:
        if not args.rolling_data.is_file():
            raise ValueError(
                f"missing raw columns {sorted(missing_raw)}; rolling data unavailable"
            )
        rolling = pd.read_csv(args.rolling_data)
        sources["rolling"] = _source_metadata(args.rolling_data)
    frame = merge_training_snapshots(ratings, rolling)
    X_all, feature_names = build_training_features(frame)
    if "date" not in frame:
        raise ValueError("snapshots require explicit date for chronological splits")
    dates = pd.to_datetime(frame["date"], errors="raise", utc=True)
    if dates.isna().any():
        raise ValueError("snapshot dates must not be missing")
    label_columns = [name for name in ("y_true", "y_team1") if name in frame]
    if not label_columns:
        raise ValueError("snapshots require a binary y_true or y_team1 label")
    y_all = _labels(frame[label_columns[0]].to_numpy(), len(frame))
    for name in label_columns[1:]:
        if not np.array_equal(y_all, _labels(frame[name].to_numpy(), len(frame))):
            raise ValueError("y_true and y_team1 label orientations conflict")
    val_date = pd.to_datetime(args.val_date, errors="raise", utc=True)
    split_date = pd.to_datetime(args.split_date, errors="raise", utc=True)
    if pd.isna(val_date) or pd.isna(split_date) or val_date >= split_date:
        raise ValueError(
            "train/calibration boundary must precede calibration/test boundary"
        )
    masks = {
        "train": (dates < val_date).to_numpy(),
        "calibration": ((dates >= val_date) & (dates < split_date)).to_numpy(),
        "test": (dates >= split_date).to_numpy(),
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("train, calibration and test partitions must all contain rows")
    scale = np.std(X_all[masks["train"]], axis=0)
    scale[scale == 0.0] = 1.0
    X_scaled = X_all / scale
    partitions = {name: (X_scaled[mask], y_all[mask]) for name, mask in masks.items()}
    split_metadata = {
        name: {
            "rows": int(mask.sum()),
            "first_date": dates[mask].min().isoformat(),
            "last_date": dates[mask].max().isoformat(),
        }
        for name, mask in masks.items()
    }
    provenance = {
        "sources": sources,
        "row_count": len(frame),
        "dropped_rows": 0,
        "split": split_metadata,
        "val_date": val_date.isoformat(),
        "split_date": split_date.isoformat(),
        "scaler_fit": "training rows only; zero centering to preserve antisymmetry",
        "calibration_fit": "provided calibration partition only; positive zero-intercept slope",
        "temporal_audit": "not established by this training script",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "gamma": args.focal_gamma,
            "d_h1": args.d_h1,
            "d_h2": args.d_h2,
            "n_members": args.n_members,
            "risk_kappa": args.risk_kappa,
        },
        "seeds": [args.seed + index * 101 for index in range(args.n_members)],
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    print(
        f"Canonical feature dimension: {len(feature_names)}; splits: {split_metadata}"
    )
    X_train, y_train = partitions["train"]
    X_val, y_val = partitions["calibration"]
    if args.tune:
        results = []
        for gamma in (0.0, 0.5, 1.0, 1.5, 2.0):
            for lr in (0.001, 0.003, 0.005):
                member = train_single_member(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=lr,
                    gamma=gamma,
                    d_h1=args.d_h1,
                    d_h2=args.d_h2,
                    seed=args.seed,
                )
                results.append(
                    {
                        "gamma": gamma,
                        "lr": lr,
                        "calibration_metrics": evaluate_ensemble(
                            [member], X_val, y_val
                        ),
                        "platt_slope": member.platt_slope,
                    }
                )
        results.sort(key=lambda row: row["calibration_metrics"]["log_loss"])
        report = {
            "experiment": EXPERIMENT_VERSION,
            "status": "development_grid_not_unbiased_selection",
            "temporal_limitations": TEMPORAL_LIMITATIONS,
            "provenance": provenance,
            "grid_results": results,
            "test_evaluated": False,
        }
        write_json_exclusive(args.output_report, report)
        print(f"Development-only grid report: {args.output_report}")
        return 0

    started = time.perf_counter()
    members = [
        train_single_member(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.focal_gamma,
            d_h1=args.d_h1,
            d_h2=args.d_h2,
            seed=seed,
        )
        for seed in provenance["seeds"]
    ]
    metrics = {
        name: evaluate_ensemble(members, X, y, risk_kappa=args.risk_kappa)
        for name, (X, y) in partitions.items()
    }
    artifact = build_model_artifact(
        members,
        feature_names,
        scale,
        risk_kappa=args.risk_kappa,
        provenance=provenance,
    )
    artifact["architecture"]["loss"] = f"focal_loss_gamma_{args.focal_gamma}"
    artifact["metrics"] = metrics
    report = {
        "experiment": EXPERIMENT_VERSION,
        "status": "experimental_not_qualified",
        "temporal_limitations": TEMPORAL_LIMITATIONS,
        "provenance": provenance,
        "metrics": metrics,
        "model_path": str(args.output_model.resolve()),
        "training_and_evaluation_seconds": time.perf_counter() - started,
    }
    write_json_exclusive(args.output_model, artifact)
    write_json_exclusive(args.output_report, report)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Exported experimental artifact (not promoted): {args.output_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
