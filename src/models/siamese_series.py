"""Pure NumPy inference for the EXP-081 Symmetrized Siamese Series Model.

Combines:
1. Exact machine anti-symmetry: z_sym = 0.5 * (f(x) - f(-x)).
2. 5-Member bootstrap-bagged ensemble trained with Focal Loss (gamma=1.0).
3. Descriptive logit spread sigma_z(x) across ensemble members.
4. Heuristic lower scores (P_low = sigmoid(z_mean - kappa * sigma_z)).
   These do not measure shared member bias or provide confidence coverage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.models.symmetric_series import (
    _REQUIRED_BASE_FIELDS,
    build_feature_mapping,
)

MODEL_NAME = "Symmetrized-Siamese-Series-EXP081"
MODEL_VERSION = "exp081-siamese-series-v1"
FEATURE_VERSION = "ratings-w20-symmetric-series-v1"


def load_artifact(path: str | Path) -> "SiameseSeriesModel":
    """Load an explicitly selected historical EXP-081 artifact."""
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SiameseSeriesModel.from_mapping(payload)


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@dataclass(frozen=True)
class SiameseMember:
    """A single calibrated Siamese MLP ensemble member."""

    w1: np.ndarray  # shape (d_in, 32)
    b1: np.ndarray  # shape (32,)
    w2: np.ndarray  # shape (32, 16)
    b2: np.ndarray  # shape (16,)
    w3: np.ndarray  # shape (16, 1)
    b3: float
    platt_slope: float

    def forward_anti_symmetric(self, x: np.ndarray) -> float:
        """Evaluate forward pass with guaranteed machine-level anti-symmetry."""
        # Positive branch: +x
        z1_pos = x @ self.w1 + self.b1
        h1_pos = np.maximum(0.0, z1_pos)
        z2_pos = h1_pos @ self.w2 + self.b2
        h2_pos = np.maximum(0.0, z2_pos)
        out_pos = float((h2_pos @ self.w3)[0] + self.b3)

        # Negative branch: -x
        z1_neg = (-x) @ self.w1 + self.b1
        h1_neg = np.maximum(0.0, z1_neg)
        z2_neg = h1_neg @ self.w2 + self.b2
        h2_neg = np.maximum(0.0, z2_neg)
        out_neg = float((h2_neg @ self.w3)[0] + self.b3)

        # Anti-symmetric raw logit
        z_sym = 0.5 * (out_pos - out_neg)
        # Platt scaled logit
        return float(z_sym * self.platt_slope)


@dataclass(frozen=True)
class SiameseSeriesModel:
    """Bagged Siamese MLP with descriptive member spread, not confidence bounds."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    members: tuple[SiameseMember, ...]
    risk_kappa: float = 0.75

    @classmethod
    def from_mapping(cls, artifact: Mapping[str, object]) -> "SiameseSeriesModel":
        scaler_data = artifact.get("scaler")
        if not isinstance(scaler_data, Mapping):
            raise ValueError("artifact is missing scaler metadata")

        names = tuple(str(v) for v in scaler_data["feature_names"])
        scales = np.array(scaler_data["scales"], dtype=np.float64)
        means_raw = scaler_data.get("means")
        if means_raw is not None:
            means = np.array(means_raw, dtype=np.float64)
        else:
            means = np.zeros(len(scales), dtype=np.float64)

        if not names or len(names) != len(scales) or len(names) != len(means):
            raise ValueError("scaler metadata dimensions mismatch")
        ensemble_data = artifact.get("ensemble")
        if not isinstance(ensemble_data, list) or not ensemble_data:
            raise ValueError("artifact is missing ensemble list")

        members = []
        for item in ensemble_data:
            m = SiameseMember(
                w1=np.array(item["w1"], dtype=np.float64),
                b1=np.array(item["b1"], dtype=np.float64),
                w2=np.array(item["w2"], dtype=np.float64),
                b2=np.array(item["b2"], dtype=np.float64),
                w3=np.array(item["w3"], dtype=np.float64),
                b3=float(item["b3"]),
                platt_slope=float(item["platt_slope"]),
            )
            members.append(m)

        arch = artifact.get("architecture", {})
        risk_kappa = float(arch.get("risk_kappa", 0.75))

        return cls(
            feature_names=names,
            means=means,
            scales=scales,
            members=tuple(members),
            risk_kappa=risk_kappa,
        )

    def _prepare_vector(self, snapshot: Mapping[str, float], *, best_of: int) -> np.ndarray:
        engineered = build_feature_mapping(snapshot, best_of=best_of)
        raw_vals = np.array([float(engineered[name]) for name in self.feature_names], dtype=np.float64)
        return (raw_vals - self.means) / self.scales

    def predict(self, snapshot: Mapping[str, float], *, best_of: int) -> float:
        """Compute the calibrated mean probability P(Team A wins series)."""
        x_scaled = self._prepare_vector(snapshot, best_of=best_of)
        logits = [m.forward_anti_symmetric(x_scaled) for m in self.members]
        z_mean = float(np.mean(logits))
        return _sigmoid(z_mean)

    def predict_with_uncertainty(
        self,
        snapshot: Mapping[str, float],
        *,
        best_of: int,
        kappa: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return mean Side A probability, logit spread, and both sidewise lower scores.

        The risk scores are sigmoid(z_mean - k*sigma_z) and
        sigmoid(-z_mean - k*sigma_z), not complementary probabilities.
        They are conservative decision scores, not statistical confidence bounds.
        """
        k = self.risk_kappa if kappa is None else float(kappa)
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("kappa must be finite and nonnegative")
        x_scaled = self._prepare_vector(snapshot, best_of=best_of)
        logits = [m.forward_anti_symmetric(x_scaled) for m in self.members]
        if not logits or not all(math.isfinite(z) for z in logits):
            raise ValueError("ensemble logits must be nonempty and finite")
        z_mean = float(np.mean(logits))
        sigma_z = float(np.std(logits))
        if not math.isfinite(z_mean) or not math.isfinite(sigma_z) or not math.isfinite(k * sigma_z):
            raise ValueError("ensemble uncertainty must be finite")
        p_mean = _sigmoid(z_mean)
        p_low = _sigmoid(z_mean - k * sigma_z)
        p_low_b = min(1.0 - p_mean, _sigmoid(-z_mean - k * sigma_z))
        return p_mean, sigma_z, p_low, p_low_b
