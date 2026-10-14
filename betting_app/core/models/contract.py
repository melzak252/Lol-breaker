"""Contract definitions for the unified model registry and prediction engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Specification and runtime descriptor for an ML/rating prediction model."""

    name: str
    version: str
    feature_version: str
    ratings_version: str = "latest-full"
    w20_version: str = "w20-latest"
    prediction_target: str = "series"  # "series" (BoN) or "map"
    has_uncertainty: bool = False
    family: str = "native_c0"  # "native_c0", "linear_symmetric", "elastic_net", etc.
    artifact_path: Path | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def target(self) -> str:
        return self.prediction_target

    @property
    def has_epistemic_uncertainty(self) -> bool:
        return self.has_uncertainty

    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "feature_version": self.feature_version,
            "ratings_version": self.ratings_version,
            "w20_version": self.w20_version,
            "prediction_target": self.prediction_target,
            "target": self.prediction_target,
            "has_uncertainty": self.has_uncertainty,
            "has_epistemic_uncertainty": self.has_uncertainty,
            "family": self.family,
            "description": self.description,
        }

@dataclass(frozen=True)
class HybridSpec:
    """Specification for blending a base prediction model with market odds."""

    base_model: ModelSpec
    hybrid_model_name: str = "Hybrid-Operational-Market"
    alpha: float = 0.50  # sports-model weight; market weight is 1 - alpha
    temperature: float = 1.0
    blending_mode: str = "logit_shrinkage"  # "logit_shrinkage" or "linear"
    custom_version: str | None = None

    @property
    def hybrid_model_version(self) -> str:
        if self.custom_version is not None:
            return self.custom_version
        return f"{self.base_model.version}-a{self.alpha:.2f}-t{self.temperature:.2f}"

    def identifier(self) -> str:
        return f"{self.hybrid_model_name}:{self.hybrid_model_version}"


@dataclass(frozen=True)
class UnifiedPredictionResult:
    """Immutable, unified prediction result returned by the PredictionEngine."""

    canonical_match_id: int | None
    prob_a: float  # Calibrated probability of Team A winning the entire match/series
    prob_b: float  # Calibrated probability of Team B winning the entire match/series
    map_prob_a: float | None = None  # Single-map probability if available/applicable
    map_prob_b: float | None = None
    p_low_a: float | None = None  # Conservative lower bound for EV gating
    p_low_b: float | None = None
    epistemic_sigma_z: float | None = None  # Epistemic uncertainty std dev
    model_name: str = ""
    model_version: str = ""
    feature_version: str = ""
    best_of: int = 1
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.prob_a <= 1.0):
            raise ValueError(f"prob_a must be in [0, 1], got {self.prob_a}")
        if not (0.0 <= self.prob_b <= 1.0):
            raise ValueError(f"prob_b must be in [0, 1], got {self.prob_b}")
        if not math.isclose(self.prob_a + self.prob_b, 1.0, abs_tol=1e-3):
            raise ValueError(f"prob_a + prob_b must equal 1.0, got {self.prob_a + self.prob_b}")
        uncertainty = (self.p_low_a, self.p_low_b, self.epistemic_sigma_z)
        if self.diagnostics.get("uncertainty_required") or any(value is not None for value in uncertainty):
            if any(value is None or not math.isfinite(value) for value in uncertainty):
                raise ValueError("both side bounds and finite epistemic uncertainty are required")
            if self.epistemic_sigma_z < 0:
                raise ValueError("epistemic uncertainty must be nonnegative")
            if not (0 <= self.p_low_a <= self.prob_a and 0 <= self.p_low_b <= self.prob_b):
                raise ValueError("side bounds must lie between zero and their own means")
