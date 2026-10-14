"""Pure inference for the leakage-audited EXP-078 series model.

The model uses only pre-match ratings, rating uncertainty, rest, W20 form, and
series format. Every engineered feature changes sign when the teams are
swapped. A zero-intercept linear logit and slope-only calibration therefore
preserve ``P(A) + P(B) == 1`` by construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Mapping

MODEL_NAME = "Symmetric-General-Series-EXP078"
MODEL_VERSION = "exp078-symmetric-series-v1"
FEATURE_VERSION = "ratings-w20-symmetric-series-v1"
ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "betting_app"
    / "models"
    / "exp078_symmetric_series_v1.json"
)

_SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")
_PROBABILITY_FIELDS = tuple(
    field for system in _SYSTEMS for field in (f"team_{system}", f"player_{system}")
)
_RAW_DIFFERENCES = (
    ("team_elo_diff", "team_elo_r1", "team_elo_r2"),
    ("team_glicko_diff", "team_gl_r1", "team_gl_r2"),
    ("team_glicko_rd_diff", "team_gl_rd2", "team_gl_rd1"),
    ("player_elo_weak_diff", "player_elo_min1", "player_elo_min2"),
    ("player_glicko_peak_diff", "player_gl_max1", "player_gl_max2"),
    ("player_glicko_rd_diff", "player_gl_rd_avg2", "player_gl_rd_avg1"),
    ("team_ts_mu_diff", "team_ts_mu1", "team_ts_mu2"),
    ("team_ts_sigma_diff", "team_ts_sigma2", "team_ts_sigma1"),
    ("player_ts_sigma_diff", "player_ts_sigma_avg2", "player_ts_sigma_avg1"),
    ("team_os_mu_diff", "team_os_mu1", "team_os_mu2"),
    ("team_os_sigma_diff", "team_os_sigma2", "team_os_sigma1"),
    ("player_os_sigma_diff", "player_os_sigma_avg2", "player_os_sigma_avg1"),
    ("team_pl_mu_diff", "team_pl_mu1", "team_pl_mu2"),
    ("team_pl_sigma_diff", "team_pl_sigma2", "team_pl_sigma1"),
    ("player_pl_sigma_diff", "player_pl_sigma_avg2", "player_pl_sigma_avg1"),
    ("team_tm_mu_diff", "team_tm_mu1", "team_tm_mu2"),
    ("team_tm_sigma_diff", "team_tm_sigma2", "team_tm_sigma1"),
    ("rest_days_diff", "days_since_last_2", "days_since_last_1"),
)
_W20_FIELDS = (
    "win_rate",
    "kills",
    "deaths",
    "gd15",
    "dpm",
    "vspm",
    "towers",
    "nashors",
    "gold",
    "duration",
)
_REQUIRED_BASE_FIELDS = frozenset(
    (*_PROBABILITY_FIELDS,
     *(left for _, left, _ in _RAW_DIFFERENCES),
     *(right for _, _, right in _RAW_DIFFERENCES),
     *(f"t1_rolling_{field}" for field in _W20_FIELDS),
     *(f"t2_rolling_{field}" for field in _W20_FIELDS))
)


def _clip_probability(value: float) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"rating probability must be finite and in [0, 1], got {value!r}")
    return min(0.999, max(0.001, probability))


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _independent_series_probability(probability: float, best_of: int) -> float:
    p = _clip_probability(probability)
    if best_of == 1:
        return p
    if best_of == 3:
        return p * p * (3.0 - 2.0 * p)
    return p**3 * (10.0 - 15.0 * p + 6.0 * p * p)


def build_feature_mapping(snapshot: Mapping[str, float], *, best_of: int) -> dict[str, float]:
    """Build the exact anti-symmetric feature mapping used by EXP-078."""

    if best_of not in {1, 3, 5}:
        raise ValueError(f"best_of must be one of 1, 3, 5; got {best_of!r}")
    missing = sorted(_REQUIRED_BASE_FIELDS.difference(snapshot))
    if missing:
        raise ValueError("missing required base features: " + ", ".join(missing))

    base: dict[str, float] = {}
    probability_logits: dict[str, float] = {}
    for field in _PROBABILITY_FIELDS:
        value = _logit(snapshot[field])
        name = f"{field}_logit"
        base[name] = value
        probability_logits[name] = value

    for name, left, right in _RAW_DIFFERENCES:
        base[name] = float(snapshot[left]) - float(snapshot[right])
    for field in _W20_FIELDS[:7]:
        base[f"w20_{field}_diff"] = (
            float(snapshot[f"t1_rolling_{field}"])
            - float(snapshot[f"t2_rolling_{field}"])
        )

    consensus = sum(probability_logits.values()) / len(probability_logits)
    base["rating_consensus_logit"] = consensus
    bo3 = float(best_of == 3)
    bo5 = float(best_of == 5)
    for name, value in (*probability_logits.items(), ("rating_consensus_logit", consensus)):
        base[f"{name}_bo3"] = value * bo3
        base[f"{name}_bo5"] = value * bo5

    for field in _W20_FIELDS[7:]:
        base[f"w20_{field}_diff"] = (
            float(snapshot[f"t1_rolling_{field}"])
            - float(snapshot[f"t2_rolling_{field}"])
        )
    for field in _PROBABILITY_FIELDS:
        base[f"{field}_series_amplification"] = (
            _logit(_independent_series_probability(snapshot[field], best_of))
            - probability_logits[f"{field}_logit"]
        )

    non_finite = sorted(name for name, value in base.items() if not math.isfinite(value))
    if non_finite:
        raise ValueError("non-finite engineered features: " + ", ".join(non_finite))
    return base


@dataclass(frozen=True)
class SymmetricSeriesModel:
    """Zero-intercept standardized logistic model with slope-only calibration."""

    feature_names: tuple[str, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    logit_calibration_slope: float

    @classmethod
    def from_mapping(cls, artifact: Mapping[str, object]) -> "SymmetricSeriesModel":
        estimator = artifact.get("estimator")
        if not isinstance(estimator, Mapping):
            raise ValueError("artifact is missing estimator metadata")
        names = tuple(str(value) for value in estimator["feature_names"])
        scales = tuple(float(value) for value in estimator["scales"])
        coefficients = tuple(float(value) for value in estimator["coefficients"])
        slope = float(estimator["logit_calibration_slope"])
        if not names or len(names) != len(scales) or len(names) != len(coefficients):
            raise ValueError("artifact feature, scale, and coefficient lengths differ")
        if len(set(names)) != len(names):
            raise ValueError("artifact contains duplicate feature names")
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("artifact scales must be finite and positive")
        if any(not math.isfinite(value) for value in coefficients):
            raise ValueError("artifact coefficients must be finite")
        if not math.isfinite(slope) or slope <= 0.0:
            raise ValueError("artifact calibration slope must be finite and positive")
        return cls(names, scales, coefficients, slope)

    @classmethod
    @cache
    def load_default(cls) -> "SymmetricSeriesModel":
        with ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if (
            artifact.get("model_name") != MODEL_NAME
            or artifact.get("model_version") != MODEL_VERSION
            or artifact.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError("default artifact identity does not match the EXP-078 contract")
        return cls.from_mapping(artifact)

    def predict(self, snapshot: Mapping[str, float], *, best_of: int) -> float:
        engineered = build_feature_mapping(snapshot, best_of=best_of)
        missing = [name for name in self.feature_names if name not in engineered]
        unexpected = sorted(set(engineered).difference(self.feature_names))
        if missing or unexpected:
            raise ValueError(
                f"artifact/feature schema mismatch; missing={missing}, unexpected={unexpected}"
            )
        raw_logit = math.fsum(
            coefficient * engineered[name] / scale
            for name, scale, coefficient in zip(
                self.feature_names, self.scales, self.coefficients, strict=True
            )
        )
        return _sigmoid(self.logit_calibration_slope * raw_logit)


def predict_series_probability(snapshot: Mapping[str, float], *, best_of: int) -> float:
    """Predict with the checked-in immutable EXP-078 artifact."""

    return SymmetricSeriesModel.load_default().predict(snapshot, best_of=best_of)
