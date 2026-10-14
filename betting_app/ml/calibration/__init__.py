"""Probability calibration and evaluation components for match outcome models."""

from __future__ import annotations

from betting_app.ml.calibration.candidate_calibration import (
    BetaCalibrator,
    TemperatureScalingCalibrator,
    UncertaintyGatedCalibrator,
    brier_score_decomposition,
    expected_calibration_error,
)
from betting_app.ml.calibration.venn_abers import (
    ConformalRiskGater,
    VennAbersCalibrator,
    VennAbersIntervals,
)

__all__ = [
    "BetaCalibrator",
    "ConformalRiskGater",
    "TemperatureScalingCalibrator",
    "UncertaintyGatedCalibrator",
    "VennAbersCalibrator",
    "VennAbersIntervals",
    "brier_score_decomposition",
    "expected_calibration_error",
]
