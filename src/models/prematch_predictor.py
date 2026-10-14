"""Saved pre-draft forecasts and strict chronological fit partitions.

These artifacts implement a retrospective daily-availability scenario. They do
not certify roster announcement times or place bets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

from betting_app.ml.calibration.candidate_calibration import TemperatureScalingCalibrator
from src.analysis.probability_metrics import clip_probabilities

BATCH_KEYS = ("players", "teams", "context", "baseline_series")


def partition_indices(metadata: pd.DataFrame, origin: str, calibration_months: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split closed TRAIN/CAL/forecast; zero CAL months trains up to the origin."""
    if isinstance(calibration_months, bool) or not isinstance(calibration_months, (int, np.integer)) or calibration_months < 0:
        raise ValueError("calibration_months must be a nonnegative integer")
    start = pd.Timestamp(origin)
    train_end = (start - pd.DateOffset(months=calibration_months)).strftime("%Y-%m-%d")
    end = (start + pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    date = metadata["date"].astype(str)
    released = metadata["release_day"].astype(str)
    train = date.lt(train_end) & released.lt(train_end)
    calibration = date.ge(train_end) & date.lt(origin) & released.lt(origin)
    test = date.ge(origin) & date.lt(end)
    return train.to_numpy(), calibration.to_numpy(), test.to_numpy()


def _batch_keys(batch: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    history_keys = ("history", "history_mask")
    present = [key in batch for key in history_keys]
    if any(present) and not all(present):
        raise ValueError("History values and their availability mask are required together")
    return BATCH_KEYS + history_keys if all(present) else BATCH_KEYS


def take_batch(arrays: Mapping[str, np.ndarray], index: np.ndarray, *, prior_roster: bool = False) -> dict[str, np.ndarray]:
    """Select one population without altering feature preprocessing or outcomes."""
    return {
        key: arrays[("prior_" + key) if prior_roster and key != "context" else key][index]
        for key in _batch_keys(arrays)
    }


def swap_batch(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = _batch_keys(batch)
    return {
        key: (batch[key] if key == "context" else 1.0 - batch[key] if key == "baseline_series" else batch[key][:, ::-1])
        for key in keys
    }


@dataclass
class FittedMember:
    """One fitted estimator, with an optional explicitly held-out calibrator."""

    configuration: dict[str, Any]
    estimator: Any
    calibrator: TemperatureScalingCalibrator | None
    probability_epsilon: float
    origin: str
    training_cutoff: str
    calibration_count: int

    def predict_proba(self, batch: Mapping[str, np.ndarray], variant: str = "temperature") -> np.ndarray:
        if variant not in ("raw", "temperature"):
            raise ValueError(f"Unknown probability variant: {variant}")
        if variant == "temperature" and self.calibrator is None:
            raise ValueError("Temperature prediction requires a fitted held-out calibrator")
        if self.configuration["kind"] == "rating":
            # This input is already integrated over the full series format.
            raw = np.asarray(batch["baseline_series"][:, self.configuration["params"]["index"]], dtype=float)
        else:
            raw = np.asarray(self.estimator.predict_proba(batch), dtype=float)
        if raw.ndim != 1 or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise ValueError("Base estimator returned invalid series probabilities")
        p = clip_probabilities(raw, epsilon=self.probability_epsilon)
        if variant == "temperature":
            p = self.calibrator.transform(logit(p))
        return clip_probabilities(p, epsilon=self.probability_epsilon)


@dataclass
class ForecastArtifact:
    """Selected recipe, corresponding fitted members, and released history state.

    Only load trusted local joblib artifacts: deserialization executes Python.
    """

    recipe: list[dict[str, Any]]
    members: dict[str, FittedMember]
    state: Any
    provenance: dict[str, Any]

    def predict_proba(self, batch: Mapping[str, np.ndarray]) -> np.ndarray:
        weights = np.asarray([part["weight"] for part in self.recipe], dtype=float)
        if not len(weights) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Forecast recipe must have nonnegative weights summing to one")
        result = np.zeros(len(batch["context"]), dtype=float)
        for part in self.recipe:
            result += part["weight"] * self.members[part["candidate"]].predict_proba(batch, part["variant"])
        return result

    def predict_match(self, team_a: str | None, roster_a: list[str] | tuple[str, ...], team_b: str | None, roster_b: list[str] | tuple[str, ...], day: str, best_of: int) -> dict[str, Any]:
        if day < self.provenance["model_origin"]:
            raise ValueError("Forecast day precedes the fitted model origin")
        feature = self.state.features(team_a, tuple(roster_a), team_b, tuple(roster_b), day, best_of)
        batch = {key: np.asarray(feature[key])[None] for key in BATCH_KEYS}
        p = float(self.predict_proba(batch)[0])
        return {
            "probability_a": p,
            "probability_b": 1.0 - p,
            "target": "series_winner",
            "best_of": best_of,
            "day": day,
            "team_a": team_a,
            "team_b": team_b,
            "roster_a": list(roster_a),
            "roster_b": list(roster_b),
            "model_origin": self.provenance["model_origin"],
            "latest_history_release_day": feature["max_release_day"],
            "recipe": self.provenance["selected_recipe"],
            "availability": "daily prior-release scenario; supplied roster treated as known, not independently verified",
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> ForecastArtifact:
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise ValueError("Not a ground-up pre-match forecasting artifact")
        return artifact


@dataclass
class ArchitectureForecastArtifact:
    """Research inference from aligned, cutoff-validated feature batches.

    This bundle deliberately has no live roster/history adapter. Callers must
    supply the historical token and summary schemas used by the frozen run.
    """

    members: dict[str, FittedMember]
    stack: Any
    expert_names: tuple[str, ...]
    expert_variant: str
    provenance: dict[str, Any]

    def predict_proba(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        from src.models.prematch_stack import stack_context

        if "history" not in _batch_keys(batch):
            raise ValueError("Architecture inference requires aligned player histories")
        predictions = {
            f"{name}__{variant}": member.predict_proba(batch, variant)
            for name, member in self.members.items()
            for variant in ("raw", "temperature")
        }
        experts = np.column_stack([
            predictions[f"{name}__{self.expert_variant}"] for name in self.expert_names
        ])
        predictions["specialist_stack"] = self.stack.predict_proba(
            experts, stack_context(batch, experts)
        )
        return predictions

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> ArchitectureForecastArtifact:
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise ValueError("Not an architecture forecasting artifact")
        return artifact
