"""Calibration utilities for walk-forward probability streams."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.analysis.probability_metrics import clip_probabilities


def logit(probability: np.ndarray | pd.Series, epsilon: float = 0.001) -> np.ndarray:
    """Return clipped logit values as a two-dimensional array.

    Args:
        probability: Probability vector.
        epsilon: Probability clipping margin.

    Returns:
        Logit-transformed probabilities with shape ``(n_samples, 1)``.
    """

    clipped = clip_probabilities(probability, epsilon=epsilon)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def expanding_platt_isotonic_calibration(
    predictions: pd.DataFrame,
    variant_value: str,
    variant_column: str = "variant",
    target_column: str = "y_true",
    probability_column: str = "y_prob",
    fold_column: str = "fold",
    date_column: str = "date",
    id_column: str = "golgg_match_id",
    min_calibration_samples: int = 1000,
    epsilon: float = 0.001,
    output_variant_column: str = "base_variant",
) -> pd.DataFrame:
    """Apply leakage-safe expanding Platt and isotonic calibration.

    For each walk-forward fold, calibrators are fitted only on predictions from
    earlier folds. If the calibration pool is too small or contains one class,
    raw probabilities are passed through unchanged and ``calibrator_available``
    is set to ``0``.

    Args:
        predictions: Long-form prediction table.
        variant_value: Variant/stream value to select and calibrate.
        variant_column: Column containing the input variant label.
        target_column: Binary target column name.
        probability_column: Raw probability column name.
        fold_column: Walk-forward fold column name.
        date_column: Date column name.
        id_column: Match identifier column name.
        min_calibration_samples: Minimum earlier-fold observations required.
        epsilon: Probability clipping margin.
        output_variant_column: Name of the output variant-label column.

    Returns:
        Long-form table containing raw, Platt and isotonic calibrated streams.
    """

    variant_data = predictions[predictions[variant_column] == variant_value].copy().sort_values([fold_column, date_column])
    calibrated_parts: list[pd.DataFrame] = []

    for fold in sorted(variant_data[fold_column].unique()):
        test_fold = variant_data[variant_data[fold_column] == fold].copy()
        calibration_pool = variant_data[variant_data[fold_column] < fold].copy()
        raw_prob = clip_probabilities(test_fold[probability_column].to_numpy(dtype=float), epsilon=epsilon)
        platt_prob = raw_prob.copy()
        isotonic_prob = raw_prob.copy()
        calibrated = False

        if len(calibration_pool) >= min_calibration_samples and calibration_pool[target_column].nunique() == 2:
            cal_y = calibration_pool[target_column].astype(int).to_numpy()
            cal_prob = clip_probabilities(calibration_pool[probability_column].to_numpy(dtype=float), epsilon=epsilon)
            platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            platt.fit(logit(cal_prob, epsilon=epsilon), cal_y)
            platt_prob = clip_probabilities(platt.predict_proba(logit(raw_prob, epsilon=epsilon))[:, 1], epsilon=epsilon)

            isotonic = IsotonicRegression(out_of_bounds="clip", y_min=epsilon, y_max=1.0 - epsilon)
            isotonic.fit(cal_prob, cal_y)
            isotonic_prob = clip_probabilities(isotonic.predict(raw_prob), epsilon=epsilon)
            calibrated = True

        for calibration_name, probability in [
            ("raw", raw_prob),
            ("platt_expanding", platt_prob),
            ("isotonic_expanding", isotonic_prob),
        ]:
            output = test_fold[[id_column, date_column, fold_column, target_column]].copy()
            output[output_variant_column] = variant_value
            output["calibration"] = calibration_name
            output[probability_column] = probability
            output["calibrator_available"] = int(calibrated)
            calibrated_parts.append(output)

    return pd.concat(calibrated_parts, ignore_index=True)
