from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DEFAULT_PROBABILITY_EPSILON = 0.001


def clip_probabilities(
    probabilities: Iterable[float] | np.ndarray | pd.Series,
    epsilon: float = DEFAULT_PROBABILITY_EPSILON,
) -> np.ndarray:
    """Clip binary probabilities to a numerically safe open interval.

    Args:
        probabilities: Predicted positive-class probabilities.
        epsilon: Lower and upper clipping margin.

    Returns:
        One-dimensional NumPy array clipped to ``[epsilon, 1 - epsilon]``.
    """

    return np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)


def calculate_ece(
    y_true: Iterable[int] | np.ndarray | pd.Series,
    y_prob: Iterable[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error for binary probabilities.

    The implementation uses equal-width bins over ``[0, 1]`` and reports the
    weighted mean absolute difference between empirical positive rate and mean
    predicted probability in each non-empty bin.

    Args:
        y_true: Binary target labels.
        y_prob: Predicted positive-class probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        Weighted average absolute calibration error.
    """

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (probabilities > lower) & (probabilities <= upper)
        weight = float(np.mean(in_bin))
        if weight == 0.0:
            continue
        ece += abs(float(np.mean(labels[in_bin])) - float(np.mean(probabilities[in_bin]))) * weight
    return ece


def binary_log_loss_vector(
    y_true: Iterable[int] | np.ndarray | pd.Series,
    y_prob: Iterable[float] | np.ndarray | pd.Series,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Return per-row binary LogLoss values.

    Args:
        y_true: Binary target labels.
        y_prob: Predicted positive-class probabilities.
        epsilon: Numerical clipping margin.

    Returns:
        Vector of per-observation LogLoss values.
    """

    labels = np.asarray(y_true, dtype=int)
    probabilities = clip_probabilities(y_prob, epsilon=epsilon)
    return -(labels * np.log(probabilities) + (1 - labels) * np.log(1.0 - probabilities))


def evaluate_binary_probabilities(
    y_true: Iterable[int] | np.ndarray | pd.Series,
    y_prob: Iterable[float] | np.ndarray | pd.Series,
    epsilon: float = DEFAULT_PROBABILITY_EPSILON,
    include_accuracy: bool = False,
) -> dict[str, float]:
    """Calculate standard binary probabilistic prediction metrics.

    Args:
        y_true: Binary target labels.
        y_prob: Predicted positive-class probabilities.
        epsilon: Probability clipping margin used for proper scoring metrics.
        include_accuracy: Whether to include threshold-0.5 accuracy.

    Returns:
        Dictionary containing sample size, AUC, LogLoss, Brier score and ECE.
    """

    labels = np.asarray(y_true, dtype=int)
    probabilities = clip_probabilities(y_prob, epsilon=epsilon)
    metrics: dict[str, float] = {
        "sample_size": float(len(labels)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": float(calculate_ece(labels, probabilities)),
    }
    if include_accuracy:
        metrics["accuracy_0_5"] = float(accuracy_score(labels, probabilities >= 0.5))
    return metrics


def evaluate_probability_column(
    data: pd.DataFrame,
    target_column: str,
    probability_column: str,
    epsilon: float = DEFAULT_PROBABILITY_EPSILON,
    include_accuracy: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate one probability column from a data frame.

    Args:
        data: Data frame containing target and probability columns.
        target_column: Binary target column name.
        probability_column: Probability column name.
        epsilon: Probability clipping margin.
        include_accuracy: Whether to include threshold-0.5 accuracy.
        metadata: Optional metadata merged into the output row.

    Returns:
        Metric row with optional metadata.
    """

    subset = data[[target_column, probability_column]].dropna().copy()
    row: dict[str, object] = dict(metadata or {})
    row["probability_column"] = probability_column
    row.update(
        evaluate_binary_probabilities(
            subset[target_column].astype(int).to_numpy(),
            subset[probability_column].to_numpy(dtype=float),
            epsilon=epsilon,
            include_accuracy=include_accuracy,
        )
    )
    row["sample_size"] = int(row["sample_size"])
    return row


def evaluate_probability_groups(
    predictions: pd.DataFrame,
    group_columns: list[str],
    target_column: str = "y_true",
    probability_column: str = "y_prob",
    epsilon: float = DEFAULT_PROBABILITY_EPSILON,
    include_accuracy: bool = True,
    calibrated_column: str = "calibrator_available",
) -> pd.DataFrame:
    """Evaluate long-form prediction streams grouped by variant columns.

    Args:
        predictions: Prediction table with labels and probabilities.
        group_columns: Columns defining compared variants.
        target_column: Binary target column name.
        probability_column: Probability column name.
        epsilon: Probability clipping margin.
        include_accuracy: Whether to include threshold-0.5 accuracy.
        calibrated_column: Optional column used to report calibrated sample rate.

    Returns:
        Metric table sorted by LogLoss.
    """

    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = dict(zip(group_columns, key_tuple, strict=True))
        row.update(
            evaluate_binary_probabilities(
                group[target_column].astype(int).to_numpy(),
                group[probability_column].to_numpy(dtype=float),
                epsilon=epsilon,
                include_accuracy=include_accuracy,
            )
        )
        row["sample_size"] = int(row["sample_size"])
        if calibrated_column in group.columns:
            row["calibrated_sample_rate"] = float(group[calibrated_column].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("logloss")


def calculate_basic_metrics(
    y_true: pd.Series,
    y_prob: pd.Series,
    label: str = "Model",
) -> Dict[str, float]:
    """Calculate basic binary classification metrics.

    Args:
        y_true: Binary target values.
        y_prob: Predicted probability for the positive class.
        label: Human-readable label printed with the metric summary.

    Returns:
        Dictionary with LogLoss, AUC, accuracy, and Brier score.
    """
    y_prob_clipped = y_prob.clip(1e-15, 1 - 1e-15)

    metrics = {
        "log_loss": log_loss(y_true, y_prob_clipped),
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
        "brier_score": brier_score_loss(y_true, y_prob),
    }

    print(f"\n--- {label} Metrics ---")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():<15}: {value:.5f}")

    return metrics


def shannon_entropy(p: float) -> float:
    """Calculate binary Shannon entropy in bits.

    Args:
        p: Probability of the positive class.

    Returns:
        Binary entropy in bits. Boundary probabilities return `0.0`.
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mean_prediction_entropy(y_prob: pd.Series) -> float:
    """Calculate average entropy of predicted probabilities.

    Args:
        y_prob: Predicted probability for the positive class.

    Returns:
        Mean binary entropy in bits.
    """
    entropies = [shannon_entropy(p) for p in y_prob]
    return float(np.mean(entropies))


def brier_decomposition(
    y_true: pd.Series,
    y_prob: pd.Series,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute binned Brier score decomposition.

    Args:
        y_true: Binary target values.
        y_prob: Predicted probability for the positive class.
        n_bins: Number of equal-width probability bins.

    Returns:
        Dictionary with reliability, resolution, and uncertainty components.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        return {
            "brier_reliability": 0.0,
            "brier_resolution": 0.0,
            "brier_uncertainty": 0.0,
        }

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob_arr, bins) - 1, 0, n_bins - 1)
    base_rate = float(np.mean(y_true_arr))

    reliability = 0.0
    resolution = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue

        weight = float(np.mean(mask))
        observed_rate = float(np.mean(y_true_arr[mask]))
        predicted_rate = float(np.mean(y_prob_arr[mask]))
        reliability += weight * (predicted_rate - observed_rate) ** 2
        resolution += weight * (observed_rate - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "brier_reliability": reliability,
        "brier_resolution": resolution,
        "brier_uncertainty": uncertainty,
    }


def calculate_detailed_metrics(
    y_true: pd.Series,
    y_prob: pd.Series,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Calculate classification, calibration, and information metrics.

    Args:
        y_true: Binary target values.
        y_prob: Predicted probability for the positive class.
        n_bins: Number of bins used for binned calibration estimates.

    Returns:
        Dictionary with base metrics, Brier decomposition, entropy metrics,
        mutual information estimate, and sample count.
    """
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    n_samples = len(y_true_arr)

    basic = calculate_basic_metrics(y_true, y_prob)
    brier_decomp = brier_decomposition(y_true, y_prob, n_bins=n_bins)

    o_bar = np.mean(y_true_arr)
    h_out = shannon_entropy(o_bar)
    h_pred = mean_prediction_entropy(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(y_prob_arr, bins) - 1, 0, n_bins - 1)

    bin_sums = np.bincount(binids, weights=y_true_arr, minlength=n_bins)
    bin_counts = np.bincount(binids, minlength=n_bins)

    mask = bin_counts > 0
    n_k = bin_counts[mask]
    o_bar_k = bin_sums[mask] / n_k

    h_cond = 0.0
    for index, count in enumerate(n_k):
        h_cond += (count / n_samples) * shannon_entropy(o_bar_k[index])

    mi = h_out - h_cond

    metrics = {
        **basic,
        **brier_decomp,
        "outcome_entropy": h_out,
        "prediction_entropy": h_pred,
        "mutual_info": mi,
        "count": float(n_samples),
    }

    return metrics
