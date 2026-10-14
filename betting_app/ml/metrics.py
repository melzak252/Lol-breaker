"""Shared model and betting metrics for production ML evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable


EPS = 1e-15


def clip_probability(probability: float) -> float:
    """Clip a probability to a numerically safe open interval."""

    return min(max(float(probability), EPS), 1.0 - EPS)


def binary_log_loss(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    """Return mean binary log loss for labels in ``{0, 1}``."""

    losses: list[float] = []
    for y, p in zip(y_true, y_prob, strict=False):
        p = clip_probability(p)
        losses.append(-(int(y) * math.log(p) + (1 - int(y)) * math.log(1.0 - p)))
    return sum(losses) / len(losses) if losses else float("nan")


def brier_score(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    """Return mean Brier score for labels in ``{0, 1}``."""

    errors = [(float(p) - int(y)) ** 2 for y, p in zip(y_true, y_prob, strict=False)]
    return sum(errors) / len(errors) if errors else float("nan")


def accuracy_from_prob(y_true: Iterable[int], y_prob: Iterable[float], threshold: float = 0.5) -> float:
    """Return classification accuracy after thresholding probabilities."""

    values = [(1 if float(p) >= threshold else 0) == int(y) for y, p in zip(y_true, y_prob, strict=False)]
    return sum(values) / len(values) if values else float("nan")


def binary_auc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    """Return rank-based binary AUC without requiring a third-party runtime."""

    pairs = sorted(
        ((float(probability), int(target)) for target, probability in zip(y_true, y_prob, strict=False)),
        key=lambda pair: pair[0],
    )
    positives = sum(target for _, target in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        mean_rank = (index + 1 + next_index) / 2.0
        rank_sum += mean_rank * sum(target for _, target in pairs[index:next_index])
        index = next_index
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    *,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error for binary predictions."""

    if bins < 1:
        raise ValueError("bins must be positive")
    pairs = [(int(target), float(probability)) for target, probability in zip(y_true, y_prob, strict=False)]
    if not pairs:
        return float("nan")
    bucketed: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for target, probability in pairs:
        index = min(int(clip_probability(probability) * bins), bins - 1)
        bucketed[index].append((target, probability))
    total = len(pairs)
    return sum(
        len(bucket) / total
        * abs(
            sum(target for target, _ in bucket) / len(bucket)
            - sum(probability for _, probability in bucket) / len(bucket)
        )
        for bucket in bucketed
        if bucket
    )


def max_drawdown(bankroll_curve: Iterable[float]) -> float:
    """Return maximum drawdown as a positive money amount."""

    peak: float | None = None
    worst = 0.0
    for value in bankroll_curve:
        value = float(value)
        peak = value if peak is None else max(peak, value)
        worst = max(worst, peak - value)
    return worst


def roi(total_profit: float, total_staked: float) -> float:
    """Return return on investment per unit staked."""

    return float(total_profit) / float(total_staked) if total_staked > 0 else 0.0
