"""Shared metrics for binary probabilistic predictions.

This module exposes the probability-quality functions used by thesis experiments
so scripts do not need to redefine ECE, LogLoss vectors or grouped metric tables.
The implementations live in :mod:`src.analysis.metrics` for backward
compatibility with older code.
"""

from src.analysis.metrics import (
    DEFAULT_PROBABILITY_EPSILON,
    binary_log_loss_vector,
    calculate_ece,
    clip_probabilities,
    evaluate_binary_probabilities,
    evaluate_probability_column,
    evaluate_probability_groups,
)

__all__ = [
    "DEFAULT_PROBABILITY_EPSILON",
    "binary_log_loss_vector",
    "calculate_ece",
    "clip_probabilities",
    "evaluate_binary_probabilities",
    "evaluate_probability_column",
    "evaluate_probability_groups",
]
