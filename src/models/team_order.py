"""Utilities for Team1/Team2 orientation diagnostics."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def binomial_columns(features: list[str], suffix: str = "_binom_series") -> list[str]:
    """Return generated binomial probability feature names.

    Args:
        features: Model feature names.
        suffix: Suffix identifying binomial series features.

    Returns:
        Feature names ending with the configured suffix.
    """

    return [feature for feature in features if feature.endswith(suffix)]


def pair_columns(features: list[str]) -> list[tuple[str, str]]:
    """Find feature pairs that represent Team 1 / Team 2 quantities.

    Args:
        features: Feature names used by a side-oriented model.

    Returns:
        Unique pairs of columns that should be swapped together.
    """

    feature_set = set(features)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for feature in features:
        if feature in seen:
            continue
        counterpart = ""
        if feature.startswith("t1_"):
            counterpart = "t2_" + feature[3:]
        elif feature.startswith("t2_"):
            counterpart = "t1_" + feature[3:]
        else:
            match = re.match(r"(.+)([12])$", feature)
            if match is not None:
                prefix, side = match.groups()
                counterpart = f"{prefix}{'2' if side == '1' else '1'}"
        if counterpart and counterpart in feature_set and counterpart not in seen:
            left, right = sorted([feature, counterpart])
            pairs.append((left, right))
            seen.update({feature, counterpart})
    return pairs


def swap_orientation(
    data: pd.DataFrame,
    features: list[str],
    rank_probability_features: list[str],
    swap_mask: np.ndarray,
    target_column: str = "y_true",
) -> pd.DataFrame:
    """Swap Team 1 and Team 2 representation for selected rows.

    Probability-like ranking and binomial features are converted with ``1 - p``;
    paired Team1/Team2 numeric features are exchanged; and the binary target is
    flipped for swapped rows.

    Args:
        data: Input modeling frame.
        features: Final model feature names.
        rank_probability_features: Ranking probability columns.
        swap_mask: Boolean mask indicating rows to swap.
        target_column: Binary target column to flip.

    Returns:
        Modeling frame with selected rows represented from the opposite side.
    """

    swapped = data.copy()
    mask = np.asarray(swap_mask, dtype=bool)
    probability_features = [
        feature
        for feature in [*rank_probability_features, *binomial_columns(features)]
        if feature in swapped.columns
    ]
    for feature in probability_features:
        swapped.loc[mask, feature] = 1.0 - swapped.loc[mask, feature].astype(float)

    for left, right in pair_columns(features):
        left_values = swapped.loc[mask, left].copy()
        swapped.loc[mask, left] = swapped.loc[mask, right].to_numpy()
        swapped.loc[mask, right] = left_values.to_numpy()

    swapped.loc[mask, target_column] = 1 - swapped.loc[mask, target_column].astype(int)
    swapped["orientation_swapped"] = mask.astype(int)
    return swapped


def symmetrize_binary_probabilities(original_prob: np.ndarray, swapped_side_prob: np.ndarray) -> np.ndarray:
    """Average original and swapped predictions in the original orientation.

    Args:
        original_prob: Probability that Team 1 wins in original orientation.
        swapped_side_prob: Probability that the swapped Team 1 wins after Team
            sides were exchanged.

    Returns:
        Symmetrized probability in original Team-1 orientation.
    """

    converted_prob = 1.0 - np.asarray(swapped_side_prob, dtype=float)
    return 0.5 * (np.asarray(original_prob, dtype=float) + converted_prob)
