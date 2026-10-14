"""Strict point-in-time eligibility for historical model evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from betting_app.ml.backtesting.types import HistoricalPrediction, MatchLabel, OddsQuote


@dataclass(frozen=True)
class TemporalPredictionSelection:
    """Latest executable prediction per match and transparent exclusion counts."""

    predictions: list[HistoricalPrediction]
    exclusions: dict[str, int]


def _is_aware(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def select_temporally_eligible_predictions(
    predictions: Iterable[HistoricalPrediction],
    labels: Iterable[MatchLabel],
) -> TemporalPredictionSelection:
    """Keep the latest prediction known before each match began.

    A prediction is eligible only when all source, prediction, and kickoff
    timestamps are timezone-aware and satisfy ``data_cutoff <= predicted <
    match_start``. Rows without that proof are excluded rather than guessed.
    """

    labels_by_match = {label.canonical_match_id: label for label in labels}
    eligible_by_match: dict[int, HistoricalPrediction] = {}
    exclusions: Counter[str] = Counter()

    for prediction in predictions:
        label = labels_by_match.get(prediction.canonical_match_id)
        if label is None:
            exclusions["missing_label"] += 1
            continue
        if not _is_aware(prediction.data_cutoff_at):
            exclusions["missing_or_unzoned_data_cutoff_at"] += 1
            continue
        if not _is_aware(prediction.predicted_at):
            exclusions["missing_or_unzoned_predicted_at"] += 1
            continue
        if not _is_aware(label.start_time):
            exclusions["missing_or_unzoned_match_start_at"] += 1
            continue
        if prediction.data_cutoff_at > prediction.predicted_at:
            exclusions["data_cutoff_after_prediction"] += 1
            continue
        if prediction.predicted_at >= label.start_time:
            exclusions["prediction_not_before_match_start"] += 1
            continue
        if not (0.0 <= prediction.prob_a <= 1.0 and 0.0 <= prediction.prob_b <= 1.0):
            exclusions["probability_out_of_bounds"] += 1
            continue
        if abs(prediction.prob_a + prediction.prob_b - 1.0) > 1e-6:
            exclusions["probabilities_not_complementary"] += 1
            continue

        current = eligible_by_match.get(prediction.canonical_match_id)
        if current is None or prediction.predicted_at >= current.predicted_at:
            eligible_by_match[prediction.canonical_match_id] = prediction

    return TemporalPredictionSelection(
        predictions=sorted(
            eligible_by_match.values(),
            key=lambda prediction: (prediction.predicted_at, prediction.canonical_match_id),
        ),
        exclusions=dict(sorted(exclusions.items())),
    )


def select_temporally_eligible_quotes(
    quotes: Iterable[OddsQuote],
    prediction: HistoricalPrediction,
    label: MatchLabel,
) -> tuple[list[OddsQuote], dict[str, int]]:
    """Return quotes available after the prediction and before kickoff."""

    exclusions: Counter[str] = Counter()
    eligible: list[OddsQuote] = []
    for quote in quotes:
        if not _is_aware(quote.scraped_at):
            exclusions["missing_or_unzoned_quote_at"] += 1
            continue
        if quote.scraped_at < prediction.predicted_at:
            exclusions["quote_before_prediction"] += 1
            continue
        if quote.scraped_at >= label.start_time:
            exclusions["quote_not_before_match_start"] += 1
            continue
        eligible.append(quote)
    return eligible, dict(sorted(exclusions.items()))
