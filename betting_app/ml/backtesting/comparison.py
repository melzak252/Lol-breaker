"""Probability-quality comparison between a model and bookmaker market."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from betting_app.core.ev import fair_market_probabilities
from betting_app.ml.backtesting.odds_selection import select_quotes_for_match
from betting_app.ml.backtesting.types import HistoricalPrediction, MatchLabel, OddsQuote, PredictionMarketComparison
from betting_app.ml.backtesting.temporal import (
    select_temporally_eligible_predictions,
    select_temporally_eligible_quotes,
)
from betting_app.ml.config import BacktestConfig
from betting_app.ml.metrics import (
    accuracy_from_prob,
    binary_auc,
    binary_log_loss,
    brier_score,
    expected_calibration_error,
)


def compare_predictions_to_market(
    predictions: Iterable[HistoricalPrediction],
    labels: Iterable[MatchLabel],
    odds_quotes: Iterable[OddsQuote],
    config: BacktestConfig | None = None,
) -> PredictionMarketComparison:
    """Compare model probabilities against no-vig market probabilities.

    One observation is one historical model prediction for a finished match with
    at least one eligible pre-match quote. For multiple bookmakers, market
    probability is averaged across eligible no-vig quotes under the configured
    odds timing policy.
    """

    config = config or BacktestConfig()
    labels_by_match = {label.canonical_match_id: label for label in labels}
    quotes_by_match: dict[int, list[OddsQuote]] = defaultdict(list)
    for quote in odds_quotes:
        quotes_by_match[quote.canonical_match_id].append(quote)

    prediction_exclusions: dict[str, int] = {}
    quote_exclusions: dict[str, int] = defaultdict(int)
    if config.strict_temporal_eligibility:
        selection = select_temporally_eligible_predictions(predictions, labels_by_match.values())
        eligible_predictions = selection.predictions
        prediction_exclusions = selection.exclusions
    else:
        eligible_predictions = list(predictions)

    y_true: list[int] = []
    model_prob: list[float] = []
    market_prob: list[float] = []

    for prediction in eligible_predictions:
        label = labels_by_match.get(prediction.canonical_match_id)
        if label is None:
            continue
        quotes_for_match = quotes_by_match.get(prediction.canonical_match_id, [])
        if config.strict_temporal_eligibility:
            quotes_for_match, exclusions = select_temporally_eligible_quotes(
                quotes_for_match, prediction, label
            )
            for reason, count in exclusions.items():
                quote_exclusions[reason] += count
        elif config.require_prediction_before_odds and prediction.predicted_at is not None:
            quotes_for_match = [q for q in quotes_for_match if q.scraped_at >= prediction.predicted_at]
        eligible_quotes = select_quotes_for_match(quotes_for_match, label, config)
        if not eligible_quotes:
            continue

        fair_a_values = [fair_market_probabilities(q.odds_a, q.odds_b)[0] for q in eligible_quotes]
        y_true.append(1 if label.winner_side == "a" else 0)
        model_prob.append(prediction.prob_a)
        market_prob.append(sum(fair_a_values) / len(fair_a_values))

    return PredictionMarketComparison(
        observations=len(y_true),
        model_log_loss=binary_log_loss(y_true, model_prob),
        market_log_loss=binary_log_loss(y_true, market_prob),
        model_brier=brier_score(y_true, model_prob),
        market_brier=brier_score(y_true, market_prob),
        model_accuracy=accuracy_from_prob(y_true, model_prob),
        market_accuracy=accuracy_from_prob(y_true, market_prob),
        eligible_predictions=len(eligible_predictions),
        prediction_exclusions=prediction_exclusions,
        quote_exclusions=dict(sorted(quote_exclusions.items())),
        model_auc=binary_auc(y_true, model_prob),
        market_auc=binary_auc(y_true, market_prob),
        model_ece=expected_calibration_error(y_true, model_prob),
        market_ece=expected_calibration_error(y_true, market_prob),
    )
