"""Point-in-time market selection for fixed pre-match evaluation horizons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from betting_app.ml.backtesting.types import HistoricalPrediction, MatchLabel, OddsQuote


@dataclass(frozen=True)
class HorizonSpec:
    """One named target time before kickoff and its maximum quote staleness."""

    name: str
    before_start: timedelta
    max_quote_age: timedelta


DEFAULT_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec("t24h", timedelta(hours=24), timedelta(hours=2)),
    HorizonSpec("t6h", timedelta(hours=6), timedelta(hours=1)),
    HorizonSpec("t1h", timedelta(hours=1), timedelta(minutes=20)),
)


def select_horizon_quotes(
    prediction: HistoricalPrediction,
    label: MatchLabel,
    quotes: Iterable[OddsQuote],
    horizon: HorizonSpec,
) -> list[OddsQuote]:
    """Select the latest fresh quote per book available at a fixed horizon.

    A row is eligible only when the prediction is fully timestamped and the
    quote was observed after that prediction, no later than the target horizon,
    and strictly before kickoff.  Missing coverage remains an exclusion rather
    than silently selecting a later closing quote.
    """
    if prediction.predicted_at is None or prediction.data_cutoff_at is None or label.start_time is None:
        return []
    predicted_at = _as_utc(prediction.predicted_at)
    cutoff_at = _as_utc(prediction.data_cutoff_at)
    start_at = _as_utc(label.start_time)
    target_at = start_at - horizon.before_start
    if cutoff_at > predicted_at or predicted_at > target_at or target_at >= start_at:
        return []

    latest_by_bookmaker: dict[int, OddsQuote] = {}
    for quote in quotes:
        quote_at = _as_utc(quote.scraped_at)
        if (
            quote.canonical_match_id != prediction.canonical_match_id
            or quote.odds_a <= 1.0
            or quote.odds_b <= 1.0
            or quote_at < predicted_at
            or quote_at > target_at
            or quote_at >= start_at
            or target_at - quote_at > horizon.max_quote_age
        ):
            continue
        current = latest_by_bookmaker.get(quote.bookmaker_id)
        if current is None or _as_utc(current.scraped_at) < quote_at:
            latest_by_bookmaker[quote.bookmaker_id] = quote
    return sorted(latest_by_bookmaker.values(), key=lambda quote: (quote.bookmaker_id, _as_utc(quote.scraped_at)))


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
