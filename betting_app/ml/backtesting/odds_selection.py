"""Policies deciding which historical odds snapshot a backtest would use."""

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from betting_app.ml.config import BacktestConfig
from betting_app.ml.backtesting.types import MatchLabel, OddsQuote


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def select_quotes_for_match(
    quotes: Iterable[OddsQuote],
    label: MatchLabel,
    config: BacktestConfig,
) -> list[OddsQuote]:
    """Return quotes eligible under the configured historical timing policy.

    The default policy, ``latest_pre_match``, answers the practical question
    "when would we have bet?" by taking the latest available pre-match snapshot
    per bookmaker, optionally requiring a minimum buffer before match start.
    """

    valid = [q for q in quotes if q.odds_a > 1.0 and q.odds_b > 1.0]
    if label.start_time is not None:
        start_utc = _as_utc(label.start_time)
        latest_allowed = start_utc - timedelta(minutes=config.min_minutes_before_start)
        valid = [q for q in valid if _as_utc(q.scraped_at) <= latest_allowed]

    policy = config.odds_policy.lower()
    if policy == "all_pre_match":
        return sorted(valid, key=lambda q: _as_utc(q.scraped_at))

    by_bookmaker: dict[int, list[OddsQuote]] = defaultdict(list)
    for quote in sorted(valid, key=lambda q: _as_utc(q.scraped_at)):
        by_bookmaker[quote.bookmaker_id].append(quote)

    selected: list[OddsQuote] = []
    if policy in ("latest_pre_match", "close_pre_match", "close"):
        for b_quotes in by_bookmaker.values():
            selected.append(b_quotes[-1])
    elif policy in ("open_pre_match", "open"):
        for b_quotes in by_bookmaker.values():
            selected.append(b_quotes[0])
    elif policy in ("mid_pre_match", "mid"):
        for b_quotes in by_bookmaker.values():
            mid_idx = (len(b_quotes) - 1) // 2
            selected.append(b_quotes[mid_idx])
    else:
        raise ValueError(f"Unsupported odds policy: {config.odds_policy}")

    return sorted(selected, key=lambda q: (q.bookmaker_id, _as_utc(q.scraped_at)))
