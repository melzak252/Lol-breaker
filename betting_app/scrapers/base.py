"""Base types for bookmaker odds scrapers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class RawOddsSnapshot:
    """Observed pair in source order; unavailable prices/live state stay null."""

    bookmaker: str
    raw_team_a: str
    raw_team_b: str
    odds_a: float | None
    odds_b: float | None
    scraped_at: str | None = None
    raw_league: str | None = None
    match_start_time: str | None = None
    source_url: str | None = None
    offer_url: str | None = None
    market_type: str = "match_winner"
    is_live: bool | None = None
    scraper_name: str | None = None
    scraper_version: str | None = None
    raw_payload: str | dict | None = None
    page_html_path: str | None = None
    screenshot_path: str | None = None
    best_of: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary and fill scrape timestamp if missing."""

        payload = asdict(self)
        payload["scraped_at"] = payload["scraped_at"] or datetime.now(UTC).isoformat()
        return payload


@dataclass(frozen=True)
class RawOutcomeOddsSnapshot:
    """Single bookmaker outcome/selection odds snapshot.

    This is the preferred format for real bookmaker APIs because a page/API often
    returns one outcome at a time (for example STS social-api popular picks), not
    a clean two-sided match-winner pair. The database can reconstruct market
    history from these atomic outcome ticks.
    """

    bookmaker: str
    bookmaker_event_id: str
    raw_team_a: str
    raw_team_b: str
    decimal_odds: float
    outcome_key: str
    outcome_name: str
    market_key: str
    market_name: str
    scraped_at: str | None = None
    match_start_time: str | None = None
    sport_id: str | None = None
    sport_name: str | None = None
    category_id: str | None = None
    category_name: str | None = None
    league_id: str | None = None
    league_name: str | None = None
    outcome_side: str | None = None
    line_id: str | None = None
    line_name: str | None = None
    is_extra_market: bool = False
    is_live: bool = False
    source_url: str | None = None
    offer_url: str | None = None
    scraper_name: str | None = None
    scraper_version: str | None = None
    scrape_run_id: int | None = None
    raw_payload: dict[str, Any] | str | None = None
    best_of: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary and fill scrape timestamp if missing."""

        payload = asdict(self)
        payload["snapshot_type"] = "outcome"
        payload["scraped_at"] = payload["scraped_at"] or datetime.now(UTC).replace(microsecond=0).isoformat()
        return payload


class OddsScraper(Protocol):
    """Protocol implemented by all bookmaker scrapers."""

    bookmaker: str

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot | RawOutcomeOddsSnapshot]:
        """Scrape upcoming LoL match winner markets."""
