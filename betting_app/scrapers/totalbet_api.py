"""TOTALbet API scraper for League of Legends prematch match-winner odds."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from betting_app.scrapers.base import RawOddsSnapshot


TOTALBET_ESPORT_URL = "https://totalbet.pl/esport"
TOTALBET_EVENTS_API = "https://totalbet.pl/dealer/bdata/v1/bet/events/esport"
SCRAPER_VERSION = "totalbet-api-lol-match-winner-0.1"


class TotalbetApiScraper:
    """API scraper for TOTALbet LoL prematch match-winner markets."""

    bookmaker = "totalbet"
    scraper_version = SCRAPER_VERSION

    def __init__(
        self,
        start_url: str = TOTALBET_ESPORT_URL,
        headless: bool | None = None,
        pages: int = 20,
        per_page: int = 100,
    ) -> None:
        self.start_url = start_url
        self.headless = headless
        self.pages = pages
        self.per_page = per_page

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Fetch paginated esport events and keep LoL prematch match-winner markets."""

        _ = self.headless
        events_seen = 0
        snapshots: list[RawOddsSnapshot] = []
        for page in range(1, self.pages + 1):
            batch = self.fetch_events_page(page)
            observed_at = datetime.now(UTC)
            if not batch:
                break
            events_seen += len(batch)
            snapshots.extend(snapshot for event in batch if (snapshot := self.parse_event(event, observed_at=observed_at)))
        print(f"TOTALbet API captured {len(snapshots)} LoL snapshots from {events_seen} esport events.")
        return snapshots

    def fetch_events_page(self, page: int) -> list[dict[str, Any]]:
        """Fetch one TOTALbet esport events page."""

        url = f"{TOTALBET_EVENTS_API}?{urllib.parse.urlencode({'page': page, 'per_page': self.per_page})}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.load(response)
        return list((payload.get("data") or {}).get("events") or [])

    def parse_event(self, event: dict[str, Any], *, observed_at: datetime | None = None) -> RawOddsSnapshot | None:
        """Convert one TOTALbet event into a two-sided match-winner snapshot."""

        path = event.get("path") or []
        if not path or str(path[0].get("name")) != "League of Legends":
            return None
        participants = event.get("participants") or []
        if len(participants) != 2:
            return None
        market = next((
            m for m in event.get("markets") or []
            if str(m.get("name", "")).strip().lower() in {
                "zwycięzca meczu", "zwyciezca meczu", "zwycięzca", "zwyciezca", "match winner",
            }
        ), None)
        if not market:
            return None
        outcomes = sorted(market.get("outcomes") or [], key=lambda item: str(item.get("sort") or ""))
        odds_a = parse_float(outcomes[0].get("odds")) if outcomes else None
        odds_b = parse_float(outcomes[1].get("odds")) if len(outcomes) > 1 else None
        inactive = (
            event.get("status") not in (None, "active") or market.get("status") not in (None, "active")
            or event.get("isActive") is False or market.get("isActive") is False
            or len(outcomes) != 2
            or any(outcome.get("status") not in (None, "active") or outcome.get("isActive") is False for outcome in outcomes[:2])
        )
        if inactive:
            odds_a = odds_b = None
        team_a = str(participants[0].get("name") or "")
        team_b = str(participants[1].get("name") or "")
        if not team_a or not team_b:
            return None
        event_type = event.get("type")
        market_type = market.get("market_type")
        is_live = True if "live" in (event_type, market_type) else False if event_type in ("prematch", "pre", "prematch_event") else None
        unknown_live = False
        for source in (event, market):
            if "isLive" in source:
                live = source["isLive"]
                if isinstance(live, bool) or type(live) is int and live in (0, 1):
                    is_live = True if live == 1 or is_live is True else False
                else:
                    unknown_live = True
        if unknown_live:
            is_live = None
        event_uuid = str(event.get("uuid"))
        league = " / ".join(str(p.get("name")) for p in path[1:] if p.get("name")) or str(event.get("tournament") or "")
        return RawOddsSnapshot(
            bookmaker=self.bookmaker,
            raw_team_a=team_a,
            raw_team_b=team_b,
            odds_a=odds_a,
            odds_b=odds_b,
            scraped_at=(observed_at or datetime.now(UTC)).isoformat(),
            raw_league=league,
            match_start_time=event.get("start_at"),
            source_url=TOTALBET_EVENTS_API,
            offer_url=f"https://totalbet.pl/esport/event-details/{event_uuid}",
            market_type="match_winner",
            is_live=is_live,
            scraper_name="totalbet_api_lol_match_winner",
            scraper_version=SCRAPER_VERSION,
            raw_payload={"event": event, "market": market, "outcomes": outcomes[:2]},
        )


def parse_float(value: Any) -> float | None:
    """Parse bookmaker odd value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(str(value).replace(",", "."))
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 1 else None
