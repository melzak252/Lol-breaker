"""Superbet NoDriver scraper for League of Legends prematch match-winner odds."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from betting_app.scrapers.base import RawOddsSnapshot
from betting_app.scrapers.nodriver_client import NoDriverClient
from betting_app.scrapers.superbet_parser import ParsedSuperbetOffer, SUPERBET_LOL_URL, parse_superbet_lol_offers


SCRAPER_VERSION = "superbet-nodriver-lol-match-winner-0.1"


class SuperbetNoDriverScraper:
    """NoDriver-based scraper for Superbet LoL match-winner markets."""

    bookmaker = "superbet"
    scraper_version = SCRAPER_VERSION

    def __init__(self, start_url: str = SUPERBET_LOL_URL, headless: bool | None = None) -> None:
        self.start_url = start_url
        self.headless = headless

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Open Superbet and return normalized LoL odds snapshots."""

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        async with NoDriverClient(headless=self.headless) as client:
            tab = await client.open(self.start_url)
            await self.wait_for_render(tab)
            await self.accept_cookies(tab)
            html_path, screenshot_path = await client.save_debug_artifacts(tab, f"superbet_{timestamp}")
            body_text = await self.extract_body_text(tab)
            body_path = client.debug_dir / f"superbet_{timestamp}_body.txt"
            body_path.write_text(body_text, encoding="utf-8")
            html_text = Path(html_path).read_text(encoding="utf-8") if html_path else ""
            cards = parse_superbet_lol_offers(body_text, html_text=html_text)
            snapshots = [
                snapshot
                for card in cards
                if (snapshot := self.parse_match_card(card, html_path or str(body_path), screenshot_path))
            ]

        print(
            f"Superbet scraper captured {len(snapshots)} snapshots. "
            f"Debug html={html_path}, body={body_path}, screenshot={screenshot_path}"
        )
        return snapshots

    async def wait_for_render(self, tab: Any, seconds: float = 8.0) -> None:
        """Wait for Superbet SPA to render content."""

        import asyncio

        _ = tab
        await asyncio.sleep(seconds)

    async def accept_cookies(self, tab: Any) -> None:
        """Best-effort cookie modal acceptance."""

        try:
            await tab.evaluate(
                """Array.from(document.querySelectorAll('button'))
                .find(button => /akcept|zgadzam|accept/i.test(button.innerText || ''))?.click()"""
            )
        except Exception:
            return

    async def extract_body_text(self, tab: Any) -> str:
        """Return rendered page body text."""

        body_text = await tab.evaluate("document.body ? document.body.innerText : ''")
        return str(body_text or "")

    def parse_match_card(
        self,
        card: ParsedSuperbetOffer,
        html_path: str | None = None,
        screenshot_path: str | None = None,
    ) -> RawOddsSnapshot | None:
        """Convert one parsed Superbet card into RawOddsSnapshot."""

        raw_payload = {
            "bookmaker_event_id": card.bookmaker_event_id,
            "market_count_label": card.market_count_label,
            "raw_text": card.raw_text,
            "source_url": self.start_url,
            "offer_url": card.offer_url,
        }
        return RawOddsSnapshot(
            bookmaker=self.bookmaker,
            raw_team_a=card.raw_team_a,
            raw_team_b=card.raw_team_b,
            odds_a=card.odds_a,
            odds_b=card.odds_b,
            scraped_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
            raw_league=card.league,
            match_start_time=card.start_time_label,
            source_url=self.start_url,
            offer_url=card.offer_url,
            market_type="match_winner",
            is_live=False,
            scraper_name="superbet_nodriver_lol_match_winner",
            scraper_version=SCRAPER_VERSION,
            raw_payload=raw_payload,
            page_html_path=str(html_path) if html_path else None,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
        )
