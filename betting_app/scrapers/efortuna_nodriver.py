"""eFortuna NoDriver scraper for League of Legends prematch match-winner odds."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from betting_app.scrapers.base import RawOddsSnapshot
from betting_app.scrapers.efortuna_parser import (
    EFORTUNA_LOL_LEAGUE_URLS,
    EFORTUNA_LOL_URL,
    ParsedEFortunaOffer,
    parse_efortuna_lol_offers,
)
from betting_app.scrapers.nodriver_client import NoDriverClient


SCRAPER_VERSION = "efortuna-nodriver-lol-match-winner-0.1"


class EFortunaNoDriverScraper:
    """NoDriver-based scraper for visible eFortuna LoL match-winner markets."""

    bookmaker = "efortuna"
    scraper_version = SCRAPER_VERSION

    def __init__(
        self,
        start_url: str = EFORTUNA_LOL_URL,
        headless: bool | None = None,
        league_urls: list[str] | None = None,
    ) -> None:
        self.start_url = start_url
        self.headless = headless
        self.league_urls = league_urls or EFORTUNA_LOL_LEAGUE_URLS

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Open eFortuna league pages and return normalized LoL odds snapshots."""

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        snapshots: list[RawOddsSnapshot] = []
        async with NoDriverClient(headless=self.headless) as client:
            for index, url in enumerate(self.league_urls, start=1):
                tab = await client.open(url)
                await self.wait_for_render(tab)
                await self.accept_cookies(tab)
                prefix = f"efortuna_{index:02d}_{timestamp}"
                html_path, screenshot_path = await client.save_debug_artifacts(tab, prefix)
                body_text = await self.extract_body_text(tab)
                body_path = client.debug_dir / f"{prefix}_body.txt"
                body_path.write_text(body_text, encoding="utf-8")
                cards = parse_efortuna_lol_offers(body_text, source_url=url)
                snapshots.extend(
                    snapshot
                    for card in cards
                    if (snapshot := self.parse_match_card(card, html_path or str(body_path), screenshot_path))
                )

        print(f"eFortuna scraper captured {len(snapshots)} snapshots from {len(self.league_urls)} league pages.")
        return snapshots

    async def wait_for_render(self, tab: Any, seconds: float = 5.0) -> None:
        """Wait for eFortuna page to render."""

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
        card: ParsedEFortunaOffer,
        html_path: str | None = None,
        screenshot_path: str | None = None,
    ) -> RawOddsSnapshot | None:
        """Convert one parsed eFortuna card into RawOddsSnapshot."""

        raw_payload = {
            "raw_text": card.raw_text,
            "source_url": card.source_url,
            "offer_url": card.offer_url,
            "note": "eFortuna fixture cards route to per-event offer pages built from the league URL and teams slug.",
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
            source_url=card.source_url,
            offer_url=card.offer_url,
            market_type="match_winner",
            is_live=False,
            scraper_name="efortuna_nodriver_lol_match_winner",
            scraper_version=SCRAPER_VERSION,
            raw_payload=raw_payload,
            page_html_path=str(html_path) if html_path else None,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            best_of=card.best_of,
        )
