"""Generic NoDriver scraper skeleton for one bookmaker page.

This is intentionally conservative: each bookmaker will need selectors tuned
after inspecting the live page. The class provides the NoDriver lifecycle and
debug artifact capture so implementing STS/Betclic/Fortuna is localized.
"""

from __future__ import annotations

from datetime import UTC, datetime

from betting_app.scrapers.base import RawOddsSnapshot
from betting_app.scrapers.nodriver_client import NoDriverClient


class GenericNoDriverScraper:
    """Skeleton scraper that opens a URL and stores debug artifacts."""

    bookmaker = "generic"
    start_url: str

    def __init__(self, bookmaker: str, start_url: str, headless: bool | None = None) -> None:
        self.bookmaker = bookmaker
        self.start_url = start_url
        self.headless = headless

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Open bookmaker page and return snapshots.

        Returns an empty list until bookmaker-specific selectors are implemented.
        Debug HTML/screenshot are saved for selector development.
        """

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        async with NoDriverClient(headless=self.headless) as client:
            tab = await client.open(self.start_url)
            html_path, screenshot_path = await client.save_debug_artifacts(tab, f"{self.bookmaker}_{timestamp}")
        print(
            f"Opened {self.start_url}. Implement selectors for {self.bookmaker}. "
            f"Debug html={html_path}, screenshot={screenshot_path}"
        )
        return []
