"""Betclic NoDriver scraper for League of Legends prematch match-winner odds."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from betting_app.scrapers.base import RawOddsSnapshot
from betting_app.scrapers.betclic_parser import (
    BETCLIC_LOL_URL,
    ParsedBetclicOffer,
    extract_event_links,
    parse_betclic_lol_offers,
    parse_event_link,
)
from betting_app.scrapers.nodriver_client import NoDriverClient


SCRAPER_VERSION = "betclic-nodriver-0.1"


BETCLIC_BASE_URL = "https://www.betclic.pl"
BETCLIC_LOL_COMPETITION_RE = re.compile(r"^https://www\.betclic\.pl/lol-slol/[^/?#]+-c\d+(?:$|[/?#])")
BETCLIC_LOL_COMPETITION_FALLBACK_URLS = [
    "https://www.betclic.pl/lol-slol/msi-c21940",
    "https://www.betclic.pl/lol-slol/esports-world-cup-c35516",
    "https://www.betclic.pl/lol-slol/lck-c23480",
    "https://www.betclic.pl/lol-slol/liga-regional-norte-c40600",
]
BETCLIC_PAGE_TIMEOUT_SECONDS = 55.0
BETCLIC_TOTAL_BUDGET_SECONDS = 240.0
BETCLIC_MAX_COMPETITION_PAGES = 5


class BetclicNoDriverScraper:
    """NoDriver-based scraper for Betclic League of Legends match-winner markets."""

    bookmaker = "betclic"

    def __init__(self, start_url: str = BETCLIC_LOL_URL, headless: bool | None = None) -> None:
        self.start_url = start_url
        self.headless = headless
        self.start_urls = [start_url]

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Open Betclic and return normalized LoL odds snapshots."""

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        snapshots: list[RawOddsSnapshot] = []
        debug_paths: list[str] = []
        started_at = time.monotonic()

        urls_to_scrape = list(dict.fromkeys(self.start_urls))
        discovered_competitions: list[str] = []
        planned_page_count = len(urls_to_scrape)
        for index, url in enumerate(urls_to_scrape):
            if self.time_budget_exhausted(started_at):
                print(f"Betclic scraper stopped before {url}: total time budget exhausted")
                break
            page_snapshots, page_debug_paths, page_competition_links = await self.scrape_page_with_timeout(
                url,
                timestamp,
                index,
                retry_forbidden=True,
            )
            snapshots.extend(page_snapshots)
            debug_paths.extend(page_debug_paths)
            discovered_competitions.extend(page_competition_links)

        if self.start_url.rstrip("/") == BETCLIC_LOL_URL.rstrip("/"):
            discovered_urls = list(dict.fromkeys(discovered_competitions))
            # Normal path: scrape every LoL competition tab discovered from the Betclic UI.
            # Safety path: if the landing page is temporarily blocked (403) and discovery
            # returns nothing, use a small known-good fallback list so the scheduled job does
            # not silently produce zero snapshots.
            competition_urls = discovered_urls or BETCLIC_LOL_COMPETITION_FALLBACK_URLS
            dynamic_urls = [url for url in competition_urls if url not in set(urls_to_scrape)]
            dynamic_urls = list(dict.fromkeys(dynamic_urls))[:BETCLIC_MAX_COMPETITION_PAGES]
            planned_page_count += len(list(dict.fromkeys(dynamic_urls)))
            for offset, url in enumerate(dynamic_urls, start=len(urls_to_scrape)):
                if self.time_budget_exhausted(started_at):
                    print(
                        f"Betclic scraper stopped before {url}: "
                        f"total time budget {BETCLIC_TOTAL_BUDGET_SECONDS:.0f}s exhausted"
                    )
                    break
                # Betclic can start returning 403 after many quick same-session navigations.
                # Use a fresh browser session per discovered competition page and a short
                # pause between pages to make the scheduled scraper more stable.
                await asyncio.sleep(1.0)
                page_snapshots, page_debug_paths, _ = await self.scrape_page_with_timeout(
                    url,
                    timestamp,
                    offset,
                    retry_forbidden=False,
                )
                snapshots.extend(page_snapshots)
                debug_paths.extend(page_debug_paths)

        snapshots = self.deduplicate_snapshots(snapshots)

        print(
            f"Betclic scraper captured {len(snapshots)} snapshots from {planned_page_count} planned pages "
            f"({len(set(discovered_competitions))} discovered competitions, "
            f"max competition pages={BETCLIC_MAX_COMPETITION_PAGES}, "
            f"elapsed={time.monotonic() - started_at:.1f}s). "
            f"Competition URLs={list(dict.fromkeys(discovered_competitions))}. "
            f"Debug bodies={debug_paths}"
        )
        return snapshots

    async def scrape_page_with_timeout(
        self,
        url: str,
        timestamp: str,
        index: int,
        *,
        retry_forbidden: bool,
    ) -> tuple[list[RawOddsSnapshot], list[str], list[str]]:
        """Scrape one page with a hard timeout so scheduler jobs cannot hang."""

        try:
            return await asyncio.wait_for(
                self.scrape_page(url, timestamp, index, retry_forbidden=retry_forbidden),
                timeout=BETCLIC_PAGE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            print(f"Betclic page scrape timed out after {BETCLIC_PAGE_TIMEOUT_SECONDS:.0f}s for {url}")
            return [], [], []

    async def scrape_page(
        self,
        url: str,
        timestamp: str,
        index: int,
        *,
        retry_forbidden: bool,
    ) -> tuple[list[RawOddsSnapshot], list[str], list[str]]:
        """Scrape one Betclic page and discover competition links visible there."""

        max_attempts = 2 if retry_forbidden else 1
        collected_debug_paths: list[str] = []
        for attempt in range(max_attempts):
            async with NoDriverClient(headless=self.headless) as client:
                tab = await client.open(url)
                await self.wait_for_render(tab)
                await self.accept_cookies(tab)
                await client.scroll_to_bottom(tab, step=500, pause=0.8, passes=2)
                await self.wait_for_render(tab, seconds=3.0)
                debug_prefix = f"betclic_{timestamp}_{index}" if attempt == 0 else f"betclic_{timestamp}_{index}_retry{attempt}"
                html_path, screenshot_path = await client.save_debug_artifacts(tab, debug_prefix)
                body_text = await self.extract_body_text(tab)
                body_path = client.debug_dir / f"{debug_prefix}_body.txt"
                body_path.write_text(body_text, encoding="utf-8")
                collected_debug_paths.append(str(body_path))

                if self.is_forbidden_body(body_text) and attempt + 1 < max_attempts:
                    print(f"Betclic returned 403 for {url}; retrying attempt {attempt + 2}/{max_attempts}")
                    await asyncio.sleep(10.0 * (attempt + 1))
                    continue

                if self.is_forbidden_body(body_text):
                    print(f"Betclic returned 403 for {url}; skipping page")
                    return [], collected_debug_paths, []

                cards = await self.extract_match_cards(tab, body_text=body_text)
                event_links = await self.extract_event_links(tab)
                competition_links = await self.extract_competition_links(tab)
                if html_path:
                    if not event_links:
                        event_links = self.extract_event_links_from_html(Path(html_path))
                    if not competition_links:
                        competition_links = self.extract_competition_links_from_html(Path(html_path))
                cards = self.attach_offer_links(cards, event_links)
                snapshots = [
                    snapshot
                    for card in cards
                    if (snapshot := self.parse_match_card(card, html_path or str(body_path), screenshot_path, url))
                ]
                return snapshots, collected_debug_paths, competition_links

        return [], collected_debug_paths, []

    @staticmethod
    def time_budget_exhausted(started_at: float) -> bool:
        """Return True when the scraper should stop before scheduler-level timeout."""

        return time.monotonic() - started_at >= BETCLIC_TOTAL_BUDGET_SECONDS

    async def wait_for_render(self, tab: Any, seconds: float = 8.0) -> None:
        """Wait for Betclic SPA to render content."""

        import asyncio

        _ = tab
        await asyncio.sleep(seconds)

    async def accept_cookies(self, tab: Any) -> None:
        """Best-effort cookie modal acceptance."""

        try:
            await tab.evaluate(
                """Array.from(document.querySelectorAll('button'))
                .find(button => button.innerText && button.innerText.includes('Zaakceptuj'))?.click()"""
            )
        except Exception:
            return

    async def extract_body_text(self, tab: Any) -> str:
        """Return rendered page body text."""

        body_text = await tab.evaluate("document.body ? document.body.innerText : ''")
        return str(body_text or "")

    async def extract_match_cards(self, tab: Any, body_text: str | None = None) -> list[ParsedBetclicOffer]:
        """Extract parsed Betclic LoL match cards from rendered text."""

        _ = tab
        return parse_betclic_lol_offers(body_text or "")

    async def extract_event_links(self, tab: Any) -> list[dict[str, str]]:
        """Extract Betclic event deep links from rendered HTML via parsel."""

        try:
            html = await tab.get_content()
        except Exception:
            return []
        return [{"text": text, "href": href} for text, href in extract_event_links(str(html or ""))]

    async def extract_competition_links(self, tab: Any) -> list[str]:
        """Extract LoL competition page URLs from the rendered Betclic navigation."""

        try:
            raw_links = await tab.evaluate(
                """
                Array.from(document.querySelectorAll('a[href]')).map(anchor => ({
                    text: anchor.innerText || anchor.textContent || '',
                    href: anchor.href || anchor.getAttribute('href') || ''
                }))
                """
            )
        except Exception:
            return []
        if not isinstance(raw_links, list):
            return []
        urls: list[str] = []
        for item in raw_links:
            if not isinstance(item, dict):
                continue
            href = str(item.get("href") or "")
            normalized = self.normalize_betclic_url(href)
            if self.is_lol_competition_url(normalized):
                urls.append(normalized)
        return list(dict.fromkeys(urls))

    def extract_event_links_from_html(self, html_path: Path) -> list[dict[str, str]]:
        """Fallback event-link extraction from saved Betclic SSR/debug HTML."""

        try:
            html = html_path.read_text(encoding="utf-8")
        except OSError:
            return []
        return [{"text": text, "href": href} for text, href in extract_event_links(html)]

    def extract_competition_links_from_html(self, html_path: Path) -> list[str]:
        """Fallback competition-link extraction from saved Betclic HTML."""

        try:
            html = html_path.read_text(encoding="utf-8")
        except OSError:
            return []
        return list(
            dict.fromkeys(
                url
                for href in re.findall(r'href=["\']([^"\']+)["\']', html)
                if self.is_lol_competition_url(url := self.normalize_betclic_url(href))
            )
        )

    @staticmethod
    def normalize_betclic_url(href: str) -> str:
        """Return an absolute Betclic URL without query/fragment noise."""

        href = href.strip()
        if href.startswith("/"):
            href = BETCLIC_BASE_URL + href
        href = href.split("#", 1)[0].split("?", 1)[0].rstrip("/")
        return href

    @staticmethod
    def is_lol_competition_url(url: str) -> bool:
        """True for Betclic LoL competition tabs, false for event pages."""

        return bool(BETCLIC_LOL_COMPETITION_RE.match(url)) and "-m" not in url

    @staticmethod
    def is_forbidden_body(body_text: str) -> bool:
        """Detect Betclic anti-bot/temporary forbidden page."""

        normalized = body_text.lower()
        return "error 403" in normalized or "forbidden" in normalized or "0x2005002" in normalized

    def attach_offer_links(
        self,
        cards: list[ParsedBetclicOffer],
        event_links: list[dict[str, str]],
    ) -> list[ParsedBetclicOffer]:
        """Attach Betclic URLs and add event-link-only cards.

        Betclic renders some valid event anchors in the HTML that are not
        present in ``document.body.innerText``.  The body parser sees only the
        visible cards, so after attaching links to those cards we also parse
        unused event-link text into standalone offers.
        """

        enriched: list[ParsedBetclicOffer] = []
        used_hrefs: set[str] = set()
        seen_keys: set[tuple[str, str, str, float, float]] = set()
        for card in cards:
            if card.source_url != BETCLIC_LOL_URL:
                seen_keys.add((card.source_url, card.raw_team_a, card.raw_team_b, card.odds_a, card.odds_b))
                enriched.append(card)
                continue
            match = self.find_link_for_card(card, event_links, used_hrefs)
            if match is None:
                seen_keys.add((card.source_url, card.raw_team_a, card.raw_team_b, card.odds_a, card.odds_b))
                enriched.append(card)
                continue
            href = str(match.get("href"))
            used_hrefs.add(href)
            linked_card = replace(card, source_url=href, bookmaker_event_id=self.extract_match_id(href))
            seen_keys.add(
                (linked_card.source_url, linked_card.raw_team_a, linked_card.raw_team_b, linked_card.odds_a, linked_card.odds_b)
            )
            enriched.append(linked_card)

        for item in event_links:
            href = str(item.get("href") or "")
            if not href or href in used_hrefs:
                continue
            parsed = parse_event_link(str(item.get("text") or ""), href)
            if parsed is None:
                continue
            key = (parsed.source_url, parsed.raw_team_a, parsed.raw_team_b, parsed.odds_a, parsed.odds_b)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            enriched.append(parsed)
        return enriched

    @staticmethod
    def find_link_for_card(
        card: ParsedBetclicOffer,
        event_links: list[dict[str, str]],
        used_hrefs: set[str],
    ) -> dict[str, str] | None:
        """Find a DOM link whose visible text contains both teams."""

        team_a = card.raw_team_a.lower()
        team_b = card.raw_team_b.lower()
        team_a_compact = compact_text(card.raw_team_a)
        team_b_compact = compact_text(card.raw_team_b)
        for item in event_links:
            href = str(item.get("href") or "")
            if href in used_hrefs:
                continue
            text = str(item.get("text") or "").lower()
            compact = compact_text(text)
            if (team_a in text and team_b in text) or (team_a_compact in compact and team_b_compact in compact):
                return item
        return None

    @staticmethod
    def extract_match_id(url: str) -> str | None:
        """Extract Betclic numeric match ID from an event URL."""

        match = re.search(r"-m(?P<id>\d+)(?:$|[/?#])", url)
        return match.group("id") if match else None

    def parse_match_card(
        self,
        card: ParsedBetclicOffer,
        html_path: str | None = None,
        screenshot_path: str | None = None,
        source_page_url: str | None = None,
    ) -> RawOddsSnapshot | None:
        """Parse one Betclic match card into the common snapshot format."""

        start_label = " ".join(part for part in [card.date_label, card.start_time_label] if part) or None
        offer_url = card.source_url if card.source_url != BETCLIC_LOL_URL else None
        source_url = source_page_url or self.start_url
        raw_payload = {
            "bookmaker_event_id": card.bookmaker_event_id,
            "market_count_label": card.market_count_label,
            "raw_text": card.raw_text,
            "source_url": source_url,
            "offer_url": offer_url,
        }
        return RawOddsSnapshot(
            bookmaker=self.bookmaker,
            raw_team_a=card.raw_team_a,
            raw_team_b=card.raw_team_b,
            odds_a=card.odds_a,
            odds_b=card.odds_b,
            scraped_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
            raw_league=card.league,
            match_start_time=start_label,
            source_url=source_url,
            offer_url=offer_url,
            market_type="match_winner",
            is_live=False,
            scraper_name="betclic_nodriver_lol_match_winner",
            scraper_version=SCRAPER_VERSION,
            raw_payload=raw_payload,
            page_html_path=str(html_path) if html_path else None,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
        )

    @staticmethod
    def deduplicate_snapshots(snapshots: list[RawOddsSnapshot]) -> list[RawOddsSnapshot]:
        """Remove duplicates when the same event appears on aggregate and competition pages."""

        deduped: list[RawOddsSnapshot] = []
        seen: set[tuple[str, str, str | None, float, float, str | None]] = set()
        for snapshot in snapshots:
            key = (
                snapshot.raw_team_a,
                snapshot.raw_team_b,
                snapshot.match_start_time,
                snapshot.odds_a,
                snapshot.odds_b,
                snapshot.offer_url,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(snapshot)
        return deduped


def compact_text(value: str) -> str:
    """Normalize text for matching team names split by DOM line wraps."""

    return re.sub(r"\s+", "", value.lower())
