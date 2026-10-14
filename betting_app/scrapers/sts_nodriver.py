"""STS League of Legends prematch scraper using nodriver browser automation.

The STS site (www.sts.pl) is a fully client-side rendered Angular SPA. When
fetched with plain HTTP, the HTML contains no odds data — only the Angular
bootstrap scripts. However, when a real browser renders the page, the Angular
app bootstraps and embeds all SBK data in a <script> tag as SSR transfer state
JSON (key "sbk-exporter-sports-ssr").

This scraper uses nodriver to:
1. Navigate to the STS LoL esport page
2. Wait for the Angular app to render and populate the DOM
3. Extract the SSR transfer state JSON from the rendered page's script tags
4. Parse the compact SBK data structure for LoL match-winner odds

No API key or CF Access headers are needed — the data comes from the rendered
DOM, same as what the user sees in the browser.

Scope: prematch League of Legends, market "Zwycięzca meczu" only.
Returns both RawOutcomeOddsSnapshot (atomic) and RawOddsSnapshot (legacy pair).
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

from betting_app.scrapers.base import RawOddsSnapshot, RawOutcomeOddsSnapshot
from betting_app.scrapers.nodriver_client import NoDriverClient


STS_LOL_URL = "https://www.sts.pl/zaklady/esport/league-of-legends"
STS_ESPORT_SPORT_ID = "156"
STS_LOL_CATEGORY_ID = "992"
STS_LOL_CATEGORY_NAME = "League of Legends"
STS_ESPORT_SPORT_NAME = "Esport"
STS_MATCH_WINNER_MARKET_NAME = "Zwycięzca meczu"
SCRAPER_VERSION = "sts-nodriver-lol-match-winner-0.3"

# JS to extract SSR transfer state from the rendered page DOM.
# The Angular app embeds a <script> tag whose text content is a JSON object
# with key "sbk-exporter-sports-ssr". We find it and return the inner data.
_EXTRACT_SSR_JS = """
(() => {
    const scripts = document.querySelectorAll('script');
    for (const s of scripts) {
        try {
            const text = s.textContent.trim();
            if (text.startsWith('{') && text.includes('sbk-exporter')) {
                const parsed = JSON.parse(text);
                const key = 'sbk-exporter-sports-ssr';
                if (key in parsed) return JSON.stringify(parsed[key]);
                if ('B' in parsed && 'P' in parsed) return JSON.stringify(parsed);
            }
        } catch(e) {}
    }
    return null;
})()
"""


class STSNoDriverScraper:
    """Scrape STS LoL prematch match-winner odds via nodriver browser automation."""

    bookmaker = "sts"
    scraper_version = SCRAPER_VERSION

    def __init__(
        self,
        start_url: str = STS_LOL_URL,
        headless: bool | None = None,
        *,
        sport_id: str = STS_ESPORT_SPORT_ID,
        category_id: str = STS_LOL_CATEGORY_ID,
        include_legacy_pair_snapshots: bool = True,
    ) -> None:
        self.start_url = start_url or STS_LOL_URL
        self.headless = headless
        self.sport_id = sport_id
        self.category_id = category_id
        self.include_legacy_pair_snapshots = include_legacy_pair_snapshots
        self.last_request_url: str | None = None
        self.last_total_count: int = 0
        self.last_fixture_count: int = 0

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot | RawOutcomeOddsSnapshot]:
        """Open STS LoL page in nodriver, extract SSR data, return match-winner odds."""

        scraped_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        self.last_request_url = self.start_url

        async with NoDriverClient(headless=self.headless) as client:
            tab = await client.open(self.start_url)
            await self._wait_for_render(tab, 8.0)
            await self._accept_cookies(tab)

            # Extract SSR data from the rendered DOM
            data = await self._extract_ssr_data(tab)

            # Save debug artifacts
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            html_path, screenshot_path = await client.save_debug_artifacts(
                tab, f"sts_{timestamp}"
            )

        if data is None:
            raise RuntimeError(
                "Could not extract SSR transfer state from STS page DOM. "
                "The Angular app may not have rendered, or the site structure changed."
            )

        snapshots = self.parse_lol_match_winner_snapshot(
            data,
            scraped_at=scraped_at,
            source_url=self.start_url,
        )
        outcome_count = sum(
            isinstance(s, RawOutcomeOddsSnapshot) for s in snapshots
        )
        pair_count = sum(isinstance(s, RawOddsSnapshot) for s in snapshots)
        print(
            f"STS nodriver captured {self.last_fixture_count} LoL fixtures, "
            f"{outcome_count} outcome odds, {pair_count} two-sided match-winner snapshots. "
            f"URL={self.start_url}"
        )
        return snapshots

    async def _wait_for_render(self, tab: Any, seconds: float = 8.0) -> None:
        """Wait for the Angular SPA to render content."""

        import asyncio

        _ = tab
        await asyncio.sleep(seconds)

    async def _accept_cookies(self, tab: Any) -> None:
        """Best-effort cookie modal acceptance."""

        try:
            await tab.evaluate(
                """Array.from(document.querySelectorAll('button'))
                .find(button => /akcept|zgadzam|accept/i.test(button.innerText || ''))?.click()"""
            )
        except Exception:
            return

    async def _extract_ssr_data(self, tab: Any) -> dict[str, Any] | None:
        """Extract the SBK SSR transfer state from the rendered page DOM.

        Uses JavaScript evaluation to find the <script> tag containing
        the "sbk-exporter-sports-ssr" key and return the inner data structure.
        """

        result = await tab.evaluate(_EXTRACT_SSR_JS)
        if result is None:
            return None

        # nodriver may return the result as a string or already-parsed object
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                return None
        elif isinstance(result, dict):
            parsed = result
        else:
            return None

        if not isinstance(parsed, dict):
            return None

        # Validate the expected structure
        if "B" in parsed and "P" in parsed:
            return parsed

        return None

    # ------------------------------------------------------------------
    # Parsing logic — unchanged from the original scraper, works with the
    # compact SBK data structure {B: {S: ...}, P: ...}
    # ------------------------------------------------------------------

    def parse_lol_match_winner_snapshot(
        self,
        data: dict[str, Any],
        *,
        scraped_at: str,
        source_url: str,
    ) -> list[RawOddsSnapshot | RawOutcomeOddsSnapshot]:
        """Parse LoL fixtures and match-winner markets from SBK SSR payload."""

        offers = data.get("P") or {}
        sports = ((data.get("B") or {}).get("S") or {})
        esport = sports.get(self.sport_id) or {}
        category = (esport.get("C") or {}).get(self.category_id) or {}
        tournaments = category.get("T") or {}
        snapshots: list[RawOddsSnapshot | RawOutcomeOddsSnapshot] = []
        fixture_count = 0

        for tournament_id, tournament in tournaments.items():
            league_name = self._name(tournament) or str(tournament_id)
            fixtures = tournament.get("FX") or tournament.get("F") or {}
            for fixture_id, fixture in fixtures.items():
                fixture_count += 1
                parsed = self.parse_fixture_match_winner(
                    fixture_id=str(fixture_id),
                    fixture=fixture,
                    offers=offers,
                    tournament_id=str(tournament_id),
                    league_name=league_name,
                    scraped_at=scraped_at,
                    source_url=source_url,
                )
                snapshots.extend(parsed)

        self.last_fixture_count = fixture_count
        self.last_total_count = len(snapshots)
        return snapshots

    def parse_fixture_match_winner(
        self,
        *,
        fixture_id: str,
        fixture: dict[str, Any],
        offers: dict[str, Any],
        tournament_id: str,
        league_name: str,
        scraped_at: str,
        source_url: str,
    ) -> list[RawOddsSnapshot | RawOutcomeOddsSnapshot]:
        """Parse match winner outcomes for one fixture."""

        home = self._name(fixture.get("H")) or ""
        away = self._name(fixture.get("A")) or ""
        starts_at = fixture.get("t") or fixture.get("T")
        offer_url = self.build_offer_url(home=home, away=away, fixture_id=fixture_id)
        offer_ids = list((fixture.get("a") or {}).keys())
        snapshots: list[RawOddsSnapshot | RawOutcomeOddsSnapshot] = []

        for offer_id in offer_ids:
            offer = offers.get(offer_id) or {}
            markets = offer.get("m") or {}
            for market_id, market in markets.items():
                lines = market.get("l") or {}
                for line_id, line in lines.items():
                    market_name = str(line.get("n") or market.get("n") or "")
                    if not self.is_match_winner_market(market_name):
                        continue
                    outcomes = line.get("o") or {}
                    if not outcomes:
                        continue
                    outcome_snapshots: list[RawOutcomeOddsSnapshot] = []
                    odds_by_side: dict[str, float] = {}
                    for outcome_id, outcome in outcomes.items():
                        if not isinstance(outcome, dict) or outcome.get("O") is None:
                            continue
                        outcome_side = self.infer_outcome_side(str(outcome_id))
                        decimal_odds = float(outcome["O"])
                        if outcome_side:
                            odds_by_side[outcome_side] = decimal_odds
                        outcome_name = self.outcome_name(
                            outcome_id=str(outcome_id),
                            outcome_side=outcome_side,
                            home=home,
                            away=away,
                        )
                        snapshot = RawOutcomeOddsSnapshot(
                            bookmaker=self.bookmaker,
                            bookmaker_event_id=fixture_id,
                            raw_team_a=home,
                            raw_team_b=away,
                            decimal_odds=decimal_odds,
                            outcome_key=f"{offer_id}:{market_id}:{line_id}:{outcome_id}",
                            outcome_name=outcome_name,
                            outcome_side=outcome_side,
                            market_key=f"{offer_id}:{market_id}:{line_id}",
                            market_name=market_name or STS_MATCH_WINNER_MARKET_NAME,
                            line_id=str(line_id),
                            line_name=market_name or STS_MATCH_WINNER_MARKET_NAME,
                            is_extra_market=False,
                            scraped_at=scraped_at,
                            match_start_time=starts_at,
                            sport_id=self.sport_id,
                            sport_name=STS_ESPORT_SPORT_NAME,
                            category_id=self.category_id,
                            category_name=STS_LOL_CATEGORY_NAME,
                            league_id=tournament_id,
                            league_name=league_name,
                            source_url=source_url,
                            offer_url=offer_url,
                            scraper_name="sts_nodriver_lol_match_winner",
                            scraper_version=SCRAPER_VERSION,
                            raw_payload={
                                "fixture_id": fixture_id,
                                "offer_id": offer_id,
                                "market_id": market_id,
                                "line_id": line_id,
                                "outcome_id": outcome_id,
                                "fixture": fixture,
                                "outcome": outcome,
                            },
                        )
                        outcome_snapshots.append(snapshot)

                    snapshots.extend(outcome_snapshots)
                    if (
                        self.include_legacy_pair_snapshots
                        and "a" in odds_by_side
                        and "b" in odds_by_side
                    ):
                        snapshots.append(
                            RawOddsSnapshot(
                                bookmaker=self.bookmaker,
                                raw_team_a=home,
                                raw_team_b=away,
                                odds_a=odds_by_side["a"],
                                odds_b=odds_by_side["b"],
                                scraped_at=scraped_at,
                                raw_league=league_name,
                                match_start_time=starts_at,
                                source_url=source_url,
                                offer_url=offer_url,
                                market_type="match_winner",
                                is_live=False,
                                scraper_name="sts_nodriver_lol_match_winner",
                                scraper_version=SCRAPER_VERSION,
                                raw_payload={
                                    "fixture_id": fixture_id,
                                    "offer_id": offer_id,
                                    "market_id": market_id,
                                    "line_id": line_id,
                                    "offer_url": offer_url,
                                    "home": home,
                                    "away": away,
                                },
                            )
                        )
        return snapshots

    @staticmethod
    def build_offer_url(*, home: str, away: str, fixture_id: str) -> str:
        """Build the STS prematch detail URL used for manual close-odds checks."""

        team_slug = slugify(f"{home} {away}") or "mecz"
        return f"https://www.sts.pl/kursy/{team_slug}/{fixture_id}"

    @staticmethod
    def _name(value: Any) -> str | None:
        """Extract display name from STS compact objects."""

        if isinstance(value, dict):
            return value.get("n") or value.get("N")
        if value is None:
            return None
        return str(value)

    @staticmethod
    def is_match_winner_market(market_name: str) -> bool:
        """Return True for STS match-winner market labels."""

        normalized = market_name.strip().lower()
        return normalized in {"zwycięzca meczu", "zwyciezca meczu", "mecz"}

    @staticmethod
    def infer_outcome_side(outcome_id: str) -> str | None:
        """Infer side for STS esports match-winner outcome IDs.

        In observed LoL prematch data outcome id 4 corresponds to home/team A,
        and outcome id 5 corresponds to away/team B. Football 1X2 uses other IDs,
        but LoL match winner is two-sided and currently uses 4/5.
        """

        if str(outcome_id) == "4":
            return "a"
        if str(outcome_id) == "5":
            return "b"
        return None

    @staticmethod
    def outcome_name(
        *, outcome_id: str, outcome_side: str | None, home: str, away: str
    ) -> str:
        """Build readable outcome name."""

        if outcome_side == "a":
            return home or "1"
        if outcome_side == "b":
            return away or "2"
        return str(outcome_id)


def slugify(value: str) -> str:
    """Create a simple STS-compatible-ish URL slug from team names."""

    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value.lower()).strip("-")
    return re.sub(r"-+", "-", slug)
