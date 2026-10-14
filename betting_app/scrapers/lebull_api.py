"""Lebull API scraper for League of Legends prematch match-winner odds."""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

from betting_app.scrapers.base import RawOddsSnapshot


LEBULL_ESPORT_URL = "https://www.lebull.pl/pl/zaklady-sportowe?page=/esport"
LEBULL_SSR_URL = "https://lebullpl-ssr.boxwebcdn.work/pl/esport?currency=PLN&parent=www.lebull.pl&&1"
LEBULL_API_BASE = "https://betting-platform.prod.sbteam.xyz"
SCRAPER_VERSION = "lebull-api-lol-match-winner-0.1"


class LebullApiScraper:
    """API scraper for Lebull LoL prematch match-winner markets."""

    bookmaker = "lebull"
    scraper_version = SCRAPER_VERSION

    def __init__(self, start_url: str = LEBULL_ESPORT_URL, headless: bool | None = None) -> None:
        self.start_url = start_url
        self.headless = headless
        self.tenant_id: str | None = None

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Fetch LoL league IDs, then event details for match-winner odds."""

        _ = self.headless
        self.tenant_id = self.fetch_tenant_id()
        league_ids = self.fetch_lol_league_ids()
        games: list[dict[str, Any]] = []
        for league_id in league_ids:
            games.extend(self.fetch_league_games(league_id))
        snapshots: list[RawOddsSnapshot] = []
        seen: set[int] = set()
        for game in games:
            event_id = int(game.get("eventId") or 0)
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            detail = self.fetch_event_detail(event_id)
            observed_at = datetime.now(UTC)
            snapshot = self.parse_event_detail(detail, observed_at=observed_at)
            if snapshot:
                snapshots.append(snapshot)
        print(f"Lebull API captured {len(snapshots)} LoL prematch snapshots from {len(league_ids)} leagues.")
        return snapshots

    def fetch_tenant_id(self) -> str:
        """Read public tenant id from Lebull SSR env block."""

        req = urllib.request.Request(LEBULL_SSR_URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*"})
        with urllib.request.urlopen(req, timeout=30) as response:
            text = response.read().decode("utf-8", "replace")
        match = re.search(r'"BETTING_PLATFORM_SVC_TENANT_ID"\s*:\s*"([^"]+)"', text)
        if not match:
            raise RuntimeError("Could not find Lebull tenant id in SSR page")
        return match.group(1)

    def request_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET Lebull betting-platform JSON with tenant header."""

        if not self.tenant_id:
            raise RuntimeError("Lebull tenant_id is not initialized")
        query = urllib.parse.urlencode(params or {})
        url = f"{LEBULL_API_BASE}{path}" + (f"?{query}" if query else "")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
                "Origin": "https://www.lebull.pl",
                "Referer": "https://www.lebull.pl/",
                "X-Auth-Tenant-Id": self.tenant_id,
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    def fetch_lol_league_ids(self) -> list[int]:
        """Return Lebull League of Legends league IDs from sports tree."""

        data = self.request_json("/sports", {"currency": "PLN", "language": "pl"})
        sports = data.get("sports") if isinstance(data, dict) else []
        league_ids: list[int] = []
        for sport in sports or []:
            if int(sport.get("sportId") or 0) != 53:
                continue
            for country in sport.get("countries") or []:
                if str(country.get("countryName")) != "League of Legends":
                    continue
                for league in country.get("leagues") or []:
                    league_id = int(league.get("leagueId") or 0)
                    if league_id:
                        league_ids.append(league_id)
        return league_ids

    def fetch_league_games(self, league_id: int) -> list[dict[str, Any]]:
        """Fetch all games for one LoL league."""

        data = self.request_json(f"/sports/53/leagues/{league_id}/games", {"currency": "PLN"})
        return list(data or [])

    def fetch_event_detail(self, event_id: int) -> dict[str, Any]:
        """Fetch prematch event detail with markets."""

        return dict(self.request_json(f"/games/{event_id}/prematch", {"currency": "PLN"}) or {})

    def parse_event_detail(self, event: dict[str, Any], *, observed_at: datetime | None = None) -> RawOddsSnapshot | None:
        """Parse Lebull event detail into RawOddsSnapshot."""

        if not event or str(event.get("countryName")) != "League of Legends":
            return None
        if event.get("isOutright"):
            return None
        team_a = str(event.get("teamA") or "").strip()
        team_b = str(event.get("teamB") or "").strip()
        if not team_a or not team_b:
            return None
        market = next(
            (st for st in event.get("stakeTypes") or [] if str(st.get("stakeTypeName", "")).lower() in {"result (2 way)", "winner"}),
            None,
        )
        if not market:
            return None
        stakes = sorted(market.get("stakes") or [], key=lambda item: str(item.get("stakeCode") or ""))
        odds_a = parse_float(stakes[0].get("betFactor")) if stakes else None
        odds_b = parse_float(stakes[1].get("betFactor")) if len(stakes) > 1 else None
        inactive = (
            event.get("isActive") is False or event.get("isDeleted") is True
            or market.get("isActive") is False or market.get("isDeleted") is True
            or len(stakes) != 2
            or any(stake.get("isActive") is not True or stake.get("isDeleted") is True for stake in stakes[:2])
        )
        if inactive:
            odds_a = odds_b = None
        live = event.get("isLive")
        is_live = live if isinstance(live, bool) else bool(live) if type(live) is int and live in (0, 1) else None
        event_id = str(event.get("eventId"))
        detail_api = f"{LEBULL_API_BASE}/games/{event_id}/prematch?currency=PLN"
        offer_url = build_offer_url(event_id=event_id, is_live=is_live)
        return RawOddsSnapshot(
            bookmaker=self.bookmaker,
            raw_team_a=team_a,
            raw_team_b=team_b,
            odds_a=odds_a,
            odds_b=odds_b,
            scraped_at=(observed_at or datetime.now(UTC)).isoformat(),
            raw_league=str(event.get("leagueName") or ""),
            match_start_time=parse_timestamp(event.get("timestamp")) or str(event.get("date") or ""),
            source_url=LEBULL_ESPORT_URL,
            offer_url=offer_url,
            market_type="match_winner",
            is_live=is_live,
            scraper_name="lebull_api_lol_match_winner",
            scraper_version=SCRAPER_VERSION,
            raw_payload={"event": event, "market": market, "stakes": stakes, "detail_api": detail_api},
        )


def build_offer_url(*, event_id: str, is_live: bool | None = None) -> str:
    """Build a user-facing Lebull SPA link for one event.

    Lebull's public shell is served at /pl/zaklady-sportowe and embeds the
    sportsbook route in the `page` query parameter. The sportsbook SSR bundle
    links event cards as `/event/<eventId>?isLive=<bool>`.
    """

    inner_route = f"/event/{event_id}"
    if is_live is not None:
        inner_route += f"?isLive={str(is_live).lower()}"
    return "https://www.lebull.pl/pl/zaklady-sportowe?" + urllib.parse.urlencode({"page": inner_route})


def parse_float(value: Any) -> float | None:
    """Parse bookmaker odd value."""

    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(str(value).replace(",", "."))
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 1 else None


def parse_timestamp(value: Any) -> str | None:
    """Convert Lebull millisecond timestamp to UTC ISO string."""

    if value is None or isinstance(value, bool):
        return None
    try:
        timestamp = int(value) / 1000
        return datetime.fromtimestamp(timestamp, tz=UTC).replace(microsecond=0).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None
