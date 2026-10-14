"""Superbet League of Legends proposition odds scraper (IDEA-018).

Extracts in-game markets (total kills, team kills, handicaps, duration,
first blood, first dragon, first baron, first tower, map winner, total maps)
from Superbet event detail API and rendered pages.
"""

from __future__ import annotations

import re
from typing import Any

from betting_app.scrapers.nodriver_client import NoDriverClient
from betting_app.scrapers.props_parser import (
    MarketType,
    ParsedMatchProps,
    ParsedPropLine,
    compute_novig_two_way,
    extract_line_number,
)
from betting_app.scrapers.superbet_parser import SUPERBET_LOL_URL

SCRAPER_VERSION = "superbet-props-0.1"


def classify_superbet_market(name: str) -> tuple[MarketType | None, int]:
    """Classify Superbet market name into MarketType and map_number."""
    n = name.strip().lower()

    # Determine map number
    map_number = 1
    m_match = re.search(r"map[a-ząćęłńóśźż]*\s*(\d+)|(\d+)\.\s*map[a-ząćęłńóśźż]*", n)
    if m_match:
        try:
            val = m_match.group(1) or m_match.group(2)
            if val:
                map_number = int(val)
        except ValueError:
            map_number = 1

    # Classify market type
    if "dokładny wynik" in n or "wynik meczu" in n or "correct score" in n:
        return "correct_score", 0
    if "suma map" in n or "liczba map" in n:
        return "total_maps", 1
    if "handicap map" in n:
        return "map_handicap", 1
    if "zwycięzca meczu" in n:
        return None, 1
    if "zwycięzca" in n and ("map" in n or "mapie" in n or "mapy" in n):
        return "map_winner", map_number
    # Kills
    if "handicap" in n and ("zabójstw" in n or "kill" in n or "frag" in n):
        return "handicap_kills", map_number
    if "drużyn" in n and "zabójstw" in n:
        return "team_kills", map_number
    if "liczba zabójstw" in n or "suma zabójstw" in n:
        return "total_kills", map_number

    # Duration
    if "czas trwania" in n or "czas gry" in n or "minut" in n:
        return "duration", map_number

    # First objectives
    if "1. krew" in n or "pierwsza krew" in n or "first blood" in n:
        return "first_blood", map_number
    if "1. smok" in n or "pierwszy smok" in n or "first dragon" in n:
        return "first_dragon", map_number
    if "1. baron" in n or "pierwszy baron" in n or "first baron" in n:
        return "first_baron", map_number
    if "1. wieża" in n or "pierwsza wieża" in n or "first tower" in n or "first turret" in n:
        return "first_tower", map_number
    if "herald" in n or "czerwie" in n or "voidgrub" in n:
        return "first_herald", map_number
    if "inhibitor" in n:
        return "first_inhibitor", map_number
    if "liczba smoków" in n or "suma smoków" in n or "total dragons" in n:
        return "total_dragons", map_number
    if "liczba baronów" in n or "suma baronów" in n or "total barons" in n:
        return "total_barons", map_number
    if "wyścig do" in n:
        return "race_to_kills", map_number
    return None, map_number


class SuperbetPropsScraper:
    """Scraper and parser for Superbet League of Legends proposition markets."""

    bookmaker: str = "superbet"
    scraper_version: str = SCRAPER_VERSION

    def __init__(self, start_url: str = SUPERBET_LOL_URL, headless: bool | None = None) -> None:
        self.start_url = start_url
        self.headless = headless

    def parse_event_props(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        match_info: dict[str, Any] | None = None,
    ) -> list[ParsedMatchProps]:
        """Parse Superbet event detail payload or market cards into ParsedMatchProps."""
        info = match_info or {}
        home = info.get("team_a") or info.get("home") or "Team A"
        away = info.get("team_b") or info.get("away") or "Team B"

        markets_list: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            home = payload.get("matchName", "").split(" · ")[0] if " · " in payload.get("matchName", "") else (payload.get("team_a") or home)
            if " · " in payload.get("matchName", ""):
                away = payload.get("matchName", "").split(" · ")[1]
            else:
                away = payload.get("team_b") or away

            raw = payload.get("markets") or payload.get("data") or []
            if isinstance(raw, list):
                markets_list = raw
            elif isinstance(raw, dict):
                markets_list = list(raw.values())
        elif isinstance(payload, list):
            markets_list = payload

        lines_by_map: dict[int, list[ParsedPropLine]] = {}

        for market in markets_list:
            if not isinstance(market, dict):
                continue
            market_name = str(market.get("name") or market.get("marketName") or "")
            market_type, map_num = classify_superbet_market(market_name)
            if not market_type:
                continue

            parsed_line = self._parse_market(market, market_type, market_name, home, away)
            if parsed_line:
                if map_num not in lines_by_map:
                    lines_by_map[map_num] = []
                lines_by_map[map_num].append(parsed_line)

        results: list[ParsedMatchProps] = []
        for map_num, lines in sorted(lines_by_map.items()):
            if lines:
                results.append(
                    ParsedMatchProps(
                        bookmaker=self.bookmaker,
                        raw_team_a=home,
                        raw_team_b=away,
                        map_number=map_num,
                        lines=lines,
                    )
                )

        return results

    def _parse_market(
        self,
        market: dict[str, Any],
        market_type: MarketType,
        market_name: str,
        home: str,
        away: str,
    ) -> ParsedPropLine | None:
        """Parse individual Superbet market into ParsedPropLine."""
        extracted_num = extract_line_number(market_name)
        line_val = extracted_num if extracted_num is not None else 0.0

        choices = market.get("choices") or market.get("odds") or market.get("outcomes") or []
        odds_over: float | None = None
        odds_under: float | None = None
        odds_cover_a: float | None = None
        odds_cover_b: float | None = None

        for c in choices:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or c.get("choiceName") or "").lower()
            try:
                odds = float(str(c.get("price") or c.get("odds") or c.get("value") or 0.0).replace(",", "."))
            except ValueError:
                continue

            if odds <= 1.0:
                continue

            if any(k in name for k in ["+", "więcej", "powyżej", "over"]):
                odds_over = odds
                num = extract_line_number(name)
                if num is not None:
                    line_val = num
            elif any(k in name for k in ["-", "mniej", "poniżej", "under"]):
                odds_under = odds
                num = extract_line_number(name)
                if num is not None:
                    line_val = num
            elif home.lower() in name or "1" in name or "team1" in name:
                odds_cover_a = odds
            elif away.lower() in name or "2" in name or "team2" in name:
                odds_cover_b = odds

        if odds_over and odds_under:
            p_o, p_u, margin = compute_novig_two_way(odds_over, odds_under)
            return ParsedPropLine(
                market_type=market_type,
                line=line_val,
                odds_over=odds_over,
                odds_under=odds_under,
                novig_prob_over=p_o,
                novig_prob_under=p_u,
                margin=margin,
                raw_market_name=market_name,
            )

        if odds_cover_a and odds_cover_b:
            p_a, p_b, margin = compute_novig_two_way(odds_cover_a, odds_cover_b)
            return ParsedPropLine(
                market_type=market_type,
                line=line_val,
                odds_cover_a=odds_cover_a,
                odds_cover_b=odds_cover_b,
                novig_prob_cover_a=p_a,
                novig_prob_cover_b=p_b,
                margin=margin,
                raw_market_name=market_name,
            )

        # Multi-outcome market handling (e.g. correct_score)
        if market_type == "correct_score" or len(choices) > 2:
            payload_dict = {}
            for c in choices:
                if isinstance(c, dict):
                    n_raw = str(c.get("name") or "")
                    try:
                        o_val = float(str(c.get("price") or "0").replace(",", "."))
                        if n_raw and o_val > 1.0:
                            payload_dict[n_raw] = o_val
                    except ValueError:
                        pass
            if payload_dict:
                tot_inv = sum(1.0 / v for v in payload_dict.values())
                margin = round(tot_inv - 1.0, 4)
                return ParsedPropLine(
                    market_type=market_type,
                    line=line_val,
                    margin=margin,
                    raw_market_name=market_name,
                    outcomes_payload=payload_dict,
                )
        return None

    async def _extract_event_urls(self, tab: Any) -> list[str]:
        import json
        js = """
        (() => {
            const links = [];
            const anchors = document.querySelectorAll('a[href*=\"/zaklady-bukmacherskie/league-of-legends/\"]');
            for (const a of anchors) {
                const href = a.getAttribute('href') || '';
                if (href.includes('/league-of-legends/') && href.match(/-\\d+$/)) {
                    const full = href.startsWith('http') ? href : 'https://superbet.pl' + href;
                    if (!links.includes(full)) links.push(full);
                }
            }
            return JSON.stringify(links);
        })()
        """
        try:
            raw = await tab.evaluate(js)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return []

    async def scrape_upcoming_props(
        self,
        max_matches: int = 10,
        match_urls: list[str] | None = None,
    ) -> list[ParsedMatchProps]:
        """Scrape Superbet upcoming match props from match views via NoDriverClient."""
        props_results: list[ParsedMatchProps] = []
        async with NoDriverClient(headless=self.headless) as client:
            target_urls: list[str] = list(match_urls or [])
            if not target_urls:
                tab = await client.open(self.start_url)
                await self._wait_for_render(tab, 5.0)
                target_urls = await self._extract_event_urls(tab)
                data = await self._extract_markets(tab)
                if data:
                    props_results.extend(self.parse_event_props(data))

            for i, match_url in enumerate(target_urls[:max_matches]):
                if i > 0:
                    import asyncio, random
                    await asyncio.sleep(random.uniform(3.0, 5.5))
                try:
                    tab = await client.open(match_url)
                    await self._wait_for_render(tab, 4.0)
                    detail_data = await self._extract_markets(tab)
                    if detail_data:
                        from dataclasses import replace
                        parsed = self.parse_event_props(detail_data)
                        for p in parsed:
                            props_results.append(replace(p, source_url=match_url))
                except Exception as err:
                    print(f"Error scraping Superbet match view {match_url}: {err}")
                    continue

        return props_results[:max_matches]
    async def _wait_for_render(self, tab: Any, seconds: float = 5.0) -> None:
        import asyncio
        await asyncio.sleep(seconds)

    async def _extract_markets(self, tab: Any) -> list[dict[str, Any]] | None:
        import json
        js = """
        (() => {
            const markets = [];
            const cards = document.querySelectorAll('.market-item, .event-market-card');
            for (const c of cards) {
                const title = c.querySelector('.market-title, h4')?.innerText?.trim() || '';
                const choices = [];
                const buttons = c.querySelectorAll('button, .odd-button');
                for (const b of buttons) {
                    const label = b.querySelector('.odd-name')?.innerText?.trim() || b.innerText?.trim() || '';
                    const price = b.querySelector('.odd-value')?.innerText?.trim() || '';
                    choices.push({ name: label, price: price });
                }
                if (title && choices.length > 0) {
                    markets.push({ name: title, choices: choices });
                }
            }
            return JSON.stringify(markets);
        })()
        """
        try:
            raw = await tab.evaluate(js)
            if raw:
                return json.loads(raw)
        except Exception:
            return None
        return None
