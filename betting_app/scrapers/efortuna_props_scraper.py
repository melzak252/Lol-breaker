"""eFortuna League of Legends proposition odds scraper (IDEA-018).

Extracts in-game markets (total kills, team kills, handicaps, duration,
first blood, first dragon, first baron, first tower, map winner, total maps)
from eFortuna event detail pages and ticket API payloads.
"""

from __future__ import annotations

import re
from typing import Any

from betting_app.scrapers.efortuna_parser import EFORTUNA_LOL_LEAGUE_URLS, EFORTUNA_LOL_URL
from betting_app.scrapers.nodriver_client import NoDriverClient
from betting_app.scrapers.props_parser import (
    MarketType,
    ParsedMatchProps,
    ParsedPropLine,
    compute_novig_two_way,
    extract_line_number,
)

SCRAPER_VERSION = "efortuna-props-0.1"


def classify_efortuna_market(name: str) -> tuple[MarketType | None, int]:
    """Classify eFortuna market header into MarketType and map_number."""
    n = name.strip().lower()

    # Determine map number
    map_number = 1
    m_match = re.search(r"(\d+)\.\s*map[a-ząćęłńóśźż]*|map[a-ząćęłńóśźż]*\s*(\d+)", n)
    if m_match:
        try:
            val = m_match.group(1) or m_match.group(2)
            if val:
                map_number = int(val)
        except ValueError:
            map_number = 1

    if "dokładny wynik" in n or "wynik meczu" in n or "correct score" in n:
        return "correct_score", 0
    if "liczba map" in n or "suma map" in n:
        return "total_maps", 1
    if "handicap map" in n:
        return "map_handicap", 1
    if "zwycięzca meczu" in n:
        return None, 1
    if "zwycięzca" in n and ("map" in n or "mapy" in n):
        return "map_winner", map_number
    if "handicap" in n and ("zabójstw" in n or "kill" in n or "frag" in n):
        return "handicap_kills", map_number
    if "drużyn" in n and "zabójstw" in n:
        return "team_kills", map_number
    if "liczba zabójstw" in n or "suma zabójstw" in n:
        return "total_kills", map_number

    # Duration
    if "czas trwania" in n or "minut" in n:
        return "duration", map_number

    # First objectives
    if "1. zabójstwo" in n or "pierwsze zabójstwo" in n or "1. krew" in n or "pierwsza krew" in n:
        return "first_blood", map_number
    if "1. smok" in n or "pierwszy smok" in n:
        return "first_dragon", map_number
    if "1. baron" in n or "pierwszy baron" in n:
        return "first_baron", map_number
    if "1. wieża" in n or "pierwsza wieża" in n or "first tower" in n:
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


class EFortunaPropsScraper:
    """Scraper and parser for eFortuna League of Legends proposition markets."""

    bookmaker: str = "efortuna"
    scraper_version: str = SCRAPER_VERSION

    def __init__(
        self,
        start_url: str = EFORTUNA_LOL_URL,
        headless: bool | None = None,
        league_urls: list[str] | None = None,
    ) -> None:
        self.start_url = start_url
        self.headless = headless
        self.league_urls = league_urls or EFORTUNA_LOL_LEAGUE_URLS

    def parse_event_props(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        match_info: dict[str, Any] | None = None,
    ) -> list[ParsedMatchProps]:
        """Parse eFortuna event detail tables/markets into ParsedMatchProps."""
        info = match_info or {}
        home = info.get("team_a") or info.get("home") or "Team A"
        away = info.get("team_b") or info.get("away") or "Team B"

        # If payload is a dict with 'tables' or 'markets'
        tables: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            home = payload.get("team_a") or home
            away = payload.get("team_b") or away
            raw_tables = payload.get("tables") or payload.get("markets") or payload.get("data") or []
            if isinstance(raw_tables, list):
                tables = raw_tables
            elif isinstance(raw_tables, dict):
                tables = list(raw_tables.values())
        elif isinstance(payload, list):
            tables = payload

        lines_by_map: dict[int, list[ParsedPropLine]] = {}

        for table in tables:
            if not isinstance(table, dict):
                continue
            table_name = str(table.get("name") or table.get("title") or "")
            market_type, map_num = classify_efortuna_market(table_name)
            if not market_type:
                continue

            parsed_line = self._parse_table_line(table, market_type, table_name, home, away)
            if parsed_line:
                if map_num not in lines_by_map:
                    lines_by_map[map_num] = []
                lines_by_map[map_num].append(parsed_line)

        # Return list of ParsedMatchProps
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

    def _parse_table_line(
        self,
        table: dict[str, Any],
        market_type: MarketType,
        table_name: str,
        home: str,
        away: str,
    ) -> ParsedPropLine | None:
        """Parse individual eFortuna table into ParsedPropLine."""
        extracted_num = extract_line_number(table_name)
        line_val = extracted_num if extracted_num is not None else 0.0

        bets = table.get("bets") or table.get("odds") or table.get("items") or []
        odds_over: float | None = None
        odds_under: float | None = None
        odds_cover_a: float | None = None
        odds_cover_b: float | None = None

        for b in bets:
            if not isinstance(b, dict):
                continue
            name = str(b.get("name") or b.get("title") or b.get("label") or "").lower()
            try:
                odds = float(str(b.get("odds") or b.get("value") or 0.0).replace(",", "."))
            except ValueError:
                continue

            if odds <= 1.0:
                continue

            # Check Over/Under or +/-
            if any(k in name for k in ["+", "więcej", "powyżej", "over"]):
                odds_over = odds
                n = extract_line_number(name)
                if n is not None:
                    line_val = n
            elif any(k in name for k in ["-", "mniej", "poniżej", "under"]):
                odds_under = odds
                n = extract_line_number(name)
                if n is not None:
                    line_val = n
            elif home.lower() in name or "1" in name or "home" in name:
                odds_cover_a = odds
            elif away.lower() in name or "2" in name or "away" in name:
                odds_cover_b = odds

        # Two-way Over/Under
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
                raw_market_name=table_name,
            )

        # Two-way Team A / Team B
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
                raw_market_name=table_name,
            )

        # Multi-outcome market handling (e.g. correct_score)
        if market_type == "correct_score" or len(bets) > 2:
            payload_dict = {}
            for b in bets:
                if isinstance(b, dict):
                    n_raw = str(b.get("name") or "")
                    try:
                        o_val = float(str(b.get("odds") or "0").replace(",", "."))
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
                    raw_market_name=table_name,
                    outcomes_payload=payload_dict,
                )
        return None

    async def _extract_match_links(self, tab: Any) -> list[str]:
        import json
        js = """
        (() => {
            const links = [];
            const anchors = document.querySelectorAll('a[href*="/zaklady-bukmacherskie/esport-lol/"]');
            for (const a of anchors) {
                const href = a.getAttribute('href') || '';
                if (href.includes('/esport-lol/') && !href.includes('tab=') && !links.includes(href)) {
                    const full = href.startsWith('http') ? href : 'https://www.efortuna.pl' + href;
                    links.push(full);
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
        """Scrape eFortuna upcoming match props from match views via NoDriverClient."""
        props_results: list[ParsedMatchProps] = []
        async with NoDriverClient(headless=self.headless) as client:
            target_urls: list[str] = list(match_urls or [])
            if not target_urls:
                for league_url in self.league_urls[:3]:
                    try:
                        tab = await client.open(league_url)
                        await self._wait_for_render(tab, 4.0)
                        discovered = await self._extract_match_links(tab)
                        for u in discovered:
                            if u not in target_urls:
                                target_urls.append(u)
                        tables_data = await self._extract_tables_from_dom(tab)
                        if tables_data:
                            props_results.extend(self.parse_event_props(tables_data))
                    except Exception:
                        continue

            for i, match_url in enumerate(target_urls[:max_matches]):
                if i > 0:
                    import asyncio, random
                    await asyncio.sleep(random.uniform(3.0, 5.5))
                try:
                    tab = await client.open(match_url)
                    await self._wait_for_render(tab, 4.0)
                    tables_data = await self._extract_tables_from_dom(tab)
                    if tables_data:
                        from dataclasses import replace
                        parsed = self.parse_event_props(tables_data)
                        for p in parsed:
                            props_results.append(replace(p, source_url=match_url))
                except Exception as err:
                    print(f"Error scraping eFortuna match view {match_url}: {err}")
                    continue

        return props_results[:max_matches]

    async def _wait_for_render(self, tab: Any, seconds: float = 5.0) -> None:
        import asyncio
        await asyncio.sleep(seconds)

    async def _extract_tables_from_dom(self, tab: Any) -> list[dict[str, Any]] | None:
        import json
        js = """
        (() => {
            const results = [];
            const tables = document.querySelectorAll('.odds-table, .bet-table, .market-table');
            for (const t of tables) {
                const title = t.querySelector('.table-header, .market-title, h3, h4')?.innerText?.trim() || '';
                const bets = [];
                const buttons = t.querySelectorAll('button, .bet-btn, .odds-value');
                for (const btn of buttons) {
                    const label = btn.querySelector('.bet-name, .label')?.innerText?.trim() || btn.innerText?.trim() || '';
                    const oddsText = btn.querySelector('.odds, .value')?.innerText?.trim() || '';
                    bets.push({ name: label, odds: oddsText });
                }
                if (title && bets.length > 0) {
                    results.push({ name: title, bets: bets });
                }
            }
            return JSON.stringify(results);
        })()
        """
        try:
            raw = await tab.evaluate(js)
            if raw:
                return json.loads(raw)
        except Exception:
            return None
        return None
