"""Parser utilities for Betclic League of Legends prematch pages.

Betclic renders a useful LoL listing at `/league-of-legends-slol`. The page can
be parsed either from real DOM/HTML text or from a markdown/text snapshot (for
example a saved reader/debug dump), because the event cards appear as links with
their full card text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import parsel


BETCLIC_LOL_PATH = "/league-of-legends-slol/"
BETCLIC_LOL_URL = "https://www.betclic.pl/league-of-legends-slol"


@dataclass(frozen=True)
class ParsedBetclicOffer:
    """One parsed Betclic LoL match-winner offer."""

    league: str
    raw_team_a: str
    raw_team_b: str
    odds_a: float
    odds_b: float
    source_url: str
    start_time_label: str | None = None
    date_label: str | None = None
    market_count_label: str | None = None
    bookmaker_event_id: str | None = None
    raw_text: str | None = None


MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https://www\.betclic\.pl/league-of-legends-slol/[^)]+-m\d+)\)")
MARKET_PREFIX_RE = re.compile(r"^(?P<league>.+?)\s+\+(?P<count>\d+)\s+zakł\.\s+(?P<rest>.+)$", re.IGNORECASE)
TIME_RE = re.compile(r"^(?P<team_a>.+?)\s+(?P<time>\d{1,2}:\d{2})\s+-\s+(?P<tail>.+)$")
ODD_RE = re.compile(r"\d+[,.]\d{2}")
ODD_SUFFIX_RE = re.compile(r"(?=(\d{1,2}[,.]\d{2}))")
MATCH_ID_RE = re.compile(r"-m(?P<id>\d+)(?:$|[/?#])")
MARKET_COUNT_LINE_RE = re.compile(r"^\+(?P<count>\d+)$")
TIME_LINE_RE = re.compile(r"^\d{1,2}:\d{2}$")


def parse_betclic_lol_offers(text: str) -> list[ParsedBetclicOffer]:
    """Parse Betclic LoL match-winner cards from HTML or markdown text."""

    offers: list[ParsedBetclicOffer] = []
    seen: set[tuple[str, str, str, float, float]] = set()
    for offer in parse_betclic_lol_body_text(text):
        key = (offer.source_url, offer.raw_team_a, offer.raw_team_b, offer.odds_a, offer.odds_b)
        if key in seen:
            continue
        seen.add(key)
        offers.append(offer)

    for raw_text, href in extract_event_links(text):
        offer = parse_event_link(raw_text, href)
        if offer is None:
            continue
        key = (offer.source_url, offer.raw_team_a, offer.raw_team_b, offer.odds_a, offer.odds_b)
        if key in seen:
            continue
        seen.add(key)
        offers.append(offer)
    return offers


def parse_betclic_lol_body_text(text: str) -> list[ParsedBetclicOffer]:
    """Parse Betclic rendered `document.body.innerText` LoL listing.

    This is the most reliable format observed with NoDriver. Prematch cards are
    line based, for example:

    ```text
    LPL
    +457
    zakł.
    TOP Esports
    11:30
    LGD Gaming
    TOP Esp
    orts
    1,15
    LGD Gaming
    4,50
    ```

    Live rows under `Teraz` do not have the `+N / zakł.` marker and are skipped
    because the MVP is prematch-only.

    Some prematch cards (typically the first in a date section) may lack the
    ``+N zakł.`` marker entirely.  The parser therefore tries two strategies:

    1. **Strict** – the classic ``League → +N → zakł. → TeamA → HH:MM → TeamB`` pattern.
    2. **Relaxed** – ``League → TeamA → HH:MM → TeamB`` without the market-count
       marker.  This is only attempted when the strict match fails, so that
       well-formed cards are still parsed with their market count intact.
    """

    lines = [normalize_space(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    offers: list[ParsedBetclicOffer] = []
    i = 0
    current_date_label: str | None = None
    while i < len(lines) - 4:
        if is_date_heading(lines[i]):
            current_date_label = lines[i]
            i += 1
            continue

        # --- Strategy 1: strict (with +N zakł.) ---
        league = lines[i]
        market_count = MARKET_COUNT_LINE_RE.match(lines[i + 1])
        if market_count and i + 2 < len(lines) and lines[i + 2].lower() == "zakł.":
            team_a = lines[i + 3]
            start_time = lines[i + 4]
            team_b = lines[i + 5]
            if TIME_LINE_RE.match(start_time):
                odd_positions = _find_two_odds(lines, i + 6)
                if len(odd_positions) >= 2:
                    offers.append(
                        ParsedBetclicOffer(
                            league=league,
                            raw_team_a=team_a,
                            raw_team_b=team_b,
                            odds_a=odd_positions[0][1],
                            odds_b=odd_positions[1][1],
                            source_url=BETCLIC_LOL_URL,
                            start_time_label=start_time,
                            date_label=current_date_label,
                            market_count_label=market_count.group("count"),
                            raw_text="\n".join(lines[i : odd_positions[1][0] + 1]),
                        )
                    )
                    i = odd_positions[1][0] + 1
                    continue

        # --- Strategy 2: relaxed (no +N zakł.) ---
        # League line followed by TeamA → HH:MM → TeamB
        relaxed_team_a = lines[i + 1]
        relaxed_time = lines[i + 2]
        if TIME_LINE_RE.match(relaxed_time) and i + 3 < len(lines):
            relaxed_team_b = lines[i + 3]
            # Make sure this isn't a false positive: the "league" line should
            # not look like a time, date heading, or odds value.
            if (
                not TIME_LINE_RE.match(league)
                and not is_date_heading(league)
                and not ODD_RE.fullmatch(league)
                and league.lower() != "zakł."
            ):
                odd_positions = _find_two_odds(lines, i + 4)
                if len(odd_positions) >= 2:
                    offers.append(
                        ParsedBetclicOffer(
                            league=league,
                            raw_team_a=relaxed_team_a,
                            raw_team_b=relaxed_team_b,
                            odds_a=odd_positions[0][1],
                            odds_b=odd_positions[1][1],
                            source_url=BETCLIC_LOL_URL,
                            start_time_label=relaxed_time,
                            date_label=current_date_label,
                            market_count_label=None,
                            raw_text="\n".join(lines[i : odd_positions[1][0] + 1]),
                        )
                    )
                    i = odd_positions[1][0] + 1
                    continue

        i += 1
    return offers


def _find_two_odds(lines: list[str], start: int) -> list[tuple[int, float]]:
    """Scan forward from *start* and return up to two (index, odd) pairs."""

    odd_positions: list[tuple[int, float]] = []
    j = start
    while j < len(lines):
        # Stop before the next prematch card marker.
        if j + 2 < len(lines) and MARKET_COUNT_LINE_RE.match(lines[j + 1]) and lines[j + 2].lower() == "zakł.":
            break
        if ODD_RE.fullmatch(lines[j]):
            odd_positions.append((j, parse_decimal_odd(lines[j])))
            if len(odd_positions) == 2:
                break
        j += 1
    return odd_positions


def is_date_heading(line: str) -> bool:
    """Detect date section headings in Betclic listing text."""

    normalized = line.strip().lower()
    if normalized in {"teraz", "dzisiaj", "jutro"}:
        return True
    return bool(re.match(r"^(pon|wt|śr|sr|czw|pt|sob|niedz)\.\s+\d{1,2}/\d{2}$", normalized))


def extract_event_links(text: str) -> list[tuple[str, str]]:
    """Extract candidate event card texts and URLs from markdown or HTML.

    HTML extraction intentionally uses parsel rather than regex so changes in
    attributes or nested markup do not break link detection.
    """

    links: list[tuple[str, str]] = []
    for match in MARKDOWN_LINK_RE.finditer(text):
        links.append((normalize_space(match.group(1)), match.group(2)))

    selector = parsel.Selector(text=text)
    for anchor in selector.css('a[href*="league-of-legends-slol"]'):
        href = anchor.css("::attr(href)").get() or ""
        if "-m" not in href:
            continue
        if href.startswith("/"):
            href = "https://www.betclic.pl" + href
        body = anchor.xpath("string(.)").get() or ""
        links.append((normalize_space(body), href))
    return links


def parse_event_link(raw_text: str, href: str) -> ParsedBetclicOffer | None:
    """Parse one Betclic event card link.

    Expected prematch shape examples:

    `LCK +268 zakł. Dplus KIA 08:00 - DRX Dplus KIA 1,17 DRX 4,20`
    `LPL +84 zakł. Anyone's Legend 06:00 - Edward Gaming Anyone's Legend 1,10 Edward Gaming 5,50`

    Live/result rows without a `HH:MM -` separator are intentionally skipped for
    now because the MVP is prematch-only.
    """

    prefixed = MARKET_PREFIX_RE.match(raw_text)
    if not prefixed:
        return None
    league = prefixed.group("league").strip()
    market_count_label = prefixed.group("count")
    rest = prefixed.group("rest").strip()

    timed = TIME_RE.match(rest)
    if not timed:
        return None
    team_a = timed.group("team_a").strip()
    start_time_label = timed.group("time")
    tail = timed.group("tail").strip()

    parsed_odds = parse_compact_event_odds(tail, team_a)
    if parsed_odds is None:
        return None
    team_b, odds_a, odds_b = parsed_odds

    match_id = None
    id_match = MATCH_ID_RE.search(href)
    if id_match:
        match_id = id_match.group("id")

    return ParsedBetclicOffer(
        league=league,
        raw_team_a=team_a,
        raw_team_b=team_b,
        odds_a=odds_a,
        odds_b=odds_b,
        source_url=href,
        start_time_label=start_time_label,
        market_count_label=market_count_label,
        bookmaker_event_id=match_id,
        raw_text=raw_text,
    )


def infer_team_b_from_before_odd_a(before_odd_a: str, team_a: str) -> str | None:
    """Fallback team B extraction for card text before first odd."""

    if before_odd_a.endswith(team_a):
        candidate = before_odd_a[: -len(team_a)].strip()
        return candidate or None
    return None


def parse_compact_event_odds(tail: str, team_a: str) -> tuple[str, float, float] | None:
    """Parse Betclic compact event tail into team B and two match-winner odds.

    Betclic sometimes renders card text without spaces between team labels and
    odds, e.g. ``KT Rolster T11,25KT Rolster3,40`` or
    ``Cloud9 Team Liquid1,70Cloud91,98``.  A plain regex sees ``T11,25`` as
    ``11,25`` and ``Cloud91,98`` as ``91,98`` even though the actual odds are
    ``1,25`` and ``1,98`` because the preceding digit belongs to the team name.

    The event shape is stable enough to use team labels as anchors:
    ``{team_b} {team_a}{odds_a}{team_b}{odds_b}``.  We therefore consider
    overlapping odd suffixes and keep only the split where the text before the
    first odd ends with ``team_a`` and the text between odds equals the inferred
    ``team_b``.  This preserves legitimate high odds like ``11,25`` when the
    full value is actually after the team label.
    """

    candidates = odd_suffix_candidates(tail)
    for odd_a_start, odd_a_end, odd_a_text in candidates:
        before_odd_a = tail[:odd_a_start].strip()
        team_b = infer_team_b_from_before_odd_a(before_odd_a, team_a)
        if not team_b:
            continue

        for odd_b_start, odd_b_end, odd_b_text in candidates:
            if odd_b_start <= odd_a_end:
                continue
            between_odds = tail[odd_a_end:odd_b_start].strip()
            if between_odds != team_b:
                continue
            trailing = tail[odd_b_end:].strip()
            if trailing:
                continue
            return team_b, parse_decimal_odd(odd_a_text), parse_decimal_odd(odd_b_text)
    return None


def odd_suffix_candidates(text: str) -> list[tuple[int, int, str]]:
    """Return overlapping decimal-odd candidates in compact Betclic text."""

    candidates: list[tuple[int, int, str]] = []
    for match in ODD_SUFFIX_RE.finditer(text):
        value = match.group(1)
        start = match.start(1)
        end = start + len(value)
        candidates.append((start, end, value))
    return candidates


def parse_decimal_odd(value: str) -> float:
    """Parse Polish decimal odd notation."""

    return float(value.replace(",", "."))


def normalize_space(value: str) -> str:
    """Collapse whitespace."""

    return re.sub(r"\s+", " ", value).strip()
