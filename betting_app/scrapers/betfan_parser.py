"""Parser utilities for Betfan League of Legends prematch pages."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

import parsel


BETFAN_ESPORT_URL = "https://betfan.pl/esport"


@dataclass(frozen=True)
class ParsedBetfanOffer:
    """One parsed Betfan LoL match-winner offer."""

    league: str
    raw_team_a: str
    raw_team_b: str
    odds_a: float
    odds_b: float
    source_url: str
    offer_url: str | None = None
    start_time_label: str | None = None
    market_count_label: str | None = None
    bookmaker_event_id: str | None = None
    raw_text: str | None = None


ODD_RE = re.compile(r"^\d+[,.]\d{2}$")
MARKET_COUNT_RE = re.compile(r"^\+\d+$")
START_TIME_RE = re.compile(
    r"^(?:dzi[śs]\s+\d{1,2}:\d{2}|jutro\s+\d{1,2}:\d{2}|\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2}|\d{1,2}:\d{2}:\d{2})$",
    re.IGNORECASE,
)
EVENT_ID_RE = re.compile(r"/(?P<id>\d+)(?:$|[/?#])")


def parse_betfan_lol_offers(body_text: str, html_text: str | None = None) -> list[ParsedBetfanOffer]:
    """Parse Betfan rendered text and attach event links from HTML if present."""

    offers = parse_betfan_lol_body_text(body_text)
    links = extract_betfan_event_links(html_text or "")
    offers = attach_betfan_links(offers, links)
    return [offer if offer.offer_url else replace(offer, offer_url=build_offer_url(offer)) for offer in offers]


def parse_betfan_lol_body_text(text: str) -> list[ParsedBetfanOffer]:
    """Parse visible Betfan LoL listing from `document.body.innerText`."""

    lines = [normalize_space(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    offers: list[ParsedBetfanOffer] = []
    i = 0
    while i < len(lines) - 8:
        league = lines[i]
        if not looks_like_league(league):
            i += 1
            continue
        if not MARKET_COUNT_RE.match(lines[i + 1]):
            i += 1
            continue
        team_a = lines[i + 2]
        start = lines[i + 3]
        team_b = lines[i + 4]
        # Betfan LoL card layout:
        # league, +N, teamA, start, teamB, teamA, oddsA, teamB, oddsB
        if (
            looks_like_start_time(start)
            and normalize_key(lines[i + 5]) == normalize_key(team_a)
            and ODD_RE.match(lines[i + 6])
            and normalize_key(lines[i + 7]) == normalize_key(team_b)
            and ODD_RE.match(lines[i + 8])
        ):
            offers.append(
                ParsedBetfanOffer(
                    league=league,
                    raw_team_a=team_a,
                    raw_team_b=team_b,
                    odds_a=parse_decimal_odd(lines[i + 6]),
                    odds_b=parse_decimal_odd(lines[i + 8]),
                    source_url=BETFAN_ESPORT_URL,
                    start_time_label=start,
                    market_count_label=lines[i + 1][1:],
                    raw_text="\n".join(lines[i : i + 9]),
                )
            )
            i += 9
            continue
        i += 1
    return offers


def looks_like_league(line: str) -> bool:
    """Heuristic for Betfan LoL league labels."""

    if line in {"Esport", "Counter Strike 2", "LoL", "Dota", "Call Of Duty", "Valorant"}:
        return False
    if line in {"1h", "3h", "6h", "Dzisiaj", "Jutro", "Pojutrze", "Rozwiń"}:
        return False
    return bool(re.match(r"^[\w .ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'-]{2,60}$", line))


def looks_like_start_time(value: str) -> bool:
    """Detect Betfan start/countdown labels."""

    return bool(START_TIME_RE.match(value.strip()))


def extract_betfan_event_links(html_text: str) -> list[dict[str, str]]:
    """Extract Betfan per-event links from HTML using parsel."""

    links: list[dict[str, str]] = []
    selector = parsel.Selector(text=html_text)
    for anchor in selector.css('a[href*="/lista-zakladow/lol/mecze/"]'):
        href = anchor.css("::attr(href)").get() or ""
        if href.startswith("/"):
            href = "https://betfan.pl" + href
        id_match = EVENT_ID_RE.search(href)
        if not id_match:
            continue
        text = normalize_space(anchor.xpath("string(.)").get() or "")
        links.append({"href": href, "text": text, "id": id_match.group("id")})
    return links


def attach_betfan_links(offers: list[ParsedBetfanOffer], links: list[dict[str, str]]) -> list[ParsedBetfanOffer]:
    """Attach direct event URLs to offers."""

    enriched: list[ParsedBetfanOffer] = []
    used: set[str] = set()
    for offer in offers:
        link = find_link_for_offer(offer, links, used)
        if not link:
            enriched.append(offer)
            continue
        href = str(link["href"])
        used.add(href)
        enriched.append(replace(offer, offer_url=href, bookmaker_event_id=str(link.get("id") or "") or None))
    return enriched


def find_link_for_offer(offer: ParsedBetfanOffer, links: list[dict[str, str]], used: set[str]) -> dict[str, str] | None:
    """Find event link containing both teams."""

    team_a = compact_text(offer.raw_team_a)
    team_b = compact_text(offer.raw_team_b)
    for link in links:
        href = str(link.get("href") or "")
        if href in used:
            continue
        combined = compact_text(f"{href} {link.get('text', '')}")
        if team_a in combined and team_b in combined:
            return link
    return None


def build_offer_url(offer: ParsedBetfanOffer) -> str:
    """Build a best-effort Betfan event URL when HTML href is unavailable."""

    league_slug = slugify(offer.league)
    teams_slug = f"{slugify(offer.raw_team_a)}-{slugify(offer.raw_team_b)}"
    suffix = f"/{offer.bookmaker_event_id}" if offer.bookmaker_event_id else ""
    return f"https://betfan.pl/lista-zakladow/lol/mecze/{league_slug}/{teams_slug}{suffix}"


def parse_decimal_odd(value: str) -> float:
    """Parse decimal odd."""

    return float(value.replace(",", "."))


def normalize_space(value: str) -> str:
    """Collapse whitespace."""

    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: str) -> str:
    """Normalize team label for exact card checks."""

    return normalize_space(value).casefold()


def compact_text(value: str) -> str:
    """Normalize text for fuzzy link matching."""

    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slugify(value: str) -> str:
    """Build URL-friendly slug."""

    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-+", "-", value)
