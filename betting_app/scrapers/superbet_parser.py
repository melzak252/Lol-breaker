"""Parser utilities for Superbet League of Legends prematch pages."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import parsel


SUPERBET_LOL_URL = "https://superbet.pl/zaklady-bukmacherskie/league-of-legends"


@dataclass(frozen=True)
class ParsedSuperbetOffer:
    """One parsed Superbet LoL match-winner offer."""

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


DATE_LINE_RE = re.compile(r"^(dzisiaj|jutro|[a-ząćęłńóśźż]{2,4}\s+\d{1,2}\.\s+[a-ząćęłńóśźż]{3,}|\d{1,2}:\d{2}|.+,\s*\d{1,2}:\d{2})$", re.IGNORECASE)
ODD_RE = re.compile(r"^\d+[,.]\d{2}$")
MARKET_COUNT_RE = re.compile(r"^\+\d+$")
EVENT_ID_RE = re.compile(r"-(?P<id>\d+)(?:$|[/?#])")


def parse_superbet_lol_offers(body_text: str, html_text: str | None = None) -> list[ParsedSuperbetOffer]:
    """Parse Superbet rendered text and attach event links from HTML if present."""

    offers = parse_superbet_lol_body_text(body_text)
    if html_text:
        links = extract_superbet_event_links(html_text)
        offers = attach_superbet_links(offers, links)
    return offers


def parse_superbet_lol_body_text(text: str) -> list[ParsedSuperbetOffer]:
    """Parse visible Superbet LoL listing from `document.body.innerText`."""

    lines = [normalize_space(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    offers: list[ParsedSuperbetOffer] = []
    known_control = {
        "LEAGUE OF LEGENDS",
        "Social",
        "Kalendarz",
        "Rozgrywki",
        "Rejestracja",
        "Zaloguj",
        "Strona Główna",
        "Live",
        "Sport",
        "Kupony",
        "Gry",
        "Wszystko",
        "Rozstrzygnięte",
        "Nadchodzące",
        "Wideo",
        "ZAKŁADY NA LEAGUE OF LEGENDS",
    }
    i = 0
    current_league: str | None = None
    while i < len(lines) - 8:
        line = lines[i]
        if line in known_control or line.startswith("+") and line[1:].endswith("h"):
            i += 1
            continue
        if looks_like_league_heading(line, lines, i):
            current_league = line
            i += 1
            continue
        if current_league and looks_like_start_time(line):
            start = line
            team_a = lines[i + 1]
            team_b = lines[i + 2]
            # Expected Superbet two-way market: 1, odds_a, 2, odds_b, +N
            if lines[i + 3] == "1" and ODD_RE.match(lines[i + 4]) and lines[i + 5] == "2" and ODD_RE.match(lines[i + 6]):
                offers.append(
                    ParsedSuperbetOffer(
                        league=current_league,
                        raw_team_a=team_a,
                        raw_team_b=team_b,
                        odds_a=parse_decimal_odd(lines[i + 4]),
                        odds_b=parse_decimal_odd(lines[i + 6]),
                        source_url=SUPERBET_LOL_URL,
                        start_time_label=start,
                        market_count_label=lines[i + 7][1:] if i + 7 < len(lines) and MARKET_COUNT_RE.match(lines[i + 7]) else None,
                        raw_text="\n".join(lines[i : min(i + 8, len(lines))]),
                    )
                )
                i += 8
                continue
        i += 1
    return offers


def looks_like_league_heading(line: str, lines: list[str], index: int) -> bool:
    """Heuristic for league headers in the Superbet listing."""

    if len(line) < 2 or len(line) > 60:
        return False
    if not re.match(r"^[A-Za-z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ .&'\-]+$", line):
        return False
    if index + 1 >= len(lines):
        return False
    return looks_like_start_time(lines[index + 1])


def looks_like_start_time(line: str) -> bool:
    """Detect Superbet date/time labels."""

    return bool(DATE_LINE_RE.match(line.strip())) and ":" in line


def extract_superbet_event_links(html_text: str) -> list[dict[str, str]]:
    """Extract event URLs and visible card text from Superbet HTML using parsel."""

    links: list[dict[str, str]] = []
    selector = parsel.Selector(text=html_text)
    for anchor in selector.css('a[href*="/kursy/league-of-legends/"]'):
        href = anchor.css("::attr(href)").get() or ""
        if href.startswith("/"):
            href = "https://superbet.pl" + href
        id_match = EVENT_ID_RE.search(href)
        if not id_match:
            continue
        text = normalize_space(anchor.xpath("string(.)").get() or "")
        links.append({"href": href, "text": text, "id": id_match.group("id")})
    return links


def attach_superbet_links(
    offers: list[ParsedSuperbetOffer],
    links: list[dict[str, str]],
) -> list[ParsedSuperbetOffer]:
    """Attach direct Superbet event links to parsed offers."""

    enriched: list[ParsedSuperbetOffer] = []
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


def find_link_for_offer(
    offer: ParsedSuperbetOffer,
    links: list[dict[str, str]],
    used: set[str],
) -> dict[str, str] | None:
    """Find event link containing both team names."""

    team_a = compact_text(offer.raw_team_a)
    team_b = compact_text(offer.raw_team_b)
    for link in links:
        href = str(link.get("href") or "")
        if href in used:
            continue
        combined = compact_text(f"{link.get('href', '')} {link.get('text', '')}")
        if team_a in combined and team_b in combined:
            return link
    return None


def parse_decimal_odd(value: str) -> float:
    """Parse decimal odd."""

    return float(value.replace(",", "."))


def normalize_space(value: str) -> str:
    """Collapse whitespace."""

    return re.sub(r"\s+", " ", value).strip()


def compact_text(value: str) -> str:
    """Normalize text for fuzzy link matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())
