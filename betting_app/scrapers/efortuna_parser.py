"""Parser utilities for eFortuna League of Legends prematch pages."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


EFORTUNA_LOL_URL = "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol"

EFORTUNA_LOL_LEAGUE_URLS = [
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lec?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lck?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lpl?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lcs-na?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lcp?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lvp?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/lfl?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/nacl?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/miedzynarodowe-4/prime-league?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/lol/italian-tournament?tab=matches&filter=all",
    "https://www.efortuna.pl/zaklady-bukmacherskie/esport-lol/lol/lplol?tab=matches&filter=all",
]


@dataclass(frozen=True)
class ParsedEFortunaOffer:
    """One parsed eFortuna LoL match-winner offer."""

    league: str
    raw_team_a: str
    raw_team_b: str
    odds_a: float
    odds_b: float
    source_url: str
    offer_url: str | None = None
    start_time_label: str | None = None
    raw_text: str | None = None
    best_of: int | None = None


TIME_RE = re.compile(r"^(dzisiaj|jutro)\s+\d{1,2}:\d{2}$", re.IGNORECASE)
FULL_DATE_RE = re.compile(r"^[a-ząćęłńóśźż]{2,8}\.,\s*\d{1,2}\.\d{2}\.\d{4},\s*\d{1,2}:\d{2}$", re.IGNORECASE)
ODD_RE = re.compile(r"^\d+[,.]\d{2}$")

# "liczba map" (number of maps) over/under line detection.
# eFortuna shows lines like "liczba map -2.5" or "liczba map 2.5" for map totals.
# -2.5 → under 2.5 maps → Bo3 (2 or 3 maps possible)
# -4.5 → under 4.5 maps → Bo5 (3, 4, or 5 maps possible)
# -0.5 or 0.5 → Bo1 (exactly 1 map)
LICZBA_MAP_RE = re.compile(r"liczba\s+map", re.IGNORECASE)
MAP_LINE_RE = re.compile(r"^[+-]?\s*(\d+[,.]\d+)$")

def _infer_best_from_map_line(line_value: str) -> int | None:
    """Infer best_of from an over/under map total line.

    eFortuna uses half-point lines (e.g. 2.5, 4.5) for map totals.
    - Line ≤ 1.5 → Bo1 (1 map total)
    - Line ≤ 2.5 → Bo3 (2-3 maps possible)
    - Line ≤ 4.5 → Bo5 (3-5 maps possible)
    """
    try:
        clean = line_value.replace(" ", "").replace(",", ".")
        val = abs(float(clean))
    except ValueError:
        return None
    if val <= 1.5:
        return 1
    if val <= 2.5:
        return 3
    if val <= 4.5:
        return 5
    return None


def _extract_best_of_from_text(lines: list[str], team_a: str, team_b: str, match_start_idx: int) -> int | None:
    """Scan surrounding lines for a 'liczba map' line and infer best_of.

    Looks forward from the match start position for a 'liczba map' line
    followed by a decimal line (the over/under value). Stops after 30 lines
    to avoid crossing into the next match.
    """
    search_limit = min(match_start_idx + 30, len(lines))
    for j in range(match_start_idx, search_limit):
        if LICZBA_MAP_RE.match(lines[j]):
            # Next non-empty line should contain the line value
            for k in range(j + 1, min(j + 4, len(lines))):
                m = MAP_LINE_RE.match(lines[k])
                if m:
                    return _infer_best_from_map_line(m.group(1))
            break
    return None


def parse_efortuna_lol_offers(text: str, *, source_url: str, offer_url: str | None = None) -> list[ParsedEFortunaOffer]:
    """Parse visible eFortuna LoL match-winner offers from rendered text."""

    lines = [normalize_space(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    league = infer_league(lines, source_url)
    offers: list[ParsedEFortunaOffer] = []
    i = 0
    while i < len(lines) - 7:
        if not is_start_time(lines[i]):
            i += 1
            continue
        start = lines[i]
        team_a = lines[i + 1]
        team_b = lines[i + 2]
        if lines[i + 3].lower() != "zwycięzca meczu":
            i += 1
            continue
        if lines[i + 4] != team_a or not ODD_RE.match(lines[i + 5]) or lines[i + 6] != team_b or not ODD_RE.match(lines[i + 7]):
            i += 1
            continue
        offers.append(
            ParsedEFortunaOffer(
                league=league,
                raw_team_a=team_a,
                raw_team_b=team_b,
                odds_a=parse_decimal_odd(lines[i + 5]),
                odds_b=parse_decimal_odd(lines[i + 7]),
                source_url=source_url,
                offer_url=offer_url or build_offer_url(source_url, team_a, team_b),
                start_time_label=start,
                raw_text="\n".join(lines[i : i + 8]),
                best_of=_extract_best_of_from_text(lines, team_a, team_b, i),
            )
        )
        i += 8
    return offers


def infer_league(lines: list[str], source_url: str) -> str:
    """Infer league label from visible breadcrumb or URL."""

    slug = source_url.rstrip("/").split("/")[-1].split("?")[0]
    slug_map = {
        "lec": "LEC",
        "lck": "LCK",
        "lpl": "LPL",
        "lcs-na": "LCS NA",
        "lcp": "LCP",
        "lvp": "LES",
        "lfl": "LFL",
        "nacl": "NACL",
        "prime-league": "Prime League",
        "italian-tournament": "LIT",
        "lplol": "LPLOL",
    }
    if slug in slug_map:
        return slug_map[slug]

    for i, line in enumerate(lines[:-1]):
        if line == "Esport LOL" and i + 3 < len(lines):
            # Shape on league pages: Esport LOL / Międzynarodowe LCK LCK ...
            candidates = [candidate for candidate in lines[i + 1 : i + 8] if candidate not in {"/", "Międzynarodowe", "LOL", "Wszystko"}]
            if candidates:
                return candidates[0]
    return slug.replace("-", " ").upper() if slug else "Esport LOL"


def is_start_time(line: str) -> bool:
    """Detect Fortuna prematch start labels."""

    return bool(TIME_RE.match(line) or FULL_DATE_RE.match(line))


def parse_decimal_odd(value: str) -> float:
    """Parse decimal odd."""

    return float(value.replace(",", "."))


def build_offer_url(source_url: str, team_a: str, team_b: str) -> str:
    """Build eFortuna event details URL used after clicking a fixture card."""

    parts = urlsplit(source_url)
    base_path = parts.path.rstrip("/")
    slug = slugify(f"{team_a}-{team_b}")
    if not slug:
        return source_url
    return urlunsplit((parts.scheme, parts.netloc, f"{base_path}/{slug}", "tab=offer&filter=all", ""))


def slugify(value: str) -> str:
    """Approximate eFortuna route slug for fixture detail pages."""

    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.replace(".", "-")
    ascii_value = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value.lower())
    return re.sub(r"-+", "-", ascii_value).strip("-")


def normalize_space(value: str) -> str:
    """Collapse whitespace."""

    return re.sub(r"\s+", " ", value).strip()
