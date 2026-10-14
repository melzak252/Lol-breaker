"""Cross-bookmaker canonical match resolution."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from betting_app.core.db import query_df, transaction
from betting_app.core.matching import normalize_team_name, similarity
from betting_app.services.team_alias_service import (
    AliasContext,
    alias_lookup_key,
    resolve_scoped_alias,
)


TEAM_ALIASES = {
    "brion": "brion",
    "hanjin brion": "brion",
    "oksavingsbank brion": "brion",
    "ok savingsbank brion": "brion",
    "dn freecs": "dn soopers",
    "dn soopers": "dn soopers",
    "soopers": "dn soopers",
    "dplus kia": "dplus kia",
    "dk": "dplus kia",
    "drx": "drx",
    "kiwoom drx": "drx",
    "giantx": "giantx",
    "giant x": "giantx",
    "top esports": "top esports",
    "top": "top esports",
    "tes": "top esports",
    "thundertalk gaming": "thundertalk",
    "thundertalk": "thundertalk",
    "tt gaming": "thundertalk",
    "tt": "thundertalk",
    "edward": "edward gaming",
    "edward gaming": "edward gaming",
    "anyones legend": "anyones legend",
    "anyone s legend": "anyones legend",
    "anyone legend": "anyones legend",
    "fearx": "bnk fearx",
    "bnk fearx": "bnk fearx",
    "karmine corp blue": "karmine corp blue",
    "kc blue": "karmine corp blue",
    "karmine corp": "karmine corp",
    "geng": "gen g",
    "gen g": "gen g",
    "gen.g": "gen g",
    "dplus": "dplus kia",
    "barca esports": "barca esports",
    "barca": "barca esports",
    "fc barcelona": "barca esports",
    # Keep Movistar KOI main roster distinct from the academy/Fenix roster.
    # The LES feeds sometimes omit the "Fenix/Academy" suffix; context rules
    # in ``canonical_team_key`` handle that, but the neutral alias must not
    # collapse both squads to a generic "koi" bucket.
    "movistar koi": "movistar koi",
    "movister koi": "movistar koi",
    "koi academy": "movistar koi fenix",
    "movistar koi academy": "movistar koi fenix",
    "movistar koi fenix": "movistar koi fenix",
    "movistar koi fénix": "movistar koi fenix",
    "mkf": "movistar koi fenix",
    "nongshim redforce": "nongshim redforce",
    "ns red force": "nongshim redforce",
    "red canids": "red canids",
    "red canids kalunga": "red canids",
    "ruddy corporation": "ruddy",
    "ruddy esports": "ruddy",
    "ruddy": "ruddy",
    "los": "los",
    "los grandes": "los",
    "ronaldo": "ronaldoteam",
    "ronaldoteam": "ronaldoteam",
    # LCK CL (Challengers League) academy teams
    "soopers challengers": "dn soopers challengers",
    "dn soopers challengers": "dn soopers challengers",
    "kdf challengers": "dn soopers challengers",
    "t1 challengers": "t1 academy",
    "t1 academy": "t1 academy",
    "t1 esports academy": "t1 academy",
    "skt t1 challengers": "t1 academy",
    "skt t1 academy": "t1 academy",
    "drx challengers": "krx challengers",
    "kiwoom drx challengers": "krx challengers",
    "krx challengers": "krx challengers",
    "kt challengers": "ktrolster challengers",
    "kt rolster challengers": "ktrolster challengers",
    "ns challengers": "nongshim academy",
    "nongshim challengers": "nongshim academy",
    "nongshim redforce challengers": "nongshim academy",
    "nongshim academy": "nongshim academy",
    "fearx challengers": "bnk fearx youth",
    "bnk fearx challengers": "bnk fearx youth",
    "fearx youth": "bnk fearx youth",
    "bnk fearx youth": "bnk fearx youth",
    "brion challengers": "hanjin brion challengers",
    "ok brion challengers": "hanjin brion challengers",
    "hanjin brion challengers": "hanjin brion challengers",
    "bro challengers": "hanjin brion challengers",
    "gen g challengers": "geng global academy",
    "geng challengers": "geng global academy",
    "gen g global academy": "geng global academy",
    "geng global academy": "geng global academy",
    "hanwha life challengers": "hanwhalife challengers",
    "hle challengers": "hanwhalife challengers",
    "hanwhalife challengers": "hanwhalife challengers",
    "dk challengers": "damwonkia challengers",
    "dplus challengers": "damwonkia challengers",
    "dplus kia challengers": "damwonkia challengers",
    "damwonkia challengers": "damwonkia challengers",
    "anyones legend": "anyones legend",
    "anyone s legend": "anyones legend",
    "anyone legend": "anyones legend",
    "we love": "wlgaming",
    "wlgaming": "wlgaming",
    "ucam club": "ucam",
    "ucam ec": "ucam",
    "ucam tokiers": "ucam",
    "ucam": "ucam",
    "e wie einfach": "e wie einfach e sports",
    "ninjas in pyjamas cn": "ninjas in pyjamas",
    "invictus gaming": "invictus gaming",
    "ig": "invictus gaming",
}


# ---------------------------------------------------------------------------
# League → best-of mapping
# ---------------------------------------------------------------------------
# Bo3 leagues (major regions + some minor that use Bo3 regular season)
# NOTE: substring matching is used, so order matters — more specific patterns
# should come first.  "lpl" would also match "lplol" (LPLOL), so we use
# "lpl" with an exclusion check inside infer_best_of().
_BO3_LEAGUE_PATTERNS: list[str] = [
    "lck",
    "lpl",
    "lec",
    "lck challengers",
    "cblo",
    "ljl",
    "lcs na",
    "lcs",
    "tcl",
    "lcp",
    "tj sports lol / lpl",
]

# Leagues whose names contain a Bo3 pattern substring but are actually Bo1.
_BO1_OVERRIDE_PATTERNS: list[str] = [
    "lplol",  # LPLOL, Inygon / LPLOL
    "lpol",
    "nacl",   # NACL (contains "lcs" substring via "na challengers league")
    "na challengers",
]

# Explicit Bo5 stages and tournament qualifiers
_BO5_LEAGUE_PATTERNS: list[str] = [
    "playoffs",
    "play-offs",
    "play off",
    "bracket",
    "finals",
    "semifinals",
    "lck road to msi",
    "regional qualifiers",
]
# Bo5 is only for playoffs / finals — we cannot reliably detect that from
# the league name alone, so default regular-season Bo3 leagues to 3.
# Everything else defaults to Bo1.

COMPETITION_MARKERS: tuple[tuple[str, str], ...] = (
    ("lck challengers", "lck_cl"),
    ("lck cl", "lck_cl"),
    ("lck challenge", "lck_cl"),
    ("lck", "lck"),
    ("lplol", "lpol"),
    ("lpol", "lpol"),
    ("lpl", "lpl"),
    ("lec", "lec"),
    ("na challengers", "nacl"),
    ("nacl", "nacl"),
    ("lcs", "lcs"),
    ("lastlap", "les"),
    ("superliga", "les"),
    ("lvp", "les"),
    ("les", "les"),
    ("lfl", "lfl"),
    ("prime league", "prime_league"),
    ("emea masters", "emea_masters"),
    ("circuito desafiante", "circuito_desafiante"),
    ("cd", "circuito_desafiante"),
    ("cblol", "cblol"),
    ("lcp", "lcp"),
    ("ljl", "ljl"),
    ("lrs", "lrs"),
    ("lrn", "lrn"),
    ("tcl", "tcl"),
    ("nlc", "nlc"),
    ("rift legends", "rift_legends"),
    ("liga regional sur", "lrs"),
    ("japan league", "ljl"),
    ("pg nationals", "pg_nationals"),
    ("kespa cup", "kespa_cup"),
    ("gll", "gll"),
    ("lit", "lit"),
    ("hellenic legends", "hll"),
    ("hll", "hll"),
    ("road of legends", "road_of_legends"),
    ("hitpoint", "hitpoint"),
    ("esports world cup", "ewc"),
    ("world cup", "ewc"),
    ("wolrd cup", "ewc"),
    ("ewc", "ewc"),
    ("msi", "msi"),
)


def competition_family(value: object) -> str | None:
    """Return a conservative canonical family for a source competition label."""

    if value is None:
        return None
    normalized = alias_lookup_key(str(value))
    if not normalized:
        return None
    for marker, family in COMPETITION_MARKERS:
        if re.search(rf"\b{re.escape(marker)}\b", normalized):
            return family
    return None


def infer_best_of(league: str | None) -> int:
    """Return the best-of value for a given league name.

    Uses substring matching so that variants like "Riot LoL / LCK" still
    resolve correctly.
    Precedence:
    1. Bo5 patterns (playoffs, finals, Road to MSI)
    2. Bo1 overrides (e.g. LPLOL, NACL)
    3. Bo3 leagues (LCK, LPL, LEC, etc.)
    4. Default: 1 (Bo1)
    """
    if not league:
        return 1
    low = league.lower()
    for pattern in _BO5_LEAGUE_PATTERNS:
        if pattern in low:
            return 5
    for pattern in _BO1_OVERRIDE_PATTERNS:
        if pattern in low:
            return 1
    for pattern in _BO3_LEAGUE_PATTERNS:
        if pattern in low:
            return 3
    return 1


def canonical_team_key(
    name: str,
    *,
    league: str | None = None,
    source_system: str | None = None,
    match_date: date | datetime | str | None = None,
) -> str:
    """Normalize a raw team name into a cross-bookmaker key."""

    scoped = resolve_scoped_alias(
        name,
        context=AliasContext(source_system=source_system, league=league, match_date=match_date),
    )
    # A scoped alias target still needs the deterministic built-in alias pass.
    # For example an old DB alias can target ``ThunderTalk Gaming`` while the
    # canonical key is ``thundertalk``. Returning ``normalized_target`` here
    # used two different identities for equivalent source spellings and could
    # make a valid bookmaker offer look unrelated to its canonical match.
    return _canonical_key_after_alias(scoped.target_name if scoped.target_name and not scoped.blocked else name, league)


def _canonical_key_after_alias(name: str, league: str | None) -> str:
    """Apply the existing deterministic normalization after one alias decision."""
    normalized = normalize_team_name(name)
    league_norm = normalize_team_name(league or "")

    # Context-sensitive KOI handling: in LEC/EWC "Movistar KOI" is the main
    # roster, but in Spanish ERL feeds (LES/LVP/Lastlap) bookmakers often show
    # the academy as just "Movistar KOI" while other books use
    # "Movistar KOI Fenix" / "KOI Academy".  Without this rule those rows are
    # merged and whichever bookmaker refreshed last controls the roster used
    # for automatic predictions.
    if normalized in {"movistar koi", "movister koi", "koi"}:
        if any(marker in league_norm for marker in ("les", "lvp", "lastlap")):
            return "movistar koi fenix"
        return "movistar koi"
    if normalized in {"movistar koi fenix", "movistar koi academy", "koi academy", "mkf"}:
        return "movistar koi fenix"

    compact = normalized.replace(" ", "")
    if normalized in TEAM_ALIASES:
        return TEAM_ALIASES[normalized]
    if compact in TEAM_ALIASES:
        return TEAM_ALIASES[compact]
    return normalized


def resolve_canonical_match(
    *,
    raw_team_a: str,
    raw_team_b: str,
    match_start_time: str | None = None,
    league: str | None = None,
    source_system: str | None = None,
    min_confidence: float = 0.78,
    best_of: int | None = None,
) -> int:
    """Find or create a canonical match shared by all bookmakers."""

    start_norm = normalize_start_time(match_start_time)
    league_norm = normalize_league(league)
    team_a_key = canonical_team_key(
        raw_team_a,
        league=league,
        source_system=source_system,
        match_date=start_norm,
    )
    team_b_key = canonical_team_key(
        raw_team_b,
        league=league,
        source_system=source_system,
        match_date=start_norm,
    )

    # Prefer scraper-provided best_of over heuristic; fall back to heuristic
    # when the scraper did not detect the format (e.g. no "liczba map" line).
    best_of_val = best_of if best_of is not None else infer_best_of(league)

    # Search across upcoming, expired, AND finished matches when we have a
    # trustworthy start time.  Including finished is essential: once GOL.GG
    # marks a match finished, later bookmaker odds for the same real match
    # must attach to the finished row, not create an expired duplicate.
    # The time_match_score naturally rejects old finished matches (decays to
    # 0.0 at 3+ days), and the identical-teams boost in canonical_match_score
    # is intentionally NOT applied to finished candidates, so finished matches
    # only match when both teams AND time are close.
    # Without a trustworthy time restrict to upcoming rows only, so a new
    # feed item is not accidentally attached to an old finished match.
    candidate_where = "WHERE status IN ('upcoming', 'expired', 'finished')" if start_norm else "WHERE status = 'upcoming'"
    candidate_params: tuple = ()

    with transaction() as connection:
        candidates = connection.execute(
            f"""
            SELECT * FROM canonical_matches
            {candidate_where}
            ORDER BY id DESC
            LIMIT 1000
            """,
            candidate_params,
        ).fetchall()
        best_id: int | None = None
        best_score = 0.0
        for candidate in candidates:
            score = canonical_match_score(
                team_a_key,
                team_b_key,
                start_norm,
                league_norm,
                dict(candidate),
            )
            if score > best_score:
                best_score = score
                best_id = int(candidate["id"])
        if best_id is not None and best_score >= min_confidence:
            # When the scraper provides best_of, always update it (the scraper
            # has direct evidence from the bookmaker page).  Otherwise only
            # fill in null values via COALESCE to preserve manual edits.
            if best_of is not None:
                best_of_sql = "best_of = ?"
                best_of_params: tuple[int, ...] = (best_of_val,)
            else:
                best_of_sql = "best_of = COALESCE(best_of, ?)"
                best_of_params = (best_of_val,)
            connection.execute(
                f"""
                UPDATE canonical_matches
                SET start_time_normalized = COALESCE(start_time_normalized, ?),
                    league = COALESCE(league, ?),
                    match_confidence = GREATEST(match_confidence, ?),
                    {best_of_sql}
                WHERE id = ?
                """,
                (start_norm, league, best_score, *best_of_params, best_id),
            )
            return best_id

        canonical_key = build_canonical_key(team_a_key, team_b_key, start_norm, league_norm)
        connection.execute(
            """
            INSERT INTO canonical_matches(
                canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b,
                start_time_normalized, league, match_confidence, best_of, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, ?, 'upcoming')
            ON CONFLICT (canonical_key) DO NOTHING
            """,
            (canonical_key, raw_team_a, raw_team_b, team_a_key, team_b_key, start_norm, league, best_of_val),
        )
        row = connection.execute("SELECT id FROM canonical_matches WHERE canonical_key = ?", (canonical_key,)).fetchone()
        return int(row["id"])


def canonical_match_score(
    team_a_key: str,
    team_b_key: str,
    start_norm: str | None,
    league_norm: str | None,
    candidate: dict[str, Any],
) -> float:
    """Score whether a raw bookmaker event belongs to a canonical match."""

    cand_a = str(candidate.get("normalized_team_a") or "")
    cand_b = str(candidate.get("normalized_team_b") or "")

    direct_marker_ok = not (
        _squad_marker_mismatch(team_a_key, cand_a)
        or _squad_marker_mismatch(team_b_key, cand_b)
    )
    swapped_marker_ok = not (
        _squad_marker_mismatch(team_a_key, cand_b)
        or _squad_marker_mismatch(team_b_key, cand_a)
    )

    direct_a = _team_identity_similarity(team_a_key, cand_a) if direct_marker_ok else 0.0
    direct_b = _team_identity_similarity(team_b_key, cand_b) if direct_marker_ok else 0.0
    direct = (direct_a + direct_b) / 2 if direct_marker_ok else 0.0
    swapped_a = _team_identity_similarity(team_a_key, cand_b) if swapped_marker_ok else 0.0
    swapped_b = _team_identity_similarity(team_b_key, cand_a) if swapped_marker_ok else 0.0
    swapped = (swapped_a + swapped_b) / 2 if swapped_marker_ok else 0.0

    # An average alone is unsafe for academy leagues: unrelated names such as
    # "Gen.G Challengers" and "DN SOOPers Challengers" share a long suffix,
    # so two unrelated teams can clear the aggregate threshold when kickoff
    # times coincide. A canonical merge requires both team identities to meet
    # the minimum similarity in at least one orientation.
    if direct_marker_ok and min(direct_a, direct_b) < 0.68:
        direct = 0.0
    if swapped_marker_ok and min(swapped_a, swapped_b) < 0.68:
        swapped = 0.0
    team_score = max(direct, swapped)
    if team_score < 0.68:
        return team_score * 0.7


    incoming_family = competition_family(league_norm)
    candidate_family = competition_family(candidate.get("league"))
    if incoming_family and candidate_family and incoming_family != candidate_family:
        return 0.0
    time_score = time_match_score(start_norm, candidate.get("start_time_normalized"))
    league_score = league_match_score(league_norm, candidate.get("league"))
    score = 0.72 * team_score + 0.23 * time_score + 0.05 * league_score

    # Boost: identical teams on an *upcoming* match means it IS the same
    # canonical match only when the start times are still close.  Bookmakers can
    # move the same event by minutes/hours, but tournament brackets can also
    # produce genuine rematches a few days apart.  Keep the boost inside the
    # 4-hour time window so those future rematches become separate canonicals.
    if team_score >= 0.95 and candidate.get("status") == "upcoming" and time_score >= 0.35:
        score = max(score, 0.85)

    return score


def _team_identity_similarity(left: str, right: str) -> float:
    """Compare organization identities without generic squad suffix inflation."""

    def organization_key(value: str) -> str:
        normalized = normalize_team_name(value)
        tokens = [
            token
            for token in normalized.split()
            if token not in {"academy", "challengers", "fenix", "pride", "bee", "cl", "eclipse", "youth", "junior"}
        ]
        return " ".join(tokens)

    left_org = organization_key(left)
    right_org = organization_key(right)
    if left_org and right_org:
        return similarity(left_org, right_org)
    return similarity(left, right)


def _squad_marker_mismatch(left: str, right: str) -> bool:
    """Return True when two names are same org but different squad level.

    This deliberately handles the common bookmaker problem where an academy
    suffix is present in one source and omitted in another.  For identical orgs
    (e.g. Movistar KOI), a main-vs-Fenix/Academy mismatch must never be rescued
    by an exact kickoff time because it corrupts rosters and automatic
    predictions.
    """

    l_norm = normalize_team_name(left)
    r_norm = normalize_team_name(right)
    if l_norm == r_norm:
        return False
    l_has = any(marker in l_norm for marker in ("academy", "challengers", "fenix", "fénix"))
    r_has = any(marker in r_norm for marker in ("academy", "challengers", "fenix", "fénix"))
    if l_has == r_has:
        return False
    l_base = re.sub(r"\b(academy|challengers|fenix|fénix)\b", "", l_norm).strip()
    r_base = re.sub(r"\b(academy|challengers|fenix|fénix)\b", "", r_norm).strip()
    return bool(l_base and r_base and similarity(l_base, r_base) >= 0.70)


def time_match_score(left: str | None, right: str | None) -> float:
    """Score start-time compatibility."""

    if not left or not right:
        return 0.45
    left_dt = parse_iso(left)
    right_dt = parse_iso(right)
    if not left_dt or not right_dt:
        return 0.45 if left == right else 0.0
    diff_minutes = abs((left_dt - right_dt).total_seconds()) / 60
    if diff_minutes <= 20:
        return 1.0
    if diff_minutes <= 90:
        return 0.75
    if diff_minutes <= 240:
        return 0.35
    return 0.0


def league_match_score(left: str | None, right: str | None) -> float:
    """Score league-name compatibility."""

    if not left or not right:
        return 0.5
    return similarity(left, right)


def normalize_start_time(value: str | None, *, now_utc: datetime | None = None) -> str | None:
    """Normalize bookmaker start labels to ISO-like UTC where possible."""
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    # Explicitly reject countdown labels before any other parser gets a chance
    # to interpret them as clock times. Betfan and similar feeds can emit a
    # changing HH:MM:SS countdown shortly before kickoff; treating that as a
    # start time creates a new match key/canonical row on every scrape and
    # corrupts odds history used for CLV analysis.
    if is_countdown_start_label(raw):
        return None
    if raw.isdigit() and len(raw) >= 12:
        return datetime.fromtimestamp(int(raw) / 1000, tz=UTC).replace(microsecond=0).isoformat()
    match = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})\s+(\d{1,2}):(\d{2})$", raw)
    if match:
        day, month, year, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute, tzinfo=UTC).isoformat()
    # eFortuna full-date format with day-of-week prefix, e.g. "śro., 14.06.2026, 22:00"
    match = re.match(r"^[a-ząćęłńóśźż]{2,8}\.,\s*(\d{2})\.(\d{2})\.(\d{4}),\s*(\d{1,2}):(\d{2})$", raw, re.IGNORECASE)
    if match:
        day, month, year, hour, minute = map(int, match.groups())
        return datetime(year, month, day, hour, minute, tzinfo=UTC).isoformat()
    # Polish bookmaker labels without year, e.g. "Śr 15. Lip, 11:20".
    # Treat the date as the next occurrence in UTC year-space. This keeps
    # future listings visible while avoiding stale rows around year rollover.
    polish_months = {
        "sty": 1,
        "lut": 2,
        "mar": 3,
        "kwi": 4,
        "maj": 5,
        "cze": 6,
        "lip": 7,
        "sie": 8,
        "wrz": 9,
        "paź": 10,
        "paz": 10,
        "lis": 11,
        "gru": 12,
    }
    match = re.match(
        r"^[a-ząćęłńóśźż]{2,8}\s+(\d{1,2})\.\s*([a-ząćęłńóśźż]{3}),\s*(\d{1,2}):(\d{2})$",
        raw,
        re.IGNORECASE,
    )
    if match:
        today = (now_utc or datetime.now(UTC)).date()
        day_s, month_s, hour_s, minute_s = match.groups()
        month = polish_months.get(month_s.lower())
        if month:
            year = today.year
            candidate = datetime(year, month, int(day_s), int(hour_s), int(minute_s), tzinfo=UTC)
            if candidate.date() < today:
                candidate = datetime(year + 1, month, int(day_s), int(hour_s), int(minute_s), tzinfo=UTC)
            return candidate.isoformat()
    # Use UTC date so "dziś/dzisiaj" resolves correctly even when the
    # scraper runs after midnight in the local timezone (CEST/UTC+2).
    today = (now_utc or datetime.now(UTC)).date()
    rel = re.match(r"^(?:dzi[śs]|dzisiaj)\s+(\d{1,2}):(\d{2})$", raw, re.IGNORECASE)
    if rel:
        hour, minute = map(int, rel.groups())
        return datetime(today.year, today.month, today.day, hour, minute, tzinfo=UTC).isoformat()
    rel = re.match(r"^jutro\s+(\d{1,2}):(\d{2})$", raw, re.IGNORECASE)
    if rel:
        target = today + timedelta(days=1)
        hour, minute = map(int, rel.groups())
        return datetime(target.year, target.month, target.day, hour, minute, tzinfo=UTC).isoformat()
    parsed = parse_iso(raw)
    if parsed:
        return parsed.replace(microsecond=0).isoformat()
    return None


def is_countdown_start_label(value: str | None) -> bool:
    """Return True for unstable countdown labels such as HH:MM:SS."""

    if not value:
        return False
    raw = str(value).strip()
    match = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", raw)
    if not match:
        return False
    hours, minutes, seconds = map(int, match.groups())
    return minutes < 60 and seconds < 60 and hours < 48


def parse_iso(value: str | None) -> datetime | None:
    """Parse ISO timestamp if possible."""

    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed.year < 2000 or parsed.year > 2100:
        return None
    return parsed


def normalize_league(value: str | None) -> str | None:
    """Normalize league text for matching."""

    normalized = normalize_team_name(value or "")
    return normalized or None


def build_canonical_key(team_a_key: str, team_b_key: str, start_norm: str | None, league_norm: str | None) -> str:
    """Build stable unique key for a canonical match."""

    left, right = sorted([team_a_key, team_b_key])
    time_bucket = start_norm[:13] if start_norm else "unknown"
    base = f"{left}|{right}|{time_bucket}|{league_norm or 'unknown'}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"{base}|{digest}"


def canonical_match_overview(limit: int = 100):
    """Return latest canonical matches with bookmaker coverage."""

    rows = query_df(
        """
        WITH ranked AS (
            SELECT
                cm.id AS canonical_match_id,
                cm.team_a_name,
                cm.team_b_name,
                cm.normalized_team_a,
                cm.normalized_team_b,
                cm.start_time_normalized,
                cm.league,
                b.name AS bookmaker,
                os.raw_team_a,
                os.raw_team_b,
                os.odds_a,
                os.odds_b,
                os.scraped_at,
                ROW_NUMBER() OVER (
                    PARTITION BY cm.id, b.name
                    ORDER BY os.scraped_at DESC, os.id DESC
                ) AS rn
            FROM canonical_matches cm
            LEFT JOIN odds_snapshots os ON os.canonical_match_id = cm.id AND os.market_type = 'match_winner'
            LEFT JOIN bookmakers b ON b.id = os.bookmaker_id
        )
        SELECT * FROM ranked
        WHERE rn = 1 OR rn IS NULL
        """,
    )
    if rows.empty:
        return rows

    aggregated: list[dict[str, Any]] = []
    for match_id, group in rows.groupby("canonical_match_id", dropna=False):
        first = group.iloc[0]
        odds_a: list[float] = []
        odds_b: list[float] = []
        bookmakers: list[str] = []
        for row in group.to_dict("records"):
            if not row.get("bookmaker"):
                continue
            aligned = align_snapshot_odds(
                str(first["normalized_team_a"]),
                str(first["normalized_team_b"]),
                str(row.get("raw_team_a") or ""),
                str(row.get("raw_team_b") or ""),
                row.get("odds_a"),
                row.get("odds_b"),
                league=first.get("league"), source_system="bookmaker",
                match_date=first.get("start_time_normalized"),
            )
            if aligned is None:
                continue
            aligned_a, aligned_b = aligned
            odds_a.append(aligned_a)
            odds_b.append(aligned_b)
            bookmakers.append(str(row["bookmaker"]))
        aggregated.append(
            {
                "canonical_match_id": int(match_id),
                "team_a_name": first["team_a_name"],
                "team_b_name": first["team_b_name"],
                "start_time_normalized": first["start_time_normalized"],
                "league": first["league"],
                "bookmaker_count": len(set(bookmakers)),
                "bookmakers": ",".join(sorted(set(bookmakers))),
                "min_odds_a": min(odds_a) if odds_a else None,
                "max_odds_a": max(odds_a) if odds_a else None,
                "min_odds_b": min(odds_b) if odds_b else None,
                "max_odds_b": max(odds_b) if odds_b else None,
                "last_scraped_at": group["scraped_at"].dropna().max() if "scraped_at" in group else None,
            }
        )
    frame = pd.DataFrame(aggregated)
    if frame.empty:
        return frame
    frame = frame.sort_values(["last_scraped_at", "start_time_normalized"], ascending=[False, True], na_position="last")
    return frame.head(limit).reset_index(drop=True)


def align_snapshot_odds(
    canonical_a: str,
    canonical_b: str,
    raw_team_a: str,
    raw_team_b: str,
    odds_a: Any,
    odds_b: Any,
    *,
    league: str | None = None,
    source_system: str | None = None,
    match_date: date | datetime | str | None = None,
) -> tuple[float, float] | None:
    """Return prices only for one exact, context-scoped participant assignment."""

    context = AliasContext(source_system=source_system, league=league, match_date=match_date)
    keys: list[str] = []
    for name in (canonical_a, canonical_b, raw_team_a, raw_team_b):
        if not isinstance(name, str) or not name.strip():
            return None
        resolution = resolve_scoped_alias(name, context=context)
        if resolution.blocked or resolution.source in {"ambiguous", "durable-ambiguous"}:
            return None
        key = _canonical_key_after_alias(resolution.target_name or name, league)
        # A global short-form expansion does not establish contextual identity.
        short_key = alias_lookup_key(name).replace(" ", "")
        if (
            0 < len(short_key) <= 3 and not resolution.target_name
            and (not league or not source_system)
            and key.replace(" ", "") != short_key
        ):
            return None
        if not key:
            return None
        keys.append(key)
    ca, cb, ra, rb = keys
    if ca == cb or ra == rb:
        return None
    direct = ca == ra and cb == rb
    reversed_pair = ca == rb and cb == ra
    if direct == reversed_pair:
        return None
    prices: list[float] = []
    for value in (odds_a, odds_b):
        if value is None or isinstance(value, bool):
            return None
        try:
            price = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(price) or price <= 1:
            return None
        prices.append(price)
    left, right = prices
    return (left, right) if direct else (right, left)
