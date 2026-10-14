"""Services for odds snapshots and upcoming matches."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from betting_app.core.db import is_pg, query_df, transaction
from betting_app.core.matching import normalize_team_name
from betting_app.services.canonical_match_service import (
    is_countdown_start_label,
    normalize_start_time,
    resolve_canonical_match,
)


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO string."""

    return datetime.now(UTC).isoformat()


def get_or_create_bookmaker(name: str, base_url: str | None = None) -> int:
    """Return bookmaker ID for an *existing* bookmaker.

    Raises ValueError if the bookmaker name is not in the known whitelist.
    This prevents phantom bookmaker rows (e.g. "fortuna", "manual") from
    being auto-created by mistake.
    """

    _KNOWN_BOOKMAKERS = {
        "sts", "betclic", "superbet", "efortuna", "betfan", "totalbet", "lebull",
        "pinnacle", "kalshi",
    }

    if name not in _KNOWN_BOOKMAKERS:
        raise ValueError(
            f"Unknown bookmaker '{name}'. Known: {sorted(_KNOWN_BOOKMAKERS)}"
        )

    with transaction() as connection:
        row = connection.execute("SELECT id FROM bookmakers WHERE name = ?", (name,)).fetchone()
        if row is None:
            default_urls = {
                "pinnacle": "https://www.pinnacle.com/",
                "kalshi": "https://kalshi.com/",
                "sts": "https://www.sts.pl/",
                "betclic": "https://www.betclic.pl/",
                "superbet": "https://superbet.pl/",
                "efortuna": "https://www.efortuna.pl/",
                "betfan": "https://betfan.pl/",
                "totalbet": "https://totalbet.pl/",
                "lebull": "https://www.lebull.pl/",
            }
            url = base_url or default_urls.get(name, "")
            connection.execute(
                "INSERT INTO bookmakers (name, base_url, is_active) VALUES (?, ?, '1')",
                (name, url),
            )
            row = connection.execute("SELECT id FROM bookmakers WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise ValueError(f"Bookmaker '{name}' is known but could not be registered.")
        return int(row["id"])

def make_match_key(bookmaker: str, raw_team_a: str, raw_team_b: str, start_time: str | None) -> str:
    """Build a stable bookmaker match key from raw data."""

    # Use the normalized start when possible and collapse unparseable labels
    # (notably countdowns such as HH:MM:SS) to "unknown".  Otherwise every
    # changing countdown string becomes a different bookmaker_match_key and can
    # cascade into duplicate upcoming/canonical rows.
    normalized_start = normalize_start_time(start_time)

    return "|".join(
        [bookmaker, normalize_team_name(raw_team_a), normalize_team_name(raw_team_b), normalized_start or "unknown"]
    )


def _resolve_golgg_team_id(team_name: str, league: str | None = None) -> int | None:
    """Resolve a GOL.GG team ID from a bookmaker team name.

    Uses suggest_mapping() to find the matching GOL.GG team name,
    then looks up the ID in the golgg_teams table.
    Returns None if no mapping or no ID found.
    """
    from betting_app.services.mapping_service import suggest_mapping

    normalized = normalize_team_name(team_name)
    league_norm = normalize_team_name(league or "")

    # Context-sensitive Movistar KOI disambiguation.  Some Spanish ERL feeds
    # omit the Fenix/Academy suffix and publish "Movistar KOI" in LES/LVP;
    # that must resolve to the academy GOL.GG team, not the LEC main roster.
    forced_golgg_name: str | None = None
    if normalized in {"movistar koi", "movister koi", "koi"}:
        forced_golgg_name = "Movistar KOI Fenix" if any(
            marker in league_norm for marker in ("les", "lvp", "lastlap")
        ) else "Movistar KOI"
    elif normalized in {"movistar koi fenix", "movistar koi academy", "koi academy", "mkf"}:
        forced_golgg_name = "Movistar KOI Fenix"

    if forced_golgg_name:
        golgg_name = forced_golgg_name
    else:
        golgg_name, conf, source = suggest_mapping(
            team_name,
            source_system="bookmaker",
            league=league,
        )
        if not golgg_name or conf <= 0:
            return None

    # Try exact team_name match first
    with transaction() as connection:
        row = connection.execute(
            "SELECT id FROM golgg_teams WHERE team_name = ?",
            (golgg_name,),
        ).fetchone()
        if row:
            return int(row["id"])

        # Fallback: try normalized_name
        norm = normalize_team_name(golgg_name)
        row = connection.execute(
            "SELECT id FROM golgg_teams WHERE normalized_name = ?",
            (norm,),
        ).fetchone()
        if row:
            return int(row["id"])

    return None


def upsert_upcoming_match(
    bookmaker: str,
    raw_team_a: str,
    raw_team_b: str,
    match_start_time: str | None = None,
    league: str | None = None,
    offer_url: str | None = None,
    canonical_team_a: str | None = None,
    canonical_team_b: str | None = None,
    best_of: int | None = None,
) -> int:
    """Create or update an upcoming match and return its ID."""

    key = make_match_key(bookmaker, raw_team_a, raw_team_b, match_start_time)
    bookmaker_id = get_or_create_bookmaker(bookmaker)
    norm_a = normalize_team_name(raw_team_a)
    norm_b = normalize_team_name(raw_team_b)
    canonical_match_id = _find_existing_canonical_for_unstable_start(
        bookmaker_id=bookmaker_id,
        norm_a=norm_a,
        norm_b=norm_b,
        match_start_time=match_start_time,
        league=league,
    )
    if canonical_match_id is None:
        canonical_match_id = resolve_canonical_match(
            raw_team_a=canonical_team_a or raw_team_a,
            raw_team_b=canonical_team_b or raw_team_b,
            match_start_time=match_start_time,
            league=league,
            source_system="bookmaker",
            best_of=best_of,
        )

    # Resolve GOL.GG team IDs using suggest_mapping()
    # Preserve the original spelling for GOL.GG resolution. Passing the
    # already-normalized value loses meaningful squad qualifiers such as
    # "GIANTX iTero" and can resolve it to the parent/other academy team.
    team_a_id = _resolve_golgg_team_id(raw_team_a, league)
    team_b_id = _resolve_golgg_team_id(raw_team_b, league)

    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO upcoming_matches(
                bookmaker_id, bookmaker_match_key, canonical_match_id,
                raw_team_a, raw_team_b, normalized_team_a, normalized_team_b,
                match_start_time, league, offer_url, last_seen_at,
                team_a_golgg_id, team_b_golgg_id, is_live
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, 0)
            ON CONFLICT(bookmaker_match_key) DO UPDATE SET
                canonical_match_id = excluded.canonical_match_id,
                raw_team_a = excluded.raw_team_a,
                raw_team_b = excluded.raw_team_b,
                normalized_team_a = excluded.normalized_team_a,
                normalized_team_b = excluded.normalized_team_b,
                match_start_time = excluded.match_start_time,
                league = excluded.league,
                offer_url = COALESCE(excluded.offer_url, upcoming_matches.offer_url),
                last_seen_at = CURRENT_TIMESTAMP,
                team_a_golgg_id = excluded.team_a_golgg_id,
                team_b_golgg_id = excluded.team_b_golgg_id
            """,
            (
                bookmaker_id,
                key,
                canonical_match_id,
                raw_team_a,
                raw_team_b,
                norm_a,
                norm_b,
                match_start_time,
                league,
                offer_url,
                team_a_id,
                team_b_id,
            ),
        )
        row = connection.execute("SELECT id FROM upcoming_matches WHERE bookmaker_match_key = ?", (key,)).fetchone()
        return int(row["id"])


def _find_existing_canonical_for_unstable_start(
    *,
    bookmaker_id: int,
    norm_a: str,
    norm_b: str,
    match_start_time: str | None,
    league: str | None = None,
) -> int | None:
    """Reuse a known canonical row when a feed only sends an unstable countdown.

    Some bookmakers switch a real kickoff label (for example "Dziś 22:00") to a
    changing countdown ("04:55:53") shortly before the event starts.  The
    countdown has no date, so resolving it globally can create orphan canonical
    rows.  If the same bookmaker already emitted the same team pair with a
    parseable/non-countdown start label, reuse that canonical id and keep the
    odds history attached to the real match.
    """

    if not is_countdown_start_label(match_start_time):
        return None

    league_norm = normalize_team_name(league or "")
    with transaction() as connection:
        rows = connection.execute(
            """
            SELECT um.canonical_match_id, um.match_start_time, um.league, cm.status
            FROM upcoming_matches um
            JOIN canonical_matches cm ON cm.id = um.canonical_match_id
            WHERE um.bookmaker_id = ?
              AND um.canonical_match_id IS NOT NULL
              AND (
                    (um.normalized_team_a = ? AND um.normalized_team_b = ?)
                 OR (um.normalized_team_a = ? AND um.normalized_team_b = ?)
              )
            ORDER BY
              CASE WHEN cm.status = 'finished' THEN 3 WHEN cm.status = 'upcoming' THEN 2 WHEN cm.status = 'expired' THEN 1 ELSE 0 END DESC,
              um.id DESC
            LIMIT 25
            """,
            (bookmaker_id, norm_a, norm_b, norm_b, norm_a),
        ).fetchall()

    best_id: int | None = None
    best_score = -1
    for row in rows:
        existing_start = str(row["match_start_time"] or "")
        if is_countdown_start_label(existing_start) or normalize_start_time(existing_start) is None:
            continue
        existing_league_norm = normalize_team_name(str(row["league"] or ""))
        league_score = 1 if not league_norm or not existing_league_norm or league_norm == existing_league_norm else 0
        status_score = {"finished": 3, "upcoming": 2, "expired": 1}.get(str(row["status"] or ""), 0)
        score = status_score * 10 + league_score
        if score > best_score:
            best_score = score
            best_id = int(row["canonical_match_id"])
    return best_id


def _available_decimal_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price > 1 else None


def insert_odds_snapshot(snapshot: dict[str, Any]) -> int:
    """Retain one observed source state without orientation or clock repairs.

    Prices that cannot represent a decimal offer are unavailable; original raw
    values remain in the payload. Only explicit bool/int 0/1 live states are
    operationally known. Timestamp-key upserts remain mutable observations.
    """

    bookmaker_name = str(snapshot.get("bookmaker", "manual"))
    bookmaker_id = get_or_create_bookmaker(bookmaker_name, snapshot.get("base_url"))
    scraped_at = str(snapshot.get("scraped_at") or utc_now_iso())
    match_id = upsert_upcoming_match(
        bookmaker=bookmaker_name,
        raw_team_a=str(snapshot["raw_team_a"]),
        raw_team_b=str(snapshot["raw_team_b"]),
        match_start_time=snapshot.get("match_start_time"),
        league=snapshot.get("raw_league"),
        offer_url=snapshot.get("offer_url"),
        canonical_team_a=snapshot.get("mapped_team_a"),
        canonical_team_b=snapshot.get("mapped_team_b"),
        best_of=snapshot.get("best_of"),
    )
    canonical_match_id = get_upcoming_canonical_match_id(match_id)

    # Retain source orientation; canonical alignment belongs to consumers.
    raw_team_a = snapshot["raw_team_a"]
    raw_team_b = snapshot["raw_team_b"]
    odds_a = _available_decimal_price(snapshot.get("odds_a"))
    odds_b = _available_decimal_price(snapshot.get("odds_b"))
    live = snapshot.get("is_live")
    is_live = int(live) if isinstance(live, bool) or type(live) is int and live in (0, 1) else None

    raw_payload = snapshot.get("raw_payload")
    if raw_payload is None:
        raw_payload = {"odds_a": snapshot.get("odds_a"), "odds_b": snapshot.get("odds_b"), "is_live": snapshot.get("is_live")}
    if not isinstance(raw_payload, str):
        raw_payload = json.dumps(raw_payload, ensure_ascii=False, default=str)

    params = (
        bookmaker_id,
        match_id,
        canonical_match_id,
        scraped_at,
        snapshot.get("source_url"),
        snapshot.get("offer_url"),
        raw_team_a,
        raw_team_b,
        odds_a,
        odds_b,
        snapshot.get("market_type", "match_winner"),
        is_live,
        raw_payload,
    )

    with transaction() as connection:
        if is_pg():
            # Production odds_snapshots is a Timescale hypertable with a partial
            # unique index:
            #   idx_odds_unique_snapshot
            #   (canonical_match_id, bookmaker_id, scraped_at)
            #   WHERE market_type = 'match_winner'
            # Scrapers can legitimately emit the same bookmaker/match/scrape
            # timestamp twice in one run (observed for STS).  Treat that as an
            # idempotent upsert instead of failing the whole scrape cycle.
            row = connection.execute(
                """
                INSERT INTO odds_snapshots(
                    bookmaker_id, match_id, canonical_match_id, scraped_at, source_url, offer_url,
                    raw_team_a, raw_team_b, odds_a, odds_b, market_type, is_live, raw_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (canonical_match_id, bookmaker_id, scraped_at)
                    WHERE market_type = 'match_winner'
                DO UPDATE SET
                    match_id = EXCLUDED.match_id,
                    source_url = EXCLUDED.source_url,
                    offer_url = EXCLUDED.offer_url,
                    raw_team_a = EXCLUDED.raw_team_a,
                    raw_team_b = EXCLUDED.raw_team_b,
                    odds_a = EXCLUDED.odds_a,
                    odds_b = EXCLUDED.odds_b,
                    is_live = EXCLUDED.is_live,
                    raw_payload = EXCLUDED.raw_payload
                RETURNING id
                """,
                params,
            ).fetchone()
            return int(row["id"]) if row else 0

        cursor = connection.execute(
            """
            INSERT INTO odds_snapshots(
                bookmaker_id, match_id, canonical_match_id, scraped_at, source_url, offer_url,
                raw_team_a, raw_team_b, odds_a, odds_b, market_type, is_live, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (canonical_match_id, bookmaker_id, scraped_at)
                WHERE market_type = 'match_winner'
            DO UPDATE SET
                match_id = excluded.match_id,
                source_url = excluded.source_url,
                offer_url = excluded.offer_url,
                raw_team_a = excluded.raw_team_a,
                raw_team_b = excluded.raw_team_b,
                odds_a = excluded.odds_a,
                odds_b = excluded.odds_b,
                is_live = excluded.is_live,
                raw_payload = excluded.raw_payload
            RETURNING id
            """,
            params,
        )
        row = cursor.fetchone()
        return int(row["id"]) if row else 0


def start_scrape_run(
    bookmaker: str,
    scraper_name: str,
    scraper_version: str | None = None,
    source_url: str | None = None,
    request_url: str | None = None,
) -> int:
    """Create a scrape run audit record and return its ID."""

    bookmaker_id = get_or_create_bookmaker(bookmaker)
    with transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO scrape_runs(
                bookmaker_id, scraper_name, scraper_version, source_url, request_url, status, started_at
            ) VALUES (?, ?, ?, ?, ?, 'running', CURRENT_TIMESTAMP)
            """,
            (bookmaker_id, scraper_name, scraper_version, source_url, request_url),
        )
        return int(cursor.lastrowid)


def finish_scrape_run(
    scrape_run_id: int,
    *,
    status: str = "success",
    items_seen: int = 0,
    items_inserted: int = 0,
    error: str | None = None,
) -> None:
    """Mark a scrape run as finished."""

    with transaction() as connection:
        connection.execute(
            """
            UPDATE scrape_runs
            SET finished_at = CURRENT_TIMESTAMP,
                status = ?,
                items_seen = ?,
                items_inserted = ?,
                error = ?
            WHERE id = ?
            """,
            (status, items_seen, items_inserted, error, scrape_run_id),
        )


def _market_type_from_name(market_name: str) -> str:
    """Map bookmaker market labels to coarse internal market types."""

    normalized = market_name.strip().lower()
    if normalized in {"zwycięzca meczu", "mecz", "zwyciezca meczu"}:
        return "match_winner"
    if "handicap" in normalized:
        return "handicap"
    if "dokładny wynik" in normalized or "dokladny wynik" in normalized:
        return "correct_score"
    if "mapa" in normalized:
        return "map_prop"
    return "other"


def upsert_bookmaker_event(snapshot: dict[str, Any]) -> int:
    """Create/update bookmaker event and linked upcoming match."""

    bookmaker_name = str(snapshot.get("bookmaker", "manual"))
    bookmaker_id = get_or_create_bookmaker(bookmaker_name, snapshot.get("base_url"))
    match_id = upsert_upcoming_match(
        bookmaker=bookmaker_name,
        raw_team_a=str(snapshot["raw_team_a"]),
        raw_team_b=str(snapshot["raw_team_b"]),
        match_start_time=snapshot.get("match_start_time"),
        league=snapshot.get("league_name") or snapshot.get("raw_league"),
        offer_url=snapshot.get("offer_url"),
        canonical_team_a=snapshot.get("mapped_team_a"),
        canonical_team_b=snapshot.get("mapped_team_b"),
        best_of=snapshot.get("best_of"),
    )
    canonical_match_id = get_upcoming_canonical_match_id(match_id)
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO bookmaker_events(
                bookmaker_id, bookmaker_event_id, canonical_match_id, raw_team_a, raw_team_b,
                match_start_time, sport_id, sport_name,
                category_id, category_name, league_id, league_name, offer_url, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(bookmaker_id, bookmaker_event_id) DO UPDATE SET
                canonical_match_id = excluded.canonical_match_id,
                raw_team_a = excluded.raw_team_a,
                raw_team_b = excluded.raw_team_b,
                match_start_time = excluded.match_start_time,
                sport_id = excluded.sport_id,
                sport_name = excluded.sport_name,
                category_id = excluded.category_id,
                category_name = excluded.category_name,
                league_id = excluded.league_id,
                league_name = excluded.league_name,
                offer_url = COALESCE(excluded.offer_url, bookmaker_events.offer_url),
                last_seen_at = CURRENT_TIMESTAMP
            """,
            (
                bookmaker_id,
                str(snapshot["bookmaker_event_id"]),
                canonical_match_id,
                snapshot["raw_team_a"],
                snapshot["raw_team_b"],
                snapshot.get("match_start_time"),
                snapshot.get("sport_id"),
                snapshot.get("sport_name"),
                snapshot.get("category_id"),
                snapshot.get("category_name"),
                snapshot.get("league_id"),
                snapshot.get("league_name"),
                snapshot.get("offer_url"),
            ),
        )
        row = connection.execute(
            "SELECT id FROM bookmaker_events WHERE bookmaker_id = ? AND bookmaker_event_id = ?",
            (bookmaker_id, str(snapshot["bookmaker_event_id"])),
        ).fetchone()
        return int(row["id"])


def get_upcoming_canonical_match_id(match_id: int) -> int | None:
    """Return canonical match ID linked to upcoming match."""

    with transaction() as connection:
        row = connection.execute("SELECT canonical_match_id FROM upcoming_matches WHERE id = ?", (match_id,)).fetchone()
        return int(row["canonical_match_id"]) if row and row["canonical_match_id"] is not None else None


def upsert_bookmaker_market(event_id: int, snapshot: dict[str, Any]) -> str:
    """Create/update bookmaker market for an event and return market_key."""

    market_name = str(snapshot.get("market_name") or snapshot.get("line_name") or "unknown")
    market_key = str(snapshot.get("market_key") or f"{snapshot.get('line_id') or 'unknown'}:{market_name}")
    
    # Get bookmaker_event_id from event_id
    with transaction() as connection:
        row = connection.execute(
            "SELECT bookmaker_event_id FROM bookmaker_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"bookmaker_event id={event_id} not found")
        bookmaker_event_id = str(row["bookmaker_event_id"])
        
        connection.execute(
            """
            INSERT INTO bookmaker_markets(
                bookmaker_event_id, bookmaker_market_key, market_name, market_type,
                line_id, line_name, is_extra_market
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bookmaker_event_id, bookmaker_market_key) DO UPDATE SET
                market_name = excluded.market_name,
                market_type = excluded.market_type,
                line_id = excluded.line_id,
                line_name = excluded.line_name,
                is_extra_market = excluded.is_extra_market
            """,
            (
                bookmaker_event_id,
                market_key,
                market_name,
                snapshot.get("market_type") or _market_type_from_name(market_name),
                snapshot.get("line_id"),
                snapshot.get("line_name") or market_name,
                int(bool(snapshot.get("is_extra_market", False))),
            ),
        )
        return market_key


def insert_outcome_snapshot(snapshot: dict[str, Any]) -> int:
    """Insert one atomic outcome odds snapshot into normalized odds tables."""

    bookmaker_name = str(snapshot.get("bookmaker", "manual"))
    bookmaker_id = get_or_create_bookmaker(bookmaker_name, snapshot.get("base_url"))
    event_id = upsert_bookmaker_event(snapshot)
    market_key = upsert_bookmaker_market(event_id, snapshot)

    # Get the string bookmaker_event_id from the integer event_id
    with transaction() as connection:
        row = connection.execute(
            "SELECT bookmaker_event_id FROM bookmaker_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        bookmaker_event_id = str(row["bookmaker_event_id"])

    raw_payload = snapshot.get("raw_payload")
    if raw_payload is not None and not isinstance(raw_payload, str):
        raw_payload = json.dumps(raw_payload, ensure_ascii=False)

    with transaction() as connection:
        cursor = connection.execute(
            """
            INSERT INTO odds_outcome_snapshots(
                bookmaker_event_id, bookmaker_market_key, scrape_run_id,
                scraped_at, source_url, offer_url, outcome_key, outcome_name, outcome_side,
                decimal_odds, raw_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bookmaker_event_id,
                market_key,
                snapshot.get("scrape_run_id"),
                str(snapshot.get("scraped_at") or utc_now_iso()),
                snapshot.get("source_url"),
                snapshot.get("offer_url"),
                str(snapshot["outcome_key"]),
                str(snapshot["outcome_name"]),
                snapshot.get("outcome_side"),
                float(snapshot["decimal_odds"]),
                raw_payload,
            ),
        )
        return int(cursor.lastrowid)


def insert_snapshot(snapshot: dict[str, Any]) -> int:
    """Insert either a legacy two-sided snapshot or normalized outcome snapshot."""

    if snapshot.get("snapshot_type") == "outcome" or "decimal_odds" in snapshot:
        return insert_outcome_snapshot(snapshot)
    return insert_odds_snapshot(snapshot)


def latest_outcome_odds(limit: int = 200) -> pd.DataFrame:
    """Return latest normalized outcome odds snapshots."""

    return query_df(
        """
        SELECT
            oos.*,
            b.name AS bookmaker,
            be.bookmaker_event_id,
            be.raw_team_a,
            be.raw_team_b,
            be.match_start_time,
            be.sport_name,
            be.category_name,
            be.league_name,
            be.offer_url AS event_offer_url,
            bm.market_name,
            bm.market_type,
            bm.line_id,
            bm.line_name
        FROM odds_outcome_snapshots oos
        JOIN bookmakers b ON b.id = oos.bookmaker_id
        JOIN bookmaker_events be ON be.id = oos.bookmaker_event_id
        JOIN bookmaker_markets bm ON bm.id = oos.bookmaker_market_id
        ORDER BY oos.scraped_at DESC, oos.id DESC
        LIMIT ?
        """,
        (limit,),
    )


def latest_odds(limit: int = 200) -> pd.DataFrame:
    """Return latest odds snapshots with bookmaker and match info."""

    return query_df(
        """
        SELECT os.*, b.name AS bookmaker, um.canonical_team_a, um.canonical_team_b, um.status AS match_status,
               um.offer_url AS match_offer_url
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        LEFT JOIN upcoming_matches um ON um.id = os.match_id
        ORDER BY os.scraped_at DESC, os.id DESC
        LIMIT ?
        """,
        (limit,),
    )


def export_snapshots(path: str | Path) -> Path:
    """Export all odds snapshots to CSV for audit/debug."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_odds(limit=1_000_000).to_csv(output, index=False)
    return output
