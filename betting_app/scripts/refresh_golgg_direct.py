"""Incrementally refresh finished GOL.GG matches directly into TimescaleDB.

Unlike refresh_golgg_results.py, this script skips the JSON cache entirely:
- Existing match IDs are read from the golgg_matches table
- Newly fetched match game data is written directly to the DB via import_golgg_batch()
- After import, newly added matches are auto-mapped to canonical_matches
  by date + team name proximity
- No golgg_matches.json file is read or written

Intended to be run periodically (every 6-12 hours) via the scheduler.
"""

from __future__ import annotations

import argparse
from collections import Counter
import asyncio
from datetime import datetime, timedelta, timezone
import logging

from betting_app.scrapers.golgg import GolggScraper

try:
    from tqdm.asyncio import tqdm
except ImportError:
    tqdm = None

# Reuse helpers from the JSON-based refresh script
from betting_app.scripts.refresh_golgg_results import (
    deduplicate_by_key,
    enrich_match_with_nested_game_metadata,
    expected_games_for_match,
    should_fetch_games,
    select_tournaments_to_fetch,
    get_tournament_matches,
    get_matches_with_games,
    CONCURRENCY_LIMIT as _REFRESH_CONCURRENCY,
)

# Import the DB writer
from betting_app.services.golgg_import_service import import_golgg_batch

# Import mapping for auto-match
from betting_app.services.mapping_service import suggest_mapping, load_golgg_team_candidates
from betting_app.services.canonical_match_service import competition_family

# -------------------------------------------------------------------- #
#  Auto-mapping: link imported GOL.GG results to canonical_matches     #
# -------------------------------------------------------------------- #

AUTO_MAP_MIN_IDENTITY_CONFIDENCE = 0.95


def result_mapping_review_reason(
    *,
    identity_confidence: float,
    canonical_date: str | None,
    golgg_date: str | None,
    canonical_competition: str | None,
    golgg_competition: str | None,
) -> str | None:
    """Return why a GOL.GG result link is ineligible for unattended mapping."""

    if identity_confidence < AUTO_MAP_MIN_IDENTITY_CONFIDENCE:
        return "identity_confidence"
    if not canonical_date or not golgg_date:
        return "missing_date"
    if canonical_date[:10] != golgg_date[:10]:
        return "date_mismatch"
    canonical_family = competition_family(canonical_competition)
    golgg_family = competition_family(golgg_competition)
    if not canonical_family or not golgg_family:
        return "unknown_competition"
    if canonical_family != golgg_family:
        return "competition_mismatch"
    return None

def auto_map_new_matches(match_ids: list[str], candidate_statuses: list[str] | None = None) -> dict:
    """Find canonical_matches that correspond to newly imported golgg matches.

    For each golgg match ID, query the DB for the match's team names + date,
    then search canonical_matches for a match with the same teams (via suggest_mapping)
    and a nearby start_time_normalized.

    Returns a dict with stats about what was mapped.
    """
    from betting_app.core.db import query_df, get_session
    from sqlalchemy import text as _sql_text

    if not match_ids:
        return {"checked": 0, "mapped": 0, "already_mapped": 0, "errors": 0}

    # Build WHERE clause for match IDs
    ids_str = ",".join(f"'{mid}'" for mid in match_ids)
    status_filter = ""
    if candidate_statuses:
        statuses = ",".join(f"'{str(status).replace("'", "''")}'" for status in candidate_statuses)
        status_filter = f"AND cm.status IN ({statuses})"
    
    # Fetch newly imported golgg matches with team names + date
    df = query_df(f"""
        SELECT match_id, team1_name, team2_name, date, tournament_name
        FROM golgg_matches
        WHERE match_id IN ({ids_str})
    """)
    
    if df.empty:
        return {
            "checked": 0,
            "mapped": 0,
            "already_mapped": 0,
            "review": 0,
            "review_reasons": {},
            "errors": 0,
        }
    
    mapped = 0
    already_mapped = 0
    errors = 0
    review = 0
    review_reasons: Counter[str] = Counter()
    checked = len(df)

    for _, row in df.iterrows():
        try:
            gmid = str(row["match_id"])
            team1 = str(row["team1_name"])
            team2 = str(row["team2_name"])
            match_date = str(row["date"]) if row.get("date") else ""
            tournament = str(row["tournament_name"]) if row.get("tournament_name") else ""

            # Check if already mapped
            existing = query_df(f"""
                SELECT id FROM golgg_match_mappings
                WHERE golgg_match_id = '{gmid}'
            """)
            if not existing.empty:
                already_mapped += 1
                continue

            from betting_app.core.matching import normalize_team_name as _ntn, similarity as _sim
            from betting_app.services.canonical_match_service import canonical_team_key as _ctk

            def _sql_escape(value: str) -> str:
                return value.replace("'", "''")

            def _team_keys(raw_team_name: str) -> set[str]:
                """Return normalized names/aliases that may identify a team.

                `suggest_mapping()` returns a display alias (for example "JD Gaming"),
                while `canonical_matches.normalized_team_*` may store either a short
                alias key ("jd") or a normalized display name ("top esports").  For
                auto-mapping we therefore consider both directions from team_aliases:
                exact alias -> normalized_name, and normalized_name -> alias.
                """

                keys = {_ntn(raw_team_name), _ctk(raw_team_name)}
                suggested, _, _ = suggest_mapping(raw_team_name)
                if suggested:
                    keys.add(_ntn(suggested))
                    keys.add(_ctk(suggested))

                raw_lower = raw_team_name.strip().lower()
                norm = _ntn(raw_team_name)
                suggested_lower = (suggested or "").strip().lower()
                alias_rows = query_df(
                    """
                    SELECT DISTINCT normalized_name, alias
                    FROM team_aliases
                    WHERE LOWER(alias) IN (?, ?)
                       OR LOWER(normalized_name) = ?
                    """,
                    (raw_lower, suggested_lower, norm),
                )
                for _, alias_row in alias_rows.iterrows():
                    normalized_name = str(alias_row.get("normalized_name") or "")
                    alias = str(alias_row.get("alias") or "")
                    if normalized_name:
                        keys.add(normalized_name.strip().lower())
                        keys.add(_ntn(normalized_name))
                        keys.add(_ctk(normalized_name))
                    if alias:
                        keys.add(_ntn(alias))
                        keys.add(_ctk(alias))
                return {key for key in keys if key}

            def _team_score(raw_team_name: str, canonical_normalized: str, keys: set[str]) -> float:
                canonical = (canonical_normalized or "").strip().lower()
                canonical_norm = _ntn(canonical)
                if not canonical:
                    return 0.0
                if canonical in keys or canonical_norm in keys:
                    return 1.0
                scores = [_sim(raw_team_name, canonical), _sim(raw_team_name, canonical_norm)]
                scores.extend(_sim(key, canonical) for key in keys)
                scores.extend(_sim(key, canonical_norm) for key in keys)
                return max(scores) if scores else 0.0

            start_date = match_date[:10] if len(match_date) >= 10 else match_date
            team1_keys = _team_keys(team1)
            team2_keys = _team_keys(team2)

            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                review += 1
                review_reasons["missing_date"] += 1
                continue

            candidates = query_df(f"""
                SELECT cm.id, cm.normalized_team_a, cm.normalized_team_b,
                       cm.start_time_normalized, cm.league, cm.status,
                       COALESCE(os.odds_snapshots, 0) AS odds_snapshots,
                       COALESCE(be.bookmaker_events, 0) AS bookmaker_events
                FROM canonical_matches cm
                LEFT JOIN golgg_match_mappings existing_gmm
                  ON existing_gmm.canonical_match_id = cm.id
                LEFT JOIN (
                    SELECT canonical_match_id, COUNT(*) AS odds_snapshots
                    FROM odds_snapshots
                    GROUP BY canonical_match_id
                ) os ON os.canonical_match_id = cm.id
                LEFT JOIN (
                    SELECT canonical_match_id, COUNT(*) AS bookmaker_events
                    FROM bookmaker_events
                    GROUP BY canonical_match_id
                ) be ON be.canonical_match_id = cm.id
                WHERE SUBSTR(start_time_normalized, 1, 10) = '{_sql_escape(start_date)}'
                  AND existing_gmm.canonical_match_id IS NULL
                  {status_filter}
            """)

            if candidates.empty:
                review += 1
                review_reasons["no_exact_date_candidate"] += 1
                continue

            eligible: list[tuple[int, float]] = []
            candidate_reasons: Counter[str] = Counter()
            for _, c in candidates.iterrows():
                ca = str(c.get("normalized_team_a") or "")
                cb = str(c.get("normalized_team_b") or "")
                cdate = str(c.get("start_time_normalized") or "")[:10]

                a1 = _team_score(team1, ca, team1_keys)
                b1 = _team_score(team2, cb, team2_keys)
                a2 = _team_score(team1, cb, team1_keys)
                b2 = _team_score(team2, ca, team2_keys)
                direct_pair = (a1, b1)
                swapped_pair = (a2, b2)
                selected_pair = (
                    direct_pair
                    if sum(direct_pair) >= sum(swapped_pair)
                    else swapped_pair
                )
                identity_confidence = min(selected_pair)
                reason = result_mapping_review_reason(
                    identity_confidence=identity_confidence,
                    canonical_date=cdate,
                    golgg_date=start_date,
                    canonical_competition=str(c.get("league") or ""),
                    golgg_competition=tournament,
                )
                if reason:
                    candidate_reasons[reason] += 1
                    continue
                eligible.append((int(c["id"]), identity_confidence))

            if len(eligible) != 1:
                review += 1
                reason = "ambiguous_candidates" if len(eligible) > 1 else (
                    candidate_reasons.most_common(1)[0][0]
                    if candidate_reasons
                    else "no_identity_candidate"
                )
                review_reasons[reason] += 1
                logger.warning(
                    "GOL.GG match %s routed to review: %s (%s vs %s)",
                    gmid,
                    reason,
                    team1,
                    team2,
                )
                continue

            best_id, best_score = eligible[0]
            with get_session() as session:
                session.execute(
                    _sql_text(
                        """
                        INSERT INTO golgg_match_mappings
                            (canonical_match_id, golgg_match_id, confidence, mapped_by)
                        VALUES (:canonical_id, :golgg_id, :confidence, 'auto-contained-v1')
                        ON CONFLICT (golgg_match_id) DO NOTHING
                        """
                    ),
                    {
                        "canonical_id": best_id,
                        "golgg_id": gmid,
                        "confidence": best_score,
                    },
                )
                session.commit()
            mapped += 1

        except Exception as exc:
            logger.warning(f"Auto-map error for match_id {row.get('match_id')}: {exc}")
            errors += 1

    return {
        "checked": checked,
        "mapped": mapped,
        "already_mapped": already_mapped,
        "review": review,
        "review_reasons": dict(sorted(review_reasons.items())),
        "errors": errors,
    }


def mark_mapped_matches_finished(match_ids: list[str] | None = None) -> dict[str, int]:
    """Mark canonical matches as finished when a mapped GOL.GG result exists.

    GOL.GG is our source of truth for completed results.  Once a GOL.GG match is
    linked to a `canonical_matches` row, update the canonical row from
    `upcoming`/`expired` (or any non-finished status) to `finished` and persist
    the winner/loser names so downstream code can settle/evaluate predictions.
    """
    from betting_app.core.db import get_session, query_df
    from betting_app.core.matching import normalize_team_name, similarity
    from sqlalchemy import text as _sql_text

    def _sql_escape(value: str) -> str:
        return value.replace("'", "''")

    # Existing deployed DBs predate result columns, so keep this migration
    # idempotent and cheap.  init.sql/model are also updated for fresh DBs.
    # (Removed ALTER TABLE statements as they are not SQLite compatible and columns were added manually)

    where_ids = ""
    if match_ids:
        ids_str = ",".join(f"'{_sql_escape(str(mid))}'" for mid in match_ids if str(mid).strip())
        if ids_str:
            where_ids = f"AND gm.match_id IN ({ids_str})"

    df = query_df(f"""
        SELECT
            cm.id AS canonical_match_id,
            cm.normalized_team_a,
            cm.normalized_team_b,
            gm.match_id AS golgg_match_id,
            gm.team1_name,
            gm.team2_name,
            gm.team1_win,
            gm.team2_win,
            gm.draw,
            gm.winner_name,
            gm.loser_name,
            gm.best_of
        FROM golgg_match_mappings gmm
        JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
        JOIN canonical_matches cm ON cm.id = gmm.canonical_match_id
        WHERE COALESCE(gm.draw, 0) = 0
          AND (COALESCE(gm.team1_win, 0) = 1 OR COALESCE(gm.team2_win, 0) = 1 OR gm.winner_name IS NOT NULL)
          {where_ids}
    """)

    checked = len(df)
    updated = 0
    skipped = 0
    errors = 0

    for _, row in df.iterrows():
        try:
            team1_win = int(row.get("team1_win") or 0) == 1
            team2_win = int(row.get("team2_win") or 0) == 1
            team1 = str(row.get("team1_name") or "")
            team2 = str(row.get("team2_name") or "")
            winner = str(row.get("winner_name") or "").strip()
            loser = str(row.get("loser_name") or "").strip()
            best_of = row.get("best_of")

            if not winner:
                winner = team1 if team1_win else team2 if team2_win else ""
            if not loser:
                loser = team2 if team1_win else team1 if team2_win else ""
            if not winner:
                skipped += 1
                continue

            winner_norm = normalize_team_name(winner)
            ca = str(row.get("normalized_team_a") or "")
            cb = str(row.get("normalized_team_b") or "")
            score_a = max(similarity(winner, ca), similarity(winner_norm, ca)) if ca else 0.0
            score_b = max(similarity(winner, cb), similarity(winner_norm, cb)) if cb else 0.0
            winner_side = "team_a" if score_a >= score_b else "team_b"

            with get_session() as session:
                result = session.execute(
                    _sql_text(
                        """
                        UPDATE canonical_matches
                        SET status = 'finished',
                            winner_name = :winner_name,
                            loser_name = :loser_name,
                            winner_normalized = :winner_normalized,
                            winner_side = :winner_side,
                            best_of = COALESCE(:best_of, best_of),
                            result_source = 'golgg',
                            result_source_match_id = :golgg_match_id,
                            result_recorded_at = NOW()
                        WHERE id = :canonical_match_id
                          AND (
                              status IS DISTINCT FROM 'finished'
                              OR winner_name IS DISTINCT FROM :winner_name
                              OR loser_name IS DISTINCT FROM :loser_name
                              OR winner_normalized IS DISTINCT FROM :winner_normalized
                              OR winner_side IS DISTINCT FROM :winner_side
                              OR best_of IS DISTINCT FROM COALESCE(:best_of, best_of)
                              OR result_source_match_id IS DISTINCT FROM :golgg_match_id
                          )
                        """
                    ),
                    {
                        "canonical_match_id": int(row["canonical_match_id"]),
                        "golgg_match_id": str(row["golgg_match_id"]),
                        "winner_name": winner,
                        "loser_name": loser or None,
                        "winner_normalized": winner_norm,
                        "winner_side": winner_side,
                        "best_of": int(best_of) if best_of is not None else None,
                    },
                )
                session.commit()
                if result.rowcount:
                    updated += int(result.rowcount)
                else:
                    skipped += 1
        except Exception as exc:
            logger.warning("Auto-finish error for match_id %s: %s", row.get("golgg_match_id"), exc)
            errors += 1

    return {"checked": checked, "updated": updated, "skipped": skipped, "errors": errors}


def backfill_finished_expired_matches(limit: int | None = None) -> dict[str, int]:
    """Try to resolve expired canonical matches using existing GOL.GG results.

    `auto_map_new_matches()` normally runs only for newly imported GOL.GG IDs.
    This backfill finds completed, not-yet-mapped GOL.GG matches in the date range
    covered by currently expired canonical matches, maps them with the same fuzzy
    logic, and then marks mapped canonical rows as `finished` with winners.
    """
    from betting_app.core.db import query_df

    window = query_df(
        """
        SELECT
            MIN(SUBSTR(start_time_normalized, 1, 10)) AS min_day,
            MAX(SUBSTR(start_time_normalized, 1, 10)) AS max_day,
            COUNT(*) AS expired_count
        FROM canonical_matches
        WHERE status = 'expired'
          AND NOT EXISTS (
              SELECT 1
              FROM golgg_match_mappings gmm
              WHERE gmm.canonical_match_id = canonical_matches.id
          )
        """
    )
    if window.empty or int(window.iloc[0].get("expired_count") or 0) == 0:
        finish_stats = mark_mapped_matches_finished(None)
        return {
            "expired_candidates": 0,
            "golgg_candidates": 0,
            "mapped": 0,
            "already_mapped": 0,
            "unmatched": 0,
            "finished_updated": finish_stats["updated"],
            "finished_checked": finish_stats["checked"],
            "finished_errors": finish_stats["errors"],
        }

    min_day = str(window.iloc[0]["min_day"])
    max_day = str(window.iloc[0]["max_day"])
    expired_count = int(window.iloc[0]["expired_count"] or 0)
    start_day = (datetime.strptime(min_day, "%Y-%m-%d") - timedelta(days=2)).strftime("%Y-%m-%d")
    end_day = (datetime.strptime(max_day, "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d")
    limit_sql = f"LIMIT {int(limit)}" if limit and int(limit) > 0 else ""

    golgg = query_df(
        f"""
        SELECT gm.match_id, gm.date
        FROM golgg_matches gm
        LEFT JOIN golgg_match_mappings gmm ON gmm.golgg_match_id = gm.match_id
        WHERE gmm.golgg_match_id IS NULL
          AND gm.date >= '{start_day}'
          AND gm.date <= '{end_day}'
          AND COALESCE(gm.draw, 0) = 0
          AND (
              COALESCE(gm.team1_win, 0) = 1
              OR COALESCE(gm.team2_win, 0) = 1
              OR gm.winner_name IS NOT NULL
          )
        ORDER BY gm.date, gm.match_id
        {limit_sql}
        """
    )

    match_ids = [str(row["match_id"]) for _, row in golgg.iterrows() if row.get("match_id")]
    map_stats = auto_map_new_matches(match_ids, candidate_statuses=["expired"]) if match_ids else {
        "checked": 0,
        "mapped": 0,
        "already_mapped": 0,
        "errors": 0,
    }
    finish_stats = mark_mapped_matches_finished(match_ids if match_ids else None)
    return {
        "expired_candidates": expired_count,
        "golgg_candidates": len(match_ids),
        "mapped": map_stats["mapped"],
        "already_mapped": map_stats["already_mapped"],
        "unmatched": map_stats["errors"],
        "finished_updated": finish_stats["updated"],
        "finished_checked": finish_stats["checked"],
        "finished_errors": finish_stats["errors"],
    }


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh GOL.GG finished matches directly into TimescaleDB"
    )
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--refresh-matches",
        action="store_true",
        help="Fetch match lists for all tournaments instead of only tournaments with missing games.",
    )
    parser.add_argument(
        "--refresh-existing-days",
        type=int,
        default=0,
        help=(
            "Upsert metadata for existing DB matches whose GOL.GG date is within this many days. "
            "Use with --refresh-matches in the scheduler so in-progress series are corrected after finishing."
        ),
    )
    parser.add_argument(
        "--include-incomplete-existing",
        action="store_true",
        help="Also fetch games for existing matches whose nested games are incomplete.",
    )
    parser.add_argument(
        "--refetch-games",
        action="store_true",
        help="Refetch games for selected matches even if they already look complete.",
    )
    parser.add_argument("--match-id", help="Fetch/refetch only one match ID.")
    parser.add_argument("--dry-run", action="store_true", help="Discover missing matches but do not write to DB.")
    parser.add_argument(
        "--backfill-finished",
        action="store_true",
        help="Map expired canonical matches to existing GOL.GG results and mark them finished, then exit.",
    )
    parser.add_argument(
        "--backfill-limit",
        type=int,
        default=None,
        help="Optional limit for GOL.GG candidate rows processed by --backfill-finished.",
    )
    return parser.parse_args()


def load_db_match_ids() -> set[str]:
    """Return set of match_id strings already in golgg_matches."""
    from betting_app.core.db import query_df

    df = query_df("SELECT match_id FROM golgg_matches WHERE match_id IS NOT NULL")
    if df.empty:
        return set()
    return {str(row["match_id"]) for _, row in df.iterrows()}


def load_db_games_per_tournament() -> dict[str, int]:
    """Return dict of tournament_name -> nested game count from DB."""
    from betting_app.core.db import query_df

    df = query_df(
        """
        SELECT gm.tournament_name, COUNT(gg.game_id) AS game_count
        FROM golgg_matches gm
        LEFT JOIN golgg_games gg ON gg.match_id = gm.match_id
        GROUP BY gm.tournament_name
        """
    )
    if df.empty:
        return {}
    return {str(row["tournament_name"]): int(row["game_count"]) for _, row in df.iterrows()}


def load_db_games_per_match() -> dict[str, int]:
    """Return dict of match_id -> nested game count from DB."""
    from betting_app.core.db import query_df

    df = query_df(
        """
        SELECT gm.match_id, COUNT(gg.game_id) AS game_count
        FROM golgg_matches gm
        LEFT JOIN golgg_games gg ON gg.match_id = gm.match_id
        GROUP BY gm.match_id
        """
    )
    if df.empty:
        return {}
    return {str(row["match_id"]): int(row["game_count"]) for _, row in df.iterrows()}


def _match_date(match: dict) -> datetime | None:
    raw_date = match.get("date")
    if not raw_date:
        return None
    try:
        return datetime.fromisoformat(str(raw_date)).replace(tzinfo=None)
    except ValueError:
        return None


def select_recent_existing_matches(
    discovered_matches: list[dict],
    known_existing_ids: set[str],
    *,
    days: int,
    now: datetime | None = None,
) -> list[dict]:
    """Return existing DB matches whose live GOL.GG metadata should be refreshed.

    This is intentionally separate from nested-game fetching: tournament match-list rows
    can change while a series is in progress (for example 1-1 -> 3-2), and those
    updates must be upserted even when the match_id already exists in the DB.
    """

    if days <= 0:
        return []

    cutoff = (now or datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=days)
    selected: list[dict] = []
    for match in deduplicate_by_key(discovered_matches, "match_id"):
        match_id = match.get("match_id")
        if not match_id or str(match_id) not in known_existing_ids:
            continue
        parsed_date = _match_date(match)
        if parsed_date is not None and parsed_date >= cutoff:
            selected.append(match)
    return selected


async def fetch_and_save_games(
    scraper: GolggScraper,
    selected_matches: list[dict],
    *,
    batch_size: int,
    concurrency: int,
) -> int:
    """Fetch games for selected matches and write each batch directly to DB.

    Returns the total number of matches whose games were successfully written.
    """
    fetched_total = 0
    for start in range(0, len(selected_matches), batch_size):
        batch = selected_matches[start : start + batch_size]
        print(
            f"Fetching batch {start // batch_size + 1}: "
            f"{start + 1}-{start + len(batch)} / {len(selected_matches)}"
        )
        fetched_match_docs = await get_matches_with_games(batch, scraper, concurrency=concurrency)
        fetched_match_docs = [
            m for m in fetched_match_docs if m.get("match_id") and m.get("games")
        ]
        if not fetched_match_docs:
            print("  No game data fetched in this batch.")
            continue

        # Write directly to DB
        try:
            stats = import_golgg_batch(fetched_match_docs)
            fetched_total += len(fetched_match_docs)
            print(
                f"  DB insert OK: {stats['matches']} matches, "
                f"{stats['games']} games, {stats['players']} players"
            )
        except Exception as exc:
            print(f"  DB insert FAILED for batch: {exc}")
            # Re-raise so caller knows something went wrong
            raise

    return fetched_total


async def main() -> None:
    args = parse_args()

    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting GOL.GG direct refresh...")

    if args.backfill_finished:
        print("Backfilling expired canonical matches from existing GOL.GG results...")
        stats = backfill_finished_expired_matches(limit=args.backfill_limit)
        print(
            "Backfill result: "
            f"expired_candidates={stats['expired_candidates']}, "
            f"golgg_candidates={stats['golgg_candidates']}, "
            f"mapped={stats['mapped']}, "
            f"already_mapped={stats['already_mapped']}, "
            f"unmatched={stats['unmatched']}, "
            f"finished_checked={stats['finished_checked']}, "
            f"finished_updated={stats['finished_updated']}, "
            f"finished_errors={stats['finished_errors']}"
        )
        return

    # ------------------------------------------------------------------ #
    # 1. Load existing state from DB
    # ------------------------------------------------------------------ #
    print("Loading existing GOL.GG match IDs from TimescaleDB...")
    known_existing_ids = load_db_match_ids()
    games_per_tournament = load_db_games_per_tournament()
    games_per_match = load_db_games_per_match()
    total_games = sum(games_per_match.values())
    print(f"DB state: {len(known_existing_ids)} matches, {total_games} nested games")

    # ------------------------------------------------------------------ #
    # 2. Fetch tournament index and discover new matches
    # ------------------------------------------------------------------ #
    async with GolggScraper(max_pages=args.max_pages) as scraper:
        print("Fetching GOL.GG tournament index...")
        all_tournaments = await scraper.get_all_tournaments()
        tournaments_to_fetch = select_tournaments_to_fetch(
            all_tournaments,
            games_per_tournament,
            refresh_matches=args.refresh_matches,
        )
        print(
            f"Tournament lists to scan: {len(tournaments_to_fetch)} / {len(all_tournaments)} "
            "(only tournaments with missing/new games unless --refresh-matches)"
        )

        discovered_matches = (
            await get_tournament_matches(scraper, tournaments_to_fetch)
            if tournaments_to_fetch
            else []
        )
        new_matches = [
            match
            for match in deduplicate_by_key(discovered_matches, "match_id")
            if match.get("match_id") and str(match["match_id"]) not in known_existing_ids
        ]
        if args.match_id:
            new_matches = [
                m for m in new_matches
                if str(m.get("match_id")) == str(args.match_id)
            ]

        print(f"Discovered new match IDs: {len(new_matches)}")

        refreshed_existing_ids: list[str] = []
        recent_existing_matches = select_recent_existing_matches(
            discovered_matches,
            known_existing_ids,
            days=args.refresh_existing_days,
        )
        if recent_existing_matches:
            print(
                f"Refreshing metadata for {len(recent_existing_matches)} existing matches "
                f"from the last {args.refresh_existing_days} days."
            )
            if not args.dry_run:
                refresh_stats = import_golgg_batch(recent_existing_matches)
                refreshed_existing_ids = [
                    str(m["match_id"]) for m in recent_existing_matches if m.get("match_id")
                ]
                print(
                    f"Existing metadata refresh OK: {refresh_stats['matches']} matches, "
                    f"{refresh_stats['games']} games, {refresh_stats['players']} players"
                )

        if args.dry_run:
            print("Dry run: would fetch games for the following new matches:")
            for m in new_matches:
                print(f"  {m.get('match_id')}: {m.get('sname_t1')} vs {m.get('sname_t2')} ({m.get('date')})")

        # ------------------------------------------------------------------ #
        # 3. Build the list of matches that need game data
        # ------------------------------------------------------------------ #
        # For the first run, we don't have match objects in memory (they're in DB).
        # We only process newly discovered matches (or refetches).
        # Build minimal match stubs from discovered_matches for the DB-fetched ones.
        selected_matches = [
            match
            for match in deduplicate_by_key(discovered_matches, "match_id")
            if should_fetch_games(
                match,
                known_existing_ids,
                games_per_match,
                include_incomplete_existing=args.include_incomplete_existing,
                refetch_games=args.refetch_games,
                match_id=args.match_id,
            )
        ]
        if args.dry_run:
            # Only consider newly discovered matches for dry run
            selected_new_ids = {
                str(m["match_id"]) for m in new_matches if m.get("match_id")
            }
            selected_matches = [
                m for m in selected_matches
                if str(m.get("match_id")) in selected_new_ids
            ]

        if not selected_matches:
            if refreshed_existing_ids:
                finish_stats = mark_mapped_matches_finished(refreshed_existing_ids)
                print(
                    f"Auto-finish refreshed existing result: checked={finish_stats['checked']}, "
                    f"updated={finish_stats['updated']}, skipped={finish_stats['skipped']}, "
                    f"errors={finish_stats['errors']}"
                )
            print("No matches need game data. All up to date.")
            return

        print(
            f"Fetching nested games for {len(selected_matches)} matches "
            "(new only by default; existing incomplete only with --include-incomplete-existing)."
        )
        if args.dry_run:
            print("Dry run: stopping before downloading/saving nested games.")
            return

        # ------------------------------------------------------------------ #
        # 4. Fetch games and write directly to DB in batches
        # ------------------------------------------------------------------ #
        fetched_total = await fetch_and_save_games(
            scraper,
            selected_matches,
            batch_size=max(1, args.batch_size),
            concurrency=args.concurrency,
        )

        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"Finished. Wrote game data for {fetched_total} matches."
        )

        # ------------------------------------------------------------------ #
        # 5. Auto-map newly imported matches to canonical_matches
        # ------------------------------------------------------------------ #
        print("Auto-mapping newly imported GOL.GG matches to canonical_matches...")
        new_match_ids = [
            str(m["match_id"]) for m in selected_matches
            if m.get("match_id")
        ]
        if new_match_ids:
            all_finish_ids = sorted(set(new_match_ids + refreshed_existing_ids))
            map_stats = auto_map_new_matches(new_match_ids)
            print(
                f"Auto-mapping result: checked={map_stats['checked']}, "
                f"mapped={map_stats['mapped']}, "
                f"already_mapped={map_stats['already_mapped']}, "
                f"unmatched={map_stats['errors']}"
            )

            finish_stats = mark_mapped_matches_finished(all_finish_ids)
            print(
                f"Auto-finish result: checked={finish_stats['checked']}, "
                f"updated={finish_stats['updated']}, "
                f"skipped={finish_stats['skipped']}, "
                f"errors={finish_stats['errors']}"
            )

        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"GOL.GG direct refresh complete."
        )


if __name__ == "__main__":
    asyncio.run(main())
