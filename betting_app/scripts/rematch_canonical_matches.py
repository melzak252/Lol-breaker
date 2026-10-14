"""Rebuild canonical cross-bookmaker match links for stored odds snapshots."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db, transaction
from betting_app.services.canonical_match_service import canonical_match_overview, resolve_canonical_match


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear canonical_matches and rebuild all links from upcoming_matches from scratch.",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Suppress the tabular canonical-match overview after rematching.",
    )
    args = parser.parse_args()

    init_db()
    if args.rebuild:
        reset_canonical_matches()
    updated = rematch_odds_snapshots()
    print(f"Rematched odds snapshots: {updated}")
    if not args.no_overview:
        overview = canonical_match_overview(limit=20)
        if not overview.empty:
            print(overview.to_string(index=False))


def rematch_odds_snapshots() -> int:
    """Resolve canonical_match_id for existing upcoming_matches and odds_snapshots."""

    # ── Phase 1: rematch upcoming_matches + odds_snapshots ────────────────
    with transaction() as connection:
        rows = connection.execute(
            """
            SELECT id, raw_team_a, raw_team_b, match_start_time, league
            FROM upcoming_matches
            WHERE canonical_match_id IS NULL
               OR NOT EXISTS (
                    SELECT 1
                    FROM canonical_matches
                    WHERE canonical_matches.id = upcoming_matches.canonical_match_id
               )
               OR EXISTS (
                    SELECT 1
                    FROM odds_snapshots
                    WHERE odds_snapshots.match_id = upcoming_matches.id
                      AND (
                            odds_snapshots.canonical_match_id IS NULL
                         OR odds_snapshots.canonical_match_id <> upcoming_matches.canonical_match_id
                      )
               )
            """
        ).fetchall()
    updated = 0
    for row in rows:
        canonical_match_id = resolve_canonical_match(
            raw_team_a=row["raw_team_a"],
            raw_team_b=row["raw_team_b"],
            match_start_time=row["match_start_time"],
            league=row["league"],
            source_system="bookmaker",
        )
        with transaction() as connection:
            connection.execute(
                "UPDATE upcoming_matches SET canonical_match_id = ? WHERE id = ?",
                (canonical_match_id, row["id"]),
            )
            removed = deduplicate_odds_for_canonical_update(
                connection,
                match_id=int(row["id"]),
                canonical_match_id=canonical_match_id,
            )
            if removed:
                print(
                    "Removed duplicate odds snapshots before canonical rematch: "
                    f"match_id={row['id']} canonical_match_id={canonical_match_id} count={removed}"
                )
            connection.execute(
                "UPDATE odds_snapshots SET canonical_match_id = ? WHERE match_id = ?",
                (canonical_match_id, row["id"]),
            )
        updated += 1

    # ── Phase 2: rematch bookmaker_events ─────────────────────────────────
    # bookmaker_events has no match_id column (dropped in migration); match
    # using its own raw_team_a/raw_team_b/match_start_time via the same
    # canonical resolution used for upcoming_matches.
    with transaction() as connection:
        events = connection.execute(
            """
            SELECT id, raw_team_a, raw_team_b, match_start_time, league_name
            FROM bookmaker_events
            WHERE canonical_match_id IS NULL
               OR canonical_match_id NOT IN (SELECT id FROM canonical_matches)
            """
        ).fetchall()
    events_updated = 0
    for event in events:
        canonical_match_id = resolve_canonical_match(
            raw_team_a=event["raw_team_a"],
            raw_team_b=event["raw_team_b"],
            match_start_time=event["match_start_time"],
            league=event.get("league_name"),
            source_system="bookmaker",
        )
        with transaction() as connection:
            connection.execute(
                "UPDATE bookmaker_events SET canonical_match_id = ? WHERE id = ?",
                (canonical_match_id, event["id"]),
            )
        events_updated += 1

    print(f"Rematched upcoming_matches: {updated}, bookmaker_events: {events_updated}")
    return updated + events_updated


def deduplicate_odds_for_canonical_update(connection, *, match_id: int, canonical_match_id: int) -> int:
    """Remove odds rows that would violate the canonical odds unique index.

    ``odds_snapshots`` has a partial unique index on
    ``(canonical_match_id, bookmaker_id, scraped_at)`` for ``match_winner``.
    During rematching, multiple scraper-specific ``upcoming_matches`` can be
    resolved to the same canonical match. If two such rows have the same
    bookmaker and ``scraped_at`` timestamp, the plain canonical-id update would
    fail with ``psycopg2.errors.UniqueViolation``. Keep the already-canonical
    row, and for duplicates within the same source match keep the highest id.
    """

    source_rows = connection.execute(
        """
        SELECT id, bookmaker_id, scraped_at
        FROM odds_snapshots
        WHERE match_id = ?
          AND market_type = 'match_winner'
        ORDER BY bookmaker_id, scraped_at, id DESC
        """,
        (match_id,),
    ).fetchall()

    duplicate_ids: set[int] = set()
    seen_source_keys: set[tuple[int, str | None]] = set()

    for row in source_rows:
        odds_id = int(row["id"])
        key = (int(row["bookmaker_id"]), row["scraped_at"])

        # Duplicates already present inside this source match. Rows are sorted
        # by id DESC, so the first row for a bookmaker/timestamp pair is kept.
        if key in seen_source_keys:
            duplicate_ids.add(odds_id)
            continue
        seen_source_keys.add(key)

        # Duplicates against rows that have already been attached to the target
        # canonical match. The production partial unique index starts with
        # canonical_match_id, so this is a cheap point lookup instead of a
        # table-wide join for every upcoming match.
        existing = connection.execute(
            """
            SELECT id
            FROM odds_snapshots
            WHERE canonical_match_id = ?
              AND bookmaker_id = ?
              AND scraped_at = ?
              AND market_type = 'match_winner'
              AND id <> ?
            LIMIT 1
            """,
            (canonical_match_id, key[0], key[1], odds_id),
        ).fetchone()
        if existing is not None:
            duplicate_ids.add(odds_id)

    for odds_id in duplicate_ids:
        connection.execute("DELETE FROM odds_snapshots WHERE id = ?", (odds_id,))

    return len(duplicate_ids)


def reset_canonical_matches() -> None:
    """Remove stale/duplicate canonical groups before a full rebuild."""

    with transaction() as connection:
        # Downstream objects reference canonical_matches.  They are rebuilt by
        # run_upcoming_prediction_pipeline immediately after rematching, so it
        # is safer to clear them than to keep predictions tied to stale IDs.
        connection.execute("DELETE FROM model_ev_signals")
        connection.execute("DELETE FROM canonical_predictions")
        connection.execute("DELETE FROM upcoming_match_features")
        connection.execute("UPDATE odds_snapshots SET canonical_match_id = NULL")
        connection.execute("UPDATE upcoming_matches SET canonical_match_id = NULL")
        connection.execute("UPDATE bookmaker_events SET canonical_match_id = NULL")
        connection.execute("DELETE FROM canonical_matches")


if __name__ == "__main__":
    main()
