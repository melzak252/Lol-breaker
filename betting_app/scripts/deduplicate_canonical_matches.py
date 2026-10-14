"""Deduplicate canonical_matches and auto-map expired matches to GOL.GG results.

This script performs two cleanup operations:

1. DEDUPLICATION
   Removes exact duplicate canonical_match rows (same normalized teams + same day),
   keeping the row with the best evidence (odds data, bookmaker events).
   Also handles close-date duplicates (same teams, dates ≤1 day apart) by merging.

2. BACKFILL MAPPING
   After cleanup, runs backfill_finished_expired_matches() to map expired
   canonical matches to existing GOL.GG results.

Usage:
    python -m betting_app.scripts.deduplicate_canonical_matches  [--dry-run] [--yes]
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from betting_app.core.db import connect, get_session, query_df
from sqlalchemy import text as _sql_text

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


FK_TABLES = [
    "bets",
    "bookmaker_events",
    "canonical_predictions",
    "golgg_match_mappings",
    "odds_snapshots",
    "upcoming_match_features",
    "upcoming_matches",
]


def _repoint_fk(old_id: int, new_id: int) -> None:
    """Update FK references from old_id to new_id. If a UNIQUE constraint
    blocks the UPDATE (e.g. upcoming_match_features has a composite unique key),
    DELETE the row instead — it's a duplicate of the one already owned by new_id.
    """
    # Process each table in its own session so a failure on one doesn't abort others
    for table in FK_TABLES:
        with get_session() as session:
            try:
                result = session.execute(_sql_text(
                    f"UPDATE {table} SET canonical_match_id = :new_id "
                    f"WHERE canonical_match_id = :old_id"
                ), {"new_id": new_id, "old_id": old_id})
                session.commit()
                if result.rowcount > 0:
                    logger.debug("  CM %d -> CM %d: repointed %d row(s) in %s",
                                 old_id, new_id, result.rowcount, table)
            except Exception as e:
                session.rollback()
                # If unique violation, DELETE the row instead (it's a duplicate)
                err_str = str(e)
                if "UniqueViolation" in err_str or "duplicate key" in err_str:
                    with get_session() as s2:
                        s2.execute(_sql_text(
                            f"DELETE FROM {table} WHERE canonical_match_id = :old_id"
                        ), {"old_id": old_id})
                        s2.commit()
                    logger.debug("  CM %d -> CM %d: deleted conflicting row(s) in %s",
                                 old_id, new_id, table)
                else:
                    logger.warning("  CM %d -> CM %d: could not repoint %s: %s",
                                   old_id, new_id, table, e)


def _get_evidence(canonical_match_id: int) -> int:
    """Count evidence (odds + bookmaker events) for a canonical match."""
    try:
        df = query_df(
            f"""
            SELECT
                COALESCE(os.odds_snapshots, 0) AS odds,
                COALESCE(be.bookmaker_events, 0) AS bookmaker
            FROM canonical_matches cm
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
            WHERE cm.id = {canonical_match_id}
            """
        )
        if not df.empty:
            return int(df.iloc[0]["odds"] or 0) + int(df.iloc[0]["bookmaker"] or 0)
    except Exception:
        pass
    return 0


# --------------------------------------------------------------------------- #
#  1. Remove exact same-day duplicates
# --------------------------------------------------------------------------- #

def remove_same_day_duplicates(dry_run: bool = False) -> dict:
    """Remove rows where same (norm_team_a, norm_team_b, date) appears > 1 time.

    Keeps the row with highest evidence, or if tie, the lowest ID.
    Before deleting, repoints FK references from removed rows to the kept row.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Removing same-day duplicates")
    logger.info("=" * 60)

    df = query_df(
        """
        SELECT id, normalized_team_a, normalized_team_b,
               SUBSTR(start_time_normalized, 1, 10) AS match_date,
               team_a_name, team_b_name, league, status
        FROM canonical_matches
        WHERE status IN ('expired', 'finished')
        ORDER BY id
        """
    )
    if df.empty:
        logger.info("  No canonical matches found.")
        return {"removed": 0, "kept": 0, "groups": 0}

    # Group by (normalized_team_a, normalized_team_b, date)
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row.get("normalized_team_a", "") or "").strip(),
               str(row.get("normalized_team_b", "") or "").strip(),
               str(row.get("match_date", "") or "").strip())
        if key[0] and key[1] and key[2]:
            groups[key].append(dict(row))

    total_removed = 0
    groups_processed = 0
    # Mapping: delete_id -> keep_id
    repoint_map: dict[int, int] = {}

    for key, rows in groups.items():
        if len(rows) <= 1:
            continue

        groups_processed += 1
        # Score each row by evidence, prefer higher, then lower ID
        scored = [(r, _get_evidence(r["id"])) for r in rows]
        scored.sort(key=lambda x: (-x[1], -x[0]["id"]))

        keep = scored[0][0]
        to_remove = [s[0] for s in scored[1:]]

        for rem in to_remove:
            logger.debug(
                "  Remove CM %d: '%s vs %s' (%s, %s) — keep CM %d (evidence %d > %d)",
                rem["id"], rem["team_a_name"], rem["team_b_name"],
                rem["match_date"], rem.get("league", ""),
                keep["id"], scored[0][1], _get_evidence(rem["id"]),
            )
            repoint_map[rem["id"]] = keep["id"]

        total_removed += len(to_remove)

    logger.info("  Found %d groups with duplicates, removing %d rows",
                groups_processed, total_removed)

    if dry_run:
        logger.info("  DRY RUN — would delete %d rows", total_removed)
    elif repoint_map:
        # 1. Repoint all FK refs to keep rows
        for del_id, keep_id in repoint_map.items():
            _repoint_fk(del_id, keep_id)

        # 2. Delete in batches
        delete_ids = list(repoint_map.keys())
        batch_size = 50
        for i in range(0, len(delete_ids), batch_size):
            batch = delete_ids[i:i + batch_size]
            ids_str = ",".join(str(x) for x in batch)
            with get_session() as session:
                session.execute(_sql_text(
                    f"DELETE FROM canonical_matches WHERE id IN ({ids_str})"
                ))
                session.commit()
        logger.info("  Deleted %d duplicate rows", total_removed)

    return {
        "removed": total_removed,
        "kept": len(df) - total_removed if not dry_run else -1,
        "groups": groups_processed,
    }


# --------------------------------------------------------------------------- #
#  2. Handle close-date duplicates (±1 day)
# --------------------------------------------------------------------------- #

def merge_close_date_duplicates(dry_run: bool = False) -> dict:
    """Merge rows where same team pair appears on consecutive days.

    Strategy: group by (norm_team_a, norm_team_b), sort by date.
    When dates are ≤1 day apart, keep the row with higher evidence,
    or if equal, the one with a GOLGG mapping. Delete others.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Merging close-date duplicates (±1 day)")
    logger.info("=" * 60)

    df = query_df(
        """
        SELECT id, normalized_team_a, normalized_team_b,
               SUBSTR(start_time_normalized, 1, 10) AS match_date,
               team_a_name, team_b_name, league, status
        FROM canonical_matches
        WHERE status IN ('expired', 'finished')
        ORDER BY normalized_team_a, normalized_team_b, start_time_normalized, id
        """
    )
    if df.empty:
        return {"removed": 0, "groups": 0}

    # Group by team pair
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row.get("normalized_team_a", "") or "").strip(),
               str(row.get("normalized_team_b", "") or "").strip())
        if key[0] and key[1]:
            groups[key].append(dict(row))

    total_removed = 0
    groups_processed = 0
    to_delete_ids: list[int] = []
    kept_ids: set[int] = set()

    for pair_key, rows in groups.items():
        if len(rows) < 2:
            continue

        # Sort by date, then ID
        rows.sort(key=lambda r: (str(r.get("match_date", "") or ""), r["id"]))

        # Walk through and find close-together groups
        i = 0
        while i < len(rows):
            j = i + 1
            close_group = [rows[i]]
            while j < len(rows):
                d1 = str(close_group[-1].get("match_date", "") or "")
                d2 = str(rows[j].get("match_date", "") or "")
                try:
                    dt1 = datetime.strptime(d1, "%Y-%m-%d")
                    dt2 = datetime.strptime(d2, "%Y-%m-%d")
                    if abs((dt2 - dt1).days) <= 1:
                        close_group.append(rows[j])
                        j += 1
                    else:
                        break
                except (ValueError, TypeError):
                    break

            if len(close_group) > 1:
                groups_processed += 1
                # Score: evidence, then has_mapping
                scored = []
                for r in close_group:
                    evidence = _get_evidence(r["id"])
                    has_mapping = False
                    try:
                        mdf = query_df(
                            f"SELECT id FROM golgg_match_mappings WHERE canonical_match_id = {r['id']} LIMIT 1"
                        )
                        has_mapping = not mdf.empty
                    except Exception:
                        pass
                    scored.append((evidence, has_mapping, -r["id"], r))

                # Best: most evidence, then has mapping, then lowest ID
                scored.sort(key=lambda x: (-x[0], -x[1] if x[1] else False, -x[2]))
                keep = scored[0][3]
                kept_ids.add(keep["id"])

                for sc in scored[1:]:
                    rem = sc[3]
                    if rem["id"] in kept_ids:
                        continue  # already kept by another group
                    logger.debug(
                        "  Merge CM %d '%s vs %s' (%s) -> keep CM %d (evidence %d, has_map=%s)",
                        rem["id"], rem.get("team_a_name", ""), rem.get("team_b_name", ""),
                        rem.get("match_date", ""),
                        keep["id"], scored[0][0], scored[0][1],
                    )
                    to_delete_ids.append(rem["id"])
                    total_removed += 1

            i = j

    logger.info("  Found %d close-date groups, removing %d rows", groups_processed, total_removed)

    if dry_run:
        logger.info("  DRY RUN — would delete %d rows", total_removed)
    elif to_delete_ids:
        # Build repoint map: delete_id -> keep_id
        # We need to map each delete ID to its keep row from the group scoring above
        # Re-derive from the pair_keys logic (same as above)
        repoint_map: dict[int, int] = {}
        for pair_key, rows in groups.items():
            if len(rows) < 2:
                continue
            rows.sort(key=lambda r: (str(r.get("match_date", "") or ""), r["id"]))
            i = 0
            while i < len(rows):
                j = i + 1
                close_group = [rows[i]]
                while j < len(rows):
                    d1 = str(close_group[-1].get("match_date", "") or "")
                    d2 = str(rows[j].get("match_date", "") or "")
                    try:
                        dt1 = datetime.strptime(d1, "%Y-%m-%d")
                        dt2 = datetime.strptime(d2, "%Y-%m-%d")
                        if abs((dt2 - dt1).days) <= 1:
                            close_group.append(rows[j])
                            j += 1
                        else:
                            break
                    except (ValueError, TypeError):
                        break
                if len(close_group) > 1:
                    scored = []
                    for r in close_group:
                        evidence = _get_evidence(r["id"])
                        has_mapping = False
                        try:
                            mdf = query_df(
                                f"SELECT id FROM golgg_match_mappings WHERE canonical_match_id = {r['id']} LIMIT 1"
                            )
                            has_mapping = not mdf.empty
                        except Exception:
                            pass
                        scored.append((evidence, has_mapping, -r["id"], r))
                    scored.sort(key=lambda x: (-x[0], -x[1] if x[1] else False, -x[2]))
                    keep = scored[0][3]
                    for sc in scored[1:]:
                        rem = sc[3]
                        if rem["id"] in repoint_map:
                            continue
                        repoint_map[rem["id"]] = keep["id"]
                i = j

        # Repoint FK references first
        for del_id, keep_id in repoint_map.items():
            _repoint_fk(del_id, keep_id)

        # Then delete
        batch_size = 50
        for i in range(0, len(to_delete_ids), batch_size):
            batch = to_delete_ids[i:i + batch_size]
            ids_str = ",".join(str(x) for x in batch)
            with get_session() as session:
                session.execute(_sql_text(
                    f"DELETE FROM canonical_matches WHERE id IN ({ids_str})"
                ))
                session.commit()
        logger.info("  Deleted %d close-date duplicate rows", total_removed)

    return {
        "removed": total_removed,
        "groups": groups_processed,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate canonical_matches and backfill GOL.GG mappings"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be done without making changes")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Detailed logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    if not args.yes and not args.dry_run:
        print("\n⚠️  This will DELETE rows from canonical_matches and create GOLGG mappings.")
        print("   Recommend: take a DB backup first.\n")
        resp = input("Continue? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # Phase 1
    r1 = remove_same_day_duplicates(dry_run=args.dry_run)
    logger.info("")

    # Phase 2
    r2 = merge_close_date_duplicates(dry_run=args.dry_run)
    logger.info("")

    # Phase 3: Backfill expired -> GOLGG
    logger.info("=" * 60)
    logger.info("PHASE 3: Backfill expired matches to GOL.GG")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("  SKIP (dry run)")
    else:
        # Run the existing backfill function
        from betting_app.scripts.refresh_golgg_direct import backfill_finished_expired_matches
        result = backfill_finished_expired_matches()
        for k, v in result.items():
            logger.info(f"  {k}: {v}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Same-day duplicates removed: {r1['removed']}")
    logger.info(f"  Close-date duplicates removed: {r2['removed']}")
    if not args.dry_run and not r1.get("kept", -1) == -1:
        logger.info(f"  Rows remaining after cleanup: ~{r1['kept'] - r2['removed']}")
    logger.info(f"  Dry run: {args.dry_run}")


if __name__ == "__main__":
    main()
