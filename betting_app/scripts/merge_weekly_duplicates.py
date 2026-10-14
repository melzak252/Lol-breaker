"""Merge weekly re-scrape duplicates in canonical_matches.

Same team pair appearing multiple times with 'expired' status across
different weeks (date gap >1 day). Keeps the row with:
1. Highest evidence (odds_snapshots + bookmaker_events count)
2. Has GOLGG mapping (tiebreaker)
3. Highest ID (final tiebreaker)

Run with --dry-run first to preview, then --yes to apply.

Usage:
    python -m betting_app.scripts.merge_weekly_duplicates --dry-run
    python -m betting_app.scripts.merge_weekly_duplicates --yes
"""

import argparse
import logging
from collections import defaultdict

from betting_app.core.db import connect, get_session, query_df
from sqlalchemy import text as _sql_text

logger = logging.getLogger(__name__)

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


def main():
    parser = argparse.ArgumentParser(description="Merge weekly re-scrape duplicates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.dry_run else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    # Find all team pair groups with multiple entries
    df = query_df("""
        SELECT id, normalized_team_a, normalized_team_b,
               start_time_normalized, team_a_name, team_b_name, league, status
        FROM canonical_matches
        WHERE status IN ('expired', 'finished')
          AND normalized_team_a IS NOT NULL
          AND normalized_team_b IS NOT NULL
        ORDER BY normalized_team_a, normalized_team_b, id
    """)
    if df.empty:
        logger.info("No canonical matches found.")
        return

    groups = defaultdict(list)
    for _, row in df.iterrows():
        teams = sorted([str(row["normalized_team_a"]).strip(), str(row["normalized_team_b"]).strip()])
        key = tuple(teams)
        groups[key].append(dict(row))

    total_remove = 0
    repoint_map = {}
    groups_found = 0

    for key, rows in groups.items():
        if len(rows) <= 1:
            continue

        # Check each has expired or finished status
        expired_rows = [r for r in rows if r.get("status") in ("expired", "finished")]
        if len(expired_rows) <= 1:
            continue

        groups_found += 1
        # Score: evidence, has mapping, highest ID
        scored = []
        for r in expired_rows:
            evidence = 0
            try:
                evidence_df = query_df(
                    f"SELECT COUNT(*) AS cnt FROM odds_snapshots WHERE canonical_match_id = {r['id']}"
                )
                evidence = int(evidence_df.iloc[0]["cnt"]) if not evidence_df.empty else 0
                be_df = query_df(
                    f"SELECT COUNT(*) AS cnt FROM bookmaker_events WHERE canonical_match_id = {r['id']}"
                )
                evidence += int(be_df.iloc[0]["cnt"]) if not be_df.empty else 0
            except Exception:
                pass
            has_map = False
            try:
                mdf = query_df(
                    f"SELECT id FROM golgg_match_mappings WHERE canonical_match_id = {r['id']} LIMIT 1"
                )
                has_map = not mdf.empty
            except Exception:
                pass
            scored.append((evidence, has_map, r["id"], r))

        # Best: most evidence, then has mapping, then highest ID
        scored.sort(key=lambda x: (-x[0], -x[1] if x[1] else False, -x[2]))
        keep = scored[0][3]

        for sc in scored[1:]:
            rem = sc[3]
            if rem["id"] in repoint_map:
                continue
            date1 = str(keep.get("start_time_normalized", "") or "")[:10]
            date2 = str(rem.get("start_time_normalized", "") or "")[:10]
            print(f"  CM {keep['id']} ({date1}) <- CM {rem['id']} ({date2})  {rem.get('team_a_name', '')[:15]} vs {rem.get('team_b_name', '')[:15]}  [{rem.get('league', '')}]")
            repoint_map[rem["id"]] = keep["id"]
            total_remove += 1

    print(f"\nFound {groups_found} groups, {total_remove} rows to remove")

    if args.dry_run:
        logger.info("DRY RUN — no changes made")
        return

    if total_remove == 0:
        logger.info("Nothing to do.")
        return

    # Apply: repoint FK, then delete
    for del_id, keep_id in repoint_map.items():
        _repoint_fk(del_id, keep_id)

    delete_ids = list(repoint_map.keys())
    batch_size = 50
    for i in range(0, len(delete_ids), batch_size):
        batch = delete_ids[i:i + batch_size]
        ids_str = ",".join(str(x) for x in batch)
        with get_session() as session:
            session.execute(_sql_text(f"DELETE FROM canonical_matches WHERE id IN ({ids_str})"))
            session.commit()
    logger.info(f"Deleted {total_remove} rows")
    logger.info("Done.")


if __name__ == "__main__":
    main()
