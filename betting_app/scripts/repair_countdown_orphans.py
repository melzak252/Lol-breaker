"""Repair canonical rows created by countdown/orphan bookmaker labels.

This is intentionally conservative and aimed at preserving odds history used
for CLV analysis:

* remap known countdown duplicate canonical rows into the real canonical match;
* correct canonical start times when a row has odds history but no GOL.GG result;
* delete only canonical rows that have no dependent history/evidence.

Dry-run by default; pass ``--execute`` to apply.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text

from betting_app.core.db import get_session, query_df


@dataclass(frozen=True)
class MergePlan:
    duplicate_id: int
    keeper_id: int
    reason: str


@dataclass(frozen=True)
class TimeFixPlan:
    canonical_id: int
    start_time_normalized: str
    reason: str


# Verified on production data: canonical 197 is a Betfan countdown duplicate of
# canonical 20 (same CCG/Conviction match). Keep all odds/upcoming history by
# remapping to the real row instead of deleting it.
MERGE_PLANS = [MergePlan(duplicate_id=197, keeper_id=20, reason="Betfan countdown duplicate of CCG vs Conviction")]

# Canonical 257 has real odds history and a Totalbet row with the correct start;
# there is no GOL.GG result in DB, so keep the row and fix the bad countdown time.
TIME_FIX_PLANS = [
    TimeFixPlan(
        canonical_id=257,
        start_time_normalized="2026-05-31T10:00:00+00:00",
        reason="Meavedron vs Esuba: correct start from Totalbet, preserve odds history",
    )
]

# Verified orphan rows: no odds_snapshots, upcoming_matches, bookmaker_events,
# GOL.GG mappings, predictions/features/signals/bets.
ORPHAN_IDS = [1, 57, 2, 58]


DEPENDENCY_TABLES = [
    "golgg_match_mappings",
    "upcoming_matches",
    "bookmaker_events",
    "odds_snapshots",
    "canonical_predictions",
    "upcoming_match_features",
    "model_ev_signals",
    "bets",
]


def _counts_for_ids(ids: list[int]) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {id_: {} for id_ in ids}
    if not ids:
        return counts
    id_list = ",".join(str(int(id_)) for id_ in ids)
    for table in DEPENDENCY_TABLES:
        df = query_df(
            f"""
            SELECT canonical_match_id, COUNT(*) AS cnt
            FROM {table}
            WHERE canonical_match_id IN ({id_list})
            GROUP BY canonical_match_id
            """
        )
        for _, row in df.iterrows():
            counts[int(row["canonical_match_id"])][table] = int(row["cnt"] or 0)
    for per_id in counts.values():
        for table in DEPENDENCY_TABLES:
            per_id.setdefault(table, 0)
    return counts


def _canonical_rows(ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    id_list = ",".join(str(int(id_)) for id_ in ids)
    df = query_df(
        f"""
        SELECT id, team_a_name, team_b_name, start_time_normalized, league, status
        FROM canonical_matches
        WHERE id IN ({id_list})
        ORDER BY id
        """
    )
    return df.to_dict("records") if not df.empty else []


def _remap_dependencies(session, *, keeper_id: int, duplicate_id: int) -> None:
    # Avoid unique conflicts before FK remaps.
    session.execute(
        sql_text(
            """
            DELETE FROM upcoming_match_features dup
            USING upcoming_match_features keep
            WHERE dup.canonical_match_id = :duplicate_id
              AND keep.canonical_match_id = :keeper_id
              AND keep.feature_version = dup.feature_version
              AND keep.ratings_version = dup.ratings_version
            """
        ),
        {"keeper_id": keeper_id, "duplicate_id": duplicate_id},
    )
    session.execute(
        sql_text(
            """
            DELETE FROM odds_snapshots dup
            USING odds_snapshots keep
            WHERE dup.canonical_match_id = :duplicate_id
              AND keep.canonical_match_id = :keeper_id
              AND keep.bookmaker_id = dup.bookmaker_id
              AND keep.scraped_at = dup.scraped_at
              AND COALESCE(keep.market_type, 'match_winner') = COALESCE(dup.market_type, 'match_winner')
              AND COALESCE(dup.market_type, 'match_winner') = 'match_winner'
            """
        ),
        {"keeper_id": keeper_id, "duplicate_id": duplicate_id},
    )

    for table in [
        "upcoming_matches",
        "bookmaker_events",
        "odds_snapshots",
        "canonical_predictions",
        "upcoming_match_features",
        "model_ev_signals",
        "bets",
    ]:
        session.execute(
            sql_text(f"UPDATE {table} SET canonical_match_id = :keeper_id WHERE canonical_match_id = :duplicate_id"),
            {"keeper_id": keeper_id, "duplicate_id": duplicate_id},
        )

    session.execute(
        sql_text(
            """
            UPDATE golgg_match_mappings
            SET canonical_match_id = :keeper_id
            WHERE canonical_match_id = :duplicate_id
              AND NOT EXISTS (
                  SELECT 1 FROM golgg_match_mappings existing
                  WHERE existing.canonical_match_id = :keeper_id
              )
            """
        ),
        {"keeper_id": keeper_id, "duplicate_id": duplicate_id},
    )


def repair_countdown_orphans(*, execute: bool = False) -> dict[str, int]:
    ids = sorted({*ORPHAN_IDS, *[p.duplicate_id for p in MERGE_PLANS], *[p.keeper_id for p in MERGE_PLANS], *[p.canonical_id for p in TIME_FIX_PLANS]})
    print("Canonical rows:")
    for row in _canonical_rows(ids):
        print(
            f"  id={row['id']} {row['status']} {row['start_time_normalized']} "
            f"{row['team_a_name']} vs {row['team_b_name']} ({row['league']})"
        )
    counts = _counts_for_ids(ids)
    print("Dependency counts:")
    for id_ in ids:
        print(f"  id={id_} {counts.get(id_, {})}")

    planned_merges = 0
    planned_time_fixes = 0
    planned_deletes = 0

    for plan in MERGE_PLANS:
        planned_merges += 1
        print(f"{'MERGE' if execute else 'DRY MERGE'} duplicate={plan.duplicate_id} -> keeper={plan.keeper_id}: {plan.reason}")

    for plan in TIME_FIX_PLANS:
        planned_time_fixes += 1
        print(f"{'FIX' if execute else 'DRY FIX'} id={plan.canonical_id} start={plan.start_time_normalized}: {plan.reason}")

    safe_orphans: list[int] = []
    for id_ in ORPHAN_IDS:
        total = sum(counts.get(id_, {}).values())
        if total == 0:
            safe_orphans.append(id_)
            planned_deletes += 1
            print(f"{'DELETE' if execute else 'DRY DELETE'} orphan id={id_}")
        else:
            print(f"SKIP orphan id={id_}: has dependencies {counts.get(id_, {})}")

    if execute:
        with get_session() as session:
            for plan in MERGE_PLANS:
                _remap_dependencies(session, keeper_id=plan.keeper_id, duplicate_id=plan.duplicate_id)
                session.execute(sql_text("DELETE FROM canonical_matches WHERE id = :id"), {"id": plan.duplicate_id})
            for plan in TIME_FIX_PLANS:
                session.execute(
                    sql_text(
                        """
                        UPDATE canonical_matches
                        SET start_time_normalized = :start_time,
                            canonical_key = canonical_key || '|timefix-' || :id
                        WHERE id = :id
                          AND start_time_normalized IS DISTINCT FROM :start_time
                        """
                    ),
                    {"id": plan.canonical_id, "start_time": plan.start_time_normalized},
                )
            for id_ in safe_orphans:
                session.execute(sql_text("DELETE FROM canonical_matches WHERE id = :id"), {"id": id_})
            session.commit()

    return {
        "planned_merges": planned_merges,
        "planned_time_fixes": planned_time_fixes,
        "planned_deleted_orphans": planned_deletes,
        "executed": int(execute),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair countdown duplicate/orphan canonical rows")
    parser.add_argument("--execute", action="store_true", help="Apply changes; default is dry-run")
    args = parser.parse_args()
    print("Result:", repair_countdown_orphans(execute=args.execute))


if __name__ == "__main__":
    main()
