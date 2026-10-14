"""Repair duplicate canonical_matches rows.

The bookmaker feeds sometimes emit unstable start labels (for example changing
countdowns).  Historically those labels were part of the bookmaker match key and
the canonical resolver only looked at `status='upcoming'`, so the same real match
could be represented by several canonical rows on the same date.

This script conservatively groups canonical rows by unordered canonical team keys
and UTC date, chooses one keeper, remaps foreign keys, and deletes duplicate
canonical rows.  It defaults to dry-run; pass --execute to write changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import text as sql_text

from betting_app.core.db import get_session, query_df
from betting_app.services.canonical_match_service import canonical_team_key


@dataclass(frozen=True)
class CanonicalRow:
    id: int
    team_a_name: str
    team_b_name: str
    normalized_team_a: str
    normalized_team_b: str
    start_time_normalized: str
    league: str | None
    status: str
    mappings: int
    odds_snapshots: int
    upcoming_matches: int
    bookmaker_events: int
    predictions: int
    features: int
    ev_signals: int
    bets: int

    @property
    def date_key(self) -> str:
        return (self.start_time_normalized or "")[:10] or "unknown"

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted([canonical_team_key(self.normalized_team_a), canonical_team_key(self.normalized_team_b)]))  # type: ignore[return-value]

    @property
    def evidence(self) -> int:
        return self.odds_snapshots + self.upcoming_matches + self.bookmaker_events


def _int(value: Any) -> int:
    return int(value or 0)


def load_rows() -> list[CanonicalRow]:
    df = query_df(
        """
        SELECT
            cm.id,
            cm.team_a_name,
            cm.team_b_name,
            cm.normalized_team_a,
            cm.normalized_team_b,
            cm.start_time_normalized,
            cm.league,
            cm.status,
            COALESCE(gmm.cnt, 0) AS mappings,
            COALESCE(os.cnt, 0) AS odds_snapshots,
            COALESCE(um.cnt, 0) AS upcoming_matches,
            COALESCE(be.cnt, 0) AS bookmaker_events,
            COALESCE(cp.cnt, 0) AS predictions,
            COALESCE(umf.cnt, 0) AS features,
            COALESCE(mes.cnt, 0) AS ev_signals,
            COALESCE(bets.cnt, 0) AS bets
        FROM canonical_matches cm
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM golgg_match_mappings GROUP BY canonical_match_id) gmm ON gmm.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM odds_snapshots GROUP BY canonical_match_id) os ON os.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM upcoming_matches GROUP BY canonical_match_id) um ON um.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM bookmaker_events GROUP BY canonical_match_id) be ON be.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM canonical_predictions GROUP BY canonical_match_id) cp ON cp.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM upcoming_match_features GROUP BY canonical_match_id) umf ON umf.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM model_ev_signals GROUP BY canonical_match_id) mes ON mes.canonical_match_id = cm.id
        LEFT JOIN (SELECT canonical_match_id, COUNT(*) cnt FROM bets GROUP BY canonical_match_id) bets ON bets.canonical_match_id = cm.id
        WHERE cm.start_time_normalized IS NOT NULL
        ORDER BY cm.id
        """
    )
    rows: list[CanonicalRow] = []
    for _, r in df.iterrows():
        rows.append(
            CanonicalRow(
                id=int(r["id"]),
                team_a_name=str(r.get("team_a_name") or ""),
                team_b_name=str(r.get("team_b_name") or ""),
                normalized_team_a=str(r.get("normalized_team_a") or ""),
                normalized_team_b=str(r.get("normalized_team_b") or ""),
                start_time_normalized=str(r.get("start_time_normalized") or ""),
                league=str(r.get("league") or "") or None,
                status=str(r.get("status") or ""),
                mappings=_int(r.get("mappings")),
                odds_snapshots=_int(r.get("odds_snapshots")),
                upcoming_matches=_int(r.get("upcoming_matches")),
                bookmaker_events=_int(r.get("bookmaker_events")),
                predictions=_int(r.get("predictions")),
                features=_int(r.get("features")),
                ev_signals=_int(r.get("ev_signals")),
                bets=_int(r.get("bets")),
            )
        )
    return rows


def keeper_sort_key(row: CanonicalRow) -> tuple[int, int, int, int, int, int]:
    status_score = {"finished": 3, "upcoming": 2, "expired": 1}.get(row.status, 0)
    # Prefer rows already connected to GOL.GG/results, then live odds evidence.
    return (
        row.mappings,
        status_score,
        row.odds_snapshots,
        row.upcoming_matches,
        row.bookmaker_events,
        -row.id,
    )


def find_duplicate_groups(rows: list[CanonicalRow]) -> list[list[CanonicalRow]]:
    groups: dict[tuple[tuple[str, str], str], list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        groups[(row.pair_key, row.date_key)].append(row)
    return [group for group in groups.values() if len(group) > 1]


def _parse_date_key(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def find_mapped_window_groups(rows: list[CanonicalRow], *, window_days: int) -> list[list[CanonicalRow]]:
    """Find date-shift duplicates around one GOL.GG-mapped row.

    Same-day repair is not enough for feeds that saved the same bookmaker match
    under a wrong/unstable start date (for example countdown/timezone labels).
    This mode is intentionally conservative: it only proposes a group when a
    pair has exactly one mapped canonical row and one or more unmapped rows for
    the same canonical team pair within +/- `window_days` of that mapped row.
    """

    by_pair: dict[tuple[str, str], list[CanonicalRow]] = defaultdict(list)
    for row in rows:
        by_pair[row.pair_key].append(row)

    groups: list[list[CanonicalRow]] = []
    seen: set[tuple[int, ...]] = set()
    for pair_rows in by_pair.values():
        mapped_rows = [row for row in pair_rows if row.mappings > 0]
        if len(mapped_rows) != 1:
            continue
        keeper = mapped_rows[0]
        keeper_day = _parse_date_key(keeper.date_key)
        if keeper_day is None:
            continue
        duplicates: list[CanonicalRow] = []
        for row in pair_rows:
            if row.id == keeper.id or row.mappings > 0:
                continue
            row_day = _parse_date_key(row.date_key)
            if row_day is None:
                continue
            if abs((row_day - keeper_day).days) <= window_days:
                duplicates.append(row)
        if duplicates:
            group = [keeper, *duplicates]
            key = tuple(sorted(row.id for row in group))
            if key not in seen:
                seen.add(key)
                groups.append(group)
    return groups


def choose_keeper(group: list[CanonicalRow]) -> CanonicalRow:
    return max(group, key=keeper_sort_key)


def merge_duplicate(session, keeper_id: int, duplicate_id: int) -> None:
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
    session.execute(sql_text("DELETE FROM canonical_matches WHERE id = :duplicate_id"), {"duplicate_id": duplicate_id})


def repair_duplicates(
    *,
    execute: bool = False,
    limit_groups: int | None = None,
    merge_mapped_window_days: int = 0,
) -> dict[str, int]:
    rows = load_rows()
    groups = find_duplicate_groups(rows)
    if merge_mapped_window_days > 0:
        for group in find_mapped_window_groups(rows, window_days=merge_mapped_window_days):
            key = tuple(sorted(row.id for row in group))
            overlapping_indexes = [
                index
                for index, existing in enumerate(groups)
                if set(row.id for row in existing) & set(key)
            ]
            if not overlapping_indexes:
                groups.append(group)
                continue

            merged_by_id = {row.id: row for row in group}
            for index in sorted(overlapping_indexes, reverse=True):
                merged_by_id.update({row.id: row for row in groups.pop(index)})
            groups.append(list(merged_by_id.values()))
    groups.sort(key=lambda group: (group[0].date_key, group[0].pair_key))
    if limit_groups is not None:
        groups = groups[:limit_groups]

    planned_groups = 0
    planned_deletes = 0
    skipped_groups = 0
    for group in groups:
        mapped_rows = [row for row in group if row.mappings > 0]
        if len(mapped_rows) > 1:
            skipped_groups += 1
            print(f"SKIP multiple mapped rows: ids={[row.id for row in group]}")
            continue
        keeper = choose_keeper(group)
        duplicates = [row for row in group if row.id != keeper.id]
        planned_groups += 1
        planned_deletes += len(duplicates)
        print(
            f"{'MERGE' if execute else 'DRY'} {group[0].date_key} {group[0].pair_key}: "
            f"keeper={keeper.id}({keeper.status},map={keeper.mappings},odds={keeper.odds_snapshots},up={keeper.upcoming_matches}) "
            f"dups={[row.id for row in duplicates]}"
        )
        for row in group:
            print(
                f"  id={row.id} {row.status} {row.start_time_normalized} "
                f"{row.team_a_name} vs {row.team_b_name} map={row.mappings} odds={row.odds_snapshots} up={row.upcoming_matches}"
            )

        if execute:
            with get_session() as session:
                for duplicate in duplicates:
                    merge_duplicate(session, keeper.id, duplicate.id)
                session.commit()

    return {
        "groups": len(groups),
        "planned_groups": planned_groups,
        "planned_deleted_duplicates": planned_deletes,
        "skipped_groups": skipped_groups,
        "executed": int(execute),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair duplicate canonical match rows")
    parser.add_argument("--execute", action="store_true", help="Apply changes; default is dry-run")
    parser.add_argument("--limit-groups", type=int, default=None, help="Process only first N duplicate groups")
    parser.add_argument(
        "--merge-mapped-window-days",
        type=int,
        default=0,
        help="Also merge unmapped same-pair rows within +/-N days into the single mapped row for that pair.",
    )
    args = parser.parse_args()
    stats = repair_duplicates(
        execute=args.execute,
        limit_groups=args.limit_groups,
        merge_mapped_window_days=args.merge_mapped_window_days,
    )
    print("Result:", stats)


if __name__ == "__main__":
    main()
