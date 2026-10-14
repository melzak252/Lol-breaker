"""Merge expired canonical_matches that are duplicates of finished matches.

Problem: once GOL.GG marks a match as finished, later bookmaker odds for the
same real match could not attach to the finished row (resolve_canonical_match
excluded finished from candidates).  Instead, a separate expired duplicate
was created with the odds attached to it.  This script finds those safe
duplicates and repoints all FK references from the expired row to the
finished target, then deletes the expired row.

Matching: fuzzy team-name similarity with alias normalization + date proximity (±3 days).
Safe threshold: team_score >= 0.80 and overall score >= 0.78.
High-confidence: team_score >= 0.92 (auto-merge).
Medium-confidence: team_score >= 0.80 (merge with warning).

Usage:
    python -m betting_app.scripts.merge_expired_duplicates --dry-run
    python -m betting_app.scripts.merge_expired_duplicates --apply
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from sqlalchemy import text as _sql
from betting_app.core.db import get_session, query_df
from betting_app.core.matching import similarity

logger = logging.getLogger(__name__)

# Tables with canonical_match_id FK that need repointing
FK_TABLES = [
    "bets",
    "bookmaker_events",
    "canonical_predictions",
    "golgg_match_mappings",
    "model_ev_signals",
    "odds_snapshots",
    "upcoming_match_features",
    "upcoming_matches",
]

# Horizon bins for coverage estimation
HORIZON_BINS = [
    ("0-2h", 0, 2),
    ("2-6h", 2, 6),
    ("6-12h", 6, 12),
    ("12-24h", 12, 24),
    ("24-48h", 24, 48),
    ("48h+", 48, 9999),
]


def _team_score(a1: str, b1: str, a2: str, b2: str) -> float:
    """Best of direct/swapped team similarity."""
    direct = (similarity(a1, a2) + similarity(b1, b2)) / 2
    swapped = (similarity(a1, b2) + similarity(b1, a2)) / 2
    return max(direct, swapped)


def _time_score(t1: str | None, t2: str | None) -> float:
    """Score start-time proximity (1.0 = same, 0.0 = >3 days)."""
    if not t1 or not t2:
        return 0.0
    try:
        dt1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
        dt2 = datetime.fromisoformat(t2.replace("Z", "+00:00"))
    except Exception:
        return 0.45 if t1 == t2 else 0.0
    diff_h = abs((dt1 - dt2).total_seconds()) / 3600
    if diff_h <= 0.33:
        return 1.0
    if diff_h <= 1.5:
        return 0.75
    if diff_h <= 4:
        return 0.35
    if diff_h <= 72:
        return 0.10
    return 0.0


def _overall_score(team: float, time_s: float) -> float:
    return 0.72 * team + 0.23 * time_s + 0.05 * 0.5  # league unknown, neutral


def find_duplicates() -> list[dict]:
    """Find expired matches that are safe duplicates of finished matches.

    Returns list of {expired_id, target_id, score, team_score, teams, dates, refs}.
    """
    # Get expired matches with start time and valid odds
    expired_df = query_df("""
        SELECT cm.id, cm.team_a_name, cm.team_b_name,
               cm.normalized_team_a, cm.normalized_team_b,
               cm.start_time_normalized, cm.league
        FROM canonical_matches cm
        WHERE cm.status = 'expired'
          AND cm.start_time_normalized IS NOT NULL
          AND cm.normalized_team_a IS NOT NULL
          AND cm.normalized_team_b IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM odds_snapshots os
              WHERE os.canonical_match_id = cm.id
                AND os.market_type = 'match_winner'
                AND COALESCE(os.is_live, 0) = 0
                AND os.odds_a > 1 AND os.odds_b > 1
          )
        ORDER BY cm.id DESC
    """)

    if expired_df.empty:
        return []

    # Get all finished matches
    finished_df = query_df("""
        SELECT id, team_a_name, team_b_name,
               normalized_team_a, normalized_team_b,
               start_time_normalized, league
        FROM canonical_matches
        WHERE status = 'finished'
          AND start_time_normalized IS NOT NULL
          AND normalized_team_a IS NOT NULL
          AND normalized_team_b IS NOT NULL
    """)

    if finished_df.empty:
        return []

    finished_rows = finished_df.to_dict("records")
    duplicates = []

    for _, exp in expired_df.iterrows():
        exp_a = str(exp["normalized_team_a"] or "")
        exp_b = str(exp["normalized_team_b"] or "")
        exp_time = str(exp["start_time_normalized"] or "")

        best_id = None
        best_score = 0.0
        best_team = 0.0

        for fin in finished_rows:
            fin_a = str(fin["normalized_team_a"] or "")
            fin_b = str(fin["normalized_team_b"] or "")
            fin_time = str(fin["start_time_normalized"] or "")

            # Quick date filter: ±3 days
            try:
                exp_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                fin_dt = datetime.fromisoformat(fin_time.replace("Z", "+00:00"))
                if abs((exp_dt - fin_dt).total_seconds()) > 72 * 3600:
                    continue
            except Exception:
                continue

            ts = _team_score(exp_a, exp_b, fin_a, fin_b)
            if ts < 0.75:
                continue
            tm = _time_score(exp_time, fin_time)
            score = _overall_score(ts, tm)

            if ts >= 0.95:
                score = max(score, 0.85)

            if score > best_score:
                best_score = score
                best_id = int(fin["id"])
                best_team = ts

        if best_id is not None and best_team >= 0.80 and best_score >= 0.78:
            # Count references
            refs = {}
            for table in FK_TABLES:
                try:
                    r = query_df(
                        f"SELECT COUNT(*) AS cnt FROM {table} WHERE canonical_match_id = :id",
                        {"id": int(exp["id"])},
                    )
                    refs[table] = int(r.iloc[0]["cnt"]) if not r.empty else 0
                except Exception:
                    refs[table] = 0

            duplicates.append({
                "expired_id": int(exp["id"]),
                "target_id": best_id,
                "score": round(best_score, 4),
                "team_score": round(best_team, 4),
                "expired_teams": f"{exp['team_a_name']} vs {exp['team_b_name']}",
                "expired_time": exp_time[:16],
                "refs": refs,
                "total_refs": sum(refs.values()),
            })

    return duplicates


def _repoint_fk(old_id: int, new_id: int) -> dict:
    """Repoint canonical_match_id FKs from old to new. Returns per-table counts."""
    moved = {}
    for table in FK_TABLES:
        with get_session() as session:
            try:
                result = session.execute(_sql(
                    f"UPDATE {table} SET canonical_match_id = :new_id "
                    f"WHERE canonical_match_id = :old_id"
                ), {"new_id": new_id, "old_id": old_id})
                session.commit()
                moved[table] = result.rowcount or 0
            except Exception as e:
                session.rollback()
                err = str(e)
                if "UniqueViolation" in err or "duplicate key" in err:
                    # Delete conflicting rows from old (target already has them)
                    with get_session() as s2:
                        s2.execute(_sql(
                            f"DELETE FROM {table} WHERE canonical_match_id = :old_id"
                        ), {"old_id": old_id})
                        s2.commit()
                    moved[table] = -1  # signal: deleted conflicts
                else:
                    logger.warning("  Could not repoint %s: %s", table, e)
                    moved[table] = 0
    return moved


def _coverage_estimate(duplicates: list[dict]) -> None:
    """Print estimated finished-match odds coverage after merge."""
    if not duplicates:
        return

    exp_ids = ",".join(str(d["expired_id"]) for d in duplicates)
    tgt_ids = ",".join(str(d["target_id"]) for d in duplicates)
    all_ids = f"{exp_ids},{tgt_ids}"

    print("\n── Coverage estimate (finished matches with valid pre-match odds) ──")
    for label, h_min, h_max in HORIZON_BINS:
        before = query_df(f"""
            SELECT COUNT(DISTINCT cm.id) AS n
            FROM canonical_matches cm
            WHERE cm.status = 'finished'
              AND cm.start_time_normalized IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM odds_snapshots os
                  WHERE os.canonical_match_id = cm.id
                    AND os.market_type = 'match_winner'
                    AND COALESCE(os.is_live, 0) = 0
                    AND os.odds_a > 1 AND os.odds_b > 1
                    AND os.scraped_at < cm.start_time_normalized::timestamptz
                    AND EXTRACT(EPOCH FROM (cm.start_time_normalized::timestamptz - os.scraped_at))/3600 <= {h_max}
                    AND EXTRACT(EPOCH FROM (cm.start_time_normalized::timestamptz - os.scraped_at))/3600 > {h_min}
              )
        """)
        after = query_df(f"""
            SELECT COUNT(DISTINCT CASE
                WHEN os.canonical_match_id IN ({tgt_ids}) THEN os.canonical_match_id
                WHEN os.canonical_match_id IN ({exp_ids}) THEN
                    (SELECT target_id FROM (VALUES {','.join(f"({d['expired_id']},{d['target_id']})" for d in duplicates)}) AS v(expired_id, target_id)
                     WHERE v.expired_id = os.canonical_match_id)
                ELSE os.canonical_match_id
            END) AS n
            FROM odds_snapshots os
            JOIN canonical_matches cm ON cm.id = os.canonical_match_id
            WHERE os.market_type = 'match_winner'
              AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a > 1 AND os.odds_b > 1
              AND cm.start_time_normalized IS NOT NULL
              AND os.scraped_at < cm.start_time_normalized::timestamptz
              AND EXTRACT(EPOCH FROM (cm.start_time_normalized::timestamptz - os.scraped_at))/3600 <= {h_max}
              AND EXTRACT(EPOCH FROM (cm.start_time_normalized::timestamptz - os.scraped_at))/3600 > {h_min}
        """)
        b = int(before.iloc[0]["n"]) if not before.empty else 0
        a = int(after.iloc[0]["n"]) if not after.empty else 0
        print(f"  {label:8s}  before={b:4d}  after_merge={a:4d}  (+{a - b})")


def main():
    parser = argparse.ArgumentParser(description="Merge expired duplicate canonical matches into finished targets")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--apply", action="store_true", help="Apply merge")
    parser.add_argument("--min-score", type=float, default=0.82, help="Min overall score (default 0.82)")
    parser.add_argument("--min-team", type=float, default=0.92, help="Min team score (default 0.92)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.dry_run and not args.apply:
        print("Use --dry-run or --apply")
        return

    print("Finding expired duplicate canonical matches...")
    duplicates = find_duplicates()

    if not duplicates:
        print("No safe duplicates found.")
        return

    # Filter by thresholds
    safe = [d for d in duplicates if d["team_score"] >= args.min_team and d["score"] >= args.min_score]
    # Deduplicate: if multiple expired map to the same target, keep all
    print(f"\nFound {len(duplicates)} candidate duplicates, {len(safe)} pass threshold")
    print(f"  (team_score >= {args.min_team}, overall_score >= {args.min_score})")

    if not safe:
        print("No safe duplicates after filtering.")
        return

    # Summary
    unique_targets = len(set(d["target_id"] for d in safe))
    total_refs = sum(d["total_refs"] for d in safe)
    print(f"  Expired rows to merge: {len(safe)}")
    print(f"  Unique finished targets: {unique_targets}")
    print(f"  Total FK references to repoint: {total_refs}")

    # Per-duplicate details
    print("\n── Duplicates ──")
    for d in safe:
        print(f"  expired {d['expired_id']:6d} -> finished {d['target_id']:6d}  "
              f"score={d['score']:.3f} team={d['team_score']:.3f}  "
              f"{d['expired_teams'][:40]}  {d['expired_time']}")
        if d["total_refs"] > 0:
            ref_summary = ", ".join(f"{t}={c}" for t, c in d["refs"].items() if c > 0)
            print(f"    refs: {ref_summary}")

    # Coverage estimate
    try:
        _coverage_estimate(safe)
    except Exception as e:
        print(f"  (coverage estimate skipped: {e})")

    if args.dry_run:
        print("\nDRY RUN — no changes made")
        return

    # Apply
    print(f"\n{'═' * 60}")
    print(f"APPLYING MERGE: {len(safe)} expired -> finished")
    print(f"{'═' * 60}")

    merged = 0
    for d in safe:
        old_id = d["expired_id"]
        new_id = d["target_id"]
        print(f"\n  Merging expired {old_id} -> finished {new_id}")

        moved = _repoint_fk(old_id, new_id)
        moved_summary = ", ".join(f"{t}={c}" for t, c in moved.items() if c != 0)
        print(f"    repointed: {moved_summary}")

        # Delete the expired canonical match
        with get_session() as session:
            session.execute(_sql("DELETE FROM canonical_matches WHERE id = :id"), {"id": old_id})
            session.commit()
        print(f"    deleted expired canonical_match {old_id}")
        merged += 1

    print(f"\n{'═' * 60}")
    print(f"Done. Merged {merged} expired duplicates into finished targets.")
    print(f"Run bootstrap and prediction pipeline to rebuild metrics.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
