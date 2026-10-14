"""
Fix wrong start_time_normalized in canonical_matches by cross-referencing
GOL.GG match dates, then re-run the GOLGG backfill.

The scraper's "Dzisiaj"/"Jutro" date resolution bug (using local timezone
instead of UTC) caused many canonical matches to have dates shifted +1 day
(or more).  This script repairs those dates where GOLGG has the same team
pair within a ±7-day window.

Usage:
    python -m betting_app.scripts.fix_canonical_dates [--yes]
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/app")

import os

os.environ.setdefault("BETTING_ENV", "production")

from datetime import UTC, date, datetime, timedelta
from typing import Any

from betting_app.core.db import connect
from betting_app.core.matching import normalize_team_name

logger = logging.getLogger(__name__)

# How many days before/after the canonical date to look for GOLGG matches.
SEARCH_WINDOW_DAYS = 7


def _golgg_team_keys(golgg_row: dict[str, Any]) -> tuple[str, str]:
    """Return (team1_key, team2_key) for a GOLGG match row."""
    return (
        normalize_team_name(golgg_row["team1_name"] or ""),
        normalize_team_name(golgg_row["team2_name"] or ""),
    )


def find_all_unmapped_expired(conn) -> list[dict[str, Any]]:
    """Return expired (or upcoming) canonical matches without GOLGG mapping."""
    rows = conn.execute(
        """
        SELECT cm.id, cm.team_a_name, cm.team_b_name,
               cm.normalized_team_a, cm.normalized_team_b,
               cm.start_time_normalized, cm.league, cm.status
        FROM canonical_matches cm
        WHERE cm.status IN ('expired', 'upcoming')
          AND cm.normalized_team_a IS NOT NULL
          AND cm.normalized_team_b IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM golgg_match_mappings gmm
              WHERE gmm.canonical_match_id = cm.id
          )
        ORDER BY cm.start_time_normalized DESC NULLS LAST
        """
    ).fetchall()
    return [dict(r) for r in rows]


def find_golgg_matches(
    conn, norm_a: str, norm_b: str, cm_date: str | None
) -> list[dict[str, Any]]:
    """
    Find GOLGG matches where the normalized team names match.
    Optionally filter by a date window around cm_date.
    """
    if cm_date:
        try:
            cm_dt = datetime.fromisoformat(cm_date)
        except (ValueError, TypeError):
            cm_dt = datetime.now(UTC)
    else:
        cm_dt = datetime.now(UTC)

    start_window = (cm_dt - timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    end_window = (cm_dt + timedelta(days=SEARCH_WINDOW_DAYS)).strftime("%Y-%m-%d")

    # Search GOLGG for team names that match either order
    # Use direct comparison on the raw names since GOLGG doesn't have normalized columns.
    rows = conn.execute(
        """
        SELECT match_id, team1_name, team2_name, tournament_name, date,
               team1_score, team2_score
        FROM golgg_matches
        WHERE date >= ? AND date <= ?
          AND (
               (team1_name ILIKE '%' || ? || '%' AND team2_name ILIKE '%' || ? || '%')
            OR (team1_name ILIKE '%' || ? || '%' AND team2_name ILIKE '%' || ? || '%')
          )
        ORDER BY date
        """,
        (start_window, end_window, norm_a, norm_b, norm_b, norm_a),
    ).fetchall()

    # Also try with normalized names on GOLGG side too — compute in Python
    # Since ILIKE may miss exact matches due to word order, do a second pass
    # using normalize_team_name on GOLGG names.
    if not rows:
        all_golgg = conn.execute(
            """
            SELECT match_id, team1_name, team2_name, tournament_name, date,
                   team1_score, team2_score
            FROM golgg_matches
            WHERE date >= ? AND date <= ?
            """,
            (start_window, end_window),
        ).fetchall()
        for g in all_golgg:
            ga, gb = _golgg_team_keys(g)
            # Normalize canonical keys too
            ca = normalize_team_name(norm_a or "")
            cb = normalize_team_name(norm_b or "")
            ga_norm = normalize_team_name(ga or "")
            gb_norm = normalize_team_name(gb or "")
            if (ga_norm == ca and gb_norm == cb) or (
                ga_norm == cb and gb_norm == ca
            ):
                rows.append(g)

    return [dict(r) for r in rows]


def fix_and_map(conn, cm: dict, golgg: dict) -> bool:
    """Update canonical match date to GOLGG date and create mapping."""
    cm_id = cm["id"]
    golgg_id = golgg["match_id"]
    golgg_date = golgg["date"]

    # Update start_time_normalized to GOLGG's date (noon UTC)
    new_start = f"{golgg_date}T12:00:00+00:00"
    conn.execute(
        "UPDATE canonical_matches SET start_time_normalized = ? WHERE id = ?",
        (new_start, cm_id),
    )

    # If GOLGG has scores, mark as finished
    if golgg.get("team1_score") is not None and golgg.get("team2_score") is not None:
        winner = (
            golgg["team1_name"]
            if golgg["team1_score"] > golgg["team2_score"]
            else golgg["team2_name"]
        )
        loser = (
            golgg["team2_name"]
            if golgg["team1_score"] > golgg["team2_score"]
            else golgg["team1_name"]
        )
        conn.execute(
            "UPDATE canonical_matches SET status = 'finished', winner_name = ?, loser_name = ? WHERE id = ?",
            (winner, loser, cm_id),
        )

    # Try to create mapping — use ON CONFLICT to safely handle duplicates.
    # The golgg_match_id unique constraint means at most one canonical
    # per GOLGG match gets the mapping; that is fine.
    conn.execute(
        "INSERT INTO golgg_match_mappings (canonical_match_id, golgg_match_id, confidence) "
        "VALUES (?, ?, 0.95) ON CONFLICT (golgg_match_id) DO NOTHING",
        (cm_id, golgg_id),
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix canonical match dates from GOLGG")
    parser.add_argument("--yes", action="store_true", help="Execute changes")
    args = parser.parse_args()

    dry_run = not args.yes

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with connect() as conn:
        unmapped = find_all_unmapped_expired(conn)
        logger.info("Found %d unmapped expired/upcoming matches", len(unmapped))

        fixed_count = 0
        skipped_no_golgg = 0
        skipped_same_date = 0
        errors = 0
        same_date_candidates: list[tuple[dict, dict]] = []

        for cm in unmapped:
            cm_id = cm["id"]
            norm_a = cm["normalized_team_a"] or ""
            norm_b = cm["normalized_team_b"] or ""
            cm_date = cm["start_time_normalized"]

            try:
                golgg_matches = find_golgg_matches(conn, norm_a, norm_b, cm_date)
            except Exception as e:
                logger.error("Error searching GOLGG for CM %d: %s", cm_id, e)
                errors += 1
                continue

            if not golgg_matches:
                skipped_no_golgg += 1
                continue

            # Prefer the match with closest date
            if cm_date:
                try:
                    cm_dt = datetime.fromisoformat(cm_date).date()
                except (ValueError, TypeError):
                    cm_dt = None
            else:
                cm_dt = None

            best = golgg_matches[0]
            if cm_dt and golgg_matches:
                best = min(
                    golgg_matches,
                    key=lambda g: abs(
                        (
                            datetime.strptime(g["date"], "%Y-%m-%d").date()
                            if g["date"]
                            else cm_dt
                        )
                        - cm_dt
                    ),
                )

            # Only update if dates actually differ
            golgg_date = best["date"]
            cm_date_short = cm_date[:10] if cm_date else None
            if golgg_date and cm_date_short and golgg_date == cm_date_short:
                # Date already matches — but may still need a mapping.
                # Handle this in Phase 2 below.
                same_date_candidates.append((cm, best))
                skipped_same_date += 1
                continue

            old_date = cm_date or "NULL"
            logger.info(
                "CM %d: %s vs %s | %s -> %s (GOLGG: %s)",
                cm_id,
                cm["team_a_name"],
                cm["team_b_name"],
                old_date,
                golgg_date,
                best["tournament_name"],
            )

            if not dry_run:
                try:
                    fix_and_map(conn, cm, best)
                    conn.commit()  # persist after each successful fix
                    fixed_count += 1
                except Exception as e:
                    logger.error("Error fixing CM %d: %s", cm_id, e)
                    errors += 1
            else:
                fixed_count += 1

        logger.info("=" * 50)
        logger.info("SUMMARY (dry_run=%s)", dry_run)
        logger.info("  Fixed (or would fix):    %d", fixed_count)
        logger.info("  Skipped (same date):     %d  (may need mapping)", skipped_same_date)
        logger.info("  Skipped (no GOLGG):      %d", skipped_no_golgg)

        # ==================== PHASE 2: Create mappings for already-correct dates ====================
        if not dry_run and same_date_candidates:
            logger.info("")
            logger.info("Phase 2: Creating mappings for %d already-correct-date matches...", len(same_date_candidates))
            mapped_in_phase2 = 0
            for cm, golgg in same_date_candidates:
                cm_id = cm["id"]
                golgg_id = golgg["match_id"]
                existing = conn.execute(
                    "SELECT id FROM golgg_match_mappings WHERE golgg_match_id = ?",
                    (golgg_id,),
                ).fetchone()
                if not existing:
                    try:
                        conn.execute(
                            "INSERT INTO golgg_match_mappings (canonical_match_id, golgg_match_id, confidence) VALUES (?, ?, 0.95)",
                            (cm_id, golgg_id),
                        )
                        # If GOLGG has scores, mark as finished
                        if golgg.get("team1_score") is not None and golgg.get("team2_score") is not None:
                            winner = (
                                golgg["team1_name"]
                                if golgg["team1_score"] > golgg["team2_score"]
                                else golgg["team2_name"]
                            )
                            loser = (
                                golgg["team2_name"]
                                if golgg["team1_score"] > golgg["team2_score"]
                                else golgg["team1_name"]
                            )
                            conn.execute(
                                "UPDATE canonical_matches SET status = 'finished', winner_name = ?, loser_name = ? WHERE id = ?",
                                (winner, loser, cm_id),
                            )
                        mapped_in_phase2 += 1
                    except Exception:
                        pass  # ignore individual mapping errors
            conn.commit()  # persist Phase 2 mappings
            logger.info("  Mapped in Phase 2:      %d", mapped_in_phase2)
        logger.info("  Errors:                  %d", errors)
        logger.info("=" * 50)

        if dry_run:
            print("\nRun with --yes to apply changes.")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
