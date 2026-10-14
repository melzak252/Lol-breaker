"""Repair missing team IDs and abbreviated team names in golgg_matches using golgg_games."""

from __future__ import annotations

import argparse
import logging

from betting_app.core.db import connect
from betting_app.core.matching import similarity

logger = logging.getLogger(__name__)


def repair_golgg_matches(*, dry_run: bool = False) -> dict[str, int]:
    """Align and populate missing team_id and full team names in golgg_matches."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT m.match_id, m.team1_name as mt1, m.team2_name as mt2,
                   g.team1_name as gt1, g.team2_name as gt2,
                   g.team1_id as gid1, g.team2_id as gid2
            FROM golgg_matches m
            JOIN (
                SELECT match_id, team1_name, team2_name, team1_id, team2_id,
                       ROW_NUMBER() OVER(PARTITION BY match_id ORDER BY game_id) as rn
                FROM golgg_games
            ) g ON g.match_id = m.match_id AND g.rn = 1
            WHERE m.team1_id IS NULL OR LENGTH(m.team1_id) = 0
               OR m.team2_id IS NULL OR LENGTH(m.team2_id) = 0
            """
        ).fetchall()

        if not rows:
            logger.info("No golgg_matches rows need repair.")
            return {"checked": 0, "repaired": 0}

        repaired = 0
        for r in rows:
            mid = str(r["match_id"])
            mt1, mt2 = str(r["mt1"] or ""), str(r["mt2"] or "")
            gt1, gt2 = str(r["gt1"] or ""), str(r["gt2"] or "")
            gid1, gid2 = str(r["gid1"] or ""), str(r["gid2"] or "")

            s_same = similarity(mt1, gt1) + similarity(mt2, gt2)
            s_swap = similarity(mt1, gt2) + similarity(mt2, gt1)

            if s_same >= s_swap:
                t1_name, t1_id, t2_name, t2_id = gt1, gid1, gt2, gid2
            else:
                t1_name, t1_id, t2_name, t2_id = gt2, gid2, gt1, gid1

            if not dry_run:
                conn.execute(
                    """
                    UPDATE golgg_matches
                    SET team1_name = :t1_name, team1_id = :t1_id,
                        team2_name = :t2_name, team2_id = :t2_id
                    WHERE match_id = :mid
                    """,
                    {
                        "mid": mid,
                        "t1_name": t1_name,
                        "t1_id": t1_id,
                        "t2_name": t2_name,
                        "t2_id": t2_id,
                    },
                )
            repaired += 1

        if not dry_run:
            conn.commit()

        logger.info("Repaired %d golgg_matches rows (dry_run=%s)", repaired, dry_run)
        return {"checked": len(rows), "repaired": repaired}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Repair missing team IDs and abbreviated team names in golgg_matches")
    parser.add_argument("--dry-run", action="store_true", help="Report repairs without applying them")
    args = parser.parse_args()
    stats = repair_golgg_matches(dry_run=args.dry_run)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
