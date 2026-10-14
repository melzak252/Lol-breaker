"""
Remove CS:GO, Valorant, Overwatch matches from canonical_matches.
All these come from Betfan which incorrectly categorizes them under /lol/ URL path.

Deletes in FK-safe order:
1. canonical_predictions
2. odds_snapshots
3. upcoming_match_features
4. upcoming_matches
5. canonical_matches
"""

import sys
import logging
from betting_app.core.db import connect

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NON_LOL_LEAGUES = [
    "IEM Cologne Major - Eliminacje H2H",
    "IEM Cologne Major",
    "European Pro League Series",
    "CCT European Series",
    "NODWIN Clutch Series",
    "United21",
    "Valorant Champions Tour Masters London",
    "Overwatch Champions Series Korea",
]

FK_ORDER = [
    "model_ev_signals",
    "canonical_predictions",
    "odds_snapshots",
    "upcoming_match_features",
    "upcoming_matches",
]


def main():
    dry_run = "--dry-run" in sys.argv

    # Phase 1: get IDs and count refs (separate session to avoid abort issues)
    with connect() as conn:
        ids_raw = conn.execute(
            "SELECT id FROM canonical_matches WHERE league = ANY(:leagues)",
            {"leagues": NON_LOL_LEAGUES},
        ).fetchall()
    ids = [dict(r)["id"] for r in ids_raw]
    log.info(f"Matches to delete: {len(ids)}")
    if not ids:
        log.info("Nothing to do.")
        return

    with connect() as conn:
        for tbl in FK_ORDER:
            cnt = dict(
                conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE canonical_match_id = ANY(:ids)",
                    {"ids": ids},
                ).fetchone()
            )["count"]
            log.info(f"  FK in {tbl}: {cnt}")
    # Check golgg mappings
    with connect() as conn:
        try:
            cnt = dict(
                conn.execute(
                    "SELECT COUNT(*) FROM golgg_match_mappings WHERE canonical_match_id = ANY(:ids)",
                    {"ids": ids},
                ).fetchone()
            )["count"]
            if cnt > 0:
                log.info(f"  FK in golgg_match_mappings: {cnt}")
        except Exception:
            pass

    if dry_run:
        log.info("Dry-run complete. Pass --yes to execute.")
        return

    # Phase 2: delete (separate fresh sessions per table)
    # model_ev_signals must be deleted by prediction_id (double FK)
    with connect() as conn:
        pred_ids = [
            dict(r)["id"]
            for r in conn.execute(
                "SELECT id FROM canonical_predictions WHERE canonical_match_id = ANY(:ids)",
                {"ids": ids},
            ).fetchall()
        ]
    with connect() as conn:
        if pred_ids:
            result = conn.execute(
                "DELETE FROM model_ev_signals WHERE canonical_prediction_id = ANY(:pids)",
                {"pids": pred_ids},
            )
            conn.commit()
            log.info(f"  Deleted {result.rowcount} from model_ev_signals")

    for tbl in [t for t in FK_ORDER if t != "model_ev_signals"]:
        with connect() as conn:
            result = conn.execute(
                f"DELETE FROM {tbl} WHERE canonical_match_id = ANY(:ids)",
                {"ids": ids},
            )
            conn.commit()
            log.info(f"  Deleted {result.rowcount} from {tbl}")

    with connect() as conn:
        result = conn.execute(
            "DELETE FROM canonical_matches WHERE league = ANY(:leagues)",
            {"leagues": NON_LOL_LEAGUES},
        )
        conn.commit()
        log.info(f"  Deleted {result.rowcount} from canonical_matches")

    log.info("All deletions committed successfully.")


if __name__ == "__main__":
    main()
