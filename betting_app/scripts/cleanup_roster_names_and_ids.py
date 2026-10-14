"""Utility script to audit and clean up player names and IDs across the database.

Fixes historical imports from Fandom / Liquipedia where player legal names
were stored in player_name (often containing HTML entities like &nbsp;)
and IGNs were stored in player_id instead of numeric GOL.GG IDs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Any

from sqlalchemy import bindparam, text

from betting_app.core.db import get_session
from betting_app.services.current_roster_service import (
    clean_player_name,
    lock_current_roster_teams,
    resolve_golgg_player_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cleanup_team_current_rosters() -> dict[str, int]:
    """Clean up all records in team_current_roster_players."""
    session = get_session()
    try:
        team_names = session.execute(text(
            "SELECT DISTINCT normalized_team_name FROM team_current_roster_players"
        )).scalars().all()
        lock_current_roster_teams(session, team_names)
        rows = session.execute(
            text(
                "SELECT id, normalized_team_name, role, player_id, player_name "
                "FROM team_current_roster_players WHERE normalized_team_name IN :teams"
            ).bindparams(bindparam("teams", expanding=True)),
            {"teams": team_names},
        ).fetchall()
        logger.info("Found %d rows in team_current_roster_players", len(rows))

        updated = 0
        resolved_to_id = 0

        for r in rows:
            row_id, team_name, role, pid, pname = r
            str_pid = str(pid or "").strip()
            str_pname = str(pname or "").strip()

            new_pid = str_pid
            new_pname = clean_player_name(str_pname)

            # If pid is not numeric, it likely holds an in-game name (IGN)
            if str_pid and not str_pid.isdigit():
                clean_ign = clean_player_name(str_pid)
                # Attempt resolving to numeric GOL.GG player_id
                resolved = resolve_golgg_player_id(session, clean_ign, team_name)
                if not resolved and str_pname:
                    resolved = resolve_golgg_player_id(session, clean_player_name(str_pname), team_name)

                if resolved:
                    resolved_id, resolved_name = resolved
                    new_pid = str(resolved_id)
                    new_pname = clean_player_name(resolved_name or clean_ign)
                    resolved_to_id += 1
                else:
                    # Player not in GOL.GG (debutant/academy) - keep IGN as name
                    new_pid = clean_ign
                    new_pname = clean_ign
            elif str_pid.isdigit():
                # Verify and clean player_name
                if "&" in str_pname or "\xa0" in str_pname:
                    new_pname = clean_player_name(str_pname)

            if new_pid != str_pid or new_pname != str_pname:
                session.execute(
                    text(
                        "UPDATE team_current_roster_players SET player_id = :pid, player_name = :pname WHERE id = :id"
                    ),
                    {"pid": new_pid, "pname": new_pname, "id": row_id},
                )
                updated += 1

        session.commit()
        logger.info(
            "team_current_roster_players cleaned: %d rows updated (%d resolved to GOL.GG IDs)",
            updated,
            resolved_to_id,
        )
        return {"total": len(rows), "updated": updated, "resolved_to_id": resolved_to_id}
    finally:
        session.close()


def cleanup_upcoming_match_features() -> dict[str, int]:
    """Clean up frozen player_ratings inside upcoming_match_features."""
    session = get_session()
    try:
        rows = session.execute(
            text("""
            SELECT id, canonical_match_id, features_json
            FROM upcoming_match_features
            ORDER BY id DESC
            """)
        ).fetchall()
        logger.info("Found %d rows in upcoming_match_features", len(rows))

        updated = 0
        for row_id, match_id, features_json in rows:
            if not features_json:
                continue
            data = json.loads(features_json) if isinstance(features_json, str) else features_json
            if not isinstance(data, dict):
                continue

            pr = data.get("player_ratings")
            if not isinstance(pr, dict):
                continue

            changed = False
            for side in ("team_a_roster", "team_b_roster"):
                roster = pr.get(side)
                if not isinstance(roster, dict):
                    continue
                players = roster.get("players")
                if not isinstance(players, list):
                    continue

                for p in players:
                    if not isinstance(p, dict):
                        continue
                    pname = str(p.get("player_name") or "")
                    pid = str(p.get("player_id") or "")

                    # Clean HTML entities
                    if "&" in pname or "\xa0" in pname:
                        p["player_name"] = clean_player_name(pname)
                        changed = True

                    # If pid is non-numeric, it might be an IGN
                    if pid and not pid.isdigit():
                        clean_ign = clean_player_name(pid)
                        team_name = roster.get("team_name")
                        resolved = resolve_golgg_player_id(session, clean_ign, team_name)
                        if resolved:
                            p["player_id"] = str(resolved[0])
                            p["player_name"] = clean_player_name(resolved[1] or clean_ign)
                            changed = True
                        elif clean_ign:
                            p["player_name"] = clean_ign
                            changed = True

            if changed:
                new_json = json.dumps(data)
                session.execute(
                    text("UPDATE upcoming_match_features SET features_json = :fj WHERE id = :id"),
                    {"fj": new_json, "id": row_id},
                )
                updated += 1

        session.commit()
        logger.info("upcoming_match_features cleaned: %d rows updated", updated)
        return {"total": len(rows), "updated": updated}
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up player names and IDs across database")
    parser.add_argument("--rosters-only", action="store_true", help="Clean only team_current_roster_players")
    parser.add_argument("--features-only", action="store_true", help="Clean only upcoming_match_features")
    args = parser.parse_args()

    if not args.features_only:
        cleanup_team_current_rosters()
    if not args.rosters_only:
        cleanup_upcoming_match_features()

    logger.info("Cleanup completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
