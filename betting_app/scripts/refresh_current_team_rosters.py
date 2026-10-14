"""Materialize the latest completed GOL.GG roster for every team.

Manual confirmations are only replaced when GOL.GG has a later completed
match, since ``upsert_current_roster`` compares source timestamps.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json

from sqlalchemy import text

from betting_app.core.db import get_session
from betting_app.services.current_roster_service import lock_current_roster_teams, upsert_current_roster


def refresh_current_team_rosters(*, include_manual_overrides: bool = False) -> dict[str, int]:
    session = get_session()
    try:
        rows = session.execute(text("""
            WITH team_match_ranked AS (
                SELECT
                    COALESCE(NULLIF(gp.team_id, ''), LOWER(gp.team_name)) AS team_key,
                    gp.team_id, gp.team_name, gp.match_id, gm.date AS match_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(NULLIF(gp.team_id, ''), LOWER(gp.team_name))
                        ORDER BY gm.date DESC, CAST(gm.match_id AS INTEGER) DESC
                    ) AS match_rank
                FROM golgg_game_players gp
                JOIN golgg_matches gm ON gm.match_id = gp.match_id
                WHERE gp.team_name IS NOT NULL AND gp.player_id IS NOT NULL
            ), latest_matches AS (
                SELECT DISTINCT team_key, team_id, team_name, match_id, match_date
                FROM team_match_ranked WHERE match_rank = 1
            ), game_ranked AS (
                SELECT
                    lm.team_key, lm.team_id, lm.team_name, lm.match_id, lm.match_date,
                    gp.game_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY lm.team_key ORDER BY CAST(gp.game_id AS INTEGER) ASC
                    ) AS game_rank
                FROM latest_matches lm
                JOIN golgg_game_players gp ON gp.match_id = lm.match_id
                    AND (gp.team_id = lm.team_id OR (lm.team_id IS NULL AND gp.team_name = lm.team_name))
            )
            SELECT gr.team_key, gr.team_id, gr.team_name, gr.match_id, gr.match_date,
                   gr.game_id, gp.player_id, gp.player_name, gp.role
            FROM game_ranked gr
            JOIN golgg_game_players gp ON gp.game_id = gr.game_id
                AND (gp.team_id = gr.team_id OR (gr.team_id IS NULL AND gp.team_name = gr.team_name))
            WHERE gr.game_rank = 1
            ORDER BY gr.team_key, gp.role
        """)).mappings().all()
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row["team_key"])].append(dict(row))
        overrides = []
        if include_manual_overrides:
            overrides = session.execute(text("""
                SELECT mo.roster_json, mo.updated_at,
                       CASE mo.team_side WHEN 'a' THEN umf.team_a_golgg_name ELSE umf.team_b_golgg_name END AS golgg_team,
                       CASE mo.team_side WHEN 'a' THEN cm.team_a_name ELSE cm.team_b_name END AS canonical_team
                FROM match_roster_overrides mo
                JOIN canonical_matches cm ON cm.id = mo.canonical_match_id
                LEFT JOIN upcoming_match_features umf ON umf.canonical_match_id = mo.canonical_match_id
            """)).mappings().all()
        lock_current_roster_teams(
            session,
            [players[0]["team_name"] for players in grouped.values()]
            + [override["golgg_team"] or override["canonical_team"] for override in overrides],
        )
        updated = skipped = 0
        for players in grouped.values():
            first = players[0]
            changed = upsert_current_roster(
                session,
                team_name=str(first["team_name"]), team_id=str(first["team_id"]) if first["team_id"] else None,
                players=players, source="auto", source_match_id=str(first["match_id"]),
                source_game_id=str(first["game_id"]), source_match_date=str(first["match_date"] or ""),
            )
            updated += int(changed)
            skipped += int(not changed)
        manual_updated = 0
        if include_manual_overrides:
            # One-time migration aid for overrides saved before this durable
            # roster table existed. Do not run this in the scheduled refresh:
            # a later real GOL.GG game must be allowed to supersede a manual
            # lineup after it has actually been played.
            for override in overrides:
                try:
                    roster = json.loads(override["roster_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(roster, dict) or not isinstance(roster.get("players"), list):
                    continue
                manual_updated += int(upsert_current_roster(
                    session,
                    team_name=str(override["golgg_team"] or override["canonical_team"]),
                    players=roster["players"], source="manual",
                    source_match_date=str(override["updated_at"] or ""),
                ))
        session.commit()
        return {"teams_seen": len(grouped), "updated": updated, "skipped": skipped, "manual_updated": manual_updated}
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh materialized current team rosters from GOL.GG games")
    parser.add_argument("--include-manual-overrides", action="store_true", help="One-time seed from existing match overrides")
    args = parser.parse_args()
    print(refresh_current_team_rosters(include_manual_overrides=args.include_manual_overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
