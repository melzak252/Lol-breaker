
from betting_app.core.db import query_df
import pandas as pd

# Check schema of golgg_matches
print("--- golgg_matches schema ---")
df_schema = query_df("PRAGMA table_info(golgg_matches)")
print(df_schema[['name', 'type']])

# Check if any 2026 matches have no players in golgg_game_players
df_missing = query_df("""
    SELECT gm.match_id, gm.team1_name, gm.team2_name, gm.date
    FROM golgg_matches gm
    LEFT JOIN golgg_game_players gp ON gm.match_id = gp.match_id
    WHERE gm.date LIKE '2026%' AND gp.player_id IS NULL
""")
print(f"\nMatches in 2026 without players in golgg_game_players: {len(df_missing)}")

# Check if load_last_roster would fail for any teams in upcoming_matches
df_upcoming = query_df("""
    SELECT DISTINCT team_a_name as team_name FROM canonical_matches WHERE status = 'upcoming'
    UNION
    SELECT DISTINCT team_b_name as team_name FROM canonical_matches WHERE status = 'upcoming'
""")

from betting_app.services.upcoming_inference_service import load_last_roster

missing_rosters = []
for team in df_upcoming['team_name']:
    roster = load_last_roster(team)
    if not roster or not roster.get('players'):
        missing_rosters.append(team)

print(f"\nTeams in upcoming matches with missing rosters: {len(missing_rosters)}")
if missing_rosters:
    print(f"Sample missing: {missing_rosters[:5]}")
