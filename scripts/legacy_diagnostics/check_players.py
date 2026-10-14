
from betting_app.core.db import query_df
import pandas as pd

# Check 2026 matches without players
df = query_df("""
    SELECT gm.match_id, gm.team1, gm.team2, gm.date
    FROM golgg_matches gm
    LEFT JOIN golgg_game_players gp ON gm.match_id = gp.match_id
    WHERE gm.date LIKE '2026%' AND gp.player_id IS NULL
""")

print(f"Matches in 2026 without players: {len(df)}")
if len(df) > 0:
    print(df.head())

# Check total matches in 2026
df_total = query_df("SELECT COUNT(*) as count FROM golgg_matches WHERE date LIKE '2026%'")
print(f"Total matches in 2026: {df_total.iloc[0]['count']}")
