import sqlite3
import json

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("""
    SELECT m.date, gp.stats_json 
    FROM golgg_matches m
    JOIN golgg_games g ON m.match_id = g.match_id
    JOIN golgg_game_players gp ON g.game_id = gp.game_id
    WHERE m.date >= '2026-01-01' AND gp.stats_json IS NOT NULL
    LIMIT 5
""")
for row in c.fetchall():
    print(f"Date: {row[0]}")
    print(row[1])
conn.close()
