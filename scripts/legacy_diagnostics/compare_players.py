import json
import sqlite3
from src.utils.golgg_schema import players1, players2

with open("data/golgg_matches.json") as f:
    matches = json.load(f)
matches = [m for m in matches if not m.get("draw")]
matches.sort(key=lambda m: (m["date"], int(m.get("match_id") or 0)))

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()

for i in range(5):
    m = matches[i]
    m_id = str(m["match_id"])
    p1_json = players1(m)
    p2_json = players2(m)
    
    c.execute("""
        SELECT gp.player_id 
        FROM golgg_games g
        JOIN golgg_game_players gp ON g.game_id = gp.game_id
        WHERE g.match_id = ? AND gp.team_id = ?
        ORDER BY CAST(g.game_id AS INTEGER) ASC
    """, (m_id, m.get("t1_id") or m.get("tid_1")))
    p1_db = [row[0] for row in c.fetchall()]
    
    print(f"Match {m_id}")
    print(f"JSON p1: {p1_json}")
    print(f"DB p1: {p1_db[:5]}")
    print("---")
