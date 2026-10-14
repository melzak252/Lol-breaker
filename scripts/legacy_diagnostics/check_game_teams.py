import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("""
    SELECT COUNT(*) 
    FROM golgg_games g
    JOIN golgg_matches m ON g.match_id = m.match_id
    WHERE g.team1_id != m.team1_id
""")
print(f"Games where game.team1_id != match.team1_id: {c.fetchone()[0]}")
conn.close()
