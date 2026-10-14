import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("""
    SELECT COUNT(*) 
    FROM golgg_games g
    JOIN golgg_matches m ON g.match_id = m.match_id
    WHERE g.team1_id != m.team1_id AND g.team1_id != m.team2_id
""")
print(f"Games where game.team1_id is neither match.team1_id nor match.team2_id: {c.fetchone()[0]}")
conn.close()
