import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM golgg_matches WHERE team1_id IS NULL OR team1_id = '' OR team2_id IS NULL OR team2_id = ''")
print(f"Empty team ids in matches: {c.fetchone()[0]}")
conn.close()
