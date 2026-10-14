import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("SELECT team1_id, team2_id FROM golgg_matches WHERE match_id = '324'")
print(c.fetchone())
conn.close()
