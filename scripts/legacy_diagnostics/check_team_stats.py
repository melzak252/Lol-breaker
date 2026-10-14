import sqlite3
import json

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("SELECT team1_stats_json FROM golgg_games WHERE team1_stats_json IS NOT NULL LIMIT 5")
for row in c.fetchall():
    print(row[0])
conn.close()
