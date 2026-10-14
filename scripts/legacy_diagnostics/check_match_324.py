import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("SELECT game_id, team_id, team_name, player_id FROM golgg_game_players WHERE match_id = '324'")
for row in c.fetchall():
    print(row)
conn.close()
