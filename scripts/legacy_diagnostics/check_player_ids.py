import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM golgg_game_players WHERE player_id IS NULL OR player_id = '' OR player_id = 'None'")
print(f"Empty player_ids: {c.fetchone()[0]}")

c.execute("SELECT COUNT(*) FROM golgg_game_players WHERE player_name IS NULL OR player_name = ''")
print(f"Empty player_names: {c.fetchone()[0]}")
conn.close()
