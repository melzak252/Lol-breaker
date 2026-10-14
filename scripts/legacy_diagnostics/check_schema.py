import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(canonical_matches)")
columns = cursor.fetchall()
for col in columns:
    print(col)
