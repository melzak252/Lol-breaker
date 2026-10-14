import sqlite3
import os

db_path = 'data/betting_app.sqlite3'

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def add_column(table, column, type):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type}")
        print(f"Added column {column} to {table}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print(f"Column {column} already exists in {table}")
        else:
            print(f"Error adding {column} to {table}: {e}")

add_column('upcoming_matches', 'last_seen_at', 'DATETIME')
add_column('canonical_matches', 'best_of', 'INTEGER')
add_column('upcoming_matches', 'team_a_golgg_id', 'INTEGER')
add_column('upcoming_matches', 'team_b_golgg_id', 'INTEGER')
add_column('team_aliases', 'alias', 'TEXT')

# Copy data if needed
try:
    cursor.execute("UPDATE team_aliases SET alias = golgg_team_name WHERE alias IS NULL AND golgg_team_name IS NOT NULL")
    print("Copied golgg_team_name to alias in team_aliases")
except sqlite3.OperationalError as e:
    print(f"Note: Could not copy data: {e}")

conn.commit()
conn.close()
print("Done.")
