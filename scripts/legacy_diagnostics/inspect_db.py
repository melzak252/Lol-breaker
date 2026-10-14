import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
cursor = conn.cursor()

print("--- Tables ---")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print(tables)

if 'alembic_version' in tables:
    print("\n--- alembic_version ---")
    cursor.execute("SELECT version_num FROM alembic_version;")
    print(cursor.fetchall())
else:
    print("\n--- alembic_version table missing ---")

print("\n--- upcoming_matches columns ---")
cursor.execute("PRAGMA table_info(upcoming_matches);")
cols = [row[1] for row in cursor.fetchall()]
print(cols)

print("\n--- automation_commands columns ---")
cursor.execute("PRAGMA table_info(automation_commands);")
cols = [row[1] for row in cursor.fetchall()]
print(cols)

print("\n--- upcoming_matches columns ---")
cursor.execute("PRAGMA table_info(upcoming_matches);")
cols = [row[1] for row in cursor.fetchall()]
print(cols)

print("\n--- alembic_version content ---")
cursor.execute("SELECT * FROM alembic_version;")
print(cursor.fetchall())

conn.close()
