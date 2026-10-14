import sqlite3

conn = sqlite3.connect('data/betting_app.sqlite3')
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS golgg_match_mappings (
    canonical_match_id INTEGER NOT NULL,
    golgg_match_id TEXT NOT NULL UNIQUE,
    confidence REAL NOT NULL,
    mapped_by TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (canonical_match_id, golgg_match_id)
);
""")

cursor.execute("ALTER TABLE canonical_matches ADD COLUMN winner_name VARCHAR(200);")
cursor.execute("ALTER TABLE canonical_matches ADD COLUMN loser_name VARCHAR(200);")
cursor.execute("ALTER TABLE canonical_matches ADD COLUMN winner_normalized VARCHAR(200);")
cursor.execute("ALTER TABLE canonical_matches ADD COLUMN winner_side VARCHAR(20);")
cursor.execute("ALTER TABLE canonical_matches ADD COLUMN result_source VARCHAR(50);")
cursor.execute("ALTER TABLE canonical_matches ADD COLUMN result_source_match_id VARCHAR(50);")
cursor.execute("ALTER TABLE canonical_matches ADD COLUMN result_recorded_at DATETIME;")

conn.commit()
print("Fixed schema.")
