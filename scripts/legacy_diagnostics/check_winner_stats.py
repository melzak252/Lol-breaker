import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_stats():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("--- canonical_matches status/winner_side distribution ---")
    cur.execute("""
        SELECT status, 
               (winner_side IS NOT NULL) as has_winner_side,
               (winner_name IS NOT NULL) as has_winner_name,
               COUNT(*)
        FROM canonical_matches
        GROUP BY status, has_winner_side, has_winner_name
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- Checking if we can recover winner_side from winner_name ---")
    cur.execute("""
        SELECT COUNT(*)
        FROM canonical_matches
        WHERE winner_side IS NULL 
          AND winner_name IS NOT NULL
          AND (winner_name = team_a_name OR winner_name = team_b_name)
    """)
    print(f"Matches where winner_side can be recovered: {cur.fetchone()['count']}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_stats()
