import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("--- Model Counts in canonical_predictions ---")
    cur.execute("""
        SELECT model_name, model_version, COUNT(*) 
        FROM canonical_predictions 
        GROUP BY model_name, model_version 
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"{row['model_name']} ({row['model_version']}): {row['count']}")
        
    print("\n--- Finished Matches with Predictions ---")
    cur.execute("""
        SELECT cp.model_name, cp.model_version, COUNT(*) 
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cm.status IN ('finished', 'completed')
        GROUP BY cp.model_name, cp.model_version
        ORDER BY count DESC
    """)
    for row in cur.fetchall():
        print(f"{row['model_name']} ({row['model_version']}): {row['count']}")

    print("\n--- Recent Finished Matches ---")
    cur.execute("""
        SELECT id, canonical_key, start_time_normalized, status, winner_side
        FROM canonical_matches
        WHERE status IN ('finished', 'completed')
        ORDER BY start_time_normalized DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_db()
