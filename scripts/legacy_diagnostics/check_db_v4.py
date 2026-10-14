import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    thesis_model = "Fusion-v2-SymAug"
    thesis_version = "v1.0"
    
    print(f"--- Sample predictions for {thesis_model} ---")
    cur.execute("""
        SELECT cp.id, cp.canonical_match_id, cm.status, cm.winner_side, cm.winner_name, cm.start_time_normalized
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = %s
          AND cp.model_version = %s
        LIMIT 10
    """, (thesis_model, thesis_version))
    for row in cur.fetchall():
        print(row)

    print("\n--- Winner side distribution for these matches ---")
    cur.execute("""
        SELECT cm.winner_side, COUNT(*)
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = %s
          AND cp.model_version = %s
        GROUP BY cm.winner_side
    """, (thesis_model, thesis_version))
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_db()
