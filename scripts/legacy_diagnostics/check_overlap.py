import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_overlap():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    thesis_model = "Fusion-v2-SymAug"
    thesis_version = "v1.0"
    
    print(f"--- Checking overlap for {thesis_model} ---")
    cur.execute("""
        SELECT COUNT(*) 
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = %s
          AND cp.model_version = %s
          AND cm.winner_side IS NOT NULL
    """, (thesis_model, thesis_version))
    print(f"Matches with prediction AND winner_side: {cur.fetchone()['count']}")

    print("\n--- Sample matches WITH winner_side but NO Fusion prediction ---")
    cur.execute("""
        SELECT cm.id, cm.team_a_name, cm.team_b_name, cm.winner_side, cm.start_time_normalized
        FROM canonical_matches cm
        LEFT JOIN canonical_predictions cp ON cp.canonical_match_id = cm.id 
            AND cp.model_name = %s AND cp.model_version = %s
        WHERE cm.winner_side IS NOT NULL
          AND cp.id IS NULL
        LIMIT 5
    """, (thesis_model, thesis_version))
    for row in cur.fetchall():
        print(row)

    print("\n--- Sample matches WITH Fusion prediction but NO winner_side ---")
    cur.execute("""
        SELECT cm.id, cm.team_a_name, cm.team_b_name, cm.status, cm.winner_name, cm.start_time_normalized
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = %s
          AND cp.model_version = %s
          AND cm.winner_side IS NULL
        LIMIT 5
    """, (thesis_model, thesis_version))
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_overlap()
