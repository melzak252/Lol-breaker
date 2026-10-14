import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_matches():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    thesis_model = "Fusion-v2-SymAug"
    thesis_version = "v1.0"
    
    print(f"--- Checking matches for {thesis_model} ---")
    cur.execute("""
        SELECT cm.id, cm.status, cm.winner_name, cm.winner_side, cm.winner_normalized, cm.team_a_name, cm.team_b_name
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = %s
          AND cp.model_version = %s
          AND cm.status IN ('finished', 'completed')
        LIMIT 20
    """, (thesis_model, thesis_version))
    
    rows = cur.fetchall()
    for row in rows:
        print(row)

    if rows:
        print("\n--- Checking if winner_name matches team_a or team_b ---")
        for row in rows:
            w_name = row['winner_name']
            t_a = row['team_a_name']
            t_b = row['team_b_name']
            match = "None"
            if w_name:
                if w_name == t_a: match = "team_a"
                elif w_name == t_b: match = "team_b"
            print(f"ID {row['id']}: Winner='{w_name}', A='{t_a}', B='{t_b}' -> Match: {match}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_matches()
