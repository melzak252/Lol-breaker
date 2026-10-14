import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, UTC

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    max_days_back = 90
    cutoff = (datetime.now(UTC) - timedelta(days=max_days_back)).isoformat()
    
    print(f"Cutoff: {cutoff}")
    
    thesis_model = "Fusion-v2-SymAug"
    thesis_version = "v1.0"
    
    cur.execute("""
        SELECT COUNT(*) 
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = %s
          AND cp.model_version = %s
          AND cm.status IN ('finished', 'completed')
          AND cm.winner_side IS NOT NULL
          AND cm.start_time_normalized > %s
    """, (thesis_model, thesis_version, cutoff))
    count = cur.fetchone()['count']
    print(f"Recent finished matches with {thesis_model} ({thesis_version}) and winner_side: {count}")

    if count == 0:
        print("\nChecking without cutoff...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
            WHERE cp.model_name = %s
              AND cp.model_version = %s
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IS NOT NULL
        """, (thesis_model, thesis_version))
        count_no_cutoff = cur.fetchone()['count']
        print(f"Total finished matches with {thesis_model} ({thesis_version}) and winner_side: {count_no_cutoff}")
        
        if count_no_cutoff > 0:
            cur.execute("""
                SELECT MAX(cm.start_time_normalized) as max_date, MIN(cm.start_time_normalized) as min_date
                FROM canonical_predictions cp
                JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
                WHERE cp.model_name = %s
                  AND cp.model_version = %s
            """, (thesis_model, thesis_version))
            dates = cur.fetchone()
            print(f"Date range for {thesis_model}: {dates['min_date']} to {dates['max_date']}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_db()
