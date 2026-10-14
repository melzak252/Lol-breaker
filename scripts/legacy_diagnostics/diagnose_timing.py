import psycopg2
import os
from datetime import datetime, timedelta, UTC

DB_DSN = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db").replace("postgresql+psycopg2://", "postgresql://")

def diagnose():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    cutoff = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    print(f"Cutoff: {cutoff}")
    
    # 1. Count Fusion-v2-SymAug predictions for finished matches in last 90 days
    cur.execute("""
        SELECT count(*) 
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = 'Fusion-v2-SymAug'
          AND cp.model_version = 'v1.0'
          AND cm.status IN ('finished', 'completed')
          AND cm.winner_side IS NOT NULL
          AND cm.start_time_normalized > %s
    """, (cutoff,))
    print(f"Fusion-v2-SymAug predictions (last 90d, finished): {cur.fetchone()[0]}")
    
    # 2. Check if any of these have odds snapshots
    cur.execute("""
        SELECT count(DISTINCT cp.canonical_match_id)
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        JOIN odds_snapshots os ON os.canonical_match_id = cm.id
        WHERE cp.model_name = 'Fusion-v2-SymAug'
          AND cp.model_version = 'v1.0'
          AND cm.status IN ('finished', 'completed')
          AND cm.winner_side IS NOT NULL
          AND cm.start_time_normalized > %s
    """, (cutoff,))
    print(f"Matches with both Fusion-v2-SymAug and Odds (last 90d): {cur.fetchone()[0]}")

    # 3. Check Hybrid-Fusion-SymAug-Market predictions
    cur.execute("""
        SELECT count(*) 
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = 'Hybrid-Fusion-SymAug-Market'
          AND cm.status IN ('finished', 'completed')
          AND cm.winner_side IS NOT NULL
          AND cm.start_time_normalized > %s
    """, (cutoff,))
    print(f"Hybrid-Fusion-SymAug-Market predictions (last 90d, finished): {cur.fetchone()[0]}")

    # 4. Check if apply_temperature_probability is defined in timing.py
    # I'll just check the file content via bash later.
    
    conn.close()

if __name__ == "__main__":
    diagnose()
