import psycopg2
import os
from datetime import datetime

DB_DSN = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db").replace("postgresql+psycopg2://", "postgresql://")

def check_matching():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    # Get some completed matches
    cur.execute("""
        SELECT id, team_a_name, team_b_name, start_time_normalized 
        FROM canonical_matches 
        WHERE status = 'completed' 
        LIMIT 5
    """)
    matches = cur.fetchall()
    
    for m_id, t1, t2, dt in matches:
        print(f"Matching: {t1} vs {t2} on {dt}")
        # Try to find in golgg_matches
        # golgg_matches has team1_name, team2_name, date
        cur.execute("""
            SELECT team1_name, team2_name, team1_win, date 
            FROM golgg_matches 
            WHERE (
                (LOWER(team1_name) = LOWER(%s) AND LOWER(team2_name) = LOWER(%s))
                OR
                (LOWER(team1_name) = LOWER(%s) AND LOWER(team2_name) = LOWER(%s))
            )
            AND date::date = %s::date
        """, (t1, t2, t2, t1, dt.date()))
        
        results = cur.fetchall()
        if results:
            for g_t1, g_t2, g_win, g_date in results:
                winner = "team_a" if (LOWER(g_t1) == LOWER(t1) and g_win == 1) or (LOWER(g_t2) == LOWER(t1) and g_win == 0) else "team_b"
                print(f"  FOUND: {g_t1} vs {g_t2}, win={g_win}, date={g_date} -> Winner: {winner}")
        else:
            print("  NOT FOUND")
            
    conn.close()

def LOWER(s):
    return s.lower() if s else ""

if __name__ == "__main__":
    check_matching()
