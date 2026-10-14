import psycopg2
import os
from datetime import datetime, timedelta

DB_DSN = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db").replace("postgresql+psycopg2://", "postgresql://")

def debug_matching():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, team_a_name, team_b_name, start_time_normalized 
        FROM canonical_matches 
        WHERE status = 'completed' AND winner_side IS NULL
        LIMIT 10
    """)
    matches = cur.fetchall()
    
    for m_id, t_a, t_b, dt in matches:
        print(f"\nCanonical: {t_a} vs {t_b} | Date: {dt}")
        
        # Search with +/- 1 day window
        d_start = dt.date() - timedelta(days=1)
        d_end = dt.date() + timedelta(days=1)
        
        cur.execute("""
            SELECT team1_name, team2_name, date, team1_win
            FROM golgg_matches 
            WHERE date::date >= %s AND date::date <= %s
            AND (
                LOWER(team1_name) LIKE LOWER(%s) OR LOWER(team2_name) LIKE LOWER(%s)
                OR LOWER(team1_name) LIKE LOWER(%s) OR LOWER(team2_name) LIKE LOWER(%s)
            )
        """, (d_start, d_end, f"%{t_a[:4]}%", f"%{t_a[:4]}%", f"%{t_b[:4]}%", f"%{t_b[:4]}%"))
        
        results = cur.fetchall()
        print(f"  Found {len(results)} potential matches in GOL.GG:")
        for g1, g2, gd, gwin in results:
            print(f"    - {g1} vs {g2} | Date: {gd} | Win: {gwin}")
            
    conn.close()

if __name__ == "__main__":
    debug_matching()
