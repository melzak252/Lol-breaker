import psycopg2
import os
from datetime import datetime

DB_DSN = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db").replace("postgresql+psycopg2://", "postgresql://")

def backfill():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    print("Fetching completed matches with NULL winner_side...")
    cur.execute("""
        SELECT id, team_a_name, team_b_name, start_time_normalized 
        FROM canonical_matches 
        WHERE status = 'completed' AND winner_side IS NULL
    """)
    matches = cur.fetchall()
    print(f"Found {len(matches)} matches to process.")
    
    updated_count = 0
    not_found_count = 0
    
    for m_id, t_a, t_b, dt in matches:
        # Try to find in golgg_matches
        cur.execute("""
            SELECT team1_name, team2_name, team1_win 
            FROM golgg_matches 
            WHERE (
                (LOWER(team1_name) = LOWER(%s) AND LOWER(team2_name) = LOWER(%s))
                OR
                (LOWER(team1_name) = LOWER(%s) AND LOWER(team2_name) = LOWER(%s))
            )
            AND date::date = %s::date
        """, (t_a, t_b, t_b, t_a, dt.date()))
        
        results = cur.fetchall()
        if results:
            # Take the first match (usually there's only one per day)
            g_t1, g_t2, g_win = results[0]
            
            winner_side = None
            if g_win == 1: # Team 1 won
                if g_t1.lower() == t_a.lower():
                    winner_side = 'team_a'
                else:
                    winner_side = 'team_b'
            elif g_win == 0: # Team 2 won
                if g_t2.lower() == t_a.lower():
                    winner_side = 'team_a'
                else:
                    winner_side = 'team_b'
            
            if winner_side:
                cur.execute("UPDATE canonical_matches SET winner_side = %s WHERE id = %s", (winner_side, m_id))
                updated_count += 1
                if updated_count % 100 == 0:
                    print(f"Updated {updated_count} matches...")
                    conn.commit()
        else:
            not_found_count += 1
            
    conn.commit()
    print(f"Finished. Updated: {updated_count}, Not found: {not_found_count}")
    conn.close()

if __name__ == "__main__":
    backfill()
