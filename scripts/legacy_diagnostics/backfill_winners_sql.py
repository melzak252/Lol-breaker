import psycopg2
import os

DB_DSN = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db").replace("postgresql+psycopg2://", "postgresql://")

def backfill_sql():
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    sql = """
    UPDATE canonical_matches cm
    SET winner_side = CASE 
        WHEN gm.team1_win = 1 AND LOWER(gm.team1_name) = LOWER(cm.team_a_name) THEN 'team_a'
        WHEN gm.team1_win = 1 AND LOWER(gm.team1_name) = LOWER(cm.team_b_name) THEN 'team_b'
        WHEN gm.team1_win = 0 AND LOWER(gm.team2_name) = LOWER(cm.team_a_name) THEN 'team_a'
        WHEN gm.team1_win = 0 AND LOWER(gm.team2_name) = LOWER(cm.team_b_name) THEN 'team_b'
    END
    FROM golgg_matches gm
    WHERE cm.status = 'completed' 
      AND cm.winner_side IS NULL
      AND gm.date::date = cm.start_time_normalized::date
      AND (
        (LOWER(gm.team1_name) = LOWER(cm.team_a_name) AND LOWER(gm.team2_name) = LOWER(cm.team_b_name))
        OR
        (LOWER(gm.team1_name) = LOWER(cm.team_b_name) AND LOWER(gm.team2_name) = LOWER(cm.team_a_name))
      );
    """
    
    print("Executing backfill SQL...")
    cur.execute(sql)
    updated = cur.rowcount
    conn.commit()
    print(f"Updated {updated} matches.")
    
    # Also update winner_name if it's NULL
    sql_name = """
    UPDATE canonical_matches cm
    SET winner_name = CASE 
        WHEN winner_side = 'team_a' THEN team_a_name
        WHEN winner_side = 'team_b' THEN team_b_name
    END
    WHERE status = 'completed' AND winner_name IS NULL AND winner_side IS NOT NULL;
    """
    cur.execute(sql_name)
    updated_names = cur.rowcount
    conn.commit()
    print(f"Updated {updated_names} winner names.")
    
    conn.close()

if __name__ == "__main__":
    backfill_sql()
