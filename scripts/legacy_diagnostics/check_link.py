import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_link():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("--- Checking result_source_match_id vs golgg_matches ---")
    cur.execute("""
        SELECT cm.id, cm.result_source, cm.result_source_match_id, gm.match_id as golgg_id
        FROM canonical_matches cm
        JOIN golgg_matches gm ON cm.result_source_match_id = gm.match_id
        WHERE cm.status = 'completed'
        LIMIT 10
    """)
    for row in cur.fetchall():
        print(row)

    cur.execute("""
        SELECT COUNT(*)
        FROM canonical_matches cm
        JOIN golgg_matches gm ON cm.result_source_match_id = gm.match_id
        WHERE cm.status = 'completed'
    """)
    print(f"Matches linked via result_source_match_id: {cur.fetchone()['count']}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_link()
