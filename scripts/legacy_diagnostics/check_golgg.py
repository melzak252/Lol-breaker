import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://admin:admin@localhost:5432/betting_db")
if "postgresql+psycopg2://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")

def check_golgg():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("--- golgg_matches winner info ---")
    cur.execute("""
        SELECT team1_win, COUNT(*) 
        FROM golgg_matches 
        GROUP BY team1_win
    """)
    for row in cur.fetchall():
        print(row)

    print("\n--- Sample golgg_matches ---")
    cur.execute("""
        SELECT match_id, team1_name, team2_name, team1_win, date
        FROM golgg_matches
        WHERE team1_win IS NOT NULL
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(row)

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_golgg()
