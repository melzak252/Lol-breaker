
from betting_app.core.db import connect
import os

def check():
    print(f"DATABASE_URL: {os.environ.get('DATABASE_URL')}")
    with connect() as conn:
        row = conn.execute("SELECT MAX(date) as max_date, COUNT(*) as total FROM golgg_matches").fetchone()
        print(f"Max date in golgg_matches: {row['max_date']}")
        print(f"Total matches in golgg_matches: {row['total']}")
        
        recent = conn.execute("SELECT match_id, date, team1_name, team2_name FROM golgg_matches ORDER BY date DESC LIMIT 5").fetchall()
        for r in recent:
            print(f"Match: {r['match_id']} | Date: {r['date']} | {r['team1_name']} vs {r['team2_name']}")

if __name__ == "__main__":
    check()
