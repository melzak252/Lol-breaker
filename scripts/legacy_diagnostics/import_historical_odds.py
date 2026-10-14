import pandas as pd
import psycopg2
import os
from datetime import datetime, timezone

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://betting:betting_local_password@timescaledb:5432/betting"
).replace("postgresql+psycopg2://", "postgresql://")

CSV_PATH = "/app/data/odds.csv"

def import_historical_odds():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}")
        return

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # 1. Get mapping of canonical_key -> id
    cur.execute("SELECT id, canonical_key FROM canonical_matches WHERE canonical_key LIKE 'golgg:%'")
    key_to_id = {row[1]: row[0] for row in cur.fetchall()}
    print(f"Found {len(key_to_id)} golgg matches in DB")

    inserted = 0
    skipped = 0

    print("Inserting odds...")
    for _, row in df.iterrows():
        golgg_id = row['golgg_match_id']
        key = f"golgg:{golgg_id}"
        
        if key not in key_to_id:
            skipped += 1
            continue
            
        cm_id = key_to_id[key]
        
        # Use avg_odds_home/away
        odds_a = row['avg_odds_home']
        odds_b = row['avg_odds_away']
        
        if pd.isna(odds_a) or pd.isna(odds_b):
            continue
            
        # Date
        date_str = row['odds_date']
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except:
            dt = datetime.now(timezone.utc)

        # Insert into odds_snapshots
        # bookmaker_id=0 for historical average
        cur.execute("""
            INSERT INTO odds_snapshots 
            (bookmaker_id, canonical_match_id, market_type, odds_a, odds_b, is_live, scraped_at, raw_payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (0, cm_id, 'match_winner', float(odds_a), float(odds_b), 0, dt, '{"source": "historical_csv"}'))
        
        inserted += 1
        if inserted % 500 == 0:
            print(f"  Inserted {inserted}...")

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done! Inserted: {inserted}, Skipped (no match in DB): {skipped}")

if __name__ == "__main__":
    import_historical_odds()
