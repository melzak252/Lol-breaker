
import os
from dotenv import load_dotenv
load_dotenv()

from betting_app.core.db import connect

def check():
    with connect() as conn:
        # Check latest completed run
        run = conn.execute("""
            SELECT id, ratings_version, status, data_cutoff_at, finished_at 
            FROM rating_runs 
            WHERE ratings_version = 'latest-full' 
            ORDER BY finished_at DESC NULLS LAST, id DESC 
            LIMIT 1
        """).fetchone()
        print(f"Latest run: {dict(run) if run else 'None'}")
        
        # Check max snapshot_at in entity_ratings
        snapshot = conn.execute("SELECT MAX(snapshot_at) as max_snapshot FROM entity_ratings WHERE ratings_version = 'latest-full'").fetchone()
        print(f"Max snapshot_at in entity_ratings: {snapshot['max_snapshot']}")
        
        # Check number of processed matches in ledger
        ledger_count = conn.execute("SELECT COUNT(*) as n FROM rating_processed_matches WHERE ratings_version = 'latest-full'").fetchone()
        print(f"Total matches in ledger: {ledger_count['n']}")

if __name__ == "__main__":
    check()
