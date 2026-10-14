import psycopg2
import os
import time

def cleanup():
    url = os.environ['DATABASE_URL'].replace('+psycopg2', '')
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    # Get IDs of matches to delete
    cur.execute("SELECT id FROM canonical_matches WHERE status = 'completed'")
    match_ids = [r[0] for r in cur.fetchall()]
    print(f'Found {len(match_ids)} matches to delete.')
    
    if not match_ids:
        print("No matches to delete.")
        return

    tables = [
        'model_ev_signals', 
        'canonical_predictions', 
        'golgg_match_mappings', 
        'upcoming_match_features', 
        'odds_snapshots', 
        'bookmaker_events', 
        'bets', 
        'upcoming_matches'
    ]
    
    batch_size = 5000
    for i in range(0, len(match_ids), batch_size):
        batch = match_ids[i:i + batch_size]
        batch_tuple = tuple(batch)
        
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} matches)...")
        
        for table in tables:
            cur.execute(f"DELETE FROM {table} WHERE canonical_match_id IN %s", (batch_tuple,))
            if cur.rowcount > 0:
                print(f"  Deleted {cur.rowcount} rows from {table}")
        
        cur.execute("DELETE FROM canonical_matches WHERE id IN %s", (batch_tuple,))
        print(f"  Deleted {cur.rowcount} rows from canonical_matches")
        
        conn.commit()
        print(f"Batch {i//batch_size + 1} committed.")

    conn.close()
    print('Cleanup successful.')

if __name__ == "__main__":
    cleanup()
