
from betting_app.core.db import connect

def check():
    with connect() as conn:
        # Check cutoff in rating_runs
        run = conn.execute("SELECT data_cutoff_at FROM rating_runs WHERE ratings_version = 'latest-full' AND status = 'completed' ORDER BY finished_at DESC LIMIT 1").fetchone()
        print(f"Cutoff in rating_runs: {run['data_cutoff_at']}")
        
        # Check matches in golgg_matches after cutoff
        matches = conn.execute("SELECT COUNT(*) as n FROM golgg_matches WHERE date > ?", (run['data_cutoff_at'],)).fetchone()
        print(f"Matches in golgg_matches after cutoff: {matches['n']}")
        
        # Check matches in rating_processed_matches after cutoff
        processed = conn.execute("SELECT COUNT(*) as n FROM rating_processed_matches WHERE ratings_version = 'latest-full' AND match_date > ?", (run['data_cutoff_at'],)).fetchone()
        print(f"Matches in rating_processed_matches after cutoff: {processed['n']}")
        
        # Check matches in golgg_matches ON cutoff date that are NOT in rating_processed_matches
        on_cutoff = conn.execute("""
            SELECT m.match_id 
            FROM golgg_matches m
            LEFT JOIN rating_processed_matches rpm ON m.match_id = rpm.match_id AND rpm.ratings_version = 'latest-full'
            WHERE m.date = ? AND rpm.match_id IS NULL
        """, (run['data_cutoff_at'],)).fetchall()
        print(f"Matches on cutoff date NOT in ledger: {len(on_cutoff)}")

        # Check matches AFTER cutoff date that are NOT in rating_processed_matches
        after_cutoff = conn.execute("""
            SELECT m.match_id, m.date
            FROM golgg_matches m
            LEFT JOIN rating_processed_matches rpm ON m.match_id = rpm.match_id AND rpm.ratings_version = 'latest-full'
            WHERE m.date > ? AND rpm.match_id IS NULL
        """, (run['data_cutoff_at'],)).fetchall()
        print(f"Matches after cutoff date NOT in ledger: {len(after_cutoff)}")
        if after_cutoff:
            print(f"First few: {after_cutoff[:5]}")

if __name__ == "__main__":
    check()
