import os
from sqlalchemy import create_engine, text
from datetime import datetime, UTC, timedelta

def check_filtering():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set")
        return
    
    engine = create_engine(url)
    now = datetime.now(UTC)
    stale_hours = 6
    stale_cutoff = now - timedelta(hours=stale_hours)
    
    ids = [63291, 64185, 63290, 65577]
    
    with engine.connect() as conn:
        for mid in ids:
            print(f"\n--- Checking CM {mid} ---")
            # 1. Canonical Match basic info
            cm = conn.execute(text("SELECT status, start_time_normalized FROM canonical_matches WHERE id = :id"), {"id": mid}).fetchone()
            if not cm:
                print("Not found in canonical_matches")
                continue
            print(f"Status: {cm.status}, Start Time: {cm.start_time_normalized}")
            
            # 2. upcoming_matches (seen_matches CTE)
            um = conn.execute(text("SELECT last_seen_at FROM upcoming_matches WHERE canonical_match_id = :id"), {"id": mid}).fetchall()
            print(f"upcoming_matches entries: {len(um)}")
            for r in um:
                print(f"  last_seen_at: {r.last_seen_at} (Stale? {r.last_seen_at < stale_cutoff if r.last_seen_at else 'N/A'})")
            
            # 3. odds_snapshots (latest CTE)
            odds = conn.execute(text("""
                SELECT bookmaker_id, market_type, is_live, scraped_at 
                FROM odds_snapshots 
                WHERE canonical_match_id = :id 
                  AND market_type='match_winner' 
                  AND COALESCE(is_live,0)=0
            """), {"id": mid}).fetchall()
            print(f"odds_snapshots (match_winner, pre-match): {len(odds)}")
            for r in odds:
                print(f"  Bookmaker: {r.bookmaker_id}, Scraped: {r.scraped_at}")

if __name__ == "__main__":
    check_filtering()
