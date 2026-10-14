"""
Migrate all data from SQLite to PostgreSQL for local development.
"""

import os
import sys
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

def log(msg: str) -> None:
    print(f"[MIGRATE] {msg}", flush=True)

def main():
    sqlite_path = "/app/data/betting_app.sqlite3"
    pg_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://betting:betting_local_password@timescaledb:5432/betting")
    
    log(f"Source: {sqlite_path}")
    log(f"Target: {pg_url}")
    
    if not os.path.exists(sqlite_path):
        log(f"ERROR: SQLite file not found at {sqlite_path}")
        sys.exit(1)
    
    # Connect to SQLite
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    
    # Connect to PostgreSQL
    engine = create_engine(pg_url, pool_pre_ping=True)
    session = Session(engine)
    
    # Disable FK triggers
    session.execute(text("SET session_replication_role = 'replica';"))
    session.commit()
    
    # Migrate bookmakers (don't use SQLite IDs, let PG generate them)
    log("Migrating bookmakers...")
    cursor = src.execute("SELECT id, name, base_url FROM bookmakers")
    rows = cursor.fetchall()
    sqlite_to_pg_bookmaker_map = {}  # SQLite ID -> PG ID
    
    for row in rows:
        # Insert without ID, let PG generate it
        result = session.execute(
            text("INSERT INTO bookmakers (name, base_url) VALUES (:name, :base_url) ON CONFLICT (name) DO UPDATE SET base_url = EXCLUDED.base_url RETURNING id"),
            {"name": row["name"], "base_url": row["base_url"]}
        )
        pg_id = result.fetchone()[0]
        sqlite_to_pg_bookmaker_map[row["id"]] = pg_id
    
    session.commit()
    log(f"  {len(rows)} bookmakers")
    log(f"  Bookmaker ID mapping: {sqlite_to_pg_bookmaker_map}")
    
    # Migrate golgg_teams
    log("Migrating golgg_teams...")
    cursor = src.execute("SELECT id, team_name, normalized_name FROM golgg_teams")
    rows = cursor.fetchall()
    count = 0
    for row in rows:
        session.execute(
            text("INSERT INTO golgg_teams (id, team_name, normalized_name) VALUES (:id, :team_name, :normalized_name) ON CONFLICT (id) DO NOTHING"),
            {"id": row["id"], "team_name": row["team_name"], "normalized_name": row["normalized_name"]}
        )
        count += 1
        if count % 500 == 0:
            session.commit()
    session.commit()
    log(f"  {len(rows)} golgg_teams")
    
    # Migrate canonical_matches
    log("Migrating canonical_matches...")
    cursor = src.execute("""
        SELECT id, canonical_key, team_a_name, team_b_name, normalized_team_a, 
               normalized_team_b, start_time_normalized, league, status, match_confidence
        FROM canonical_matches
    """)
    rows = cursor.fetchall()
    for row in rows:
        session.execute(
            text("""
                INSERT INTO canonical_matches 
                (id, canonical_key, team_a_name, team_b_name, normalized_team_a, 
                 normalized_team_b, start_time_normalized, league, status, match_confidence)
                VALUES (:id, :canonical_key, :team_a_name, :team_b_name, :normalized_team_a,
                        :normalized_team_b, :start_time_normalized, :league, :status, :match_confidence)
                ON CONFLICT (id) DO NOTHING
            """),
            dict(row)
        )
    session.commit()
    log(f"  {len(rows)} canonical_matches")
    
    # Migrate upcoming_matches
    log("Migrating upcoming_matches...")
    cursor = src.execute("""
        SELECT id, canonical_team_a, canonical_team_b, raw_team_a, raw_team_b,
               match_start_time, league, bookmaker_match_key, offer_url, canonical_match_id
        FROM upcoming_matches
    """)
    rows = cursor.fetchall()
    
    count = 0
    skipped = 0
    for row in rows:
        key_parts = row["bookmaker_match_key"].split("|")
        bookmaker_name = key_parts[0] if key_parts else "unknown"
        
        if bookmaker_name == "manual":
            skipped += 1
            continue
        
        # Find SQLite bookmaker ID for this name
        sqlite_bm_cursor = src.execute("SELECT id FROM bookmakers WHERE name = ?", (bookmaker_name,))
        sqlite_bm_row = sqlite_bm_cursor.fetchone()
        if not sqlite_bm_row:
            skipped += 1
            continue
        
        bookmaker_id = sqlite_to_pg_bookmaker_map.get(sqlite_bm_row["id"])
        if not bookmaker_id:
            skipped += 1
            continue
        
        normalized_a = row["canonical_team_a"] or row["raw_team_a"]
        normalized_b = row["canonical_team_b"] or row["raw_team_b"]
        
        try:
            session.execute(
                text("""
                    INSERT INTO upcoming_matches 
                    (bookmaker_id, bookmaker_match_key, canonical_match_id, raw_team_a, raw_team_b,
                     normalized_team_a, normalized_team_b, match_start_time, league, offer_url, is_live)
                    VALUES (:bookmaker_id, :bookmaker_match_key, :canonical_match_id, :raw_team_a, :raw_team_b,
                            :normalized_team_a, :normalized_team_b, :match_start_time, :league, :offer_url, 0)
                    ON CONFLICT (bookmaker_match_key) DO NOTHING
                """),
                {
                    "bookmaker_id": bookmaker_id,
                    "bookmaker_match_key": row["bookmaker_match_key"],
                    "canonical_match_id": row["canonical_match_id"],
                    "raw_team_a": row["raw_team_a"],
                    "raw_team_b": row["raw_team_b"],
                    "normalized_team_a": normalized_a,
                    "normalized_team_b": normalized_b,
                    "match_start_time": row["match_start_time"],
                    "league": row["league"],
                    "offer_url": row["offer_url"],
                }
            )
            count += 1
        except Exception as e:
            log(f"  ERROR: {e}")
            session.rollback()
            continue
        
        if count % 100 == 0:
            session.commit()
    
    session.commit()
    log(f"  {count} upcoming_matches ({skipped} skipped)")
    
    # Migrate odds_snapshots
    log("Migrating odds_snapshots...")
    cursor = src.execute("""
        SELECT id, bookmaker_id, canonical_match_id, market_type, raw_team_a, raw_team_b,
               odds_a, odds_b, is_live, scraped_at, source_url, offer_url
        FROM odds_snapshots
        WHERE bookmaker_id NOT IN (SELECT id FROM bookmakers WHERE name = 'manual')
    """)
    rows = cursor.fetchall()
    
    count = 0
    for row in rows:
        # Map SQLite bookmaker_id to PG bookmaker_id
        pg_bookmaker_id = sqlite_to_pg_bookmaker_map.get(row["bookmaker_id"])
        if not pg_bookmaker_id:
            continue
        
        try:
            session.execute(
                text("""
                    INSERT INTO odds_snapshots
                    (bookmaker_id, canonical_match_id, market_type, raw_team_a, raw_team_b,
                     odds_a, odds_b, is_live, scraped_at, source_url, offer_url)
                    VALUES (:bookmaker_id, :canonical_match_id, :market_type, :raw_team_a, :raw_team_b,
                            :odds_a, :odds_b, :is_live, :scraped_at, :source_url, :offer_url)
                """),
                {
                    "bookmaker_id": pg_bookmaker_id,
                    "canonical_match_id": row["canonical_match_id"],
                    "market_type": row["market_type"],
                    "raw_team_a": row["raw_team_a"],
                    "raw_team_b": row["raw_team_b"],
                    "odds_a": row["odds_a"],
                    "odds_b": row["odds_b"],
                    "is_live": row["is_live"],
                    "scraped_at": row["scraped_at"],
                    "source_url": row["source_url"],
                    "offer_url": row["offer_url"],
                }
            )
            count += 1
        except Exception as e:
            session.rollback()
            continue
        
        if count % 500 == 0:
            session.commit()
            log(f"  {count}/{len(rows)}")
    
    session.commit()
    log(f"  {count} odds_snapshots")
    
    # Re-enable FK triggers
    session.execute(text("SET session_replication_role = 'origin';"))
    session.commit()
    
    src.close()
    session.close()
    engine.dispose()
    
    log("Migration complete!")

if __name__ == "__main__":
    main()
