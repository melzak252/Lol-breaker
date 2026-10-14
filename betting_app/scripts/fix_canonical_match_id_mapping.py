"""Fix canonical_match_id mapping in upcoming_matches and odds_snapshots.

After migration, canonical_match_id references SQLite IDs (2635-2767) 
but should reference PostgreSQL IDs (1-257). This script builds a mapping
based on canonical_key and updates the foreign keys.
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from betting_app.core.db import get_db_path as sqlite_path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fix_canonical_match_id_mapping() -> None:
    """Build SQLite→PG canonical_match_id mapping and update foreign keys."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        log("ERROR: DATABASE_URL not set")
        sys.exit(1)

    sqlite_db = sqlite_path()
    if not sqlite_db or not sqlite_db.exists():
        log(f"ERROR: SQLite file not found at {sqlite_db}")
        sys.exit(1)

    log(f"SQLite: {sqlite_db}")
    log(f"PG: {database_url}")

    # Load SQLite canonical_matches (id, canonical_key)
    log("Loading SQLite canonical_matches...")
    src = sqlite3.connect(str(sqlite_db))
    src.row_factory = sqlite3.Row
    sqlite_map = {}  # canonical_key → sqlite_id
    try:
        cursor = src.execute("SELECT id, canonical_key FROM canonical_matches")
        for row in cursor:
            sqlite_map[row["canonical_key"]] = row["id"]
    finally:
        src.close()
    
    log(f"  Loaded {len(sqlite_map)} SQLite canonical_matches")

    # Connect to PG and build sqlite_id → pg_id mapping
    log("Building SQLite→PG ID mapping...")
    engine = create_engine(database_url)
    with Session(engine) as session:
        # Load PG canonical_matches
        pg_map = {}  # canonical_key → pg_id
        result = session.execute(text("SELECT id, canonical_key FROM canonical_matches"))
        for row in result:
            pg_map[row[1]] = row[0]
        
        log(f"  Loaded {len(pg_map)} PG canonical_matches")

        # Build sqlite_id → pg_id mapping
        id_mapping = {}  # sqlite_id → pg_id
        for canonical_key, sqlite_id in sqlite_map.items():
            pg_id = pg_map.get(canonical_key)
            if pg_id:
                id_mapping[sqlite_id] = pg_id
        
        log(f"  Built mapping for {len(id_mapping)} canonical_matches")

        if not id_mapping:
            log("ERROR: No mapping built - check canonical_key format")
            sys.exit(1)

        # Update upcoming_matches.canonical_match_id
        log("Updating upcoming_matches.canonical_match_id...")
        updated_upcoming = 0
        for sqlite_id, pg_id in id_mapping.items():
            result = session.execute(
                text("UPDATE upcoming_matches SET canonical_match_id = :pg_id WHERE canonical_match_id = :sqlite_id"),
                {"pg_id": pg_id, "sqlite_id": sqlite_id}
            )
            updated_upcoming += result.rowcount
        
        log(f"  Updated {updated_upcoming} upcoming_matches rows")

        # Update odds_snapshots.canonical_match_id
        log("Updating odds_snapshots.canonical_match_id...")
        updated_odds = 0
        for sqlite_id, pg_id in id_mapping.items():
            result = session.execute(
                text("UPDATE odds_snapshots SET canonical_match_id = :pg_id WHERE canonical_match_id = :sqlite_id"),
                {"pg_id": pg_id, "sqlite_id": sqlite_id}
            )
            updated_odds += result.rowcount
        
        log(f"  Updated {updated_odds} odds_snapshots rows")

        session.commit()
        log("✓ Mapping fix completed")

        # Verify
        result = session.execute(text("""
            SELECT COUNT(DISTINCT cm.id) as canonical_with_odds 
            FROM canonical_matches cm 
            JOIN upcoming_matches um ON um.canonical_match_id = cm.id 
            WHERE cm.status = 'upcoming'
        """))
        row = result.fetchone()
        log(f"  Canonical matches with upcoming_matches: {row[0]}")

    engine.dispose()


if __name__ == "__main__":
    fix_canonical_match_id_mapping()
