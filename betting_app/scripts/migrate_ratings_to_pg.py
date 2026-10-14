"""
Migrate rating_runs and entity_ratings from SQLite to PostgreSQL.

This script properly handles:
- rating_runs: migrates all runs with their IDs
- entity_ratings: migrates with rating_run_id FK and snapshot_at for TimescaleDB partitioning
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


def migrate_rating_runs(pg_session: Session, sqlite_db: str) -> dict[int, int]:
    """Migrate rating_runs and return mapping of old_id -> new_id."""
    src = sqlite3.connect(sqlite_db)
    src.row_factory = sqlite3.Row
    
    try:
        cursor = src.execute("""
            SELECT id, ratings_version, source, data_cutoff_at, started_at, finished_at, 
                   status, systems_json, matches_processed, games_processed, players_processed, error
            FROM rating_runs
            ORDER BY id
        """)
        rows = cursor.fetchall()
    except Exception as e:
        log(f"ERROR reading rating_runs from SQLite: {e}")
        src.close()
        return {}
    
    if not rows:
        log("No rating_runs found in SQLite")
        src.close()
        return {}
    
    log(f"Migrating {len(rows)} rating_runs...")
    id_mapping = {}
    
    for row in rows:
        old_id = row['id']
        
        # Check if this version already exists in PG
        pg_session.begin()
        try:
            existing = pg_session.execute(
                text("SELECT id FROM rating_runs WHERE ratings_version = :version"),
                {"version": row['ratings_version']}
            ).fetchone()
            
            if existing:
                new_id = existing[0]
                log(f"  SKIP rating_runs id={old_id} version={row['ratings_version']} (already exists as id={new_id})")
                id_mapping[old_id] = new_id
                pg_session.rollback()
                continue
            
            # Insert new rating_run
            pg_session.execute(
                text("""
                    INSERT INTO rating_runs (
                        ratings_version, source, data_cutoff_at, started_at, finished_at,
                        status, systems_json, matches_processed, games_processed, players_processed, error
                    ) VALUES (
                        :ratings_version, :source, :data_cutoff_at, :started_at, :finished_at,
                        :status, :systems_json, :matches_processed, :games_processed, :players_processed, :error
                    )
                    RETURNING id
                """),
                {
                    "ratings_version": row['ratings_version'],
                    "source": row['source'],
                    "data_cutoff_at": row['data_cutoff_at'],
                    "started_at": row['started_at'],
                    "finished_at": row['finished_at'],
                    "status": row['status'],
                    "systems_json": row['systems_json'],
                    "matches_processed": row['matches_processed'],
                    "games_processed": row['games_processed'],
                    "players_processed": row['players_processed'],
                    "error": row['error'],
                }
            )
            result = pg_session.execute(text("SELECT LASTVAL()")).scalar()
            new_id = result
            id_mapping[old_id] = new_id
            pg_session.commit()
            log(f"  Migrated rating_runs id={old_id} -> {new_id} (version={row['ratings_version']})")
            
        except Exception as e:
            pg_session.rollback()
            log(f"  ERROR migrating rating_runs id={old_id}: {e}")
    
    src.close()
    return id_mapping


def migrate_entity_ratings(pg_session: Session, sqlite_db: str, id_mapping: dict[int, int]) -> int:
    """Migrate entity_ratings with proper rating_run_id mapping and snapshot_at."""
    src = sqlite3.connect(sqlite_db)
    src.row_factory = sqlite3.Row
    
    try:
        cursor = src.execute("""
            SELECT rating_run_id, ratings_version, snapshot_at, entity_type, entity_name,
                   normalized_entity_name, team_name, role, rating_system, rating_value,
                   rd, sigma, games_played, last_match_at, state_json
            FROM entity_ratings
            ORDER BY rating_run_id, id
        """)
        rows = cursor.fetchall()
    except Exception as e:
        log(f"ERROR reading entity_ratings from SQLite: {e}")
        src.close()
        return 0
    
    if not rows:
        log("No entity_ratings found in SQLite")
        src.close()
        return 0
    
    log(f"Migrating {len(rows)} entity_ratings...")
    migrated = 0
    batch_size = 1000
    batch = []
    
    for row in rows:
        old_run_id = row['rating_run_id']
        new_run_id = id_mapping.get(old_run_id)
        
        if new_run_id is None:
            log(f"  SKIP entity_rating: no mapping for rating_run_id={old_run_id}")
            continue
        
        batch.append({
            "rating_run_id": new_run_id,
            "ratings_version": row['ratings_version'],
            "snapshot_at": row['snapshot_at'],
            "entity_type": row['entity_type'],
            "entity_name": row['entity_name'],
            "normalized_entity_name": row['normalized_entity_name'] or row['entity_name'].lower(),
            "team_name": row['team_name'],
            "role": row['role'],
            "rating_system": row['rating_system'],
            "rating_value": row['rating_value'],
            "rd": row['rd'],
            "sigma": row['sigma'],
            "games_played": row['games_played'],
            "last_match_at": row['last_match_at'],
            "state_json": row['state_json'],
        })
        
        if len(batch) >= batch_size:
            pg_session.begin()
            try:
                for params in batch:
                    pg_session.execute(
                        text("""
                            INSERT INTO entity_ratings (
                                rating_run_id, ratings_version, snapshot_at, entity_type, entity_name,
                                normalized_entity_name, team_name, role, rating_system, rating_value,
                                rd, sigma, games_played, last_match_at, state_json
                            ) VALUES (
                                :rating_run_id, :ratings_version, :snapshot_at, :entity_type, :entity_name,
                                :normalized_entity_name, :team_name, :role, :rating_system, :rating_value,
                                :rd, :sigma, :games_played, :last_match_at, :state_json
                            )
                        """),
                        params
                    )
                pg_session.commit()
                migrated += len(batch)
                log(f"  {migrated}/{len(rows)}")
            except Exception as e:
                pg_session.rollback()
                log(f"  ERROR batch: {e}")
            batch = []
    
    # Process remaining batch
    if batch:
        pg_session.begin()
        try:
            for params in batch:
                pg_session.execute(
                    text("""
                        INSERT INTO entity_ratings (
                            rating_run_id, ratings_version, snapshot_at, entity_type, entity_name,
                            normalized_entity_name, team_name, role, rating_system, rating_value,
                            rd, sigma, games_played, last_match_at, state_json
                        ) VALUES (
                            :rating_run_id, :ratings_version, :snapshot_at, :entity_type, :entity_name,
                            :normalized_entity_name, :team_name, :role, :rating_system, :rating_value,
                            :rd, :sigma, :games_played, :last_match_at, :state_json
                        )
                    """),
                    params
                )
            pg_session.commit()
            migrated += len(batch)
            log(f"  {migrated}/{len(rows)}")
        except Exception as e:
            pg_session.rollback()
            log(f"  ERROR final batch: {e}")
    
    src.close()
    return migrated


def main() -> None:
    pg_url = os.environ.get("DATABASE_URL", "")
    if not pg_url:
        log("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    sqlite_db = sqlite_path()
    if not sqlite_db or not sqlite_db.exists():
        log(f"ERROR: SQLite file not found at {sqlite_db}")
        sys.exit(1)
    
    log(f"Source SQLite:  {sqlite_db}")
    log(f"Target PG/TimescaleDB: {pg_url}")
    log("")
    
    engine = create_engine(pg_url, pool_pre_ping=True)
    with Session(engine) as session:
        # Migrate rating_runs first
        id_mapping = migrate_rating_runs(session, str(sqlite_db))
        log("")
        
        # Migrate entity_ratings with ID mapping
        migrated = migrate_entity_ratings(session, str(sqlite_db), id_mapping)
        log(f"\nMigrated {migrated} entity_ratings")
    
    log("Migration complete.")
    engine.dispose()


if __name__ == "__main__":
    main()
