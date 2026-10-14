#!/usr/bin/env python3
"""Migrate entity_ratings from SQLite to PostgreSQL with correct rating_run_id mapping."""

import sqlite3
import psycopg2
from psycopg2.extras import execute_values

SQLITE_PATH = "/home/melzak/dev/inzynierka/data/betting_app.sqlite3"
PG_DSN = "postgresql://betting:betting_local_password@localhost:5432/betting"

def migrate():
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_cur = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(PG_DSN)
    pg_cur = pg_conn.cursor()
    
    # Get rating_run_id mapping
    # SQLite: latest-full has id=5
    # PostgreSQL: latest-full has id=3
    sqlite_cur.execute("SELECT id, ratings_version FROM rating_runs")
    sqlite_runs = {row[1]: row[0] for row in sqlite_cur.fetchall()}
    
    pg_cur.execute("SELECT id, ratings_version FROM rating_runs")
    pg_runs = {row[1]: row[0] for row in pg_cur.fetchall()}
    
    print("SQLite rating_runs:", sqlite_runs)
    print("PostgreSQL rating_runs:", pg_runs)
    
    # Migrate entity_ratings for latest-full (rating_run_id=5 in SQLite → id=3 in PG)
    sqlite_rating_run_id = sqlite_runs['latest-full']  # 5
    pg_rating_run_id = pg_runs['latest-full']  # 3
    
    print(f"\nMigrating entity_ratings: SQLite rating_run_id={sqlite_rating_run_id} → PG rating_run_id={pg_rating_run_id}")
    
    # Read from SQLite
    sqlite_cur.execute("""
        SELECT entity_type, entity_id, system_name, rating_value, rd_value, 
               volatility_value, games_count, last_game_at, updated_at
        FROM entity_ratings
        WHERE rating_run_id = ?
    """, (sqlite_rating_run_id,))
    
    rows = sqlite_cur.fetchall()
    print(f"Read {len(rows)} rows from SQLite")
    
    if not rows:
        print("No rows to migrate")
        return
    
    # Delete existing rows with this rating_run_id in PG
    pg_cur.execute("DELETE FROM entity_ratings WHERE rating_run_id = %s", (pg_rating_run_id,))
    print(f"Deleted existing rows with rating_run_id={pg_rating_run_id} from PG")
    
    # Insert into PostgreSQL
    insert_sql = """
        INSERT INTO entity_ratings 
        (rating_run_id, entity_type, entity_id, system_name, rating_value, rd_value, 
         volatility_value, games_count, last_game_at, updated_at)
        VALUES %s
    """
    
    # Prepare data with correct rating_run_id
    values = [
        (pg_rating_run_id, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8])
        for row in rows
    ]
    
    execute_values(pg_cur, insert_sql, values, page_size=1000)
    pg_conn.commit()
    
    print(f"Inserted {len(values)} rows into PostgreSQL")
    
    # Verify
    pg_cur.execute("SELECT COUNT(*) FROM entity_ratings WHERE rating_run_id = %s", (pg_rating_run_id,))
    count = pg_cur.fetchone()[0]
    print(f"Verification: {count} rows in PG with rating_run_id={pg_rating_run_id}")
    
    # Cleanup
    sqlite_conn.close()
    pg_conn.close()
    
    print("\nMigration complete!")

if __name__ == "__main__":
    migrate()
