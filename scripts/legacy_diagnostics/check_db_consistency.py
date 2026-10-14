import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

print("--- Versions in rating_processed_matches ---")
query_versions = text("SELECT DISTINCT ratings_version FROM rating_processed_matches;")
with engine.connect() as conn:
    versions = conn.execute(query_versions).fetchall()
    for v in versions:
        print(f"  - {v[0]}")

print("\n--- Versions in entity_ratings ---")
query_er_versions = text("SELECT DISTINCT ratings_version FROM entity_ratings;")
with engine.connect() as conn:
    versions = conn.execute(query_er_versions).fetchall()
    for v in versions:
        print(f"  - {v[0]}")

print("\n--- Latest 5 runs for 'latest-full' ---")
query_runs = text("""
    SELECT id, status, data_cutoff_at, finished_at, error 
    FROM rating_runs 
    WHERE ratings_version = 'latest-full' 
    ORDER BY id DESC 
    LIMIT 5;
""")
with engine.connect() as conn:
    runs = conn.execute(query_runs).fetchall()
    for r in runs:
        print(f"ID: {r.id}, Status: {r.status}, Cutoff: {r.data_cutoff_at}, Finished: {r.finished_at}, Error: {r.error}")
