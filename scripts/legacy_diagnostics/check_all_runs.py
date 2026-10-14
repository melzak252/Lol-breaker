import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("SELECT id, ratings_version, status, data_cutoff_at, finished_at, error FROM rating_runs ORDER BY id DESC LIMIT 20;")
with engine.connect() as conn:
    results = conn.execute(query).fetchall()
    print("Latest rating runs (all versions):")
    for row in results:
        print(f"ID: {row.id}, Version: {row.ratings_version}, Status: {row.status}, Cutoff: {row.data_cutoff_at}, Finished: {row.finished_at}, Error: {row.error}")
