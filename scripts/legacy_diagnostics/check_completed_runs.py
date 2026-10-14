import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("""
    SELECT id, status, data_cutoff_at, finished_at 
    FROM rating_runs 
    WHERE status = 'completed' 
    ORDER BY finished_at DESC 
    LIMIT 5;
""")

with engine.connect() as conn:
    results = conn.execute(query).fetchall()
    print("Latest completed rating runs:")
    for row in results:
        print(f"ID: {row.id}, Cutoff: {row.data_cutoff_at}, Finished: {row.finished_at}")
