import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("""
    SELECT match_id, match_date, processed_at 
    FROM rating_processed_matches 
    WHERE ratings_version = 'latest-full' 
      AND match_date > '2026-05-28'
    ORDER BY processed_at DESC
    LIMIT 10;
""")

with engine.connect() as conn:
    results = conn.execute(query).fetchall()
    print("Latest entries in rating_processed_matches for 'latest-full' after 2026-05-28:")
    for row in results:
        print(f"Match ID: {row.match_id}, Date: {row.match_date}, Processed At: {row.processed_at}")
