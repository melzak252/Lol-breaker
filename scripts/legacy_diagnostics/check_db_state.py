import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

with engine.connect() as conn:
    res1 = conn.execute(text("SELECT MAX(snapshot_at) FROM entity_ratings")).scalar()
    print(f"Max snapshot_at in entity_ratings: {res1}")
    
    res2 = conn.execute(text("SELECT COUNT(*) FROM rating_processed_matches WHERE ratings_version = 'latest-full'")).scalar()
    print(f"Count in rating_processed_matches: {res2}")
    
    res3 = conn.execute(text("SELECT MAX(match_date) FROM rating_processed_matches WHERE ratings_version = 'latest-full'")).scalar()
    print(f"Max match_date in rating_processed_matches: {res3}")
