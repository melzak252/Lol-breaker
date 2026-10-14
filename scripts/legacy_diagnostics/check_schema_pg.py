import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

with engine.connect() as conn:
    # Check columns of entity_ratings
    res = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'entity_ratings'
    """))
    print("Columns in entity_ratings:")
    for row in res:
        print(f" - {row[0]}")

    # Check columns of rating_processed_matches
    res = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'rating_processed_matches'
    """))
    print("\nColumns in rating_processed_matches:")
    for row in res:
        print(f" - {row[0]}")
