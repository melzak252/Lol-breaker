import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("SELECT MAX(created_at) FROM entity_rating_history;")
with engine.connect() as conn:
    max_date = conn.execute(query).scalar()
    print(f"Max date in entity_rating_history: {max_date}")

query_count = text("SELECT COUNT(*) FROM entity_ratings;")
with engine.connect() as conn:
    count = conn.execute(query_count).scalar()
    print(f"Total rows in entity_ratings: {count}")

# Check a specific player's history to see if it stops at May 28
query_sample = text("""
    SELECT h.created_at, h.rating_value 
    FROM entity_rating_history h
    JOIN entities e ON h.entity_id = e.id
    WHERE e.type = 'player'
    ORDER BY h.created_at DESC
    LIMIT 5;
""")
with engine.connect() as conn:
    results = conn.execute(query_sample).fetchall()
    print("Latest player rating history entries:")
    for row in results:
        print(f"  {row[0]}: {row[1]}")
