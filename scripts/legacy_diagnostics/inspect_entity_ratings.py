import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'entity_ratings';
""")

with engine.connect() as conn:
    columns = conn.execute(query).fetchall()
    print("Columns in entity_ratings:")
    for col in columns:
        print(f"  - {col[0]} ({col[1]})")

query_sample = text("SELECT * FROM entity_ratings LIMIT 1;")
with engine.connect() as conn:
    sample = conn.execute(query_sample).fetchone()
    if sample:
        print("\nSample row keys:", sample._mapping.keys())
        print("Sample row values:", sample)
