import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'rating_runs';
""")

with engine.connect() as conn:
    columns = conn.execute(query).fetchall()
    print("Columns in rating_runs:")
    for col in columns:
        print(f"  - {col[0]} ({col[1]})")
