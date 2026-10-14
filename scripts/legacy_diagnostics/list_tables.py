import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public';
""")

with engine.connect() as conn:
    tables = conn.execute(query).fetchall()
    print("Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
