import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

db_url = os.getenv("DATABASE_URL")
print(f"Connecting to: {db_url}")
engine = create_engine(db_url)

with engine.connect() as conn:
    # Check tables
    res = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """))
    tables = [row[0] for row in res]
    print(f"Tables: {tables}")
    
    if 'golgg_match_mappings' in tables:
        res = conn.execute(text("SELECT COUNT(*) FROM golgg_match_mappings"))
        count = res.scalar()
        print(f"golgg_match_mappings count: {count}")
        if count > 0:
            res = conn.execute(text("SELECT * FROM golgg_match_mappings LIMIT 5"))
            print("Sample mappings:")
            for row in res:
                print(row)
    else:
        print("golgg_match_mappings table MISSING in PostgreSQL!")

    # Check canonical_matches in May-June
    res = conn.execute(text("""
        SELECT status, COUNT(*) 
        FROM canonical_matches 
        WHERE start_time_normalized >= '2026-05-01' 
          AND start_time_normalized < '2026-07-01'
        GROUP BY status
    """))
    print("\nCanonical matches (May-June 2026):")
    for row in res:
        print(row)
