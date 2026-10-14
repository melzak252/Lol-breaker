from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    res = conn.execute(text("SELECT MIN(created_at), MAX(created_at), COUNT(*) FROM entity_ratings")).fetchone()
    print(f"Ratings range: {res[0]} to {res[1]} (Total: {res[2]})")

    # Check ratings for a specific date in June
    res_june = conn.execute(text("SELECT COUNT(*) FROM entity_ratings WHERE created_at >= '2026-06-01'")).fetchone()
    print(f"Ratings from June 2026: {res_june[0]}")

    # Check distinct rating types
    res_types = conn.execute(text("SELECT DISTINCT rating_type FROM entity_ratings")).fetchall()
    print(f"Rating types: {[r[0] for r in res_types]}")
