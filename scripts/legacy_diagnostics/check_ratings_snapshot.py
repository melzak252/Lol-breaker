from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    res = conn.execute(text("SELECT MIN(snapshot_at), MAX(snapshot_at), COUNT(*) FROM entity_ratings")).fetchone()
    print(f"Ratings range (snapshot_at): {res[0]} to {res[1]} (Total: {res[2]})")

    # Check ratings from June 2026
    res_june = conn.execute(text("SELECT COUNT(*) FROM entity_ratings WHERE snapshot_at >= '2026-06-01'")).fetchone()
    print(f"Ratings from June 2026: {res_june[0]}")
