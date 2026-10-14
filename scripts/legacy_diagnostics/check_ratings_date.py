from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    res = conn.execute(text("SELECT MIN(created_at), MAX(created_at), COUNT(*) FROM player_ratings")).fetchone()
    print(f"Ratings range: {res[0]} to {res[1]} (Total: {res[2]})")

    # Check ratings for a specific date in June
    res_june = conn.execute(text("SELECT COUNT(*) FROM player_ratings WHERE created_at >= '2026-06-01'")).fetchone()
    print(f"Ratings from June 2026: {res_june[0]}")
