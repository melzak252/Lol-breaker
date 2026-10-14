import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

with engine.connect() as conn:
    res = conn.execute(text("SELECT COUNT(*) FROM golgg_matches WHERE date > '2026-05-28'")).scalar()
    print(f"Matches in golgg_matches after 2026-05-28: {res}")
    
    res = conn.execute(text("SELECT MIN(date), MAX(date) FROM golgg_matches WHERE date > '2026-05-28'")).fetchone()
    print(f"Date range: {res}")
