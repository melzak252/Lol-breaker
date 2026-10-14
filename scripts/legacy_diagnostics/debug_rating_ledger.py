import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("""
    SELECT COUNT(*) 
    FROM rating_processed_matches 
    WHERE match_id IN (SELECT match_id FROM golgg_matches WHERE date > '2026-05-28');
""")

with engine.connect() as conn:
    count = conn.execute(query).scalar()
    print(f"Matches after 2026-05-28 already in rating_processed_matches: {count}")

query_total = text("SELECT COUNT(*) FROM golgg_matches WHERE date > '2026-05-28';")
with engine.connect() as conn:
    total = conn.execute(query_total).scalar()
    print(f"Total matches after 2026-05-28 in golgg_matches: {total}")

query_last_processed = text("SELECT MAX(date) FROM golgg_matches WHERE match_id IN (SELECT match_id FROM rating_processed_matches);")
with engine.connect() as conn:
    last_date = conn.execute(query_last_processed).scalar()
    print(f"Last processed match date: {last_date}")
