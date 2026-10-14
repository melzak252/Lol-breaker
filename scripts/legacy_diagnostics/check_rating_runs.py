import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

query = text("SELECT * FROM rating_runs ORDER BY id DESC LIMIT 10;")
with engine.connect() as conn:
    results = conn.execute(query).fetchall()
    print("Latest rating runs:")
    for row in results:
        print(f"ID: {row.id}, Status: {row.status}, Mode: {row.mode}, Created: {row.created_at}, Finished: {row.finished_at}, Error: {row.error_message}")
