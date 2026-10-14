import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM rating_runs ORDER BY id DESC LIMIT 5;"))
    print(result.keys())
    for row in result:
        print(row)
