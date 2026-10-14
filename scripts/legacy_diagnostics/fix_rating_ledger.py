import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

# 1. Clear ledger for 'latest-full' after 2026-05-28
delete_query = text("""
    DELETE FROM rating_processed_matches 
    WHERE ratings_version = 'latest-full' 
      AND match_date > '2026-05-28';
""")

with engine.connect() as conn:
    result = conn.execute(delete_query)
    print(f"Deleted {result.rowcount} entries from rating_processed_matches.")
    conn.commit()

# 2. Verify deletion
check_query = text("""
    SELECT COUNT(*) 
    FROM rating_processed_matches 
    WHERE ratings_version = 'latest-full' 
      AND match_date > '2026-05-28';
""")

with engine.connect() as conn:
    count = conn.execute(check_query).scalar()
    print(f"Remaining entries after 2026-05-28: {count}")
