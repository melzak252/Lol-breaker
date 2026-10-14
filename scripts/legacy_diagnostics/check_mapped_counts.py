import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

with engine.connect() as conn:
    res = conn.execute(text("""
        SELECT cm.status, COUNT(*)
        FROM canonical_matches cm
        JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        WHERE cm.start_time_normalized >= '2026-05-01' 
          AND cm.start_time_normalized < '2026-07-01'
        GROUP BY cm.status
    """))
    print("Mapped matches by status:")
    for row in res:
        print(row)

    res = conn.execute(text("""
        SELECT COUNT(*)
        FROM canonical_matches cm
        JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
        WHERE cm.start_time_normalized >= '2026-05-01' 
          AND cm.start_time_normalized < '2026-07-01'
    """))
    print(f"Total matches with mapping and GOL.GG data: {res.scalar()}")
