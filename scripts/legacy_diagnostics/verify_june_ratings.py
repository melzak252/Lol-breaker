import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

teams = ["Anyone's Legend", "Team WE", "Karmine Corp", "GIANTX"]

with engine.connect() as conn:
    for team in teams:
        print(f"\nRatings for {team}:")
        res = conn.execute(text("""
            SELECT rating_system, rating_value, rd, last_match_at, games_played
            FROM entity_ratings
            WHERE ratings_version = 'latest-full'
              AND entity_type = 'team'
              AND entity_name = :name
        """), {"name": team})
        for row in res:
            print(f"  {row[0]}: val={row[1]:.2f}, rd={row[2] if row[2] else 'N/A'}, last={row[3]}, games={row[4]}")
