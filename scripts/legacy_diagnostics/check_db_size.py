
from betting_app.core.db import connect

def check_db():
    with connect() as connection:
        print("--- Rating Runs ---")
        runs = connection.execute("SELECT id, ratings_version, status, finished_at FROM rating_runs").fetchall()
        for run in runs:
            print(dict(run))
        
        print("\n--- Entity Ratings Counts ---")
        counts = connection.execute("SELECT ratings_version, COUNT(*) as count FROM entity_ratings GROUP BY ratings_version").fetchall()
        for c in counts:
            print(dict(c))

if __name__ == "__main__":
    check_db()
