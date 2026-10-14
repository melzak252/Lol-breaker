import psycopg2
import os

dsn = os.environ.get('DATABASE_URL').replace('postgresql+psycopg2://', 'postgresql://')
conn = psycopg2.connect(dsn)
cur = conn.cursor()

tables = [
    'canonical_matches',
    'canonical_predictions',
    'model_artifacts',
    'model_ev_signals',
    'odds_snapshots',
    'upcoming_match_features',
    'golgg_matches'
]

print("Table counts:")
for table in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table}: {count}")
    except Exception as e:
        print(f"  {table}: Error - {e}")
        conn.rollback()

cur.execute("SELECT status, COUNT(*) FROM canonical_matches GROUP BY status")
print(f"Canonical matches by status: {cur.fetchall()}")

cur.execute("SELECT model_name, COUNT(*) FROM canonical_predictions cp JOIN model_artifacts ma ON cp.model_artifact_id = ma.id GROUP BY model_name")
print(f"Predictions by model: {cur.fetchall()}")
