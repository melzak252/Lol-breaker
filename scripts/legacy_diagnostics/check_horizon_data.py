import psycopg2
import os
import json

dsn = os.environ.get('DATABASE_URL').replace('postgresql+psycopg2://', 'postgresql://')
conn = psycopg2.connect(dsn)
cur = conn.cursor()

# Check statuses
cur.execute("SELECT status, COUNT(*) FROM canonical_matches GROUP BY status")
print(f"Statuses: {cur.fetchall()}")

# Check odds snapshots count
cur.execute("SELECT COUNT(*) FROM odds_snapshots")
print(f"Total odds snapshots: {cur.fetchone()[0]}")

# Find matches with odds snapshots
cur.execute("""
    SELECT cm.status, COUNT(DISTINCT cm.id)
    FROM canonical_matches cm
    JOIN odds_snapshots os ON os.canonical_match_id = cm.id
    GROUP BY cm.status
""")
print(f"Matches with odds by status: {cur.fetchall()}")

# Check recent odds snapshots
cur.execute("""
    SELECT os.canonical_match_id, os.timestamp, cm.status, cm.start_time_normalized
    FROM odds_snapshots os
    JOIN canonical_matches cm ON cm.id = os.canonical_match_id
    ORDER BY os.timestamp DESC
    LIMIT 5
""")
print(f"Recent odds snapshots: {cur.fetchall()}")
if match_ids:
    cur.execute("""
        SELECT canonical_match_id FROM upcoming_match_features
        WHERE canonical_match_id IN %s
    """, (tuple(match_ids),))
    feature_ids = [r[0] for r in cur.fetchall()]
    print(f'Matches with features: {len(feature_ids)}')
    
    if len(feature_ids) < len(match_ids):
        missing = list(set(match_ids) - set(feature_ids))
        print(f'Missing features for {len(missing)} matches.')
        # Print first 5 missing
        for m_id in missing[:5]:
            cur.execute("SELECT canonical_key, team_a_name, team_b_name, start_time_normalized FROM canonical_matches WHERE id = %s", (m_id,))
            print(f"  Missing: {cur.fetchone()}")
else:
    print("No matches with odds found in the last 90 days.")
