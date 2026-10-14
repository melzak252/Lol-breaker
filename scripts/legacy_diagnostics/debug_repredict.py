from betting_app.core.db import query_df
import pandas as pd

df = query_df("""
    SELECT cm.id, cm.status, cm.start_time_normalized, cm.team_a_name, cm.team_b_name,
           gmm.golgg_match_id
    FROM canonical_matches cm
    LEFT JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
    WHERE cm.start_time_normalized >= '2026-05-01' 
      AND cm.start_time_normalized < '2026-07-01'
""")
print(f"Total matches in range: {len(df)}")
print("\nStatus counts:")
print(df['status'].value_counts())
print("\nMatches with GOL.GG mapping:")
print(df['golgg_match_id'].notnull().value_counts())

finished_mapped = df[(df['status'].isin(['finished', 'completed'])) & (df['golgg_match_id'].notnull())]
print(f"\nFinished and mapped matches: {len(finished_mapped)}")

if len(finished_mapped) > 0:
    print("\nSample finished and mapped:")
    print(finished_mapped.head())
else:
    print("\nChecking why no finished and mapped matches found...")
    print("Sample of matches in range:")
    print(df.head(10))
