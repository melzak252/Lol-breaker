
from betting_app.core.db import query_df
from collections import defaultdict

df = query_df("""
    SELECT id, normalized_team_a, normalized_team_b, start_time_normalized, status, league
    FROM canonical_matches
    WHERE normalized_team_a IS NOT NULL AND normalized_team_b IS NOT NULL
    ORDER BY start_time_normalized
""")

groups = defaultdict(list)
for _, row in df.iterrows():
    teams = tuple(sorted([str(row['normalized_team_a']), str(row['normalized_team_b'])]))
    groups[teams].append(row.to_dict())

for teams, matches in groups.items():
    if len(matches) > 1:
        # Check if any are close in time or just multiple entries
        print(f"Group {teams}:")
        for m in matches:
            print(f"  ID {m['id']}: {m['start_time_normalized']} [{m['status']}] {m['league']}")
        print("-" * 20)
