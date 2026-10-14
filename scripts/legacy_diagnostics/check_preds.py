import pandas as pd
from betting_app.core.db import query_df

query = """
SELECT cp.id, cp.model_name, cp.model_version, cp.predicted_at, cp.prob_a, cp.prob_b, 
       cm.team_a_name, cm.team_b_name, cp.diagnostics_json
FROM canonical_predictions cp
JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
WHERE cm.team_a_name LIKE '%BRION%' OR cm.team_b_name LIKE '%BRION%'
ORDER BY cp.predicted_at DESC
LIMIT 10
"""
df = query_df(query)
print(df[['model_version', 'predicted_at', 'team_a_name', 'team_b_name', 'prob_a']])
