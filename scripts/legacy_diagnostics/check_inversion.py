import pandas as pd
from sklearn.metrics import log_loss
import numpy as np
from betting_app.core.db import query_df
from betting_app.core.matching import normalize_team_name

query = """
WITH latest_predictions AS (
    SELECT cp.*
    FROM canonical_predictions cp
    JOIN (
        SELECT canonical_match_id, model_name, MAX(predicted_at) as max_predicted_at
        FROM canonical_predictions
        WHERE model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
        GROUP BY canonical_match_id, model_name
    ) latest ON cp.canonical_match_id = latest.canonical_match_id 
             AND cp.model_name = latest.model_name 
             AND cp.predicted_at = latest.max_predicted_at
)
SELECT 
    cm.team_a_name, cm.team_b_name, lp.prob_a,
    gm.team1_name, gm.team2_name, gm.team1_win, gm.team2_win
FROM canonical_matches cm
JOIN latest_predictions lp ON cm.id = lp.canonical_match_id
JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
WHERE cm.status IN ('finished', 'completed')
"""

df = query_df(query)

def get_team_a_win(row):
    norm_a = normalize_team_name(row['team_a_name'])
    norm_g1 = normalize_team_name(row['team1_name'])
    norm_g2 = normalize_team_name(row['team2_name'])
    if norm_a == norm_g1 or norm_a in norm_g1 or norm_g1 in norm_a: return row['team1_win']
    if norm_a == norm_g2 or norm_a in norm_g2 or norm_g2 in norm_a: return row['team2_win']
    return np.nan

df['team_a_win'] = df.apply(get_team_a_win, axis=1)
df = df.dropna(subset=['team_a_win'])

y_true = df['team_a_win'].astype(int)
y_pred = df['prob_a']

print(f"Original LogLoss: {log_loss(y_true, y_pred, labels=[0, 1]):.4f}")
print(f"Inverted LogLoss: {log_loss(y_true, 1 - y_pred, labels=[0, 1]):.4f}")
