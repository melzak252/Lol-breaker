import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import numpy as np
from betting_app.core.db import query_df
from betting_app.core.matching import normalize_team_name

# Query to get predictions for finished canonical matches and link them to golgg_matches via golgg_match_mappings
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
    cm.id as canonical_match_id,
    cm.team_a_name,
    cm.team_b_name,
    cm.start_time_normalized,
    lp.prob_a,
    lp.prob_b,
    lp.predicted_at,
    gm.match_id as golgg_match_id,
    gm.team1_name as golgg_team1_name,
    gm.team2_name as golgg_team2_name,
    gm.team1_win,
    gm.team2_win,
    gm.date as golgg_date
FROM canonical_matches cm
JOIN latest_predictions lp ON cm.id = lp.canonical_match_id
JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
WHERE cm.status IN ('finished', 'completed')
"""

df = query_df(query)

if len(df) == 0:
    print("No finished matches with predictions found.")
else:
    print(f"Found {len(df)} predictions for finished canonical matches linked to GOL.GG.")
    
    def get_team_a_win(row):
        norm_a = normalize_team_name(row['team_a_name'])
        norm_b = normalize_team_name(row['team_b_name'])
        norm_g1 = normalize_team_name(row['golgg_team1_name'])
        norm_g2 = normalize_team_name(row['golgg_team2_name'])
        
        # Check direct match or substring match
        a_is_1 = norm_a == norm_g1 or norm_a in norm_g1 or norm_g1 in norm_a
        a_is_2 = norm_a == norm_g2 or norm_a in norm_g2 or norm_g2 in norm_a
        
        if a_is_1 and not a_is_2:
            return row['team1_win']
        elif a_is_2 and not a_is_1:
            return row['team2_win']
        
        # Fallback to checking team B
        b_is_1 = norm_b == norm_g1 or norm_b in norm_g1 or norm_g1 in norm_b
        b_is_2 = norm_b == norm_g2 or norm_b in norm_g2 or norm_g2 in norm_b
        
        if b_is_2 and not b_is_1:
            return row['team1_win'] # If B is 2, A must be 1
        elif b_is_1 and not b_is_2:
            return row['team2_win'] # If B is 1, A must be 2
            
        return np.nan
        
    df['team_a_win'] = df.apply(get_team_a_win, axis=1)
    df_valid = df.dropna(subset=['team_a_win']).copy()
    
    print(f"Valid results for {len(df_valid)} matches (could resolve sides).")
    
    if len(df_valid) > 0:
        y_true = df_valid['team_a_win'].astype(int)
        y_pred = df_valid['prob_a']
        
        ll = log_loss(y_true, y_pred, labels=[0, 1])
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')
            
        acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        
        print("\n--- Evaluation Results ---")
        print(f"LogLoss: {ll:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        
        print("\nSample predictions:")
        print(df_valid[['team_a_name', 'team_b_name', 'prob_a', 'team_a_win', 'golgg_date']].head(10))
        
        # Let's also check if there are any really bad predictions
        df_valid['brier_score'] = (df_valid['prob_a'] - df_valid['team_a_win'])**2
        bad_preds = df_valid.sort_values('brier_score', ascending=False).head(5)
        print("\nWorst predictions (highest Brier score):")
        print(bad_preds[['team_a_name', 'team_b_name', 'prob_a', 'team_a_win', 'brier_score']])
