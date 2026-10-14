import sqlite3
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import numpy as np

conn = sqlite3.connect('data/betting_app.sqlite3')

# Query to get predictions for canonical matches that are in the past
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
),
match_ids AS (
    SELECT 
        cm.id as canonical_match_id,
        cm.team_a_name,
        cm.team_b_name,
        cm.start_time_normalized,
        (SELECT team_a_golgg_id FROM upcoming_matches WHERE canonical_match_id = cm.id ORDER BY last_seen_at DESC LIMIT 1) as team_a_golgg_id,
        (SELECT team_b_golgg_id FROM upcoming_matches WHERE canonical_match_id = cm.id ORDER BY last_seen_at DESC LIMIT 1) as team_b_golgg_id
    FROM canonical_matches cm
    WHERE cm.start_time_normalized < datetime('now')
)
SELECT 
    m.canonical_match_id,
    m.team_a_name,
    m.team_b_name,
    m.start_time_normalized,
    m.team_a_golgg_id,
    m.team_b_golgg_id,
    lp.prob_a,
    lp.prob_b,
    lp.predicted_at,
    gm.match_id as golgg_match_id,
    gm.team1_id as golgg_team1_id,
    gm.team2_id as golgg_team2_id,
    gm.team1_win,
    gm.team2_win,
    gm.date as golgg_date
FROM match_ids m
JOIN latest_predictions lp ON m.canonical_match_id = lp.canonical_match_id
LEFT JOIN golgg_matches gm ON 
    (
        (CAST(m.team_a_golgg_id AS TEXT) = gm.team1_id AND CAST(m.team_b_golgg_id AS TEXT) = gm.team2_id)
        OR 
        (CAST(m.team_a_golgg_id AS TEXT) = gm.team2_id AND CAST(m.team_b_golgg_id AS TEXT) = gm.team1_id)
    )
    AND date(gm.date) >= date(m.start_time_normalized, '-2 day')
    AND date(gm.date) <= date(m.start_time_normalized, '+2 day')
"""

df = pd.read_sql_query(query, conn)

if len(df) == 0:
    print("No past matches with predictions found.")
else:
    print(f"Found {len(df)} predictions for past canonical matches.")
    
    # Filter out those we couldn't link to golgg_matches
    df_linked = df.dropna(subset=['golgg_match_id']).copy()
    print(f"Successfully linked {len(df_linked)} matches to golgg_matches.")
    
    if len(df_linked) > 0:
        def get_team_a_win(row):
            if str(row['team_a_golgg_id']) == str(row['golgg_team1_id']):
                return row['team1_win']
            elif str(row['team_a_golgg_id']) == str(row['golgg_team2_id']):
                return row['team2_win']
            return np.nan
            
        df_linked['team_a_win'] = df_linked.apply(get_team_a_win, axis=1)
        df_linked = df_linked.dropna(subset=['team_a_win'])
        
        print(f"Valid results for {len(df_linked)} matches.")
        
        if len(df_linked) > 0:
            y_true = df_linked['team_a_win'].astype(int)
            y_pred = df_linked['prob_a']
            
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
            print(df_linked[['team_a_name', 'team_b_name', 'prob_a', 'team_a_win', 'golgg_date']].head(10))
            
            df_linked['brier_score'] = (df_linked['prob_a'] - df_linked['team_a_win'])**2
            bad_preds = df_linked.sort_values('brier_score', ascending=False).head(5)
            print("\nWorst predictions (highest Brier score):")
            print(bad_preds[['team_a_name', 'team_b_name', 'prob_a', 'team_a_win', 'brier_score']])
