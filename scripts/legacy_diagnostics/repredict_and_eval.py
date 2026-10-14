import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from betting_app.core.db import query_df
from betting_app.services.thesis_inference_service import build_thesis_features_for_match, _load_model, _swap_feature_vector, _symmetrize, _logit
from betting_app.core.matching import normalize_team_name
from dotenv import load_dotenv

load_dotenv()

# 1. Get the 73 matches
query = """
SELECT 
    cm.id as canonical_match_id,
    cm.team_a_name,
    cm.team_b_name,
    cm.best_of,
    gm.team1_name as golgg_team1_name,
    gm.team2_name as golgg_team2_name,
    gm.team1_win,
    gm.team2_win,
    um.team_a_golgg_id,
    um.team_b_golgg_id
FROM canonical_matches cm
JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
LEFT JOIN upcoming_matches um ON um.canonical_match_id = cm.id
WHERE cm.status IN ('finished', 'completed')
  AND cm.start_time_normalized >= '2026-05-01'
  AND cm.start_time_normalized < '2026-07-01'
"""

df = query_df(query)
print(f"Query returned {len(df)} rows.")
if len(df) > 0:
    print("Columns:", df.columns.tolist())
    df = df.sort_values('canonical_match_id').groupby('canonical_match_id').first().reset_index()
else:
    print("No matches found in the specified date range.")
    exit()

print(f"Processing {len(df)} matches for re-prediction...")

pipeline, calibrator = _load_model()
EPSILON = 0.001

results = []

for _, row in df.iterrows():
    team_a = row['team_a_name']
    team_b = row['team_b_name']
    best_of = int(row['best_of']) if pd.notnull(row['best_of']) else 1
    golgg_a = int(row['team_a_golgg_id']) if pd.notnull(row['team_a_golgg_id']) else None
    golgg_b = int(row['team_b_golgg_id']) if pd.notnull(row['team_b_golgg_id']) else None
    
    # Build features
    feature_vec, diag = build_thesis_features_for_match(
        team_a, team_b,
        team_a_golgg_id=golgg_a,
        team_b_golgg_id=golgg_b,
        best_of=best_of
    )
    
    if feature_vec is None:
        continue
        
    # Predict
    orig_prob = float(np.clip(pipeline.predict_proba(feature_vec)[0, 1], EPSILON, 1.0 - EPSILON))
    swapped_vec = _swap_feature_vector(feature_vec)
    swapped_prob = float(np.clip(pipeline.predict_proba(swapped_vec)[0, 1], EPSILON, 1.0 - EPSILON))
    sym_prob = _symmetrize(orig_prob, swapped_prob)
    calibrated_prob = float(np.clip(
        calibrator.predict_proba(_logit(np.array([sym_prob])))[0, 1],
        EPSILON, 1.0 - EPSILON,
    ))
    
    # Resolve winner
    norm_a = normalize_team_name(team_a)
    norm_g1 = normalize_team_name(row['golgg_team1_name'])
    norm_g2 = normalize_team_name(row['golgg_team2_name'])
    
    win_a = np.nan
    if norm_a == norm_g1 or norm_a in norm_g1 or norm_g1 in norm_a: win_a = row['team1_win']
    elif norm_a == norm_g2 or norm_a in norm_g2 or norm_g2 in norm_a: win_a = row['team2_win']
    
    if not np.isnan(win_a):
        results.append({
            'match': f"{team_a} vs {team_b}",
            'prob_a': calibrated_prob,
            'win_a': int(win_a)
        })

res_df = pd.DataFrame(results)
if len(res_df) > 0:
    y_true = res_df['win_a']
    y_pred = res_df['prob_a']
    print("\n--- Re-prediction Results (May-June 2026) ---")
    print(f"Matches: {len(res_df)}")
    print(f"LogLoss: {log_loss(y_true, y_pred, labels=[0, 1]):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, (y_pred > 0.5).astype(int)):.4f}")
else:
    print("No valid re-predictions generated.")
