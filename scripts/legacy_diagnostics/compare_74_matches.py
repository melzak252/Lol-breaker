import json
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from betting_app.core.db import query_df
from betting_app.core.matching import normalize_team_name

def main():
    # Query the 74 predictions we just made
    sql = """
    SELECT cp.prob_a, cp.diagnostics_json
    FROM canonical_predictions cp
    WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
      AND cp.model_version = 'exp-039-golgg-corrected'
    ORDER BY cp.predicted_at DESC
    LIMIT 74
    """
    df = query_df(sql)
    
    if df.empty:
        print("No predictions found.")
        return

    results = []
    for _, row in df.iterrows():
        diag = json.loads(row['diagnostics_json'])
        
        # Resolve y_true
        golgg_team1 = diag.get('golgg_team1', '')
        golgg_winner = diag.get('golgg_winner', '')
        if not golgg_winner:
            continue
            
        y_true = 1 if normalize_team_name(golgg_team1) == normalize_team_name(golgg_winner) else 0
        
        # Extract base probabilities from feature_vector
        # ALL_FEATURES order: elo, gl, ts, os, pl, tm, ...
        fvec = diag.get('feature_vector', [[]])[0]
        
        results.append({
            'y_true': y_true,
            'ensemble': row['prob_a'],
            'elo': fvec[0],
            'gl': fvec[1],
            'ts': fvec[2],
            'os': fvec[3],
            'pl': fvec[4],
            'tm': fvec[5]
        })

    res_df = pd.DataFrame(results)
    
    print(f"Comparison for May-June 2026 (N={len(res_df)}):")
    print("-" * 60)
    print(f"{'System':<12} | {'LogLoss':<8} | {'AUC':<8} | {'Acc':<8}")
    print("-" * 60)
    
    systems = ['ensemble', 'elo', 'gl', 'ts', 'os', 'pl', 'tm']
    for s in systems:
        y_pred = res_df[s].clip(0.001, 0.999)
        ll = log_loss(res_df['y_true'], y_pred)
        auc = roc_auc_score(res_df['y_true'], y_pred)
        acc = accuracy_score(res_df['y_true'], (y_pred > 0.5).astype(int))
        print(f"{s:<12} | {ll:.4f}   | {auc:.4f}   | {acc:.4f}")

if __name__ == "__main__":
    main()
