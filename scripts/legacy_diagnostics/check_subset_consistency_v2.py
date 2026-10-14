import json
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from betting_app.core.db import query_df

def main():
    # 1. Get the 74 PG predictions
    sql = """
    SELECT gmm.golgg_match_id, cp.prob_a as pg_prob, cp.diagnostics_json
    FROM canonical_predictions cp
    JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cp.canonical_match_id
    WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
      AND cp.model_version = 'exp-039-golgg-corrected'
    """
    df_pg = query_df(sql)
    
    # 2. Load the CSV (which has the 1779 matches from the 0.54 LogLoss evaluation)
    df_csv = pd.read_csv('data/golgg_y_predicts.csv')
    
    # Ensure IDs are strings
    df_pg['golgg_match_id'] = df_pg['golgg_match_id'].astype(str)
    df_csv['golgg_match_id'] = df_csv['golgg_match_id'].astype(str)
    
    # 3. Merge
    merged = pd.merge(df_pg, df_csv, on='golgg_match_id', how='inner')
    
    print(f"Successfully matched {len(merged)} out of 74 matches.")
    
    if len(merged) == 0:
        # Try matching by team names and date if IDs differ
        print("Attempting fuzzy match by teams and date...")
        # ... (omitted for now, let's see if IDs work)
        pass

    if len(merged) > 0:
        # Calculate LogLoss for PG ensemble on this subset
        y_true = merged['y_true']
        pg_probs = merged['pg_prob'].clip(0.001, 0.999)
        ll_pg = log_loss(y_true, pg_probs)
        
        # Calculate LogLoss for Glicko-2 from CSV on this subset
        gl_probs = merged['player_gl'].clip(0.001, 0.999)
        ll_gl = log_loss(y_true, gl_probs)
        
        print(f"\nResults for the {len(merged)} matched matches:")
        print(f"PG Ensemble LogLoss: {ll_pg:.4f}")
        print(f"CSV Glicko-2 LogLoss: {ll_gl:.4f}")
        
        # Check if we have the ensemble prediction in another CSV? 
        # No, the 0.54 LogLoss came from evaluate_2026_full.py which ran on the fly.
    else:
        # List some IDs from both to debug
        print("\nPG IDs (first 5):", df_pg['golgg_match_id'].head().tolist())
        print("CSV IDs (first 5):", df_csv['golgg_match_id'].head().tolist())

if __name__ == "__main__":
    main()
