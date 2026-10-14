import json
import pandas as pd
from betting_app.core.db import query_df

def main():
    # Get the 74 matches and their golgg_match_ids
    sql = """
    SELECT gmm.golgg_match_id, cp.prob_a as pg_prob, cp.diagnostics_json
    FROM canonical_predictions cp
    JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cp.canonical_match_id
    WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
      AND cp.model_version = 'exp-039-golgg-corrected'
    """
    df_pg = query_df(sql)
    
    # Load the full 2026 dataset from CSV
    df_csv = pd.read_csv('data/golgg_y_predicts.csv')
    
    # Merge to find the same matches
    # Note: golgg_match_id in CSV might be different format or missing
    # Let's check CSV columns again
    # team1_id,team2_id,team1_name,team2_name,BoN,date,golgg_match_id,y_true
    
    df_csv['golgg_match_id'] = df_csv['golgg_match_id'].astype(str)
    df_pg['golgg_match_id'] = df_pg['golgg_match_id'].astype(str)
    
    merged = pd.merge(df_pg, df_csv, on='golgg_match_id', how='inner')
    
    print(f"Found {len(merged)} matches in both PG and CSV.")
    
    if len(merged) > 0:
        from sklearn.metrics import log_loss
        
        # In CSV, we don't have the ensemble prediction, only base ratings.
        # Wait, I want to know if the 0.54 LogLoss from the full evaluation 
        # is consistent with these 74 matches.
        
        # Let's calculate LogLoss for base ratings in this subset
        systems = ['player_elo', 'player_gl', 'player_ts']
        print("\nLogLoss on the 74-match subset (from CSV data):")
        for s in systems:
            ll = log_loss(merged['y_true'], merged[s].clip(0.001, 0.999))
            print(f"{s:10} | {ll:.4f}")
            
        print("\nLogLoss on the 74-match subset (from PG re-prediction):")
        # We need to make sure y_true is aligned.
        # In PG re-prediction, we calculated y_true in compare_74_matches.py
        # Let's just run a quick check here.
        
        y_true_pg = []
        for diag_json in merged['diagnostics_json']:
            diag = json.loads(diag_json)
            g1 = diag.get('golgg_team1', '')
            gw = diag.get('golgg_winner', '')
            y_true_pg.append(1 if g1 == gw else 0)
            
        ll_pg = log_loss(y_true_pg, merged['pg_prob'].clip(0.001, 0.999))
        print(f"{'ensemble':10} | {ll_pg:.4f}")

if __name__ == "__main__":
    main()
