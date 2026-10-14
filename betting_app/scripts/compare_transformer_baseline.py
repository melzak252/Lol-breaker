import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import os

def main():
    # 1. Load Transformer test set match IDs
    data_path = 'data/transformer_team_sequences_v1.json'
    with open(data_path, 'r') as f:
        matches = json.load(f)
    if isinstance(matches, dict) and 'matches' in matches:
        matches = matches['matches']
        
    test_match_ids = []
    for m in matches:
        date_str = m['date']
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            if dt.year >= 2025:
                test_match_ids.append(str(m['match_id']))
        except:
            pass
            
    print(f"Transformer test matches: {len(test_match_ids)}")
    
    # 2. Load LR model predictions (player-based)
    # The thesis model uses player_elo, etc. 
    # We have golgg_y_predicts.csv which contains the base rating probabilities.
    predicts_path = 'data/golgg_y_predicts.csv'
    if not os.path.exists(predicts_path):
        print("golgg_y_predicts.csv not found")
        return
        
    df_predicts = pd.read_csv(predicts_path)
    df_predicts['golgg_match_id'] = df_predicts['golgg_match_id'].astype(str)
    
    # Filter to test matches
    df_test = df_predicts[df_predicts['golgg_match_id'].isin(test_match_ids)].copy()
    print(f"Overlap with player-based predictions: {len(df_test)}")
    
    if len(df_test) > 0:
        # Use 'player_elo' as a proxy for the baseline rating model
        # (The full thesis model is a metamodel on top of these, but player_elo is the strongest single feature)
        y_true = df_test['y_true'].astype(int)
        y_prob = df_test['player_elo']
        
        ll = log_loss(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        acc = np.mean((y_prob > 0.5) == y_true)
        
        print(f"\nBaseline (Player Elo Rating) on {len(df_test)} matches:")
        print(f"LogLoss: {ll:.4f}")
        print(f"AUC:     {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
    else:
        print("No overlap found to compare.")

if __name__ == "__main__":
    main()
