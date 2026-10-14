import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import os
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from betting_app.models.transformer.team_transformer import MatchPredictor

class TransformerDataset(Dataset):
    def __init__(self, matches: List[Dict[str, Any]], max_seq_len: int = 15):
        self.matches = matches
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        
        seq_a = torch.tensor(match['t1_seq'], dtype=torch.float)
        seq_b = torch.tensor(match['t2_seq'], dtype=torch.float)
        
        label = torch.tensor([float(match['y'])], dtype=torch.float)
        match_id = str(match['match_id'])
        
        mask_a = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask_b = torch.zeros(self.max_seq_len, dtype=torch.bool)
        
        return seq_a, seq_b, mask_a, mask_b, label, match_id

def get_transformer_predictions(loader, model, device):
    """Generate predictions from Transformer model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_match_ids = []
    
    with torch.no_grad():
        for seq_a, seq_b, mask_a, mask_b, labels, match_ids in loader:
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            mask_a, mask_b = mask_a.to(device), mask_b.to(device)
            
            logits = model(seq_a, seq_b, mask_a, mask_b)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_match_ids.extend(match_ids)
            
    return np.array(all_preds).flatten(), np.array(all_labels).flatten(), all_match_ids

def main():
    print("=" * 60)
    print("LATE FUSION: Ensemble of Baseline + Transformer")
    print("=" * 60)
    
    # Paths
    transformer_data_path = 'data/transformer_team_sequences_v1.json'
    transformer_model_path = 'models/transformer_best.pt'
    baseline_predictions_path = 'data/golgg_y_predicts.csv'
    
    # Check files exist
    for path in [transformer_data_path, transformer_model_path, baseline_predictions_path]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            return
    
    # 1. Load Transformer data and generate predictions
    print("\n[1/4] Loading Transformer data and generating predictions...")
    with open(transformer_data_path, 'r') as f:
        matches = json.load(f)
    if isinstance(matches, dict) and 'matches' in matches:
        matches = matches['matches']
    
    # Split into train/val/test
    train_matches, val_matches, test_matches = [], [], []
    for m in matches:
        date_str = m['date']
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            if dt.year < 2024:
                train_matches.append(m)
            elif dt.year == 2024:
                val_matches.append(m)
            else:  # 2025+
                test_matches.append(m)
        except:
            pass
    
    print(f"  Train: {len(train_matches)}, Val: {len(val_matches)}, Test: {len(test_matches)}")
    
    # Load Transformer model
    model = MatchPredictor(
        input_dim=9,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    )
    model.load_state_dict(torch.load(transformer_model_path, map_location='cpu'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"  Using device: {device}")
    
    # Generate predictions for all splits
    train_loader = DataLoader(TransformerDataset(train_matches), batch_size=128, shuffle=False)
    val_loader = DataLoader(TransformerDataset(val_matches), batch_size=128, shuffle=False)
    test_loader = DataLoader(TransformerDataset(test_matches), batch_size=128, shuffle=False)
    
    train_preds, train_labels, train_ids = get_transformer_predictions(train_loader, model, device)
    val_preds, val_labels, val_ids = get_transformer_predictions(val_loader, model, device)
    test_preds, test_labels, test_ids = get_transformer_predictions(test_loader, model, device)
    
    print(f"  Generated predictions for {len(test_preds)} test matches")
    
    # 2. Load Baseline predictions
    print("\n[2/4] Loading Baseline predictions...")
    df_baseline = pd.read_csv(baseline_predictions_path)
    df_baseline['golgg_match_id'] = df_baseline['golgg_match_id'].astype(str)
    
    # Create DataFrames with Transformer predictions
    df_train = pd.DataFrame({
        'match_id': train_ids,
        'transformer_pred': train_preds,
        'y_true': train_labels
    })
    df_val = pd.DataFrame({
        'match_id': val_ids,
        'transformer_pred': val_preds,
        'y_true': val_labels
    })
    df_test = pd.DataFrame({
        'match_id': test_ids,
        'transformer_pred': test_preds,
        'y_true': test_labels
    })
    
    # Merge with baseline predictions
    df_train = df_train.merge(df_baseline[['golgg_match_id', 'player_elo']], 
                              left_on='match_id', right_on='golgg_match_id', how='inner')
    df_val = df_val.merge(df_baseline[['golgg_match_id', 'player_elo']], 
                          left_on='match_id', right_on='golgg_match_id', how='inner')
    df_test = df_test.merge(df_baseline[['golgg_match_id', 'player_elo']], 
                            left_on='match_id', right_on='golgg_match_id', how='inner')
    
    print(f"  Overlap - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # 3. Train Ensemble model (Logistic Regression on predictions)
    print("\n[3/4] Training Ensemble model...")
    
    # Prepare features (predictions from both models)
    X_train = df_train[['transformer_pred', 'player_elo']].values
    y_train = df_train['y_true'].values
    
    X_val = df_val[['transformer_pred', 'player_elo']].values
    y_val = df_val['y_true'].values
    
    X_test = df_test[['transformer_pred', 'player_elo']].values
    y_test = df_test['y_true'].values
    
    # Train simple logistic regression
    ensemble = LogisticRegression(random_state=42, max_iter=1000)
    ensemble.fit(X_train, y_train)
    
    print(f"  Ensemble weights: Transformer={ensemble.coef_[0][0]:.4f}, Baseline={ensemble.coef_[0][1]:.4f}")
    print(f"  Intercept: {ensemble.intercept_[0]:.4f}")
    
    # 4. Evaluate all models
    print("\n[4/4] Evaluating models on test set (2025+)...")
    print(f"  Test set size: {len(X_test)} matches")
    
    # Baseline only
    baseline_ll = log_loss(y_test, X_test[:, 1])
    baseline_auc = roc_auc_score(y_test, X_test[:, 1])
    baseline_acc = np.mean((X_test[:, 1] > 0.5) == y_test)
    
    # Transformer only
    transformer_ll = log_loss(y_test, X_test[:, 0])
    transformer_auc = roc_auc_score(y_test, X_test[:, 0])
    transformer_acc = np.mean((X_test[:, 0] > 0.5) == y_test)
    
    # Ensemble
    ensemble_probs = ensemble.predict_proba(X_test)[:, 1]
    ensemble_ll = log_loss(y_test, ensemble_probs)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_acc = np.mean((ensemble_probs > 0.5) == y_test)
    
    # Simple average ensemble (for comparison)
    avg_probs = (X_test[:, 0] + X_test[:, 1]) / 2
    avg_ll = log_loss(y_test, avg_probs)
    avg_auc = roc_auc_score(y_test, avg_probs)
    avg_acc = np.mean((avg_probs > 0.5) == y_test)
    
    print("\n" + "=" * 60)
    print("RESULTS ON TEST SET (2025+)")
    print("=" * 60)
    print(f"\nBaseline (Player Elo):")
    print(f"  LogLoss:  {baseline_ll:.4f}")
    print(f"  AUC:      {baseline_auc:.4f}")
    print(f"  Accuracy: {baseline_acc:.4f}")
    
    print(f"\nTransformer (Team Form):")
    print(f"  LogLoss:  {transformer_ll:.4f}")
    print(f"  AUC:      {transformer_auc:.4f}")
    print(f"  Accuracy: {transformer_acc:.4f}")
    
    print(f"\nEnsemble (Logistic Regression):")
    print(f"  LogLoss:  {ensemble_ll:.4f}")
    print(f"  AUC:      {ensemble_auc:.4f}")
    print(f"  Accuracy: {ensemble_acc:.4f}")
    
    print(f"\nSimple Average Ensemble:")
    print(f"  LogLoss:  {avg_ll:.4f}")
    print(f"  AUC:      {avg_auc:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    
    print("\n" + "=" * 60)
    
    # Find best model
    best_model = min([
        ('Baseline', baseline_ll, baseline_auc),
        ('Transformer', transformer_ll, transformer_auc),
        ('Ensemble LR', ensemble_ll, ensemble_auc),
        ('Simple Average', avg_ll, avg_auc)
    ], key=lambda x: x[1])
    
    print(f"\n🏆 Best model by LogLoss: {best_model[0]} (LL={best_model[1]:.4f}, AUC={best_model[2]:.4f})")
    print("=" * 60)

if __name__ == "__main__":
    main()
