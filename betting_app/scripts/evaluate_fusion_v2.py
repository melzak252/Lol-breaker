#!/usr/bin/env python3
"""
Evaluate Fusion Model v2 on test set and calculate calibration metrics.
Saves compact results (metrics only, no full prediction arrays).
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import pandas as pd
import sys

# Model architecture (must match training)
class FusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    
    return ece

def main():
    print("Loading data...", flush=True)
    
    # Load baseline features
    baseline_df = pd.read_csv('data/golgg_y_predicts.csv')
    baseline_df = baseline_df.sort_values('date').reset_index(drop=True)
    
    # Load transformer embeddings
    with open('data/transformer_embeddings_v2.json', 'r') as f:
        embeddings_data = json.load(f)
    
    emb_lookup = {int(item['match_id']): item['embedding'] for item in embeddings_data}
    
    # Match baseline with embeddings
    baseline_df['has_emb'] = baseline_df['golgg_match_id'].apply(lambda x: int(x) in emb_lookup)
    matched_df = baseline_df[baseline_df['has_emb']].copy()
    
    print(f"Matched samples: {len(matched_df)}", flush=True)
    
    # Prepare features - must match training script's EXCLUDE_COLS
    exclude_cols = ['team1_name', 'team2_name', 'date', 'golgg_match_id', 'y_true', 'match_id_str', 'has_emb']
    feature_cols = [c for c in matched_df.columns if c not in exclude_cols and matched_df[c].dtype in ['float64', 'int64']]
    
    baseline_features = matched_df[feature_cols].values.astype(np.float32)
    
    # Get embeddings
    embeddings = np.array([emb_lookup[int(mid)] for mid in matched_df['golgg_match_id']], dtype=np.float32)
    
    # Combine features
    all_features = np.concatenate([baseline_features, embeddings], axis=1)
    labels = matched_df['y_true'].values.astype(np.float32)
    match_ids = matched_df['golgg_match_id'].values.astype(int)
    
    print(f"Feature dim: {all_features.shape[1]} ({len(feature_cols)} baseline + 192 transformer)", flush=True)
    
    # Split: test set = 2025+
    dates = pd.to_datetime(matched_df['date'])
    test_mask = dates >= '2025-01-01'
    
    test_features = all_features[test_mask]
    test_labels = labels[test_mask]
    test_match_ids = match_ids[test_mask]
    
    print(f"Test set: {len(test_labels)} matches (2025+)", flush=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    model = FusionModel(input_dim=all_features.shape[1]).to(device)
    checkpoint = torch.load('models/fusion_v2_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Apply StandardScaler normalization (using mean/scale from training)
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    print(f"Applying StandardScaler normalization...", flush=True)
    all_features = (all_features - scaler_mean) / scaler_scale
    
    # Re-extract test features after normalization
    test_features = all_features[test_mask]
    
    # Create dataloader
    test_dataset = TensorDataset(
        torch.from_numpy(test_features),
        torch.from_numpy(test_labels)
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Inference
    print("\nRunning inference on test set...", flush=True)
    test_preds = []
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            test_preds.extend(preds.tolist())
    
    test_preds = np.array(test_preds)
    
    # Calculate metrics
    test_loss = log_loss(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
    ece = expected_calibration_error(test_labels, test_preds, n_bins=10)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Test Results (2025+, {len(test_labels)} matches):", flush=True)
    print(f"  LogLoss:  {test_loss:.4f}", flush=True)
    print(f"  AUC:      {test_auc:.4f}", flush=True)
    print(f"  Accuracy: {test_acc:.4f}", flush=True)
    print(f"  ECE:      {ece:.4f}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Calibration analysis
    print("\nCalibration by confidence bin:", flush=True)
    bin_boundaries = np.linspace(0, 1, 11)
    calibration_bins = []
    for i in range(10):
        mask = (test_preds >= bin_boundaries[i]) & (test_preds < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = float(test_labels[mask].mean())
            bin_conf = float(test_preds[mask].mean())
            bin_size = int(mask.sum())
            gap = abs(bin_acc - bin_conf)
            print(f"  [{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}]: "
                  f"n={bin_size:4d}, acc={bin_acc:.3f}, conf={bin_conf:.3f}, "
                  f"gap={gap:.3f}", flush=True)
            calibration_bins.append({
                'range': f"{bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f}",
                'n': bin_size,
                'accuracy': round(bin_acc, 4),
                'confidence': round(bin_conf, 4),
                'gap': round(gap, 4)
            })
    
    # Confidence-based accuracy
    print("\nAccuracy by confidence threshold:", flush=True)
    confidence_stats = {}
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4]:
        conf = np.abs(test_preds - 0.5)
        mask = conf >= thresh
        if mask.sum() > 0:
            acc = float(test_labels[mask] == (test_preds[mask] > 0.5).astype(int))
            acc = float(((test_preds[mask] > 0.5).astype(int) == test_labels[mask]).mean())
            n = int(mask.sum())
            pct = n / len(test_labels) * 100
            print(f"  conf >= {thresh:.1f}: acc={acc:.4f}, n={n} ({pct:.1f}%)", flush=True)
            confidence_stats[f">={thresh}"] = {
                'accuracy': round(acc, 4),
                'n': n,
                'pct': round(pct, 1)
            }
    
    # Save compact results (no full arrays)
    results = {
        'metrics': {
            'logloss': round(float(test_loss), 6),
            'auc': round(float(test_auc), 6),
            'accuracy': round(float(test_acc), 6),
            'ece': round(float(ece), 6)
        },
        'calibration_bins': calibration_bins,
        'confidence_stats': confidence_stats,
        'n_test': len(test_labels),
        'n_features': int(all_features.shape[1]),
        'prediction_stats': {
            'mean': round(float(test_preds.mean()), 4),
            'std': round(float(test_preds.std()), 4),
            'min': round(float(test_preds.min()), 4),
            'max': round(float(test_preds.max()), 4)
        }
    }
    
    with open('data/fusion_v2_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: data/fusion_v2_eval_results.json", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
