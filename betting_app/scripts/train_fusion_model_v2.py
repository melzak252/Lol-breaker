#!/usr/bin/env python3
"""
Fusion Model v2: Baseline features + Transformer embeddings
Łączy tradycyjne cechy (Elo, ratingi) z embeddingami z transformera
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from pathlib import Path

# Paths
BASELINE_CSV = "data/golgg_y_predicts.csv"
EMBEDDINGS_JSON = "data/transformer_embeddings_v2.json"
MODEL_PATH = "models/fusion_v2_best.pt"

# Features to exclude from baseline (non-numeric or identifiers)
EXCLUDE_COLS = ['team1_name', 'team2_name', 'date', 'golgg_match_id']


class FusionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


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
        return self.net(x).squeeze(-1)


def load_and_merge_data():
    """Wczytuje baseline CSV i transformer embeddings, łączy po match_id"""
    print("Loading baseline features...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    print(f"  Baseline: {len(baseline_df)} rows, {len(baseline_df.columns)} columns")
    
    print("Loading transformer embeddings...")
    with open(EMBEDDINGS_JSON, 'r') as f:
        embeddings_data = json.load(f)
    print(f"  Embeddings: {len(embeddings_data)} samples")
    
    # Create embeddings lookup: match_id -> embedding
    emb_lookup = {str(item['match_id']): item['embedding'] for item in embeddings_data}
    
    # Filter baseline to only include matches with embeddings
    baseline_df['match_id_str'] = baseline_df['golgg_match_id'].astype(str)
    baseline_df = baseline_df[baseline_df['match_id_str'].isin(emb_lookup)].copy()
    print(f"  Matched: {len(baseline_df)} samples")
    
    # Extract baseline numeric features
    feature_cols = [c for c in baseline_df.columns if c not in EXCLUDE_COLS + ['y_true', 'match_id_str']]
    baseline_features = baseline_df[feature_cols].values
    print(f"  Baseline features: {baseline_features.shape[1]} dimensions")
    
    # Extract transformer embeddings
    transformer_features = np.array([emb_lookup[mid] for mid in baseline_df['match_id_str']])
    print(f"  Transformer features: {transformer_features.shape[1]} dimensions")
    
    # Concatenate features
    all_features = np.concatenate([baseline_features, transformer_features], axis=1)
    labels = baseline_df['y_true'].values
    dates = pd.to_datetime(baseline_df['date'])
    
    print(f"  Total features: {all_features.shape[1]} dimensions")
    print(f"  Labels: {labels.sum()} positive, {len(labels) - labels.sum()} negative")
    
    return all_features, labels, dates, feature_cols


def train_model():
    # Load data
    features, labels, dates, baseline_cols = load_and_merge_data()
    
    # Chronological split: train/val before 2025, test 2025+
    test_mask = dates >= '2025-01-01'
    train_val_mask = ~test_mask
    
    # Split train/val (80/20 from pre-2025 data)
    train_val_indices = np.where(train_val_mask)[0]
    np.random.seed(42)
    np.random.shuffle(train_val_indices)
    split_idx = int(0.8 * len(train_val_indices))
    train_indices = train_val_indices[:split_idx]
    val_indices = train_val_indices[split_idx:]
    test_indices = np.where(test_mask)[0]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    # Normalize features
    scaler = StandardScaler()
    features[train_indices] = scaler.fit_transform(features[train_indices])
    features[val_indices] = scaler.transform(features[val_indices])
    features[test_indices] = scaler.transform(features[test_indices])
    
    # Create datasets
    train_dataset = FusionDataset(features[train_indices], labels[train_indices])
    val_dataset = FusionDataset(features[val_indices], labels[val_indices])
    test_dataset = FusionDataset(features[test_indices], labels[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    input_dim = features.shape[1]
    model = FusionModel(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining Fusion Model ({input_dim} input features)...")
    print("=" * 70)
    
    for epoch in range(50):
        # Train
        model.train()
        train_losses = []
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(batch_labels.numpy())
        
        val_loss = log_loss(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
        
        train_loss = np.mean(train_losses)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:2d} | Train LL {train_loss:.4f} | Val LL {val_loss:.4f} | AUC {val_auc:.4f} | Acc {val_acc:.4f}", end="")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'input_dim': input_dim,
                'baseline_cols': baseline_cols,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_
            }, MODEL_PATH)
            print(" ⭐ (best)")
            patience_counter = 0
        else:
            print()
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating best model on test set (2025+)...")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            test_labels.extend(batch_labels.numpy())
    
    test_loss = log_loss(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, (np.array(test_preds) > 0.5).astype(int))
    
    print(f"\nTest Results (2025+, {len(test_labels)} matches):")
    print(f"  LogLoss: {test_loss:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Save predictions for ensemble comparison
    test_results = {
        'match_ids': test_indices.tolist(),
        'predictions': [float(p) for p in test_preds],
        'labels': [int(l) for l in test_labels],
        'metrics': {
            'logloss': float(test_loss),
            'auc': float(test_auc),
            'accuracy': float(test_acc)
        }
    }
    
    with open('data/fusion_v2_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Test results saved to: data/fusion_v2_test_results.json")


if __name__ == "__main__":
    train_model()
