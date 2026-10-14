#!/usr/bin/env python3
"""
Fusion Model v2 with Symmetrization: Baseline features + Transformer embeddings
Adds data augmentation by swapping team1↔team2 and inference-time averaging.
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
MODEL_PATH = "models/fusion_v2_sym_best.pt"

# Features to exclude from baseline (non-numeric or identifiers)
EXCLUDE_COLS = ['team1_name', 'team2_name', 'date', 'golgg_match_id']

# Baseline column swap pairs: when swapping team1↔team2, swap these column pairs
# Format: (team1_col, team2_col)
BASELINE_SWAP_PAIRS = [
    ('team_elo_r1', 'team_elo_r2'),
    ('player_elo_min1', 'player_elo_min2'),
    ('team_gl_r1', 'team_gl_r2'),
    ('team_gl_rd1', 'team_gl_rd2'),
    ('player_gl_max1', 'player_gl_max2'),
    ('player_gl_rd_avg1', 'player_gl_rd_avg2'),
    ('team_ts_mu1', 'team_ts_mu2'),
    ('team_ts_sigma1', 'team_ts_sigma2'),
    ('player_ts_sigma_avg1', 'player_ts_sigma_avg2'),
    ('team_os_mu1', 'team_os_mu2'),
    ('team_os_sigma1', 'team_os_sigma2'),
    ('player_os_sigma_avg1', 'player_os_sigma_avg2'),
    ('team_pl_mu1', 'team_pl_mu2'),
    ('team_pl_sigma1', 'team_pl_sigma2'),
    ('player_pl_sigma_avg1', 'player_pl_sigma_avg2'),
    ('team_tm_mu1', 'team_tm_mu2'),
    ('team_tm_sigma1', 'team_tm_sigma2'),
    ('player_tm_sigma_avg1', 'player_tm_sigma_avg2'),
    ('days_since_last_1', 'days_since_last_2'),
    ('team1_id', 'team2_id'),
]

# Columns that get negated when swapping (differences)
BASELINE_NEGATE_COLS = ['days_diff']

# Symmetric columns (unchanged when swapping): team_elo, player_elo, team_gl, player_gl,
# team_ts, player_ts, team_os, player_os, team_pl, player_pl, team_tm, player_tm, BoN


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


def swap_baseline_features(features_df, feature_cols):
    """
    Create swapped version of baseline features (team1↔team2).
    Returns numpy array with swapped features.
    """
    swapped = features_df[feature_cols].copy()
    
    # Swap paired columns
    for col1, col2 in BASELINE_SWAP_PAIRS:
        if col1 in swapped.columns and col2 in swapped.columns:
            tmp = swapped[col1].copy()
            swapped[col1] = swapped[col2].values
            swapped[col2] = tmp.values
    
    # Negate difference columns
    for col in BASELINE_NEGATE_COLS:
        if col in swapped.columns:
            swapped[col] = -swapped[col].values
    
    return swapped.values


def swap_embedding(embedding):
    """
    Swap transformer embedding for team1↔team2.
    Embedding structure: [emb_t1(64), emb_t2(64), emb_t1-emb_t2(64)]
    After swap: [emb_t2(64), emb_t1(64), emb_t2-emb_t1(64)]
    """
    emb_t1 = embedding[:64]
    emb_t2 = embedding[64:128]
    emb_diff = embedding[128:192]
    return np.concatenate([emb_t2, emb_t1, -emb_diff])


def load_and_merge_data(augment=True):
    """Wczytuje baseline CSV i transformer embeddings, łączy po match_id.
    If augment=True, adds symmetrized copies of each sample."""
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
    
    if augment:
        print("\nApplying symmetrization (data augmentation)...")
        # Create swapped baseline features
        swapped_baseline = swap_baseline_features(baseline_df, feature_cols)
        # Create swapped embeddings
        swapped_transformer = np.array([swap_embedding(emb) for emb in transformer_features])
        # Concatenate swapped features
        swapped_features = np.concatenate([swapped_baseline, swapped_transformer], axis=1)
        # Flip labels
        swapped_labels = 1.0 - labels
        
        # Augment: original + swapped
        all_features = np.concatenate([all_features, swapped_features], axis=0)
        labels = np.concatenate([labels, swapped_labels], axis=0)
        dates = pd.concat([dates, dates], ignore_index=True)
        
        print(f"  After augmentation: {len(all_features)} samples (doubled)")
        print(f"  Labels: {labels.sum()} positive, {len(labels) - labels.sum()} negative")
    
    return all_features, labels, dates, feature_cols


def symmetrized_predict(model, features, device, batch_size=256):
    """
    Symmetrized inference: predict for both orderings and average.
    p = (f(t1,t2) + 1 - f(t2,t1)) / 2
    
    Features structure: [baseline(54), embedding(192)]
    """
    model.eval()
    n = len(features)
    
    # Original predictions
    preds_original = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.FloatTensor(features[i:i+batch_size]).to(device)
            outputs = model(batch)
            preds_original.extend(torch.sigmoid(outputs).cpu().numpy())
    preds_original = np.array(preds_original)
    
    # Swapped predictions
    swapped_features = features.copy()
    
    # Swap baseline pairs (first 54 columns)
    # Build column index mapping for swap pairs
    # We need to know which indices correspond to which columns
    # Since features are [baseline(54), embedding(192)], baseline is indices 0-53
    # We'll swap based on the known column structure
    
    # For the embedding part (indices 54-245): swap [0:64]↔[64:128], negate [128:192]
    emb_start = 54  # baseline has 54 features
    swapped_features[:, emb_start:emb_start+64], swapped_features[:, emb_start+64:emb_start+128] = \
        swapped_features[:, emb_start+64:emb_start+128].copy(), swapped_features[:, emb_start:emb_start+64].copy()
    swapped_features[:, emb_start+128:emb_start+192] = -swapped_features[:, emb_start+128:emb_start+192]
    
    # For baseline, we need to swap the paired columns
    # We need the feature column names to determine indices
    # This will be handled by passing baseline_cols info
    # For now, we'll handle it in the main function
    
    preds_swapped = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.FloatTensor(swapped_features[i:i+batch_size]).to(device)
            outputs = model(batch)
            preds_swapped.extend(torch.sigmoid(outputs).cpu().numpy())
    preds_swapped = np.array(preds_swapped)
    
    # Symmetrized prediction: p = (p_original + 1 - p_swapped) / 2
    symmetrized_preds = (preds_original + 1.0 - preds_swapped) / 2.0
    
    return symmetrized_preds


def train_model():
    # Load data with augmentation
    features, labels, dates, baseline_cols = load_and_merge_data(augment=True)
    
    # Chronological split: train/val before 2025, test 2025+
    # Note: augmented data has original + swapped, both with same dates
    # Test set should only contain original (non-augmented) samples from 2025+
    n_original = len(features) // 2  # first half is original, second half is augmented
    
    test_mask_orig = dates.iloc[:n_original] >= '2025-01-01'
    test_indices_orig = np.where(test_mask_orig)[0]
    # Also include augmented versions of test samples
    test_indices = np.concatenate([test_indices_orig, test_indices_orig + n_original])
    
    train_val_mask_orig = ~test_mask_orig
    train_val_indices_orig = np.where(train_val_mask_orig)[0]
    
    # Split train/val (80/20 from pre-2025 data, including augmented)
    np.random.seed(42)
    np.random.shuffle(train_val_indices_orig)
    split_idx = int(0.8 * len(train_val_indices_orig))
    train_indices_orig = train_val_indices_orig[:split_idx]
    val_indices_orig = train_val_indices_orig[split_idx:]
    
    # Include augmented versions
    train_indices = np.concatenate([train_indices_orig, train_indices_orig + n_original])
    val_indices = np.concatenate([val_indices_orig, val_indices_orig + n_original])
    
    print(f"\nData split:")
    print(f"  Train: {len(train_indices)} samples (with augmentation)")
    print(f"  Val: {len(val_indices)} samples (with augmentation)")
    print(f"  Test: {len(test_indices)} samples (with augmentation)")
    
    # Normalize features (fit on train only)
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
    
    print(f"\nTraining Fusion Model v2 + Symmetrization ({input_dim} input features)...")
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
        
        # Validate - use symmetrized prediction
        model.eval()
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels_list.extend(batch_labels.numpy())
        
        val_loss = log_loss(val_labels_list, val_preds)
        val_auc = roc_auc_score(val_labels_list, val_preds)
        val_acc = accuracy_score(val_labels_list, (np.array(val_preds) > 0.5).astype(int))
        
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
    
    # Load best model and evaluate on test set with symmetrized inference
    print("\n" + "=" * 70)
    print("Evaluating best model on test set (2025+) with symmetrized inference...")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # --- Standard evaluation (no symmetrized inference) ---
    test_preds_standard = []
    test_labels_list = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            test_preds_standard.extend(torch.sigmoid(outputs).cpu().numpy())
            test_labels_list.extend(batch_labels.numpy())
    
    # Only use original (non-augmented) test samples for metrics
    n_test_orig = len(test_indices_orig)
    test_preds_orig = np.array(test_preds_standard[:n_test_orig])
    test_labels_orig = np.array(test_labels_list[:n_test_orig])
    
    std_loss = log_loss(test_labels_orig, test_preds_orig)
    std_auc = roc_auc_score(test_labels_orig, test_preds_orig)
    std_acc = accuracy_score(test_labels_orig, (test_preds_orig > 0.5).astype(int))
    
    print(f"\nStandard evaluation (original order only, {n_test_orig} matches):")
    print(f"  LogLoss: {std_loss:.4f}")
    print(f"  AUC: {std_auc:.4f}")
    print(f"  Accuracy: {std_acc:.4f}")
    
    # --- Symmetrized evaluation ---
    # Get predictions for both original and swapped test features
    test_features_orig = features[test_indices_orig]
    
    # Create swapped test features
    test_features_swapped = test_features_orig.copy()
    
    # Swap baseline pairs using column indices
    # Build index mapping from baseline_cols
    swap_index_pairs = []
    negate_indices = []
    for col1, col2 in BASELINE_SWAP_PAIRS:
        if col1 in baseline_cols and col2 in baseline_cols:
            idx1 = baseline_cols.index(col1)
            idx2 = baseline_cols.index(col2)
            swap_index_pairs.append((idx1, idx2))
    for col in BASELINE_NEGATE_COLS:
        if col in baseline_cols:
            negate_indices.append(baseline_cols.index(col))
    
    # Apply baseline swaps
    for idx1, idx2 in swap_index_pairs:
        test_features_swapped[:, idx1], test_features_swapped[:, idx2] = \
            test_features_swapped[:, idx2].copy(), test_features_swapped[:, idx1].copy()
    # Negate difference columns
    for idx in negate_indices:
        test_features_swapped[:, idx] = -test_features_swapped[:, idx]
    
    # Swap embedding segments
    emb_start = len(baseline_cols)  # 54
    test_features_swapped[:, emb_start:emb_start+64], test_features_swapped[:, emb_start+64:emb_start+128] = \
        test_features_swapped[:, emb_start+64:emb_start+128].copy(), test_features_swapped[:, emb_start:emb_start+64].copy()
    test_features_swapped[:, emb_start+128:emb_start+192] = -test_features_swapped[:, emb_start+128:emb_start+192]
    
    # Predict on swapped features
    swapped_preds = []
    with torch.no_grad():
        for i in range(0, len(test_features_swapped), 256):
            batch = torch.FloatTensor(test_features_swapped[i:i+256]).to(device)
            outputs = model(batch)
            swapped_preds.extend(torch.sigmoid(outputs).cpu().numpy())
    swapped_preds = np.array(swapped_preds)
    
    # Symmetrized: p = (p_orig + 1 - p_swapped) / 2
    sym_preds = (test_preds_orig + 1.0 - swapped_preds) / 2.0
    
    sym_loss = log_loss(test_labels_orig, sym_preds)
    sym_auc = roc_auc_score(test_labels_orig, sym_preds)
    sym_acc = accuracy_score(test_labels_orig, (sym_preds > 0.5).astype(int))
    
    print(f"\nSymmetrized evaluation (averaged both orderings, {n_test_orig} matches):")
    print(f"  LogLoss: {sym_loss:.4f}")
    print(f"  AUC: {sym_auc:.4f}")
    print(f"  Accuracy: {sym_acc:.4f}")
    
    # Calculate ECE for symmetrized predictions
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    print(f"\nCalibration (symmetrized):")
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i+1]
        mask = (sym_preds > low) & (sym_preds <= high) if i > 0 else (sym_preds >= low) & (sym_preds <= high)
        if mask.sum() > 0:
            bin_acc = test_labels_orig[mask].mean()
            bin_conf = sym_preds[mask].mean()
            bin_ece = abs(bin_acc - bin_conf) * mask.sum()
            ece += bin_ece
            print(f"  [{low:.1f}-{high:.1f}]: n={mask.sum()}, acc={bin_acc:.3f}, conf={bin_conf:.3f}, gap={abs(bin_acc-bin_conf):.3f}")
    ece /= len(test_labels_orig)
    print(f"  ECE: {ece:.4f}")
    
    # Save results
    test_results = {
        'standard_metrics': {
            'logloss': float(std_loss),
            'auc': float(std_auc),
            'accuracy': float(std_acc)
        },
        'symmetrized_metrics': {
            'logloss': float(sym_loss),
            'auc': float(sym_auc),
            'accuracy': float(sym_acc),
            'ece': float(ece)
        },
        'predictions_standard': [float(p) for p in test_preds_orig],
        'predictions_symmetrized': [float(p) for p in sym_preds],
        'labels': [float(l) for l in test_labels_orig],
        'match_ids': test_indices_orig.tolist()
    }
    
    with open('data/fusion_v2_sym_test_results.json', 'w') as f:
        json.dump(test_results, f)
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Test results saved to: data/fusion_v2_sym_test_results.json")
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON (test set 2025+, {n_test_orig} matches):")
    print(f"{'='*70}")
    print(f"{'Metric':<12} {'Baseline':<12} {'Fusion v2':<12} {'Fusion v2+Sym':<12}")
    print(f"{'-'*48}")
    print(f"{'LogLoss':<12} {'0.5869':<12} {'0.5582':<12} {sym_loss:<12.4f}")
    print(f"{'AUC':<12} {'0.7528':<12} {'0.7822':<12} {sym_auc:<12.4f}")
    print(f"{'Accuracy':<12} {'0.6868':<12} {'0.7131':<12} {sym_acc:<12.4f}")
    print(f"{'ECE':<12} {'N/A':<12} {'0.0217':<12} {ece:<12.4f}")


if __name__ == "__main__":
    train_model()
