#!/usr/bin/env python3
"""
Fusion Model v2 with Architectural Symmetrization.
Instead of data augmentation, the network architecture enforces symmetry:
  p = (f(t1,t2) + 1 - f(t2,t1)) / 2
Plus an optional symmetry loss penalty: λ * |f(t1,t2) + f(t2,t1) - 1|

This ensures the model is EXACTLY symmetric by construction.
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
MODEL_PATH = "models/fusion_v2_archsym_best.pt"

# Features to exclude from baseline (non-numeric or identifiers)
EXCLUDE_COLS = ['team1_name', 'team2_name', 'date', 'golgg_match_id']

# Baseline column swap pairs
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

# Columns that get negated when swapping
BASELINE_NEGATE_COLS = ['days_diff']

# Hyperparameters
SYMMETRY_LOSS_WEIGHT = 0.1  # Weight for symmetry penalty loss
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATES = (0.3, 0.2, 0.1)
HIDDEN_DIMS = (256, 128, 64)
PATIENCE = 15
MAX_EPOCHS = 50
BATCH_SIZE = 256


class FusionDataset(Dataset):
    def __init__(self, features, labels, swapped_features=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.swapped_features = torch.FloatTensor(swapped_features) if swapped_features is not None else None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.swapped_features is not None:
            return self.features[idx], self.labels[idx], self.swapped_features[idx]
        return self.features[idx], self.labels[idx]


class FusionMLP(nn.Module):
    """Base MLP that processes a single feature vector."""
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout_rates=DROPOUT_RATES):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class SymmetricFusionModel(nn.Module):
    """
    Architecturally symmetric fusion model.
    
    Forward pass computes:
        p = sigmoid( (f(t1,t2) + (-f(t2,t1))) / 2 )
    
    This is equivalent to:
        p = (sigmoid(f(t1,t2)) + 1 - sigmoid(f(t2,t1))) / 2
    
    But done in logit space for better gradient flow.
    
    The model is EXACTLY symmetric by construction:
        p(t1,t2) = 1 - p(t2,t1)
    """
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout_rates=DROPOUT_RATES):
        super().__init__()
        self.mlp = FusionMLP(input_dim, hidden_dims, dropout_rates)
    
    def forward(self, x, x_swapped=None):
        """
        If x_swapped is provided, uses architectural symmetrization.
        Otherwise, just runs the base MLP (for inference without pre-computed swap).
        """
        logit_orig = self.mlp(x)
        
        if x_swapped is not None:
            logit_swapped = self.mlp(x_swapped)
            # Symmetrized logit: (logit(t1,t2) - logit(t2,t1)) / 2
            # This ensures: sigmoid(sym_logit) + sigmoid(-sym_logit) = 1
            # i.e., p(t1,t2) = 1 - p(t2,t1) exactly
            sym_logit = (logit_orig - logit_swapped) / 2.0
            return sym_logit, logit_orig, logit_swapped
        
        return logit_orig, None, None
    
    def predict_symmetric(self, x, swap_fn, device, batch_size=256):
        """
        Inference with symmetrization. swap_fn creates the swapped version of features.
        """
        self.eval()
        n = len(x)
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.FloatTensor(x[i:i+batch_size]).to(device)
                batch_swapped = torch.FloatTensor(swap_fn(x[i:i+batch_size])).to(device)
                
                logit_orig = self.mlp(batch)
                logit_swapped = self.mlp(batch_swapped)
                sym_logit = (logit_orig - logit_swapped) / 2.0
                
                preds = torch.sigmoid(sym_logit).cpu().numpy()
                all_preds.extend(preds)
        
        return np.array(all_preds)


def swap_features(features, baseline_cols, n_baseline):
    """
    Swap features for team1↔team2.
    features: numpy array [n_samples, n_features]
    Returns swapped copy.
    """
    swapped = features.copy()
    
    # Swap baseline pairs
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
    
    for idx1, idx2 in swap_index_pairs:
        swapped[:, idx1], swapped[:, idx2] = swapped[:, idx2].copy(), swapped[:, idx1].copy()
    for idx in negate_indices:
        swapped[:, idx] = -swapped[:, idx]
    
    # Swap embedding segments: [emb_t1(64), emb_t2(64), emb_diff(64)]
    emb_start = n_baseline
    swapped[:, emb_start:emb_start+64], swapped[:, emb_start+64:emb_start+128] = \
        swapped[:, emb_start+64:emb_start+128].copy(), swapped[:, emb_start:emb_start+64].copy()
    swapped[:, emb_start+128:emb_start+192] = -swapped[:, emb_start+128:emb_start+192]
    
    return swapped


def load_and_merge_data():
    """Load baseline CSV and transformer embeddings, merge by match_id."""
    print("Loading baseline features...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    print(f"  Baseline: {len(baseline_df)} rows, {len(baseline_df.columns)} columns")
    
    print("Loading transformer embeddings...")
    with open(EMBEDDINGS_JSON, 'r') as f:
        embeddings_data = json.load(f)
    print(f"  Embeddings: {len(embeddings_data)} samples")
    
    # Create embeddings lookup
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
    # Load data (NO augmentation)
    features, labels, dates, baseline_cols = load_and_merge_data()
    n_baseline = len(baseline_cols)
    
    # Create swapped features for architectural symmetrization
    print("\nPre-computing swapped features for symmetrization...")
    swapped_features = swap_features(features, baseline_cols, n_baseline)
    
    # Chronological split
    test_mask = dates >= '2025-01-01'
    test_indices = np.where(test_mask)[0]
    train_val_indices = np.where(~test_mask)[0]
    
    # Split train/val (80/20)
    np.random.seed(42)
    np.random.shuffle(train_val_indices)
    split_idx = int(0.8 * len(train_val_indices))
    train_indices = train_val_indices[:split_idx]
    val_indices = train_val_indices[split_idx:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples (2025+)")
    
    # Normalize features (fit on train only)
    scaler = StandardScaler()
    features[train_indices] = scaler.fit_transform(features[train_indices])
    features[val_indices] = scaler.transform(features[val_indices])
    features[test_indices] = scaler.transform(features[test_indices])
    
    # Also normalize swapped features with same scaler
    swapped_features[train_indices] = scaler.transform(swapped_features[train_indices])
    swapped_features[val_indices] = scaler.transform(swapped_features[val_indices])
    swapped_features[test_indices] = scaler.transform(swapped_features[test_indices])
    
    # Create datasets (with swapped features)
    train_dataset = FusionDataset(features[train_indices], labels[train_indices], swapped_features[train_indices])
    val_dataset = FusionDataset(features[val_indices], labels[val_indices], swapped_features[val_indices])
    test_dataset = FusionDataset(features[test_indices], labels[test_indices], swapped_features[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    input_dim = features.shape[1]
    model = SymmetricFusionModel(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining Architecturally-Symmetric Fusion Model v2 ({input_dim} input features)...")
    print(f"Symmetry loss weight: {SYMMETRY_LOSS_WEIGHT}")
    print("=" * 80)
    
    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_losses = []
        train_bce_losses = []
        train_sym_losses = []
        
        for batch_features, batch_labels, batch_swapped in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_swapped = batch_swapped.to(device)
            
            optimizer.zero_grad()
            
            # Forward with symmetrization
            sym_logit, logit_orig, logit_swapped = model(batch_features, batch_swapped)
            
            # BCE loss on symmetrized logits
            bce_loss = criterion(sym_logit, batch_labels)
            
            # Symmetry penalty: |logit(t1,t2) + logit(t2,t1)|
            # For a perfectly symmetric model: logit(t1,t2) = -logit(t2,t1)
            # So logit(t1,t2) + logit(t2,t1) should be 0
            sym_penalty = torch.mean(torch.abs(logit_orig + logit_swapped))
            
            # Total loss
            loss = bce_loss + SYMMETRY_LOSS_WEIGHT * sym_penalty
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_bce_losses.append(bce_loss.item())
            train_sym_losses.append(sym_penalty.item())
        
        # Validate
        model.eval()
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for batch_features, batch_labels, batch_swapped in val_loader:
                batch_features = batch_features.to(device)
                batch_swapped = batch_swapped.to(device)
                
                sym_logit, _, _ = model(batch_features, batch_swapped)
                val_preds.extend(torch.sigmoid(sym_logit).cpu().numpy())
                val_labels_list.extend(batch_labels.numpy())
        
        val_loss = log_loss(val_labels_list, val_preds)
        val_auc = roc_auc_score(val_labels_list, val_preds)
        val_acc = accuracy_score(val_labels_list, (np.array(val_preds) > 0.5).astype(int))
        
        train_loss = np.mean(train_losses)
        train_bce = np.mean(train_bce_losses)
        train_sym = np.mean(train_sym_losses)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:2d} | Loss {train_loss:.4f} (BCE {train_bce:.4f} + Sym {train_sym:.4f}) | "
              f"Val LL {val_loss:.4f} | AUC {val_auc:.4f} | Acc {val_acc:.4f}", end="")
        
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
                'scaler_scale': scaler.scale_,
                'symmetry_loss_weight': SYMMETRY_LOSS_WEIGHT,
                'architecture': 'architectural_symmetrization',
            }, MODEL_PATH)
            print(" ⭐ (best)")
            patience_counter = 0
        else:
            print()
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating best model on test set (2025+)...")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate with architectural symmetrization
    test_preds = []
    test_labels_list = []
    with torch.no_grad():
        for batch_features, batch_labels, batch_swapped in test_loader:
            batch_features = batch_features.to(device)
            batch_swapped = batch_swapped.to(device)
            
            sym_logit, logit_orig, logit_swapped = model(batch_features, batch_swapped)
            test_preds.extend(torch.sigmoid(sym_logit).cpu().numpy())
            test_labels_list.extend(batch_labels.numpy())
    
    test_preds = np.array(test_preds)
    test_labels_arr = np.array(test_labels_list)
    
    test_loss = log_loss(test_labels_arr, test_preds)
    test_auc = roc_auc_score(test_labels_arr, test_preds)
    test_acc = accuracy_score(test_labels_arr, (test_preds > 0.5).astype(int))
    
    print(f"\nArchitecturally-Symmetric Fusion v2 (test set 2025+, {len(test_indices)} matches):")
    print(f"  LogLoss: {test_loss:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    # Verify symmetry: p(t1,t2) should equal 1 - p(t2,t1)
    print("\nVerifying symmetry...")
    orig_preds = []
    swapped_preds = []
    with torch.no_grad():
        for batch_features, batch_labels, batch_swapped in test_loader:
            batch_features = batch_features.to(device)
            batch_swapped = batch_swapped.to(device)
            
            _, logit_orig, logit_swapped = model(batch_features, batch_swapped)
            orig_preds.extend(torch.sigmoid(logit_orig).cpu().numpy())
            swapped_preds.extend(torch.sigmoid(logit_swapped).cpu().numpy())
    
    orig_preds = np.array(orig_preds)
    swapped_preds = np.array(swapped_preds)
    
    # Symmetry: p(t1,t2) + p(t2,t1) should equal 1
    symmetry_diff = np.abs(orig_preds + swapped_preds - 1.0)
    print(f"  Mean |p(t1,t2) + p(t2,t1) - 1|: {symmetry_diff.mean():.6f}")
    print(f"  Max |p(t1,t2) + p(t2,t1) - 1|: {symmetry_diff.max():.6f}")
    print(f"  Symmetrized pred: Mean |p_sym - (p_orig + 1 - p_swapped)/2|: {np.abs(test_preds - (orig_preds + 1 - swapped_preds)/2).mean():.6f}")
    
    # Disagreement analysis
    orig_class = (orig_preds > 0.5).astype(int)
    swapped_class = (1 - swapped_preds > 0.5).astype(int)
    disagreements = np.sum(orig_class != swapped_class)
    print(f"  Prediction disagreements (orig vs sym): {disagreements}/{len(orig_preds)} ({100*disagreements/len(orig_preds):.2f}%)")
    
    # Calculate ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    print(f"\nCalibration:")
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i+1]
        mask = (test_preds > low) & (test_preds <= high) if i > 0 else (test_preds >= low) & (test_preds <= high)
        if mask.sum() > 0:
            bin_acc = test_labels_arr[mask].mean()
            bin_conf = test_preds[mask].mean()
            bin_ece = abs(bin_acc - bin_conf) * mask.sum()
            ece += bin_ece
            print(f"  [{low:.1f}-{high:.1f}]: n={mask.sum()}, acc={bin_acc:.3f}, conf={bin_conf:.3f}, gap={abs(bin_acc-bin_conf):.3f}")
    ece /= len(test_labels_arr)
    print(f"  ECE: {ece:.4f}")
    
    # Save results
    test_results = {
        'model': 'fusion_v2_architectural_symmetrization',
        'symmetry_loss_weight': SYMMETRY_LOSS_WEIGHT,
        'metrics': {
            'logloss': float(test_loss),
            'auc': float(test_auc),
            'accuracy': float(test_acc),
            'ece': float(ece),
        },
        'symmetry_verification': {
            'mean_asymmetry': float(symmetry_diff.mean()),
            'max_asymmetry': float(symmetry_diff.max()),
            'disagreements': int(disagreements),
            'disagreement_pct': float(100 * disagreements / len(orig_preds)),
        },
        'predictions': [float(p) for p in test_preds],
        'labels': [float(l) for l in test_labels_arr],
        'match_ids': test_indices.tolist(),
    }
    
    with open('data/fusion_v2_archsym_test_results.json', 'w') as f:
        json.dump(test_results, f)
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Test results saved to: data/fusion_v2_archsym_test_results.json")
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON (test set 2025+, {len(test_indices)} matches):")
    print(f"{'='*80}")
    print(f"{'Metric':<12} {'Baseline':<12} {'Fusion v2':<12} {'Fusion+SymAug':<12} {'Fusion+ArchSym':<12}")
    print(f"{'-'*60}")
    print(f"{'LogLoss':<12} {'0.5869':<12} {'0.5582':<12} {'0.5575':<12} {test_loss:<12.4f}")
    print(f"{'AUC':<12} {'0.7528':<12} {'0.7822':<12} {'0.7817':<12} {test_auc:<12.4f}")
    print(f"{'Accuracy':<12} {'0.6868':<12} {'0.7131':<12} {'0.7086':<12} {test_acc:<12.4f}")
    print(f"{'ECE':<12} {'N/A':<12} {'0.0217':<12} {'0.0225':<12} {ece:<12.4f}")


if __name__ == "__main__":
    train_model()
