#!/usr/bin/env python3
"""
Generate fusion model predictions for ALL matches and save as JSON.
This allows local comparison with odds on the common subset.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sys

# Model architectures (must match training scripts)

class FusionModel(nn.Module):
    """Standard fusion model (v2, v2+sym)."""
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


class FusionMLP(nn.Module):
    """Base MLP used by SymmetricFusionModel."""
    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.1]
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
    """Architecturally symmetric fusion model (v2+archsym)."""
    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        super().__init__()
        self.mlp = FusionMLP(input_dim, hidden_dims, dropout_rates)
    
    def forward(self, x, x_swapped=None):
        logit_orig = self.mlp(x)
        if x_swapped is not None:
            logit_swapped = self.mlp(x_swapped)
            sym_logit = (logit_orig - logit_swapped) / 2.0
            return sym_logit, logit_orig, logit_swapped
        return logit_orig


def swap_features(features, baseline_cols, n_baseline, n_embedding=192):
    """Swap team1↔team2 in features for symmetrization."""
    swapped = features.clone()
    
    swap_pairs = [
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
    
    col_to_idx = {col: i for i, col in enumerate(baseline_cols)}
    
    for col1, col2 in swap_pairs:
        if col1 in col_to_idx and col2 in col_to_idx:
            idx1, idx2 = col_to_idx[col1], col_to_idx[col2]
            swapped[:, idx1] = features[:, idx2]
            swapped[:, idx2] = features[:, idx1]
    
    if 'days_diff' in col_to_idx:
        idx = col_to_idx['days_diff']
        swapped[:, idx] = -features[:, idx]
    
    emb_start = n_baseline
    swapped[:, emb_start:emb_start+64] = features[:, emb_start+64:emb_start+128]
    swapped[:, emb_start+64:emb_start+128] = features[:, emb_start:emb_start+64]
    swapped[:, emb_start+128:emb_start+192] = -features[:, emb_start+128:emb_start+192]
    
    return swapped


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load baseline data
    print("Loading baseline data...")
    baseline_df = pd.read_csv('data/golgg_y_predicts.csv')
    print(f"Baseline: {len(baseline_df)} rows")
    
    # Load embeddings - format: list of {'match_id': str, 'embedding': list[192]}
    print("Loading embeddings...")
    with open('data/transformer_embeddings_v2.json', 'r') as f:
        embeddings_data = json.load(f)
    
    # Build match_id -> embedding mapping
    emb_map = {}
    for item in embeddings_data:
        emb_map[str(item['match_id'])] = item['embedding']
    print(f"Embeddings: {len(emb_map)} samples")
    
    # Exclude non-feature columns
    exclude_cols = ['team1_name', 'team2_name', 'date', 'golgg_match_id', 'y_true', 'match_id_str']
    baseline_cols = [c for c in baseline_df.columns if c not in exclude_cols]
    print(f"Baseline features: {len(baseline_cols)}")
    
    # Align embeddings with baseline by match_id
    match_ids = baseline_df['golgg_match_id'].values
    aligned_embeddings = []
    missing = 0
    for mid in match_ids:
        mid_str = str(mid)
        if mid_str in emb_map:
            aligned_embeddings.append(emb_map[mid_str])
        else:
            missing += 1
            aligned_embeddings.append([0.0] * 192)
    if missing > 0:
        print(f"WARNING: {missing} matches missing embeddings")
    
    embedding_features = np.array(aligned_embeddings, dtype=np.float32)
    print(f"Embedding features shape: {embedding_features.shape}")
    
    # Prepare baseline features
    baseline_features = baseline_df[baseline_cols].values.astype(np.float32)
    
    # --- Fusion v2 (no sym) ---
    print("\n--- Fusion v2 (no sym) ---")
    ckpt = torch.load('models/fusion_v2_best.pt', map_location=device, weights_only=False)
    input_dim = ckpt.get('input_dim', 246)
    model = FusionModel(input_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    saved_cols = ckpt.get('baseline_cols', baseline_cols)
    scaler_mean = ckpt['scaler_mean']
    scaler_scale = ckpt['scaler_scale']
    
    # Use saved baseline cols for feature alignment
    if saved_cols != baseline_cols:
        print(f"Adjusting baseline cols: model has {len(saved_cols)}, current {len(baseline_cols)}")
        baseline_features_aligned = baseline_df[saved_cols].values.astype(np.float32)
    else:
        baseline_features_aligned = baseline_features
    
    features = np.hstack([baseline_features_aligned, embedding_features])
    features = (features - scaler_mean) / scaler_scale
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        probs_v2 = torch.sigmoid(model(features_tensor)).cpu().numpy()
    print(f"Fusion v2: mean prob = {probs_v2.mean():.4f}")
    
    # --- Fusion v2+SymAug (symmetrized inference) ---
    print("\n--- Fusion v2+SymAug ---")
    ckpt_sym = torch.load('models/fusion_v2_sym_best.pt', map_location=device, weights_only=False)
    model_sym = FusionModel(ckpt_sym.get('input_dim', 246))
    model_sym.load_state_dict(ckpt_sym['model_state_dict'])
    model_sym.to(device)
    model_sym.eval()
    
    saved_cols_sym = ckpt_sym.get('baseline_cols', baseline_cols)
    if saved_cols_sym != saved_cols:
        print(f"Sym model has different baseline cols, realigning...")
        baseline_features_sym = baseline_df[saved_cols_sym].values.astype(np.float32)
        features_sym = np.hstack([baseline_features_sym, embedding_features])
        features_sym = (features_sym - ckpt_sym['scaler_mean']) / ckpt_sym['scaler_scale']
        features_tensor_sym = torch.tensor(features_sym, dtype=torch.float32).to(device)
    else:
        features_tensor_sym = features_tensor
    
    with torch.no_grad():
        logits_orig = model_sym(features_tensor_sym)
        probs_orig = torch.sigmoid(logits_orig)
        
        swapped = swap_features(features_tensor_sym, saved_cols_sym, len(saved_cols_sym))
        logits_swap = model_sym(swapped)
        probs_swap = torch.sigmoid(logits_swap)
        
        probs_sym = ((probs_orig + (1 - probs_swap)) / 2).cpu().numpy()
    print(f"Fusion SymAug: mean prob = {probs_sym.mean():.4f}")
    
    # --- Fusion v2+ArchSym ---
    print("\n--- Fusion v2+ArchSym ---")
    ckpt_arch = torch.load('models/fusion_v2_archsym_best.pt', map_location=device, weights_only=False)
    model_arch = SymmetricFusionModel(ckpt_arch.get('input_dim', 246))
    model_arch.load_state_dict(ckpt_arch['model_state_dict'])
    model_arch.to(device)
    model_arch.eval()
    
    saved_cols_arch = ckpt_arch.get('baseline_cols', baseline_cols)
    if saved_cols_arch != saved_cols:
        print(f"ArchSym model has different baseline cols, realigning...")
        baseline_features_arch = baseline_df[saved_cols_arch].values.astype(np.float32)
        features_arch = np.hstack([baseline_features_arch, embedding_features])
        features_arch = (features_arch - ckpt_arch['scaler_mean']) / ckpt_arch['scaler_scale']
        features_tensor_arch = torch.tensor(features_arch, dtype=torch.float32).to(device)
    else:
        features_tensor_arch = features_tensor
    
    with torch.no_grad():
        # Use architectural symmetrization: pass both original and swapped features
        swapped_arch = swap_features(features_tensor_arch, saved_cols_arch, len(saved_cols_arch))
        sym_logit, _, _ = model_arch(features_tensor_arch, swapped_arch)
        probs_arch = torch.sigmoid(sym_logit).cpu().numpy()
    print(f"Fusion ArchSym: mean prob = {probs_arch.mean():.4f}")
    
    # Save predictions
    results = {}
    for i, mid in enumerate(match_ids):
        results[str(mid)] = {
            'fusion_v2': float(probs_v2[i]),
            'fusion_v2_sym': float(probs_sym[i]),
            'fusion_v2_archsym': float(probs_arch[i]),
            'y_true': int(baseline_df.iloc[i]['y_true']),
            'player_elo': float(baseline_df.iloc[i]['player_elo']),
            'date': str(baseline_df.iloc[i]['date']),
        }
    
    output_path = 'data/fusion_predictions_all.json'
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"\nSaved {len(results)} predictions to {output_path}")
    print(f"Sample: {list(results.items())[:2]}")

if __name__ == '__main__':
    main()
