import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import sys

# Load model
ckpt = torch.load('models/fusion_v2_best.pt', map_location='cpu', weights_only=False)
baseline_cols = ckpt['baseline_cols']
scaler_mean = ckpt['scaler_mean']
scaler_scale = ckpt['scaler_scale']
if hasattr(scaler_mean, 'numpy'):
    scaler_mean = scaler_mean.numpy()
    scaler_scale = scaler_scale.numpy()

class FusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = FusionModel(246)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load data
df = pd.read_csv('data/golgg_y_predicts.csv')
with open('data/transformer_embeddings_v2.json', 'r') as f:
    emb_data = json.load(f)
emb_lookup = {str(item['match_id']): item['embedding'] for item in emb_data}

df['date'] = pd.to_datetime(df['date'])
test_df = df[df['date'] >= '2025-01-01'].copy()
test_df = test_df[test_df['golgg_match_id'].astype(str).isin(emb_lookup)].copy()
test_df = test_df.sort_values('date').reset_index(drop=True)
print(f"Test samples: {len(test_df)}", flush=True)

# Build swap indices for baseline pairs
pair_swaps = [
    ('team_elo_r1', 'team_elo_r2'), ('player_elo_min1', 'player_elo_min2'),
    ('team_gl_r1', 'team_gl_r2'), ('team_gl_rd1', 'team_gl_rd2'),
    ('player_gl_max1', 'player_gl_max2'), ('player_gl_rd_avg1', 'player_gl_rd_avg2'),
    ('team_ts_mu1', 'team_ts_mu2'), ('team_ts_sigma1', 'team_ts_sigma2'),
    ('player_ts_sigma_avg1', 'player_ts_sigma_avg2'),
    ('team_os_mu1', 'team_os_mu2'), ('team_os_sigma1', 'team_os_sigma2'),
    ('player_os_sigma_avg1', 'player_os_sigma_avg2'),
    ('team_pl_mu1', 'team_pl_mu2'), ('team_pl_sigma1', 'team_pl_sigma2'),
    ('player_pl_sigma_avg1', 'player_pl_sigma_avg2'),
    ('team_tm_mu1', 'team_tm_mu2'), ('team_tm_sigma1', 'team_tm_sigma2'),
    ('player_tm_sigma_avg1', 'player_tm_sigma_avg2'),
    ('days_since_last_1', 'days_since_last_2'), ('team1_id', 'team2_id'),
]
swap_indices = []
for c1, c2 in pair_swaps:
    if c1 in baseline_cols and c2 in baseline_cols:
        swap_indices.append((baseline_cols.index(c1), baseline_cols.index(c2)))

days_diff_idx = baseline_cols.index('days_diff') if 'days_diff' in baseline_cols else None

# Process in batches
BATCH = 512
all_preds_orig = []
all_preds_swap = []
all_labels = []

for start in range(0, len(test_df), BATCH):
    end = min(start + BATCH, len(test_df))
    batch_df = test_df.iloc[start:end]
    
    feats_orig = []
    feats_swap = []
    batch_labels = []
    
    for _, row in batch_df.iterrows():
        mid = str(int(row['golgg_match_id']))
        emb = np.array(emb_lookup[mid])
        baseline = row[baseline_cols].values.astype(float)
        feat_orig = np.concatenate([baseline, emb])
        
        # Swap
        baseline_swap = baseline.copy()
        for i1, i2 in swap_indices:
            baseline_swap[i1], baseline_swap[i2] = baseline_swap[i2], baseline_swap[i1]
        if days_diff_idx is not None:
            baseline_swap[days_diff_idx] = -baseline_swap[days_diff_idx]
        
        emb_swap = emb.copy()
        emb_swap[:64], emb_swap[64:128] = emb[64:128].copy(), emb[:64].copy()
        emb_swap[128:192] = -emb_swap[128:192]
        
        feat_swap = np.concatenate([baseline_swap, emb_swap])
        feats_orig.append(feat_orig)
        feats_swap.append(feat_swap)
        batch_labels.append(row['y_true'])
    
    fo = (np.array(feats_orig) - scaler_mean) / scaler_scale
    fs = (np.array(feats_swap) - scaler_mean) / scaler_scale
    
    with torch.no_grad():
        lo = model(torch.FloatTensor(fo)).numpy()
        ls = model(torch.FloatTensor(fs)).numpy()
    
    po = 1 / (1 + np.exp(-lo))
    ps = 1 / (1 + np.exp(-ls))
    all_preds_orig.extend(po)
    all_preds_swap.extend(ps)
    all_labels.extend(batch_labels)
    
    if start % 2000 == 0:
        print(f"  Processed {start}/{len(test_df)}", flush=True)

po = np.array(all_preds_orig)
ps = np.array(all_preds_swap)
labels = np.array(all_labels)

# Asymmetry
diff = po - (1 - ps)
print(f"\n=== ASYMMETRY ===", flush=True)
print(f"Mean |f(t1,t2) - (1-f(t2,t1))|: {np.mean(np.abs(diff)):.6f}", flush=True)
print(f"Max asymmetry: {np.max(np.abs(diff)):.6f}", flush=True)
print(f"Std asymmetry: {np.std(diff):.6f}", flush=True)
print(f"Mean asymmetry: {np.mean(diff):.6f}", flush=True)

# Disagreements
orig_class = (po > 0.5).astype(int)
swap_class_corrected = 1 - (ps > 0.5).astype(int)
disagree = np.sum(orig_class != swap_class_corrected)
print(f"Prediction disagreements: {disagree}/{len(labels)} ({100*disagree/len(labels):.2f}%)", flush=True)

# Symmetrized
p_sym = (po + (1 - ps)) / 2

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
print(f"\n=== RESULTS ===", flush=True)
print(f"Original:  LL={log_loss(labels, po):.4f}  AUC={roc_auc_score(labels, po):.4f}  Acc={accuracy_score(labels, (po>0.5).astype(int)):.4f}", flush=True)
print(f"Symmetr:   LL={log_loss(labels, p_sym):.4f}  AUC={roc_auc_score(labels, p_sym):.4f}  Acc={accuracy_score(labels, (p_sym>0.5).astype(int)):.4f}", flush=True)

print(f"\n=== TEAM1 BIAS ===", flush=True)
print(f"Mean pred team1 wins: {np.mean(po):.4f}", flush=True)
print(f"Actual team1 win rate: {np.mean(labels):.4f}", flush=True)
print(f"Mean pred swap: {np.mean(ps):.4f}", flush=True)

# Distribution of asymmetry
print(f"\n=== ASYMMETRY DISTRIBUTION ===", flush=True)
for thresh in [0.01, 0.02, 0.05, 0.10, 0.20]:
    n = np.sum(np.abs(diff) > thresh)
    print(f"|diff| > {thresh}: {n} ({100*n/len(diff):.2f}%)", flush=True)
