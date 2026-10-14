"""
Compare ALL models on the SAME common subset of matches (those with valid odds).
This ensures fair comparison - no different sample sizes.
"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss

# Load data
print("Loading data...")
odds = pd.read_csv('data/odds.csv')
baseline = pd.read_csv('data/golgg_y_predicts.csv')

# Load fusion predictions
with open('data/fusion_predictions_all.json', 'r') as f:
    fusion_preds = json.load(f)
print(f"Fusion predictions: {len(fusion_preds)} matches")

print(f"Odds: {len(odds)} rows, Baseline: {len(baseline)} rows")

# Merge odds with baseline on golgg_match_id
merged = odds.merge(baseline, on='golgg_match_id', how='inner', suffixes=('_odds', '_base'))
print(f"Merged: {len(merged)} rows")

# Check team alignment
merged['teams_match'] = (merged['golgg_team1'] == merged['team1_name']) & (merged['golgg_team2'] == merged['team2_name'])
merged['teams_swapped'] = (merged['golgg_team1'] == merged['team2_name']) & (merged['golgg_team2'] == merged['team1_name'])
merged['teams_neither'] = ~merged['teams_match'] & ~merged['teams_swapped']

print(f"\nTeam alignment:")
print(f"  Match (same order): {merged['teams_match'].sum()}")
print(f"  Swapped: {merged['teams_swapped'].sum()}")
print(f"  Neither: {merged['teams_neither'].sum()}")

# Filter to rows with valid avg odds
merged = merged[merged['avg_odds_home'].notna() & merged['avg_odds_away'].notna()].copy()
print(f"\nWith valid avg odds: {len(merged)} rows")

# Compute bookmaker implied probabilities (aligned to baseline team1)
merged['bm_p_team1'] = np.where(
    merged['teams_match'],
    1.0 / merged['avg_odds_home'],
    np.where(merged['teams_swapped'], 1.0 / merged['avg_odds_away'], np.nan)
)
merged['bm_p_team2'] = np.where(
    merged['teams_match'],
    1.0 / merged['avg_odds_away'],
    np.where(merged['teams_swapped'], 1.0 / merged['avg_odds_home'], np.nan)
)

# Remove rows where teams don't match
merged = merged[merged['bm_p_team1'].notna()].copy()
print(f"After removing non-matching teams: {len(merged)} rows")

# Bookmaker margin
merged['bm_margin'] = merged['bm_p_team1'] + merged['bm_p_team2'] - 1
print(f"Average bookmaker margin: {merged['bm_margin'].mean():.4f}")

# Normalize probabilities (remove margin)
merged['bm_p_team1_norm'] = merged['bm_p_team1'] / (merged['bm_p_team1'] + merged['bm_p_team2'])

# Actual odds aligned to baseline team1/team2
merged['odds_team1'] = np.where(
    merged['teams_match'],
    merged['avg_odds_home'],
    np.where(merged['teams_swapped'], merged['avg_odds_away'], np.nan)
)
merged['odds_team2'] = np.where(
    merged['teams_match'],
    merged['avg_odds_away'],
    np.where(merged['teams_swapped'], merged['avg_odds_home'], np.nan)
)

# Add fusion predictions
merged['match_id_str'] = merged['golgg_match_id'].astype(str)
merged['fusion_v2'] = merged['match_id_str'].map(lambda x: fusion_preds.get(x, {}).get('fusion_v2', np.nan))
merged['fusion_v2_sym'] = merged['match_id_str'].map(lambda x: fusion_preds.get(x, {}).get('fusion_v2_sym', np.nan))
merged['fusion_v2_archsym'] = merged['match_id_str'].map(lambda x: fusion_preds.get(x, {}).get('fusion_v2_archsym', np.nan))

# Check fusion coverage
fusion_coverage = merged['fusion_v2'].notna().mean() * 100
print(f"Fusion predictions coverage: {fusion_coverage:.1f}%")

# Filter to 2025+ test set
merged['date_base'] = pd.to_datetime(merged['date'])
test = merged[merged['date_base'] >= '2025-01-01'].copy()
print(f"\n2025+ test set: {len(test)} rows")

# Also check all-data stats
all_data = merged.copy()

# ============================================
# Metrics computation
# ============================================
def compute_metrics(y_true, y_pred, name="Model"):
    """Compute LogLoss, AUC, Accuracy, ECE, Brier score."""
    ll = log_loss(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = float('nan')
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    brier = brier_score_loss(y_true, y_pred)
    
    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred > bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_pred) * abs(y_pred[mask].mean() - y_true[mask].mean())
    
    return {'name': name, 'logloss': ll, 'auc': auc, 'accuracy': acc, 'ece': ece, 'brier': brier}

def print_metrics(metrics):
    print(f"  LogLoss:  {metrics['logloss']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ECE:      {metrics['ece']:.4f}")
    print(f"  Brier:    {metrics['brier']:.4f}")

# ============================================
# Evaluate on 2025+ test set (SAME subset)
# ============================================
print("\n" + "="*70)
print(f"ALL MODELS ON SAME SUBSET: 2025+ with valid odds ({len(test)} matches)")
print("="*70)

y_true = test['y_true'].values

results = []

# 1. Bookmaker
bm = compute_metrics(y_true, test['bm_p_team1_norm'].values, "Bookmaker (avg odds)")
results.append(bm)
print(f"\n{bm['name']}:")
print_metrics(bm)

# 2. Baseline (Player Elo)
base = compute_metrics(y_true, test['player_elo'].values, "Baseline (Player Elo)")
results.append(base)
print(f"\n{base['name']}:")
print_metrics(base)

# 3. Fusion v2 (no sym)
mask_fv2 = test['fusion_v2'].notna()
if mask_fv2.sum() > 0:
    fv2 = compute_metrics(y_true[mask_fv2], test['fusion_v2'].values[mask_fv2], "Fusion v2 (no sym)")
    results.append(fv2)
    print(f"\n{fv2['name']} ({mask_fv2.sum()} matches):")
    print_metrics(fv2)

# 4. Fusion v2+SymAug
mask_fsym = test['fusion_v2_sym'].notna()
if mask_fsym.sum() > 0:
    fsym = compute_metrics(y_true[mask_fsym], test['fusion_v2_sym'].values[mask_fsym], "Fusion v2+SymAug")
    results.append(fsym)
    print(f"\n{fsym['name']} ({mask_fsym.sum()} matches):")
    print_metrics(fsym)

# 5. Fusion v2+ArchSym
mask_farch = test['fusion_v2_archsym'].notna()
if mask_farch.sum() > 0:
    farch = compute_metrics(y_true[mask_farch], test['fusion_v2_archsym'].values[mask_farch], "Fusion v2+ArchSym")
    results.append(farch)
    print(f"\n{farch['name']} ({mask_farch.sum()} matches):")
    print_metrics(farch)

# ============================================
# Also evaluate on the COMMON subset where ALL models have predictions
# ============================================
print("\n" + "="*70)
common_mask = test['fusion_v2'].notna() & test['fusion_v2_sym'].notna() & test['fusion_v2_archsym'].notna()
common_test = test[common_mask].copy()
print(f"COMMON SUBSET (all models available): {len(common_test)} matches")
print("="*70)

y_true_common = common_test['y_true'].values

results_common = []

# Bookmaker
bm_c = compute_metrics(y_true_common, common_test['bm_p_team1_norm'].values, "Bookmaker (avg odds)")
results_common.append(bm_c)
print(f"\n{bm_c['name']}:")
print_metrics(bm_c)

# Baseline
base_c = compute_metrics(y_true_common, common_test['player_elo'].values, "Baseline (Player Elo)")
results_common.append(base_c)
print(f"\n{base_c['name']}:")
print_metrics(base_c)

# Fusion v2
fv2_c = compute_metrics(y_true_common, common_test['fusion_v2'].values, "Fusion v2 (no sym)")
results_common.append(fv2_c)
print(f"\n{fv2_c['name']}:")
print_metrics(fv2_c)

# Fusion v2+SymAug
fsym_c = compute_metrics(y_true_common, common_test['fusion_v2_sym'].values, "Fusion v2+SymAug")
results_common.append(fsym_c)
print(f"\n{fsym_c['name']}:")
print_metrics(fsym_c)

# Fusion v2+ArchSym
farch_c = compute_metrics(y_true_common, common_test['fusion_v2_archsym'].values, "Fusion v2+ArchSym")
results_common.append(farch_c)
print(f"\n{farch_c['name']}:")
print_metrics(farch_c)

# ============================================
# Summary table
# ============================================
print("\n" + "="*70)
print(f"SUMMARY TABLE - Common subset ({len(common_test)} matches, 2025+)")
print("="*70)
print(f"{'Model':<25} {'LogLoss':>8} {'AUC':>8} {'Accuracy':>8} {'ECE':>8} {'Brier':>8}")
print("-" * 70)
for r in results_common:
    print(f"{r['name']:<25} {r['logloss']:>8.4f} {r['auc']:>8.4f} {r['accuracy']:>8.4f} {r['ece']:>8.4f} {r['brier']:>8.4f}")

# ============================================
# Betting simulation
# ============================================
print("\n" + "="*70)
print("BETTING SIMULATION (2025+, common subset)")
print("="*70)

def betting_simulation(y_true, model_probs, odds_t1, odds_t2, name="Model"):
    """Simulate betting using actual decimal odds."""
    print(f"\n{name}:")
    print(f"{'Edge>':>8} {'Bets':>6} {'Wins':>6} {'Profit':>8} {'ROI':>8} {'WinRate':>8}")
    
    for min_edge in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        profit = 0.0
        bets = 0
        wins = 0
        
        for i in range(len(y_true)):
            p_model = model_probs[i]
            o1 = odds_t1[i]
            o2 = odds_t2[i]
            
            if np.isnan(o1) or np.isnan(o2) or o1 <= 1 or o2 <= 1:
                continue
            
            imp1 = 1.0 / o1
            imp2 = 1.0 / o2
            
            edge1 = p_model - imp1
            edge2 = (1 - p_model) - imp2
            
            if edge1 > min_edge:
                bets += 1
                if y_true[i] == 1:
                    profit += (o1 - 1)
                    wins += 1
                else:
                    profit -= 1
            elif edge2 > min_edge:
                bets += 1
                if y_true[i] == 0:
                    profit += (o2 - 1)
                    wins += 1
                else:
                    profit -= 1
        
        roi = profit / bets * 100 if bets > 0 else 0
        win_rate = wins / bets * 100 if bets > 0 else 0
        print(f"{min_edge:>7.2f} {bets:>6d} {wins:>6d} {profit:>8.1f} {roi:>7.1f}% {win_rate:>7.1f}%")

# Run betting simulation for each model on common subset
betting_simulation(
    y_true_common, common_test['player_elo'].values,
    common_test['odds_team1'].values, common_test['odds_team2'].values,
    "Baseline (Player Elo)"
)

betting_simulation(
    y_true_common, common_test['fusion_v2'].values,
    common_test['odds_team1'].values, common_test['odds_team2'].values,
    "Fusion v2 (no sym)"
)

betting_simulation(
    y_true_common, common_test['fusion_v2_sym'].values,
    common_test['odds_team1'].values, common_test['odds_team2'].values,
    "Fusion v2+SymAug"
)

betting_simulation(
    y_true_common, common_test['fusion_v2_archsym'].values,
    common_test['odds_team1'].values, common_test['odds_team2'].values,
    "Fusion v2+ArchSym"
)

# ============================================
# Also: all-data comparison (not just 2025+)
# ============================================
print("\n" + "="*70)
print(f"ALL DATA with valid odds ({len(all_data)} matches)")
print("="*70)

all_results = []

bm_all = compute_metrics(all_data['y_true'].values, all_data['bm_p_team1_norm'].values, "Bookmaker (avg odds)")
all_results.append(bm_all)

base_all = compute_metrics(all_data['y_true'].values, all_data['player_elo'].values, "Baseline (Player Elo)")
all_results.append(base_all)

# Fusion on all data with valid odds
mask_all_fv2 = all_data['fusion_v2'].notna()
if mask_all_fv2.sum() > 0:
    fv2_all = compute_metrics(all_data['y_true'].values[mask_all_fv2], all_data['fusion_v2'].values[mask_all_fv2], "Fusion v2 (no sym)")
    all_results.append(fv2_all)

mask_all_fsym = all_data['fusion_v2_sym'].notna()
if mask_all_fsym.sum() > 0:
    fsym_all = compute_metrics(all_data['y_true'].values[mask_all_fsym], all_data['fusion_v2_sym'].values[mask_all_fsym], "Fusion v2+SymAug")
    all_results.append(fsym_all)

mask_all_farch = all_data['fusion_v2_archsym'].notna()
if mask_all_farch.sum() > 0:
    farch_all = compute_metrics(all_data['y_true'].values[mask_all_farch], all_data['fusion_v2_archsym'].values[mask_all_farch], "Fusion v2+ArchSym")
    all_results.append(farch_all)

print(f"{'Model':<25} {'LogLoss':>8} {'AUC':>8} {'Accuracy':>8} {'ECE':>8} {'Brier':>8}")
print("-" * 70)
for r in all_results:
    print(f"{r['name']:<25} {r['logloss']:>8.4f} {r['auc']:>8.4f} {r['accuracy']:>8.4f} {r['ece']:>8.4f} {r['brier']:>8.4f}")

# Save comprehensive results
output = {
    'common_subset_2025': {
        'n_matches': len(common_test),
        'models': {r['name']: {k: v for k, v in r.items() if k != 'name'} for r in results_common}
    },
    'all_data_with_odds': {
        'n_matches': len(all_data),
        'models': {r['name']: {k: v for k, v in r.items() if k != 'name'} for r in all_results}
    }
}

with open('data/comprehensive_odds_comparison.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to data/comprehensive_odds_comparison.json")
