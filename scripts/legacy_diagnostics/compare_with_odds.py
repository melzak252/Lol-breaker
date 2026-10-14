"""
Compare model predictions with bookmaker odds.
Handles team ordering mismatch between odds.csv and baseline data.
"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

# Load data
print("Loading data...")
odds = pd.read_csv('data/odds.csv')
baseline = pd.read_csv('data/golgg_y_predicts.csv')

print(f"Odds: {len(odds)} rows, Baseline: {len(baseline)} rows")

# Merge on golgg_match_id
merged = odds.merge(baseline, on='golgg_match_id', how='inner', suffixes=('_odds', '_base'))
print(f"Merged: {len(merged)} rows")

# Check team alignment
# odds has golgg_team1, golgg_team2
# baseline has team1_name, team2_name
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

# Compute bookmaker implied probabilities
# For matches where teams match: p_home = 1/odds_home, p_away = 1/odds_away
# For swapped matches: we need to flip the odds
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

# Remove rows where teams don't match at all
merged = merged[merged['bm_p_team1'].notna()].copy()
print(f"After removing non-matching teams: {len(merged)} rows")

# Bookmaker margin
merged['bm_margin'] = merged['bm_p_team1'] + merged['bm_p_team2'] - 1
print(f"Average bookmaker margin: {merged['bm_margin'].mean():.4f}")

# Normalize probabilities (remove margin)
merged['bm_p_team1_norm'] = merged['bm_p_team1'] / (merged['bm_p_team1'] + merged['bm_p_team2'])

# y_true is from baseline (1 = team1 wins)
# For swapped matches, baseline team1 = odds team2, so y_true already refers to baseline's team1
# No need to flip y_true since we already aligned the odds

# Filter to 2025+ test set
merged['date_base'] = pd.to_datetime(merged['date'])
test = merged[merged['date_base'] >= '2025-01-01'].copy()
print(f"\n2025+ test set: {len(test)} rows")

# Also compute for all data
all_data = merged.copy()

# ============================================
# Bookmaker metrics
# ============================================
def compute_metrics(y_true, y_pred, name="Model"):
    ll = log_loss(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = float('nan')
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    
    # ECE
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred > bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_pred) * abs(y_pred[mask].mean() - y_true[mask].mean())
    
    print(f"\n{name}:")
    print(f"  LogLoss:  {ll:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ECE:      {ece:.4f}")
    return {'logloss': ll, 'auc': auc, 'accuracy': acc, 'ece': ece}

# Bookmaker on all data
print("\n" + "="*60)
print("ALL DATA (with valid odds)")
print("="*60)
bm_all = compute_metrics(all_data['y_true'].values, all_data['bm_p_team1_norm'].values, "Bookmaker (avg odds)")

# Baseline on all data (with valid odds)
base_all = compute_metrics(all_data['y_true'].values, all_data['player_elo'].values, "Baseline (Player Elo)")

# Bookmaker on 2025+
print("\n" + "="*60)
print("2025+ TEST SET")
print("="*60)
bm_test = compute_metrics(test['y_true'].values, test['bm_p_team1_norm'].values, "Bookmaker (avg odds)")
base_test = compute_metrics(test['y_true'].values, test['player_elo'].values, "Baseline (Player Elo)")

# ============================================
# Betting simulation
# ============================================
def betting_simulation(y_true, y_pred, bm_probs, name="Model", min_edge=0.0):
    """
    Simulate betting: bet when model disagrees with bookmaker by enough edge.
    Edge = model_prob - bm_prob (for team1) or (1-model_prob) - (1-bm_prob) (for team2)
    """
    results = []
    
    for min_edge_val in [0.0, 0.02, 0.05, 0.10]:
        profit = 0
        bets = 0
        wins = 0
        
        for i in range(len(y_true)):
            p_model = y_pred[i]
            p_bm = bm_probs[i]
            
            # Bet on team1 if model thinks team1 is undervalued
            edge1 = p_model - p_bm
            # Bet on team2 if model thinks team2 is undervalued
            edge2 = (1 - p_model) - (1 - p_bm)
            
            if edge1 > min_edge_val:
                # Bet on team1
                bets += 1
                odds = 1.0 / p_bm if p_bm > 0 else 0  # approximate decimal odds from bm prob
                # Actually use real odds
                # For now, use implied odds (without margin removal for payout)
                if y_true[i] == 1:
                    profit += (1/p_bm - 1)  # win: profit = (odds - 1) * stake
                    wins += 1
                else:
                    profit -= 1  # lose: lose stake
            elif edge2 > min_edge_val:
                # Bet on team2
                bets += 1
                p_bm_2 = 1 - p_bm
                if y_true[i] == 0:
                    profit += (1/p_bm_2 - 1)
                    wins += 1
                else:
                    profit -= 1
        
        roi = profit / bets * 100 if bets > 0 else 0
        win_rate = wins / bets * 100 if bets > 0 else 0
        results.append({
            'min_edge': min_edge_val,
            'bets': bets,
            'wins': wins,
            'profit': profit,
            'roi': roi,
            'win_rate': win_rate
        })
    
    print(f"\n{name} - Betting Simulation:")
    print(f"{'Edge>':>8} {'Bets':>6} {'Wins':>6} {'Profit':>8} {'ROI':>8} {'WinRate':>8}")
    for r in results:
        print(f"{r['min_edge']:>7.2f} {r['bets']:>6d} {r['wins']:>6d} {r['profit']:>8.1f} {r['roi']:>7.1f}% {r['win_rate']:>7.1f}%")
    return results

# Betting simulation on 2025+ test set
print("\n" + "="*60)
print("BETTING SIMULATION (2025+)")
print("="*60)

# For betting simulation we need the raw bookmaker odds (not normalized)
# Re-derive: for matched teams, odds_team1 = avg_odds_home; for swapped, odds_team1 = avg_odds_away
test['odds_team1'] = np.where(
    test['teams_match'],
    test['avg_odds_home'],
    np.where(test['teams_swapped'], test['avg_odds_away'], np.nan)
)
test['odds_team2'] = np.where(
    test['teams_match'],
    test['avg_odds_away'],
    np.where(test['teams_swapped'], test['avg_odds_home'], np.nan)
)

# Better betting simulation using actual odds
def betting_sim_v2(y_true, model_probs, odds_team1, odds_team2, name="Model"):
    """
    Bet when model thinks a team has higher probability than bookmaker implies.
    Uses actual decimal odds for payout calculation.
    """
    print(f"\n{name} - Betting Simulation (using actual odds):")
    print(f"{'Edge>':>8} {'Bets':>6} {'Wins':>6} {'Profit':>8} {'ROI':>8} {'WinRate':>8}")
    
    for min_edge in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        profit = 0.0
        bets = 0
        wins = 0
        
        for i in range(len(y_true)):
            p_model = model_probs[i]
            o1 = odds_team1[i]
            o2 = odds_team2[i]
            
            if np.isnan(o1) or np.isnan(o2) or o1 <= 1 or o2 <= 1:
                continue
            
            # Implied probabilities (with margin)
            imp1 = 1.0 / o1
            imp2 = 1.0 / o2
            
            # Edge: model probability - implied probability
            edge1 = p_model - imp1
            edge2 = (1 - p_model) - imp2
            
            if edge1 > min_edge:
                # Bet on team1
                bets += 1
                if y_true[i] == 1:
                    profit += (o1 - 1)  # win
                    wins += 1
                else:
                    profit -= 1  # lose
            elif edge2 > min_edge:
                # Bet on team2
                bets += 1
                if y_true[i] == 0:
                    profit += (o2 - 1)  # win
                    wins += 1
                else:
                    profit -= 1  # lose
        
        roi = profit / bets * 100 if bets > 0 else 0
        win_rate = wins / bets * 100 if bets > 0 else 0
        print(f"{min_edge:>7.2f} {bets:>6d} {wins:>6d} {profit:>8.1f} {roi:>7.1f}% {win_rate:>7.1f}%")

betting_sim_v2(
    test['y_true'].values,
    test['player_elo'].values,
    test['odds_team1'].values,
    test['odds_team2'].values,
    "Baseline (Player Elo)"
)

# ============================================
# Summary comparison table
# ============================================
print("\n" + "="*60)
print("SUMMARY COMPARISON (2025+)")
print("="*60)
print(f"{'Model':<25} {'LogLoss':>8} {'AUC':>8} {'Accuracy':>8} {'ECE':>8}")
print("-" * 60)
print(f"{'Bookmaker (avg odds)':<25} {bm_test['logloss']:>8.4f} {bm_test['auc']:>8.4f} {bm_test['accuracy']:>8.4f} {bm_test['ece']:>8.4f}")
print(f"{'Baseline (Player Elo)':<25} {base_test['logloss']:>8.4f} {base_test['auc']:>8.4f} {base_test['accuracy']:>8.4f} {base_test['ece']:>8.4f}")
print(f"{'Fusion v2 (no sym)':<25} {'0.5582':>8} {'0.7822':>8} {'0.7131':>8} {'0.0217':>8}")
print(f"{'Fusion v2+SymAug':<25} {'0.5575':>8} {'0.7817':>8} {'0.7086':>8} {'0.0225':>8}")

# Save results
results = {
    'bookmaker_2025': bm_test,
    'baseline_2025': base_test,
    'bookmaker_all': bm_all,
    'baseline_all': base_all,
    'n_test_2025': len(test),
    'n_all_with_odds': len(all_data),
    'team_match_count': int(merged['teams_match'].sum()) if 'teams_match' in merged.columns else 0,
    'team_swapped_count': int(merged['teams_swapped'].sum()) if 'teams_swapped' in merged.columns else 0,
}

with open('data/odds_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to data/odds_comparison_results.json")
