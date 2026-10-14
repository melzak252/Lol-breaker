#!/usr/bin/env python3
"""
Hybrid Fusion + Market with Temperature Scaling
hybrid_prob = alpha * model_prob + (1-alpha) * market_prob
calibrated = sigmoid(logit(hybrid_prob) / T)

Grid search over alpha and T for each Fusion model.
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, log_loss
from scipy.special import logit as sp_logit, expit as sp_expit
import warnings
warnings.filterwarnings('ignore')

DB_DSN = "postgresql://betting:betting_local_password@192.168.1.17:5432/betting"

def load_data():
    odds = pd.read_csv('/home/melzak/dev/inzynierka/data/odds.csv')
    odds['golgg_date'] = pd.to_datetime(odds['golgg_date'])
    odds_2025 = odds[odds['golgg_date'] >= '2025-01-01'].copy()
    odds_2025['golgg_match_id'] = odds_2025['golgg_match_id'].astype(str)
    
    # Market implied probs (remove margin)
    odds_2025['market_prob_home'] = 1 / odds_2025['avg_odds_home']
    odds_2025['market_prob_away'] = 1 / odds_2025['avg_odds_away']
    total = odds_2025['market_prob_home'] + odds_2025['market_prob_away']
    odds_2025['market_prob_home'] /= total
    odds_2025['market_prob_away'] /= total
    odds_2025['actual_home_win'] = odds_2025['t1_win'].astype(int)
    odds_2025['date'] = odds_2025['golgg_date'].dt.date
    
    conn = psycopg2.connect(DB_DSN)
    fusion_df = pd.read_sql_query("""
        SELECT cm.canonical_key, cm.team_a_name, cm.team_b_name,
               cm.start_time_normalized, cm.winner_name,
               cp.model_name, cp.prob_a, cp.prob_b
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name IN ('Fusion-v2', 'Fusion-v2-SymAug', 'Fusion-v2-ArchSym')
          AND cm.start_time_normalized >= '2025-01-01'
    """, conn)
    conn.close()
    
    fusion_df['date'] = pd.to_datetime(fusion_df['start_time_normalized'], format='ISO8601', utc=True).dt.date
    
    return odds_2025, fusion_df

def match_data(odds_df, fusion_df, model_name):
    model_preds = fusion_df[fusion_df['model_name'] == model_name].copy()
    
    # Exact match
    merged = odds_df.merge(
        model_preds,
        left_on=['golgg_team1', 'golgg_team2', 'date'],
        right_on=['team_a_name', 'team_b_name', 'date'],
        how='inner'
    )
    
    if len(merged) < 100:
        odds_df['t1_lower'] = odds_df['golgg_team1'].str.lower()
        odds_df['t2_lower'] = odds_df['golgg_team2'].str.lower()
        model_preds['ta_lower'] = model_preds['team_a_name'].str.lower()
        model_preds['tb_lower'] = model_preds['team_b_name'].str.lower()
        merged = odds_df.merge(
            model_preds,
            left_on=['t1_lower', 't2_lower', 'date'],
            right_on=['ta_lower', 'tb_lower', 'date'],
            how='inner'
        )
    
    return merged

def temperature_scale(probs, T):
    """Apply temperature scaling: sigmoid(logit(p) / T)"""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return sp_expit(sp_logit(p) / T)

def calc_metrics(y_true, y_prob):
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask]
    y_prob = np.clip(y_prob[mask], 0.01, 0.99)
    if len(y_true) < 10:
        return {'auc': 0, 'brier': 1, 'log_loss': 10, 'accuracy': 0, 'n': len(y_true)}
    return {
        'auc': roc_auc_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'accuracy': accuracy_score(y_true, (y_prob > 0.5).astype(int)),
        'n': len(y_true),
    }

def calc_roi(y_true, y_prob, odds, threshold=0.05):
    mask = ~(np.isnan(y_true) | np.isnan(y_prob) | np.isnan(odds))
    y_true = y_true[mask]
    y_prob = np.clip(y_prob[mask], 0.01, 0.99)
    odds = odds[mask]
    ev = y_prob * (odds - 1) - (1 - y_prob)
    bet_mask = ev > threshold
    if bet_mask.sum() == 0:
        return {'n_bets': 0, 'roi': 0, 'profit': 0, 'win_rate': 0}
    
    bets_y = y_true[bet_mask]
    bets_odds = odds[bet_mask]
    profit = np.where(bets_y == 1, bets_odds - 1, -1)
    
    return {
        'n_bets': int(bet_mask.sum()),
        'roi': profit.sum() / len(profit),
        'profit': profit.sum(),
        'win_rate': bets_y.mean(),
    }

def grid_search_hybrid(df, model_col='prob_a', market_col='market_prob_home', odds_col='avg_odds_home'):
    """Grid search over alpha and temperature T"""
    
    y_true = df['actual_home_win'].values
    model_p = df[model_col].values
    market_p = df[market_col].values
    odds = df[odds_col].values
    
    alphas = np.arange(0.0, 1.05, 0.05)
    temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    
    results = []
    
    for alpha in alphas:
        for T in temps:
            # Hybrid blend
            hybrid_p = alpha * model_p + (1 - alpha) * market_p
            # Temperature scaling
            calibrated_p = temperature_scale(hybrid_p, T)
            
            m = calc_metrics(y_true, calibrated_p)
            r = calc_roi(y_true, calibrated_p, odds, threshold=0.05)
            
            results.append({
                'alpha': round(alpha, 2),
                'T': T,
                'auc': m['auc'],
                'brier': m['brier'],
                'log_loss': m['log_loss'],
                'accuracy': m['accuracy'],
                'n_bets': r['n_bets'],
                'roi': r['roi'],
                'profit': r['profit'],
                'win_rate': r['win_rate'],
            })
    
    return pd.DataFrame(results)

def print_top_results(results_df, model_name, n=10):
    """Print top results by different criteria"""
    
    print(f"\n{'='*90}")
    print(f"  {model_name} — TOP HYBRID CONFIGURATIONS")
    print(f"{'='*90}")
    
    # By AUC
    top_auc = results_df.nlargest(5, 'auc')
    print(f"\n  TOP 5 by AUC:")
    print(f"  {'α':>5} {'T':>5} {'AUC':>7} {'Brier':>7} {'LogLoss':>8} {'Acc':>6} {'Bets':>5} {'ROI%':>7} {'Profit':>8}")
    for _, r in top_auc.iterrows():
        print(f"  {r['alpha']:5.2f} {r['T']:5.1f} {r['auc']:7.4f} {r['brier']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.3f} {int(r['n_bets']):5d} {r['roi']*100:7.2f} {r['profit']:8.1f}")
    
    # By ROI (min 50 bets)
    valid_roi = results_df[results_df['n_bets'] >= 50]
    if len(valid_roi) > 0:
        top_roi = valid_roi.nlargest(5, 'roi')
        print(f"\n  TOP 5 by ROI (min 50 bets):")
        print(f"  {'α':>5} {'T':>5} {'AUC':>7} {'Brier':>7} {'LogLoss':>8} {'Acc':>6} {'Bets':>5} {'ROI%':>7} {'Profit':>8}")
        for _, r in top_roi.iterrows():
            print(f"  {r['alpha']:5.2f} {r['T']:5.1f} {r['auc']:7.4f} {r['brier']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.3f} {int(r['n_bets']):5d} {r['roi']*100:7.2f} {r['profit']:8.1f}")
    
    # By profit (min 50 bets)
    if len(valid_roi) > 0:
        top_profit = valid_roi.nlargest(5, 'profit')
        print(f"\n  TOP 5 by Total Profit (min 50 bets):")
        print(f"  {'α':>5} {'T':>5} {'AUC':>7} {'Brier':>7} {'LogLoss':>8} {'Acc':>6} {'Bets':>5} {'ROI%':>7} {'Profit':>8}")
        for _, r in top_profit.iterrows():
            print(f"  {r['alpha']:5.2f} {r['T']:5.1f} {r['auc']:7.4f} {r['brier']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.3f} {int(r['n_bets']):5d} {r['roi']*100:7.2f} {r['profit']:8.1f}")
    
    # Best balanced (AUC * ROI rank)
    if len(valid_roi) > 0:
        valid_roi = valid_roi.copy()
        valid_roi['score'] = valid_roi['auc'] * 0.4 + valid_roi['roi'] * 0.3 + (1 - valid_roi['brier']) * 0.3
        top_balanced = valid_roi.nlargest(5, 'score')
        print(f"\n  TOP 5 Balanced (0.4*AUC + 0.3*ROI + 0.3*(1-Brier)):")
        print(f"  {'α':>5} {'T':>5} {'AUC':>7} {'Brier':>7} {'LogLoss':>8} {'Acc':>6} {'Bets':>5} {'ROI%':>7} {'Profit':>8}")
        for _, r in top_balanced.iterrows():
            print(f"  {r['alpha']:5.2f} {r['T']:5.1f} {r['auc']:7.4f} {r['brier']:7.4f} {r['log_loss']:8.4f} {r['accuracy']:6.3f} {int(r['n_bets']):5d} {r['roi']*100:7.2f} {r['profit']:8.1f}")

def print_baselines(df):
    """Print baseline metrics for comparison"""
    y_true = df['actual_home_win'].values
    market_p = df['market_prob_home'].values
    odds = df['avg_odds_home'].values
    
    print(f"\n{'='*90}")
    print(f"  BASELINES ({len(df)} matches)")
    print(f"{'='*90}")
    
    m = calc_metrics(y_true, market_p)
    r = calc_roi(y_true, market_p, odds, threshold=0.05)
    print(f"\n  Market (odds):")
    print(f"    AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  Acc={m['accuracy']:.3f}")
    print(f"    ROI bets(EV>5%): {r['n_bets']}  ROI={r['roi']*100:.2f}%  Profit={r['profit']:.1f}")

def main():
    print("=" * 90)
    print("HYBRID FUSION + MARKET WITH TEMPERATURE SCALING")
    print("hybrid = α * model + (1-α) * market")
    print("calibrated = sigmoid(logit(hybrid) / T)")
    print("=" * 90)
    
    odds_df, fusion_df = load_data()
    print(f"Odds 2025+: {len(odds_df)} matches")
    print(f"Fusion predictions: {len(fusion_df)}")
    
    for model_name in ['Fusion-v2', 'Fusion-v2-SymAug', 'Fusion-v2-ArchSym']:
        merged = match_data(odds_df.copy(), fusion_df, model_name)
        print(f"\n{model_name}: {len(merged)} matched matches")
        
        if len(merged) < 50:
            print(f"  Too few matches, skipping")
            continue
        
        # Baselines
        print_baselines(merged)
        
        # Pure model baselines
        y_true = merged['actual_home_win'].values
        model_p = merged['prob_a'].values
        odds = merged['avg_odds_home'].values
        
        m = calc_metrics(y_true, model_p)
        r = calc_roi(y_true, model_p, odds, threshold=0.05)
        print(f"\n  Pure {model_name} (α=1.0, T=1.0):")
        print(f"    AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  Acc={m['accuracy']:.3f}")
        print(f"    ROI bets(EV>5%): {r['n_bets']}  ROI={r['roi']*100:.2f}%  Profit={r['profit']:.1f}")
        
        # Grid search
        results = grid_search_hybrid(merged)
        print_top_results(results, model_name)
        
        # Save full grid
        results.to_csv(f'/home/melzak/dev/inzynierka/hybrid_grid_{model_name.replace("-", "_")}.csv', index=False)
    
    print(f"\n{'='*90}")
    print("DONE")
    print(f"{'='*90}")

if __name__ == '__main__':
    main()
