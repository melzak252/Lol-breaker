#!/usr/bin/env python3
"""
Evaluate Fusion models on historical 2025+ data vs market odds.
Compares:
- Fusion-v2, Fusion-v2-SymAug, Fusion-v2-ArchSym
- Market implied probabilities (from avg_odds)
- Hybrid-Thesis-Market (baseline model)
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Database connection
DB_DSN = "postgresql://betting:betting_local_password@192.168.1.17:5432/betting"

def load_odds_data():
    """Load odds.csv with 2025+ filter"""
    odds = pd.read_csv('/home/melzak/dev/inzynierka/data/odds.csv')
    odds['golgg_date'] = pd.to_datetime(odds['golgg_date'])
    odds_2025 = odds[odds['golgg_date'] >= '2025-01-01'].copy()
    
    # Convert golgg_match_id to string for DB join
    odds_2025['golgg_match_id'] = odds_2025['golgg_match_id'].astype(str)
    
    # Calculate market implied probabilities (remove margin)
    odds_2025['market_prob_home'] = 1 / odds_2025['avg_odds_home']
    odds_2025['market_prob_away'] = 1 / odds_2025['avg_odds_away']
    
    # Normalize to sum to 1 (remove bookmaker margin)
    total = odds_2025['market_prob_home'] + odds_2025['market_prob_away']
    odds_2025['market_prob_home'] = odds_2025['market_prob_home'] / total
    odds_2025['market_prob_away'] = odds_2025['market_prob_away'] / total
    
    # True outcome
    odds_2025['actual_home_win'] = odds_2025['t1_win'].astype(int)
    
    print(f"Loaded {len(odds_2025)} matches from 2025+")
    print(f"Date range: {odds_2025['golgg_date'].min()} to {odds_2025['golgg_date'].max()}")
    
    return odds_2025

def load_fusion_predictions(conn):
    """Load all fusion predictions from database"""
    query = """
    SELECT 
        cm.canonical_key,
        cm.team_a_name,
        cm.team_b_name,
        cm.start_time_normalized,
        cm.winner_name,
        cp.model_name,
        cp.prob_a,
        cp.prob_b
    FROM canonical_predictions cp
    JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
    WHERE cp.model_name IN ('Fusion-v2', 'Fusion-v2-SymAug', 'Fusion-v2-ArchSym')
      AND cm.start_time_normalized >= '2025-01-01'
    """
    
    df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(df)} fusion predictions from DB")
    
    return df

def load_hybrid_predictions(conn):
    """Load Hybrid-Thesis-Market predictions for comparison"""
    query = """
    SELECT 
        cm.canonical_key,
        cm.team_a_name,
        cm.team_b_name,
        cm.start_time_normalized,
        cm.winner_name,
        cp.model_name,
        cp.prob_a,
        cp.prob_b
    FROM canonical_predictions cp
    JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
    WHERE cp.model_name = 'Hybrid-Thesis-Market'
      AND cm.start_time_normalized >= '2025-01-01'
    """
    
    df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(df)} Hybrid predictions from DB")
    
    return df

def match_odds_with_predictions(odds_df, pred_df, model_name):
    """Match odds.csv with predictions using team names and dates"""
    
    # Filter predictions for this model
    model_preds = pred_df[pred_df['model_name'] == model_name].copy()
    
    # Parse dates
    model_preds['date'] = pd.to_datetime(model_preds['start_time_normalized'], format='ISO8601', utc=True).dt.date
    odds_df['date'] = odds_df['golgg_date'].dt.date
    
    # Merge on team names and date (fuzzy matching)
    # Try exact match first
    merged = odds_df.merge(
        model_preds,
        left_on=['golgg_team1', 'golgg_team2', 'date'],
        right_on=['team_a_name', 'team_b_name', 'date'],
        how='inner'
    )
    
    print(f"  {model_name}: matched {len(merged)} predictions (exact)")
    
    # If low match, try case-insensitive
    if len(merged) < 100:
        odds_df['team1_lower'] = odds_df['golgg_team1'].str.lower()
        odds_df['team2_lower'] = odds_df['golgg_team2'].str.lower()
        model_preds['team_a_lower'] = model_preds['team_a_name'].str.lower()
        model_preds['team_b_lower'] = model_preds['team_b_name'].str.lower()
        
        merged = odds_df.merge(
            model_preds,
            left_on=['team1_lower', 'team2_lower', 'date'],
            right_on=['team_a_lower', 'team_b_lower', 'date'],
            how='inner'
        )
        print(f"  {model_name}: matched {len(merged)} predictions (case-insensitive)")
    
    return merged

def calculate_metrics(df, prob_col, label='Model'):
    """Calculate evaluation metrics"""
    
    # Filter valid rows
    valid = df[[prob_col, 'actual_home_win']].dropna()
    
    if len(valid) < 10:
        return None
    
    y_true = valid['actual_home_win'].values
    y_prob = valid[prob_col].values
    
    # Clip probabilities to [0.01, 0.99] to avoid log(0)
    y_prob = np.clip(y_prob, 0.01, 0.99)
    
    metrics = {
        'label': label,
        'n_matches': len(valid),
        'auc': roc_auc_score(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'accuracy': accuracy_score(y_true, (y_prob > 0.5).astype(int)),
        'avg_prob': y_prob.mean(),
        'avg_actual': y_true.mean(),
    }
    
    return metrics

def calculate_roi(df, prob_col, odds_col, label='Model', threshold=0.05):
    """Calculate ROI for betting strategy"""
    
    valid = df[[prob_col, odds_col, 'actual_home_win']].dropna()
    
    if len(valid) < 10:
        return None
    
    # Calculate EV for each bet
    valid['ev'] = (valid[prob_col] * (valid[odds_col] - 1)) - (1 - valid[prob_col])
    
    # Bet only when EV > threshold
    bets = valid[valid['ev'] > threshold].copy()
    
    if len(bets) == 0:
        return {'label': label, 'n_bets': 0, 'roi': 0, 'win_rate': 0}
    
    # Calculate profit/loss
    bets['profit'] = np.where(
        bets['actual_home_win'] == 1,
        bets[odds_col] - 1,  # Win: profit = odds - 1
        -1  # Loss: -1 unit
    )
    
    total_staked = len(bets)
    total_profit = bets['profit'].sum()
    roi = total_profit / total_staked
    
    return {
        'label': label,
        'n_bets': len(bets),
        'roi': roi,
        'total_profit': total_profit,
        'win_rate': bets['actual_home_win'].mean(),
        'avg_ev': bets['ev'].mean(),
    }

def plot_calibration(models_data, output_path):
    """Plot calibration curves for all models"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label, df, prob_col in models_data:
        valid = df[[prob_col, 'actual_home_win']].dropna()
        if len(valid) < 10:
            continue
        
        y_true = valid['actual_home_win'].values
        y_prob = np.clip(valid[prob_col].values, 0.01, 0.99)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform'
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=label)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curves (2025+ Historical Data)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved calibration plot to {output_path}")

def main():
    print("=" * 80)
    print("FUSION MODEL EVALUATION ON 2025+ HISTORICAL DATA")
    print("=" * 80)
    
    # Load data
    odds_df = load_odds_data()
    
    conn = psycopg2.connect(DB_DSN)
    fusion_df = load_fusion_predictions(conn)
    hybrid_df = load_hybrid_predictions(conn)
    conn.close()
    
    # Match odds with predictions
    print("\nMatching predictions with odds...")
    
    models = [
        ('Market', odds_df, 'market_prob_home'),
        ('Fusion-v2', fusion_df, 'prob_a'),
        ('Fusion-v2-SymAug', fusion_df, 'prob_a'),
        ('Fusion-v2-ArchSym', fusion_df, 'prob_a'),
        ('Hybrid-Thesis-Market', hybrid_df, 'prob_a'),
    ]
    
    matched_data = {}
    for model_name, pred_df, prob_col in models:
        if model_name == 'Market':
            matched_data[model_name] = odds_df
        else:
            matched = match_odds_with_predictions(odds_df, pred_df, model_name)
            matched_data[model_name] = matched
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    all_metrics = []
    
    # Market baseline
    metrics = calculate_metrics(odds_df, 'market_prob_home', 'Market (odds)')
    if metrics:
        all_metrics.append(metrics)
        print(f"\nMarket (odds):")
        print(f"  Matches: {metrics['n_matches']}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Brier: {metrics['brier']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Fusion models
    for model_name in ['Fusion-v2', 'Fusion-v2-SymAug', 'Fusion-v2-ArchSym']:
        df = matched_data[model_name]
        metrics = calculate_metrics(df, 'prob_a', model_name)
        if metrics:
            all_metrics.append(metrics)
            print(f"\n{model_name}:")
            print(f"  Matches: {metrics['n_matches']}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Brier: {metrics['brier']:.4f}")
            print(f"  Log Loss: {metrics['log_loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Hybrid baseline
    df = matched_data['Hybrid-Thesis-Market']
    metrics = calculate_metrics(df, 'prob_a', 'Hybrid-Thesis-Market')
    if metrics:
        all_metrics.append(metrics)
        print(f"\nHybrid-Thesis-Market:")
        print(f"  Matches: {metrics['n_matches']}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Brier: {metrics['brier']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # ROI analysis
    print("\n" + "=" * 80)
    print("ROI ANALYSIS (EV > 5% threshold)")
    print("=" * 80)
    
    all_roi = []
    
    # Market (no edge by definition)
    print(f"\nMarket (odds):")
    print(f"  ROI: 0.00% (by definition, after margin)")
    
    # Fusion models
    for model_name in ['Fusion-v2', 'Fusion-v2-SymAug', 'Fusion-v2-ArchSym']:
        df = matched_data[model_name]
        roi = calculate_roi(df, 'prob_a', 'avg_odds_home', model_name, threshold=0.05)
        if roi:
            all_roi.append(roi)
            print(f"\n{model_name}:")
            print(f"  Bets: {roi['n_bets']}")
            print(f"  ROI: {roi['roi']*100:.2f}%")
            print(f"  Total Profit: {roi['total_profit']:.2f} units")
            print(f"  Win Rate: {roi['win_rate']*100:.1f}%")
            print(f"  Avg EV: {roi['avg_ev']*100:.2f}%")
    
    # Hybrid
    df = matched_data['Hybrid-Thesis-Market']
    roi = calculate_roi(df, 'prob_a', 'avg_odds_home', 'Hybrid-Thesis-Market', threshold=0.05)
    if roi:
        all_roi.append(roi)
        print(f"\nHybrid-Thesis-Market:")
        print(f"  Bets: {roi['n_bets']}")
        print(f"  ROI: {roi['roi']*100:.2f}%")
        print(f"  Total Profit: {roi['total_profit']:.2f} units")
        print(f"  Win Rate: {roi['win_rate']*100:.1f}%")
        print(f"  Avg EV: {roi['avg_ev']*100:.2f}%")
    
    # Calibration plot
    print("\n" + "=" * 80)
    print("CALIBRATION PLOT")
    print("=" * 80)
    
    models_for_plot = [
        ('Market', odds_df, 'market_prob_home'),
        ('Fusion-v2', matched_data['Fusion-v2'], 'prob_a'),
        ('Fusion-v2-SymAug', matched_data['Fusion-v2-SymAug'], 'prob_a'),
        ('Fusion-v2-ArchSym', matched_data['Fusion-v2-ArchSym'], 'prob_a'),
        ('Hybrid-Thesis-Market', matched_data['Hybrid-Thesis-Market'], 'prob_a'),
    ]
    
    plot_calibration(models_for_plot, '/home/melzak/dev/inzynierka/fusion_calibration_2025.png')
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df[['label', 'n_matches', 'auc', 'brier', 'log_loss', 'accuracy']]
    summary_df.columns = ['Model', 'Matches', 'AUC', 'Brier', 'LogLoss', 'Accuracy']
    
    print("\n" + summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv('/home/melzak/dev/inzynierka/fusion_evaluation_2025.csv', index=False)
    print("\nSaved summary to fusion_evaluation_2025.csv")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
