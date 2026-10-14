import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_curve, auc
import os
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece

def main():
    print("Running 04_evaluate_ratings.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    print("Loading golgg_y_predicts.csv...")
    df = pd.read_csv("data/golgg_y_predicts.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter since 2020
    df = df[df['date'] >= datetime(2020, 1, 1)].copy()
    df = df[pd.notna(df['y_true'])].copy()
    
    rating_systems = {
        'Player Glicko2': 'player_gl',
        'Player Elo': 'player_elo',
        'Player TrueSkill': 'player_ts',
        'Player OpenSkill': 'player_os',
        'Player Plackett-Luce': 'player_pl',
        'Player Thurstone-Mosteller': 'player_tm',
        'Team Glicko2': 'team_gl',
        'Team Elo': 'team_elo',
        'Team TrueSkill': 'team_ts',
        'Team OpenSkill': 'team_os',
        'Team Plackett-Luce': 'team_pl',
        'Team Thurstone-Mosteller': 'team_tm'
    }
    
    # 1. Rating Distributions
    plt.figure(figsize=(12, 6))
    for name, col in rating_systems.items():
        if 'Player' in name and col in df.columns:
            sns.kdeplot(df[col].dropna(), label=name, fill=True, alpha=0.1)
    plt.title("Distribution of Player Ratings (Probabilities)")
    plt.xlabel("Predicted Probability of T1 Win")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/ratings_distributions.png")
    
    metrics = []
    
    # For ROC plot
    plt.figure(figsize=(6, 6))
    roc_fig = plt.gcf()
    cal_fig = plt.figure(figsize=(6, 6))
    
    for name, col in rating_systems.items():
        if col not in df.columns:
            continue
            
        y_prob = df[col].to_numpy()
        y_true = df['y_true'].to_numpy()
        
        mask = ~np.isnan(y_prob)
        y_prob = y_prob[mask]
        y_true = y_true[mask]
        
        if len(y_true) == 0:
            continue
            
        # Calibration Curve (Only for Player models)
        if 'Player' in name:
            plt.figure(cal_fig.number)
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=name)
        
        # ROC Curve (Only for Player models)
        if 'Player' in name:
            plt.figure(roc_fig.number)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        else:
            roc_auc = auc(*roc_curve(y_true, y_prob)[:2])
        
        # Metrics
        brier = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        ece = calculate_ece(y_true, y_prob)
        
        metrics.append({
            'System': name,
            'Brier Score': brier,
            'Log Loss': ll,
            'ECE': ece,
            'AUC': roc_auc,
            'Sample Size': len(y_true)
        })
        
    # Finalize Calibration Plot
    plt.figure(cal_fig.number)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves: Player-Level Rating Systems')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/ratings_calibration_curves.png")
    
    # Finalize ROC Plot
    plt.figure(roc_fig.number)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Player-Level Rating Systems')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/ratings_roc_curves.png")
    
    # Save metrics table
    metrics_df = pd.DataFrame(metrics).sort_values('Brier Score')
    metrics_df.to_csv("docs/assets/ratings_calibration_metrics.csv", index=False)
    
    print("\n04_evaluate_ratings.py completed successfully.")

if __name__ == "__main__":
    main()
