import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

def main():
    print("Running 06b_metamodel_deep_dive.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # Load data
    df = pd.read_csv("data/golgg_stacking_results.csv")
    
    # 1. Correlation Analysis (Stage 1 Features)
    features_stage1 = [
        "player_elo", "player_gl", "player_ts", "player_os",
        "player_pl", "player_tm",
        "team_elo", "team_gl", "team_ts", "team_os",
        "team_pl", "team_tm"
    ]
    
    corr_matrix = df[features_stage1].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Rating Systems (Probabilities)")
    plt.tight_layout()
    plt.savefig("docs/assets/metamodel_feature_correlation.png")
    
    # 2. VIF Analysis
    vif_data = pd.DataFrame()
    vif_data["feature"] = features_stage1
    vif_data["VIF"] = [variance_inflation_factor(df[features_stage1].values, i) for i in range(len(features_stage1))]
    print("\n=== VIF Analysis ===")
    print(vif_data.sort_values("VIF", ascending=False))
    vif_data.to_csv("docs/assets/metamodel_vif_results.csv", index=False)
    
    # 3. Residual Analysis (Biggest Blunders)
    # Residual = |y_true - y_prob|
    df['residual'] = np.abs(df['y_true'] - df['metamodel_lgbm_prob'])
    
    # Find cases where model was very confident (>80%) but lost
    blunders = df[((df['metamodel_lgbm_prob'] > 0.8) & (df['y_true'] == 0)) | 
                  ((df['metamodel_lgbm_prob'] < 0.2) & (df['y_true'] == 1))].copy()
    
    print(f"\nTotal Blunders (Confidence > 80%, Wrong Result): {len(blunders)}")
    
    # Analyze blunders by BoN
    blunder_bon = blunders['BoN'].value_counts(normalize=True)
    print("\n=== Blunders by Match Format (BoN) ===")
    print(blunder_bon)
    
    # Plot Blunders by BoN
    plt.figure(figsize=(8, 6))
    sns.countplot(x='BoN', data=blunders, palette='viridis', hue='BoN', legend=False)
    plt.title("Number of Major Blunders by Match Format (BoN)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("docs/assets/metamodel_blunders_bon.png")
    
    # Top 10 biggest blunders
    top_blunders = blunders.sort_values('residual', ascending=False).head(10)
    # Check if tournament exists, if not use what's available
    cols_to_save = [c for c in ['date', 'tournament', 'metamodel_lgbm_prob', 'y_true', 'BoN'] if c in top_blunders.columns]
    top_blunders[cols_to_save].to_csv("docs/assets/metamodel_top_blunders.csv", index=False)
    
    print("\n06b_metamodel_deep_dive.py completed successfully.")

if __name__ == "__main__":
    main()
