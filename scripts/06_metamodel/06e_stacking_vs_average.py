import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Running 06e_stacking_vs_average.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # Load stacking results
    df = pd.read_csv("data/golgg_stacking_results.csv")
    
    # Features (Stage 1 Probabilities)
    features = [
        "player_elo", "player_gl", "player_ts", "player_os",
        "player_pl", "player_tm",
        "team_elo", "team_gl", "team_ts", "team_os",
        "team_pl", "team_tm"
    ]
    
    # 1. Simple Average
    df['simple_avg_prob'] = df[features].mean(axis=1)
    
    # 2. Stacking (LGBM) - already in df as metamodel_lgbm_prob
    
    # Evaluation (since 2020 for consistency)
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'] >= '2020-01-01'].copy()
    
    results = []
    for name, col in [("Simple Average", "simple_avg_prob"), ("Stacking (LGBM)", "metamodel_lgbm_prob")]:
        auc = roc_auc_score(test_df['y_true'], test_df[col])
        ll = log_loss(test_df['y_true'], test_df[col])
        results.append({"Method": name, "AUC": auc, "LogLoss": ll})
        
    res_df = pd.DataFrame(results)
    print("\n=== Stacking vs. Simple Average ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("docs/assets/metamodel_stacking_vs_avg.csv", index=False)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Method", y="AUC", data=res_df, palette="viridis", hue="Method", legend=False)
    plt.title("Model Architecture: Stacking vs. Simple Average (AUC)")
    plt.ylim(0.7, 0.76)
    for i, v in enumerate(res_df['AUC']):
        plt.text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("docs/assets/metamodel_stacking_vs_avg_auc.png")
    
    print("06e_stacking_vs_average.py completed.")

if __name__ == "__main__":
    main()
