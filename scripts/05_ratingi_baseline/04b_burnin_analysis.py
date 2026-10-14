import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import os

def main():
    print("Running 04b_burnin_analysis.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # Load predictions
    df = pd.read_csv("data/golgg_y_predicts.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. System Maturity: AUC over Years (Proxy for long-term stability)
    years = df['date'].dt.year.unique()
    yearly_auc = []
    for year in years:
        subset = df[df['date'].dt.year == year]
        if len(subset) > 100:
            auc = roc_auc_score(subset['y_true'], subset['player_gl'])
            yearly_auc.append({"Year": year, "AUC": auc})
            
    yearly_df = pd.DataFrame(yearly_auc)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Year", y="AUC", data=yearly_df, marker='o')
    plt.title("System Maturity: AUC over Years (Player Glicko-2)")
    plt.grid(alpha=0.3)
    plt.savefig("docs/assets/ratings_burnin_years.png")
    
    # 2. Actual Burn-in: AUC vs. Cumulative Matches
    # We group matches into bins of 1000 to see how AUC improves as system processes more data
    bin_size = 1000
    df['match_index'] = np.arange(len(df))
    df['bin'] = (df['match_index'] // bin_size) * bin_size
    
    bin_auc = []
    for b in sorted(df['bin'].unique()):
        subset = df[df['bin'] == b]
        if len(subset) >= 500: # Ensure enough matches for a stable AUC
            auc = roc_auc_score(subset['y_true'], subset['player_gl'])
            bin_auc.append({
                "Matches": b + bin_size, # End of the bin
                "AUC": auc
            })
            
    bin_df = pd.DataFrame(bin_auc)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plotting AUC over Cumulative Matches
    plt.plot(bin_df['Matches'], bin_df['AUC'], color='royalblue', marker='o', linewidth=2, markersize=8, label='AUC (Glicko-2)')
    
    # Add trend line (Lowess)
    sns.regplot(x="Matches", y="AUC", data=bin_df, scatter=False, color='red', lowess=True, 
                line_kws={'linestyle': '--', 'alpha': 0.6, 'label': 'Trend (Lowess)'})

    plt.title("Analiza Burn-in: Skuteczność systemu vs. Liczba przetworzonych meczów", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Skumulowana liczba meczów w bazie danych", fontsize=12)
    plt.ylabel("AUC (Player Glicko-2)", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.ylim(0.60, 0.80)
    
    # Mark key periods
    burnin_limit = 2000
    plt.axvspan(0, burnin_limit, color='red', alpha=0.1, label='Okres Burn-in (Rozgrzewka)')
    plt.axvspan(burnin_limit, len(df), color='green', alpha=0.05, label='Okres Stabilizacji')
    
    # Add text annotation for the burn-in limit
    plt.axvline(burnin_limit, color='black', linestyle=':', alpha=0.5)
    plt.text(burnin_limit + 200, 0.61, f'Próg stabilizacji: {burnin_limit} meczów', fontsize=10, fontweight='bold')

    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.savefig("docs/assets/ratings_burnin_matches.png", dpi=300, bbox_inches='tight')
    
    print(f"Saved improved docs/assets/ratings_burnin_matches.png with {bin_size}-match bins")
    print("04b_burnin_analysis.py completed.")

if __name__ == "__main__":
    main()
