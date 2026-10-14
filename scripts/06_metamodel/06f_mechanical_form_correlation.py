import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Running 06f_mechanical_form_correlation.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # Load rolling stats and stacking results
    df_rolling = pd.read_csv("data/golgg_rolling_stats.csv")
    df_stacking = pd.read_csv("data/golgg_stacking_results.csv")
    
    # Merge to get y_true and s1_prob
    df = pd.merge(df_stacking[['golgg_match_id', 'y_true', 's1_prob']], df_rolling, on='golgg_match_id')
    
    # Calculate relative stats (T1 - T2)
    df['rel_gd15'] = df['t1_rolling_gd15'] - df['t2_rolling_gd15']
    df['rel_dpm'] = df['t1_rolling_dpm'] - df['t2_rolling_dpm']
    df['rel_kills'] = df['t1_rolling_kills'] - df['t2_rolling_kills']
    
    # Correlation Analysis
    cols = ['y_true', 's1_prob', 'rel_gd15', 'rel_dpm', 'rel_kills']
    corr = df[cols].corr()
    print("\n=== Correlation Matrix: Mechanical Form vs Ratings ===")
    print(corr)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".3f")
    plt.title("Correlation: Mechanical Form vs. Win & Ratings")
    plt.tight_layout()
    plt.savefig("docs/assets/context_mechanical_correlation.png")
    
    # Scatter plot: rel_gd15 vs s1_prob
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='rel_gd15', y='s1_prob', hue='y_true', data=df.sample(2000), alpha=0.5)
    plt.title("Independence Check: Ratings (S1) vs. Early Game Form (GD15)")
    plt.xlabel("Relative Rolling GD15 (T1 - T2)")
    plt.ylabel("S1 Probability (Ratings)")
    plt.grid(alpha=0.3)
    plt.savefig("docs/assets/context_gd15_vs_ratings.png")
    
    print("06f_mechanical_form_correlation.py completed.")

if __name__ == "__main__":
    main()
