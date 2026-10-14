import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("Running 02b_eda_gd15.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    with open("data/golgg_matches.json", "r", encoding="utf-8") as f:
        matches = json.load(f)
        
    game_data = []
    for m in matches:
        for g in m['games']:
            t1_gd15 = 0
            t2_gd15 = 0
            
            # Sum GD15 for team 1
            for role, p_data in g.get('t1_players', {}).items():
                gd15 = p_data.get('stats', {}).get('gd@15')
                if gd15 is not None:
                    t1_gd15 += gd15
            
            # Sum GD15 for team 2
            for role, p_data in g.get('t2_players', {}).items():
                gd15 = p_data.get('stats', {}).get('gd@15')
                if gd15 is not None:
                    t2_gd15 += gd15
            
            # Team 1 GD15 is t1_gd15 - t2_gd15 (relative)
            # But wait, usually gd@15 in GOL.GG is already relative to the opponent in that role?
            # Let's check. If it's relative, then sum(t1_gd15) should be the team GD15.
            # Actually, in GOL.GG, gd@15 for a player is their gold minus opponent's gold.
            # So sum of all players in team 1 should be the team's GD15.
            
            if g.get('t1_players'): # Check if we have player data
                game_data.append({
                    'team_gd15': t1_gd15,
                    'win': 1 if g['t1_win'] else 0
                })

    df = pd.DataFrame(game_data)
    df = df.dropna()
    
    # Bin GD15
    df['gd15_bin'] = pd.cut(df['team_gd15'], bins=np.arange(-10000, 10001, 1000))
    
    # Calculate Win Rate per bin
    bin_stats = df.groupby('gd15_bin', observed=True)['win'].agg(['mean', 'count'])
    bin_stats = bin_stats[bin_stats['count'] > 50] # Filter out small samples
    bin_stats['mid'] = bin_stats.index.map(lambda x: x.mid)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    from scipy.optimize import curve_fit

    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    # Initial guess for sigmoid parameters
    p0 = [1, 0, 0.001, 0]
    x_data = bin_stats['mid'].astype(float).values
    y_data = bin_stats['mean'].astype(float).values
    popt, _ = curve_fit(sigmoid, x_data, y_data, p0, method='dogbox')

    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_sigmoid = sigmoid(x_range, *popt)

    plt.scatter(x_data, y_data, s=bin_stats['count']/5, alpha=0.6, color='royalblue', label='Punkty danych (rozmiar=liczba gier)')
    plt.plot(x_range, y_sigmoid, color='red', linewidth=3, label='Dopasowanie Sigmoidalne')
    
    plt.title("Zależność Win Rate od różnicy złota w 15 minucie (GD15)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Różnica złota drużyny (GD15)", fontsize=12)
    plt.ylabel("Win Rate (Szansa na wygraną)", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.ylim(0, 1)
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_gd15_vs_winrate.png", dpi=300, bbox_inches='tight')
    
    print("02b_eda_gd15.py completed successfully.")

if __name__ == "__main__":
    main()
