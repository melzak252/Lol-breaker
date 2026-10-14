import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Running 02_eda_market.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # Load Odds Data
    print("Loading odds.csv...")
    df = pd.read_csv("data/odds.csv")
    df['date'] = pd.to_datetime(df['odds_date'])
    df['year'] = df['date'].dt.year
    
    # Calculate BoN from scores if not present
    if 'BoN' not in df.columns:
        df['BoN'] = df['t1_score'] + df['t2_score']
        # Map to standard BoN formats (1, 3, 5)
        df['BoN'] = df['BoN'].apply(lambda x: 1 if x <= 1 else (3 if x <= 3 else 5))
    
    bookies = ["betclic", "betfan", "efortuna", "lv_bet", "sts", "superbet"]
    
    # 1. Bookmaker Margins
    margins = []
    for b in bookies:
        o1 = df[f'odds1_{b}_open']
        o2 = df[f'odds2_{b}_open']
        valid = (o1 > 1) & (o2 > 1)
        margin = (1/o1[valid] + 1/o2[valid] - 1) * 100
        margins.append({'Bookmaker': b.upper(), 'Margin (%)': margin.mean()})
        
    margins_df = pd.DataFrame(margins).sort_values('Margin (%)')
    avg_margin = margins_df['Margin (%)'].mean()
    print(f"Average Market Margin: {avg_margin:.2f}%")
    
    # Improved Plotting
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Use a professional color palette
    colors = sns.color_palette("viridis", len(margins_df))
    
    barplot = sns.barplot(
        x='Bookmaker', 
        y='Margin (%)', 
        data=margins_df, 
        palette='viridis', 
        hue='Bookmaker', 
        legend=False
    )
    
    # Add data labels on top of bars
    for p in barplot.patches:
        barplot.annotate(
            format(p.get_height(), '.2f') + '%', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', va = 'center', 
            xytext = (0, 9), 
            textcoords = 'offset points',
            fontweight='bold',
            fontsize=11
        )

    plt.title("Średnia marża bukmacherska (Vig) u polskich operatorów", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Marża (%)", fontsize=12)
    plt.xlabel("Bukmacher", fontsize=12)
    
    plt.axhline(avg_margin, color='red', linestyle='--', linewidth=2, label=f"Średnia rynkowa: {avg_margin:.2f}%")
    
    # Improve aesthetics
    plt.ylim(0, max(margins_df['Margin (%)']) * 1.15)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig("docs/assets/eda_bookie_margins.png", dpi=300, bbox_inches='tight')
    print("Saved improved docs/assets/eda_bookie_margins.png")
    
    # 2. Arbitrage Opportunities
    raw_arbs = []
    tax_arbs = []
    
    for year in df['year'].unique():
        year_df = df[df['year'] == year]
        raw_count = 0
        tax_count = 0
        
        for _, row in year_df.iterrows():
            max_o1 = max([row.get(f'odds1_{b}_open', 0) for b in bookies if pd.notnull(row.get(f'odds1_{b}_open'))] + [0])
            max_o2 = max([row.get(f'odds2_{b}_open', 0) for b in bookies if pd.notnull(row.get(f'odds2_{b}_open'))] + [0])
            
            if max_o1 > 1 and max_o2 > 1:
                if (1/max_o1 + 1/max_o2) < 1.0:
                    raw_count += 1
                if (1/(max_o1 * 0.88) + 1/(max_o2 * 0.88)) < 1.0:
                    tax_count += 1
                    
        raw_arbs.append({'Year': year, 'Count': raw_count})
        tax_arbs.append({'Year': year, 'Count': tax_count})
        
    raw_df = pd.DataFrame(raw_arbs).sort_values('Year')
    tax_df = pd.DataFrame(tax_arbs).sort_values('Year')
    
    print(f"Total Raw Arbitrages: {raw_df['Count'].sum()}")
    print(f"Total Tax-Adjusted Arbitrages: {tax_df['Count'].sum()}")
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    barplot = sns.barplot(x='Year', y='Count', data=raw_df, palette='Blues', hue='Year', legend=False)
    plt.title(f"Okazje arbitrażowe (Raw, Suma: {raw_df['Count'].sum()})", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba okazji", fontsize=12)
    plt.xlabel("Rok", fontsize=12)
    for p in barplot.patches:
        if p.get_height() > 0:
            barplot.annotate(format(p.get_height(), '.0f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points',
                           fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_arbitrages_per_year_raw.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 7))
    barplot = sns.barplot(x='Year', y='Count', data=tax_df, palette='Reds', hue='Year', legend=False)
    plt.title(f"Okazje arbitrażowe po 12% podatku (Suma: {tax_df['Count'].sum()})", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba okazji", fontsize=12)
    plt.xlabel("Rok", fontsize=12)
    for p in barplot.patches:
        if p.get_height() > 0:
            barplot.annotate(format(p.get_height(), '.0f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points',
                           fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_arbitrages_per_year_tax.png", dpi=300, bbox_inches='tight')
    
    # 3. Market Corrections & Odds Movement
    def get_prob(o1, o2):
        if pd.isnull(o1) or pd.isnull(o2) or o1 <= 1 or o2 <= 1:
            return np.nan
        margin = 1/o1 + 1/o2
        return (1/o1) / margin

    df['prob_avg_open'] = df.apply(lambda x: get_prob(x['avg_open_home'], x['avg_open_away']), axis=1)
    df['prob_avg_close'] = df.apply(lambda x: get_prob(x['avg_odds_home'], x['avg_odds_away']), axis=1)
    
    plt.figure(figsize=(12, 7))
    prob_shift = (df['prob_avg_close'] - df['prob_avg_open']).dropna()
    sns.histplot(x=prob_shift, bins=100, kde=True, color='royalblue')
    plt.title("Korekty rynkowe: Zmiana prawdopodobieństwa (Zamknięcie - Otwarcie)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Zmiana prawdopodobieństwa (T1)", fontsize=12)
    plt.ylabel("Częstotliwość", fontsize=12)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_market_corrections.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(12, 7))
    odds_move = ((df['avg_odds_home'] - df['avg_open_home']) / df['avg_open_home'] * 100).dropna()
    odds_move = odds_move[odds_move.between(-50, 50)]
    sns.histplot(x=odds_move, bins=100, kde=True, color='darkorange')
    plt.title("Dystrybucja zmian kursów otwarcia (%)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Procentowa zmiana (Otwarcie do Zamknięcia)", fontsize=12)
    plt.ylabel("Częstotliwość", fontsize=12)
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_odds_movement_dist.png", dpi=300, bbox_inches='tight')
    
    # 4. Favorite Probability
    plt.figure(figsize=(12, 7))
    fav_prob = df['prob_avg_open'].apply(lambda p: max(p, 1-p)).dropna()
    sns.histplot(x=fav_prob, bins=50, kde=True, color='forestgreen')
    plt.title("Dystrybucja prawdopodobieństwa faworyta (Otwarcie)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Implikowane prawdopodobieństwo faworyta", fontsize=12)
    plt.ylabel("Częstotliwość", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_favorite_prob_dist.png", dpi=300, bbox_inches='tight')
    
    # 5. Regional Efficiency
    from sklearn.metrics import roc_auc_score
    tier1_keywords = [
        "LEC", "LCS", "LTA", "LCK", "LPL", 
        "European Championship", "Championship Series", "Champions Korea", "Pro League",
        "Mid-Season Invitational", "Mid Season Invitational", "First Stand", 
        "World Championship", "Mistrzostwa Świata"
    ]
    def is_tier1(tournament):
        t = str(tournament)
        if "Pro League" in t and "Oceanic" not in t and "Continental" not in t:
            return True
        return any(kw in t for kw in tier1_keywords)

    df['is_tier1'] = df['tournament'].apply(is_tier1)
    
    tier_results = []
    for tier_name, mask in [("Tier 1", df['is_tier1']), ("Regional/ERL", ~df['is_tier1'])]:
        tier_df = df[mask].dropna(subset=['t1_win', 'prob_avg_open'])
        if len(tier_df) > 0:
            auc = roc_auc_score(tier_df['t1_win'], tier_df['prob_avg_open'])
            print(f"{tier_name} AUC: {auc:.4f}")
            tier_results.append({'Tier': tier_name, 'Bookie AUC': auc})
            
    plt.figure(figsize=(8, 9))
    res_df = pd.DataFrame(tier_results)
    barplot = sns.barplot(x='Tier', y='Bookie AUC', data=res_df, palette='coolwarm', hue='Tier', legend=False)
    plt.title("Efektywność rynku: AUC bukmacherów wg Tieru ligi", fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0.7, 0.76)
    plt.ylabel("AUC (Kursy Otwarcia)", fontsize=12)
    plt.xlabel("Poziom rozgrywek (Tier)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(res_df['Bookie AUC']):
        plt.text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_regional_efficiency.png", dpi=300, bbox_inches='tight')
    
    # 6. BoN Efficiency
    bon_results = []
    for bon in [1, 3, 5]:
        subset = df[df['BoN'] == bon].dropna(subset=['t1_win', 'prob_avg_open'])
        if len(subset) > 0:
            auc = roc_auc_score(subset['t1_win'], subset['prob_avg_open'])
            print(f"Bo{bon} AUC: {auc:.4f}")
            bon_results.append({'Format': f'Best-of-{bon}', 'Bookmaker AUC': auc})
            
    plt.figure(figsize=(8, 9))
    bon_df = pd.DataFrame(bon_results)
    barplot = sns.barplot(x='Format', y='Bookmaker AUC', data=bon_df, palette='viridis', hue='Format', legend=False)
    plt.title("Efektywność rynku: AUC bukmacherów wg formatu meczu", fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0.5, 0.80)
    plt.ylabel("AUC (Kursy Otwarcia)", fontsize=12)
    plt.xlabel("Format rozgrywek", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(bon_df['Bookmaker AUC']):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_bookie_auc_by_bon.png", dpi=300, bbox_inches='tight')
    
    print("02_eda_market.py completed successfully.")

if __name__ == "__main__":
    main()
