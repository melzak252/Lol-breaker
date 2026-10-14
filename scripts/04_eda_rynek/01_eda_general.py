import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def main():
    print("Running 01_eda_general.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # 1. Load GOL.GG Data
    print("Loading golgg_matches.json...")
    with open("data/golgg_matches.json", "r", encoding="utf-8") as f:
        matches = json.load(f)
        
    df_golgg = pd.DataFrame([{
        'match_id': m['match_id'],
        'date': m['date'],
        'tournament': m['tournament'],
        'BoN': m['BoN'],
        't1_win': m['t1_win'],
        'games_count': len(m['games'])
    } for m in matches if not m['draw']])
    
    df_golgg['date'] = pd.to_datetime(df_golgg['date'])
    df_golgg['year'] = df_golgg['date'].dt.year
    
    # 2. Load Odds Data
    print("Loading odds.csv...")
    df_odds = pd.read_csv("data/odds.csv")
    df_odds['date'] = pd.to_datetime(df_odds['odds_date'])
    df_odds['year'] = df_odds['date'].dt.year
    
    # 3. Matches per Year (GOL.GG)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    year_counts = df_golgg.groupby('year').size()
    barplot = sns.barplot(x=year_counts.index, y=year_counts.values, palette="viridis", hue=year_counts.index, legend=False)
    
    # Add data labels
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontweight='bold')

    plt.title("Liczba meczów w skali roku (Zbiór GOL.GG)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba meczów", fontsize=12)
    plt.xlabel("Rok", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_matches_per_year.png", dpi=300, bbox_inches='tight')
    
    # 4. Games per Year (GOL.GG)
    plt.figure(figsize=(12, 7))
    games_counts = df_golgg.groupby('year')['games_count'].sum()
    barplot = sns.barplot(x=games_counts.index, y=games_counts.values, palette="magma", hue=games_counts.index, legend=False)
    
    # Add data labels
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontweight='bold')

    plt.title("Liczba indywidualnych gier w skali roku (Zbiór GOL.GG)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba gier", fontsize=12)
    plt.xlabel("Rok", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_games_per_year.png", dpi=300, bbox_inches='tight')
    
    # 5. Matches per Year (Odds)
    plt.figure(figsize=(12, 7))
    odds_year_counts = df_odds.groupby('year').size()
    barplot = sns.barplot(x=odds_year_counts.index, y=odds_year_counts.values, palette="plasma", hue=odds_year_counts.index, legend=False)
    
    # Add data labels
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontweight='bold')

    plt.title("Liczba meczów w skali roku (Zbiór OddsPortal)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba meczów", fontsize=12)
    plt.xlabel("Rok", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_matches_per_year_odds.png", dpi=300, bbox_inches='tight')
    
    # 6. BoN Distribution (Odds)
    plt.figure(figsize=(10, 7))
    # Calculate BoN from scores if not present
    if 'BoN' not in df_odds.columns:
        df_odds['BoN'] = df_odds['t1_score'] + df_odds['t2_score']
        # Map to standard BoN formats (1, 3, 5)
        df_odds['BoN'] = df_odds['BoN'].apply(lambda x: 1 if x <= 1 else (3 if x <= 3 else 5))
        
    bon_counts = df_odds['BoN'].value_counts().sort_index()
    barplot = sns.barplot(x=bon_counts.index, y=bon_counts.values, palette="viridis", hue=bon_counts.index, legend=False)
    plt.title("Dystrybucja formatów meczów (Best-of-N)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba meczów", fontsize=12)
    plt.xlabel("Format (BoN)", fontsize=12)
    for i, v in enumerate(bon_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_bon_distribution_odds.png", dpi=300, bbox_inches='tight')

    # 6b. Format Evolution Over Time (GOL.GG)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Prepare data for evolution
    # Ensure BoN is standard
    df_golgg['format'] = df_golgg['BoN'].apply(lambda x: f"Bo{x}" if x in [1, 3, 5] else "Inne")
    evo_data = df_golgg.groupby(['year', 'format']).size().unstack(fill_value=0)
    
    # Reorder columns to Bo1, Bo3, Bo5
    cols = [c for c in ["Bo1", "Bo3", "Bo5"] if c in evo_data.columns]
    evo_data = evo_data[cols]
    
    evo_data.plot(kind='bar', stacked=True, ax=plt.gca(), color=sns.color_palette("viridis", len(cols)))
    
    plt.title("Ewolucja formatów meczów na przestrzeni lat (GOL.GG)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Liczba meczów", fontsize=12)
    plt.xlabel("Rok", fontsize=12)
    plt.legend(title="Format", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total count on top of each bar
    totals = evo_data.sum(axis=1)
    for i, total in enumerate(totals):
        plt.text(i, total + 50, str(int(total)), ha='center', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    plt.savefig("docs/assets/eda_format_evolution.png", dpi=300, bbox_inches='tight')
    print("Saved docs/assets/eda_format_evolution.png")
    
    # 7. Win Distribution (Odds)
    plt.figure(figsize=(10, 8))
    win_counts = df_odds['t1_win'].value_counts().sort_index()
    labels = ['Wygrana Drużyny 2 (Red)', 'Wygrana Drużyny 1 (Blue)']
    colors = sns.color_palette("coolwarm", 2)
    plt.pie(win_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0), 
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title("Dystrybucja wygranych: Blue Side vs Red Side", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_win_distribution_odds.png", dpi=300, bbox_inches='tight', transparent=True)
    
    # 8. Tier Breakdown
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
        
    df_odds['is_tier1'] = df_odds['tournament'].apply(is_tier1)
    tier_stats = df_odds.groupby('is_tier1').size()
    print("\n=== Tier Breakdown (Odds Dataset) ===")
    print(f"Tier 1: {tier_stats.get(True, 0)}")
    print(f"Regional/ERL: {tier_stats.get(False, 0)}")
    
    print("\n01_eda_general.py completed successfully.")

if __name__ == "__main__":
    main()
