import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Running 07_market_benchmarking.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # Load odds data
    df = pd.read_csv("data/odds.csv")
    df['odds_date'] = pd.to_datetime(df['odds_date'])
    
    # Calculate probabilities from average odds
    df['prob_avg_close'] = 1 / df['avg_odds_home'] / ((1/df['avg_odds_home']) + (1/df['avg_odds_away']))
    df['prob_avg_open'] = 1 / df['avg_open_home'] / ((1/df['avg_open_home']) + (1/df['avg_open_away']))
    
    # Max odds for arbitrage
    df['max_open_t1'] = df[['odds1_sts_open', 'odds1_superbet_open', 'odds1_betclic_open', 'odds1_efortuna_open', 'odds1_lv_bet_open', 'odds1_betfan_open']].max(axis=1)
    df['max_open_t2'] = df[['odds2_sts_open', 'odds2_superbet_open', 'odds2_betclic_open', 'odds2_efortuna_open', 'odds2_lv_bet_open', 'odds2_betfan_open']].max(axis=1)
    
    # Filter for matches with both open and close odds
    df = df.dropna(subset=['prob_avg_open', 'prob_avg_close', 't1_win', 'max_open_t1', 'max_open_t2'])
    
    # Convert t1_win to int
    df['t1_win'] = df['t1_win'].astype(int)
    
    # 1. Opening vs Closing Efficiency
    metrics = []
    for stage, col in [("Opening Odds", "prob_avg_open"), ("Closing Odds", "prob_avg_close")]:
        auc = roc_auc_score(df['t1_win'], df[col])
        ll = log_loss(df['t1_win'], df[col])
        metrics.append({"Stage": stage, "AUC": auc, "LogLoss": ll})
    
    metrics_df = pd.DataFrame(metrics)
    print("\n=== Market Efficiency: Open vs Close ===")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv("docs/assets/market_open_vs_close_metrics.csv", index=False)

    # Filter since 2020
    df = df[df['odds_date'] >= '2020-01-01'].copy()
    
    # Calculate probabilities from average odds
    df['prob_avg_close'] = 1 / df['avg_odds_home'] / ((1/df['avg_odds_home']) + (1/df['avg_odds_away']))
    df['prob_avg_open'] = 1 / df['avg_open_home'] / ((1/df['avg_open_home']) + (1/df['avg_open_away']))
    
    # Max odds for arbitrage and EV
    df['max_open_t1'] = df[['odds1_sts_open', 'odds1_superbet_open', 'odds1_betclic_open', 'odds1_efortuna_open', 'odds1_lv_bet_open', 'odds1_betfan_open']].max(axis=1)
    df['max_open_t2'] = df[['odds2_sts_open', 'odds2_superbet_open', 'odds2_betclic_open', 'odds2_efortuna_open', 'odds2_lv_bet_open', 'odds2_betfan_open']].max(axis=1)
    df['max_close_t1'] = df[['odds1_sts_close', 'odds1_superbet_close', 'odds1_betclic_close', 'odds1_efortuna_close', 'odds1_lv_bet_close', 'odds1_betfan_close']].max(axis=1)
    df['max_close_t2'] = df[['odds2_sts_close', 'odds2_superbet_close', 'odds2_betclic_close', 'odds2_efortuna_close', 'odds2_lv_bet_close', 'odds2_betfan_close']].max(axis=1)
    
    # Filter for matches with both open and close odds
    df = df.dropna(subset=['prob_avg_open', 'prob_avg_close', 't1_win', 'max_open_t1', 'max_open_t2', 'max_close_t1', 'max_close_t2'])
    
    # Convert t1_win to int
    df['t1_win'] = df['t1_win'].astype(int)

    # 2. Wisdom of the Crowd: ROI Simulation (Blind Strategies)
    results_roi = []
    
    # EV Betting: Closing Odds as "True" Probability
    df['true_prob'] = df['prob_avg_close']
    
    # EV for Opening Odds
    df['ev_open_t1'] = (df['max_open_t1'] * 0.88 * df['true_prob']) - 1
    df['ev_open_t2'] = (df['max_open_t2'] * 0.88 * (1 - df['true_prob'])) - 1
    
    # EV for Closing Odds (Should be near zero or negative due to tax)
    df['ev_close_t1'] = (df['max_close_t1'] * 0.88 * df['true_prob']) - 1
    df['ev_close_t2'] = (df['max_close_t2'] * 0.88 * (1 - df['true_prob'])) - 1
    
    # Simulation: Bankroll Over Time
    df = df.sort_values('odds_date')
    bankroll_open = [100]
    bankroll_close = [100]
    bankroll_arb = [100]
    bankroll_pct_cap = [100]
    dates = [df['odds_date'].min()]
    
    current_open = 100
    current_close = 100
    current_arb = 100
    current_pct_cap = 100
    
    count_open = 0
    count_close = 0
    count_arb = 0
    turnover_open = 0
    turnover_close = 0
    turnover_arb = 0

    for _, row in df.iterrows():
        # 1. EV Bet on Opening Odds (Fixed 10$)
        if row['ev_open_t1'] > 0.05:
            count_open += 1
            turnover_open += 10
            profit = (10 * row['max_open_t1'] * 0.88) - 10 if row['t1_win'] == 1 else -10
            current_open += profit
            
            # % Bankroll with Cap 100$
            stake = min(current_pct_cap * 0.02, 100)
            profit_cap = (stake * row['max_open_t1'] * 0.88) - stake if row['t1_win'] == 1 else -stake
            current_pct_cap += profit_cap
            
        elif row['ev_open_t2'] > 0.05:
            count_open += 1
            turnover_open += 10
            profit = (10 * row['max_open_t2'] * 0.88) - 10 if row['t1_win'] == 0 else -10
            current_open += profit
            
            # % Bankroll with Cap 100$
            stake = min(current_pct_cap * 0.02, 100)
            profit_cap = (stake * row['max_open_t2'] * 0.88) - stake if row['t1_win'] == 0 else -stake
            current_pct_cap += profit_cap
            
        # 2. EV Bet on Closing Odds (Fixed 10$)
        if row['ev_close_t1'] > 0.05:
            count_close += 1
            turnover_close += 10
            profit = (10 * row['max_close_t1'] * 0.88) - 10 if row['t1_win'] == 1 else -10
            current_close += profit
        elif row['ev_close_t2'] > 0.05:
            count_close += 1
            turnover_close += 10
            profit = (10 * row['max_close_t2'] * 0.88) - 10 if row['t1_win'] == 0 else -10
            current_close += profit

        # 3. Arbitrage (Tax 12%) (Fixed 20$)
        # We check if arb_tax < 1.0
        arb_val = (1/(row['max_open_t1']*0.88)) + (1/(row['max_open_t2']*0.88))
        if arb_val < 1.0:
            count_arb += 1
            turnover_arb += 20
            # Profit = Stake * (1 - arb) / arb
            profit_arb = 20 * (1 - arb_val) / arb_val
            current_arb += profit_arb
            
        bankroll_open.append(current_open)
        bankroll_close.append(current_close)
        bankroll_arb.append(current_arb)
        bankroll_pct_cap.append(current_pct_cap)
        dates.append(row['odds_date'])

    # Plot Bankroll Comparison (Fixed)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    plt.plot(dates, bankroll_open, label='EV Otwarcie (Stała stawka 10$)', color='blue', linewidth=2)
    plt.plot(dates, bankroll_close, label='EV Zamknięcie (Stała stawka 10$)', color='red', linewidth=2)
    plt.plot(dates, bankroll_arb, label='Arbitraż z podatkiem 12% (20$)', color='purple', linestyle='--', linewidth=2)
    plt.title("Ewolucja kapitału: Strategie EV vs. Arbitraż (Start: 100$)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Kapitał ($)", fontsize=12)
    plt.xlabel("Data", fontsize=12)
    plt.axhline(100, color='black', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/assets/market_bankroll_comparison.png", dpi=300, bbox_inches='tight')

    # Plot Bankroll % with Cap
    plt.figure(figsize=(12, 7))
    plt.plot(dates, bankroll_pct_cap, label='2% Kapitału (Limit 100$)', color='green', linewidth=2)
    plt.title("Ewolucja kapitału: Stawkowanie procentowe (2% z limitem 100$)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Kapitał ($)", fontsize=12)
    plt.xlabel("Data", fontsize=12)
    plt.axhline(100, color='black', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/assets/market_bankroll_pct_cap.png", dpi=300, bbox_inches='tight')

    results_roi.append({
        "Strategy": "EV Bet (Opening)", 
        "Yield": (current_open - 100) / turnover_open if turnover_open > 0 else 0, 
        "ROI": (current_open - 100) / 100,
        "Count": count_open
    })
    results_roi.append({
        "Strategy": "EV Bet (Closing)", 
        "Yield": (current_close - 100) / turnover_close if turnover_close > 0 else 0, 
        "ROI": (current_close - 100) / 100,
        "Count": count_close
    })
    results_roi.append({
        "Strategy": "Arbitrage (Tax 12%)", 
        "Yield": (current_arb - 100) / turnover_arb if turnover_arb > 0 else 0, 
        "ROI": (current_arb - 100) / 100,
        "Count": count_arb
    })

    roi_df = pd.DataFrame(results_roi)
    print("\n=== Blind Strategy ROI (Tax 12% included where applicable) ===")
    print(roi_df.to_string(index=False))
    roi_df.to_csv("docs/assets/market_blind_roi.csv", index=False)

    # 3. Bookmaker Strength (AUC per Bookie)
    bookies = [
        ('sts', 'odds1_sts_open', 'odds2_sts_open'),
        ('superbet', 'odds1_superbet_open', 'odds2_superbet_open'),
        ('betclic', 'odds1_betclic_open', 'odds2_betclic_open'),
        ('efortuna', 'odds1_efortuna_open', 'odds2_efortuna_open'),
        ('lv_bet', 'odds1_lv_bet_open', 'odds2_lv_bet_open'),
        ('betfan', 'odds1_betfan_open', 'odds2_betfan_open')
    ]
    bookie_auc = []
    total_matches = len(df)
    
    # Individual Bookmakers
    for name, c1, c2 in bookies:
        if c1 in df.columns and c2 in df.columns:
            prob_col = f'prob_{name}_open'
            df[prob_col] = (1/df[c1]) / ((1/df[c1]) + (1/df[c2]))
            subset = df.dropna(subset=[prob_col, 't1_win'])
            count = len(subset)
            if count > 100:
                auc = roc_auc_score(subset['t1_win'], subset[prob_col])
                bookie_auc.append({
                    "Bookmaker": name.upper(), 
                    "AUC": auc, 
                    "Count": count,
                    "Coverage (%)": (count / total_matches) * 100,
                    "Type": "Individual"
                })
    
    # Market Average (Wisdom of the Crowd)
    avg_auc = roc_auc_score(df['t1_win'], df['prob_avg_open'])
    bookie_auc.append({
        "Bookmaker": "ŚREDNIA RYNKOWA", 
        "AUC": avg_auc, 
        "Count": total_matches,
        "Coverage (%)": 100.0,
        "Type": "Benchmark"
    })
    
    b_auc_df = pd.DataFrame(bookie_auc).sort_values("AUC", ascending=False)
    
    # Plot Bookmaker Strength with Coverage
    fig, ax1 = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Color palette: Highlight Market Avg
    colors = ['#ff7f0e' if x == 'ŚREDNIA RYNKOWA' else '#1f77b4' for x in b_auc_df['Bookmaker']]
    
    barplot = sns.barplot(x="Bookmaker", y="AUC", data=b_auc_df, palette=colors, ax=ax1, hue="Bookmaker", legend=False)
    ax1.set_ylabel("AUC (Kursy Otwarcia)", fontsize=12)
    ax1.set_xlabel("Bukmacher", fontsize=12)
    ax1.set_ylim(0.65, 0.76)
    
    ax2 = ax1.twinx()
    sns.lineplot(x="Bookmaker", y="Coverage (%)", data=b_auc_df, marker='o', color='red', ax=ax2, label='Pokrycie rynku %', linewidth=2)
    ax2.set_ylabel("Pokrycie rynku (%)", fontsize=12)
    ax2.set_ylim(0, 110)
    
    plt.title("Siła predykcyjna bukmacherów: Indywidualnie vs. Średnia Rynkowa", fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, row in enumerate(b_auc_df.itertuples(index=False)):
        ax1.text(i, row.AUC + 0.002, f"n={row.Count}", ha='center', fontweight='bold', color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig("docs/assets/market_bookie_strength.png", dpi=300, bbox_inches='tight')

    print("\n07_market_benchmarking.py completed.")

if __name__ == "__main__":
    main()
