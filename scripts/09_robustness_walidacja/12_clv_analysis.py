import pandas as pd
import numpy as np
import os

def main():
    print("Running 12_clv_analysis.py...")
    
    # Load data
    df_meta = pd.read_csv("data/golgg_stacking_results.csv")
    df_odds = pd.read_csv("data/odds.csv")
    df_meta['golgg_match_id'] = df_meta['golgg_match_id'].astype(str)
    df_odds['golgg_match_id'] = df_odds['golgg_match_id'].astype(str)
    
    # Calculate Market Prob (Wisdom of the Crowd)
    df_odds['prob_market_open'] = 1 / df_odds['avg_open_home'] / ((1/df_odds['avg_open_home']) + (1/df_odds['avg_open_away']))
    df_odds['prob_market_close'] = 1 / df_odds['avg_odds_home'] / ((1/df_odds['avg_odds_home']) + (1/df_odds['avg_odds_away']))
    
    # Max odds for EV
    bookies = ['sts', 'superbet', 'betclic', 'efortuna', 'lv_bet', 'betfan']
    df_odds['max_open_t1'] = df_odds[[f'odds1_{b}_open' for b in bookies]].max(axis=1)
    df_odds['max_open_t2'] = df_odds[[f'odds2_{b}_open' for b in bookies]].max(axis=1)
    
    # Merge
    df = pd.merge(df_meta[['golgg_match_id', 'metamodel_lgbm_prob', 'y_true', 'date']], 
                  df_odds[['golgg_match_id', 'prob_market_open', 'prob_market_close', 'max_open_t1', 'max_open_t2', 'avg_odds_home', 'avg_odds_away']], 
                  on='golgg_match_id')
    
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2024-01-01'].sort_values('date').reset_index(drop=True)
    
    # Hybrid Model (Alpha = 0.5168)
    alpha = 0.5168
    df['prob_hybrid'] = alpha * df['metamodel_lgbm_prob'] + (1 - alpha) * df['prob_market_open']
    
    net_mult = 0.88
    
    clv_results = []
    
    for _, row in df.iterrows():
        p = row['prob_hybrid']
        ev1 = (row['max_open_t1'] * net_mult * p) - 1
        ev2 = (row['max_open_t2'] * net_mult * (1-p)) - 1
        
        if ev1 > 0.05:
            # We bet on T1 at max_open_t1
            # Closing price (fair) is 1 / prob_market_close
            fair_close_odds = 1 / row['prob_market_close']
            clv = (row['max_open_t1'] / fair_close_odds) - 1
            clv_results.append(clv)
        elif ev2 > 0.05:
            # We bet on T2 at max_open_t2
            fair_close_odds = 1 / (1 - row['prob_market_close'])
            clv = (row['max_open_t2'] / fair_close_odds) - 1
            clv_results.append(clv)
            
    avg_clv = np.mean(clv_results)
    beating_market_pct = np.mean([1 if x > 0 else 0 for x in clv_results]) * 100
    
    print(f"\n=== CLV ANALYSIS (2024-2026) ===")
    print(f"Average CLV: {avg_clv:.2%}")
    print(f"Bets Beating Closing Line: {beating_market_pct:.2f}%")
    
    with open("docs/assets/final_clv_stats.txt", "w") as f:
        f.write(f"Average CLV: {avg_clv:.2%}\n")
        f.write(f"Bets Beating Closing Line: {beating_market_pct:.2f}%\n")

    print("\n12_clv_analysis.py completed.")

if __name__ == "__main__":
    main()
