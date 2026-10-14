import pandas as pd
import numpy as np
import os

def main():
    print("Running 11_final_breakdowns.py...")
    
    # Load data
    df_meta = pd.read_csv("data/golgg_stacking_results.csv")
    df_odds = pd.read_csv("data/odds.csv")
    df_meta['golgg_match_id'] = df_meta['golgg_match_id'].astype(str)
    df_odds['golgg_match_id'] = df_odds['golgg_match_id'].astype(str)
    
    # Calculate Market Prob
    df_odds['prob_market'] = 1 / df_odds['avg_odds_home'] / ((1/df_odds['avg_odds_home']) + (1/df_odds['avg_odds_away']))
    
    # Max odds for EV
    bookies = ['sts', 'superbet', 'betclic', 'efortuna', 'lv_bet', 'betfan']
    for b in bookies:
        df_odds[f'odds1_{b}_open_net'] = df_odds[f'odds1_{b}_open'] * 0.88
        df_odds[f'odds2_{b}_open_net'] = df_odds[f'odds2_{b}_open'] * 0.88

    df_odds['max_open_t1'] = df_odds[[f'odds1_{b}_open' for b in bookies]].max(axis=1)
    df_odds['max_open_t2'] = df_odds[[f'odds2_{b}_open' for b in bookies]].max(axis=1)
    
    # Merge
    # Don't drop all NaNs yet, only essential ones
    df = pd.merge(df_meta[['golgg_match_id', 'metamodel_lgbm_prob', 'y_true', 'date']], 
                  df_odds[['golgg_match_id', 'prob_market', 'max_open_t1', 'max_open_t2', 'tournament'] + [f'odds1_{b}_open' for b in bookies] + [f'odds2_{b}_open' for b in bookies]], 
                  on='golgg_match_id')
    
    df = df.dropna(subset=['metamodel_lgbm_prob', 'prob_market', 'y_true', 'max_open_t1', 'max_open_t2'])
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2024-01-01'].sort_values('date').reset_index(drop=True)
    
    # Hybrid Model (Alpha = 0.5168)
    alpha = 0.5168
    df['prob_hybrid'] = alpha * df['metamodel_lgbm_prob'] + (1 - alpha) * df['prob_market']
    
    net_mult = 0.88
    f_kelly = 0.25
    stake_cap = 100.0
    min_stake = 2.0
    
    curr_br = 100.0
    results = []
    
    for _, row in df.iterrows():
        p = row['prob_hybrid']
        ev1 = (row['max_open_t1'] * net_mult * p) - 1
        ev2 = (row['max_open_t2'] * net_mult * (1-p)) - 1
        
        if ev1 > 0.05:
            o, prob, is_win = row['max_open_t1'], p, (1 if row['y_true'] == 1 else 0)
            # Find which bookie(s) gave the max odds
            best_bs = []
            for b in bookies:
                if row[f'odds1_{b}_open'] == o:
                    best_bs.append(b.upper())
            # If multiple, pick the first one for simplicity in breakdown
            best_b = best_bs[0] if best_bs else "UNKNOWN"
        elif ev2 > 0.05:
            o, prob, is_win = row['max_open_t2'], 1-p, (1 if row['y_true'] == 0 else 0)
            best_bs = []
            for b in bookies:
                if row[f'odds2_{b}_open'] == o:
                    best_bs.append(b.upper())
            best_b = best_bs[0] if best_bs else "UNKNOWN"
        else: continue
            
        b_net = (o * net_mult) - 1
        stake = min(curr_br * ((b_net * prob - (1 - prob)) / b_net) * f_kelly, stake_cap)
        if stake < min_stake or curr_br < stake: continue
        
        profit = (stake * o * net_mult) - stake if is_win else -stake
        curr_br += profit
        
        results.append({
            "Tournament": row['tournament'],
            "Bookmaker": best_b,
            "Profit": profit,
            "Stake": stake
        })
        
    res_df = pd.DataFrame(results)
    
    # 1. Top Tournaments
    top_tournaments = res_df.groupby('Tournament')['Profit'].sum().sort_values(ascending=False).head(10)
    print("\n=== TOP 10 PROFITABLE TOURNAMENTS ===")
    print(top_tournaments)
    top_tournaments.to_csv("docs/assets/final_top_tournaments.csv")
    
    # 2. Bookmaker Profits
    bookie_profits = res_df.groupby('Bookmaker').agg({'Profit': 'sum', 'Stake': 'count'}).rename(columns={'Stake': 'Bets'})
    bookie_profits['Yield'] = bookie_profits['Profit'] / (res_df.groupby('Bookmaker')['Stake'].sum())
    print("\n=== PROFITS PER BOOKMAKER ===")
    print(bookie_profits.sort_values('Profit', ascending=False))
    bookie_profits.to_csv("docs/assets/final_bookmaker_profits.csv")
    
    print("\n11_final_breakdowns.py completed.")

if __name__ == "__main__":
    main()
