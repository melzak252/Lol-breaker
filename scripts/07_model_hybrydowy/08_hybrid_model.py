import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss
from scipy.optimize import minimize
import os

def main():
    print("Running 08_hybrid_model.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    # 1. Load Data
    df_meta = pd.read_csv("data/golgg_stacking_results.csv")
    df_odds = pd.read_csv("data/odds.csv")
    
    # Ensure IDs are strings
    df_meta['golgg_match_id'] = df_meta['golgg_match_id'].astype(str)
    df_odds['golgg_match_id'] = df_odds['golgg_match_id'].astype(str)
    
    # Calculate Market Prob (Wisdom of the Crowd)
    df_odds['prob_market'] = 1 / df_odds['avg_odds_home'] / ((1/df_odds['avg_odds_home']) + (1/df_odds['avg_odds_away']))
    
    # Max odds for simulation
    df_odds['max_open_t1'] = df_odds[['odds1_sts_open', 'odds1_superbet_open', 'odds1_betclic_open', 'odds1_efortuna_open', 'odds1_lv_bet_open', 'odds1_betfan_open']].max(axis=1)
    df_odds['max_open_t2'] = df_odds[['odds2_sts_open', 'odds2_superbet_open', 'odds2_betclic_open', 'odds2_efortuna_open', 'odds2_lv_bet_open', 'odds2_betfan_open']].max(axis=1)

    # Merge
    df = pd.merge(df_meta[['golgg_match_id', 'metamodel_lgbm_prob', 'y_true', 'date']], 
                  df_odds[['golgg_match_id', 'prob_market', 'max_open_t1', 'max_open_t2']], 
                  on='golgg_match_id')
    
    df = df.dropna(subset=['metamodel_lgbm_prob', 'prob_market', 'y_true', 'max_open_t1', 'max_open_t2'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter since 2024 for final evaluation (Modern Era)
    df = df[df['date'] >= '2024-01-01'].copy()
    
    # 2. Optimize Hybrid Weight (Alpha)
    def objective(alpha):
        p_hybrid = alpha * df['metamodel_lgbm_prob'] + (1 - alpha) * df['prob_market']
        return log_loss(df['y_true'], p_hybrid)
    
    res = minimize(objective, 0.5, bounds=[(0, 1)])
    best_alpha = res.x[0]
    print(f"Optimal Alpha (Metamodel Weight) since 2024: {best_alpha:.4f}")
    
    df['prob_hybrid'] = best_alpha * df['metamodel_lgbm_prob'] + (1 - best_alpha) * df['prob_market']
    
    # 3. Simulation Parameters
    start_bankroll = 100.0
    stake = 10.0
    tax = 0.12
    net_mult = 0.88
    
    df = df.sort_values('date')
    br_meta = [start_bankroll]
    br_market = [start_bankroll]
    br_hybrid = [start_bankroll]
    dates = [df['date'].min()]
    
    curr_meta = start_bankroll
    curr_market = start_bankroll
    curr_hybrid = start_bankroll
    
    stats = {
        "Metamodel": {"turnover": 0, "count": 0},
        "Market (Avg)": {"turnover": 0, "count": 0},
        "Hybrid Model": {"turnover": 0, "count": 0}
    }

    for _, row in df.iterrows():
        y = row['y_true']
        
        # Metamodel
        p = row['metamodel_lgbm_prob']
        ev1 = (row['max_open_t1'] * net_mult * p) - 1
        ev2 = (row['max_open_t2'] * net_mult * (1-p)) - 1
        if ev1 > 0.05:
            stats["Metamodel"]["count"] += 1
            stats["Metamodel"]["turnover"] += stake
            curr_meta += (stake * row['max_open_t1'] * net_mult) - stake if y == 1 else -stake
        elif ev2 > 0.05:
            stats["Metamodel"]["count"] += 1
            stats["Metamodel"]["turnover"] += stake
            curr_meta += (stake * row['max_open_t2'] * net_mult) - stake if y == 0 else -stake
            
        # Market
        p = row['prob_market']
        ev1 = (row['max_open_t1'] * net_mult * p) - 1
        ev2 = (row['max_open_t2'] * net_mult * (1-p)) - 1
        if ev1 > 0.05:
            stats["Market (Avg)"]["count"] += 1
            stats["Market (Avg)"]["turnover"] += stake
            curr_market += (stake * row['max_open_t1'] * net_mult) - stake if y == 1 else -stake
        elif ev2 > 0.05:
            stats["Market (Avg)"]["count"] += 1
            stats["Market (Avg)"]["turnover"] += stake
            curr_market += (stake * row['max_open_t2'] * net_mult) - stake if y == 0 else -stake

        # Hybrid
        p = row['prob_hybrid']
        ev1 = (row['max_open_t1'] * net_mult * p) - 1
        ev2 = (row['max_open_t2'] * net_mult * (1-p)) - 1
        if ev1 > 0.05:
            stats["Hybrid Model"]["count"] += 1
            stats["Hybrid Model"]["turnover"] += stake
            curr_hybrid += (stake * row['max_open_t1'] * net_mult) - stake if y == 1 else -stake
        elif ev2 > 0.05:
            stats["Hybrid Model"]["count"] += 1
            stats["Hybrid Model"]["turnover"] += stake
            curr_hybrid += (stake * row['max_open_t2'] * net_mult) - stake if y == 0 else -stake
            
        br_meta.append(curr_meta)
        br_market.append(curr_market)
        br_hybrid.append(curr_hybrid)
        dates.append(row['date'])

    # Final Stats
    results = []
    for name in ["Metamodel", "Market (Avg)", "Hybrid Model"]:
        auc = roc_auc_score(df['y_true'], df['metamodel_lgbm_prob' if name=="Metamodel" else ('prob_market' if name=="Market (Avg)" else 'prob_hybrid')])
        ll = log_loss(df['y_true'], df['metamodel_lgbm_prob' if name=="Metamodel" else ('prob_market' if name=="Market (Avg)" else 'prob_hybrid')])
        
        br_history = br_meta if name=="Metamodel" else (br_market if name=="Market (Avg)" else br_hybrid)
        br_history = np.array(br_history)
        
        final_br = br_history[-1]
        final_profit = final_br - start_bankroll
        yield_val = final_profit / stats[name]["turnover"] if stats[name]["turnover"] > 0 else 0
        roi_val = final_profit / start_bankroll
        
        # Max Drawdown
        peak = np.maximum.accumulate(br_history)
        drawdown = (peak - br_history) / peak
        max_dd = np.max(drawdown) * 100
        
        # Min Bankroll
        min_br = np.min(br_history)
        
        results.append({
            "Model": name, 
            "AUC": auc, 
            "LogLoss": ll, 
            "Yield": yield_val, 
            "ROI (Total)": roi_val, 
            "Max DD (%)": max_dd,
            "Min Bankroll": min_br,
            "Bets": stats[name]["count"]
        })
    
    res_df = pd.DataFrame(results)
    print("\n=== FINAL COMPARISON (SINCE 2024) ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("docs/assets/hybrid_model_results_2024.csv", index=False)
    
    # 4. Bankroll Comparison Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, br_meta, label='Metamodel Only', color='blue', alpha=0.7)
    plt.plot(dates, br_market, label='Market Only (EV)', color='red', alpha=0.7)
    plt.plot(dates, br_hybrid, label='Hybrid Model', color='green', linewidth=2)
    plt.title("Bankroll Over Time: Metamodel vs. Market vs. Hybrid (2024-2026)")
    plt.ylabel("Bankroll ($)")
    plt.xlabel("Date")
    plt.axhline(100, color='black', linestyle='--')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("docs/assets/hybrid_bankroll_comparison_2024.png")
    
    # 5. Alpha Optimization Plot
    alphas = np.linspace(0, 1, 100)
    losses = [log_loss(df['y_true'], a * df['metamodel_lgbm_prob'] + (1 - a) * df['prob_market']) for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, losses, label='Log Loss')
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best Alpha: {best_alpha:.2f}')
    plt.title("Hybrid Model: Alpha Optimization (Log Loss Minimization)")
    plt.xlabel("Metamodel Weight (Alpha)")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("docs/assets/hybrid_alpha_optimization.png")
    
    # 6. Calibration Plot
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(8, 8))
    for name, col in [("Metamodel", "metamodel_lgbm_prob"), 
                      ("Market", "prob_market"), 
                      ("Hybrid", "prob_hybrid")]:
        prob_true, prob_pred = calibration_curve(df['y_true'], df[col], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.title("Calibration Curve: Metamodel vs. Market vs. Hybrid")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("docs/assets/hybrid_calibration_comparison.png")
    
    print("\n08_hybrid_model.py completed.")

if __name__ == "__main__":
    main()
