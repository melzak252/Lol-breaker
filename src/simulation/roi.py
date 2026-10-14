import pandas as pd
from typing import List, Dict, Any, Tuple

def calculate_kelly(p: float, decimal_odds: float, fraction: float = 0.5) -> float:
    """Calculate the Kelly Criterion bet size."""
    if decimal_odds <= 1: 
        return 0
    b = decimal_odds - 1
    q = 1 - p
    f_star = (p * b - q) / b
    return max(0, f_star * fraction)

def run_simulation(df_eval: pd.DataFrame, matched_data_list: List[Dict[str, Any]], bookie_name: str, ev_threshold: float = 0.05, initial_bankroll: float = 1000.0, max_bet_fraction: float = 0.05, tax_rate: float = 0.12, use_closing: bool = False) -> Tuple[float, int, float, float, pd.DataFrame]:
    """
    Run a betting simulation using the Kelly Criterion.
    """
    bankroll = initial_bankroll
    trajectory = [{"date": df_eval['date_dt'].min(), "bankroll": bankroll}]
    total_bets, total_invested = 0, 0
    
    # Ensure df_eval is indexed by match_id for fast lookup
    if "match_id" in df_eval.columns:
        df_eval_indexed = df_eval.set_index("match_id")
    else:
        df_eval_indexed = df_eval
        
    # Update matched_data with probabilities from df_eval
    sim_data = []
    for m in matched_data_list:
        mid = m["match_id"]
        if mid in df_eval_indexed.index:
            row = df_eval_indexed.loc[mid]
            if isinstance(row, pd.Series):
                prob = row["prob_calibrated"]
                y_true = row["y_true"]
            else:
                prob = row["prob_calibrated"].iloc[0]
                y_true = row["y_true"].iloc[0]
                
            m_copy = m.copy()
            m_copy["prob_home"] = prob
            m_copy["y_true"] = y_true
            sim_data.append(m_copy)
            
    sim_data.sort(key=lambda x: x["date_dt"])
    
    max_ev_seen = -1.0

    for m in sim_data:
        p_home = m["prob_home"]
        p_away = 1 - p_home
        
        target_bookies = ["STS", "LV BET", "Superbet", "eFortuna", "BETFAN"]
        
        def get_odds(b_name, side):
            mid = m["match_id"]
            row = df_eval_indexed.loc[mid]
            if not isinstance(row, pd.Series): row = row.iloc[0]
            
            suffix = "" if use_closing else "_opening"
            col = f"{b_name}{suffix}_odds{1 if side=='home' else 2}"
            val = row.get(col)
            return float(val) if pd.notna(val) and val > 0 else None

        if bookie_name == "MAX":
            h_odds_list = [get_odds(b, "home") for b in target_bookies]
            a_odds_list = [get_odds(b, "away") for b in target_bookies]
            h_odds_list = [o for o in h_odds_list if o is not None]
            a_odds_list = [o for o in a_odds_list if o is not None]
            
            if not h_odds_list or not a_odds_list: continue
            odds_home_raw = max(h_odds_list)
            odds_away_raw = max(a_odds_list)
        else:
            odds_home_raw = get_odds(bookie_name, "home")
            odds_away_raw = get_odds(bookie_name, "away")
            if odds_home_raw is None or odds_away_raw is None: continue

        odds_home_eff = odds_home_raw * (1 - tax_rate)
        odds_away_eff = odds_away_raw * (1 - tax_rate)
        
        ev_home = p_home * odds_home_eff - 1
        ev_away = p_away * odds_away_eff - 1
        
        max_ev_seen = max(max_ev_seen, ev_home, ev_away)
        
        bet_amount, odds_taken_eff, won = 0, 0, False
        
        if ev_home >= ev_threshold:
            bet_amount = bankroll * calculate_kelly(p_home, odds_home_eff, 0.5)
            odds_taken_eff, won = odds_home_eff, m["y_true"] == 1
        elif ev_away >= ev_threshold:
            bet_amount = bankroll * calculate_kelly(p_away, odds_away_eff, 0.5)
            odds_taken_eff, won = odds_away_eff, m["y_true"] == 0
            
        if bet_amount > 0:
            bet_amount = min(bet_amount, bankroll * max_bet_fraction)
            total_bets += 1
            total_invested += bet_amount
            bankroll += (bet_amount * (odds_taken_eff - 1)) if won else -bet_amount
            trajectory.append({"date": m["date_dt"], "bankroll": bankroll})
            
    roi = (bankroll - initial_bankroll) / total_invested if total_invested > 0 else 0
    # print(f"DEBUG: {bookie_name} Max EV seen: {max_ev_seen:.4f}")
    return bankroll, total_bets, total_invested, roi, pd.DataFrame(trajectory)
