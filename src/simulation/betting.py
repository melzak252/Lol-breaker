import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def calculate_kelly_stake(p: float, odds: float, fraction: float = 0.5) -> float:
    """
    Calculates the Kelly Criterion stake as a fraction of the bankroll.
    
    Args:
        p: Predicted probability of winning.
        odds: Decimal odds offered by the bookmaker.
        fraction: Fraction of Kelly to use (e.g., 0.5 for half-Kelly).
        
    Returns:
        Stake as a fraction of the bankroll (0 to 1).
    """
    if odds <= 1:
        return 0.0
    b = odds - 1
    q = 1 - p
    f_star = (p * b - q) / b
    return max(0.0, f_star * fraction)

def simulate_betting(
    df: pd.DataFrame,
    prob_col: str = 'meta_prob',
    odds1_col: str = 'avg_odds1',
    odds2_col: str = 'avg_odds2',
    target_col: str = 'y_true',
    initial_bankroll: float = 100.0,
    kelly_fraction: float = 0.5,
    max_stake_pct: float = 0.05,
    tax_rate: float = 0.0,
    ev_threshold: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates a betting strategy based on predicted probabilities and odds.
    
    Args:
        df: DataFrame containing probabilities, odds, and true outcomes.
        prob_col: Column name for predicted probability of team 1 winning.
        odds1_col: Column name for decimal odds of team 1 winning.
        odds2_col: Column name for decimal odds of team 2 winning.
        target_col: Column name for true outcome (1 if team 1 won, 0 if team 2 won).
        initial_bankroll: Starting bankroll.
        kelly_fraction: Fraction of Kelly Criterion to use.
        max_stake_pct: Maximum stake as a percentage of the bankroll.
        tax_rate: Tax rate on the stake (e.g., 0.12 for 12% tax).
        ev_threshold: Minimum expected value (EV) to place a bet.
        
    Returns:
        Tuple containing the history DataFrame and a dictionary of summary metrics.
    """
    bankroll = initial_bankroll
    history = []
    
    total_wagered = 0.0
    total_profit = 0.0
    num_bets = 0
    
    for _, row in df.iterrows():
        p1 = row[prob_col]
        p2 = 1 - p1
        
        o1_raw = row[odds1_col]
        o2_raw = row[odds2_col]
        
        if pd.isna(o1_raw) or pd.isna(o2_raw) or o1_raw <= 1 or o2_raw <= 1:
            continue
            
        # Effective odds after tax
        o1_eff = o1_raw * (1 - tax_rate)
        o2_eff = o2_raw * (1 - tax_rate)
        
        # Calculate EV
        ev1 = p1 * o1_eff - 1
        ev2 = p2 * o2_eff - 1
        
        stake_pct = 0.0
        profit = 0.0
        bet_on = None
        
        if ev1 >= ev_threshold:
            s1 = calculate_kelly_stake(p1, o1_eff, kelly_fraction)
            if s1 > 0:
                stake_pct = min(s1, max_stake_pct)
                bet_on = 1
                stake = bankroll * stake_pct
                if row[target_col] == 1:
                    profit = stake * (o1_eff - 1)
                else:
                    profit = -stake
                    
        elif ev2 >= ev_threshold:
            s2 = calculate_kelly_stake(p2, o2_eff, kelly_fraction)
            if s2 > 0:
                stake_pct = min(s2, max_stake_pct)
                bet_on = 2
                stake = bankroll * stake_pct
                if row[target_col] == 0:
                    profit = stake * (o2_eff - 1)
                else:
                    profit = -stake
        
        if stake_pct > 0:
            total_wagered += bankroll * stake_pct
            total_profit += profit
            num_bets += 1
            bankroll += profit
            
        history.append({
            'date': row.get('date'),
            'bankroll': bankroll,
            'stake_pct': stake_pct,
            'profit': profit,
            'bet_on': bet_on,
            'ev': ev1 if bet_on == 1 else (ev2 if bet_on == 2 else 0.0)
        })
        
    hist_df = pd.DataFrame(history)
    
    roi = total_profit / total_wagered if total_wagered > 0 else 0.0
    max_drawdown = 0.0
    if not hist_df.empty:
        cummax = hist_df['bankroll'].cummax()
        drawdown = (cummax - hist_df['bankroll']) / cummax
        max_drawdown = drawdown.max()
        
    metrics = {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'total_profit': total_profit,
        'total_wagered': total_wagered,
        'roi': roi,
        'num_bets': num_bets,
        'max_drawdown': max_drawdown,
        'growth_pct': (bankroll / initial_bankroll - 1)
    }
    
    return hist_df, metrics
