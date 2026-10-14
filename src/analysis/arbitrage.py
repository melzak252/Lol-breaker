import pandas as pd
from typing import List, Dict, Any, Optional

def find_arbitrage_opportunities(
    df: pd.DataFrame,
    bookies: List[str],
    tax_rate: float = 0.12,
    odds1_suffix: str = "_odds1",
    odds2_suffix: str = "_odds2",
    prefix: str = ""
) -> pd.DataFrame:
    """
    Finds arbitrage opportunities in a dataset across multiple bookmakers.
    
    Args:
        df: Input DataFrame.
        bookies: List of bookmaker names.
        tax_rate: Tax rate on the stake.
        odds1_suffix: Suffix for team 1 odds columns.
        odds2_suffix: Suffix for team 2 odds columns.
        prefix: Prefix for bookmaker names (e.g., "odds1_" or "").
        
    Returns:
        DataFrame containing arbitrage opportunities.
    """
    results = []
    
    for _, row in df.iterrows():
        h_list = []
        a_list = []
        
        for b in bookies:
            # Handle different column naming conventions
            col1 = f"{prefix}{b}{odds1_suffix}" if prefix else f"{b}{odds1_suffix}"
            col2 = f"{prefix}{b}{odds2_suffix}" if prefix else f"{b}{odds2_suffix}"
            
            # Special case for OddsPortal format: odds1_bookie_type
            if "odds1_" in col1 and "_" in odds1_suffix:
                # Already handled by the caller or suffix
                pass
            
            h = row.get(col1)
            a = row.get(col2)
            
            if pd.notna(h) and h > 0: h_list.append((h, b))
            if pd.notna(a) and a > 0: a_list.append((a, b))
            
        if not h_list or not a_list:
            continue
            
        best_h, b_h = max(h_list, key=lambda x: x[0])
        best_a, b_a = max(a_list, key=lambda x: x[0])
        
        # Pure Arb (No Tax)
        margin = (1/best_h) + (1/best_a)
        
        # Tax Adjusted Arb
        tax_margin = margin / (1 - tax_rate)
        
        if margin < 1.0:
            results.append({
                "match_id": row.get("match_id"),
                "date": row.get("date"),
                "team_1": row.get("team_1") or row.get("team1"),
                "team_2": row.get("team_2") or row.get("team2"),
                "best_h": best_h,
                "bookie_h": b_h,
                "best_a": best_a,
                "bookie_a": b_a,
                "pure_margin": margin,
                "profit_no_tax": (1/margin) - 1,
                "tax_margin": tax_margin,
                "profit_with_tax": (1/tax_margin) - 1 if tax_margin < 1 else -1
            })
            
    return pd.DataFrame(results)
