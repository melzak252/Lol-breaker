import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics import log_loss

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

def calculate_market_logloss():
    # Fix: outcome_side is 'a'/'b', and join on bookmaker_event_id
    query = """
    WITH latest_odds AS (
        SELECT 
            oos.bookmaker_event_id,
            oos.bookmaker_market_key,
            oos.outcome_side,
            oos.decimal_odds,
            ROW_NUMBER() OVER (
                PARTITION BY oos.bookmaker_event_id, oos.bookmaker_market_key, oos.outcome_side 
                ORDER BY oos.scraped_at DESC
            ) as rn
        FROM odds_outcome_snapshots oos
        JOIN bookmaker_markets bm ON oos.bookmaker_market_key = bm.bookmaker_market_key
        WHERE bm.market_type = 'match_winner'
    ),
    market_probs AS (
        SELECT 
            bookmaker_event_id,
            MAX(CASE WHEN outcome_side = 'a' THEN 1.0/decimal_odds END) as raw_prob_a,
            MAX(CASE WHEN outcome_side = 'b' THEN 1.0/decimal_odds END) as raw_prob_b
        FROM latest_odds
        WHERE rn = 1
        GROUP BY bookmaker_event_id
    )
    SELECT 
        cm.id as canonical_match_id,
        cm.team_a_name,
        cm.team_b_name,
        mp.raw_prob_a,
        mp.raw_prob_b,
        gm.team1_win,
        gm.team2_win,
        gm.team1_name as golgg_team1,
        gm.team2_name as golgg_team2
    FROM market_probs mp
    JOIN bookmaker_events be ON mp.bookmaker_event_id = be.bookmaker_event_id
    JOIN canonical_matches cm ON be.canonical_match_id = cm.id
    JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
    JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
    WHERE mp.raw_prob_a IS NOT NULL AND mp.raw_prob_b IS NOT NULL
      AND cm.status IN ('finished', 'completed')
    """
    
    df = pd.read_sql(query, engine)
    if len(df) == 0:
        return "No market data found for finished matches."
    
    # Normalize probabilities (remove margin)
    df['sum_inv_odds'] = df['raw_prob_a'] + df['raw_prob_b']
    df['prob_a'] = df['raw_prob_a'] / df['sum_inv_odds']
    df['prob_b'] = df['raw_prob_b'] / df['sum_inv_odds']
    
    # Resolve winner for team_a
    from betting_app.core.matching import normalize_team_name
    
    def get_team_a_win(row):
        norm_a = normalize_team_name(row['team_a_name'])
        norm_g1 = normalize_team_name(row['golgg_team1'])
        norm_g2 = normalize_team_name(row['golgg_team2'])
        
        if norm_a == norm_g1: return row['team1_win']
        if norm_a == norm_g2: return row['team2_win']
        
        # Substring match fallback
        if norm_a in norm_g1 or norm_g1 in norm_a: return row['team1_win']
        if norm_a in norm_g2 or norm_g2 in norm_a: return row['team2_win']
        
        return None

    df['team_a_win'] = df.apply(get_team_a_win, axis=1)
    df_valid = df.dropna(subset=['team_a_win']).copy()
    
    if len(df_valid) == 0:
        return "Could not resolve winners for market data."
        
    ll = log_loss(df_valid['team_a_win'], df_valid['prob_a'], labels=[0, 1])
    
    # Also calculate for our model on the SAME matches for comparison
    match_ids = tuple(df_valid['canonical_match_id'].unique().tolist())
    if len(match_ids) == 1:
        match_ids_str = f"({match_ids[0]})"
    else:
        match_ids_str = str(match_ids)
        
    model_query = f"""
    SELECT canonical_match_id, prob_a
    FROM canonical_predictions
    WHERE model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
      AND canonical_match_id IN {match_ids_str}
    """
    model_df = pd.read_sql(model_query, engine)
    
    # Merge and compare
    merged = df_valid.merge(model_df, on='canonical_match_id', suffixes=('_bookie', '_model'))
    
    if len(merged) > 0:
        ll_model = log_loss(merged['team_a_win'], merged['prob_a_model'], labels=[0, 1])
        return (f"Comparison on N={len(merged)} matches:\n"
                f"Bookmaker LogLoss: {ll:.4f}\n"
                f"Thesis Model LogLoss: {ll_model:.4f}")
    
    return f"Bookmaker LogLoss (N={len(df_valid)}): {ll:.4f} (No model predictions for these matches)"

if __name__ == "__main__":
    print(calculate_market_logloss())
