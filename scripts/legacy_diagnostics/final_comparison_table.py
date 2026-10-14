import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics import log_loss

# Add project root to path for imports
sys.path.append(os.getcwd())

from betting_app.core.matching import normalize_team_name
from betting_app.services.thesis_inference_service import (
    build_thesis_features_for_match, 
    _load_model, 
    _logit
)

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

def calculate_final_comparison():
    # 1. Get finished matches with GOL.GG results
    query = """
    SELECT 
        cm.id as canonical_match_id,
        cm.team_a_name,
        cm.team_b_name,
        gm.team1_id as team_a_golgg_id,
        gm.team2_id as team_b_golgg_id,
        cm.start_time_normalized,
        gm.team1_win,
        gm.team2_win,
        gm.team1_name as golgg_t1,
        gm.team2_name as golgg_t2
    FROM canonical_matches cm
    JOIN golgg_match_mappings gmm ON cm.id = gmm.canonical_match_id
    JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
    WHERE cm.status IN ('finished', 'completed')
    """
    matches = pd.read_sql(query, engine)
    
    def resolve_team_a_win(row):
        na = normalize_team_name(row['team_a_name'])
        ng1 = normalize_team_name(row['golgg_t1'])
        ng2 = normalize_team_name(row['golgg_t2'])
        if na == ng1 or na in ng1 or ng1 in na: return row['team1_win']
        if na == ng2 or na in ng2 or ng2 in na: return row['team2_win']
        return None

    matches['team_a_win'] = matches.apply(resolve_team_a_win, axis=1)
    matches = matches.dropna(subset=['team_a_win'])
    
    match_ids = matches['canonical_match_id'].unique().tolist()
    if not match_ids:
        return "No matches found."
    match_ids_tuple = tuple(match_ids)
    
    # 2. Get Bookmaker Odds
    odds_query = f"""
    WITH latest_odds AS (
        SELECT 
            be.canonical_match_id,
            be.bookmaker_id,
            b.name as bookmaker_name,
            oos.outcome_side,
            oos.decimal_odds,
            ROW_NUMBER() OVER (
                PARTITION BY be.canonical_match_id, be.bookmaker_id, oos.outcome_side 
                ORDER BY oos.scraped_at DESC
            ) as rn
        FROM odds_outcome_snapshots oos
        JOIN bookmaker_events be ON oos.bookmaker_event_id = be.bookmaker_event_id
        JOIN bookmakers b ON be.bookmaker_id = b.id
        JOIN bookmaker_markets bm ON oos.bookmaker_market_key = bm.bookmaker_market_key
        WHERE bm.market_type = 'match_winner'
          AND be.canonical_match_id IN {match_ids_tuple}
    )
    SELECT canonical_match_id, bookmaker_name, outcome_side, decimal_odds
    FROM latest_odds
    WHERE rn = 1
    """
    odds = pd.read_sql(odds_query, engine)
    
    bookie_probs = []
    for (mid, bname), group in odds.groupby(['canonical_match_id', 'bookmaker_name']):
        row_a = group[group['outcome_side'] == 'a']
        row_b = group[group['outcome_side'] == 'b']
        if not row_a.empty and not row_b.empty:
            pa_raw = 1.0 / row_a.iloc[0]['decimal_odds']
            pb_raw = 1.0 / row_b.iloc[0]['decimal_odds']
            margin_sum = pa_raw + pb_raw
            bookie_probs.append({
                'canonical_match_id': mid,
                'bookmaker': bname,
                'prob_a': pa_raw / margin_sum
            })
    
    bookie_df = pd.DataFrame(bookie_probs)
    
    # 3. Get Glicko-2 and Elo Ratings (Latest)
    ratings_query = """
    SELECT normalized_entity_name, rating_system, rating_value
    FROM entity_ratings
    WHERE snapshot_at = (SELECT MAX(snapshot_at) FROM entity_ratings)
    """
    ratings_df = pd.read_sql(ratings_query, engine)
    
    def get_rating_prob(row, system):
        na = normalize_team_name(row['team_a_name'])
        nb = normalize_team_name(row['team_b_name'])
        r_a = ratings_df[(ratings_df['normalized_entity_name'] == na) & (ratings_df['rating_system'] == system)]
        r_b = ratings_df[(ratings_df['normalized_entity_name'] == nb) & (ratings_df['rating_system'] == system)]
        if not r_a.empty and not r_b.empty:
            val_a = r_a.iloc[0]['rating_value']
            val_b = r_b.iloc[0]['rating_value']
            return 1.0 / (1.0 + pow(10, -(val_a - val_b) / 400.0))
        return None

    matches['prob_a_glicko2'] = matches.apply(lambda r: get_rating_prob(r, 'glicko2'), axis=1)
    matches['prob_a_elo'] = matches.apply(lambda r: get_rating_prob(r, 'elo'), axis=1)
    
    # 4. Re-predict with Thesis Model (Corrected)
    pipeline, calibrator = _load_model()
    
    def get_thesis_prob(row):
        try:
            # Note: build_thesis_features_for_match handles team alignment internally if IDs are provided
            feature_vec, diag = build_thesis_features_for_match(
                row['team_a_name'], row['team_b_name'],
                team_a_golgg_id=row['team_a_golgg_id'],
                team_b_golgg_id=row['team_b_golgg_id']
            )
            if feature_vec is not None:
                # Original prob
                p_orig = pipeline.predict_proba(feature_vec)[0, 1]
                # Platt calibration
                p_cal = calibrator.predict_proba(_logit(np.array([p_orig])))[0, 1]
                return p_cal
        except Exception:
            pass
        return None

    # Filter to matches that have bookmaker data for a fair comparison
    matches_with_bookie = matches[matches['canonical_match_id'].isin(bookie_df['canonical_match_id'].unique())].copy()
    print(f"Re-predicting for {len(matches_with_bookie)} matches with bookmaker data...")
    matches_with_bookie['prob_a_thesis'] = matches_with_bookie.apply(get_thesis_prob, axis=1)
    
    results = []
    
    # Thesis Model (Corrected)
    df_thesis = matches_with_bookie.dropna(subset=['prob_a_thesis'])
    if not df_thesis.empty:
        ll = log_loss(df_thesis['team_a_win'], df_thesis['prob_a_thesis'], labels=[0,1])
        results.append({'Model': 'Thesis Model (Corrected)', 'LogLoss': ll, 'N': len(df_thesis)})
        
    # Glicko-2
    df_glicko = matches_with_bookie.dropna(subset=['prob_a_glicko2'])
    if not df_glicko.empty:
        ll_g = log_loss(df_glicko['team_a_win'], df_glicko['prob_a_glicko2'], labels=[0,1])
        results.append({'Model': 'Glicko-2 (Base)', 'LogLoss': ll_g, 'N': len(df_glicko)})

    # Elo
    df_elo = matches_with_bookie.dropna(subset=['prob_a_elo'])
    if not df_elo.empty:
        ll_e = log_loss(df_elo['team_a_win'], df_elo['prob_a_elo'], labels=[0,1])
        results.append({'Model': 'Elo (Base)', 'LogLoss': ll_e, 'N': len(df_elo)})

    # Bookmakers
    if not bookie_df.empty:
        # Wisdom of the Crowd
        wotc = bookie_df.groupby('canonical_match_id')['prob_a'].mean().reset_index()
        merged_wotc = matches_with_bookie.merge(wotc, on='canonical_match_id')
        ll_wotc = log_loss(merged_wotc['team_a_win'], merged_wotc['prob_a'], labels=[0,1])
        results.append({'Model': 'Wisdom of the Crowd', 'LogLoss': ll_wotc, 'N': len(merged_wotc)})
        
        for bname, bgroup in bookie_df.groupby('bookmaker'):
            merged_b = matches_with_bookie.merge(bgroup, on='canonical_match_id')
            if len(merged_b) > 5:
                ll_b = log_loss(merged_b['team_a_win'], merged_b['prob_a'], labels=[0,1])
                results.append({'Model': f'Bookmaker: {bname}', 'LogLoss': ll_b, 'N': len(merged_b)})

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = calculate_final_comparison()
    if isinstance(df, str):
        print(df)
    else:
        print("\n### Final Model Comparison (May-June 2026) ###")
        print(df.sort_values('LogLoss').to_markdown(index=False))
