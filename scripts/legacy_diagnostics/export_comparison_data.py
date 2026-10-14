import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.getcwd())

from betting_app.core.matching import normalize_team_name
from betting_app.services.thesis_inference_service import (
    build_thesis_features_for_match, 
    _load_model, 
    _logit
)

load_dotenv()
# Use the local connection string for the remote server (it connects to the container)
engine = create_engine("postgresql+psycopg2://betting:betting_local_password@localhost:5432/betting")

def export_data():
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
        gm.team2_name as golgg_t2,
        gm.date as match_date
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
        print("No matches found.")
        return
    match_ids_tuple = tuple(match_ids)
    
    # 2. Get Bookmaker Odds (STS)
    odds_query = f"""
    WITH latest_odds AS (
        SELECT 
            be.canonical_match_id,
            oos.outcome_side,
            oos.decimal_odds,
            ROW_NUMBER() OVER (
                PARTITION BY be.canonical_match_id, oos.outcome_side 
                ORDER BY oos.scraped_at DESC
            ) as rn
        FROM odds_outcome_snapshots oos
        JOIN bookmaker_events be ON oos.bookmaker_event_id = be.bookmaker_event_id
        JOIN bookmaker_markets bm ON oos.bookmaker_market_key = bm.bookmaker_market_key
        WHERE bm.market_type = 'match_winner'
          AND be.bookmaker_id = 2 -- STS
          AND be.canonical_match_id IN {match_ids_tuple}
    )
    SELECT canonical_match_id, outcome_side, decimal_odds
    FROM latest_odds
    WHERE rn = 1
    """
    odds = pd.read_sql(odds_query, engine)
    
    bookie_probs = []
    for mid, group in odds.groupby('canonical_match_id'):
        row_a = group[group['outcome_side'] == 'a']
        row_b = group[group['outcome_side'] == 'b']
        if not row_a.empty and not row_b.empty:
            pa_raw = 1.0 / row_a.iloc[0]['decimal_odds']
            pb_raw = 1.0 / row_b.iloc[0]['decimal_odds']
            margin_sum = pa_raw + pb_raw
            bookie_probs.append({
                'canonical_match_id': mid,
                'prob_a_sts': pa_raw / margin_sum
            })
    
    bookie_df = pd.DataFrame(bookie_probs)
    matches = matches.merge(bookie_df, on='canonical_match_id', how='left')
    
    # 3. Re-predict with Thesis Model
    pipeline, calibrator = _load_model()
    
    def get_thesis_prob(row):
        try:
            feature_vec, diag = build_thesis_features_for_match(
                row['team_a_name'], row['team_b_name'],
                team_a_golgg_id=row['team_a_golgg_id'],
                team_b_golgg_id=row['team_b_golgg_id']
            )
            if feature_vec is not None:
                p_orig = pipeline.predict_proba(feature_vec)[0, 1]
                p_cal = calibrator.predict_proba(_logit(np.array([p_orig])))[0, 1]
                return p_cal
        except Exception:
            pass
        return None

    print(f"Predicting for {len(matches)} matches...")
    matches['prob_a_thesis'] = matches.apply(get_thesis_prob, axis=1)
    
    # Save to CSV
    matches.to_csv('comparison_data_remote.csv', index=False)
    print(f"Exported {len(matches)} matches to comparison_data_remote.csv")

if __name__ == "__main__":
    export_data()
