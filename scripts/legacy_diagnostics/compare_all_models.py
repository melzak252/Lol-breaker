import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics import log_loss
from betting_app.core.matching import normalize_team_name

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

def get_glicko2_prob(r1, rd1, r2, rd2):
    import math
    q = math.log(10) / 400
    g_rd = lambda rd: 1 / math.sqrt(1 + 3 * (q**2) * (rd**2) / (math.pi**2))
    E = 1 / (1 + 10**(-g_rd(math.sqrt(rd1**2 + rd2**2)) * (r1 - r2) / 400))
    return E

def compare_all():
    # 1. Get matches with GOL.GG results and Thesis predictions
    query = """
    WITH latest_thesis AS (
        SELECT cp.canonical_match_id, cp.prob_a
        FROM canonical_predictions cp
        JOIN (
            SELECT canonical_match_id, MAX(predicted_at) as max_predicted_at
            FROM canonical_predictions
            WHERE model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
            GROUP BY canonical_match_id
        ) latest ON cp.canonical_match_id = latest.canonical_match_id 
                 AND cp.predicted_at = latest.max_predicted_at
        WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
    )
    SELECT 
        cm.id as canonical_match_id,
        cm.team_a_name,
        cm.team_b_name,
        cm.start_time_normalized,
        lt.prob_a as thesis_prob,
        gm.team1_win,
        gm.team2_win,
        gm.team1_name as golgg_t1,
        gm.team2_name as golgg_t2,
        gm.team1_id as golgg_t1_id,
        gm.team2_id as golgg_t2_id
    FROM canonical_matches cm
    JOIN latest_thesis lt ON cm.id = lt.canonical_match_id
    JOIN golgg_match_mappings gmm ON cm.id = gmm.canonical_match_id
    JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
    WHERE cm.status IN ('finished', 'completed')
    """
    
    df = pd.read_sql(query, engine)
    if len(df) == 0:
        return "No matches found."

    # Resolve winner side and map canonical teams to GOL.GG IDs
    def resolve_mapping(row):
        norm_a = normalize_team_name(row['team_a_name'])
        norm_g1 = normalize_team_name(row['golgg_t1'])
        if norm_a == norm_g1:
            return row['team1_win'], row['golgg_t1_id'], row['golgg_t2_id']
        return row['team2_win'], row['golgg_t2_id'], row['golgg_t1_id']
    
    resolved = df.apply(resolve_mapping, axis=1, result_type='expand')
    df['y_true'] = resolved[0]
    df['team_a_golgg_id'] = resolved[1]
    df['team_b_golgg_id'] = resolved[2]
    
    # 2. Get Glicko-2 Ratings from entity_ratings
    ratings_query = """
    SELECT entity_id, rating_value, rating_deviation, timestamp
    FROM entity_ratings
    WHERE rating_system = 'glicko2'
    """
    ratings_df = pd.read_sql(ratings_query, engine)
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
    
    def get_glicko_prob_for_row(row):
        match_time = pd.to_datetime(row['start_time_normalized'])
        
        def get_latest_rating(golgg_id):
            if not golgg_id: return 1500, 350
            subset = ratings_df[(ratings_df['entity_id'] == str(golgg_id)) & (ratings_df['timestamp'] <= match_time)]
            if subset.empty: return 1500, 350
            latest = subset.sort_values('timestamp', ascending=False).iloc[0]
            return latest['rating_value'], latest['rating_deviation']
        
        r1, rd1 = get_latest_rating(row['team_a_golgg_id'])
        r2, rd2 = get_latest_rating(row['team_b_golgg_id'])
        return get_glicko2_prob(r1, rd1, r2, rd2)

    df['glicko2_prob'] = df.apply(get_glicko_prob_for_row, axis=1)

    # 3. Get Bookmaker Odds
    odds_query = """
    SELECT 
        be.canonical_match_id,
        b.name as bookmaker_name,
        oos.outcome_side,
        oos.decimal_odds,
        oos.scraped_at
    FROM odds_outcome_snapshots oos
    JOIN bookmaker_markets bm ON oos.bookmaker_market_key = bm.bookmaker_market_key
    JOIN bookmaker_events be ON oos.bookmaker_event_id = be.bookmaker_event_id
    JOIN bookmakers b ON be.bookmaker_id = b.id
    WHERE bm.market_type = 'match_winner'
    """
    odds_df = pd.read_sql(odds_query, engine)
    
    results = []
    for _, match in df.iterrows():
        m_id = match['canonical_match_id']
        m_time = pd.to_datetime(match['start_time_normalized'])
        m_odds = odds_df[(odds_df['canonical_match_id'] == m_id) & (pd.to_datetime(odds_df['scraped_at']) <= m_time)]
        
        bookie_probs = {}
        all_probs = []
        
        for bookie in m_odds['bookmaker_name'].unique():
            b_odds = m_odds[m_odds['bookmaker_name'] == bookie]
            latest_a = b_odds[b_odds['outcome_side'] == 'a'].sort_values('scraped_at', ascending=False)
            latest_b = b_odds[b_odds['outcome_side'] == 'b'].sort_values('scraped_at', ascending=False)
            
            if not latest_a.empty and not latest_b.empty:
                oa = latest_a.iloc[0]['decimal_odds']
                ob = latest_b.iloc[0]['decimal_odds']
                pa = (1.0/oa) / (1.0/oa + 1.0/ob)
                bookie_probs[bookie] = pa
                all_probs.append(pa)
        
        wisdom_prob = np.mean(all_probs) if all_probs else None
        
        res = {
            'canonical_match_id': m_id,
            'y_true': match['y_true'],
            'thesis_prob': match['thesis_prob'],
            'glicko2_prob': match['glicko2_prob'],
            'wisdom_prob': wisdom_prob
        }
        for b, p in bookie_probs.items():
            res[f'bookie_{b}'] = p
        results.append(res)

    res_df = pd.DataFrame(results)
    
    metrics = []
    for col in res_df.columns:
        if col in ['canonical_match_id', 'y_true']: continue
        
        valid = res_df.dropna(subset=[col, 'y_true'])
        if len(valid) < 5: continue
        
        ll = log_loss(valid['y_true'], valid[col], labels=[0, 1])
        metrics.append({
            'Model/Bookmaker': col.replace('bookie_', '').replace('_prob', '').capitalize(),
            'LogLoss': ll,
            'N': len(valid)
        })
    
    return pd.DataFrame(metrics).sort_values('LogLoss')

if __name__ == "__main__":
    print(compare_all().to_markdown(index=False))
