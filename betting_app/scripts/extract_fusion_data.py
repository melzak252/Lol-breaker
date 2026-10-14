import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import get_session
from sqlalchemy import text

def fetch_data():
    """Fetch games, player stats and ratings."""
    print("Fetching games and player stats from DB...")
    query_games = """
    SELECT 
        g.game_id, g.match_id, g.date, g.team1_id, g.team2_id, 
        g.team1_win, g.team1_side, g.game_duration,
        g.team1_stats_json, g.team2_stats_json
    FROM golgg_games g
    ORDER BY g.date ASC, g.game_id ASC
    """
    
    query_player_stats = """
    SELECT 
        game_id, team_id, 
        COALESCE(SUM((stats_json::jsonb->>'gd@15')::float), 0) as team_gd15,
        COALESCE(SUM((stats_json::jsonb->>'csd@15')::float), 0) as team_csd15,
        COALESCE(SUM((stats_json::jsonb->>'xpd@15')::float), 0) as team_xpd15,
        COALESCE(SUM((stats_json::jsonb->>'dpm')::float), 0) as team_dpm,
        COALESCE(SUM((stats_json::jsonb->>'vspm')::float), 0) as team_vspm
    FROM golgg_game_players
    GROUP BY game_id, team_id
    """
    
    with get_session() as session:
        df_games = pd.read_sql(text(query_games), session.connection())
        df_player_stats = pd.read_sql(text(query_player_stats), session.connection())
    
    # Fill any remaining NaNs just in case
    df_player_stats = df_player_stats.fillna(0)
    
    print("Loading ratings from CSV...")
    df_ratings = pd.read_csv(PROJECT_ROOT / "data" / "golgg_y_predicts.csv")
    df_ratings['golgg_match_id'] = df_ratings['golgg_match_id'].astype(str)
    
    return df_games, df_player_stats, df_ratings

def build_fusion_dataset(df_games, df_player_stats, df_ratings, window_size=15):
    # Index player stats for fast lookup
    pstats = df_player_stats.set_index(['game_id', 'team_id']).to_dict('index')
    
    # Index ratings by match_id
    # We'll use player_elo as a proxy for team strength in history
    ratings_map = df_ratings.set_index('golgg_match_id').to_dict('index')
    
    team_history = {} # team_id -> deque
    team_last_date = {} # team_id -> last game date
    
    dataset = []
    
    # Features in sequence (16 features):
    # [win, side, duration, kills, deaths, gold, towers, dragons, nashors, 
    #  gd15, csd15, xpd15, dpm, vspm, days_rest, opp_elo]
    
    for _, game in tqdm(df_games.iterrows(), total=len(df_games), desc="Building sequences"):
        gid = game['game_id']
        mid = str(game['match_id'])
        t1_id = game['team1_id']
        t2_id = game['team2_id']
        date = pd.to_datetime(game['date'])
        
        if mid not in ratings_map:
            continue
            
        match_ratings = ratings_map[mid]
        
        # Get history BEFORE this game
        t1_seq = list(team_history.get(t1_id, []))
        t2_seq = list(team_history.get(t2_id, []))
        
        def pad_sequence(seq, size):
            if len(seq) >= size: return seq[-size:]
            return [[0.0] * 16] * (size - len(seq)) + seq

        t1_padded = pad_sequence(t1_seq, window_size)
        t2_padded = pad_sequence(t2_seq, window_size)
        
        # Static features (the 46 features from exp-039)
        # We'll take them from the CSV. 
        # Note: In the CSV, player_elo is for team1.
        static_cols = [
            "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
            "player_elo_min1", "player_elo_min2", "player_gl_max1", "player_gl_max2",
            "player_gl_rd_avg1", "player_gl_rd_avg2", "player_ts_sigma_avg1", "player_ts_sigma_avg2",
            "player_os_sigma_avg1", "player_os_sigma_avg2", "player_pl_sigma_avg1", "player_pl_sigma_avg2",
            "player_tm_sigma_avg1", "player_tm_sigma_avg2"
        ]
        # Adding rolling features would be good too, but let's start with these core ones
        static_feats = [float(match_ratings.get(c, 0.5)) for c in static_cols]
        
        dataset.append({
            'game_id': int(gid),
            't1_seq': t1_padded,
            't2_seq': t2_padded,
            'static_feats': static_feats,
            'y': 1 if game['team1_win'] == 1 else 0,
            'date': date.isoformat()
        })
        
        # Update history AFTER this game
        s1_json = json.loads(game['team1_stats_json'] or '{}')
        s2_json = json.loads(game['team2_stats_json'] or '{}')
        
        def get_pstats(gid, tid):
            res = pstats.get((gid, tid), {})
            return [
                float(res.get('team_gd15') or 0) / 2000.0,
                float(res.get('team_csd15') or 0) / 50.0,
                float(res.get('team_xpd15') or 0) / 2000.0,
                float(res.get('team_dpm') or 0) / 10000.0,
                float(res.get('team_vspm') or 0) / 20.0
            ]

        def get_days_rest(tid, current_date):
            if tid not in team_last_date: return 1.0 # Default 30 days (normalized)
            diff = (current_date - team_last_date[tid]).days
            return min(float(diff) / 30.0, 1.0)

        # Team 1 update
        f1 = [
            float(game['team1_win'] or 0),
            1.0 if game['team1_side'] == 'Blue' else 0.0,
            float(game['game_duration'] or 1800) / 1800.0,
            float(s1_json.get('kills') or 0) / 20.0,
            float(s2_json.get('kills') or 0) / 20.0,
            float(s1_json.get('gold') or 50000) / 60000.0,
            float(s1_json.get('towers') or 0) / 11.0,
            float(s1_json.get('dragons') or 0) / 4.0,
            float(s1_json.get('nashors') or 0) / 2.0,
        ] + get_pstats(gid, t1_id) + [get_days_rest(t1_id, date), float(match_ratings.get('player_elo_min2', 0.5))]
        
        if t1_id not in team_history: team_history[t1_id] = deque(maxlen=window_size)
        team_history[t1_id].append(f1)
        team_last_date[t1_id] = date

        # Team 2 update
        f2 = [
            1.0 - float(game['team1_win'] or 0),
            0.0 if game['team1_side'] == 'Blue' else 1.0,
            float(game['game_duration'] or 1800) / 1800.0,
            float(s2_json.get('kills') or 0) / 20.0,
            float(s1_json.get('kills') or 0) / 20.0,
            float(s2_json.get('gold') or 50000) / 60000.0,
            float(s2_json.get('towers') or 0) / 11.0,
            float(s2_json.get('dragons') or 0) / 4.0,
            float(s2_json.get('nashors') or 0) / 2.0,
        ] + get_pstats(gid, t2_id) + [get_days_rest(t2_id, date), float(match_ratings.get('player_elo_min1', 0.5))]
        
        if t2_id not in team_history: team_history[t2_id] = deque(maxlen=window_size)
        team_history[t2_id].append(f2)
        team_last_date[t2_id] = date

    return dataset

if __name__ == "__main__":
    df_games, df_player_stats, df_ratings = fetch_data()
    dataset = build_fusion_dataset(df_games, df_player_stats, df_ratings)
    
    output_path = PROJECT_ROOT / "data" / "fusion_dataset_v1.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f)
    print(f"Saved {len(dataset)} games to {output_path}")
