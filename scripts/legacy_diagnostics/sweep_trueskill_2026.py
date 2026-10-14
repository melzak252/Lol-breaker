import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from src.ratings.trueskill_rating import TrueSkillRating
import json
from tqdm import tqdm
import multiprocessing

def evaluate_params(args):
    df_matches, df_games, beta, tau = args
    ts = TrueSkillRating(beta=beta, tau=tau)
    
    results = []
    
    # Map games to matches
    games_by_match = df_games.groupby('match_id')
    
    for _, match in df_matches.iterrows():
        match_id = match['match_id']
        if match_id not in games_by_match.groups:
            continue
            
        match_games = games_by_match.get_group(match_id)
        
        # Get players for the first game to predict match
        first_game = match_games.iloc[0]
        t1_players = first_game['t1_players_list']
        t2_players = first_game['t2_players_list']
        
        p1 = t1_players if t1_players else [f"dummy_{match['team1_id']}"]
        p2 = t2_players if t2_players else [f"dummy_{match['team2_id']}"]
        
        if match['is_2026']:
            # Predict match outcome BEFORE any games are processed
            prob = ts.predict_player_win_prob(p1, p2)
            results.append({
                'match_id': match_id,
                'prob': prob,
                'y_true': 1.0 if match['team1_win'] else 0.0
            })
            
        # Update ratings for each game in the match
        for _, game in match_games.iterrows():
            gp1 = game['t1_players_list'] if game['t1_players_list'] else [f"dummy_{match['team1_id']}"]
            gp2 = game['t2_players_list'] if game['t2_players_list'] else [f"dummy_{match['team2_id']}"]
            
            ts.update_player(gp1, gp2, 1 if game['team1_win'] else 0, 0 if game['team1_win'] else 1)

    eval_df = pd.DataFrame(results)
    if len(eval_df) == 0:
        return beta, tau, None
        
    ll = log_loss(eval_df['y_true'], eval_df['prob'])
    auc = roc_auc_score(eval_df['y_true'], eval_df['prob'])
    acc = accuracy_score(eval_df['y_true'], eval_df['prob'] > 0.5)
    
    return beta, tau, (ll, auc, acc)

if __name__ == "__main__":
    conn = sqlite3.connect("data/betting_app.sqlite3")
    
    print("Loading matches...")
    df_matches = pd.read_sql_query("""
        SELECT match_id, date as begin_at, team1_id, team2_id, team1_win
        FROM golgg_matches 
        WHERE date IS NOT NULL
        ORDER BY date ASC, match_id ASC
    """, conn)
    
    print("Loading games...")
    df_games = pd.read_sql_query("""
        SELECT 
            g.match_id, g.game_id, g.team1_win,
            (SELECT json_group_array(player_id) FROM golgg_game_players WHERE game_id = g.game_id AND team_id = m.team1_id) as t1_players,
            (SELECT json_group_array(player_id) FROM golgg_game_players WHERE game_id = g.game_id AND team_id = m.team2_id) as t2_players
        FROM golgg_games g
        JOIN golgg_matches m ON g.match_id = m.match_id
        ORDER BY m.date ASC, g.game_id ASC
    """, conn)
    conn.close()

    print("Pre-processing data...")
    df_games['t1_players_list'] = df_games['t1_players'].apply(json.loads)
    df_games['t2_players_list'] = df_games['t2_players'].apply(json.loads)
    df_matches['is_2026'] = df_matches['begin_at'].str.startswith('2026')
    
    # Grid search parameters
    betas = [4.167, 8.333, 12.5, 16.667, 25.0/2.0]
    taus = [0.01, 0.0833, 0.166, 0.25, 0.5]
    
    tasks = []
    for b in betas:
        for t in taus:
            tasks.append((df_matches, df_games, b, t))
            
    print(f"Starting sweep with {len(tasks)} combinations...")
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_params, tasks), total=len(tasks)))
        
    print(f"\n{'Beta':>6} | {'Tau':>8} | {'LogLoss':>8} | {'AUC':>8} | {'Acc':>8}")
    print("-" * 50)
    
    best_ll = 2.0
    best_params = None
    
    for b, t, res in results:
        if res:
            ll, auc, acc = res
            print(f"{b:6.3f} | {t:8.4f} | {ll:8.4f} | {auc:8.4f} | {acc:8.4f}")
            if ll < best_ll:
                best_ll = ll
                best_params = (b, t)
                    
    print("-" * 50)
    if best_params:
        print(f"Best LogLoss: {best_ll:.4f} with Beta={best_params[0]}, Tau={best_params[1]}")
