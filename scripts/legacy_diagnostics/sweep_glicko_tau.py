
import json
import math
import os
import sys
from collections import deque
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import connect, query_df
from src.ratings.manager import RatingManager

def run_evaluation(tau: float, all_matches, games_by_match, players_by_game):
    params = {
        "elo": {"k_player": 48, "k_team": 64},
        "gl": {"tau": tau},
        "ts": {"mu": 25.0, "sigma": 8.333, "beta": 4.16, "tau": 0.25},
    }
    manager = RatingManager(params)
    
    data_test = []
    
    for _, m in all_matches.iterrows():
        m_id = str(m['match_id'])
        t1_id = str(m['team1_id'])
        t2_id = str(m['team2_id'])
        m_date = date.fromisoformat(m['date'])
        
        match_games = games_by_match.get(m_id)
        if match_games is None or match_games.empty: continue
        match_games = match_games.sort_values('game_id')
        
        first_game_id = str(match_games.iloc[0]['game_id'])
        game_players = players_by_game.get(first_game_id)
        if game_players is None:
            p1, p2 = [], []
        else:
            p1 = game_players[game_players['team_id'].astype(str) == t1_id]['player_id'].astype(str).tolist()
            p2 = game_players[game_players['team_id'].astype(str) == t2_id]['player_id'].astype(str).tolist()
        
        manager.update_before_match(t1_id, t2_id, p1, p2, m_date)
        
        # Predict
        preds = manager.predict_match(t1_id, t2_id, p1, p2)
        
        # Actual outcome
        t1_wins = (match_games['team1_win'] == 1).sum()
        t2_wins = (match_games['team2_win'] == 1).sum()
        y_true = 1.0 if t1_wins > t2_wins else 0.0
        
        if m_date.year >= 2026:
            data_test.append({"y_true": y_true, "p_gl": preds["player_gl"]})

        # Update
        scores = []
        for _, g in match_games.iterrows():
            s1 = 1 if g['team1_win'] == 1 else 0
            s2 = 1 - s1
            scores.append(s1)
            manager.update_after_game(t1_id, t2_id, p1, p2, s1, s2)
        manager.update_after_match(t1_id, t2_id, p1, p2, scores)

    df_test = pd.DataFrame(data_test)
    ll = log_loss(df_test["y_true"], df_test["p_gl"])
    auc = roc_auc_score(df_test["y_true"], df_test["p_gl"])
    return ll, auc

def main():
    print("Loading data...")
    all_matches = query_df("""
        SELECT match_id, team1_id, team2_id, date, best_of, draw
        FROM golgg_matches
        WHERE date IS NOT NULL AND draw = 0
        ORDER BY date ASC, CAST(match_id AS INTEGER) ASC
    """)
    
    all_games = query_df("SELECT game_id, match_id, team1_win, team2_win FROM golgg_games")
    games_by_match = {str(m_id): group for m_id, group in all_games.groupby('match_id')}
    
    all_players = query_df("SELECT game_id, player_id, team_id FROM golgg_game_players")
    players_by_game = {str(g_id): group for g_id, group in all_players.groupby('game_id')}
    
    taus = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    results = []
    
    print(f"Sweeping tau over {taus}...")
    for tau in taus:
        ll, auc = run_evaluation(tau, all_matches, games_by_match, players_by_game)
        print(f"tau={tau:.1f} | LogLoss: {ll:.4f} | AUC: {auc:.4f}")
        results.append({"tau": tau, "log_loss": ll, "auc": auc})
    
    best_ll = min(results, key=lambda x: x['log_loss'])
    best_auc = max(results, key=lambda x: x['auc'])
    
    print(f"\nBest LogLoss: tau={best_ll['tau']} ({best_ll['log_loss']:.4f})")
    print(f"Best AUC: tau={best_auc['tau']} ({best_auc['auc']:.4f})")

if __name__ == "__main__":
    main()
