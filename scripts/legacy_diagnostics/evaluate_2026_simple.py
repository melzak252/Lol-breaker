
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import connect, query_df
from src.ratings.manager import RatingManager

# --- Configuration ---
RATING_SYSTEM_PARAMS = {
    "elo": {"k_player": 48, "k_team": 64},
    "ts": {"mu": 25.0, "sigma": 8.333, "beta": 4.16, "tau": 0.25},
    "os": {"mu": 25.0, "sigma": 3.5},
    "pl": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
    "tm": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
}

def parse_stats(sj):
    if not sj: return {}
    try: return json.loads(sj)
    except: return {}

def main():
    manager = RatingManager(RATING_SYSTEM_PARAMS)
    
    print("Loading matches from DB...")
    all_matches = query_df("""
        SELECT match_id, team1_id, team2_id, date, best_of, draw
        FROM golgg_matches
        WHERE date IS NOT NULL AND draw = 0
        ORDER BY date ASC, CAST(match_id AS INTEGER) ASC
    """)
    
    print("Loading games from DB...")
    all_games = query_df("""
        SELECT game_id, match_id, team1_win, team2_win
        FROM golgg_games
    """)
    games_by_match = {str(m_id): group for m_id, group in all_games.groupby('match_id')}
    
    print("Loading players from DB...")
    all_players = query_df("""
        SELECT game_id, player_id, team_id
        FROM golgg_game_players
    """)
    players_by_game = {str(g_id): group for g_id, group in all_players.groupby('game_id')}
    
    data_train = []
    data_test = []
    
    print(f"Processing {len(all_matches)} matches...")
    for _, m in tqdm(all_matches.iterrows(), total=len(all_matches)):
        m_id = str(m['match_id'])
        t1_id = str(m['team1_id'])
        t2_id = str(m['team2_id'])
        m_date = date.fromisoformat(m['date'])
        
        match_games = games_by_match.get(m_id)
        if match_games is None or match_games.empty: continue
        match_games = match_games.sort_values('game_id')
        
        # Get players from first game
        first_game_id = str(match_games.iloc[0]['game_id'])
        game_players = players_by_game.get(first_game_id)
        if game_players is None:
            p1, p2 = [], []
        else:
            p1 = game_players[game_players['team_id'].astype(str) == t1_id]['player_id'].astype(str).tolist()
            p2 = game_players[game_players['team_id'].astype(str) == t2_id]['player_id'].astype(str).tolist()
        
        # Update ratings before prediction
        manager.update_before_match(t1_id, t2_id, p1, p2, m_date)
        
        # Predict
        preds = manager.predict_match(t1_id, t2_id, p1, p2)
        
        # Actual outcome
        t1_wins = (match_games['team1_win'] == 1).sum()
        t2_wins = (match_games['team2_win'] == 1).sum()
        y_true = 1.0 if t1_wins > t2_wins else 0.0
        
        row = {
            "y_true": y_true,
            "p_elo": preds["player_elo"],
            "p_gl": preds["player_gl"],
            "p_ts": preds["player_ts"],
        }
        
        if m_date.year < 2026:
            data_train.append(row)
        else:
            data_test.append(row)

        # Update ratings after match
        scores = []
        for _, g in match_games.iterrows():
            s1 = 1 if g['team1_win'] == 1 else 0
            s2 = 1 - s1
            scores.append(s1)
            manager.update_after_game(t1_id, t2_id, p1, p2, s1, s2)
        manager.update_after_match(t1_id, t2_id, p1, p2, scores)

    if not data_test:
        print("No matches found for 2026.")
        return

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)
    
    features = ["p_elo", "p_gl", "p_ts"]
    X_train = df_train[features]
    y_train = df_train["y_true"]
    X_test = df_test[features]
    y_test = df_test["y_true"]
    
    # 1. Simple Average
    df_test["p_avg"] = df_test[features].mean(axis=1)
    
    # 2. Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    df_test["p_lr"] = lr.predict_proba(X_test)[:, 1]
    
    print(f"\nEvaluation for 2026 (N={len(df_test)} matches):")
    for col in ["p_elo", "p_gl", "p_ts", "p_avg", "p_lr"]:
        ll = log_loss(y_test, df_test[col])
        auc = roc_auc_score(y_test, df_test[col])
        acc = ((df_test[col] >= 0.5) == y_test).mean()
        print(f"{col:12} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Acc: {acc:.2%}")
    
    print("\nLR Coefficients:")
    for f, c in zip(features, lr.coef_[0]):
        print(f"{f}: {c:.4f}")
    print(f"Intercept: {lr.intercept_[0]:.4f}")

if __name__ == "__main__":
    main()
