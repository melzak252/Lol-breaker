
import json
import math
import os
import sys
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

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

CONTEXT_WINDOW = 20

OPTUNA_BASE_FEATURES = [
    "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
    "player_elo_min1", "player_elo_min2",
    "player_gl_max1", "player_gl_max2",
    "player_gl_rd_avg1", "player_gl_rd_avg2",
    "player_ts_sigma_avg1", "player_ts_sigma_avg2",
    "player_os_sigma_avg1", "player_os_sigma_avg2",
    "player_pl_sigma_avg1", "player_pl_sigma_avg2",
    "player_tm_sigma_avg1", "player_tm_sigma_avg2",
]

ROLLING_FULL_FEATURES = [
    "t1_rolling_win_rate", "t2_rolling_win_rate",
    "t1_rolling_kills", "t2_rolling_kills",
    "t1_rolling_deaths", "t2_rolling_deaths",
    "t1_rolling_gd15", "t2_rolling_gd15",
    "t1_rolling_dpm", "t2_rolling_dpm",
    "t1_rolling_vspm", "t2_rolling_vspm",
    "t1_rolling_towers", "t2_rolling_towers",
    "t1_rolling_nashors", "t2_rolling_nashors",
    "t1_rolling_gold", "t2_rolling_gold",
    "t1_rolling_duration", "t2_rolling_duration",
]

_DEFAULT_TEAM_STATS = {
    "win_rate": 0.5, "kills": 12.0, "deaths": 12.0, "gd15": 0.0, "dpm": 1800.0,
    "vspm": 7.0, "towers": 5.0, "nashors": 0.5, "gold": 55000.0, "duration": 1800.0,
}

def _average_history(history: deque | None) -> dict[str, float]:
    if not history: return dict(_DEFAULT_TEAM_STATS)
    rows = list(history)
    return {k: float(np.mean([r[k] for r in rows])) for k in _DEFAULT_TEAM_STATS.keys()}

def _update_w20_history(team_history: dict[str, deque], team_id: str, game: dict, window_size: int) -> None:
    is_team_1 = str(game.get("t1_id")) == str(team_id)
    win = bool(game.get("t1_win")) if is_team_1 else bool(game.get("t2_win"))
    p_stats = game.get("t1_player_stats" if is_team_1 else "t2_player_stats", [])
    
    def _f(val):
        if val is None: return 0.0
        try: return float(val)
        except: return 0.0

    game_stats = {
        "win": float(win),
        "kills": sum(_f(p.get("kills")) for p in p_stats),
        "deaths": sum(_f(p.get("deaths")) for p in p_stats),
        "dpm": sum(_f(p.get("dpm")) for p in p_stats),
        "vspm": sum(_f(p.get("vspm")) for p in p_stats),
        "gd15": sum(_f(p.get("gd@15")) for p in p_stats),
        "towers": _f(game.get("t1_towers" if is_team_1 else "t2_towers")),
        "nashors": _f(game.get("t1_nashors" if is_team_1 else "t2_nashors")),
        "gold": _f(game.get("t1_gold" if is_team_1 else "t2_gold")),
        "duration": _f(game.get("game_duration")),
    }
    if team_id not in team_history: team_history[team_id] = deque(maxlen=window_size)
    team_history[team_id].append(game_stats)

def parse_stats(sj):
    if not sj: return {}
    try: return json.loads(sj)
    except: return {}

def main():
    manager = RatingManager(RATING_SYSTEM_PARAMS)
    team_history = {}
    
    print("Loading reference data...")
    df_preds_ref = pd.read_csv("data/golgg_y_predicts.csv")
    df_rolling_ref = pd.read_csv("data/golgg_rolling_stats.csv")
    df_preds_ref['golgg_match_id'] = df_preds_ref['golgg_match_id'].astype(str)
    df_rolling_ref['golgg_match_id'] = df_rolling_ref['golgg_match_id'].astype(str)
    
    # Pick some matches from 2025 to compare
    target_matches = df_preds_ref[df_preds_ref['date'].str.startswith('2025')]['golgg_match_id'].unique()[:1]
    print(f"Target match for comparison: {target_matches}")

    print("Loading matches from DB...")
    # Only load matches up to the target match to speed up
    target_date = df_preds_ref[df_preds_ref['golgg_match_id'] == target_matches[0]]['date'].iloc[0]
    all_matches = query_df(f"""
        SELECT match_id, team1_id, team2_id, team1_name, team2_name, date, best_of, draw
        FROM golgg_matches
        WHERE date IS NOT NULL AND draw = 0 AND date <= '{target_date}'
        ORDER BY date ASC, CAST(match_id AS INTEGER) ASC
    """)
    
    print("Loading games from DB...")
    all_games = query_df("""
        SELECT game_id, match_id, team1_win, team2_win, game_duration,
               team1_stats_json, team2_stats_json
        FROM golgg_games
    """)
    games_by_match = {str(m_id): group for m_id, group in all_games.groupby('match_id')}
    
    print("Loading players from DB...")
    all_players = query_df("""
        SELECT game_id, player_id, team_id, stats_json
        FROM golgg_game_players
    """)
    players_by_game = {str(g_id): group for g_id, group in all_players.groupby('game_id')}
    
    comparison_results = []
    
    print(f"Processing matches...")
    for _, m in tqdm(all_matches.iterrows(), total=len(all_matches)):
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
        
        if m_id in target_matches:
            manager.update_before_match(t1_id, t2_id, p1, p2, m_date)
            preds = manager.predict_match(t1_id, t2_id, p1, p2)
            
            t1_stats = _average_history(team_history.get(t1_id))
            t2_stats = _average_history(team_history.get(t2_id))
            
            # Compare with reference
            ref_pred = df_preds_ref[df_preds_ref['golgg_match_id'] == m_id].iloc[0]
            ref_roll = df_rolling_ref[df_rolling_ref['golgg_match_id'] == m_id].iloc[0]
            
            diffs = {}
            for f in OPTUNA_BASE_FEATURES:
                diffs[f] = preds.get(f, 0.0) - ref_pred.get(f, 0.0)
            
            for k in _DEFAULT_TEAM_STATS.keys():
                diffs[f"t1_rolling_{k}"] = t1_stats[k] - ref_roll.get(f"t1_rolling_{k}", 0.0)
                diffs[f"t2_rolling_{k}"] = t2_stats[k] - ref_roll.get(f"t2_rolling_{k}", 0.0)
            
            comparison_results.append({
                "match_id": m_id,
                "diffs": diffs
            })
            
            if len(comparison_results) == len(target_matches):
                break
        else:
            manager.update_before_match(t1_id, t2_id, p1, p2, m_date)

        # Update
        scores = []
        for _, g in match_games.iterrows():
            g_id = str(g['game_id'])
            s1 = 1 if g['team1_win'] == 1 else 0
            s2 = 1 - s1
            scores.append(s1)
            
            gp_df = players_by_game.get(g_id)
            t1_stats_json = parse_stats(g['team1_stats_json'])
            t2_stats_json = parse_stats(g['team2_stats_json'])
            
            g_data = {
                "t1_id": t1_id, "t2_id": t2_id, "t1_win": s1 == 1, "t2_win": s2 == 1,
                "game_duration": g['game_duration'],
                "t1_towers": t1_stats_json.get('towers', 0), "t2_towers": t2_stats_json.get('towers', 0),
                "t1_nashors": t1_stats_json.get('nashors', 0), "t2_nashors": t2_stats_json.get('nashors', 0),
                "t1_gold": t1_stats_json.get('gold', 0), "t2_gold": t2_stats_json.get('gold', 0),
                "t1_player_stats": [parse_stats(sj) for sj in gp_df[gp_df['team_id'].astype(str) == t1_id]['stats_json']] if gp_df is not None else [],
                "t2_player_stats": [parse_stats(sj) for sj in gp_df[gp_df['team_id'].astype(str) == t2_id]['stats_json']] if gp_df is not None else [],
            }
            manager.update_after_game(t1_id, t2_id, p1, p2, s1, s2)
            _update_w20_history(team_history, t1_id, g_data, CONTEXT_WINDOW)
            _update_w20_history(team_history, t2_id, g_data, CONTEXT_WINDOW)
            
        manager.update_after_match(t1_id, t2_id, p1, p2, scores)

    for res in comparison_results:
        print(f"\nMatch {res['match_id']} differences:")
        sorted_diffs = sorted(res['diffs'].items(), key=lambda x: abs(x[1]), reverse=True)
        for f, d in sorted_diffs[:10]:
            if abs(d) > 1e-5:
                print(f"  {f:25}: {d:.6f}")
            else:
                break

if __name__ == "__main__":
    main()
