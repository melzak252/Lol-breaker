
import json
import math
import os
import sys
from collections import deque
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import connect, query_df
from betting_app.core.matching import normalize_team_name
from src.ratings.manager import RatingManager

# --- Configuration ---
PIPELINE_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"
CALIBRATOR_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib"

RATING_SYSTEM_PARAMS = {
    "elo": {"k_player": 48, "k_team": 64},
    "ts": {"mu": 25.0, "sigma": 8.333, "beta": 4.16, "tau": 0.25},
    "os": {"mu": 25.0, "sigma": 3.5},
    "pl": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
    "tm": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
}

CONTEXT_WINDOW = 20
EPSILON = 0.001

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

RANK_PROB_FEATURES = ["player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm"]
ALL_FEATURES = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + [f + "_binom_series" for f in RANK_PROB_FEATURES]

_DEFAULT_TEAM_STATS = {
    "win_rate": 0.5, "kills": 12.0, "deaths": 12.0, "gd15": 0.0, "dpm": 1800.0,
    "vspm": 7.0, "towers": 5.0, "nashors": 0.5, "gold": 55000.0, "duration": 1800.0,
}

# --- Helpers ---
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
        "win_rate": float(win),
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

def series_probability(prob: float, best_of: int) -> float:
    prob = max(0.001, min(0.999, prob))
    if best_of <= 1: return prob
    needed = best_of // 2 + 1
    series_prob = 0.0
    for wins in range(needed, best_of + 1):
        series_prob += math.comb(best_of, wins) * (prob**wins) * ((1.0 - prob)**(best_of - wins))
    return max(0.001, min(0.999, series_prob))

def _logit(p: float) -> float:
    p = max(EPSILON, min(1.0 - EPSILON, p))
    return math.log(p / (1.0 - p))

def parse_stats(sj):
    if not sj: return {}
    try: return json.loads(sj)
    except: return {}

# --- Main ---
def main():
    print("Loading models...")
    pipeline = joblib.load(PIPELINE_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    manager = RatingManager(RATING_SYSTEM_PARAMS)
    team_history = {}
    
    print("Loading matches from DB...")
    all_matches = query_df("""
        SELECT match_id, team1_id, team2_id, team1_name, team2_name, date, best_of, draw
        FROM golgg_matches
        WHERE date IS NOT NULL AND draw = 0
        ORDER BY date ASC, CAST(match_id AS INTEGER) ASC
    """)
    
    print("Loading games from DB...")
    all_games = query_df("""
        SELECT game_id, match_id, team1_id, team2_id, team1_win, team2_win, game_duration,
               team1_stats_json, team2_stats_json
        FROM golgg_games
    """)
    # Group games by match_id for fast lookup
    games_by_match = {str(m_id): group for m_id, group in all_games.groupby('match_id')}
    
    print("Loading players from DB...")
    # This might be large, but let's try.
    all_players = query_df("""
        SELECT game_id, player_id, team_id, stats_json
        FROM golgg_game_players
    """)
    # Group players by game_id for fast lookup
    players_by_game = {str(g_id): group for g_id, group in all_players.groupby('game_id')}
    
    results_2026 = []
    
    print(f"Processing {len(all_matches)} matches...")
    for _, m in tqdm(all_matches.iterrows(), total=len(all_matches)):
        m_id = str(m['match_id'])
        t1_id = str(m['team1_id'])
        t2_id = str(m['team2_id'])
        m_date = date.fromisoformat(m['date'])
        best_of = int(m['best_of'] or 1)
        
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
        
        # Prediction (only for 2026)
        if not p1 or not p2:
            continue

        if m_date.year == 2026:
            manager.update_before_match(t1_id, t2_id, p1, p2, m_date)
            preds = manager.predict_match(t1_id, t2_id, p1, p2)
            rolling = {}
            t1_stats = _average_history(team_history.get(t1_id))
            t2_stats = _average_history(team_history.get(t2_id))
            for k, v in t1_stats.items(): rolling[f"t1_rolling_{k}"] = v
            for k, v in t2_stats.items(): rolling[f"t2_rolling_{k}"] = v
            
            features = {f: preds.get(f, 0.5) for f in OPTUNA_BASE_FEATURES}
            features.update(rolling)
            for f in RANK_PROB_FEATURES:
                features[f"{f}_binom_series"] = series_probability(features[f], best_of)
            
            # Ensemble model with symmetry
            vec = np.array([[features[f] for f in ALL_FEATURES]])
            p_orig = pipeline.predict_proba(vec)[0, 1]
            
            # Swapped
            swapped_features = features.copy()
            for k in _DEFAULT_TEAM_STATS.keys():
                swapped_features[f"t1_rolling_{k}"], swapped_features[f"t2_rolling_{k}"] = features[f"t2_rolling_{k}"], features[f"t1_rolling_{k}"]
            for f in RANK_PROB_FEATURES:
                swapped_features[f] = 1.0 - features[f]
                swapped_features[f"{f}_binom_series"] = 1.0 - features[f"{f}_binom_series"]
            swapped_features["player_elo_min1"], swapped_features["player_elo_min2"] = features["player_elo_min2"], features["player_elo_min1"]
            swapped_features["player_gl_max1"], swapped_features["player_gl_max2"] = features["player_gl_max2"], features["player_gl_max1"]
            swapped_features["player_gl_rd_avg1"], swapped_features["player_gl_rd_avg2"] = features["player_gl_rd_avg2"], features["player_gl_rd_avg1"]
            for s in ["ts", "os", "pl", "tm"]:
                swapped_features[f"player_{s}_sigma_avg1"], swapped_features[f"player_{s}_sigma_avg2"] = features[f"player_{s}_sigma_avg2"], features[f"player_{s}_sigma_avg1"]
            
            vec_swapped = np.array([[swapped_features[f] for f in ALL_FEATURES]])
            p_swapped = pipeline.predict_proba(vec_swapped)[0, 1]
            p_sym = (p_orig + (1.0 - p_swapped)) / 2.0
            p_cal = calibrator.predict_proba(np.array([[_logit(p_sym)]]))[0, 1]
            
            # Actual outcome
            t1_wins = 0
            t2_wins = 0
            for _, g in match_games.iterrows():
                g_t1_id = str(g['team1_id'])
                if g_t1_id == t1_id:
                    if g['team1_win'] == 1: t1_wins += 1
                    else: t2_wins += 1
                else:
                    if g['team2_win'] == 1: t1_wins += 1
                    else: t2_wins += 1
            y_true = 1.0 if t1_wins > t2_wins else 0.0
            
            results_2026.append({
                "y_true": y_true,
                "p_ensemble": p_cal,
                "p_elo": preds["player_elo"],
                "p_gl": preds["player_gl"],
                "p_ts": preds["player_ts"],
                "p_os": preds["player_os"],
                "p_pl": preds["player_pl"],
                "p_tm": preds["player_tm"],
            })
        else:
            manager.update_before_match(t1_id, t2_id, p1, p2, m_date)

        # Update
        scores = []
        for _, g in match_games.iterrows():
            g_id = str(g['game_id'])
            g_t1_id = str(g['team1_id'])
            
            # Map game teams to match teams
            if g_t1_id == t1_id:
                s1 = 1 if g['team1_win'] == 1 else 0
                s2 = 1 - s1
                t1_stats = parse_stats(g['team1_stats_json'])
                t2_stats = parse_stats(g['team2_stats_json'])
            else:
                s1 = 1 if g['team2_win'] == 1 else 0
                s2 = 1 - s1
                t1_stats = parse_stats(g['team2_stats_json'])
                t2_stats = parse_stats(g['team1_stats_json'])
                
            scores.append(s1)
            
            # Get player stats for W20
            gp_df = players_by_game.get(g_id)
            
            g_data = {
                "t1_id": t1_id, "t2_id": t2_id, "t1_win": s1 == 1, "t2_win": s2 == 1,
                "game_duration": g['game_duration'],
                "t1_towers": t1_stats.get('towers', 0), "t2_towers": t2_stats.get('towers', 0),
                "t1_nashors": t1_stats.get('nashors', 0), "t2_nashors": t2_stats.get('nashors', 0),
                "t1_gold": t1_stats.get('gold', 0), "t2_gold": t2_stats.get('gold', 0),
                "t1_player_stats": [parse_stats(sj) for sj in gp_df[gp_df['team_id'].astype(str) == t1_id]['stats_json']] if gp_df is not None else [],
                "t2_player_stats": [parse_stats(sj) for sj in gp_df[gp_df['team_id'].astype(str) == t2_id]['stats_json']] if gp_df is not None else [],
            }
            manager.update_after_game(t1_id, t2_id, p1, p2, s1, s2)
            _update_w20_history(team_history, t1_id, g_data, CONTEXT_WINDOW)
            _update_w20_history(team_history, t2_id, g_data, CONTEXT_WINDOW)
            
        manager.update_after_match(t1_id, t2_id, p1, p2, scores)

    # Evaluation
    if not results_2026:
        print("No matches found for 2026.")
        return

    df = pd.DataFrame(results_2026)
    print(f"\nEvaluation for 2026 (N={len(df)} matches):")
    for col in df.columns:
        if col == "y_true": continue
        ll = log_loss(df['y_true'], df[col])
        auc = roc_auc_score(df['y_true'], df[col])
        acc = ((df[col] >= 0.5) == df['y_true']).mean()
        print(f"{col:12} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Acc: {acc:.2%}")

if __name__ == "__main__":
    main()
