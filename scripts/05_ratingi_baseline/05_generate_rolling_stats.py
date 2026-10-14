import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.getcwd())

from src.utils.golgg_schema import games, team1_id, team2_id

def main():
    print("Running 05_generate_rolling_stats.py...")
    os.makedirs("docs/assets", exist_ok=True)
    
    print("Loading golgg_matches.json...")
    with open("data/golgg_matches.json", "r", encoding="utf-8") as f:
        matches = json.load(f)

    # Sort matches by date to ensure rolling window is chronological
    matches.sort(key=lambda x: x['date'])

    team_history = {} # tid -> deque of recent game stats
    WINDOW_SIZE = 10

    results = []

    print("Processing matches for rolling stats...")
    for match in tqdm(matches):
        match_id = str(match['match_id'])
        t1_id = team1_id(match)
        t2_id = team2_id(match)

        # 1. Get current rolling stats for both teams (BEFORE this match)
        def get_avg_stats(tid):
            if tid not in team_history or not team_history[tid]:
                return {
                    'win_rate': 0.5, 'kills': 12.0, 'deaths': 12.0, 
                    'gd15': 0.0, 'dpm': 1800.0, 'vspm': 7.0
                }
            
            hist = list(team_history[tid])
            return {
                'win_rate': np.mean([h['win'] for h in hist]),
                'kills': np.mean([h['kills'] for h in hist]),
                'deaths': np.mean([h['deaths'] for h in hist]),
                'gd15': np.mean([h['gd15'] for h in hist]),
                'dpm': np.mean([h['dpm'] for h in hist]),
                'vspm': np.mean([h['vspm'] for h in hist])
            }

        t1_stats = get_avg_stats(t1_id)
        t2_stats = get_avg_stats(t2_id)

        # Store features for this match
        res = {'golgg_match_id': match_id}
        for stat, val in t1_stats.items(): res[f't1_rolling_{stat}'] = val
        for stat, val in t2_stats.items(): res[f't2_rolling_{stat}'] = val

        results.append(res)

        # 2. Update history with the results of THIS match (for future matches)
        def update_history(tid, is_t1, match_data):
            for game in games(match_data):
                is_team_1_in_game = str(game['t1_id']) == str(tid)
                win = game['t1_win'] if is_team_1_in_game else game['t2_win']
                
                player_stats_key = 't1_players' if is_team_1_in_game else 't2_players'
                players = game[player_stats_key]
                
                kills = sum(p['stats']['kills'] or 0 for p in players.values())
                deaths = sum(p['stats']['deaths'] or 0 for p in players.values())
                dpm = sum(p['stats']['dpm'] or 0 for p in players.values())
                vspm = sum(p['stats']['vspm'] or 0 for p in players.values())
                gd15 = sum(p['stats']['gd@15'] or 0 for p in players.values())

                if tid not in team_history:
                    team_history[tid] = deque(maxlen=WINDOW_SIZE)
                
                team_history[tid].append({
                    'win': float(win),
                    'kills': kills,
                    'deaths': deaths,
                    'dpm': dpm,
                    'vspm': vspm,
                    'gd15': gd15
                })

        update_history(t1_id, True, match)
        update_history(t2_id, False, match)

    df = pd.DataFrame(results)
    df.to_csv("data/golgg_rolling_stats.csv", index=False)
    print(f"Saved rolling stats to golgg_rolling_stats.csv. Total matches: {len(df)}")
    
    # Generate GD15 vs Win Rate plot
    print("Generating GD15 vs Win Rate plot...")
    df_preds = pd.read_csv("data/golgg_y_predicts.csv")
    df_preds['golgg_match_id'] = df_preds['golgg_match_id'].astype(str)
    df_merged = pd.merge(df_preds[['golgg_match_id', 'y_true']], df, on="golgg_match_id")
    
    plt.figure(figsize=(10, 6))
    df_merged['gd15_bin'] = pd.cut(df_merged['t1_rolling_gd15'], bins=np.arange(-3000, 3001, 500))
    win_rate_gd15 = df_merged.groupby('gd15_bin', observed=True)['y_true'].mean()
    
    x_labels = [str(x) for x in win_rate_gd15.index]
    y_values = [float(x) for x in win_rate_gd15.values]
    plt.plot(x_labels, y_values, marker='o', color='gold', linewidth=2)
    plt.title("Impact of Early Game Dominance: Win Rate vs. Rolling GD15")
    plt.xlabel("Rolling Gold Difference at 15 min (T1)")
    plt.ylabel("T1 Win Rate")
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/eda_winrate_vs_gd15.png")
    print("Saved docs/assets/eda_winrate_vs_gd15.png")
    
    print("05_generate_rolling_stats.py completed successfully.")

if __name__ == "__main__":
    main()
