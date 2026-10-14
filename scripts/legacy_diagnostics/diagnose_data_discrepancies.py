
import json
import math
import os
import sys
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import query_df

def main():
    print("Checking matches with empty rosters in DB...")
    # A match has an empty roster if its first game has no players in golgg_game_players
    df_empty = query_df("""
        SELECT COUNT(*) as count FROM golgg_matches m
        WHERE NOT EXISTS (
            SELECT 1 FROM golgg_games g
            JOIN golgg_game_players gp ON gp.game_id = g.game_id
            WHERE g.match_id = m.match_id
        )
        AND m.draw = 0 AND m.date IS NOT NULL
    """)
    print(f"Matches with NO players in DB: {df_empty.iloc[0]['count']}")

    df_total = query_df("""
        SELECT COUNT(*) as count FROM golgg_matches
        WHERE draw = 0 AND date IS NOT NULL
    """)
    print(f"Total matches in DB: {df_total.iloc[0]['count']}")

    print("\nChecking player_id vs player_name in golgg_game_players...")
    df_ids = query_df("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN player_id IS NULL THEN 1 ELSE 0 END) as null_ids,
            SUM(CASE WHEN player_name IS NULL THEN 1 ELSE 0 END) as null_names
        FROM golgg_game_players
    """)
    print(df_ids.to_string(index=False))

    print("\nChecking gd@15 values in DB...")
    df_gd = query_df("""
        SELECT stats_json FROM golgg_game_players 
        WHERE stats_json LIKE '%"gd@15": %' 
        LIMIT 5
    """)
    for sj in df_gd['stats_json']:
        data = json.loads(sj)
        print(f"gd@15: {data.get('gd@15')}")

    print("\nComparing feature distributions (2025 vs 2026) from golgg_y_predicts.csv...")
    df_ref = pd.read_csv("data/golgg_y_predicts.csv")
    df_ref['year'] = df_ref['date'].str[:4]
    
    features_to_check = ["player_gl", "player_ts", "player_elo", "player_gl_rd_avg1"]
    for f in features_to_check:
        print(f"\nFeature: {f}")
        stats = df_ref.groupby('year')[f].agg(['mean', 'std', 'min', 'max', 'count'])
        print(stats.loc[['2024', '2025', '2026']] if '2026' in stats.index else stats.tail(3))

if __name__ == "__main__":
    main()
