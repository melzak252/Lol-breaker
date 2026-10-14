"""
Extract transformer dataset v2 with expanded features.

Features per game (51 total):
  Team-level (10):
    0: win (0/1)
    1: side (blue=1, red=0)
    2: duration (normalized by 1800s)
    3: team kills (/20)
    4: team deaths (/20)
    5: team gold (/60000)
    6: towers (/11)
    7: dragons (/4)
    8: nashors (/2)
    9: gold_diff (/10000) - team gold minus opponent gold

  Per-role player stats (5 roles x 8 features = 40):
    For each role (TOP, JUNGLE, MID, ADC, SUPPORT):
      kills (/15), deaths (/10), assists (/20),
      dpm (/800), gpm (/500), gd@15 (/3000),
      xpd@15 (/1000), wards_placed (/50)

  Metadata (1):
    50: days_since_last_match (/30) - capped at 30 days
"""

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

# Feature dimension per game
FEATURE_DIM = 51
WINDOW_SIZE = 15
ROLES = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']

# Player feature keys and normalization factors
PLAYER_FEATURES = [
    ('kills', 15.0),
    ('deaths', 10.0),
    ('assists', 20.0),
    ('dpm', 800.0),
    ('gpm', 500.0),
    ('gd@15', 3000.0),
    ('xpd@15', 1000.0),
    ('wards_placed', 50.0),
]


def fetch_all_data_from_db():
    """Fetch all games, match info, and player stats from Postgres."""
    # Games query
    games_query = """
    SELECT 
        g.game_id,
        g.match_id,
        g.date,
        g.team1_id,
        g.team2_id,
        g.team1_win,
        g.team2_win,
        g.team1_side,
        g.team2_side,
        g.game_duration,
        g.team1_stats_json,
        g.team2_stats_json,
        m.team1_win as match_team1_win
    FROM golgg_games g
    JOIN golgg_matches m ON g.match_id = m.match_id
    ORDER BY g.date ASC, g.game_id ASC
    """
    print("Fetching games from database...")
    with get_session() as session:
        df_games = pd.read_sql(text(games_query), session.connection())
    df_games['date'] = pd.to_datetime(df_games['date'])
    print(f"  {len(df_games)} games loaded")

    # Player stats query
    player_query = """
    SELECT 
        game_id,
        team_id,
        side,
        role,
        stats_json
    FROM golgg_game_players
    WHERE stats_json IS NOT NULL
    """
    print("Fetching player stats from database...")
    with get_session() as session:
        df_players = pd.read_sql(text(player_query), session.connection())
    print(f"  {len(df_players)} player records loaded")

    # Parse stats_json and build lookup dict: game_id -> {side -> {role -> stats_dict}}
    print("Building player stats lookup...")
    player_lookup = {}
    for _, row in tqdm(df_players.iterrows(), total=len(df_players), desc="Parsing player stats"):
        gid = str(row['game_id'])
        side = row['side']  # 't1' or 't2'
        role = row['role']
        try:
            stats = json.loads(row['stats_json']) if isinstance(row['stats_json'], str) else row['stats_json']
        except:
            stats = {}

        if gid not in player_lookup:
            player_lookup[gid] = {'t1': {}, 't2': {}}
        player_lookup[gid][side][role] = stats

    return df_games, player_lookup


def safe_float(val, default=0.0):
    """Safely convert to float, returning default for None/NaN."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default


def extract_player_features(player_stats, role):
    """Extract normalized player features for a given role.
    Returns list of 8 floats.
    """
    stats = player_stats.get(role, {})
    features = []
    for key, norm in PLAYER_FEATURES:
        val = safe_float(stats.get(key))
        features.append(val / norm)
    return features


def build_game_features(game_row, team_side, player_lookup, last_match_date):
    """
    Build feature vector for one team in one game.
    team_side: 't1' or 't2'
    Returns list of FEATURE_DIM floats.
    """
    gid = str(game_row['game_id'])
    opp_side = 't2' if team_side == 't1' else 't1'

    # Team stats
    if team_side == 't1':
        team_stats = extract_team_stats(game_row['team1_stats_json'])
        opp_stats = extract_team_stats(game_row['team2_stats_json'])
        win = safe_float(game_row['team1_win'])
        side_val = 1.0 if game_row['team1_side'] == 'Blue' else 0.0
    else:
        team_stats = extract_team_stats(game_row['team2_stats_json'])
        opp_stats = extract_team_stats(game_row['team1_stats_json'])
        win = safe_float(game_row['team2_win'])
        side_val = 1.0 if game_row['team2_side'] == 'Blue' else 0.0

    team_gold = safe_float(team_stats.get('gold'))
    opp_gold = safe_float(opp_stats.get('gold'))
    gold_diff = (team_gold - opp_gold) / 10000.0

    # Team-level features (10)
    features = [
        win,
        side_val,
        safe_float(game_row['game_duration']) / 1800.0,
        safe_float(team_stats.get('kills')) / 20.0,
        safe_float(opp_stats.get('kills')) / 20.0,  # team deaths = opponent kills
        team_gold / 60000.0,
        safe_float(team_stats.get('towers')) / 11.0,
        safe_float(team_stats.get('dragons')) / 4.0,
        safe_float(team_stats.get('nashors')) / 2.0,
        gold_diff,
    ]

    # Per-role player features (5 roles x 8 = 40)
    game_players = player_lookup.get(gid, {'t1': {}, 't2': {}})
    team_players = game_players.get(team_side, {})

    for role in ROLES:
        role_features = extract_player_features(team_players, role)
        features.extend(role_features)

    # Days since last match (1)
    match_date = game_row['date']
    if last_match_date is not None:
        days_diff = (match_date - last_match_date).days
        days_diff = min(days_diff, 30)  # Cap at 30 days
        features.append(days_diff / 30.0)
    else:
        features.append(0.0)

    assert len(features) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(features)}"
    return features


def extract_team_stats(stats_json_str):
    if not stats_json_str:
        return {}
    try:
        return json.loads(stats_json_str) if isinstance(stats_json_str, str) else stats_json_str
    except:
        return {}


def build_transformer_dataset(df_games, player_lookup, window_size=WINDOW_SIZE):
    """
    Build dataset where each sample is a match with sequences of last N games
    for both teams.
    """
    # Track per-team: history of feature vectors + last match date
    team_history = {}      # team_id -> deque of feature vectors
    team_last_date = {}    # team_id -> last match date

    dataset = []
    matches = df_games.groupby('match_id', sort=False)

    for match_id, match_games in tqdm(matches, desc="Processing matches"):
        first_game = match_games.iloc[0]
        t1_id = first_game['team1_id']
        t2_id = first_game['team2_id']

        # Get history for both teams BEFORE this match
        t1_seq = list(team_history.get(t1_id, []))
        t2_seq = list(team_history.get(t2_id, []))

        def pad_sequence(seq, size):
            if len(seq) >= size:
                return seq[-size:]
            padding = [[0.0] * FEATURE_DIM] * (size - len(seq))
            return padding + seq

        t1_padded = pad_sequence(t1_seq, window_size)
        t2_padded = pad_sequence(t2_seq, window_size)

        # Target: match-level win
        dataset.append({
            'match_id': match_id,
            't1_id': t1_id,
            't2_id': t2_id,
            't1_seq': t1_padded,
            't2_seq': t2_padded,
            'y': int(first_game['match_team1_win']),
            'date': first_game['date']
        })

        # Update history with games from THIS match
        for _, game in match_games.iterrows():
            # Team 1 features
            t1_last = team_last_date.get(t1_id)
            f1 = build_game_features(game, 't1', player_lookup, t1_last)
            if t1_id not in team_history:
                team_history[t1_id] = deque(maxlen=window_size)
            team_history[t1_id].append(f1)
            team_last_date[t1_id] = game['date']

            # Team 2 features
            t2_last = team_last_date.get(t2_id)
            f2 = build_game_features(game, 't2', player_lookup, t2_last)
            if t2_id not in team_history:
                team_history[t2_id] = deque(maxlen=window_size)
            team_history[t2_id].append(f2)
            team_last_date[t2_id] = game['date']

    return dataset


if __name__ == "__main__":
    df_games, player_lookup = fetch_all_data_from_db()
    print(f"Total games: {len(df_games)}")
    print(f"Feature dim: {FEATURE_DIM}")
    print(f"Window size: {WINDOW_SIZE}")

    dataset = build_transformer_dataset(df_games, player_lookup, window_size=WINDOW_SIZE)
    print(f"Dataset size: {len(dataset)}")

    # Save to file
    output_path = PROJECT_ROOT / "data" / "transformer_team_sequences_v2.json"

    def default_serializer(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(output_path, 'w') as f:
        json.dump(dataset, f, default=default_serializer)

    print(f"Saved dataset to {output_path}")

    # Quick validation
    sample = dataset[1000]
    print(f"\nSample match {sample['match_id']}:")
    print(f"  t1_seq shape: {len(sample['t1_seq'])} x {len(sample['t1_seq'][0])}")
    print(f"  t2_seq shape: {len(sample['t2_seq'])} x {len(sample['t2_seq'][0])}")
    print(f"  y: {sample['y']}")
    # Show non-zero features in last game of t1_seq
    last_game = sample['t1_seq'][-1]
    non_zero = [(i, round(v, 3)) for i, v in enumerate(last_game) if v != 0.0]
    print(f"  t1 last game non-zero features ({len(non_zero)}): {non_zero[:20]}...")
