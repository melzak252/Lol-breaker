#!/usr/bin/env python3
"""
Generate fusion model predictions for matches with features and odds.
Targets both 'upcoming' and 'finished' matches to populate Horizon subpage.
"""
import json
import math
import sys
import os
from datetime import datetime, timezone
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text

# ─── Config ───────────────────────────────────────────────────────────────────
DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://betting:betting_local_password@timescaledb:5432/betting"
).replace("postgresql+psycopg2://", "postgresql://")
MODELS_DIR = "/app/models"
FEATURE_DIM = 51       # per-game transformer input dim
WINDOW_SIZE = 15       # transformer sequence length
ROLES = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']
PLAYER_FEATURES = [
    ('kills', 15.0), ('deaths', 10.0), ('assists', 20.0),
    ('dpm', 800.0), ('gpm', 500.0), ('gd@15', 3000.0),
    ('xpd@15', 1000.0), ('wards_placed', 50.0),
]

# ─── Hybrid Configuration ─────────────────────────────────────────────────────
HYBRID_ALPHA = 0.60        # 60% model, 40% market
HYBRID_TEMPERATURE = 0.8   # Temperature scaling
HYBRID_MODEL_NAME = "Hybrid-Fusion-SymAug-Market"
HYBRID_MODEL_VERSION = "a0.60-t0.80"

# ─── Model Classes (inlined) ─────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TeamEncoder(nn.Module):
    def __init__(self, input_dim=51, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        if mask is not None and mask.all():
            return torch.zeros(x.size(0), self.input_projection.out_features, device=x.device)
            
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).float()
            x = x * (1.0 - expanded_mask)
            sum_x = x.sum(dim=1)
            count_x = (1.0 - expanded_mask).sum(dim=1).clamp(min=1.0)
            embedding = sum_x / count_x
        else:
            embedding = x.mean(dim=1)
        return embedding


class MatchPredictor(nn.Module):
    def __init__(self, input_dim=51, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.team_encoder = TeamEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, seq_a, seq_b, mask_a=None, mask_b=None):
        emb_a = self.team_encoder(seq_a, mask_a)
        emb_b = self.team_encoder(seq_b, mask_b)
        diff = emb_a - emb_b
        combined = torch.cat([emb_a, emb_b, diff], dim=-1)
        return self.classifier(combined)

    def get_embedding(self, seq_a, seq_b, mask_a=None, mask_b=None):
        with torch.no_grad():
            emb_a = self.team_encoder(seq_a, mask_a)
            emb_b = self.team_encoder(seq_b, mask_b)
            diff = emb_a - emb_b
            return torch.cat([emb_a, emb_b, diff], dim=-1)


class FusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.1]
        layers = []
        prev_dim = input_dim
        for h_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(drop_rate)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SymmetricFusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        super().__init__()
        self.mlp = FusionMLP(input_dim, hidden_dims, dropout_rates)

    def forward(self, x, x_swapped=None):
        logit_orig = self.mlp(x)
        if x_swapped is not None:
            logit_swapped = self.mlp(x_swapped)
            sym_logit = (logit_orig - logit_swapped) / 2.0
            return sym_logit, logit_orig, logit_swapped
        return logit_orig


def swap_features(features, baseline_cols, n_baseline, n_embedding=192):
    swapped = features.clone()
    swap_pairs = [
        ('team_elo_r1', 'team_elo_r2'),
        ('player_elo_min1', 'player_elo_min2'),
        ('team_gl_r1', 'team_gl_r2'),
        ('team_gl_rd1', 'team_gl_rd2'),
        ('player_gl_max1', 'player_gl_max2'),
        ('player_gl_rd_avg1', 'player_gl_rd_avg2'),
        ('team_ts_mu1', 'team_ts_mu2'),
        ('team_ts_sigma1', 'team_ts_sigma2'),
        ('player_ts_sigma_avg1', 'player_ts_sigma_avg2'),
        ('team_os_mu1', 'team_os_mu2'),
        ('team_os_sigma1', 'team_os_sigma2'),
        ('player_os_sigma_avg1', 'player_os_sigma_avg2'),
        ('team_pl_mu1', 'team_pl_mu2'),
        ('team_pl_sigma1', 'team_pl_sigma2'),
        ('player_pl_sigma_avg1', 'player_pl_sigma_avg2'),
        ('team_tm_mu1', 'team_tm_mu2'),
        ('team_tm_sigma1', 'team_tm_sigma2'),
        ('player_tm_sigma_avg1', 'player_tm_sigma_avg2'),
        ('days_since_last_1', 'days_since_last_2'),
        ('team1_id', 'team2_id'),
    ]
    col_to_idx = {col: i for i, col in enumerate(baseline_cols)}
    for col1, col2 in swap_pairs:
        if col1 in col_to_idx and col2 in col_to_idx:
            idx1, idx2 = col_to_idx[col1], col_to_idx[col2]
            swapped[:, idx1] = features[:, idx2]
            swapped[:, idx2] = features[:, idx1]
    if 'days_diff' in col_to_idx:
        idx = col_to_idx['days_diff']
        swapped[:, idx] = -features[:, idx]
    emb_start = n_baseline
    swapped[:, emb_start:emb_start+64] = features[:, emb_start+64:emb_start+128]
    swapped[:, emb_start+64:emb_start+128] = features[:, emb_start:emb_start+64]
    swapped[:, emb_start+128:emb_start+192] = -features[:, emb_start+128:emb_start+192]
    return swapped


# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default


def extract_team_stats(stats_json_str):
    if not stats_json_str:
        return {}
    try:
        return json.loads(stats_json_str) if isinstance(stats_json_str, str) else stats_json_str
    except:
        return {}


def extract_player_features(player_stats, role):
    stats = player_stats.get(role, {})
    features = []
    for key, norm in PLAYER_FEATURES:
        val = safe_float(stats.get(key))
        features.append(val / norm)
    return features


def build_game_features(game_row, team_side, player_lookup, last_match_date):
    gid = str(game_row['game_id'])
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

    features = [
        win, side_val,
        safe_float(game_row['game_duration']) / 1800.0,
        safe_float(team_stats.get('kills')) / 20.0,
        safe_float(opp_stats.get('kills')) / 20.0,
        team_gold / 60000.0,
        safe_float(team_stats.get('towers')) / 11.0,
        safe_float(team_stats.get('dragons')) / 4.0,
        safe_float(team_stats.get('nashors')) / 2.0,
        gold_diff,
    ]

    game_players = player_lookup.get(gid, {'t1': {}, 't2': {}})
    team_players = game_players.get(team_side, {})
    for role in ROLES:
        features.extend(extract_player_features(team_players, role))

    match_date = game_row['date']
    if isinstance(match_date, str):
        try:
            match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            match_date = None
    if last_match_date is not None and match_date is not None:
        if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
            match_date = match_date.replace(tzinfo=None)
        if hasattr(last_match_date, 'tzinfo') and last_match_date.tzinfo is not None:
            last_match_date = last_match_date.replace(tzinfo=None)
        days_diff = (match_date - last_match_date).days
        days_diff = min(days_diff, 30)
        features.append(days_diff / 30.0)
    else:
        features.append(0.0)

    return features


# ─── DB Queries ───────────────────────────────────────────────────────────────

def get_matches_to_predict(conn):
    """Get matches with odds and features (upcoming or finished)."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                cm.id AS canonical_match_id,
                cm.canonical_key,
                cm.team_a_name,
                cm.team_b_name,
                cm.start_time_normalized,
                cm.best_of,
                cm.status,
                umf.team_a_golgg_name,
                umf.team_b_golgg_name,
                umf.features_json
            FROM canonical_matches cm
            JOIN upcoming_match_features umf ON umf.canonical_match_id = cm.id
            WHERE cm.status IN ('upcoming', 'finished')
              AND EXISTS (
                  SELECT 1 FROM odds_snapshots os WHERE os.canonical_match_id = cm.id
              )
            ORDER BY cm.start_time_normalized ASC
        """)
        return cur.fetchall()


def get_team_id_mapping(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT team1_name, team1_id FROM golgg_matches
            WHERE team1_id IS NOT NULL
            UNION
            SELECT DISTINCT team2_name, team2_id FROM golgg_matches
            WHERE team2_id IS NOT NULL
        """)
        mapping = {}
        for row in cur.fetchall():
            mapping[row[0]] = row[1]
        return mapping


def get_team_game_history(conn, team_golgg_names):
    if not team_golgg_names:
        return {}
    print(f"  Fetching history for {len(team_golgg_names)} teams...")
    placeholders = ','.join(['%s'] * len(team_golgg_names))
    games_query = f"""
        SELECT
            g.game_id, g.match_id, g.date,
            g.team1_id, g.team2_id,
            g.team1_name, g.team2_name,
            g.team1_win, g.team2_win,
            g.team1_side, g.team2_side,
            g.game_duration,
            g.team1_stats_json, g.team2_stats_json
        FROM golgg_games g
        WHERE g.team1_name IN ({placeholders})
           OR g.team2_name IN ({placeholders})
        ORDER BY g.date ASC, g.game_id ASC
    """
    player_query = f"""
        SELECT gp.game_id, gp.team_id, gp.side, gp.role, gp.stats_json
        FROM golgg_game_players gp
        JOIN golgg_games g ON gp.game_id = g.game_id
        WHERE g.team1_name IN ({placeholders})
           OR g.team2_name IN ({placeholders})
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        params = list(team_golgg_names) * 2
        print("    Executing games query...")
        cur.execute(games_query, params)
        games = cur.fetchall()
        print(f"    Found {len(games)} games.")
        print("    Executing player query...")
        cur.execute(player_query, params)
        players = cur.fetchall()
        print(f"    Found {len(players)} player records.")

    print("    Building player lookup...")
    player_lookup = {}
    for i, p in enumerate(players):
        if i % 50000 == 0:
            print(f"      {i}/{len(players)} player records processed...")
        gid = str(p['game_id'])
        side = p['side']
        role = p['role']
        stats = p['stats_json']
        if isinstance(stats, str):
            try:
                stats = json.loads(stats)
            except:
                stats = {}
        if gid not in player_lookup:
            player_lookup[gid] = {'t1': {}, 't2': {}}
        player_lookup[gid][side][role] = stats or {}

    print("    Processing game features...")
    team_history = {}      # team_name -> list of (features, game_date) tuples
    team_last_date = {}    # team_name -> last game date (for days_since calc)
    name_set = set(team_golgg_names)
    for i, game in enumerate(games):
        if i % 5000 == 0:
            print(f"      {i}/{len(games)} games processed...")
        game_date = game['date']
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date, '%Y-%m-%d')
            except:
                game_date = None
        # Normalize tzinfo for consistent date comparison later
        if game_date is not None and hasattr(game_date, 'tzinfo') and game_date.tzinfo is not None:
            game_date = game_date.replace(tzinfo=None)
        for team_name, team_side in [(game['team1_name'], 't1'), (game['team2_name'], 't2')]:
            if team_name not in name_set:
                continue
            last_date = team_last_date.get(team_name)
            features = build_game_features(game, team_side, player_lookup, last_date)
            if team_name not in team_history:
                team_history[team_name] = []  # store all games for date-aware filtering
            team_history[team_name].append((features, game_date))
            team_last_date[team_name] = game_date
    return team_history


def extract_baseline_features(features_json, team_a_golgg, team_b_golgg,
                               team_id_map, best_of_n, match_date):
    fj = json.loads(features_json) if isinstance(features_json, str) else features_json
    ratings = fj.get('ratings', {})
    player_ratings = fj.get('player_ratings', {})

    def get_team_rating(system, team_key):
        team_data = ratings.get(team_key, {}).get(system, {})
        return {
            'r': safe_float(team_data.get('rating_value')),
            'rd': safe_float(team_data.get('rd')),
            'sigma': safe_float(team_data.get('sigma')),
        }

    def get_player_rating(system, team_key):
        pr = player_ratings.get(team_key, {}).get(system, {})
        players = pr.get('players', [])
        if not players:
            return {'min': 0.0, 'max': 0.0, 'rd_avg': 0.0, 'sigma_avg': 0.0}
        rating_vals = [safe_float(p.get('rating_value')) for p in players]
        rd_vals = [safe_float(p.get('rd')) for p in players]
        sigma_vals = [safe_float(p.get('sigma')) for p in players]
        return {
            'min': min(rating_vals) if rating_vals else 0.0,
            'max': max(rating_vals) if rating_vals else 0.0,
            'rd_avg': np.mean(rd_vals) if rd_vals else 0.0,
            'sigma_avg': np.mean(sigma_vals) if sigma_vals else 0.0,
        }

    def get_rating_prob(system, level='team'):
        probs = ratings.get('probabilities', {}) if level == 'team' else player_ratings.get('probabilities', {})
        return safe_float(probs.get(system))

    features = []
    for sys in ['elo', 'gl', 'ts', 'os', 'pl', 'tm']:
        features.append(get_rating_prob(sys, 'team'))
        features.append(get_rating_prob(sys, 'player'))
        ta = get_team_rating(sys, 'team_a')
        tb = get_team_rating(sys, 'team_b')
        if sys == 'elo':
            features.extend([ta['r'], tb['r']])
            pa = get_player_rating(sys, 'team_a')
            pb = get_player_rating(sys, 'team_b')
            features.extend([pa['min'], pb['min']])
        elif sys == 'gl':
            features.extend([ta['r'], ta['rd'], tb['r'], tb['rd']])
            pa = get_player_rating(sys, 'team_a')
            pb = get_player_rating(sys, 'team_b')
            features.extend([pa['max'], pb['max'], pa['rd_avg'], pb['rd_avg']])
        else:
            features.extend([ta['r'], ta['sigma'], tb['r'], tb['sigma']])
            pa = get_player_rating(sys, 'team_a')
            pb = get_player_rating(sys, 'team_b')
            features.extend([pa['sigma_avg'], pb['sigma_avg']])

    ta_last = ratings.get('team_a', {}).get('elo', {}).get('last_match_at')
    tb_last = ratings.get('team_b', {}).get('elo', {}).get('last_match_at')
    now = match_date if match_date else datetime.now(timezone.utc)
    if isinstance(now, str):
        try:
            now = datetime.fromisoformat(now.replace('Z', '+00:00'))
        except:
            now = datetime.now(timezone.utc)

    def days_since(last_at):
        if not last_at: return 0.0
        try:
            last_dt = datetime.fromisoformat(last_at.replace('Z', '+00:00')) if isinstance(last_at, str) else last_at
            now_tz = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
            last_dt_tz = last_dt if last_dt.tzinfo else last_dt.replace(tzinfo=timezone.utc)
            return max(0, (now_tz - last_dt_tz).days)
        except: return 0.0

    d1, d2 = days_since(ta_last), days_since(tb_last)
    features.extend([float(d1), float(d2), float(d1 - d2)])
    t1_id, t2_id = team_id_map.get(team_a_golgg, ''), team_id_map.get(team_b_golgg, '')
    features.extend([float(t1_id) if t1_id else 0.0, float(t2_id) if t2_id else 0.0])
    features.append(float(best_of_n) if best_of_n else 3.0)
    return np.array(features, dtype=np.float32)


def build_transformer_sequences(team_history, team_a_golgg, team_b_golgg, match_date=None):
    """Build game sequences, filtering to only include games before match_date
    to prevent look-ahead bias (future games leaking into team history).
    """
    if isinstance(match_date, str):
        match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
    if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
        match_date = match_date.replace(tzinfo=None)
    def pad_sequence(history_tuples):
        # history_tuples is list of (features, game_date)
        if match_date is not None:
            # Filter to games strictly before the match date
            filtered = [f for f, d in history_tuples if d is not None and d < match_date]
        else:
            filtered = [f for f, d in history_tuples]
        if len(filtered) >= WINDOW_SIZE:
            return filtered[-WINDOW_SIZE:]
        return [[0.0] * FEATURE_DIM] * (WINDOW_SIZE - len(filtered)) + filtered
    seq_a = np.array(pad_sequence(team_history.get(team_a_golgg, [])), dtype=np.float32)
    seq_b = np.array(pad_sequence(team_history.get(team_b_golgg, [])), dtype=np.float32)
    return seq_a, seq_b


def get_market_odds(conn, canonical_match_id, team_a_name=None, team_b_name=None):
    """
    Get fair market probabilities, ensuring odds_a maps to team_a_name.
    If raw_team_a != team_a_name (swap case), swap odds.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT odds_a, odds_b, raw_team_a, raw_team_b FROM odds_snapshots
            WHERE canonical_match_id = %s AND market_type = 'match_winner'
              AND COALESCE(is_live, 0) = 0 AND odds_a IS NOT NULL AND odds_b IS NOT NULL
            ORDER BY scraped_at DESC LIMIT 1
        """, (canonical_match_id,))
        row = cur.fetchone()
        if not row: return None, None
        odds_a, odds_b = float(row['odds_a']), float(row['odds_b'])
        # Detect swap: if raw_team_a matches team_b_name instead of team_a_name
        if (team_a_name is not None and team_b_name is not None and
                row['raw_team_a'] is not None and row['raw_team_b'] is not None):
            if (row['raw_team_a'].lower() == team_b_name.lower() and
                    row['raw_team_b'].lower() == team_a_name.lower()):
                odds_a, odds_b = odds_b, odds_a
        implied_a, implied_b = 1.0 / odds_a, 1.0 / odds_b
        total = implied_a + implied_b
        return (implied_a / total, implied_b / total) if total > 0 else (None, None)


def hybridize_probability(model_prob, market_prob, alpha=HYBRID_ALPHA, temperature=HYBRID_TEMPERATURE):
    if market_prob is None: return model_prob
    hybrid = max(1e-7, min(1 - 1e-7, alpha * model_prob + (1 - alpha) * market_prob))
    logit = math.log(hybrid / (1 - hybrid))
    return 1.0 / (1.0 + math.exp(-(logit / temperature)))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    conn = psycopg2.connect(DB_DSN)
    print("Connected to DB")
    matches = get_matches_to_predict(conn)
    print(f"Found {len(matches)} matches to process")
    if not matches:
        print("No matches to process."); return
    team_id_map = get_team_id_mapping(conn)
    print(f"Team ID map: {len(team_id_map)} teams")
    all_team_names = set()
    for m in matches:
        if m['team_a_golgg_name']: all_team_names.add(m['team_a_golgg_name'])
        if m['team_b_golgg_name']: all_team_names.add(m['team_b_golgg_name'])
    print(f"Unique teams: {len(all_team_names)}")
    team_history = get_team_game_history(conn, list(all_team_names))
    print(f"Team history: {len(team_history)} teams")

    print("Loading models...")
    transformer = MatchPredictor(input_dim=FEATURE_DIM, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    transformer_ckpt = torch.load(os.path.join(MODELS_DIR, "transformer_best.pt"), map_location=device, weights_only=False)
    transformer.load_state_dict(transformer_ckpt['model_state_dict'] if 'model_state_dict' in transformer_ckpt else transformer_ckpt)
    transformer.to(device).eval()

    fusion_configs = [
        ("Fusion-v2", "fusion_v2_best.pt", "standard"),
        ("Fusion-v2-SymAug", "fusion_v2_sym_best.pt", "symaug"),
        ("Fusion-v2-ArchSym", "fusion_v2_archsym_best.pt", "archsym")
    ]
    fusion_models, fusion_scalers, baseline_cols = {}, {}, None
    for name, ckpt_file, variant in fusion_configs:
        ckpt = torch.load(os.path.join(MODELS_DIR, ckpt_file), map_location=device, weights_only=False)
        if baseline_cols is None: baseline_cols = ckpt.get('baseline_cols', [])
        model = SymmetricFusionModel(ckpt.get('input_dim', 246)) if variant == "archsym" else FusionModel(ckpt.get('input_dim', 246))
        model.load_state_dict(ckpt['model_state_dict'])
        fusion_models[name] = (model.to(device).eval(), variant)
        fusion_scalers[name] = (ckpt['scaler_mean'], ckpt['scaler_scale'])

    n_baseline = len(baseline_cols)
    predictions = []
    for match in matches:
        cm_id = match['canonical_match_id']
        team_a, team_b = match['team_a_golgg_name'] or match['team_a_name'], match['team_b_golgg_name'] or match['team_b_name']
        market_prob_a, _ = get_market_odds(conn, cm_id, team_a, team_b)
        baseline = extract_baseline_features(match['features_json'], team_a, team_b, team_id_map, match['best_of'] or 3, match['start_time_normalized'])
        seq_a, seq_b = build_transformer_sequences(team_history, team_a, team_b, match['start_time_normalized'])
        seq_a_t, seq_b_t = torch.tensor(seq_a, dtype=torch.float32).unsqueeze(0).to(device), torch.tensor(seq_b, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if (seq_a_t.abs().sum() == 0) and (seq_b_t.abs().sum() == 0):
                embedding_np = np.zeros(192, dtype=np.float32)
            else:
                embedding_np = transformer.get_embedding(seq_a_t, seq_b_t).cpu().numpy()[0]

        full_features = np.concatenate([baseline, embedding_np])
        match_preds = {}
        for model_name, (model, variant) in fusion_models.items():
            mean, scale = fusion_scalers[model_name]
            x = torch.tensor((full_features - mean) / scale, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if variant == "standard": prob = torch.sigmoid(model(x)).item()
                elif variant == "symaug":
                    p_orig = torch.sigmoid(model(x))
                    p_swap = torch.sigmoid(model(swap_features(x, baseline_cols, n_baseline)))
                    prob = ((p_orig + (1 - p_swap)) / 2).item()
                elif variant == "archsym":
                    sym_logit, _, _ = model(x, swap_features(x, baseline_cols, n_baseline))
                    prob = torch.sigmoid(sym_logit).item()
            match_preds[model_name] = prob
        
        if 'Fusion-v2-SymAug' in match_preds:
            match_preds[HYBRID_MODEL_NAME] = hybridize_probability(match_preds['Fusion-v2-SymAug'], market_prob_a)
        predictions.append({'cm_id': cm_id, 'preds': match_preds, 'status': match['status']})

    engine = create_engine(DB_DSN)
    with engine.begin() as db:
        artifact_rows = db.execute(
            text("""
                SELECT id, model_name, model_version
                FROM model_artifacts
                WHERE model_name LIKE :fusion_prefix
                   OR model_name = :hybrid_model_name
            """),
            {
                "fusion_prefix": "Fusion%",
                "hybrid_model_name": HYBRID_MODEL_NAME,
            },
        ).mappings()
        artifacts = {f"{r['model_name']}/{r['model_version']}": r["id"] for r in artifact_rows}
        if f"{HYBRID_MODEL_NAME}/{HYBRID_MODEL_VERSION}" not in artifacts:
            artifacts[f"{HYBRID_MODEL_NAME}/{HYBRID_MODEL_VERSION}"] = db.execute(
                text("""
                    INSERT INTO model_artifacts (model_name, model_version, status)
                    VALUES (:model_name, :model_version, 'registered')
                    RETURNING id
                """),
                {
                    "model_name": HYBRID_MODEL_NAME,
                    "model_version": HYBRID_MODEL_VERSION,
                },
            ).scalar_one()

        for p in predictions:
            for mname, prob in p['preds'].items():
                mver = HYBRID_MODEL_VERSION if mname == HYBRID_MODEL_NAME else 'v1.0'
                aid = artifacts.get(f"{mname}/{mver}")
                if not aid: continue
                db.execute(
                    text("""
                        DELETE FROM model_ev_signals
                        WHERE canonical_prediction_id IN (
                            SELECT id
                            FROM canonical_predictions
                            WHERE canonical_match_id = :canonical_match_id
                              AND model_artifact_id = :model_artifact_id
                        )
                    """),
                    {
                        "canonical_match_id": p['cm_id'],
                        "model_artifact_id": aid,
                    },
                )
                db.execute(
                    text("""
                        DELETE FROM canonical_predictions
                        WHERE canonical_match_id = :canonical_match_id
                          AND model_artifact_id = :model_artifact_id
                    """),
                    {
                        "canonical_match_id": p['cm_id'],
                        "model_artifact_id": aid,
                    },
                )
                db.execute(
                    text("""
                        INSERT INTO canonical_predictions (
                            canonical_match_id, model_artifact_id, model_name,
                            model_version, prob_a, prob_b, prediction_status
                        )
                        VALUES (
                            :canonical_match_id, :model_artifact_id, :model_name,
                            :model_version, :prob_a, :prob_b, :prediction_status
                        )
                    """),
                    {
                        "canonical_match_id": p['cm_id'],
                        "model_artifact_id": aid,
                        "model_name": mname,
                        "model_version": mver,
                        "prob_a": prob,
                        "prob_b": 1.0 - prob,
                        "prediction_status": 'active' if p['status'] == 'upcoming' else 'finished',
                    },
                )
    print(f"Processed {len(predictions)} matches.")
    engine.dispose()
    conn.close()

if __name__ == '__main__': main()
