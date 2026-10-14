#!/usr/bin/env python3
"""
Generate fusion model predictions for UPCOMING matches.

Self-contained: all model classes inlined (no external imports).
Reads features from PostgreSQL, builds transformer sequences from golgg data,
runs 3 fusion model variants, inserts into canonical_predictions.

Also generates HYBRID predictions for Fusion-v2-SymAug:
  hybrid = α * model + (1-α) * market
  calibrated = sigmoid(logit(hybrid) / T)
  where α=0.60, T=0.8 (optimal from historical evaluation)
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
    """Siamese transformer. get_embedding() returns 192-dim."""
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
        """Return 192-dim embedding [emb_a, emb_b, emb_a - emb_b]."""
        with torch.no_grad():
            emb_a = self.team_encoder(seq_a, mask_a)
            emb_b = self.team_encoder(seq_b, mask_b)
            diff = emb_a - emb_b
            return torch.cat([emb_a, emb_b, diff], dim=-1)


class FusionModel(nn.Module):
    """Standard fusion model (v2, v2+sym)."""
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
    """Base MLP used by SymmetricFusionModel."""
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
    """Architecturally symmetric fusion model (v2+archsym)."""
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
    """Swap team1↔team2 in features for symmetrization."""
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
    """Build 51-dim feature vector for one team in one game."""
    gid = str(game_row['game_id'])
    opp_side = 't2' if team_side == 't1' else 't1'

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
        if isinstance(last_match_date, str):
            try:
                last_match_date = datetime.fromisoformat(last_match_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                last_match_date = None
    if last_match_date is not None and match_date is not None:
        # Make both offset-naive for subtraction
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

def get_upcoming_matches(conn):
    """Get upcoming matches with odds and features."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                cm.id AS canonical_match_id,
                cm.canonical_key,
                cm.team_a_name,
                cm.team_b_name,
                cm.start_time_normalized,
                cm.best_of,
                umf.team_a_golgg_name,
                umf.team_b_golgg_name,
                umf.features_json
            FROM canonical_matches cm
            JOIN upcoming_match_features umf ON umf.canonical_match_id = cm.id
            WHERE cm.status = 'upcoming'
              AND EXISTS (
                  SELECT 1 FROM odds_snapshots os WHERE os.canonical_match_id = cm.id
              )
            ORDER BY cm.start_time_normalized ASC
        """)
        return cur.fetchall()


def get_team_id_mapping(conn):
    """Build team_name -> team_id mapping from golgg_matches."""
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


def get_team_game_history(conn, team_golgg_names, cutoff_date=None):
    """
    Fetch recent games for given team names from golgg_games + golgg_game_players.
    Returns: dict[team_name] -> list of (date, game_features_dict) sorted by date.
    """
    if not team_golgg_names:
        return {}

    placeholders = ','.join(['%s'] * len(team_golgg_names))

    # Fetch games where these teams played
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

    # Fetch player stats for those games
    player_query = f"""
        SELECT gp.game_id, gp.team_id, gp.side, gp.role, gp.stats_json
        FROM golgg_game_players gp
        JOIN golgg_games g ON gp.game_id = g.game_id
        WHERE g.team1_name IN ({placeholders})
           OR g.team2_name IN ({placeholders})
    """

    print(f"  Fetching games for {len(team_golgg_names)} teams...")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        params = list(team_golgg_names) * 2
        cur.execute(games_query, params)
        games = cur.fetchall()
    print(f"  {len(games)} games loaded")

    print(f"  Fetching player stats...")
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(player_query, params)
        players = cur.fetchall()
    print(f"  {len(players)} player records loaded")

    # Build player lookup: game_id -> {side -> {role -> stats}}
    player_lookup = {}
    for p in players:
        gid = str(p['game_id'])
        side = p['side']
        role = p['role']
        try:
            stats = json.loads(p['stats_json']) if isinstance(p['stats_json'], str) else (p['stats_json'] or {})
        except:
            stats = {}
        if gid not in player_lookup:
            player_lookup[gid] = {'t1': {}, 't2': {}}
        player_lookup[gid][side][role] = stats

    # Build per-team history
    team_history = {}   # team_name -> deque of feature vectors
    team_last_date = {} # team_name -> last match date

    name_set = set(team_golgg_names)

    for game in games:
        game_date = game['date']
        if isinstance(game_date, str):
            try:
                game_date = datetime.strptime(game_date, '%Y-%m-%d')
            except:
                game_date = None

        for team_name, team_side, opp_side in [
            (game['team1_name'], 't1', 't2'),
            (game['team2_name'], 't2', 't1'),
        ]:
            if team_name not in name_set:
                continue

            last_date = team_last_date.get(team_name)
            features = build_game_features(game, team_side, player_lookup, last_date)

            if team_name not in team_history:
                team_history[team_name] = deque(maxlen=WINDOW_SIZE)
            team_history[team_name].append(features)
            team_last_date[team_name] = game_date

    return team_history


def extract_baseline_features(features_json, team_a_golgg, team_b_golgg,
                               team_id_map, best_of_n, match_date):
    """
    Extract 54-dim baseline feature vector from features_json.
    Order MUST match checkpoint baseline_cols exactly:
      For each system (elo, gl, ts, os, pl, tm):
        team_prob, player_prob, team_raw_values, player_raw_values
      Then: days_since_last_1, days_since_last_2, days_diff, team1_id, team2_id, BoN
    Returns numpy array of shape (54,).
    """
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
        if level == 'team':
            probs = ratings.get('probabilities', {})
        else:
            probs = player_ratings.get('probabilities', {})
        return safe_float(probs.get(system))

    features = []

    # ── ELO block (6 features) ──
    # team_elo, player_elo, team_elo_r1, team_elo_r2, player_elo_min1, player_elo_min2
    features.append(get_rating_prob('elo', 'team'))
    features.append(get_rating_prob('elo', 'player'))
    ta_elo = get_team_rating('elo', 'team_a')
    tb_elo = get_team_rating('elo', 'team_b')
    features.extend([ta_elo['r'], tb_elo['r']])
    pe_a = get_player_rating('elo', 'team_a')
    pe_b = get_player_rating('elo', 'team_b')
    features.extend([pe_a['min'], pe_b['min']])

    # ── GL block (8 features) ──
    # team_gl, player_gl, team_gl_r1, team_gl_rd1, team_gl_r2, team_gl_rd2,
    # player_gl_max1, player_gl_max2, player_gl_rd_avg1, player_gl_rd_avg2
    features.append(get_rating_prob('gl', 'team'))
    features.append(get_rating_prob('gl', 'player'))
    ta_gl = get_team_rating('gl', 'team_a')
    tb_gl = get_team_rating('gl', 'team_b')
    features.extend([ta_gl['r'], ta_gl['rd'], tb_gl['r'], tb_gl['rd']])
    pg_a = get_player_rating('gl', 'team_a')
    pg_b = get_player_rating('gl', 'team_b')
    features.extend([pg_a['max'], pg_b['max'], pg_a['rd_avg'], pg_b['rd_avg']])

    # ── TS block (8 features) ──
    # team_ts, player_ts, team_ts_mu1, team_ts_sigma1, team_ts_mu2, team_ts_sigma2,
    # player_ts_sigma_avg1, player_ts_sigma_avg2
    features.append(get_rating_prob('ts', 'team'))
    features.append(get_rating_prob('ts', 'player'))
    ta_ts = get_team_rating('ts', 'team_a')
    tb_ts = get_team_rating('ts', 'team_b')
    features.extend([ta_ts['r'], ta_ts['sigma'], tb_ts['r'], tb_ts['sigma']])
    pt_a = get_player_rating('ts', 'team_a')
    pt_b = get_player_rating('ts', 'team_b')
    features.extend([pt_a['sigma_avg'], pt_b['sigma_avg']])

    # ── OS block (8 features) ──
    features.append(get_rating_prob('os', 'team'))
    features.append(get_rating_prob('os', 'player'))
    ta_os = get_team_rating('os', 'team_a')
    tb_os = get_team_rating('os', 'team_b')
    features.extend([ta_os['r'], ta_os['sigma'], tb_os['r'], tb_os['sigma']])
    po_a = get_player_rating('os', 'team_a')
    po_b = get_player_rating('os', 'team_b')
    features.extend([po_a['sigma_avg'], po_b['sigma_avg']])

    # ── PL block (8 features) ──
    features.append(get_rating_prob('pl', 'team'))
    features.append(get_rating_prob('pl', 'player'))
    ta_pl = get_team_rating('pl', 'team_a')
    tb_pl = get_team_rating('pl', 'team_b')
    features.extend([ta_pl['r'], ta_pl['sigma'], tb_pl['r'], tb_pl['sigma']])
    pp_a = get_player_rating('pl', 'team_a')
    pp_b = get_player_rating('pl', 'team_b')
    features.extend([pp_a['sigma_avg'], pp_b['sigma_avg']])

    # ── TM block (8 features) ──
    features.append(get_rating_prob('tm', 'team'))
    features.append(get_rating_prob('tm', 'player'))
    ta_tm = get_team_rating('tm', 'team_a')
    tb_tm = get_team_rating('tm', 'team_b')
    features.extend([ta_tm['r'], ta_tm['sigma'], tb_tm['r'], tb_tm['sigma']])
    pm_a = get_player_rating('tm', 'team_a')
    pm_b = get_player_rating('tm', 'team_b')
    features.extend([pm_a['sigma_avg'], pm_b['sigma_avg']])

    # ── Days features (3 features) ──
    ta_elo_data = ratings.get('team_a', {}).get('elo', {})
    tb_elo_data = ratings.get('team_b', {}).get('elo', {})
    ta_last = ta_elo_data.get('last_match_at')
    tb_last = tb_elo_data.get('last_match_at')

    now = match_date if match_date else datetime.now(timezone.utc)
    if isinstance(now, str):
        try:
            now = datetime.fromisoformat(now.replace('Z', '+00:00'))
        except:
            now = datetime.now(timezone.utc)

    def days_since(last_at):
        if not last_at:
            return 0.0
        try:
            if isinstance(last_at, str):
                last_dt = datetime.fromisoformat(last_at.replace('Z', '+00:00'))
            else:
                last_dt = last_at
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now_tz = now.replace(tzinfo=timezone.utc)
            else:
                now_tz = now
            return max(0, (now_tz - last_dt).days)
        except:
            return 0.0

    d1 = days_since(ta_last)
    d2 = days_since(tb_last)
    features.extend([float(d1), float(d2), float(d1 - d2)])

    # ── Team IDs (2 features) ──
    t1_id = team_id_map.get(team_a_golgg, '')
    t2_id = team_id_map.get(team_b_golgg, '')
    t1_num = float(hash(t1_id) % 10000) if isinstance(t1_id, str) and t1_id else safe_float(t1_id)
    t2_num = float(hash(t2_id) % 10000) if isinstance(t2_id, str) and t2_id else safe_float(t2_id)
    features.extend([t1_num, t2_num])

    # ── BoN (1 feature) ──
    features.append(float(best_of_n) if best_of_n else 3.0)

    assert len(features) == 54, f"Expected 54 baseline features, got {len(features)}"
    return np.array(features, dtype=np.float32)


def build_transformer_sequences(team_history, team_a_golgg, team_b_golgg):
    """
    Build 15x51 transformer input sequences for both teams.
    Returns (seq_a, seq_b) as numpy arrays of shape (15, 51).
    """
    def pad_sequence(history_list):
        if len(history_list) >= WINDOW_SIZE:
            return history_list[-WINDOW_SIZE:]
        padding = [[0.0] * FEATURE_DIM] * (WINDOW_SIZE - len(history_list))
        return padding + history_list

    t1_hist = list(team_history.get(team_a_golgg, []))
    t2_hist = list(team_history.get(team_b_golgg, []))

    seq_a = np.array(pad_sequence(t1_hist), dtype=np.float32)
    seq_b = np.array(pad_sequence(t2_hist), dtype=np.float32)
    return seq_a, seq_b


def get_market_odds(conn, canonical_match_id, team_a_name=None, team_b_name=None):
    """
    Get latest market odds for a match and convert to fair probabilities.
    If raw_team_a != team_a_name (swap case), swap odds to align.
    Returns (prob_a, prob_b) or (None, None) if no odds available.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT odds_a, odds_b, raw_team_a, raw_team_b
            FROM odds_snapshots
            WHERE canonical_match_id = %s
              AND market_type = 'match_winner'
              AND COALESCE(is_live, 0) = 0
              AND odds_a IS NOT NULL
              AND odds_b IS NOT NULL
            ORDER BY scraped_at DESC
            LIMIT 1
        """, (canonical_match_id,))
        row = cur.fetchone()
        
        if not row:
            return None, None
        
        odds_a = float(row['odds_a'])
        odds_b = float(row['odds_b'])
        
        # Detect swap: if raw_team_a matches team_b_name instead of team_a_name
        if (team_a_name is not None and team_b_name is not None and
                row['raw_team_a'] is not None and row['raw_team_b'] is not None):
            if (row['raw_team_a'].lower() == team_b_name.lower() and
                    row['raw_team_b'].lower() == team_a_name.lower()):
                odds_a, odds_b = odds_b, odds_a
        
        # Convert to implied probabilities and remove overround
        implied_a = 1.0 / odds_a
        implied_b = 1.0 / odds_b
        total = implied_a + implied_b
        
        if total <= 0:
            return None, None
        
        # Fair probabilities (normalized)
        prob_a = implied_a / total
        prob_b = implied_b / total
        
        return prob_a, prob_b


def hybridize_probability(model_prob, market_prob, alpha=HYBRID_ALPHA, temperature=HYBRID_TEMPERATURE):
    """
    Create hybrid probability: blend model + market, then apply temperature scaling.
    
    hybrid = α * model + (1-α) * market
    calibrated = sigmoid(logit(hybrid) / T)
    
    Args:
        model_prob: Model's predicted probability for team A
        market_prob: Market's implied probability for team A
        alpha: Blend weight (0=market only, 1=model only)
        temperature: Temperature scaling (< 1 = sharper, > 1 = softer)
    
    Returns:
        Calibrated hybrid probability for team A
    """
    if market_prob is None:
        # No market odds available, use model only
        return model_prob
    
    # Blend
    hybrid = alpha * model_prob + (1 - alpha) * market_prob
    
    # Clip to avoid log(0) or log(inf)
    hybrid = max(1e-7, min(1 - 1e-7, hybrid))
    
    # Temperature scaling via logit space
    logit = math.log(hybrid / (1 - hybrid))
    scaled_logit = logit / temperature
    calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
    
    return calibrated


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    conn = psycopg2.connect(DB_DSN)
    print("Connected to database")

    # 1. Get upcoming matches
    matches = get_upcoming_matches(conn)
    print(f"Found {len(matches)} upcoming matches with odds and features")
    if not matches:
        print("No upcoming matches to process. Exiting.")
        return

    # 2. Get team ID mapping
    team_id_map = get_team_id_mapping(conn)
    print(f"Team ID mapping: {len(team_id_map)} teams")

    # 3. Collect all unique team golgg names
    all_team_names = set()
    for m in matches:
        if m['team_a_golgg_name']:
            all_team_names.add(m['team_a_golgg_name'])
        if m['team_b_golgg_name']:
            all_team_names.add(m['team_b_golgg_name'])
    print(f"Unique teams in upcoming matches: {len(all_team_names)}")

    # 4. Build transformer sequences for all teams
    print("\nBuilding transformer sequences from game history...")
    team_history = get_team_game_history(conn, list(all_team_names))
    print(f"Team history built for {len(team_history)} teams")
    for tn in sorted(all_team_names):
        hist_len = len(team_history.get(tn, []))
        print(f"  {tn}: {hist_len} recent games")

    # 5. Load transformer model
    print("\nLoading transformer model...")
    transformer = MatchPredictor(input_dim=FEATURE_DIM, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    ckpt_path = os.path.join(MODELS_DIR, "transformer_best.pt")
    transformer_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Raw state_dict (not wrapped)
    if 'model_state_dict' in transformer_ckpt:
        transformer.load_state_dict(transformer_ckpt['model_state_dict'])
    else:
        transformer.load_state_dict(transformer_ckpt)
    transformer.to(device)
    transformer.eval()
    print("Transformer model loaded")

    # 6. Load fusion models
    print("\nLoading fusion models...")
    fusion_configs = [
        ("Fusion-v2", "fusion_v2_best.pt", "standard"),
        ("Fusion-v2-SymAug", "fusion_v2_sym_best.pt", "symaug"),
        ("Fusion-v2-ArchSym", "fusion_v2_archsym_best.pt", "archsym"),
    ]
    fusion_models = {}
    fusion_scalers = {}
    baseline_cols = None

    for name, ckpt_file, variant in fusion_configs:
        ckpt = torch.load(os.path.join(MODELS_DIR, ckpt_file), map_location=device, weights_only=False)
        input_dim = ckpt.get('input_dim', 246)
        saved_cols = ckpt.get('baseline_cols', [])

        if baseline_cols is None:
            baseline_cols = saved_cols
            print(f"  Baseline cols ({len(baseline_cols)}): {baseline_cols[:5]}...{baseline_cols[-3:]}")

        if variant == "archsym":
            model = SymmetricFusionModel(input_dim)
        else:
            model = FusionModel(input_dim)

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()

        fusion_models[name] = (model, variant)
        fusion_scalers[name] = (ckpt['scaler_mean'], ckpt['scaler_scale'])
        print(f"  {name} ({variant}) loaded, input_dim={input_dim}")

    n_baseline = len(baseline_cols)
    print(f"Baseline features: {n_baseline}, Total input: {n_baseline + 192}")

    # 7. Generate predictions for each match
    print(f"\nGenerating predictions for {len(matches)} matches...")
    predictions = []

    for i, match in enumerate(matches):
        cm_id = match['canonical_match_id']
        team_a = match['team_a_golgg_name'] or match['team_a_name']
        team_b = match['team_b_golgg_name'] or match['team_b_name']
        bon = match['best_of'] or 3
        match_date = match['start_time_normalized']

        print(f"\n  [{i+1}/{len(matches)}] {match['team_a_name']} vs {match['team_b_name']}")
        print(f"    Golgg names: {team_a} vs {team_b}")

        # Get market odds for hybridization
        market_prob_a, market_prob_b = get_market_odds(conn, cm_id, team_a, team_b)
        if market_prob_a is not None:
            print(f"    Market odds: {market_prob_a:.3f} / {market_prob_b:.3f}")
        else:
            print(f"    Market odds: NOT AVAILABLE (will use model-only for hybrid)")

        # Extract baseline features (54-dim)
        baseline = extract_baseline_features(
            match['features_json'], team_a, team_b,
            team_id_map, bon, match_date
        )

        # Build transformer sequences and get embeddings (192-dim)
        seq_a, seq_b = build_transformer_sequences(team_history, team_a, team_b)
        seq_a_t = torch.tensor(seq_a, dtype=torch.float32).unsqueeze(0).to(device)
        seq_b_t = torch.tensor(seq_b, dtype=torch.float32).unsqueeze(0).to(device)

        # Check if sequences are all-zero (no game history for team)
        a_all_zero = (seq_a_t.abs().sum() == 0).item()
        b_all_zero = (seq_b_t.abs().sum() == 0).item()

        if a_all_zero and b_all_zero:
            # Both teams have no history — use zero embeddings
            embedding_np = np.zeros(192, dtype=np.float32)
            print(f"    WARNING: Both teams have no game history, using zero embeddings")
        elif a_all_zero or b_all_zero:
            # One team has no history — use zero embedding for that team, run transformer for the other
            with torch.no_grad():
                if a_all_zero:
                    emb_a = torch.zeros(1, 64).to(device)
                    mask_b = (seq_b_t.abs().sum(dim=-1) == 0)
                    emb_b = transformer.team_encoder(seq_b_t, mask_b)
                else:
                    mask_a = (seq_a_t.abs().sum(dim=-1) == 0)
                    emb_a = transformer.team_encoder(seq_a_t, mask_a)
                    emb_b = torch.zeros(1, 64).to(device)
                diff = emb_a - emb_b
                embedding = torch.cat([emb_a, emb_b, diff], dim=-1)
            embedding_np = embedding.cpu().numpy()[0]
            print(f"    WARNING: {'Team A' if a_all_zero else 'Team B'} has no game history, using zero embedding")
        else:
            # Create padding masks (True for padded/zero rows) — shape (batch=1, seq_len=15)
            with torch.no_grad():
                embedding = transformer.get_embedding(seq_a_t, seq_b_t)
            embedding_np = embedding.cpu().numpy()[0]  # (192,)

        # Combine: 54 baseline + 192 embedding = 246
        full_features = np.concatenate([baseline, embedding_np])
        print(f"    Features shape: {full_features.shape}")

        # Run each fusion model
        match_preds = {}
        for model_name, (model, variant) in fusion_models.items():
            scaler_mean, scaler_scale = fusion_scalers[model_name]
            scaled = (full_features - scaler_mean) / scaler_scale
            x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                if variant == "standard":
                    logit = model(x)
                    prob = torch.sigmoid(logit).item()
                elif variant == "symaug":
                    logit_orig = model(x)
                    p_orig = torch.sigmoid(logit_orig)
                    x_swap = swap_features(x, baseline_cols, n_baseline)
                    logit_swap = model(x_swap)
                    p_swap = torch.sigmoid(logit_swap)
                    prob = ((p_orig + (1 - p_swap)) / 2).item()
                elif variant == "archsym":
                    x_swap = swap_features(x, baseline_cols, n_baseline)
                    sym_logit, _, _ = model(x, x_swap)
                    prob = torch.sigmoid(sym_logit).item()

            match_preds[model_name] = prob
            print(f"    {model_name}: {prob:.4f}")

        # Generate HYBRID prediction for Fusion-v2-SymAug
        symaug_prob = match_preds.get('Fusion-v2-SymAug')
        if symaug_prob is not None:
            hybrid_prob_a = hybridize_probability(symaug_prob, market_prob_a)
            hybrid_prob_b = 1.0 - hybrid_prob_a
            match_preds[HYBRID_MODEL_NAME] = hybrid_prob_a
            print(f"    {HYBRID_MODEL_NAME}: {hybrid_prob_a:.4f} (α={HYBRID_ALPHA}, T={HYBRID_TEMPERATURE})")

        predictions.append({
            'canonical_match_id': cm_id,
            'match_key': match['canonical_key'],
            'team_a': match['team_a_name'],
            'team_b': match['team_b_name'],
            'predictions': match_preds,
        })

    # 8. Insert predictions into canonical_predictions
    print(f"\n\nInserting predictions...")

    # Get model artifact IDs for fusion models
    with conn.cursor() as cur:
        cur.execute("SELECT id, model_name, model_version FROM model_artifacts WHERE model_name LIKE %s", ('Fusion%',))
        artifacts = {f"{row[1]}/{row[2]}": row[0] for row in cur.fetchall()}

    # Create or get hybrid model artifact
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id FROM model_artifacts
            WHERE model_name = %s AND model_version = %s
        """, (HYBRID_MODEL_NAME, HYBRID_MODEL_VERSION))
        row = cur.fetchone()
        if row:
            hybrid_artifact_id = row[0]
            print(f"Hybrid artifact exists: id={hybrid_artifact_id}")
        else:
            cur.execute("""
                INSERT INTO model_artifacts (model_name, model_version, status)
                VALUES (%s, %s, 'registered')
                RETURNING id
            """, (HYBRID_MODEL_NAME, HYBRID_MODEL_VERSION))
            hybrid_artifact_id = cur.fetchone()[0]
            print(f"Created hybrid artifact: id={hybrid_artifact_id}")
    conn.commit()

    # Add hybrid artifact to lookup
    artifacts[f"{HYBRID_MODEL_NAME}/{HYBRID_MODEL_VERSION}"] = hybrid_artifact_id
    print(f"Model artifacts: {artifacts}")

    inserted = 0
    with conn.cursor() as cur:
        for pred in predictions:
            cm_id = pred['canonical_match_id']
            for model_name, prob in pred['predictions'].items():
                # Find artifact ID
                artifact_id = None
                for key, aid in artifacts.items():
                    if key.startswith(model_name):
                        artifact_id = aid
                        break

                if artifact_id is None:
                    print(f"  WARNING: No artifact found for {model_name}, skipping")
                    continue

                # DELETE dependent EV signals first, then predictions
                cur.execute("""
                    DELETE FROM model_ev_signals
                    WHERE canonical_prediction_id IN (
                        SELECT id FROM canonical_predictions
                        WHERE canonical_match_id = %s AND model_artifact_id = %s
                    )
                """, (cm_id, artifact_id))

                cur.execute("""
                    DELETE FROM canonical_predictions
                    WHERE canonical_match_id = %s AND model_artifact_id = %s
                """, (cm_id, artifact_id))

                cur.execute("""
                    INSERT INTO canonical_predictions
                        (canonical_match_id, model_artifact_id, model_name, model_version,
                         prob_a, prob_b, prediction_status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'active')
                """, (cm_id, artifact_id, model_name,
                      HYBRID_MODEL_VERSION if model_name == HYBRID_MODEL_NAME else 'v1.0',
                      prob, 1.0 - prob))
                inserted += 1

    conn.commit()
    print(f"Inserted/updated {inserted} predictions")

    # 9. Verify
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM canonical_predictions cp
            JOIN canonical_matches cm ON cp.canonical_match_id = cm.id
            WHERE cm.status = 'upcoming'
        """)
        total = cur.fetchone()[0]
    print(f"\nTotal upcoming predictions in DB: {total}")

    # Print summary
    print("\n=== PREDICTION SUMMARY ===")
    for pred in predictions:
        print(f"\n{pred['team_a']} vs {pred['team_b']} (match_id={pred['canonical_match_id']})")
        for model_name, prob in pred['predictions'].items():
            print(f"  {model_name}: {prob:.4f}")

    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
