"""Causal champion-pool feature extraction for pre-match series prediction.

Extracts causal champion-pool breadth, entropy, concentration, and team-level
flexibility features strictly as-of the match cutoff from historical sequence banks.
All features preserve side-swap anti-symmetry:
    Delta X(A, B) = - Delta X(B, A)
and neutral cold-start stability (all zeros when 0 valid games).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Tuple

import numpy as np

ROLES: list[str] = ["top", "jungle", "mid", "adc", "support"]

PLAYER_METRICS: list[str] = [
    "pool_size_norm",
    "entropy_norm",
    "top1_concentration",
    "top2_concentration",
    "active_games_norm",
]

TEAM_METRICS: list[str] = [
    "team_mean_entropy",
    "team_min_entropy",
    "team_mean_pool_size",
    "team_max_top1",
    "team_flex_champions",
]

CHAMPION_FEATURE_NAMES: list[str] = [
    f"{role}_{metric}" for role in ROLES for metric in PLAYER_METRICS
] + TEAM_METRICS

DIFF_FEATURE_NAMES: list[str] = [f"diff_{name}" for name in CHAMPION_FEATURE_NAMES]

assert len(CHAMPION_FEATURE_NAMES) == 30, f"Expected 30 feature names, got {len(CHAMPION_FEATURE_NAMES)}"
assert len(DIFF_FEATURE_NAMES) == 30, f"Expected 30 diff feature names, got {len(DIFF_FEATURE_NAMES)}"


def extract_champion_pool_features(
    champions: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract player- and team-level champion pool features.

    Args:
        champions: Array of shape (N, 2, 5, 16), dtype uint16 (or integer).
            Champion IDs in player recent history. 0 denotes empty / invalid slot.
        mask: Array of shape (N, 2, 5, 16), dtype uint8 (or integer / bool).
            1 denotes valid historical map slot, 0 denotes invalid / masked.

    Returns:
        Tuple of:
            side_features: Array of shape (N, 2, 30), dtype float32.
                Features for side 0 (team 1) and side 1 (team 2).
            diff_features: Array of shape (N, 30), dtype float32.
                Differential features: side_features[:, 0, :] - side_features[:, 1, :].
    """
    if champions.ndim != 4 or champions.shape[1:] != (2, 5, 16):
        raise ValueError(f"Expected champions of shape (N, 2, 5, 16), got {champions.shape}")
    if mask.ndim != 4 or mask.shape != champions.shape:
        raise ValueError(f"Expected mask matching champions shape {champions.shape}, got {mask.shape}")

    n_series = champions.shape[0]
    if n_series == 0:
        side_features = np.zeros((0, 2, 30), dtype=np.float32)
        diff_features = np.zeros((0, 30), dtype=np.float32)
        return side_features, diff_features

    # Valid champion is champ > 0 and mask == 1
    valid_champ = np.where((mask == 1) & (champions > 0), champions, 0)

    # Flatten to (n_players, 16) where n_players = n_series * 2 * 5
    n_players = n_series * 2 * 5
    flat_champs = valid_champ.reshape(n_players, 16)

    # Find max champion ID to dimension the count array safely
    max_champ_id = int(flat_champs.max())
    n_champ_bins = max(174, max_champ_id + 1)

    # Accumulate counts per player: shape (n_players, n_champ_bins)
    counts = np.zeros((n_players, n_champ_bins), dtype=np.int16)
    row_idx = np.repeat(np.arange(n_players, dtype=np.int64), 16)
    col_idx = flat_champs.ravel().astype(np.int64)
    np.add.at(counts, (row_idx, col_idx), 1)

    # Column 0 is non-valid / empty slot; champion IDs start at 1
    champ_counts = counts[:, 1:]  # shape (n_players, n_champ_bins - 1)
    valid_games = champ_counts.sum(axis=1)  # shape (n_players,)
    has_games = valid_games > 0

    # 1. active_games_norm: valid games count / 16.0
    active_games_norm = np.zeros(n_players, dtype=np.float32)
    active_games_norm[has_games] = valid_games[has_games].astype(np.float32) / 16.0

    # 2. pool_size_norm: unique champions / 16.0
    unique_champs = (champ_counts > 0).sum(axis=1)
    pool_size_norm = np.zeros(n_players, dtype=np.float32)
    pool_size_norm[has_games] = unique_champs[has_games].astype(np.float32) / 16.0

    # 3. entropy_norm: H / 4.0 where H = - sum(p_c * log2(p_c))
    # Using identity: H = log2(K) - sum(count * log2(count)) / K
    entropy_norm = np.zeros(n_players, dtype=np.float32)
    nz_rows, nz_cols = np.nonzero(champ_counts)
    if len(nz_rows) > 0:
        nz_counts = champ_counts[nz_rows, nz_cols].astype(np.float32)
        nz_terms = nz_counts * np.log2(nz_counts)
        sum_terms = np.zeros(n_players, dtype=np.float32)
        np.add.at(sum_terms, nz_rows, nz_terms)
        k_valid = valid_games[has_games].astype(np.float32)
        h_entropy = np.log2(k_valid) - (sum_terms[has_games] / k_valid)
        entropy_norm[has_games] = np.clip(h_entropy / 4.0, 0.0, 1.0)

    # 4 & 5. top1 and top2 concentration
    if champ_counts.shape[1] >= 2:
        top2_vals = np.partition(champ_counts, -2, axis=1)[:, -2:]
        top1_count = top2_vals.max(axis=1).astype(np.float32)
        second_count = top2_vals.min(axis=1).astype(np.float32)
    else:
        top1_count = champ_counts.max(axis=1).astype(np.float32)
        second_count = np.zeros(n_players, dtype=np.float32)

    top1_conc = np.zeros(n_players, dtype=np.float32)
    top2_conc = np.zeros(n_players, dtype=np.float32)
    if np.any(has_games):
        k_val = valid_games[has_games].astype(np.float32)
        top1_conc[has_games] = np.clip(top1_count[has_games] / k_val, 0.0, 1.0)
        top2_conc[has_games] = np.clip((top1_count[has_games] + second_count[has_games]) / k_val, 0.0, 1.0)

    # Reshape player features to (N, 2, 5, 5)
    player_feats = np.stack(
        [
            pool_size_norm,
            entropy_norm,
            top1_conc,
            top2_conc,
            active_games_norm,
        ],
        axis=1,
    ).reshape(n_series, 2, 5, 5)

    # Flatten the 5 roles to (N, 2, 25)
    player_feats_25 = player_feats.reshape(n_series, 2, 25)

    # Team features: (N, 2, 5)
    team_mean_entropy = player_feats[:, :, :, 1].mean(axis=2)  # (N, 2)
    team_min_entropy = player_feats[:, :, :, 1].min(axis=2)    # (N, 2)
    team_mean_pool_size = player_feats[:, :, :, 0].mean(axis=2)  # (N, 2)
    team_max_top1 = player_feats[:, :, :, 2].max(axis=2)       # (N, 2)

    # Flex picks: unique champion IDs appearing in >= 2 roles' recent histories
    champ_presence = (champ_counts > 0).reshape(n_series, 2, 5, -1)
    roles_per_champ = champ_presence.sum(axis=2)  # (N, 2, n_champs)
    team_flex_champions = (roles_per_champ >= 2).sum(axis=2).astype(np.float32)  # (N, 2)

    team_feats_5 = np.stack(
        [
            team_mean_entropy,
            team_min_entropy,
            team_mean_pool_size,
            team_max_top1,
            team_flex_champions,
        ],
        axis=-1,
    )  # (N, 2, 5)

    side_features = np.concatenate([player_feats_25, team_feats_5], axis=-1).astype(np.float32)
    diff_features = (side_features[:, 0, :] - side_features[:, 1, :]).astype(np.float32)

    return side_features, diff_features


def extract_champion_pool_features_from_bank(
    bank_dir: Path | str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load canonical sequence bank and compute champion pool features.

    Args:
        bank_dir: Path to directory containing champions.npy, mask.npy, metadata.parquet.

    Returns:
        Tuple of (side_features, diff_features, metadata_sha256).
    """
    bank_path = Path(bank_dir).resolve()
    champions_path = bank_path / "champions.npy"
    mask_path = bank_path / "mask.npy"
    meta_path = bank_path / "metadata.parquet"

    if not champions_path.is_file():
        raise FileNotFoundError(f"Missing {champions_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing {mask_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta_bytes = meta_path.read_bytes()
    meta_sha256 = hashlib.sha256(meta_bytes).hexdigest()

    champions = np.load(champions_path)
    mask = np.load(mask_path)

    side_features, diff_features = extract_champion_pool_features(champions, mask)
    return side_features, diff_features, meta_sha256


def verify_champion_features_invariants(
    side_features: np.ndarray,
    diff_features: np.ndarray,
    champions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Verify mathematical invariants, bounds, and side symmetry of champion features.

    Raises:
        AssertionError if any invariant is violated.

    Returns:
        Dictionary of verification summary metrics.
    """
    n_series = champions.shape[0]

    # 1. Shapes and dtypes
    assert side_features.shape == (n_series, 2, 30), f"Invalid side shape {side_features.shape}"
    assert diff_features.shape == (n_series, 30), f"Invalid diff shape {diff_features.shape}"
    assert side_features.dtype == np.float32, f"Invalid side dtype {side_features.dtype}"
    assert diff_features.dtype == np.float32, f"Invalid diff dtype {diff_features.dtype}"

    # 2. No NaNs or Infs
    assert not np.isnan(side_features).any(), "NaNs detected in side_features"
    assert not np.isinf(side_features).any(), "Infs detected in side_features"
    assert not np.isnan(diff_features).any(), "NaNs detected in diff_features"
    assert not np.isinf(diff_features).any(), "Infs detected in diff_features"

    # 3. Bounds on normalized player features (first 25 columns) and first 4 team features
    player_feats = side_features[:, :, :25]
    assert np.all(player_feats >= 0.0), f"Negative player features: {player_feats.min()}"
    assert np.all(player_feats <= 1.0 + 1e-6), f"Player features exceed 1.0: {player_feats.max()}"

    team_norm_feats = side_features[:, :, 25:29]
    assert np.all(team_norm_feats >= 0.0), f"Negative team normalized features: {team_norm_feats.min()}"
    assert np.all(team_norm_feats <= 1.0 + 1e-6), f"Team normalized features exceed 1.0: {team_norm_feats.max()}"

    team_flex = side_features[:, :, 29]
    assert np.all(team_flex >= 0.0), f"Negative flex features: {team_flex.min()}"

    # 4. Cold-start invariant: players with 0 valid games must have exactly 0.0 for all 5 features
    valid_map_count = ((mask == 1) & (champions > 0)).sum(axis=-1)  # shape (N, 2, 5)
    cold_players = (valid_map_count == 0)
    if np.any(cold_players):
        player_reshaped = side_features[:, :, :25].reshape(n_series, 2, 5, 5)
        cold_values = player_reshaped[cold_players]
        max_cold_abs = np.abs(cold_values).max()
        assert max_cold_abs == 0.0, f"Cold player features not exactly 0.0: max abs {max_cold_abs}"

    # 5. Side-swap anti-symmetry: Delta X(B, A) = - Delta X(A, B)
    champions_swapped = champions[:, [1, 0], :, :]
    mask_swapped = mask[:, [1, 0], :, :]
    _, diff_swapped = extract_champion_pool_features(champions_swapped, mask_swapped)
    anti_symm_err = float(np.abs(diff_features + diff_swapped).max())
    assert anti_symm_err < 1.0e-7, f"Anti-symmetry violation: {anti_symm_err}"

    return {
        "status": "PASSED",
        "n_series": n_series,
        "max_anti_symmetry_error": anti_symm_err,
        "player_features_min": float(player_feats.min()),
        "player_features_max": float(player_feats.max()),
        "team_flex_max": float(team_flex.max()),
        "diff_features_min": float(diff_features.min()),
        "diff_features_max": float(diff_features.max()),
    }
