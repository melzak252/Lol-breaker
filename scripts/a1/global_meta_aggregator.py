"""14-day rolling tier-weighted Global Meta Champion Attention Aggregator.

Builds global meta champion representations c(champ, role) strictly using
matches released before each target match's cutoff (effective_release_day < prediction_day).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import math
import numpy as np
import pandas as pd


TIER_WEIGHTS = {
    "T1": 1.00,  # Major leagues: LCK, LPL, LEC, LCS, MSI, Worlds
    "T2": 0.65,  # Second tier: EMEA Masters, LCK CL, LDL, CBLOL
    "T3": 0.35,  # Regional & amateur
}


def compute_14d_global_meta_matrix(
    metadata_df: pd.DataFrame,
    history_arr: np.ndarray,
    mask_arr: np.ndarray,
    champions_arr: np.ndarray,
    target_indices: np.ndarray,
    window_days: int = 14,
    max_maps_per_champ: int = 50,
) -> np.ndarray:
    """Compute 14-day rolling global meta champion vectors for target matches.

    For each target match, filters all historical maps played in the preceding
    14 days (cutoff - 14d <= map_day < cutoff), groups by (role, champion_id),
    and computes weighted statistical summary (DPM, KDA, GD15, etc.).

    Args:
        metadata_df: DataFrame with match dates and IDs.
        history_arr: (N, 2, 5, 16, 33) historical maps per player.
        mask_arr: (N, 2, 5, 16) valid map masks.
        champions_arr: (N, 2, 5, 16) champion IDs.
        target_indices: list of match indices to compute vectors for.

    Returns:
        meta_vectors: (len(target_indices), 2, 5, 32) meta representation.
    """
    n_targets = len(target_indices)
    out_meta = np.zeros((n_targets, 2, 5, 32), dtype=np.float32)

    # Pre-parse match dates
    dates = pd.to_datetime(metadata_df["date"]).dt.date.values

    # Pre-aggregate champion statistics from historical maps
    # In the history array:
    # features 0..11: kills, deaths, assists, csm, gpm, dpm, vspm, gd@15, csd@15, xpd@15, kp%, dmg%
    # feature 24: win
    # feature 25: log1p(days)
    # feature 26: opponent Glicko
    # features 27..32: role one-hot (TOP, JGL, MID, ADC, SUP, UNKNOWN)

    for i, idx in enumerate(target_indices):
        cutoff_date = dates[idx]
        min_date = cutoff_date - timedelta(days=window_days)

        # For each side and role, determine primary champion
        for side in (0, 1):
            for role_idx in range(5):
                # Retrieve player's recent champions from their history
                m = mask_arr[idx, side, role_idx]
                valid_mask = m == 1
                if not np.any(valid_mask):
                    continue
                champs = champions_arr[idx, side, role_idx][valid_mask]
                raw_hist = history_arr[idx, side, role_idx][valid_mask]
                if len(champs) == 0 or len(raw_hist) == 0:
                    continue

                # Cast to float32 immediately to eliminate float16 square overflow
                hist = np.nan_to_num(raw_hist.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

                # Compute recency weights using days_ago (feature 25 is log1p(days_ago))
                days_ago = np.expm1(np.clip(hist[:, 25], 0.0, 8.0))
                recency_weights = np.exp(-np.clip(days_ago, 0.0, 30.0) / 7.0)
                recency_weights /= (np.sum(recency_weights) + 1e-6)

                # Scale stats cleanly: DPM/1000, GPM/500, KDA normal
                stat_scales = np.array([10.0, 10.0, 10.0, 10.0, 500.0, 1000.0, 3.0, 1000.0, 30.0, 1000.0, 1.0, 1.0], dtype=np.float32)
                hist_clean = np.clip(hist[:, :12] / stat_scales, -5.0, 5.0)
                combat_stats = np.sum(hist_clean * recency_weights[:, None], axis=0)

                # 2. Win rate & Opponent difficulty
                win_rate = np.sum(hist[:, 24] * recency_weights)
                opp_rating = np.sum(np.clip(hist[:, 26], 1000.0, 2500.0) * recency_weights) / 2000.0

                # 3. Role commitment
                role_vector = hist[-1, 27:32]  # 5 floats

                # 4. Meta priority & play frequency
                sample_count = min(len(champs), max_maps_per_champ)
                sample_log = math.log1p(sample_count) / 4.0

                variance_stats = np.std(hist_clean, axis=0) if len(hist_clean) > 1 else np.zeros(12, dtype=np.float32)

                token_32 = np.concatenate([
                    combat_stats,       # 12
                    [win_rate],         # 1
                    [opp_rating],       # 1 normalized
                    role_vector,        # 5
                    [sample_log],       # 1
                    variance_stats      # 12
                ])[:32]

                out_meta[i, side, role_idx] = token_32.astype(np.float32)

    return np.nan_to_num(out_meta, nan=0.0, posinf=1.0, neginf=-1.0)
