"""Build champion match-sequence training dataset using pure last N matches on role.

No arbitrary calendar day cutoff: takes the sequential stream of the last N professional maps
played on each (champion, role) pair worldwide, ordered chronologically.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ChampionSequenceDataset(Dataset):
    """Dataset yielding sequential streams of pro maps for champion representation learning."""

    def __init__(
        self,
        sequences: list[np.ndarray],     # (K, 16)
        masks: list[np.ndarray],         # (K,)
        win_rates: list[float],          # [0, 1]
        target_stats: list[np.ndarray],  # (12,)
        roles: list[int],                # 0..4
    ):
        self.sequences = sequences
        self.masks = masks
        self.win_rates = win_rates
        self.target_stats = target_stats
        self.roles = roles

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": torch.as_tensor(self.sequences[idx], dtype=torch.float32),
            "mask": torch.as_tensor(self.masks[idx], dtype=torch.bool),
            "win_rate": torch.as_tensor(self.win_rates[idx], dtype=torch.float32),
            "target_stats": torch.as_tensor(self.target_stats[idx], dtype=torch.float32),
            "role": torch.as_tensor(self.roles[idx], dtype=torch.long),
        }


def extract_champion_role_sequences(
    meta_df: pd.DataFrame,
    history_arr: np.ndarray,
    mask_arr: np.ndarray,
    champions_arr: np.ndarray,
    n_recent_maps: int = 30,
    max_samples_per_pair: int = 25,
) -> ChampionSequenceDataset:
    """Extract chronological sequence of pro maps grouped strictly by (champion, role)."""
    # Key: (champion_id, role_idx) -> list of (map_token_16, win, stats_12)
    champ_role_history = defaultdict(list)

    # Sort chronological match indices
    sorted_match_indices = np.argsort(pd.to_datetime(meta_df["date"]).values)

    stat_scales = np.array(
        [10.0, 10.0, 10.0, 10.0, 500.0, 1000.0, 3.0, 1000.0, 30.0, 1000.0, 1.0, 1.0],
        dtype=np.float32,
    )

    # Stream through historical matches chronologically
    for idx in sorted_match_indices:
        for side in (0, 1):
            for role_idx in range(5):
                m = mask_arr[idx, side, role_idx]
                valid_idx = np.where(m == 1)[0]
                if len(valid_idx) == 0:
                    continue

                champs = champions_arr[idx, side, role_idx][valid_idx]
                hists = history_arr[idx, side, role_idx][valid_idx]

                for c_id, h in zip(champs, hists):
                    if c_id == 0:
                        continue
                    stats_raw = np.nan_to_num(h[:12].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    win = float(h[24]) if len(h) > 24 else 0.5
                    opp_gl = float(h[26]) / 2000.0 if len(h) > 26 else 0.8

                    token_16 = np.zeros(16, dtype=np.float32)
                    token_16[:12] = np.clip(stats_raw / stat_scales, -5.0, 5.0)
                    token_16[12] = win
                    token_16[13] = float(role_idx) / 4.0
                    token_16[14] = np.clip(opp_gl, 0.5, 1.5)

                    champ_role_history[(int(c_id), role_idx)].append((token_16, win, token_16[:12]))

    sequences = []
    masks = []
    win_rates = []
    target_stats = []
    roles = []

    # Assemble sequential sliding windows of length n_recent_maps
    for (c_id, r_idx), map_list in champ_role_history.items():
        if len(map_list) < 5:
            continue

        step = max(5, len(map_list) // max_samples_per_pair)
        for i in range(0, len(map_list), step):
            # Take the window of maps up to i (pure historical context)
            window = map_list[max(0, i - n_recent_maps) : i]
            if len(window) < 3:
                continue

            seq_arr = np.zeros((n_recent_maps, 16), dtype=np.float32)
            mask = np.zeros(n_recent_maps, dtype=bool)

            wins = []
            stats = []
            for j, (t, w, s) in enumerate(window):
                seq_arr[j] = t
                mask[j] = True
                wins.append(w)
                stats.append(s)

            sequences.append(seq_arr)
            masks.append(mask)
            win_rates.append(float(np.mean(wins)))
            target_stats.append(np.mean(stats, axis=0).astype(np.float32))
            roles.append(r_idx)

    print(f"Extracted {len(sequences)} pure sequential windows across {len(champ_role_history)} (champion, role) pairs.")
    return ChampionSequenceDataset(sequences, masks, win_rates, target_stats, roles)
