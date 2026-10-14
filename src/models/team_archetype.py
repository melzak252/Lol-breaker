"""Team Archetype and Macro Playstyle Vector Modeling for Competitive LoL.

Computes an 8-dimensional continuous macroeconomic playstyle vector for each team:
1. Pace / Bloodiness (kills + deaths per minute)
2. Scaling Bias (game duration normalized)
3. Early Lane Dominance (GD@15 in k gold)
4. Objective Priority (Dragon / Nashor vs Turret bias)
5. Vision & Map Control Intensity (VSPM normalized)
6. Combat Damage Output (DPM normalized)
7. Role Focus Skew (Top vs Bot rating differential)
8. Gold Conversion Efficiency (DPM per 1k gold)

Guarantees exact bilateral anti-symmetry:
StyleClash(A, B) = -StyleClash(B, A).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def compute_team_archetype_vectors(
    w20_arr: np.ndarray,
    players_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 8-dimensional continuous archetype vectors for Team A and Team B.

    Args:
        w20_arr: shape (N, 2, 10) rolling W20 match stats
        players_arr: shape (N, 2, 5, 21) player skill & stats

    Returns:
        archetype_a: shape (N, 8)
        archetype_b: shape (N, 8)
    """
    def _extract_side(w20, pl):
        duration_min = np.clip(w20[:, 9] / 60.0, 20.0, 50.0)

        # 1. Pace / Bloodiness (kills + deaths per minute)
        pace = (w20[:, 1] + w20[:, 2]) / duration_min

        # 2. Scaling bias (game duration relative to 30 min)
        scaling = (duration_min - 30.0) / 10.0

        # 3. Early lane dominance (GD15 in k gold)
        early_dominance = w20[:, 5] / 1000.0

        # 4. Objective priority (dragons per tower)
        dragons = np.clip(w20[:, 7], 0.0, 4.0)
        towers = np.clip(w20[:, 6], 1.0, 11.0)
        obj_bias = dragons / towers

        # 5. Vision control intensity (VSPM relative to 6.5)
        vision = (w20[:, 4] - 6.5) / 2.0

        # 6. Combat damage output (DPM in thousands)
        dpm_norm = (w20[:, 3] - 2500.0) / 1000.0

        # 7. Top vs Bot rating skew: (Top - Bot) / 200
        top_rating = pl[:, 0, 0]  # Top Glicko
        bot_rating = pl[:, 3, 0]  # ADC Glicko
        role_skew = (top_rating - bot_rating) / 200.0

        # 8. Gold conversion efficiency (DPM per 1k gold)
        gold_k = np.clip(w20[:, 8] / 1000.0, 30.0, 100.0)
        gold_eff = (w20[:, 3] / gold_k - 50.0) / 20.0

        archetype = np.stack([
            np.nan_to_num(pace, nan=0.8),
            np.nan_to_num(scaling, nan=0.0),
            np.nan_to_num(early_dominance, nan=0.0),
            np.nan_to_num(obj_bias, nan=0.3),
            np.nan_to_num(vision, nan=0.0),
            np.nan_to_num(dpm_norm, nan=0.0),
            np.nan_to_num(role_skew, nan=0.0),
            np.nan_to_num(gold_eff, nan=0.0),
        ], axis=-1).astype(np.float32)
        return archetype

    arch_a = _extract_side(w20_arr[:, 0], players_arr[:, 0])
    arch_b = _extract_side(w20_arr[:, 1], players_arr[:, 1])
    return arch_a, arch_b


class TeamStyleClashLayer(nn.Module):
    """Neural projection layer for style clash between two team archetype vectors."""

    def __init__(self, in_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.normal_(self.proj[-1].weight, std=0.01)

    def forward(self, arch_a: torch.Tensor, arch_b: torch.Tensor) -> torch.Tensor:
        """Evaluate bilateral anti-symmetric style clash logit offset.

        Args:
            arch_a: shape (batch, 8)
            arch_b: shape (batch, 8)

        Returns:
            logit_offset: shape (batch, 1) strictly anti-symmetric
        """
        diff_ab = arch_a - arch_b
        diff_ba = arch_b - arch_a
        u_ab = self.proj(diff_ab)
        u_ba = self.proj(diff_ba)
        # Exact anti-symmetry: f(A, B) = 0.5 * (u(A, B) - u(B, A))
        return 0.5 * (u_ab - u_ba)
