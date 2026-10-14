"""Research residual wrapper around an explicitly supplied Causal A0 series forecast.

The macro MLP requires fitted weights. Optional lane differences are a fixed
linear correction, not cross-attention. Tier scaling is a research parameter;
this module does not load A0, certify artifacts, or establish production fitness.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


class AntiSymmetricMacroMLP(nn.Module):
    """Anti-symmetric 2-layer MLP projecting macroeconomic team differentials to logit offsets."""

    def __init__(self, in_dim: int = 7, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, diff_x: torch.Tensor) -> torch.Tensor:
        # Machine-level anti-symmetry: f(x) = 0.5 * (net(x) - net(-x))
        return 0.5 * (self.net(diff_x) - self.net(-diff_x))


class A1PredictorEngine:
    """Apply explicit research residual weights to Causal A0 series probabilities.

    The caller owns checkpoint/fit provenance. Supplying a PyTorch module is not
    proof of training; this research API is not an operational model loader.
    """

    def __init__(
        self,
        macro_mlp: AntiSymmetricMacroMLP | None = None,
        w_lane: np.ndarray | None = None,
        tier_scale: float = 0.94,
    ):
        if macro_mlp is None:
            raise ValueError("A1 requires explicitly supplied macro weights; implicit random initialization is forbidden.")
        if not math.isfinite(tier_scale) or tier_scale <= 0:
            raise ValueError("tier_scale must be finite and positive.")
        self.macro_mlp = macro_mlp
        self.macro_mlp.eval()
        # Historical development recipe; not independently validated coefficients.
        self.w_lane = w_lane if w_lane is not None else np.array(
            [-0.00030, -0.00040, +0.00010, +0.00020, -0.00010], dtype=np.float32
        )
        self.tier_scale = tier_scale

    @staticmethod
    def extract_macro_diff(w20_a: np.ndarray, w20_b: np.ndarray) -> np.ndarray:
        """Compute normalized 7-dimensional macroeconomic differential vector.

        Features:
        0: Duration diff (600s units)
        1: Early GD15 diff (1000g units)
        2: Death diff (10 deaths units)
        3: Gold diff (10000g units)
        4: Tower diff (5 towers units)
        5: Drake diff (2 drakes units)
        6: Kills diff (10 kills units)
        """
        if w20_a.ndim != 2 or w20_a.shape[1] != 10 or w20_b.shape != w20_a.shape:
            raise ValueError("W20 inputs must have matching (N, 10) research-bank shapes.")
        if not np.isfinite(w20_a).all() or not np.isfinite(w20_b).all():
            raise ValueError("W20 inputs must be finite.")
        dur_diff = (w20_a[:, 9] - w20_b[:, 9]) / 600.0
        gd15_diff = (w20_a[:, 5] - w20_b[:, 5]) / 1000.0
        death_diff = (w20_a[:, 2] - w20_b[:, 2]) / 10.0
        gold_diff = (w20_a[:, 8] - w20_b[:, 8]) / 10000.0
        tower_diff = (w20_a[:, 6] - w20_b[:, 6]) / 5.0
        drake_diff = (w20_a[:, 7] - w20_b[:, 7]) / 2.0
        kills_diff = (w20_a[:, 1] - w20_b[:, 1]) / 10.0

        return np.stack(
            [dur_diff, gd15_diff, death_diff, gold_diff, tower_diff, drake_diff, kills_diff],
            axis=1,
        ).astype(np.float32)

    def predict(
        self,
        p_a0: np.ndarray,
        w20_a: np.ndarray,
        w20_b: np.ndarray,
        players_a: np.ndarray | None = None,
        players_b: np.ndarray | None = None,
        competition_tiers: np.ndarray | None = None,
        regional_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a series forecast using the research-bank W20 schema.

        Args:
            p_a0: shape (N,) baseline Causal A0 SERIES probabilities in [0, 1].
            w20_a: shape (N, 10), bank ordering; GD15=5, drakes=7, duration=9.
            w20_b: same schema, not the application's differently ordered W20.
            players_a: optional shape (N, 5, 21) player stats for Team A
            players_b: optional shape (N, 5, 21) player stats for Team B
            competition_tiers: optional shape (N,) array of tier strings ('major', 'development', etc.)

        Returns:
            p_a1: shape (N,) calibrated A1 win probabilities for Team A
        """
        p_a0 = np.asarray(p_a0)
        if p_a0.ndim != 1 or len(p_a0) != len(w20_a):
            raise ValueError("A0 probabilities must have shape (N,) matching W20 rows.")
        if not np.isfinite(p_a0).all() or np.any((p_a0 < 0) | (p_a0 > 1)):
            raise ValueError("A0 probabilities must be finite and in [0, 1].")
        p_clamped = np.clip(p_a0, 1e-6, 1.0 - 1e-6)
        z_a0 = np.log(p_clamped / (1.0 - p_clamped))

        # 1. P0: Team Macro MLP offset
        X_macro = self.extract_macro_diff(w20_a, w20_b)
        with torch.no_grad():
            parameter = next(self.macro_mlp.parameters())
            t_macro = torch.as_tensor(X_macro, device=parameter.device, dtype=parameter.dtype)
            macro_output = self.macro_mlp(t_macro).squeeze(-1).cpu()
            if macro_output.dtype == torch.bfloat16:
                macro_output = macro_output.float()
            z_macro = macro_output.numpy()

        # P2 is a linear lane-rating difference, not opponent attention.
        if players_a is not None and players_b is not None:
            lane_diffs = (players_a[:, :, 0] - players_b[:, :, 0]) / 200.0  # (N, 5)
            z_matchup = lane_diffs @ self.w_lane
        else:
            z_matchup = 0.0

        # Combine offsets
        z_combined = z_a0 + z_macro + z_matchup

        # 3. Tier-Conditioned Parity Scaling
        if competition_tiers is not None:
            is_major_or_dev = np.isin(competition_tiers, ["major", "development"])
            z_combined[is_major_or_dev] *= self.tier_scale

        # 4. Regional Family Scaling (cross-regional tournament competition)
        if regional_offsets is not None:
            z_combined += regional_offsets

        p_a1 = 1.0 / (1.0 + np.exp(-z_combined))
        return np.clip(p_a1, 1e-6, 1.0 - 1e-6)
