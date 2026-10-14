"""Fixed-window C0 correction with explicit player activity and native support.

The offline caller supplies TRAIN-normalized 180-day player summaries, the
unchanged 30-day meta inputs, and eight activity channels. Only the first three
activity channels are scaled; the five support/identity flags remain binary.
No feature fitting, native-model training, or best-of projection occurs here.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from src.models.c0_meta import C0MetaCorrection


class StableC0MetaCorrection(C0MetaCorrection):
    """Minimal activity-aware extension of the frozen-series correction.

    Activity channels, in order, are log1p(recent-30 count), log1p(long-180
    count), log1p(days since last prior played), identity-history-known,
    recent-supported, long-supported, native-supported, and identity-known.
    Unknown inactivity is zero with identity-history-known false. A profile is
    usable only when native-supported is true, independently of long support.
    ``player_valid`` denotes long-supported OR native-supported; known identity
    alone cannot activate a correction.

    Champion width 8, hidden widths 16, shared role-meta encoder, five-slot
    pooling, and the bounded antisymmetric series residual are unchanged. The
    only additional learned weights are the eight activity inputs to the player
    encoder. Both branches retain identical topology and player support.

    Save ``state_dict``, ``vocabulary_size`` and ``use_meta`` and reload with
    this constructor. Normalization remains caller-owned. Role-meta export and
    series probability prediction use the inherited public interfaces.
    """

    def __init__(self, vocabulary_size: int, use_meta: bool) -> None:
        super().__init__(vocabulary_size, use_meta)
        self.player_encoder = nn.Linear(96 + 40 + 5 + 8 + 8, 16)

    def _validate_inputs(self, inputs: Mapping[str, Tensor]) -> None:
        super()._validate_inputs(inputs)
        expected = (inputs["base_probability"].shape[0], 2, 5, 8)
        activity = inputs.get("player_activity")
        if not isinstance(activity, Tensor) or tuple(activity.shape) != expected:
            raise ValueError(f"player_activity must have shape {expected}")
        weight = self.champion_embedding.weight
        if activity.device != weight.device:
            raise ValueError(f"player_activity must be on model device {weight.device}")
        if not activity.is_floating_point():
            raise ValueError("player_activity must have floating dtype")
        if activity.dtype != weight.dtype:
            raise ValueError(f"player_activity must have model dtype {weight.dtype}")

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Return corrected SERIES logits, retaining base when no player is usable."""
        self._validate_inputs(inputs)
        valid = inputs["player_valid"]
        mask = valid.unsqueeze(-1)
        activity = torch.where(mask, inputs["player_activity"], 0.)
        native_mask = mask & (activity[..., 6:7] == 1.)
        # Long support can activate player summaries without activating an
        # unsupported native profile, whose learned bias may be nonzero or NaN.
        profile = torch.where(native_mask, inputs["profile"], 0.)
        stats = torch.where(mask, inputs["player_stats"], 0.)
        roles = torch.where(mask, inputs["player_roles"], 0.)
        champions = torch.where(mask, inputs["player_champions"], 0.) @ self.champion_embedding.weight
        players = torch.tanh(self.player_encoder(torch.cat((
            profile, stats, roles, champions, activity,
        ), dim=-1)))
        role_meta = self._role_meta(inputs)
        mixtures = roles.flatten(1, 2)
        global_meta = torch.bmm(mixtures, role_meta[:, 0]).reshape_as(players)
        local_meta = torch.bmm(mixtures, role_meta[:, 1]).reshape_as(players)
        readout_input = torch.cat((
            players, global_meta, local_meta, players * global_meta, players * local_meta,
        ), dim=-1)
        player_scores = self.readout(torch.tanh(self.readout_hidden(readout_input))).squeeze(-1)
        player_scores = torch.where(valid, player_scores, 0.)
        side_scores = player_scores.mean(dim=-1)
        delta = .5 * torch.tanh(side_scores[:, 0] - side_scores[:, 1])
        return torch.logit(inputs["base_probability"]) + delta
