"""Small role-aware residual on an already projected, frozen C0 series logit.

All features and scaling are supplied by the offline caller. This module does
not fit a base model, construct a target draft, or project a game probability to
best-of. ``best_of`` is retained in the input schema solely for audit: native
``base_probability`` already describes the full series.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class C0MetaCorrection(nn.Module):
    """Shared player/meta encoders with a bounded antisymmetric series residual.

    Fixed widths are champion=8 and player/meta=16. The parameter budget is
    ``4544 + 8 * vocabulary_size`` for either branch. Champion bin zero is a
    learned unknown-observed champion, not padding; absent observations have
    zero frequency. Inputs use the fixed historical vocabulary and contain no
    future-map draft, patch, or role assignment.

    Each player mixes the SAME global and pooled-local role embeddings using
    that player's prior known-role frequencies. The readout sees the player,
    both aligned contexts, and their two elementwise player-context products.
    Shared player scores are masked and averaged over the five roster slots.
    The only correction is ``.5 * tanh(score_A - score_B)``. Its zero-initialized
    final layer starts exactly at the native series logit, including endpoints.

    Save a standard ``state_dict`` together with ``vocabulary_size`` and
    ``use_meta``; reload with the same constructor and ``load_state_dict``.
    Feature normalization and base probabilities remain caller-owned.
    """

    def __init__(self, vocabulary_size: int, use_meta: bool) -> None:
        super().__init__()
        if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError("vocabulary_size must be a positive integer including unknown bin zero")
        if not isinstance(use_meta, bool):
            raise ValueError("use_meta must be boolean")
        self.vocabulary_size = vocabulary_size
        self.use_meta = use_meta
        self.champion_embedding = nn.Embedding(vocabulary_size, 8)
        self.player_encoder = nn.Linear(96 + 40 + 5 + 8, 16)
        # No bias: zero meta inputs produce exactly zero role embeddings, even
        # after training. Both branches nevertheless have identical topology.
        self.meta_encoder = nn.Linear(40 + 8 + 2 + 2, 16, bias=False)
        self.readout_hidden = nn.Linear(16 * 5, 16)
        self.readout = nn.Linear(16, 1, bias=False)
        nn.init.zeros_(self.readout.weight)

    def _validate_inputs(self, inputs: Mapping[str, Tensor]) -> None:
        """Check the tensor schema without device copies or per-batch scans.

        Numeric validity, frequency construction and scaling are enforced by
        the caller's feature pipeline. Unsupported player tensors may contain
        arbitrary values, including NaNs: they are sanitized before encoding.
        """
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a mapping of named tensors")
        probability = inputs.get("base_probability")
        if not isinstance(probability, Tensor) or probability.ndim != 1:
            raise ValueError("base_probability must have shape [N]")
        batch_size = probability.shape[0]
        shapes = {
            "profile": (batch_size, 2, 5, 96),
            "player_stats": (batch_size, 2, 5, 40),
            "player_roles": (batch_size, 2, 5, 5),
            "player_champions": (batch_size, 2, 5, self.vocabulary_size),
            "player_valid": (batch_size, 2, 5),
            "meta_stats": (batch_size, 2, 5, 40),
            "meta_champions": (batch_size, 2, 5, self.vocabulary_size),
            "meta_support": (batch_size, 2, 5, 2),
            "meta_patch": (batch_size, 2, 5, 2),
            "base_probability": (batch_size,),
            "best_of": (batch_size,),
        }
        weight = self.champion_embedding.weight
        for name, expected in shapes.items():
            value = inputs.get(name)
            if not isinstance(value, Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if value.device != weight.device:
                raise ValueError(f"{name} must be on model device {weight.device}")
            if name == "player_valid":
                if value.dtype != torch.bool:
                    raise ValueError("player_valid must be boolean")
            elif name == "best_of":
                if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                    raise ValueError("best_of must have integer dtype")
            elif not value.is_floating_point():
                raise ValueError(f"{name} must have floating dtype")
            elif name != "base_probability" and value.dtype != weight.dtype:
                raise ValueError(f"{name} must have model dtype {weight.dtype}")

    def _role_meta(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if self.use_meta:
            champions = inputs["meta_champions"] @ self.champion_embedding.weight
            meta_input = torch.cat((
                inputs["meta_stats"], champions, inputs["meta_support"], inputs["meta_patch"],
            ), dim=-1)
        else:
            # Do not multiply by zero: ignored meta tensors may contain NaNs.
            meta_input = inputs["meta_stats"].new_zeros((*inputs["meta_stats"].shape[:-1], 52))
        return torch.tanh(self.meta_encoder(meta_input))

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Return corrected SERIES logits, shape [N], suitable for BCE fitting."""
        self._validate_inputs(inputs)
        valid = inputs["player_valid"]
        mask = valid.unsqueeze(-1)
        profile = torch.where(mask, inputs["profile"], 0.)
        stats = torch.where(mask, inputs["player_stats"], 0.)
        roles = torch.where(mask, inputs["player_roles"], 0.)
        champions = torch.where(mask, inputs["player_champions"], 0.) @ self.champion_embedding.weight
        players = torch.tanh(self.player_encoder(torch.cat((profile, stats, roles, champions), dim=-1)))
        role_meta = self._role_meta(inputs)
        # Flatten only roster axes. bmm consumes the same role basis for both
        # teams without allocating a per-team copy of the shared context.
        mixtures = roles.flatten(1, 2)
        global_meta = torch.bmm(mixtures, role_meta[:, 0]).reshape_as(players)
        local_meta = torch.bmm(mixtures, role_meta[:, 1]).reshape_as(players)
        readout_input = torch.cat((
            players, global_meta, local_meta, players * global_meta, players * local_meta,
        ), dim=-1)
        player_scores = self.readout(torch.tanh(self.readout_hidden(readout_input))).squeeze(-1)
        player_scores = torch.where(valid, player_scores, 0.)
        # A fixed five-slot denominator preserves support coverage and avoids
        # giving a single supported player the weight of a complete roster.
        side_scores = player_scores.mean(dim=-1)
        delta = .5 * torch.tanh(side_scores[:, 0] - side_scores[:, 1])
        return torch.logit(inputs["base_probability"]) + delta

    def predict_proba(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Return P(team A wins the series), without any further BO projection."""
        return torch.sigmoid(self(inputs))

    @torch.no_grad()
    def export_role_meta(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Export detached [N, global/local, role, 16] embeddings for audit.

        The tensor stays on the model device. Call ``.cpu().numpy()`` when an
        offline report requires NumPy. Player-only exports exact zeros. This
        method uses the current learned state, not a separate fitted encoder.
        """
        self._validate_inputs(inputs)
        return self._role_meta(inputs).detach()
