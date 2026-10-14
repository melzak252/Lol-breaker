"""Research A1 consuming caller-supplied dynamic champion embeddings.

This variant has no champion-ID embedding table. It does not itself train or run
the upstream champion encoder, nor establish the supplied embeddings' chronology.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from src.models.a1.architecture import TeamSetTransformer


@dataclass(frozen=True)
class A1DynamicConfig:
    """Configuration for A1 with pure dynamic champion embeddings."""

    dynamic_champ_dim: int = 32
    player_feat_dim: int = 21
    history_feat_dim: int = 33
    history_len: int = 16
    player_hidden_dim: int = 64
    num_roles: int = 5
    team_heads: int = 4
    team_ff_dim: int = 128
    dropout: float = 0.05
    context_dim: int = 28
    head_hidden_dim: int = 64
    num_series_classes: int = 3


class A1DynamicModel(nn.Module):
    """A1 Architecture with pure dynamic inductive champion embeddings (zero static ID table)."""

    def __init__(self, config: A1DynamicConfig | None = None):
        super().__init__()
        self.config = config or A1DynamicConfig()

        # 1. Player personal history encoder (16 maps attention)
        self.history_encoder = nn.Linear(self.config.history_feat_dim, 32)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.map_attention = nn.Linear(32, 1)

        # 2. Dynamic Champion Projector (maps the 32-dim dynamic vector)
        self.dynamic_projector = nn.Sequential(
            nn.Linear(self.config.dynamic_champ_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )

        # 3. Player-to-Champion Fusion Layer:
        # Concatenates: ratings (21) + history (32) + dynamic_champ (32) = 85 features
        in_player_dim = self.config.player_feat_dim + 32 + 32
        self.player_fusion = nn.Sequential(
            nn.Linear(in_player_dim, self.config.player_hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.config.player_hidden_dim),
        )

        # 4. 5-node Team Set Transformer (Intra-team synergy attention)
        self.team_transformer = TeamSetTransformer(
            d_model=self.config.player_hidden_dim,
            nhead=self.config.team_heads,
            dim_feedforward=self.config.team_ff_dim,
            dropout=self.config.dropout,
        )

        # 5. Bilateral Anti-Symmetric Utility Head
        in_head_dim = self.config.context_dim + (self.config.player_hidden_dim * 2)
        self.utility_head = nn.Sequential(
            nn.Linear(in_head_dim, self.config.head_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.head_hidden_dim, self.config.num_series_classes, bias=False),
        )
        nn.init.normal_(self.utility_head[-1].weight, std=0.01)

    def encode_players_side(
        self,
        player_ratings: torch.Tensor,
        player_history: torch.Tensor,
        player_mask: torch.Tensor,
        dynamic_champions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 5 players on one side using their ratings, history, and dynamic champion embeddings.

        Args:
            player_ratings: shape (batch, 5, 21)
            player_history: shape (batch, 5, 16, 33)
            player_mask: shape (batch, 5, 16)
            dynamic_champions: shape (batch, 5, 32) dynamic embedding from Transformer

        Returns:
            player_vectors: shape (batch, 5, player_hidden_dim)
        """
        batch_size = player_ratings.shape[0]

        # Mask invalid source values before projection so padded NaNs cannot leak.
        h_flat = player_history.reshape(-1, self.config.history_len, self.config.history_feat_dim)
        m_flat = player_mask.reshape(-1, self.config.history_len).bool()
        h_flat = torch.where(m_flat.unsqueeze(-1), h_flat, torch.zeros_like(h_flat))

        encoded_maps = torch.tanh(self.history_encoder(h_flat))
        empty_exp = self.empty_token.expand(encoded_maps.shape[0], 1, -1)
        encoded_seq = torch.cat([empty_exp, encoded_maps], dim=1)
        valid_mask = torch.cat([torch.ones_like(m_flat[:, :1]), m_flat], dim=1)

        attn_scores = self.map_attention(encoded_seq).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        history_pooled = (encoded_seq * attn_weights.unsqueeze(-1)).sum(dim=1).reshape(batch_size, 5, 32)

        # Dynamic Champion projection
        dyn_proj = self.dynamic_projector(dynamic_champions)  # (batch, 5, 32)

        # Fuse player components: [ratings, history, dynamic_champion]
        fusion_input = torch.cat([player_ratings, history_pooled, dyn_proj], dim=-1)
        return self.player_fusion(fusion_input)

    def forward(
        self,
        context: torch.Tensor,
        player_ratings: torch.Tensor,
        player_history: torch.Tensor,
        player_mask: torch.Tensor,
        dynamic_champions: torch.Tensor,
        best_of: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate forward pass with guaranteed machine-level anti-symmetry."""
        if not torch.all((best_of == 1) | (best_of == 3) | (best_of == 5)):
            raise ValueError("A1 has trained series heads only for best_of 1, 3, or 5.")
        bo_idx = (best_of.long() - 1) // 2

        # Encode side A and side B players
        players_a = self.encode_players_side(
            player_ratings[:, 0],
            player_history[:, 0],
            player_mask[:, 0],
            dynamic_champions[:, 0],
        )
        players_b = self.encode_players_side(
            player_ratings[:, 1],
            player_history[:, 1],
            player_mask[:, 1],
            dynamic_champions[:, 1],
        )

        # Team Set Transformer
        team_a_vec, _ = self.team_transformer(players_a)
        team_b_vec, _ = self.team_transformer(players_b)

        diff_ab = team_a_vec - team_b_vec
        diff_ba = team_b_vec - team_a_vec

        head_in_fwd = torch.cat([context, diff_ab], dim=-1)
        head_in_rev = torch.cat([-context, diff_ba], dim=-1)

        u_fwd = self.utility_head(head_in_fwd)
        u_rev = self.utility_head(head_in_rev)

        # Anti-symmetric logit
        z_sym = 0.5 * (u_fwd - u_rev)

        batch_indices = torch.arange(context.shape[0], device=context.device)
        logit_selected = torch.clamp(z_sym[batch_indices, bo_idx], -30.0, 30.0)

        return torch.sigmoid(logit_selected)
