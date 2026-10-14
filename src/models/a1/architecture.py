"""Research A1: player history, champion context and intrateam attention.

Champion IDs and optional external context are fused with player history before
attention over the five roles. External context provenance belongs to the caller;
this module does not establish a causal global-meta window. The signed bilateral
utility head preserves side symmetry during deterministic inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class A1Config:
    """Hyperparameter configuration for the A1 architecture."""

    # Level 1: Global Meta Champion
    meta_dim: int = 32
    num_champions: int = 256
    champ_embed_dim: int = 32

    # Level 2: Player & History
    player_feat_dim: int = 21
    history_feat_dim: int = 33
    history_len: int = 16
    player_hidden_dim: int = 64

    # Level 3: Team Transformer
    num_roles: int = 5
    team_heads: int = 4
    team_layers: int = 1
    team_ff_dim: int = 128
    dropout: float = 0.05

    # Level 4: Head & Context
    context_dim: int = 28
    head_hidden_dim: int = 64
    num_series_classes: int = 3  # Bo1, Bo3, Bo5


class ScalarRating(nn.Module):
    """Linear or residual compressor for player/team ratings."""

    def __init__(self, width: int, nonlinear: bool = False, is_player: bool = False):
        super().__init__()
        self.linear = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.linear.weight)
        if is_player:
            with torch.no_grad():
                self.linear.weight[0, 1] = 1.0
        self.residual = None
        if nonlinear:
            self.residual = nn.Sequential(
                nn.Linear(width, 16),
                nn.Tanh(),
                nn.Linear(16, 1, bias=False)
            )
            nn.init.normal_(self.residual[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out


class TeamSetTransformer(nn.Module):
    """5-node Set Transformer over the 5 competitive roles [TOP, JGL, MID, ADC, SUP]."""

    def __init__(self, d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.05):
        super().__init__()
        # Learnable role-identity position embeddings (5 roles: Top, Jungle, Mid, Bot, Support)
        self.role_embeddings = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, player_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process 5 player vectors into synergy-contextualized team representation.

        Args:
            player_vectors: Tensor of shape (batch, 5, d_model)

        Returns:
            team_pooled: Tensor of shape (batch, d_model * 2) combining mean and std.
            contextualized_players: Tensor of shape (batch, 5, d_model).
        """
        # Add role position encodings
        x = player_vectors + self.role_embeddings
        x = self.encoder(x)
        x = self.layer_norm(x)

        # Permutation-invariant summary with mean and std
        mean_pooled = x.mean(dim=1)
        std_pooled = x.std(dim=1, unbiased=False)
        team_pooled = torch.cat([mean_pooled, std_pooled], dim=-1)
        return team_pooled, x


class A1Model(nn.Module):
    """Next-Generation A1 Model Architecture with Team Attention and Global Meta Champions."""

    def __init__(self, config: A1Config | None = None):
        super().__init__()
        self.config = config or A1Config()

        # 1. Champion embeddings & Global Meta projector
        self.champion_embed = nn.Embedding(
            self.config.num_champions, self.config.champ_embed_dim, padding_idx=0
        )
        self.meta_projector = nn.Linear(self.config.meta_dim, self.config.champ_embed_dim)

        # 2. Player personal history encoder (last 16 maps attention)
        self.history_encoder = nn.Linear(self.config.history_feat_dim, 32)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.map_attention = nn.Linear(32, 1)

        # 3. Player-to-Champion Fusion Layer
        # Concatenates: ratings (21) + history (32) + champion_embed (32) + meta (32) = 117
        in_player_dim = (
            self.config.player_feat_dim
            + 32
            + self.config.champ_embed_dim
            + self.config.champ_embed_dim
        )
        self.player_fusion = nn.Sequential(
            nn.Linear(in_player_dim, self.config.player_hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.config.player_hidden_dim),
        )

        # 4. Team Set Transformer
        self.team_transformer = TeamSetTransformer(
            d_model=self.config.player_hidden_dim,
            nhead=self.config.team_heads,
            dim_feedforward=self.config.team_ff_dim,
            dropout=self.config.dropout,
        )

        # 5. Bilateral Anti-Symmetric Utility Head
        # Team representation is mean + std = 64 * 2 = 128
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
        player_champions: torch.Tensor,
        meta_champions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode 5 players on one side into their contextualized player states.

        Args:
            player_ratings: shape (batch, 5, 21)
            player_history: shape (batch, 5, 16, 33)
            player_mask: shape (batch, 5, 16)
            player_champions: shape (batch, 5, 16)
            meta_champions: shape (batch, 5, 32)

        Returns:
            player_vectors: shape (batch, 5, player_hidden_dim)
        """
        batch_size = player_ratings.shape[0]

        # Mask invalid source values before either learned projection or ID lookup:
        # masked padding may contain arbitrary IDs or non-finite feature values.
        h_flat = player_history.reshape(-1, self.config.history_len, self.config.history_feat_dim)
        m_flat = player_mask.reshape(-1, self.config.history_len).bool()
        c_flat = player_champions.reshape(-1, self.config.history_len)
        h_flat = torch.where(m_flat.unsqueeze(-1), h_flat, torch.zeros_like(h_flat))
        c_flat = c_flat.masked_fill(~m_flat, 0)

        # Linear projection + champion embedding
        encoded_maps = torch.tanh(self.history_encoder(h_flat))
        champ_emb = self.champion_embed(c_flat)
        encoded_maps = encoded_maps + champ_emb

        # Keep the learned prior token available alongside valid historical maps.
        empty_exp = self.empty_token.expand(encoded_maps.shape[0], 1, -1)
        encoded_seq = torch.cat([empty_exp, encoded_maps], dim=1)  # (batch*5, 17, 32)
        valid_mask = torch.cat([torch.ones_like(m_flat[:, :1]), m_flat], dim=1)

        # Attentive pooling excludes padded maps.
        attn_scores = self.map_attention(encoded_seq).squeeze(-1)  # (batch*5, 17)
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        history_pooled = (encoded_seq * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch*5, 32)
        history_pooled = history_pooled.reshape(batch_size, 5, 32)

        # Global meta champion projection
        meta_projected = torch.tanh(self.meta_projector(meta_champions))  # (batch, 5, 32)

        # History is stored oldest-to-newest; select the most recent valid ID.
        positions = torch.arange(self.config.history_len, device=m_flat.device)
        last_valid = torch.where(m_flat, positions.unsqueeze(0), -1).amax(dim=1)
        primary_champ_id = c_flat.gather(1, last_valid.clamp_min(0).unsqueeze(1)).squeeze(1)
        primary_champ_id = primary_champ_id.masked_fill(last_valid < 0, 0)
        primary_champ_emb = self.champion_embed(primary_champ_id).reshape(batch_size, 5, -1)

        # Fuse player components: [ratings, history, recent_champ, meta_champ]
        fusion_input = torch.cat(
            [player_ratings, history_pooled, primary_champ_emb, meta_projected], dim=-1
        )
        player_vectors = self.player_fusion(fusion_input)
        return player_vectors

    def forward(
        self,
        context: torch.Tensor,
        player_ratings: torch.Tensor,
        player_history: torch.Tensor,
        player_mask: torch.Tensor,
        player_champions: torch.Tensor,
        meta_champions: torch.Tensor,
        best_of: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate forward pass with guaranteed machine-level anti-symmetry.

        Args:
            context: shape (batch, 28) signed team, W20, gate, and core differences
            player_ratings: shape (batch, 2, 5, 21)
            player_history: shape (batch, 2, 5, 16, 33)
            player_mask: shape (batch, 2, 5, 16)
            player_champions: shape (batch, 2, 5, 16)
            meta_champions: shape (batch, 2, 5, 32)
            best_of: shape (batch,) int (1, 3, or 5)

        Returns:
            probability_a: shape (batch,) in (0, 1) strictly symmetric
        """
        if not torch.all((best_of == 1) | (best_of == 3) | (best_of == 5)):
            raise ValueError("A1 has trained series heads only for best_of 1, 3, or 5.")
        bo_idx = (best_of.long() - 1) // 2

        # 1. Encode Side A and Side B players
        players_a = self.encode_players_side(
            player_ratings[:, 0],
            player_history[:, 0],
            player_mask[:, 0],
            player_champions[:, 0],
            meta_champions[:, 0],
        )
        players_b = self.encode_players_side(
            player_ratings[:, 1],
            player_history[:, 1],
            player_mask[:, 1],
            player_champions[:, 1],
            meta_champions[:, 1],
        )

        # 2. Team Cross-Attention (Set Transformer)
        team_a_vec, _ = self.team_transformer(players_a)
        team_b_vec, _ = self.team_transformer(players_b)

        # 3. Bilateral Anti-Symmetric Utility
        diff_ab = team_a_vec - team_b_vec
        diff_ba = team_b_vec - team_a_vec

        head_in_fwd = torch.cat([context, diff_ab], dim=-1)
        head_in_rev = torch.cat([-context, diff_ba], dim=-1)
        u_fwd = self.utility_head(head_in_fwd)
        u_rev = self.utility_head(head_in_rev)

        # Exact anti-symmetry: z_sym = 0.5 * (u(A, B) - u(B, A))
        z_sym = 0.5 * (u_fwd - u_rev)

        # Select corresponding best_of logit and clamp to prevent numerical saturation
        batch_indices = torch.arange(context.shape[0], device=context.device)
        logit_selected = torch.clamp(z_sym[batch_indices, bo_idx], -30.0, 30.0)

        return torch.sigmoid(logit_selected)
