"""Self-Supervised Dynamic Champion Encoder Architecture.

Encodes an arbitrary sequence of K recent pro maps played on a (champion, role) pair
into a rich d-dimensional latent representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ChampionEncoderConfig:
    """Hyperparameters for the self-supervised dynamic champion encoder."""

    map_feature_dim: int = 16
    latent_dim: int = 32
    num_heads: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.05
    max_history_maps: int = 50


class DynamicChampionEncoder(nn.Module):
    """Transformer-based dynamic champion encoder."""

    def __init__(self, config: ChampionEncoderConfig | None = None):
        super().__init__()
        self.config = config or ChampionEncoderConfig()

        # 1. Project input map statistics to model hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.map_feature_dim, self.config.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.latent_dim),
        )

        # 2. Learned context query token (CLS token equivalent for summary)
        self.summary_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim) * 0.02)

        # 3. Transformer Encoder over recent maps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.latent_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_encoder_layers
        )

        self.out_norm = nn.LayerNorm(self.config.latent_dim)

    def forward(
        self,
        map_sequences: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a sequence of pro maps for a champion into a single latent vector.

        Args:
            map_sequences: shape (batch, num_maps, map_feature_dim)
            sequence_mask: shape (batch, num_maps) boolean mask where True indicates valid map

        Returns:
            embedding: shape (batch, latent_dim) normalized representation
        """
        batch_size = map_sequences.shape[0]

        # Project map tokens
        proj_maps = self.input_projection(map_sequences)  # (batch, num_maps, latent_dim)

        # Prepend summary query token
        summary_exp = self.summary_token.expand(batch_size, -1, -1)  # (batch, 1, latent_dim)
        full_seq = torch.cat([summary_exp, proj_maps], dim=1)  # (batch, num_maps + 1, latent_dim)

        # Build key padding mask for transformer (True = ignored/padded)
        if sequence_mask is not None:
            # First token (summary) is always valid (~False)
            summary_valid = torch.zeros((batch_size, 1), dtype=torch.bool, device=map_sequences.device)
            key_padding = torch.cat([summary_valid, ~sequence_mask.bool()], dim=1)
        else:
            key_padding = None

        encoded_seq = self.transformer(full_seq, src_key_padding_mask=key_padding)

        # Extract the summary token vector at index 0
        latent_vector = self.out_norm(encoded_seq[:, 0])

        # L2-normalize for clean metric space
        return F.normalize(latent_vector, p=2, dim=-1)


class MultiTaskPretrainingHead(nn.Module):
    """Auxiliary multi-task pretraining heads for self-supervised representation learning.

    1. Win-rate classification head: BCE loss
    2. Boxscore statistics reconstruction head: MSE loss (autoencoder denoising)
    3. Contrastive projector: InfoNCE projection
    """

    def __init__(self, latent_dim: int = 32, stat_dim: int = 12, contrast_dim: int = 32):
        super().__init__()
        self.win_head = nn.Linear(latent_dim, 1)
        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, stat_dim),
        )
        self.contrast_projector = nn.Sequential(
            nn.Linear(latent_dim, contrast_dim),
            nn.GELU(),
            nn.Linear(contrast_dim, contrast_dim),
        )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        win_logit = self.win_head(embedding).squeeze(-1)
        reconstruction = self.recon_head(embedding)
        contrastive_proj = F.normalize(self.contrast_projector(embedding), p=2, dim=-1)
        return {
            "win_logit": win_logit,
            "reconstruction": reconstruction,
            "contrast_proj": contrastive_proj,
        }
