"""Unit tests for Dynamic Champion Encoder and multi-task heads."""

import pytest
import torch
import torch.nn.functional as F

from src.models.champion_encoder.model import (
    DynamicChampionEncoder,
    ChampionEncoderConfig,
    MultiTaskPretrainingHead,
)


def test_dynamic_champion_encoder_shapes():
    """Encoder must output normalized embeddings of shape (batch, latent_dim)."""
    batch_size = 4
    num_maps = 20
    map_dim = 16
    latent_dim = 32

    config = ChampionEncoderConfig(map_feature_dim=map_dim, latent_dim=latent_dim)
    encoder = DynamicChampionEncoder(config)
    encoder.eval()

    inputs = torch.randn(batch_size, num_maps, map_dim)
    mask = torch.ones(batch_size, num_maps, dtype=torch.bool)
    mask[:, 15:] = False  # last 5 maps padded

    with torch.no_grad():
        embeddings = encoder(inputs, sequence_mask=mask)

    assert embeddings.shape == (batch_size, latent_dim)
    # L2-normalized: norms should be exactly 1.0
    norms = torch.norm(embeddings, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)


def test_multitask_pretraining_heads():
    """MultiTaskPretrainingHead must produce win_logits, reconstruction, and contrast_proj."""
    batch_size = 8
    latent_dim = 32
    stat_dim = 12

    heads = MultiTaskPretrainingHead(latent_dim=latent_dim, stat_dim=stat_dim)
    heads.eval()

    embeddings = torch.randn(batch_size, latent_dim)
    out = heads(embeddings)

    assert "win_logit" in out and out["win_logit"].shape == (batch_size,)
    assert "reconstruction" in out and out["reconstruction"].shape == (batch_size, stat_dim)
    assert "contrast_proj" in out and out["contrast_proj"].shape == (batch_size, 32)
