"""Unit test for A1DynamicModel anti-symmetry and shape invariants without ID embeddings."""

import torch
from src.models.a1.architecture_dynamic import A1DynamicModel, A1DynamicConfig


def test_a1_dynamic_model_forward_and_symmetry():
    """A1DynamicModel must evaluate without ID embeddings and maintain exact anti-symmetry."""
    batch_size = 4
    config = A1DynamicConfig()
    model = A1DynamicModel(config)
    model.eval()

    context = torch.randn(batch_size, config.context_dim)
    player_ratings = torch.randn(batch_size, 2, 5, 21)
    player_history = torch.randn(batch_size, 2, 5, 16, 33)
    player_mask = torch.ones(batch_size, 2, 5, 16, dtype=torch.uint8)
    dynamic_champions = torch.randn(batch_size, 2, 5, 32)
    best_of = torch.tensor([1, 3, 5, 3], dtype=torch.long)

    with torch.no_grad():
        p_ab = model(
            context=context,
            player_ratings=player_ratings,
            player_history=player_history,
            player_mask=player_mask,
            dynamic_champions=dynamic_champions,
            best_of=best_of,
        )

        p_ba = model(
            context=-context,
            player_ratings=player_ratings.flip(dims=[1]),
            player_history=player_history.flip(dims=[1]),
            player_mask=player_mask.flip(dims=[1]),
            dynamic_champions=dynamic_champions.flip(dims=[1]),
            best_of=best_of,
        )

    assert p_ab.shape == (batch_size,)
    assert torch.all(p_ab >= 0.0) and torch.all(p_ab <= 1.0)
    assert torch.allclose(p_ab + p_ba, torch.ones(batch_size), atol=1e-5)


def test_masked_nonfinite_history_uses_empty_token_without_leaking_nan():
    model = A1DynamicModel(A1DynamicConfig(dropout=0.0)).eval()
    ratings = torch.zeros(1, 5, 21)
    history = torch.zeros(1, 5, 16, 33)
    mask = torch.zeros(1, 5, 16, dtype=torch.bool)
    dynamic_champions = torch.zeros(1, 5, 32)

    with torch.no_grad():
        expected = model.encode_players_side(ratings, history, mask, dynamic_champions)
        history[:, 0] = torch.nan
        actual = model.encode_players_side(ratings, history, mask, dynamic_champions)

    assert torch.equal(actual, expected)
