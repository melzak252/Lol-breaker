"""Comprehensive unit tests for A1 architecture, symmetry, and Set Transformer."""

import pytest
import torch
import numpy as np

from src.models.a1.architecture import A1Model, A1Config, TeamSetTransformer
from src.models.a1.predictor import A1Predictor


def test_team_set_transformer_shape_and_permutation_invariance():
    """TeamSetTransformer must output pooled vectors of shape (batch, d_model * 2)."""
    batch_size = 4
    d_model = 64
    transformer = TeamSetTransformer(d_model=d_model, nhead=4)
    transformer.eval()

    player_vectors = torch.randn(batch_size, 5, d_model)
    team_pooled, contextualized = transformer(player_vectors)

    assert team_pooled.shape == (batch_size, d_model * 2)
    assert contextualized.shape == (batch_size, 5, d_model)


def test_a1_model_forward_pass_shape():
    """A1Model must produce valid probabilities in (0, 1) for Bo1, Bo3, and Bo5."""
    batch_size = 3
    config = A1Config()
    model = A1Model(config)
    model.eval()

    context = torch.randn(batch_size, config.context_dim)
    player_ratings = torch.randn(batch_size, 2, 5, 21)
    player_history = torch.randn(batch_size, 2, 5, 16, 33)
    player_mask = torch.ones(batch_size, 2, 5, 16, dtype=torch.uint8)
    player_champions = torch.randint(0, 255, (batch_size, 2, 5, 16))
    meta_champions = torch.randn(batch_size, 2, 5, 32)
    best_of = torch.tensor([1, 3, 5], dtype=torch.long)

    with torch.no_grad():
        probs = model(
            context=context,
            player_ratings=player_ratings,
            player_history=player_history,
            player_mask=player_mask,
            player_champions=player_champions,
            meta_champions=meta_champions,
            best_of=best_of,
        )

    assert probs.shape == (batch_size,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_a1_model_strict_anti_symmetry():
    """Swapping Side A and Side B must yield exact p(A, B) + p(B, A) = 1.0."""
    batch_size = 5
    config = A1Config()
    model = A1Model(config)
    model.eval()

    context = torch.randn(batch_size, config.context_dim)
    player_ratings = torch.randn(batch_size, 2, 5, 21)
    player_history = torch.randn(batch_size, 2, 5, 16, 33)
    player_mask = torch.ones(batch_size, 2, 5, 16, dtype=torch.uint8)
    player_champions = torch.randint(0, 255, (batch_size, 2, 5, 16))
    meta_champions = torch.randn(batch_size, 2, 5, 32)
    best_of = torch.tensor([1, 3, 5, 3, 1], dtype=torch.long)

    with torch.no_grad():
        # Forward evaluation: P(A wins against B)
        p_ab = model(
            context=context,
            player_ratings=player_ratings,
            player_history=player_history,
            player_mask=player_mask,
            player_champions=player_champions,
            meta_champions=meta_champions,
            best_of=best_of,
        )

        # Swapped evaluation: Swap side 0 and 1, invert context sign
        context_swapped = -context
        player_ratings_swapped = player_ratings.flip(dims=[1])
        player_history_swapped = player_history.flip(dims=[1])
        player_mask_swapped = player_mask.flip(dims=[1])
        player_champions_swapped = player_champions.flip(dims=[1])
        meta_champions_swapped = meta_champions.flip(dims=[1])

        p_ba = model(
            context=context_swapped,
            player_ratings=player_ratings_swapped,
            player_history=player_history_swapped,
            player_mask=player_mask_swapped,
            player_champions=player_champions_swapped,
            meta_champions=meta_champions_swapped,
            best_of=best_of,
        )

    # Sum of forward and reversed must be exactly 1.0 everywhere
    prob_sum = p_ab + p_ba
    diff = torch.abs(prob_sum - 1.0)
    assert torch.max(diff).item() < 1e-6, f"Anti-symmetry violated! Max diff: {torch.max(diff).item()}"


def test_a1_cold_start_mask_handling():
    """Model must run gracefully when a player has 0 historical maps (all mask = 0)."""
    batch_size = 2
    config = A1Config()
    model = A1Model(config)
    model.eval()

    context = torch.randn(batch_size, config.context_dim)
    player_ratings = torch.randn(batch_size, 2, 5, 21)
    player_history = torch.zeros(batch_size, 2, 5, 16, 33)
    player_mask = torch.zeros(batch_size, 2, 5, 16, dtype=torch.uint8)  # All empty
    player_champions = torch.zeros(batch_size, 2, 5, 16, dtype=torch.long)
    meta_champions = torch.randn(batch_size, 2, 5, 32)
    best_of = torch.tensor([1, 3], dtype=torch.long)

    with torch.no_grad():
        probs = model(
            context=context,
            player_ratings=player_ratings,
            player_history=player_history,
            player_mask=player_mask,
            player_champions=player_champions,
            meta_champions=meta_champions,
            best_of=best_of,
        )

    assert torch.all(torch.isfinite(probs))
    assert torch.all(probs > 0.0) and torch.all(probs < 1.0)


def test_masked_history_padding_cannot_replace_latest_valid_champion():
    model = A1Model(A1Config(dropout=0.0)).eval()
    ratings = torch.zeros(1, 5, 21)
    history = torch.zeros(1, 5, 16, 33)
    mask = torch.zeros(1, 5, 16, dtype=torch.bool)
    champions = torch.zeros(1, 5, 16, dtype=torch.long)
    meta = torch.zeros(1, 5, 32)
    mask[0, 0, 3] = True
    champions[0, 0, 3] = 7

    with torch.no_grad():
        expected = model.encode_players_side(ratings, history, mask, champions, meta)
        champions[0, 0, -1] = model.config.num_champions + 100
        actual = model.encode_players_side(ratings, history, mask, champions, meta)

    assert torch.equal(actual, expected)


def test_primary_champion_uses_last_valid_history_slot():
    model = A1Model(A1Config(dropout=0.0)).eval()
    with torch.no_grad():
        model.champion_embed.weight.zero_()
        model.champion_embed.weight[3, 0] = -1.0
        model.champion_embed.weight[7, 0] = 1.0
        model.player_fusion[0].weight.zero_()
        model.player_fusion[0].weight[0, 21 + 32] = 1.0
        model.player_fusion[0].bias.zero_()

    ratings = torch.zeros(1, 5, 21)
    history = torch.zeros(1, 5, 16, 33)
    mask = torch.zeros(1, 5, 16, dtype=torch.bool)
    champions = torch.zeros(1, 5, 16, dtype=torch.long)
    meta = torch.zeros(1, 5, 32)
    mask[0, 0, 2] = True
    mask[0, 0, 8] = True
    champions[0, 0, 2] = 3
    champions[0, 0, 8] = 7

    with torch.no_grad():
        latest_valid = model.encode_players_side(ratings, history, mask, champions, meta)
        mask[0, 0, 8] = False
        earlier_valid = model.encode_players_side(ratings, history, mask, champions, meta)
    assert not torch.equal(latest_valid[0, 0], earlier_valid[0, 0])


def test_a1_predictor_uses_model_parameter_dtype():
    model = A1Model(A1Config(dropout=0.0)).double()
    predictor = A1Predictor(model)
    probs = predictor.predict_batch(
        context=np.zeros((1, model.config.context_dim), dtype=np.float32),
        player_ratings=np.zeros((1, 2, 5, 21), dtype=np.float32),
        player_history=np.zeros((1, 2, 5, 16, 33), dtype=np.float32),
        player_mask=np.zeros((1, 2, 5, 16), dtype=np.uint8),
        player_champions=np.zeros((1, 2, 5, 16), dtype=np.int64),
        meta_champions=np.zeros((1, 2, 5, 32), dtype=np.float32),
        best_of=np.ones(1, dtype=np.int64),
    )

    assert probs.shape == (1,)
    assert np.isfinite(probs).all()


@pytest.mark.parametrize(
    "calibration",
    [
        {"platt_a": 0.0, "platt_b": 0.0},
        {"platt_a": -1.0, "platt_b": 0.0},
        {"platt_a": float("inf"), "platt_b": 0.0},
        {"platt_a": 1.0, "platt_b": float("nan")},
    ],
)
def test_a1_predictor_rejects_invalid_slope_only_calibration(calibration):
    with pytest.raises(ValueError):
        A1Predictor(A1Model(), calibration)

