"""Unit tests for official consolidated A1 model architecture and predictor engine."""

import pytest
import numpy as np
import torch

from src.models.a1.engine import A1PredictorEngine, AntiSymmetricMacroMLP


def test_anti_symmetric_macro_mlp_exact_symmetry():
    """AntiSymmetricMacroMLP must evaluate to exact negative under inverted inputs."""
    mlp = AntiSymmetricMacroMLP(in_dim=7, hidden_dim=16)
    mlp.eval()

    x = torch.randn(5, 7)
    with torch.no_grad():
        out_fwd = mlp(x)
        out_rev = mlp(-x)

    assert torch.allclose(out_fwd, -out_rev, atol=1e-6)


def test_a1_predictor_engine_predict_and_symmetry():
    """A1PredictorEngine must maintain strict bilateral symmetry p(A, B) + p(B, A) = 1.0."""
    engine = A1PredictorEngine(macro_mlp=AntiSymmetricMacroMLP(), tier_scale=0.94)

    N = 4
    p_a0 = np.array([0.70, 0.45, 0.90, 0.20])
    w20_a = np.random.randn(N, 10).astype(np.float32)
    w20_b = np.random.randn(N, 10).astype(np.float32)
    pl_a = np.random.randn(N, 5, 21).astype(np.float32)
    pl_b = np.random.randn(N, 5, 21).astype(np.float32)
    tiers = np.array(["major", "regional", "development", "minor_top_level"])

    # Forward evaluation
    p_ab = engine.predict(p_a0, w20_a, w20_b, pl_a, pl_b, competition_tiers=tiers)

    # Inverted evaluation (Side B vs Side A, p_a0 inverted to 1 - p_a0)
    p_ba = engine.predict(1.0 - p_a0, w20_b, w20_a, pl_b, pl_a, competition_tiers=tiers)

    assert p_ab.shape == (N,)
    assert np.all(p_ab > 0.0) and np.all(p_ab < 1.0)
    assert np.allclose(p_ab + p_ba, np.ones(N), atol=1e-5)


def test_a1_tier_scale_compression():
    """Major and development tier matches must have compressed logits compared to regional."""
    engine = A1PredictorEngine(macro_mlp=AntiSymmetricMacroMLP(), tier_scale=0.90)

    p_a0 = np.array([0.95, 0.95])
    w20_zeros = np.zeros((2, 10), dtype=np.float32)
    tiers = np.array(["major", "regional"])

    p_out = engine.predict(p_a0, w20_zeros, w20_zeros, competition_tiers=tiers)

    # Major tier probability should be softened (closer to 0.50 than 0.95)
    assert p_out[0] < p_out[1]
    assert p_out[0] < 0.95
    assert p_out[1] == pytest.approx(0.95, abs=1e-4)


def test_explicit_macro_supports_bfloat16_inference():
    with torch.random.fork_rng():
        torch.manual_seed(20260926)
        macro = AntiSymmetricMacroMLP().to(dtype=torch.bfloat16)
    engine = A1PredictorEngine(macro_mlp=macro)
    history = np.zeros((1, 10), dtype=np.float32)
    probability = engine.predict(np.array([0.7]), history, history)
    np.testing.assert_allclose(probability, [0.7], rtol=0, atol=1e-7)
