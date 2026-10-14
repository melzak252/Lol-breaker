"""Unit tests and mathematical invariants for causal champion-pool feature extraction."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.champion_pool_features import (
    CHAMPION_FEATURE_NAMES,
    DIFF_FEATURE_NAMES,
    extract_champion_pool_features,
    verify_champion_features_invariants,
)


def test_feature_shapes_and_dtypes() -> None:
    """Verify output shapes and data types."""
    n = 25
    rng = np.random.default_rng(101)
    champs = rng.integers(1, 150, size=(n, 2, 5, 16), dtype=np.uint16)
    mask = rng.integers(0, 2, size=(n, 2, 5, 16), dtype=np.uint8)

    side_feats, diff_feats = extract_champion_pool_features(champs, mask)

    assert side_feats.shape == (n, 2, 30)
    assert diff_feats.shape == (n, 30)
    assert side_feats.dtype == np.float32
    assert diff_feats.dtype == np.float32
    assert len(CHAMPION_FEATURE_NAMES) == 30
    assert len(DIFF_FEATURE_NAMES) == 30


def test_feature_bounds_and_no_nan() -> None:
    """Verify bounded normalized values and absence of NaNs or Infs."""
    n = 50
    rng = np.random.default_rng(202)
    champs = rng.integers(0, 174, size=(n, 2, 5, 16), dtype=np.uint16)
    mask = rng.integers(0, 2, size=(n, 2, 5, 16), dtype=np.uint8)

    side_feats, diff_feats = extract_champion_pool_features(champs, mask)

    assert not np.isnan(side_feats).any()
    assert not np.isinf(side_feats).any()
    assert not np.isnan(diff_feats).any()
    assert not np.isinf(diff_feats).any()

    # Player features (indices 0..24) must be in [0, 1]
    player_feats = side_feats[:, :, :25]
    assert np.all(player_feats >= 0.0)
    assert np.all(player_feats <= 1.0 + 1e-6)

    # First 4 team features (indices 25..28) must be in [0, 1]
    team_norm_feats = side_feats[:, :, 25:29]
    assert np.all(team_norm_feats >= 0.0)
    assert np.all(team_norm_feats <= 1.0 + 1e-6)

    # Flex feature (index 29) must be non-negative integer count
    team_flex = side_feats[:, :, 29]
    assert np.all(team_flex >= 0.0)
    assert np.all(np.equal(np.mod(team_flex, 1), 0))


def test_cold_start_exact_zeros() -> None:
    """Cold start players (0 valid games) must produce exact 0.0 features."""
    n = 10
    champs = np.zeros((n, 2, 5, 16), dtype=np.uint16)
    mask = np.zeros((n, 2, 5, 16), dtype=np.uint8)

    side_feats, diff_feats = extract_champion_pool_features(champs, mask)

    assert np.all(side_feats == 0.0)
    assert np.all(diff_feats == 0.0)


def test_side_swap_anti_symmetry() -> None:
    """Verify exact anti-symmetry: Delta X(B, A) = - Delta X(A, B)."""
    n = 60
    rng = np.random.default_rng(303)
    champs_ab = rng.integers(0, 100, size=(n, 2, 5, 16), dtype=np.uint16)
    mask_ab = rng.integers(0, 2, size=(n, 2, 5, 16), dtype=np.uint8)

    side_ab, diff_ab = extract_champion_pool_features(champs_ab, mask_ab)

    # Swap side axis (axis 1)
    champs_ba = champs_ab[:, [1, 0], :, :]
    mask_ba = mask_ab[:, [1, 0], :, :]

    side_ba, diff_ba = extract_champion_pool_features(champs_ba, mask_ba)

    assert np.allclose(side_ba[:, 0, :], side_ab[:, 1, :])
    assert np.allclose(side_ba[:, 1, :], side_ab[:, 0, :])

    anti_symm_err = np.abs(diff_ab + diff_ba).max()
    assert anti_symm_err < 1.0e-7


def test_entropy_and_concentration_mathematics() -> None:
    """Verify analytical entropy and concentration values on known patterns."""
    champs = np.zeros((4, 2, 5, 16), dtype=np.uint16)
    mask = np.zeros((4, 2, 5, 16), dtype=np.uint8)

    # Case 0: 1 game on 1 champion (role 0, side 0)
    champs[0, 0, 0, 0] = 42
    mask[0, 0, 0, 0] = 1

    # Case 1: 16 games on 16 unique champions (role 0, side 0)
    champs[1, 0, 0, :] = np.arange(1, 17, dtype=np.uint16)
    mask[1, 0, 0, :] = 1

    # Case 2: 16 games on 1 single champion (role 0, side 0)
    champs[2, 0, 0, :] = 55
    mask[2, 0, 0, :] = 1

    # Case 3: 4 games on 2 champions (2 on champ 10, 2 on champ 20)
    champs[3, 0, 0, 0:2] = 10
    champs[3, 0, 0, 2:4] = 20
    mask[3, 0, 0, 0:4] = 1

    side_feats, _ = extract_champion_pool_features(champs, mask)

    # Features for role 0, side 0 are columns 0..4:
    # 0: pool_size_norm, 1: entropy_norm, 2: top1_conc, 3: top2_conc, 4: active_games_norm

    # Case 0: 1 game, 1 champ
    f0 = side_feats[0, 0, :5]
    assert np.isclose(f0[0], 1.0 / 16.0)
    assert np.isclose(f0[1], 0.0)  # log2(1) = 0
    assert np.isclose(f0[2], 1.0)
    assert np.isclose(f0[3], 1.0)
    assert np.isclose(f0[4], 1.0 / 16.0)

    # Case 1: 16 games, 16 unique champs
    f1 = side_feats[1, 0, :5]
    assert np.isclose(f1[0], 1.0)
    assert np.isclose(f1[1], 1.0)  # log2(16) / 4 = 1.0
    assert np.isclose(f1[2], 1.0 / 16.0)
    assert np.isclose(f1[3], 2.0 / 16.0)
    assert np.isclose(f1[4], 1.0)

    # Case 2: 16 games, 1 single champ
    f2 = side_feats[2, 0, :5]
    assert np.isclose(f2[0], 1.0 / 16.0)
    assert np.isclose(f2[1], 0.0)
    assert np.isclose(f2[2], 1.0)
    assert np.isclose(f2[3], 1.0)
    assert np.isclose(f2[4], 1.0)

    # Case 3: 4 games, 2 champs (2 each)
    f3 = side_feats[3, 0, :5]
    assert np.isclose(f3[0], 2.0 / 16.0)
    # H = - (0.5 * (-1) + 0.5 * (-1)) = 1.0 bit -> 1.0 / 4.0 = 0.25
    assert np.isclose(f3[1], 0.25)
    assert np.isclose(f3[2], 0.5)
    assert np.isclose(f3[3], 1.0)
    assert np.isclose(f3[4], 4.0 / 16.0)


def test_team_flex_champions_logic() -> None:
    """Verify flex champion detection across roles."""
    champs = np.zeros((2, 2, 5, 16), dtype=np.uint16)
    mask = np.zeros((2, 2, 5, 16), dtype=np.uint8)

    # Match 0: Top (role 0) and Mid (role 2) both play champion 99
    # Jg (role 1) and Sup (role 4) both play champion 77
    champs[0, 0, 0, 0] = 99
    mask[0, 0, 0, 0] = 1
    champs[0, 0, 2, 0] = 99
    mask[0, 0, 2, 0] = 1

    champs[0, 0, 1, 0] = 77
    mask[0, 0, 1, 0] = 1
    champs[0, 0, 4, 0] = 77
    mask[0, 0, 4, 0] = 1

    # ADC plays unique champ 55
    champs[0, 0, 3, 0] = 55
    mask[0, 0, 3, 0] = 1

    # Match 1: All roles play completely disjoint champions
    for r in range(5):
        champs[1, 0, r, 0] = r + 10
        mask[1, 0, r, 0] = 1

    side_feats, _ = extract_champion_pool_features(champs, mask)

    # team_flex_champions is index 29
    assert side_feats[0, 0, 29] == 2.0  # champions 99 and 77 are flex
    assert side_feats[1, 0, 29] == 0.0  # no flex champions


def test_empty_input_handling() -> None:
    """Verify empty input handling."""
    champs = np.zeros((0, 2, 5, 16), dtype=np.uint16)
    mask = np.zeros((0, 2, 5, 16), dtype=np.uint8)

    side_feats, diff_feats = extract_champion_pool_features(champs, mask)
    assert side_feats.shape == (0, 2, 30)
    assert diff_feats.shape == (0, 30)


def test_invariants_verification_function() -> None:
    """Verify that verify_champion_features_invariants runs and passes."""
    rng = np.random.default_rng(404)
    champs = rng.integers(0, 100, size=(20, 2, 5, 16), dtype=np.uint16)
    mask = rng.integers(0, 2, size=(20, 2, 5, 16), dtype=np.uint8)

    side_feats, diff_feats = extract_champion_pool_features(champs, mask)
    summary = verify_champion_features_invariants(side_feats, diff_feats, champs, mask)
    assert summary["status"] == "PASSED"
    assert summary["n_series"] == 20
    assert summary["max_anti_symmetry_error"] < 1.0e-7
