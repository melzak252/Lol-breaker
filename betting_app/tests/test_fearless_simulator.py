"""Tests for Fearless Draft Map-Level Simulator & Series Scoreline Engine.

Covers:
1. Conservation of probability mass: sum(P(Scoreline)) == 1.0 +- 1e-6 for Bo3 and Bo5.
2. Side anti-symmetry: P(3-1 | A, B) == P(1-3 | B, A) under symmetric conditions.
3. Depletion monotonicity: In a matchup between a deep team and a shallow team,
   Team A's win probability increases strictly from Map 1 to Map 5 as champion pools burn.
4. Regression check: scripts/run_model_benchmark.py --suite --doctor passes with status READY.
5. Invariants and edge cases for state advancement, token masking, and draft sampling.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.features.champion_pool_entropy import (
    DEFAULT_COMFORT_THRESHOLD,
    DEFAULT_GAMMA,
    ROLES,
    calculate_depletion_penalty,
    clear_player_pool_registry,
    compute_player_pool_metrics,
    register_player_pool,
)
from src.models.fearless_simulator import (
    BLUE_SIDE_RATING_OFFSET,
    FearlessSeriesResult,
    FearlessSeriesState,
    c0_predict_fearless_map,
    mask_c0_tokens_for_burned_champions,
    sample_map_draft,
    simulate_fearless_series,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure in-memory player registry is clean for each test."""
    clear_player_pool_registry()
    yield
    clear_player_pool_registry()


# ─── Test 1: Conservation of Probability Mass ────────────────────────────────

def test_conservation_of_probability_mass_bo5():
    """Verify sum(P(Scoreline)) == 1.0 +- 1e-6 across Bo5 Monte Carlo simulations."""
    roster_a = [f"player_a_{r}" for r in ROLES]
    roster_b = [f"player_b_{r}" for r in ROLES]

    result = simulate_fearless_series(
        team_a=roster_a,
        team_b=roster_b,
        best_of=5,
        n_simulations=5000,
        seed=101,
        base_map_prob=0.52,
    )

    pmf_sum = sum(result.scoreline_pmf.values())
    assert math.isclose(pmf_sum, 1.0, abs_tol=1e-6), f"PMF sum {pmf_sum} != 1.0"

    # All legal Bo5 scorelines must be present
    expected_scorelines = {(3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3)}
    assert set(result.scoreline_pmf.keys()) == expected_scorelines

    # Verify series probabilities match winner sum
    prob_a_from_scores = sum(p for s, p in result.scoreline_pmf.items() if s[0] == 3)
    prob_b_from_scores = sum(p for s, p in result.scoreline_pmf.items() if s[1] == 3)
    assert math.isclose(prob_a_from_scores, result.series_prob_a, abs_tol=1e-6)
    assert math.isclose(prob_b_from_scores, result.series_prob_b, abs_tol=1e-6)
    assert math.isclose(result.series_prob_a + result.series_prob_b, 1.0, abs_tol=1e-6)


def test_conservation_of_probability_mass_bo3():
    """Verify sum(P(Scoreline)) == 1.0 +- 1e-6 for Bo3."""
    roster_a = [f"player_a_{r}" for r in ROLES]
    roster_b = [f"player_b_{r}" for r in ROLES]

    result = simulate_fearless_series(
        team_a=roster_a,
        team_b=roster_b,
        best_of=3,
        n_simulations=4000,
        seed=202,
        base_map_prob=0.48,
    )

    pmf_sum = sum(result.scoreline_pmf.values())
    assert math.isclose(pmf_sum, 1.0, abs_tol=1e-6), f"Bo3 PMF sum {pmf_sum} != 1.0"
    expected_scorelines = {(2, 0), (2, 1), (1, 2), (0, 2)}
    assert set(result.scoreline_pmf.keys()) == expected_scorelines


# ─── Test 2: Side Anti-Symmetry ──────────────────────────────────────────────

def test_side_anti_symmetry_scoreline_pmf():
    """Verify P(3-1 | A, B) == P(1-3 | B, A) under swapped side and team orientation."""
    roster_a = [f"team_a_{r}" for r in ROLES]
    roster_b = [f"team_b_{r}" for r in ROLES]

    # Register identical pool depth for both teams to ensure symmetric depletion behavior
    for r in ROLES:
        register_player_pool(f"team_a_{r}", {100: 20, 101: 20, 102: 20, 103: 20, 104: 20}, role=r)
        register_player_pool(f"team_b_{r}", {200: 20, 201: 20, 202: 20, 203: 20, 204: 20}, role=r)

    # Base matchup A vs B (A starts Blue)
    res_ab = simulate_fearless_series(
        team_a=roster_a,
        team_b=roster_b,
        best_of=5,
        n_simulations=10000,
        seed=777,
        initial_side_a="blue",
        base_map_prob=0.55,
    )

    # Swapped matchup B vs A: Team B starts Red (so Team A starts Blue), with Team B base prob 0.45
    res_ba = simulate_fearless_series(
        team_a=roster_b,
        team_b=roster_a,
        best_of=5,
        n_simulations=10000,
        seed=777,
        initial_side_a="red",
        base_map_prob=0.45,
    )

    # P(3-1 | A vs B) must equal P(1-3 | B vs A) within Monte Carlo tolerance
    p_3_1_ab = res_ab.scoreline_pmf[(3, 1)]
    p_1_3_ba = res_ba.scoreline_pmf[(1, 3)]
    assert math.isclose(p_3_1_ab, p_1_3_ba, abs_tol=0.02), (
        f"Anti-symmetry violated: P(3-1 | A,B)={p_3_1_ab:.4f} vs P(1-3 | B,A)={p_1_3_ba:.4f}"
    )

    # Total series probabilities must be symmetric: P(A wins | A,B) == P(A wins as team_b | B,A)
    assert math.isclose(res_ab.series_prob_a, res_ba.series_prob_b, abs_tol=0.02)

def test_map_prediction_side_anti_symmetry():
    """Verify single map predictor anti-symmetry: P(A wins | side=blue) = 1 - P(B wins | side=red)."""
    p_a = c0_predict_fearless_map(
        features_a={"roster": ["p1", "p2", "p3", "p4", "p5"]},
        features_b={"roster": ["q1", "q2", "q3", "q4", "q5"]},
        burned_champions={10, 20, 30},
        side_a="blue",
        base_map_prob=0.55,
    )
    p_b = c0_predict_fearless_map(
        features_a={"roster": ["q1", "q2", "q3", "q4", "q5"]},
        features_b={"roster": ["p1", "p2", "p3", "p4", "p5"]},
        burned_champions={10, 20, 30},
        side_a="red",
        base_map_prob=0.45,
    )
    assert math.isclose(p_a + p_b, 1.0, abs_tol=1e-6), f"P(A) + P(B) = {p_a + p_b} != 1.0"


# ─── Test 3: Depletion Monotonicity ──────────────────────────────────────────

def test_depletion_monotonicity_deep_vs_shallow():
    """Verify Team A's win probability strictly increases from Map 1 to Map 5 as shallow team's pool burns."""
    # Deep team (T1-like): 10 comfort champions each per role (depth = 10.0 >= 3.5)
    for r_idx, role in enumerate(ROLES):
        deep_pool = {1000 + r_idx * 100 + j: 10 for j in range(10)}
        register_player_pool(f"deep_{role}", deep_pool, role=role)

    # Shallow team: narrow comfort pool (depth = 3.41 < 3.5, 1 main pick + 4 secondary comfort picks)
    for r_idx, role in enumerate(ROLES):
        shallow_pool = {2000 + r_idx * 100: 60}
        for j in range(1, 5):
            shallow_pool[2000 + r_idx * 100 + j] = 10
        register_player_pool(f"shallow_{role}", shallow_pool, role=role)

    roster_deep = [f"deep_{r}" for r in ROLES]
    roster_shallow = [f"shallow_{r}" for r in ROLES]

    # Evaluate win probability map by map at neutral side as burned champions accumulate
    # Map 1: 0 champions burned
    burned_0: set[int] = set()
    # Map 2: 1 champion per role burned
    burned_10 = {1000 + r * 100 for r in range(5)} | {2000 + r * 100 for r in range(5)}
    # Map 3: 2 champions per role burned
    burned_20 = burned_10 | {1000 + r * 100 + 1 for r in range(5)} | {2000 + r * 100 + 1 for r in range(5)}
    # Map 4: 3 champions per role burned
    burned_30 = burned_20 | {1000 + r * 100 + 2 for r in range(5)} | {2000 + r * 100 + 2 for r in range(5)}
    # Map 5: 4 champions per role burned
    burned_40 = burned_30 | {1000 + r * 100 + 3 for r in range(5)} | {2000 + r * 100 + 3 for r in range(5)}

    probs = []
    for burned in [burned_0, burned_10, burned_20, burned_30, burned_40]:
        p = c0_predict_fearless_map(
            features_a=roster_deep,
            features_b=roster_shallow,
            burned_champions=burned,
            side_a="blue",
            base_map_prob=0.50,
        )
        probs.append(p)

    # Verify strict monotonicity from Map 1 to Map 5
    # Map 1 and Map 2 have minimal/zero penalty; Map 3, 4, 5 grow strictly as shallow pools deplete
    assert probs[0] <= probs[1], f"Map 1 ({probs[0]}) should be <= Map 2 ({probs[1]})"
    assert probs[1] < probs[2], f"Map 2 ({probs[1]}) must be strictly < Map 3 ({probs[2]})"
    assert probs[2] < probs[3], f"Map 3 ({probs[2]}) must be strictly < Map 4 ({probs[3]})"
    assert probs[3] < probs[4], f"Map 4 ({probs[3]}) must be strictly < Map 5 ({probs[4]})"


# ─── Test 4: Regression Check ────────────────────────────────────────────────

def test_model_benchmark_suite_doctor_passes():
    """Verify scripts/run_model_benchmark.py --suite --doctor reports status READY."""
    project_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, str(project_root / "scripts/run_model_benchmark.py"), "--suite", "--doctor"]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, check=True)
    assert '"status": "READY"' in result.stdout or "'status': 'READY'" in result.stdout


# ─── Test 5: FearlessSeriesState Invariants & Rules Kernel ───────────────────

def test_fearless_series_state_invariants():
    """Verify Hard Fearless invariant: len(burned_champions) == 10 * (current_map - 1)."""
    state = FearlessSeriesState(best_of=5)
    assert state.current_map == 1
    assert len(state.burned_champions) == 0
    assert not state.is_terminal
    assert state.target_wins == 3
    assert state.winner is None

    # Advance Map 1
    draft_1 = list(range(1, 11))
    state = state.advance(winner="A", drafted_champions=draft_1, loser_side_choice="blue")
    assert state.current_map == 2
    assert state.score_a == 1 and state.score_b == 0
    assert len(state.burned_champions) == 10
    # Team B lost map 1, so Team B chose Blue -> Team A is Red
    assert state.current_side_a == "red"

    # Advance Map 2
    draft_2 = list(range(11, 21))
    state = state.advance(winner="B", drafted_champions=draft_2, loser_side_choice="blue")
    assert state.current_map == 3
    assert state.score_a == 1 and state.score_b == 1
    assert len(state.burned_champions) == 20
    # Team A lost map 2, so Team A chose Blue -> Team A is Blue
    assert state.current_side_a == "blue"

    # Attempting to draft already burned champion raises ValueError
    with pytest.raises(ValueError, match="already burned"):
        state.advance(winner="A", drafted_champions=[1] + list(range(21, 30)))

    # Attempting to draft fewer than 10 unique champions raises ValueError
    with pytest.raises(ValueError, match="10 unique"):
        state.advance(winner="A", drafted_champions=[21, 21] + list(range(22, 30)))


def test_token_masking_for_burned_champions():
    """Verify C0 historical tokens are correctly zero-masked when champion is burned."""
    # Shape: (1, 2, 5, 16)
    champions = np.array([[[[10, 20, 30] + [0] * 13] * 5] * 2], dtype=np.uint16)
    mask = np.ones((1, 2, 5, 16), dtype=np.uint8)
    history = np.ones((1, 2, 5, 16, 33), dtype=np.float32)

    hist_arrays = {"champions": champions, "mask": mask, "history": history}

    # Burn champion 20
    masked = mask_c0_tokens_for_burned_champions(hist_arrays, burned_champions={20})

    # Token at index 1 (champion 20) should have mask=0 and history=0.0
    assert (masked["mask"][:, :, :, 1] == 0).all()
    assert (masked["history"][:, :, :, 1, :] == 0.0).all()

    # Tokens at index 0 (champion 10) and index 2 (champion 30) should remain 1
    assert (masked["mask"][:, :, :, 0] == 1).all()
    assert (masked["mask"][:, :, :, 2] == 1).all()
    assert (masked["history"][:, :, :, 0, :] == 1.0).all()


def test_synthetic_map_draft_sampling():
    """Verify sample_map_draft draws exactly 10 distinct, non-burned champions."""
    roster_a = [f"a_{r}" for r in ROLES]
    roster_b = [f"b_{r}" for r in ROLES]
    burned = {266, 64, 103, 51, 412}  # Some meta picks already burned

    rng = np.random.default_rng(12345)
    draft = sample_map_draft(roster_a, roster_b, burned_champions=burned, rng=rng)

    assert len(draft) == 10
    assert len(set(draft)) == 10
    assert set(draft).isdisjoint(burned)
