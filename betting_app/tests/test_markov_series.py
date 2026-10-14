"""Unit tests for Hierarchical Markov Series Simulator.

Tests:
1. Exact binary symmetry invariant across Bo1, Bo3, and Bo5:
   predict_series_proba(1.0 - p, not priority, best_of) == 1.0 - predict_series_proba(p, priority, best_of)
2. Probability bounds in [0.0, 1.0] and edge cases (0.0, 1.0, NaNs, invalid inputs).
3. Blue side loser-advantage dynamics (reduced sweeps, increased decider frequency).
4. Score distribution summation to 1.0 and symmetry under team swapping.
5. Exact analytic state reach probabilities across the series Markov tree.
6. Vectorization and broadcasting across arrays of probabilities and priorities.
7. Simulator API calling conventions and Monte Carlo simulation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from betting_app.ml.models.markov_series import (
    MarkovSeriesSimulator,
    compute_state_probabilities,
    predict_score_distribution,
    predict_series_proba,
    series_expected_games,
    single_game_proba,
)


class TestMarkovSeriesSymmetry:
    """Test strict binary symmetry invariant across all series configurations."""

    @pytest.mark.parametrize("best_of", [1, 3, 5])
    @pytest.mark.parametrize("priority", [True, False])
    @pytest.mark.parametrize("decider_rule", ["loser_picks", "priority_picks"])
    def test_series_symmetry_invariant(
        self,
        best_of: int,
        priority: bool,
        decider_rule: str,
    ) -> None:
        """Verify predict_series_proba(1 - p, not priority) == 1 - predict_series_proba(p, priority)."""
        test_probs = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95]
        for p in test_probs:
            prob_a = predict_series_proba(
                p,
                team_a_has_game1_priority=priority,
                best_of=best_of,
                decider_rule=decider_rule,
            )
            prob_b = predict_series_proba(
                1.0 - p,
                team_a_has_game1_priority=not priority,
                best_of=best_of,
                decider_rule=decider_rule,
            )

            # Strict binary symmetry
            assert np.allclose(prob_a, 1.0 - prob_b, atol=1e-14), (
                f"Symmetry failed for Bo{best_of}, p={p}, prio={priority}, rule={decider_rule}: "
                f"prob_a={prob_a}, 1 - prob_b={1.0 - prob_b}"
            )

    def test_exact_bit_level_symmetry_on_exact_pairs(self) -> None:
        """Verify bit-level equality on floating point pairs with exact binary representation."""
        # 0.5, 0.25, 0.75, 0.375, 0.625 have exact IEEE-754 representations
        exact_probs = [0.25, 0.375, 0.5, 0.625, 0.75]
        for bo in [1, 3, 5]:
            for prio in [True, False]:
                for p in exact_probs:
                    val_a = predict_series_proba(p, prio, best_of=bo)
                    val_b = predict_series_proba(1.0 - p, not prio, best_of=bo)
                    assert val_a == 1.0 - val_b

    def test_fifty_fifty_neutral_advantage_distribution(self) -> None:
        """When neutral win rate is 50%, Game 1 priority grants > 50% series win rate."""
        for bo in [1, 3, 5]:
            p_prio = float(predict_series_proba(0.5, team_a_has_game1_priority=True, best_of=bo))
            p_noprio = float(predict_series_proba(0.5, team_a_has_game1_priority=False, best_of=bo))

            # Priority team has advantage because Blue side bonus > 0
            assert p_prio > 0.50
            assert p_noprio < 0.50
            # Symmetrically opposite
            assert math.isclose(p_prio + p_noprio, 1.0, abs_tol=1e-15)

    def test_zero_side_bonus_recovers_independent_bernoulli_symmetry(self) -> None:
        """With blue_side_bonus=0, series probabilities are invariant to priority."""
        for bo in [1, 3, 5]:
            for p in [0.3, 0.5, 0.7]:
                p_with_prio = predict_series_proba(p, True, best_of=bo, blue_side_bonus=0.0)
                p_without_prio = predict_series_proba(p, False, best_of=bo, blue_side_bonus=0.0)
                assert np.allclose(p_with_prio, p_without_prio, atol=1e-14)


class TestProbabilityBoundsAndEdgeCases:
    """Test numerical bounds and robust edge case handling."""

    def test_probabilities_bounded_in_unit_interval(self) -> None:
        """All predicted probabilities must lie strictly in [0.0, 1.0]."""
        fine_grid = np.linspace(0.0, 1.0, 101)
        for bo in [1, 3, 5]:
            for prio in [True, False]:
                preds = predict_series_proba(fine_grid, prio, best_of=bo)
                assert np.all(preds >= 0.0)
                assert np.all(preds <= 1.0)
                assert np.all(np.isfinite(preds))

    def test_boundary_zero_and_one(self) -> None:
        """Certain win (1.0) or certain loss (0.0) remain strictly 1.0 or 0.0."""
        for bo in [1, 3, 5]:
            for prio in [True, False]:
                assert predict_series_proba(0.0, prio, best_of=bo) == 0.0
                assert predict_series_proba(1.0, prio, best_of=bo) == 1.0

    def test_nan_propagation_without_error(self) -> None:
        """NaN values in input arrays propagate cleanly without exceptions."""
        inputs = np.array([0.2, np.nan, 0.8])
        prios = np.array([True, True, False])
        res = predict_series_proba(inputs, prios, best_of=3)

        assert np.isfinite(res[0])
        assert np.isnan(res[1])
        assert np.isfinite(res[2])

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Invalid series length, decider rule, bonus, or probabilities raise ValueError."""
        # Even series length is invalid for LoL
        with pytest.raises(ValueError, match="best_of must be an odd positive integer"):
            predict_series_proba(0.5, True, best_of=2)

        with pytest.raises(ValueError, match="best_of must be an odd positive integer"):
            predict_series_proba(0.5, True, best_of=0)

        with pytest.raises(ValueError, match="best_of must be an odd positive integer"):
            predict_series_proba(0.5, True, best_of=-3)

        # Invalid decider rule
        with pytest.raises(ValueError, match="decider_rule must be"):
            predict_series_proba(0.5, True, best_of=3, decider_rule="coin_flip")

        # Negative bonus
        with pytest.raises(ValueError, match="blue_side_bonus must be a finite non-negative float"):
            predict_series_proba(0.5, True, best_of=3, blue_side_bonus=-0.1)

        # Score distribution input bounds
        with pytest.raises(ValueError, match="p_neutral_a must be a valid probability"):
            predict_score_distribution(-0.1, True, best_of=3)

        with pytest.raises(ValueError, match="p_neutral_a must be a valid probability"):
            predict_score_distribution(1.1, True, best_of=3)


class TestBlueSideLoserAdvantageDynamics:
    """Test the LoL rotating side dynamics (loser gets Blue side in subsequent game)."""

    def test_game2_loser_boost(self) -> None:
        """Game 1 loser gets Blue side in Game 2, receiving a positive logit boost."""
        p_neutral = 0.5
        bonus = 0.22

        # Team A plays Red in Game 2 if Team A won Game 1
        p_g2_after_win = single_game_proba(p_neutral, a_is_blue=False, blue_side_bonus=bonus)
        # Team A plays Blue in Game 2 if Team A lost Game 1
        p_g2_after_loss = single_game_proba(p_neutral, a_is_blue=True, blue_side_bonus=bonus)

        # Losing Game 1 strictly increases win probability in Game 2
        assert float(p_g2_after_loss) > float(p_g2_after_win)
        assert math.isclose(float(p_g2_after_loss + p_g2_after_win), 1.0, abs_tol=1e-15)

    def test_bo3_sweeps_reduced_and_decider_increased(self) -> None:
        """Rotating Blue side selection reduces 2-0 / 0-2 sweeps and inflates 2-1 / 1-2 deciders."""
        p_neutral = 0.5
        # 1. Independent Bernoulli trials without side advantage
        sc_no_side = predict_score_distribution(p_neutral, True, best_of=3, blue_side_bonus=0.0)
        p_sweep_no_side = sc_no_side["2-0"] + sc_no_side["0-2"]
        p_decider_no_side = sc_no_side["2-1"] + sc_no_side["1-2"]

        # 2. Markov model with rotating Blue side advantage (+0.25 log-odds)
        sc_side = predict_score_distribution(p_neutral, True, best_of=3, blue_side_bonus=0.25)
        p_sweep_side = sc_side["2-0"] + sc_side["0-2"]
        p_decider_side = sc_side["2-1"] + sc_side["1-2"]

        # Loser advantage compresses sweeps and expands deciders
        assert p_sweep_side < p_sweep_no_side
        assert p_decider_side > p_decider_no_side

    def test_bo5_sweeps_reduced_and_decider_increased(self) -> None:
        """In Bo5, rotating Blue side selection inflates Game 5 decider frequency."""
        p_neutral = 0.5
        sc_no_side = predict_score_distribution(p_neutral, True, best_of=5, blue_side_bonus=0.0)
        sc_side = predict_score_distribution(p_neutral, True, best_of=5, blue_side_bonus=0.25)

        p_sweep_no_side = sc_no_side["3-0"] + sc_no_side["0-3"]
        p_decider_no_side = sc_no_side["3-2"] + sc_no_side["2-3"]

        p_sweep_side = sc_side["3-0"] + sc_side["0-3"]
        p_decider_side = sc_side["3-2"] + sc_side["2-3"]

        assert p_sweep_side < p_sweep_no_side
        assert p_decider_side > p_decider_no_side

    def test_expected_games_increases_with_side_advantage(self) -> None:
        """Expected total games in a series increases as Blue side advantage grows."""
        for bo in [3, 5]:
            exp_zero = series_expected_games(0.5, True, best_of=bo, blue_side_bonus=0.0)
            exp_mod = series_expected_games(0.5, True, best_of=bo, blue_side_bonus=0.20)
            exp_high = series_expected_games(0.5, True, best_of=bo, blue_side_bonus=0.35)

            assert exp_high > exp_mod > exp_zero


class TestScoreDistribution:
    """Test score distribution calculation, completeness, and symmetry."""

    def test_score_distribution_keys_bo1_bo3_bo5(self) -> None:
        """Verify the exact score keys produced for Bo1, Bo3, and Bo5."""
        sc1 = predict_score_distribution(0.6, True, best_of=1)
        assert list(sc1.keys()) == ["1-0", "0-1"]

        sc3 = predict_score_distribution(0.6, True, best_of=3)
        assert list(sc3.keys()) == ["2-0", "2-1", "1-2", "0-2"]

        sc5 = predict_score_distribution(0.6, True, best_of=5)
        assert list(sc5.keys()) == ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]

    @pytest.mark.parametrize("bo", [1, 3, 5])
    @pytest.mark.parametrize("p", [0.0, 0.2, 0.5, 0.73, 1.0])
    @pytest.mark.parametrize("priority", [True, False])
    def test_score_distribution_sums_to_one(self, bo: int, p: float, priority: bool) -> None:
        """Probabilities across all series outcomes must sum to exactly 1.0."""
        sc = predict_score_distribution(p, priority, best_of=bo)
        total = sum(sc.values())
        assert math.isclose(total, 1.0, abs_tol=1e-14)
        for prob in sc.values():
            assert prob >= 0.0

    def test_score_distribution_matches_series_probability(self) -> None:
        """Sum of winning score probabilities equals predicted series probability."""
        for bo in [1, 3, 5]:
            for p in [0.3, 0.55, 0.8]:
                sc = predict_score_distribution(p, True, best_of=bo)
                series_p = float(predict_series_proba(p, True, best_of=bo))

                needed = (bo + 1) // 2
                winning_sum = sum(prob for score, prob in sc.items() if int(score.split("-")[0]) == needed)
                assert math.isclose(winning_sum, series_p, abs_tol=1e-14)

    def test_score_distribution_role_reversal_symmetry(self) -> None:
        """Swapping teams (1 - p, not priority) reverses the score distribution exactly."""
        for bo in [1, 3, 5]:
            sc_a = predict_score_distribution(0.65, True, best_of=bo)
            sc_b = predict_score_distribution(0.35, False, best_of=bo)

            for score_a, prob_a in sc_a.items():
                wa, wb = score_a.split("-")
                rev_score = f"{wb}-{wa}"
                assert math.isclose(prob_a, sc_b[rev_score], abs_tol=1e-14)

    def test_score_distribution_boundary_deterministic(self) -> None:
        """At p=1.0 and p=0.0, the score distribution collapses to pure sweep."""
        sc_win = predict_score_distribution(1.0, True, best_of=3)
        assert sc_win["2-0"] == 1.0
        assert sc_win["2-1"] == 0.0
        assert sc_win["1-2"] == 0.0
        assert sc_win["0-2"] == 0.0

        sc_loss = predict_score_distribution(0.0, True, best_of=3)
        assert sc_loss["2-0"] == 0.0
        assert sc_loss["2-1"] == 0.0
        assert sc_loss["1-2"] == 0.0
        assert sc_loss["0-2"] == 1.0


class TestMarkovStateGraph:
    """Test full Markov transition graph state reach probabilities."""

    def test_bo1_states(self) -> None:
        """Bo1 has states '0-0', '1-0', '0-1'."""
        st = compute_state_probabilities(0.6, True, best_of=1)
        assert set(st.keys()) == {"0-0", "1-0", "0-1"}
        assert st["0-0"] == 1.0
        assert math.isclose(st["1-0"] + st["0-1"], 1.0, abs_tol=1e-15)

    def test_bo3_states_exact_transition_equations(self) -> None:
        """Verify the 8 exact Markov states in Bo3 and transition consistency."""
        p_neutral = 0.60
        bonus = 0.22
        st = compute_state_probabilities(p_neutral, True, best_of=3, blue_side_bonus=bonus)

        expected_keys = {"0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "2-1", "1-2"}
        assert set(st.keys()) == expected_keys
        assert st["0-0"] == 1.0

        # Game 1 Blue probability for Team A
        p_blue = float(single_game_proba(p_neutral, a_is_blue=True, blue_side_bonus=bonus))
        p_red = float(single_game_proba(p_neutral, a_is_blue=False, blue_side_bonus=bonus))

        assert math.isclose(st["1-0"], p_blue, abs_tol=1e-14)
        assert math.isclose(st["0-1"], 1.0 - p_blue, abs_tol=1e-14)

        # 2-0 is reached when A wins G1 (Blue) and A wins G2 (Red)
        assert math.isclose(st["2-0"], p_blue * p_red, abs_tol=1e-14)

        # 0-2 is reached when B wins G1 (Red) and B wins G2 (Blue)
        assert math.isclose(st["0-2"], (1.0 - p_blue) * (1.0 - p_blue), abs_tol=1e-14)

        # Conservation: terminal states sum to 1.0
        terminal_sum = st["2-0"] + st["2-1"] + st["1-2"] + st["0-2"]
        assert math.isclose(terminal_sum, 1.0, abs_tol=1e-14)

    def test_bo5_state_tree_conservation(self) -> None:
        """Verify probability conservation across the Bo5 state tree."""
        st = compute_state_probabilities(0.55, True, best_of=5)
        terminal_keys = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
        terminal_sum = sum(st[k] for k in terminal_keys)
        assert math.isclose(terminal_sum, 1.0, abs_tol=1e-14)


class TestVectorizationAndBroadcasting:
    """Test vectorization across numpy arrays."""

    def test_1d_array_probabilities_scalar_priority(self) -> None:
        """Compute series win probabilities for an array of matches with scalar priority."""
        probs = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        preds = predict_series_proba(probs, team_a_has_game1_priority=True, best_of=3)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (5,)
        # Monotonicity check
        assert np.all(np.diff(preds) > 0.0)

    def test_1d_array_probabilities_array_priority(self) -> None:
        """Compute predictions when both probability and priority are arrays."""
        probs = np.array([0.5, 0.5])
        prios = np.array([True, False])
        preds = predict_series_proba(probs, prios, best_of=3)

        assert preds.shape == (2,)
        assert preds[0] > 0.5
        assert preds[1] < 0.5
        assert math.isclose(preds[0] + preds[1], 1.0, abs_tol=1e-14)

    def test_2d_array_broadcasting(self) -> None:
        """Broadcasting across 2D matrices."""
        probs_2d = np.array([[0.3, 0.7], [0.4, 0.6]])
        preds = predict_series_proba(probs_2d, True, best_of=3)

        assert preds.shape == (2, 2)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)


class TestSimulatorApiAndSimulation:
    """Test class instantiation, dual-method calling, and Monte Carlo simulation."""

    def test_dual_calling_conventions(self) -> None:
        """Verify instance call, class call, and module-level function yield identical results."""
        sim = MarkovSeriesSimulator(default_best_of=3, default_blue_side_bonus=0.22)

        res_inst = sim.predict_series_proba(0.65, True)
        res_class = MarkovSeriesSimulator.predict_series_proba(0.65, True, best_of=3, blue_side_bonus=0.22)
        res_func = predict_series_proba(0.65, True, best_of=3, blue_side_bonus=0.22)

        assert np.array_equal(res_inst, res_class)
        assert np.array_equal(res_inst, res_func)

    def test_custom_instance_defaults(self) -> None:
        """Instance defaults are used when arguments are omitted."""
        sim_bo5 = MarkovSeriesSimulator(default_best_of=5, default_blue_side_bonus=0.30)
        p_auto = sim_bo5.predict_series_proba(0.6, True)
        p_explicit = predict_series_proba(0.6, True, best_of=5, blue_side_bonus=0.30)

        assert np.array_equal(p_auto, p_explicit)

    def test_monte_carlo_simulation(self) -> None:
        """Monte Carlo simulator returns valid game-by-game progression and score."""
        score, history = MarkovSeriesSimulator.simulate_series(
            0.6,
            team_a_has_game1_priority=True,
            best_of=3,
            rng=123,
        )

        assert score in {"2-0", "2-1", "1-2", "0-2"}
        assert len(history) in {2, 3}

        # Verify Game 1 side assignment matches priority
        assert history[0]["team_a_side"] == "Blue"
        assert history[0]["team_b_side"] == "Red"

        # Verify Game 2 side assignment matches loser-picks-side rule
        g1_winner = history[0]["winner"]
        if g1_winner == "A":
            # Loser was B -> B selects Blue -> A is Red
            assert history[1]["team_a_side"] == "Red"
        else:
            # Loser was A -> A selects Blue -> A is Blue
            assert history[1]["team_a_side"] == "Blue"

    def test_monte_carlo_reproducibility_with_seed(self) -> None:
        """Simulation is deterministic when seeded."""
        score1, hist1 = MarkovSeriesSimulator.simulate_series(0.55, True, best_of=5, rng=42)
        score2, hist2 = MarkovSeriesSimulator.simulate_series(0.55, True, best_of=5, rng=42)

        assert score1 == score2
        assert len(hist1) == len(hist2)
        assert hist1 == hist2
