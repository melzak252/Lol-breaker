"""Unit tests for EXP-040 candidate features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from betting_app.ml.features.candidate_features import (
    assemble_symmetric_candidate_features,
    compute_patch_decay_weights,
    compute_roster_continuity,
    compute_series_side_priority,
    compute_side_advantage,
)


class TestSideAdvantage:
    """Test side advantage and series side priority calculation."""

    def test_side_advantage_team_a_blue(self):
        assert compute_side_advantage("T1", "T1", "Gen.G") == 1.0
        assert compute_side_advantage("t1", " T1 ", "Gen.G") == 1.0

    def test_side_advantage_team_b_blue(self):
        assert compute_side_advantage("Gen.G", "T1", "Gen.G") == -1.0
        assert compute_side_advantage("gen.g", "T1", "Gen.G ") == -1.0

    def test_side_advantage_unknown_or_neither(self):
        assert compute_side_advantage(None, "T1", "Gen.G") == 0.0
        assert compute_side_advantage("", "T1", "Gen.G") == 0.0
        assert compute_side_advantage("KT Rolster", "T1", "Gen.G") == 0.0

    def test_side_advantage_identical_teams(self):
        assert compute_side_advantage("T1", "T1", "T1") == 0.0

    def test_side_advantage_antisymmetry(self):
        for blue in ["T1", "Gen.G", "Unknown", None]:
            val_ab = compute_side_advantage(blue, "T1", "Gen.G")
            val_ba = compute_side_advantage(blue, "Gen.G", "T1")
            assert val_ab == -val_ba

    def test_series_side_priority_team_a_higher(self):
        assert compute_series_side_priority("T1", "T1", "Gen.G") == 1.0

    def test_series_side_priority_team_b_higher(self):
        assert compute_series_side_priority("Gen.G", "T1", "Gen.G") == -1.0

    def test_series_side_priority_unknown(self):
        assert compute_series_side_priority(None, "T1", "Gen.G") == 0.0
        assert compute_series_side_priority("", "T1", "Gen.G") == 0.0
        assert compute_series_side_priority("BLG", "T1", "Gen.G") == 0.0

    def test_series_side_priority_antisymmetry(self):
        for seed in ["T1", "Gen.G", "BLG", None]:
            val_ab = compute_series_side_priority(seed, "T1", "Gen.G")
            val_ba = compute_series_side_priority(seed, "Gen.G", "T1")
            assert val_ab == -val_ba


class TestPatchDecayWeights:
    """Test exponential time-decay and patch adaptation lag penalty."""

    def test_empty_dates(self):
        weights = compute_patch_decay_weights([], [], datetime(2026, 9, 1), "14.10")
        assert len(weights) == 0
        assert isinstance(weights, np.ndarray)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_patch_decay_weights([datetime(2026, 9, 1)], [], datetime(2026, 9, 2), "14.10")

    def test_invalid_half_life_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_patch_decay_weights(
                [datetime(2026, 9, 1)],
                ["14.10"],
                datetime(2026, 9, 2),
                "14.10",
                half_life_days=0.0,
            )

    def test_pure_time_decay_on_same_patch(self):
        target_date = datetime(2026, 9, 22)
        target_patch = "14.10"
        half_life = 21.0

        game_dates = [
            target_date,  # 0 days elapsed -> decay = 1.0
            target_date - timedelta(days=21),  # 21 days elapsed -> decay = 0.5
            target_date - timedelta(days=42),  # 42 days elapsed -> decay = 0.25
        ]
        game_patches = ["14.10", "14.10", "14.10"]

        weights = compute_patch_decay_weights(
            game_dates,
            game_patches,
            target_date,
            target_patch,
            half_life_days=half_life,
            old_patch_multiplier=0.4,
        )

        assert weights[0] == pytest.approx(1.0, rel=1e-5)
        assert weights[1] == pytest.approx(0.5, rel=1e-5)
        assert weights[2] == pytest.approx(0.25, rel=1e-5)

    def test_patch_adaptation_lag_penalty(self):
        target_date = datetime(2026, 9, 22)
        target_patch = "14.10"
        old_multiplier = 0.4

        game_dates = [
            target_date,  # Same patch "14.10" -> 1.0 * 1.0 = 1.0
            target_date,  # Older patch "14.9" -> 1.0 * 0.4 = 0.4
            target_date,  # Older patch "13.24" -> 1.0 * 0.4 = 0.4
            target_date,  # Subversion of same patch "14.10.1" -> 1.0 * 1.0 = 1.0
        ]
        game_patches = ["14.10", "14.9", "13.24", "14.10.1"]

        weights = compute_patch_decay_weights(
            game_dates,
            game_patches,
            target_date,
            target_patch,
            half_life_days=21.0,
            old_patch_multiplier=old_multiplier,
        )

        assert weights[0] == pytest.approx(1.0, rel=1e-5)
        assert weights[1] == pytest.approx(0.4, rel=1e-5)
        assert weights[2] == pytest.approx(0.4, rel=1e-5)
        assert weights[3] == pytest.approx(1.0, rel=1e-5)

    def test_combined_decay_and_patch_penalty(self):
        target_date = datetime(2026, 9, 22)
        target_patch = "14.10"
        half_life = 21.0
        old_multiplier = 0.4

        # 21 days ago on older patch: 0.5 * 0.4 = 0.2
        game_dates = [target_date - timedelta(days=21)]
        game_patches = ["14.9"]

        weights = compute_patch_decay_weights(
            game_dates,
            game_patches,
            target_date,
            target_patch,
            half_life_days=half_life,
            old_patch_multiplier=old_multiplier,
        )

        assert weights[0] == pytest.approx(0.2, rel=1e-5)

    def test_timezone_aware_and_naive_handling(self):
        target_date = datetime(2026, 9, 22, tzinfo=timezone.utc)
        game_dates = [datetime(2026, 9, 22)]  # naive datetime
        game_patches = ["14.10"]

        weights = compute_patch_decay_weights(
            game_dates,
            game_patches,
            target_date,
            "14.10",
        )
        assert weights[0] == pytest.approx(1.0, rel=1e-5)


class TestRosterContinuity:
    """Test roster continuity metrics: lineup cohesion, substitute count, games together."""

    def test_empty_lineup(self):
        metrics = compute_roster_continuity([], [])
        assert metrics["lineup_cohesion"] == 0.0
        assert metrics["substitute_count"] == 0.0
        assert metrics["games_together"] == 0.0

    def test_empty_past_lineups(self):
        lineup = [1, 2, 3, 4, 5]
        metrics = compute_roster_continuity(lineup, [])
        assert metrics["lineup_cohesion"] == 0.0
        # All 5 players have 0 games (< 3), so 5 substitutes
        assert metrics["substitute_count"] == 5.0
        assert metrics["games_together"] == 0.0

    def test_perfect_continuity(self):
        lineup = [1, 2, 3, 4, 5]
        past = [lineup.copy() for _ in range(10)]
        metrics = compute_roster_continuity(lineup, past, max_games=20)

        # Jaccard similarity is 1.0 for every game
        assert metrics["lineup_cohesion"] == 1.0
        # All 5 played 10 games (>= 3), so 0 substitutes
        assert metrics["substitute_count"] == 0.0
        # Every pair played 10 games together
        assert metrics["games_together"] == 10.0

    def test_substitute_count_threshold(self):
        # Lineup: players 1, 2, 3, 4, 5
        lineup = [1, 2, 3, 4, 5]
        # P1-P3 played 5 games (>= 3 -> not subs)
        # P4 played 2 games (< 3 -> sub)
        # P5 played 0 games (< 3 -> sub)
        past = [
            [1, 2, 3, 4, 99],
            [1, 2, 3, 4, 99],
            [1, 2, 3, 98, 99],
            [1, 2, 3, 98, 99],
            [1, 2, 3, 98, 99],
        ]
        metrics = compute_roster_continuity(lineup, past)
        assert metrics["substitute_count"] == 2.0

    def test_lineup_cohesion_partial_overlap(self):
        # Current lineup: {1, 2, 3, 4, 5}
        # Past lineup 1: {1, 2, 3, 4, 6} -> intersection=4, union=6 -> Jaccard = 4/6 = 2/3
        # Past lineup 2: {1, 2, 3, 6, 7} -> intersection=3, union=7 -> Jaccard = 3/7
        lineup = [1, 2, 3, 4, 5]
        past = [
            [1, 2, 3, 4, 6],
            [1, 2, 3, 6, 7],
        ]
        metrics = compute_roster_continuity(lineup, past)
        expected_cohesion = (4 / 6 + 3 / 7) / 2
        assert metrics["lineup_cohesion"] == pytest.approx(expected_cohesion, rel=1e-4)

    def test_max_games_window_applied_to_cohesion(self):
        # If max_games=2, only the last 2 games in past_lineups should be used for cohesion
        lineup = [1, 2, 3, 4, 5]
        past = [
            [90, 91, 92, 93, 94],  # 0 overlap (older game)
            [1, 2, 3, 4, 5],       # 1.0 overlap
            [1, 2, 3, 4, 5],       # 1.0 overlap
        ]
        metrics = compute_roster_continuity(lineup, past, max_games=2)
        # Last 2 games both have 1.0 overlap
        assert metrics["lineup_cohesion"] == 1.0

    def test_games_together_pairwise(self):
        # Current lineup: 1, 2, 3, 4, 5
        lineup = [1, 2, 3, 4, 5]
        # Past: 2 games with 1, 2, 3, 4, 6
        # Players 1, 2, 3, 4 played 2 games together (6 pairs * 2 = 12 pair-games)
        # Player 5 played 0 games (4 pairs with 5 * 0 = 0 pair-games)
        # Total pair games = 12 / 10 pairs = 1.2
        past = [
            [1, 2, 3, 4, 6],
            [1, 2, 3, 4, 6],
        ]
        metrics = compute_roster_continuity(lineup, past)
        assert metrics["games_together"] == pytest.approx(1.2, rel=1e-5)

    def test_mixed_int_and_str_identifiers(self):
        lineup = ["1", 2, "Faker", 4, 5]
        past = [
            [1, "2", "FAKER", 4, 5],
            ["1", 2, "faker ", "4", "5"],
            [1, 2, "Faker", 4, 5],
        ]
        metrics = compute_roster_continuity(lineup, past)
        assert metrics["lineup_cohesion"] == 1.0
        assert metrics["substitute_count"] == 0.0
        assert metrics["games_together"] == 3.0


class TestAssembleSymmetricCandidateFeatures:
    """Test assembly of candidate features and strict antisymmetry under team swapping."""

    def test_basic_differential_features(self):
        stats_a = {"gold_diff_15": 500.0, "win_rate": 0.6}
        stats_b = {"gold_diff_15": 200.0, "win_rate": 0.55}
        context = {
            "team_a": "T1",
            "team_b": "Gen.G",
            "game1_blue_team": "T1",
            "higher_seed_team": "T1",
        }

        features = assemble_symmetric_candidate_features(stats_a, stats_b, context)

        assert features["delta_gold_diff_15"] == pytest.approx(300.0)
        assert features["delta_win_rate"] == pytest.approx(0.05)
        assert features["side_advantage"] == 1.0
        assert features["series_side_priority"] == 1.0

    def test_missing_stats_handling(self):
        stats_a = {"rating_elo": 1600.0}
        stats_b = {"rating_glicko": 1700.0}
        context = {"team_a": "A", "team_b": "B"}

        features = assemble_symmetric_candidate_features(stats_a, stats_b, context)
        assert features["delta_rating_elo"] == 1600.0
        assert features["delta_rating_glicko"] == -1700.0

    def test_continuity_differentials(self):
        lineup_a = [1, 2, 3, 4, 5]
        past_a = [lineup_a.copy() for _ in range(5)]  # Cohesion 1.0, subs 0, games_together 5.0

        lineup_b = [10, 20, 30, 40, 50]
        past_b = []  # Cohesion 0.0, subs 5, games_together 0.0

        context = {
            "team_a": "TeamA",
            "team_b": "TeamB",
            "team_a_current_lineup": lineup_a,
            "team_a_past_lineups": past_a,
            "team_b_current_lineup": lineup_b,
            "team_b_past_lineups": past_b,
        }

        features = assemble_symmetric_candidate_features({}, {}, context)
        assert features["delta_lineup_cohesion"] == pytest.approx(1.0)
        assert features["delta_substitute_count"] == pytest.approx(-5.0)
        assert features["delta_games_together"] == pytest.approx(5.0)

    def test_exact_antisymmetry_under_team_swapping(self):
        """Verify features(B, A, swapped_context) == -features(A, B, context)."""
        stats_a = {
            "gold_diff_15": 650.0,
            "win_rate": 0.62,
            "dpm": 2100.0,
            "dragons": 2.5,
        }
        stats_b = {
            "gold_diff_15": -150.0,
            "win_rate": 0.48,
            "dpm": 1850.0,
            "towers": 6.2,
        }

        target_date = datetime(2026, 9, 15, tzinfo=timezone.utc)
        target_patch = "14.10"

        lineup_a = [1, 2, 3, 4, 5]
        past_a = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [1, 2, 3, 4, 5],
        ]

        lineup_b = [11, 12, 13, 14, 15]
        past_b = [
            [11, 12, 13, 14, 99],
            [11, 12, 13, 98, 99],
        ]

        dates_a = [target_date - timedelta(days=5), target_date - timedelta(days=15)]
        patches_a = ["14.10", "14.9"]
        dates_b = [target_date - timedelta(days=2), target_date - timedelta(days=25)]
        patches_b = ["14.10", "14.8"]

        context_ab = {
            "team_a": "T1",
            "team_b": "Gen.G",
            "game1_blue_team": "T1",
            "higher_seed_team": "Gen.G",
            "team_a_current_lineup": lineup_a,
            "team_a_past_lineups": past_a,
            "team_b_current_lineup": lineup_b,
            "team_b_past_lineups": past_b,
            "team_a_game_dates": dates_a,
            "team_a_game_patches": patches_a,
            "team_b_game_dates": dates_b,
            "team_b_game_patches": patches_b,
            "target_date": target_date,
            "target_patch": target_patch,
        }

        context_ba = {
            "team_a": "Gen.G",
            "team_b": "T1",
            "game1_blue_team": "T1",       # unchanged physical team
            "higher_seed_team": "Gen.G",   # unchanged physical team
            "team_a_current_lineup": lineup_b,
            "team_a_past_lineups": past_b,
            "team_b_current_lineup": lineup_a,
            "team_b_past_lineups": past_a,
            "team_a_game_dates": dates_b,
            "team_a_game_patches": patches_b,
            "team_b_game_dates": dates_a,
            "team_b_game_patches": patches_a,
            "target_date": target_date,
            "target_patch": target_patch,
        }

        features_ab = assemble_symmetric_candidate_features(stats_a, stats_b, context_ab)
        features_ba = assemble_symmetric_candidate_features(stats_b, stats_a, context_ba)

        assert set(features_ab.keys()) == set(features_ba.keys())

        for k in features_ab:
            val_ab = features_ab[k]
            val_ba = features_ba[k]
            # Exact antisymmetry: val_ba == -val_ab
            assert val_ba == pytest.approx(-val_ab, abs=1e-6), (
                f"Antisymmetry violation for feature {k}: {val_ba} != -{val_ab}"
            )
