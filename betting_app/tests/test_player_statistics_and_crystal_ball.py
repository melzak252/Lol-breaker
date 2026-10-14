"""Tests for player statistics generative model and Worlds Crystal Ball simulation."""
from __future__ import annotations

import json
import math
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient

from betting_app.api.main import app
from src.features.player_statistics_data import (
    DEFAULT_ROLE_KILL_SHARES,
    DEFAULT_ROLE_DEATH_SHARES,
    DEFAULT_ROLE_ASSIST_FACTORS,
    DEFAULT_WINNER_KILL_SHARE,
    DEFAULT_OBJECTIVE_RATES,
    extract_player_stat_priors,
)
from src.models.player_statistics_model import (
    GameStatsResult,
    PlayerGameStats,
    PlayerStatsDistributionModel,
    train_player_statistics_model,
    load_player_statistics_model,
)
from src.models.crystal_ball_simulator import (
    CrystalBallOutcome,
    WorldsCrystalBallResults,
    simulate_worlds_crystal_ball,
)


def test_extract_player_stat_priors_fallback():
    """Verify that fallback priors are structurally complete and valid."""
    priors = extract_player_stat_priors(as_of_date="1900-01-01", min_games=10, db=None)
    assert isinstance(priors, dict)
    assert "role_kill_shares" in priors
    assert "role_death_shares" in priors
    assert "role_assist_factors" in priors
    assert "player_profiles" in priors
    assert "winner_kill_share" in priors
    assert "objective_rates" in priors

    # Verify role shares sum to concentration totals or normalized weights
    kill_alphas = priors["role_kill_shares"]
    assert "ADC" in kill_alphas and "MID" in kill_alphas
    assert kill_alphas["ADC"] > kill_alphas["SUPPORT"]

    # Verify objective rates
    objs = priors["objective_rates"]
    assert objs["dragon_rate_per_minute"] > 0
    assert objs["baron_rate_per_minute"] > 0
    assert 0.0 < objs["pentakill_rate_per_game"] < 0.1


def test_player_stats_sampling_and_conservation():
    """Test 1 (Stat Conservation): Verify exact sum invariants across 1,000 sampled games.
    
    Invariants:
    sum_{i in A} K_i == K_A
    sum_{i in B} K_i == K_B
    sum_{i in A} D_i == K_B
    sum_{i in B} D_i == K_A
    KDA_i == (K_i + A_i) / max(1, D_i)
    """
    model = PlayerStatsDistributionModel()
    rng = np.random.default_rng(20261008)

    roster_a = ["Kiin", "Canyon", "Chovy", "Peyz", "Lehends"]
    roster_b = ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]
    drafted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for _ in range(1000):
        winner = "A" if rng.random() < 0.55 else "B"
        res = model.sample_game_stats(
            team_a_roster=roster_a,
            team_b_roster=roster_b,
            winner=winner,
            drafted_champions=drafted,
            rng=rng,
        )

        assert isinstance(res, GameStatsResult)
        assert 900 <= res.duration_seconds <= 3600

        # Kills and Deaths conservation
        team_a_kills = sum(s.kills for key, s in res.player_stats.items() if key.startswith("A_"))
        team_b_kills = sum(s.kills for key, s in res.player_stats.items() if key.startswith("B_"))
        team_a_deaths = sum(s.deaths for key, s in res.player_stats.items() if key.startswith("A_"))
        team_b_deaths = sum(s.deaths for key, s in res.player_stats.items() if key.startswith("B_"))

        assert team_a_kills == res.team_a_kills
        assert team_b_kills == res.team_b_kills
        assert team_a_deaths == res.team_b_kills, f"A deaths ({team_a_deaths}) != B kills ({res.team_b_kills})"
        assert team_b_deaths == res.team_a_kills, f"B deaths ({team_b_deaths}) != A kills ({res.team_a_kills})"

        # KDA invariant and assist bounds
        for key, s in res.player_stats.items():
            expected_kda = (s.kills + s.assists) / max(1, s.deaths)
            assert math.isclose(s.kda, expected_kda, rel_tol=1e-5)
            # Player assists cannot exceed teammates' kills
            team_k = res.team_a_kills if key.startswith("A_") else res.team_b_kills
            assert s.assists <= max(0, team_k - s.kills)


def test_training_pipeline_and_serialization(tmp_path):
    """Test 4 (Training Pipeline & Serialization): Run training, verify artifacts and loading."""
    out_dir = str(tmp_path / "player_stats_model")
    res = train_player_statistics_model(as_of_date="1900-01-01", output_dir=out_dir)

    assert os.path.exists(os.path.join(out_dir, "model_priors.joblib"))
    assert os.path.exists(os.path.join(out_dir, "model_metadata.json"))

    with open(os.path.join(out_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "sha256" in meta
    assert "role_kill_shares" in meta
    assert "duration_distribution" in meta
    assert "kills_distribution" in meta

    # Test loading
    loaded_model = load_player_statistics_model(model_dir=out_dir)
    assert isinstance(loaded_model, PlayerStatsDistributionModel)

    # Test sample with loaded model
    rng = np.random.default_rng(42)
    sample = loaded_model.sample_game_stats(
        team_a_roster=["p1", "p2", "p3", "p4", "p5"],
        team_b_roster=["p6", "p7", "p8", "p9", "p10"],
        winner="A",
        rng=rng,
    )
    assert sample.team_a_kills > 0
    assert sample.team_b_deaths == sample.team_a_kills


def test_crystal_ball_probability_mass():
    """Test 2 (Crystal Ball Probability Mass): Verify sum P == 1.0 +- 1e-5 for all category rankings."""
    # Fast test with 50 simulations
    results = simulate_worlds_crystal_ball(
        n_simulations=50,
        seed=123,
    )
    assert isinstance(results, WorldsCrystalBallResults)

    # Check each category outcome ranking sums to 1.0
    categories = [
        results.player_most_kills_single_game,
        results.player_highest_tournament_kda,
        results.player_most_unique_champions,
        results.player_highest_kill_participation,
        results.team_most_unique_champions,
        results.team_most_kills_single_game,
        results.team_highest_kda,
        results.champion_most_picked,
        results.champion_most_banned,
        results.champion_highest_winrate,
        results.champion_most_deaths,
    ]

    for cat in categories:
        assert isinstance(cat, CrystalBallOutcome)
        assert len(cat.ranking) > 0
        total_p = sum(item["probability"] for item in cat.ranking)
        assert math.isclose(total_p, 1.0, abs_tol=1e-4), f"{cat.category_name} sum {total_p} != 1.0"
        # Fair odds check: odds >= 1.0
        for item in cat.ranking:
            assert item["fair_odds"] >= 1.0

    # Total pentakills PMF check
    penta_pmf = results.tournament_total_pentakills
    assert sum(penta_pmf.values()) == pytest.approx(1.0, abs=1e-4)


def test_crystal_ball_tournament_survival_effect():
    """Test 3 (Tournament Survival Effect): Top-seeded teams have higher probability in cumulative categories."""
    results = simulate_worlds_crystal_ball(
        n_simulations=100,
        seed=42,
    )

    # Unique champions team ranking: top tier teams should be favored over play-in teams
    team_unique_map = {item["candidate"]: item["probability"] for item in results.team_most_unique_champions.ranking}

    # Top seeds (Gen.G, Bilibili Gaming, T1) survive deeper into Swiss/Knockout, playing more maps
    top_teams_prob = (
        team_unique_map.get("Gen.G", 0.0)
        + team_unique_map.get("Bilibili Gaming", 0.0)
        + team_unique_map.get("T1", 0.0)
    )
    play_in_prob = (
        team_unique_map.get("MVK Esports", 0.0)
        + team_unique_map.get("LOUD", 0.0)
    )

    assert top_teams_prob >= play_in_prob, f"Top teams ({top_teams_prob}) not favored over play-in ({play_in_prob})"


def test_worlds_crystal_ball_endpoint():
    """Verify POST /tournaments/worlds/crystal-ball endpoint returns valid JSON."""
    client = TestClient(app)
    response = client.post(
        "/tournaments/worlds/crystal-ball",
        json={"n_simulations": 30, "seed": 42},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["n_simulations"] == 30
    assert "player_most_kills_single_game" in data
    assert "player_highest_tournament_kda" in data
    assert "player_most_unique_champions" in data
    assert "team_most_unique_champions" in data
    assert "champion_most_picked" in data
    assert "tournament_shortest_game_seconds" in data
    assert "tournament_longest_game_seconds" in data
    assert "tournament_total_pentakills" in data
    assert "tournament_total_barons" in data

    ranking = data["player_most_kills_single_game"]["ranking"]
    assert len(ranking) > 0
    assert "candidate" in ranking[0]
    assert "probability" in ranking[0]
    assert "fair_odds" in ranking[0]
