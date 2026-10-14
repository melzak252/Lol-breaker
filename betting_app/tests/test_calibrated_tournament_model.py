"""Unit tests for the Calibrated Tournament Simulation Engine."""

from __future__ import annotations

import math
import pytest

from src.models.calibrated_tournament_model import (
    CalibratedTournamentEngine,
    calibrate_pairwise_probability,
    series_win_probability_from_game_prob,
)


def test_series_win_probability_combinatorics():
    """Verify exact Negative Binomial series projection across Bo1, Bo3, Bo5."""
    # Bo1 is identity
    assert series_win_probability_from_game_prob(0.60, 1) == pytest.approx(0.60)
    assert series_win_probability_from_game_prob(0.50, 1) == pytest.approx(0.50)

    # Bo3: p^2 * (3 - 2p)
    p = 0.60
    expected_bo3 = p**2 * (3.0 - 2.0 * p)
    assert series_win_probability_from_game_prob(p, 3) == pytest.approx(expected_bo3)
    assert series_win_probability_from_game_prob(0.50, 3) == pytest.approx(0.50)

    # Bo5: p^3 * (10 - 15p + 6p^2)
    expected_bo5 = p**3 * (10.0 - 15.0 * p + 6.0 * p**2)
    assert series_win_probability_from_game_prob(p, 5) == pytest.approx(expected_bo5)
    assert series_win_probability_from_game_prob(0.50, 5) == pytest.approx(0.50)

    # Monotonic amplification for favorite
    assert series_win_probability_from_game_prob(0.65, 5) > series_win_probability_from_game_prob(0.65, 3) > 0.65


def test_calibrate_pairwise_probability_anti_symmetry():
    """Verify that calibrated pairwise probability strictly preserves anti-symmetry."""
    for p in [0.20, 0.35, 0.50, 0.65, 0.85, 0.95]:
        p_ab = calibrate_pairwise_probability(p, cap=0.88, temperature=1.12, dampening=0.08)
        p_ba = calibrate_pairwise_probability(1.0 - p, cap=0.88, temperature=1.12, dampening=0.08)
        assert p_ab + p_ba == pytest.approx(1.0, abs=1e-6)


def test_calibrate_pairwise_probability_variance_ceiling():
    """Verify that extreme pairwise probabilities are truncated to cap."""
    cap = 0.88
    # Extreme favorite
    p_extreme = calibrate_pairwise_probability(0.99, cap=cap)
    assert p_extreme <= cap

    # Extreme underdog
    p_underdog = calibrate_pairwise_probability(0.01, cap=cap)
    assert p_underdog >= 1.0 - cap


def test_regional_strength_discounting():
    """Verify that regional discounting correctly shifts probability for major vs lower tier."""
    # Symmetrical baseline (50% map probability)
    p_base = 0.50
    # LCK (+0.463) vs Minor Tier 1 (-0.456)
    p_adj = calibrate_pairwise_probability(
        p_base,
        region_a="LCK",
        region_b="Minor Tier 1",
        regional_gamma=0.70,
        regional_offsets={"LCK": 0.463, "Minor Tier 1": -0.456},
    )
    # LCK should be favored significantly over minor region
    assert p_adj > 0.60

    # Opposite direction must be anti-symmetric
    p_rev = calibrate_pairwise_probability(
        1.0 - p_base,
        region_a="Minor Tier 1",
        region_b="LCK",
        regional_gamma=0.70,
        regional_offsets={"LCK": 0.463, "Minor Tier 1": -0.456},
    )
    assert p_adj + p_rev == pytest.approx(1.0, abs=1e-6)


def test_calibrated_tournament_simulation_end_to_end():
    """Execute end-to-end tournament simulation and verify probability mass conservation."""
    engine = CalibratedTournamentEngine(
        pairwise_cap=0.88,
        temperature=1.12,
        entropy_dampening=0.08,
    )

    teams = ["Gen.G", "BLG", "T1", "TES", "G2", "TL", "FNC", "PSG"]
    base_map_probs = {
        ("Gen.G", "BLG"): 0.53,
        ("Gen.G", "T1"): 0.56,
        ("Gen.G", "TES"): 0.60,
        ("Gen.G", "G2"): 0.70,
        ("Gen.G", "TL"): 0.80,
        ("Gen.G", "FNC"): 0.75,
        ("Gen.G", "PSG"): 0.82,
        ("BLG", "T1"): 0.53,
        ("BLG", "TES"): 0.57,
        ("BLG", "G2"): 0.68,
        ("BLG", "TL"): 0.78,
        ("BLG", "FNC"): 0.74,
        ("BLG", "PSG"): 0.80,
        ("T1", "TES"): 0.54,
        ("T1", "G2"): 0.65,
        ("T1", "TL"): 0.75,
        ("T1", "FNC"): 0.70,
        ("T1", "PSG"): 0.78,
        ("TES", "G2"): 0.62,
        ("TES", "TL"): 0.72,
        ("TES", "FNC"): 0.68,
        ("TES", "PSG"): 0.75,
        ("G2", "TL"): 0.60,
        ("G2", "FNC"): 0.58,
        ("G2", "PSG"): 0.65,
        ("TL", "FNC"): 0.48,
        ("TL", "PSG"): 0.55,
        ("FNC", "PSG"): 0.58,
    }

    result = engine.simulate(
        spec_or_profile_id="msi-2023-2026-double-elimination",
        teams=teams,
        base_match_probabilities=base_map_probs,
        simulations=5000,
        seed=42,
    )

    # 1. Mass conservation verification
    audit = result.verify_probability_conservation()
    assert audit["is_valid"] is True
    assert audit["champion_probability_sum"] == pytest.approx(1.0, abs=1e-4)
    assert audit["finalists_probability_sum"] == pytest.approx(2.0, abs=1e-4)

    # 2. Ranking order: Gen.G and BLG must lead champion probabilities
    assert result.champion_prob["Gen.G"] > result.champion_prob["T1"]
    assert result.champion_prob["BLG"] > result.champion_prob["G2"]

    # 3. Finals reach calibration: Gen.G finals reach should not exceed 75%
    assert result.final_prob["Gen.G"] < 0.75

    # 4. Scoring methods
    ll, brier = result.score_champion(actual_champion="Gen.G")
    assert 0.0 < ll < 2.5
    assert 0.0 < brier < 1.0


@pytest.mark.parametrize("best_of,slope", [(1, 1.0), (3, 1.34), (5, 1.62)])
def test_transformed_regional_proxy_matches_independent_formula(tmp_path, best_of, slope):
    """The honestly named proxy is a fixed transform, not a native checkpoint."""
    from src.models.tournament_pair_adapter import TournamentPairAdapter

    source = tmp_path / "series_results.csv"
    source.write_text(
        "status,scheduled_at_utc,team1,team2,winner_side,best_of\n",
        encoding="utf-8",
    )
    adapter = TournamentPairAdapter(source)
    regional = adapter.predict_pair_table(
        ["GEN", "GAM"], "2023-10-18T00:00:00Z",
        model_kind="regional_glicko_proxy", best_ofs=[best_of],
    )
    transformed = adapter.predict_pair_table(
        ["GEN", "GAM"], "2023-10-18T00:00:00Z",
        model_kind="calibrated_regional_glicko_proxy", best_ofs=[best_of],
    )

    # No history => equal Glicko states; the only logit difference is regional.
    regional_logit = slope * 0.70 * (0.463 - (-0.456)) / 1.08
    expected_regional = 1.0 / (1.0 + math.exp(-regional_logit))
    expected_transformed = min(
        0.88, max(0.12, 0.92 / (1.0 + math.exp(-regional_logit / 1.12)) + 0.04),
    )
    assert regional["GEN", "GAM", best_of] == pytest.approx(expected_regional, abs=1e-12)
    assert transformed["GEN", "GAM", best_of] == pytest.approx(expected_transformed, abs=1e-12)
    assert transformed["GAM", "GEN", best_of] == pytest.approx(1.0 - expected_transformed, abs=1e-12)
    assert 0.5 < expected_transformed < expected_regional
    for model in ("regional_glicko_proxy", "calibrated_regional_glicko_proxy"):
        receipt = adapter.predictor_receipt(
            ["GEN", "GAM"], "2023-10-18T00:00:00Z", model, best_ofs=[best_of],
        )
        assert receipt["predictor_kind"] == "glicko_derived_proxy"
        assert receipt["configuration"]["format_slopes"] == {str(best_of): slope}
        assert "calibration" not in receipt["configuration"]
        assert "calibration_policy" not in receipt["configuration"]
    with pytest.raises(ValueError, match="unavailable_prior_calibrated_glicko_formats"):
        adapter.predict_pair_table(
            ["GEN", "GAM"], "2023-10-18T00:00:00Z", "glicko_format",
            best_ofs=[best_of],
        )
