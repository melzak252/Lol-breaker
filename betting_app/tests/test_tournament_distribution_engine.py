"""Unit tests for tournament distribution and bracket simulation engine.

Verifies mathematical invariants:
1. Scoreline PMF mass conservation: sum(P(w_A, w_B)) == 1.0 +- 1.0e-7.
2. Side anti-symmetry: P(w_B, w_A | B, A) == P(w_A, w_B | A, B).
3. Swiss marginal advance conservation: sum(P(Advance)) == 8.0 in 16-team Swiss.
4. Finalist probability mass: sum(P(Final)) == 2.0.
5. Champion probability mass: sum(P(Champ)) == 1.0.
6. LP network flow feasibility via SciPy HiGHS.
7. Pair adapter probability bounds (unclipped fitted Glicko, clipped fixed proxies) and anti-symmetry.
8. Ground-truth extraction and RPS calculation accuracy.
"""
from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from src.analysis.tournament_metrics import (
    compute_advancement_brier,
    compute_champion_brier,
    compute_champion_log_loss,
    compute_finalists_brier,
    compute_scoreline_log_loss,
    compute_series_length_rps,
    compute_swiss_record_rps,
    extract_tournament_ground_truth,
    score_research_event,
    verify_network_flow_conservation,
)
from src.models.tournament_bracket_simulator import (
    SWISS_RECORD_BANDS,
    TournamentBracketSimulator,
    extract_swiss_record_probabilities,
)
from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_pair_adapter import (
    TEAM_ALIASES,
    TEAM_REGIONS,
    TournamentPairAdapter,
    normalize_team_id,
)
from src.models.tournament_scorelines import (
    ConditionalScorelineModel,
    build_score_distributions,
    compute_losing_map_pmf,
    compute_scoreline_pmf,
    compute_series_length_pmf,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "conf/base/tournament_manifest.json"


# =========================================================================
# 1. Scoreline PMF and Mathematical Invariants
# =========================================================================

@pytest.mark.parametrize("p", [0.01, 0.15, 0.35, 0.50, 0.72, 0.88, 0.99])
@pytest.mark.parametrize("bo", [1, 3, 5])
def test_scoreline_pmf_mass_conservation(p: float, bo: int) -> None:
    """Verify sum of scoreline probabilities is strictly 1.0."""
    pmf = compute_scoreline_pmf(p, bo)
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1.0e-7, f"Mass conservation violated for p={p}, bo={bo}: sum={total}"
    assert all(0.0 <= prob <= 1.0 for prob in pmf.values())


@pytest.mark.parametrize("p", [0.05, 0.25, 0.50, 0.68, 0.91])
@pytest.mark.parametrize("bo", [1, 3, 5])
def test_scoreline_pmf_side_anti_symmetry(p: float, bo: int) -> None:
    """Verify side symmetry: P(w_B, w_A | B, A) == P(w_A, w_B | A, B)."""
    pmf_ab = compute_scoreline_pmf(p, bo)
    pmf_ba = compute_scoreline_pmf(1.0 - p, bo)

    for (w1, w2), prob_ab in pmf_ab.items():
        prob_ba = pmf_ba.get((w2, w1), 0.0)
        assert abs(prob_ab - prob_ba) < 1.0e-7, (
            f"Anti-symmetry violated for score ({w1}, {w2}) at p={p}, bo={bo}: "
            f"P_ab={prob_ab}, P_ba={prob_ba}"
        )


@pytest.mark.parametrize("bo", [3, 5])
def test_losing_map_pmf_normalization(bo: int) -> None:
    """Verify losing map distribution sums to 1.0."""
    for p in [0.20, 0.50, 0.80]:
        l_pmf = compute_losing_map_pmf(p, bo)
        expected_len = bo // 2 + 1
        assert len(l_pmf) == expected_len
        assert abs(sum(l_pmf) - 1.0) < 1.0e-7


# =========================================================================
# 2. Pair Adapter Invariants and Backends
# =========================================================================

@pytest.mark.parametrize(
    "model",
    ["uniform", "glicko_format", "regional_glicko_proxy", "calibrated_regional_glicko_proxy"],
)
def test_pair_adapter_all_backends_bounds_and_anti_symmetry(model: str, tmp_path) -> None:
    """Verify real fitted/fixed predictors with legitimate prior CAL support."""
    source = tmp_path / "series_results.csv"
    rows = [
        "completed,2022-06-01T00:00:00Z,GEN,G2,1,3\n",
        "completed,2022-06-01T00:00:00Z,T1,BLG,2,3\n",
    ]
    for bo in (1, 3, 5):
        rows.extend([
            f"completed,2022-07-01T00:00:00Z,GEN,G2,1,{bo}\n",
            f"completed,2022-07-01T00:00:00Z,T1,BLG,2,{bo}\n",
            f"completed,2022-07-01T00:00:00Z,GEN,G2,2,{bo}\n",
        ])
    source.write_text(
        "status,scheduled_at_utc,team1,team2,winner_side,best_of\n" + "".join(rows),
        encoding="utf-8",
    )
    adapter = TournamentPairAdapter(source)
    teams = ["GEN", "T1", "BLG", "G2"]
    cutoff = "2023-10-18T00:00:00Z"

    table = adapter.predict_pair_table(teams, cutoff, model_kind=model)
    # Expected pairs: 4 * 3 * 3 = 36 ordered pairs across BO 1, 3, 5
    assert len(table) == 36

    for (a, b, bo), prob_ab in table.items():
        lower, upper = (0.0, 1.0) if model == "glicko_format" else (0.001, 0.999)
        assert lower <= prob_ab <= upper, f"Probability out of bounds: {prob_ab} for {a} vs {b}"
        prob_ba = table[(b, a, bo)]
        assert abs(prob_ab + prob_ba - 1.0) < 1.0e-7, (
            f"Anti-symmetry violated for {model} {a} vs {b}: {prob_ab} + {prob_ba} != 1.0"
        )


def test_pair_adapter_temporal_cutoff_isolation() -> None:
    """Verify that earlier cutoff excludes later matches."""
    adapter = TournamentPairAdapter()
    # At early 2021 vs late 2024
    r_early = adapter.compute_team_ratings("2021-01-01T00:00:00Z")
    r_late = adapter.compute_team_ratings("2024-10-01T00:00:00Z")
    # Teams have more experience and lower RD in late 2024
    assert r_late["T1"].rd < r_early["T1"].rd


# =========================================================================
# 3. Bracket Simulator and Mass Conservation
# =========================================================================

def test_bracket_simulator_swiss_mass_conservation() -> None:
    """Verify Swiss bracket simulation preserves exact participant flow."""
    adapter = TournamentPairAdapter()
    teams = [
        "GEN", "BLG", "JDG", "NRG", "G2", "T1", "KT", "WBG",
        "LNG", "DK", "FNC", "MAD", "C9", "TL", "BDS", "GAM"
    ]
    cutoff = "2023-10-18T00:00:00Z"
    spec = instantiate_profile("worlds-2023-swiss-knockout", teams)
    probs = adapter.predict_pair_table(teams, cutoff, model_kind="calibrated_regional_glicko_proxy")

    simulator = TournamentBracketSimulator(simulations=50, seed=123)
    results = simulator.simulate(spec, probs)

    # 1. Champion probability sums to 1.0
    assert abs(sum(results["champion_prob"].values()) - 1.0) < 1.0e-7

    # 2. Finalist probability sums to 2.0
    assert abs(sum(results["final_prob"].values()) - 2.0) < 1.0e-7

    # 3. Exactly 8.0 teams advance from 16-team Swiss
    assert abs(sum(results["advance_prob"]["swiss"].values()) - 8.0) < 1.0e-7

    # 4. Swiss record distribution sums to 1.0 for each team
    for team in teams:
        assert team in results["swiss_record_prob"]
        rec_sum = sum(results["swiss_record_prob"][team].values())
        assert abs(rec_sum - 1.0) < 1.0e-7


def test_bracket_simulator_double_elimination_conservation() -> None:
    """Verify double elimination bracket simulation preserves probability mass."""
    adapter = TournamentPairAdapter()
    teams = ["GEN", "JDG", "MAD", "C9", "T1", "BLG", "G2", "GG"]
    cutoff = "2023-05-08T00:00:00Z"
    spec = instantiate_profile("msi-2023-2026-double-elimination", teams)
    probs = adapter.predict_pair_table(teams, cutoff, model_kind="calibrated_regional_glicko_proxy")

    simulator = TournamentBracketSimulator(simulations=50, seed=456)
    results = simulator.simulate(spec, probs)

    assert abs(sum(results["champion_prob"].values()) - 1.0) < 1.0e-7
    assert abs(sum(results["final_prob"].values()) - 2.0) < 1.0e-7


def _small_cup(best_of=1):
    return {
        "version": 1, "id": "small-cup", "teams": list("ABCD"),
        "stages": [{"id": "cup", "kind": "single_elimination",
                    "entrants": list("ABCD"), "best_of": best_of}],
        "champion": {"stage": "cup", "group": "all", "rank": 1},
    }


def _cup_probabilities(spec, p=0.5):
    return {(a, b, spec["stages"][0]["best_of"]): p
            for a, b in combinations(spec["teams"], 2)}


@pytest.mark.parametrize("fault", ["missing", "unknown_opponent", "self_pair", "contradictory_reverse"])
def test_bracket_rejects_invalid_pair_coverage(fault):
    spec = _small_cup()
    probabilities = _cup_probabilities(spec)
    if fault == "missing":
        del probabilities["A", "D", 1]
    elif fault == "unknown_opponent":
        probabilities["A", "OUTSIDE", 1] = 0.5
    elif fault == "self_pair":
        probabilities["A", "A", 1] = 0.5
    else:
        probabilities["B", "A", 1] = 0.7
    with pytest.raises(ValueError):
        TournamentBracketSimulator(simulations=2).simulate(spec, probabilities)


def test_bracket_rejects_duplicate_rows_from_pair_mapping():
    class DuplicateRows(dict):
        def items(self):
            yield from super().items()
            yield ("A", "B", 1), self["A", "B", 1]

    spec = _small_cup()
    with pytest.raises(ValueError):
        TournamentBracketSimulator(simulations=2).simulate(
            spec, DuplicateRows(_cup_probabilities(spec)),
        )


def test_bracket_consistent_reverse_preserves_outcome_distribution():
    spec = _small_cup()
    probabilities = _cup_probabilities(spec, 0.7)
    ordered = {**probabilities, **{(b, a, bo): 1 - p for (a, b, bo), p in probabilities.items()}}
    simulator = TournamentBracketSimulator(simulations=100, seed=47)
    canonical = simulator.simulate(spec, probabilities)
    bidirectional = simulator.simulate(spec, ordered)
    assert bidirectional["champion_prob"] == canonical["champion_prob"]
    assert bidirectional["joint_final_prob"] == canonical["joint_final_prob"]


@pytest.mark.parametrize("mass", [0.6, 1.4])
def test_swiss_record_extraction_rejects_structural_mass_error(mass):
    ranks = [mass / 16] * 16
    with pytest.raises(ValueError):
        extract_swiss_record_probabilities({"all": {"A": ranks}})


def test_shared_latent_selection_persists_from_semifinal_to_final():
    spec = _small_cup()
    n = 2000
    draws = {team: np.zeros(n) for team in spec["teams"]}
    draws["A"] = np.tile([-40.0, 40.0], n // 2)
    simulator = TournamentBracketSimulator(simulations=n, seed=17)
    independent = simulator.simulate(spec, _cup_probabilities(spec))
    coherent = simulator.simulate(spec, _cup_probabilities(spec), team_strength_draws=draws)
    # All A-opponent integrated pair probabilities are .5. Winning selects
    # the same high latent for the next match, rather than drawing it again.
    assert coherent["final_prob"]["A"] == pytest.approx(0.5, abs=0.001)
    assert coherent["champion_prob"]["A"] == pytest.approx(0.5, abs=0.001)
    assert independent["champion_prob"]["A"] == pytest.approx(0.25, abs=0.04)


@pytest.mark.parametrize("conditional", [False, True])
def test_supplied_score_law_controls_actual_stopped_scores(conditional):
    spec = _small_cup(3)
    kwargs = ({"conditional_score_distributions": {
        "probability_bins": [0, 0.5, 1], "distributions": {3: [[1, 0], [0, 1]]},
    }} if conditional else {"scorelines": {3: [0, 1]}})
    result = TournamentBracketSimulator(simulations=4).simulate(
        spec, _cup_probabilities(spec, 1.0), **kwargs,
    )
    assert result["champion_prob"]["A"] == 1.0
    for match in result["trace"]:
        assert sorted([match["score_a"], match["score_b"]]) == [1, 2]


def test_coherent_score_loss_integrates_winner_and_score_law_together():
    spec = _small_cup(3)
    spec["teams"] = spec["stages"][0]["entrants"] = ["A", "B"]
    # Half the frozen latent draws imply p=.1, the other half p=.9. High
    # winner probability means a sweep, low winner probability a deciding map.
    # Therefore P(A wins 2-1)=.5*.1=.05, not K(E[p]) * E[p].
    forecast = TournamentBracketSimulator(simulations=4, seed=73).simulate(
        spec, {("A", "B", 3): 0.5},
        conditional_score_distributions={
            "probability_bins": [0, 0.5, 1],
            "distributions": {3: [[0, 1], [1, 0]]},
        },
        team_strength_draws={
            "A": [-math.log(9), math.log(9), -math.log(9), math.log(9)],
            "B": [0, 0, 0, 0],
        },
    )
    event = {"id": spec["id"], "edition_id": "analytic-edition", "year": 2026,
             "teams": ["A", "B"], "spec": spec, "scope": "full_tournament"}
    labels = {
        "event_id": event["id"], "edition_id": event["edition_id"],
        "champion": "A", "finalists": ["A", "B"], "qualifiers": {},
        "placement_bands": None, "swiss_records": None,
        "matches": [{"team_a": "A", "team_b": "B", "best_of": 3,
                     "wins_a": 2, "wins_b": 1, "date": "2026-06-02", "match_id": "observed"}],
    }
    scores = score_research_event(
        event, forecast, labels, pair_table=forecast["pair_table"],
        scoreline_model=ConditionalScorelineModel(),
    )
    # The supplied continuous law must not overwrite the actual compiled law.
    assert scores["scoreline_log_loss"] == pytest.approx(-math.log(0.05))
    assert scores["series_length_rps"] == pytest.approx(0.9 ** 2)


def test_qualification_only_phase_has_no_fabricated_champion_or_final():
    spec = {
        "version": 1, "id": "qualification", "teams": list("ABCD"),
        "stages": [{"id": "league", "kind": "round_robin", "entrants": list("ABCD"),
                    "best_of": 1, "tiebreakers": ["wins", "random"], "advance": 2}],
    }
    result = TournamentBracketSimulator(simulations=40, seed=53).simulate(
        spec, {(a, b, 1): 0.5 for a, b in combinations(spec["teams"], 2)},
    )
    assert result["champion_prob"] is None
    assert result["final_prob"] is None
    assert result["joint_final_prob"] is None
    assert sum(result["advance_prob"]["league"].values()) == pytest.approx(2.0)
    assert sum(result["joint_advance_prob"]["league"].values()) == pytest.approx(1.0)


def test_official_tied_placement_probabilities_retain_whole_field():
    spec = _small_cup()
    spec["placement_bands"] = [
        {"label": label, "placements": [
            {"stage": "cup", "group": "all", "rank": rank} for rank in ranks]}
        for label, ranks in [("1", [1]), ("2", [2]), ("3-4", [3, 4])]
    ]
    result = TournamentBracketSimulator(simulations=40, seed=29).simulate(
        spec, _cup_probabilities(spec),
    )
    for team in spec["teams"]:
        assert sum(result["placement_prob"][team].values()) == pytest.approx(1.0)
    assert sum(row["3-4"] for row in result["placement_prob"].values()) == pytest.approx(2.0)
    assert result["placement_prob"]["A"]["1"] == result["champion_prob"]["A"]
    assert result["placement_prob"]["A"]["1"] + result["placement_prob"]["A"]["2"] == pytest.approx(
        result["final_prob"]["A"],
    )


# =========================================================================
# 4. Network Flow Feasibility
# =========================================================================

def test_network_flow_conservation_highs_solver() -> None:
    """Verify SciPy HiGHS LP solver correctly verifies flow conservation."""
    # Synthetic consistent marginals: Champ <= Final <= Advance <= 1.0
    champ_prob = {"A": 0.60, "B": 0.40}
    final_prob = {"A": 1.00, "B": 1.00}
    advance_prob = {"A": 1.00, "B": 1.00}

    res = verify_network_flow_conservation(champ_prob, final_prob, advance_prob)
    assert res["feasible"] is True
    assert res["status"] == 0
    assert res["max_flow_violation"] == 0.0


def test_network_flow_conservation_detects_violations() -> None:
    """Verify SciPy HiGHS LP solver detects illegal flow leakage."""
    # Leaking flow: Champ > Final
    champ_prob = {"A": 0.80, "B": 0.20}
    final_prob = {"A": 0.50, "B": 1.50}

    res = verify_network_flow_conservation(champ_prob, final_prob, tolerance=1.0e-5)
    assert res["feasible"] is False
    assert res["max_flow_violation"] > 0.0


# =========================================================================
# 5. Ground Truth and Scoring Rules
# =========================================================================

def test_ground_truth_extraction_worlds_2023() -> None:
    """Verify historical ground-truth extraction for Worlds 2023."""
    gt = extract_tournament_ground_truth("2023 Season World Championship/Main Event")
    assert gt.champion == "T1"
    assert set(gt.finalists) == {"T1", "WBG"}
    assert len(gt.swiss_qualifiers) == 8
    assert "T1" in gt.swiss_qualifiers
    assert "WBG" in gt.swiss_qualifiers
    assert gt.swiss_records["GEN"] == "3-0"
    assert gt.swiss_records["JDG"] == "3-0"
    assert gt.swiss_records["TL"] == "0-3"


def test_ground_truth_extraction_msi_2023() -> None:
    """Verify historical ground-truth extraction for MSI 2023."""
    gt = extract_tournament_ground_truth("2023 Mid-Season Invitational")
    assert gt.champion == "JDG"
    assert set(gt.finalists) == {"BLG", "JDG"}


def test_ranked_probability_score_calculation() -> None:
    """Verify RPS calculation on controlled distributions."""
    # Perfect forecast
    perfect_dist = {"T1": {"3-0": 1.0, "3-1": 0.0, "3-2": 0.0, "2-3": 0.0, "1-3": 0.0, "0-3": 0.0}}
    actual = {"T1": "3-0"}
    rps_perfect = compute_swiss_record_rps(perfect_dist, actual)
    assert rps_perfect == pytest.approx(0.0, abs=1.0e-9)

    # Worst possible forecast (placed all mass on opposite tail 0-3 when true was 3-0)
    worst_dist = {"T1": {"3-0": 0.0, "3-1": 0.0, "3-2": 0.0, "2-3": 0.0, "1-3": 0.0, "0-3": 1.0}}
    rps_worst = compute_swiss_record_rps(worst_dist, actual)
    # (1-0)^2 * 5 / 5 = 1.0
    assert rps_worst == pytest.approx(1.0, abs=1.0e-9)
