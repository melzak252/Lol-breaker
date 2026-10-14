"""Seed-dependent completed Americas formats, exercised with counterfactual teams."""
from collections import Counter
from itertools import combinations


from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_formats import simulate_tournament


def _run(profile, count, *, favored=None, probability=1.0, seed=81):
    teams = [f"seed-{i + 1}" for i in range(count)]
    spec = instantiate_profile(profile, teams)
    probabilities = {
        (a, b, bo): (float(a == favored) if favored in (a, b) else probability)
        for a, b in combinations(teams, 2) for bo in (1, 3, 5)
    }
    return simulate_tournament(
        spec, probabilities, simulations=1, seed=seed,
        score_distributions={3: [1, 0], 5: [1, 0, 0]},
    )


def test_spring_lower_seed_gets_one_life_but_can_win_without_final_reset():
    result = _run("lcs-six-team-spring-two-qualifiers", 6, favored="seed-6")
    assert result["champion_prob"]["seed-6"] == 1
    assert result["expected_series"] == 8
    matches = [m for m in result["trace"] if "seed-6" in (m["team_a"], m["team_b"])]
    assert [m["round"] for m in matches] == ["l2", "ls", "lf", "gf"]
    assert all(m["winner"] == "seed-6" for m in matches)
    ordinary = _run("lcs-six-team-spring-two-qualifiers", 6)
    losses = Counter(m["loser"] for m in ordinary["trace"])
    assert losses["seed-5"] == losses["seed-6"] == 1
    assert losses["seed-3"] == losses["seed-4"] == 2


def test_championship_byes_and_lower_starts_do_not_create_fake_rounds():
    result = _run("lcs-eight-team-championship", 8, favored="seed-8")
    assert result["champion_prob"]["seed-8"] == 1
    assert result["expected_series"] == 12
    opening = {m["round"]: {m["team_a"], m["team_b"]} for m in result["trace"]}
    assert opening["u1"] == {"seed-3", "seed-6"}
    assert opening["u2"] == {"seed-4", "seed-5"}
    assert {"seed-1", "seed-2"}.isdisjoint(opening["u1"] | opening["u2"])
    assert sum(result["advance_prob"]["playoffs"].values()) == 3


def test_lock_in_quarterfinals_and_semifinals_use_different_series_units():
    result = _run("lcs-lock-in-eight-knockout", 8)
    assert Counter(m["best_of"] for m in result["trace"]) == {3: 4, 5: 3}
    assert result["champion_prob"]["seed-1"] == 1


def test_cross_group_seeding_matches_preserve_both_teams_but_elimination_does_not():
    result = _run("lta-split2-cross-group-battles", 8)
    assert result["expected_series"] == 4
    qualified = result["advance_prob"]["cross_group"]
    assert {team for team, p in qualified.items() if p == 1} == {
        "seed-1", "seed-2", "seed-3", "seed-5", "seed-6", "seed-4",
    }
    assert qualified["seed-7"] == qualified["seed-8"] == 0


def test_promotion_play_in_loser_is_eliminated_without_entering_double_elimination():
    result = _run("lta-five-team-promotion", 5, favored="seed-5")
    play_in = next(m for m in result["trace"] if m["round"] == "play_in")
    assert {play_in["team_a"], play_in["team_b"]} == {"seed-4", "seed-5"}
    assert all("seed-4" not in (m["team_a"], m["team_b"])
               for m in result["trace"] if m["round"] != "play_in")
    assert result["champion_prob"]["seed-5"] == 1
    assert result["expected_series"] == 7


def test_regional_upper_loser_keeps_a_worlds_qualification_path():
    result = _run("lta-six-team-regional-championship", 6)
    first = next(m for m in result["trace"] if m["round"] == "juggernaut")
    assert {first["team_a"], first["team_b"]} == {"seed-1", "seed-4"}
    assert first["loser"] in {m["team_a"] for m in result["trace"] if m["round"] == "lf"}
    assert result["advance_prob"]["regional"][first["loser"]] == 1
    assert result["expected_series"] == 6


def test_fixed_round_swiss_plays_all_teams_then_only_two_last_chance_entrants():
    result = _run("lcs-2026-lock-in-fixed-swiss", 8, probability=0.5)
    swiss = [m for m in result["trace"] if m["stage"] == "swiss"]
    assert len(swiss) == 12
    appearances = Counter(t for m in swiss for t in (m["team_a"], m["team_b"]))
    assert set(appearances.values()) == {3}
    records = result["stage_records"]["swiss"]
    assert Counter(r["wins"] for r in records.values()) == {3: 1, 2: 3, 1: 3, 0: 1}
    last_chance = [m for m in result["trace"] if m["stage"] == "qualification"]
    assert len(last_chance) == 1
    decider = last_chance[0]
    assert decider["best_of"] == 1
    assert records[decider["team_a"]]["wins"] == records[decider["team_b"]]["wins"] == 1
    advancement = result["advance_prob"]["qualification"]
    assert advancement[decider["winner"]] == 1
    assert advancement[decider["loser"]] == 0
    assert sum(advancement.values()) == 6


def test_pick_and_play_uses_new_standings_to_select_each_week():
    result = _run("lta-2025-pick-and-play", 8)
    by_round = {
        round_number: [m for m in result["trace"] if m["round"] == round_number]
        for round_number in (1, 2, 3)
    }
    assert all(len(matches) == 4 for matches in by_round.values())
    assert {frozenset((m["team_a"], m["team_b"])) for m in by_round[1]} == {
        frozenset((f"seed-{i}", f"seed-{9-i}")) for i in range(1, 5)
    }
    assert {by_round[2][0]["team_a"], by_round[2][0]["team_b"]} == {"seed-4", "seed-8"}
    assert {by_round[3][0]["team_a"], by_round[3][0]["team_b"]} == {"seed-4", "seed-8"}
    assert all(r["wins"] + r["losses"] == 3
               for r in result["stage_records"]["pick_and_play"].values())


def test_split3_upper_losers_choose_from_dynamic_lower_survivors_by_seed():
    result = _run("lta-eight-team-split3-elimination", 8, favored="seed-4")
    matches = {m["round"]: m for m in result["trace"]}
    assert matches["u1"]["loser"] == "seed-1"
    assert matches["u2"]["loser"] == "seed-3"
    assert {matches["l3"]["team_a"], matches["l3"]["team_b"]} == {"seed-1", "seed-6"}
    assert {matches["l4"]["team_a"], matches["l4"]["team_b"]} == {"seed-3", "seed-5"}
    assert result["champion_prob"]["seed-4"] == 1
    assert result["expected_series"] == 10


def test_americas_cup_changes_to_bo5_after_opening_upper_round():
    result = _run("americas-cup-four-team-double-elimination", 4, favored="seed-4")
    assert Counter(m["best_of"] for m in result["trace"]) == {3: 2, 5: 4}
    assert result["champion_prob"]["seed-4"] == 1
