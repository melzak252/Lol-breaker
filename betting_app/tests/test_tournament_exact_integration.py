"""Independent path oracles for weighted tournament integration, not MC hit tests."""
from itertools import combinations, product
import json

import pytest

from src.models.tournament_formats import simulate_tournament
from src.models.tournament_exact import UnsupportedIntegrationError


def ref(stage, rank):
    return {"stage": stage, "group": "all", "rank": rank}


def pair_key(teams):
    return json.dumps(sorted(teams), separators=(",", ":"))


def probabilities(teams, formats=(1, 3, 5)):
    return {(a, b, bo): 0.19 + 0.11 * i + 0.03 * formats.index(bo)
            for i, (a, b) in enumerate(combinations(teams, 2)) for bo in formats}


def pwin(table, a, b, bo):
    return table[a, b, bo] if (a, b, bo) in table else 1 - table[b, a, bo]


def knockout_spec(teams):
    return {"version": 1, "id": "oracle", "teams": teams,
            "stages": [{"id": "ko", "kind": "single_elimination", "entrants": teams,
                        "best_of": 5, "seeding": "ordered"}],
            "champion": ref("ko", 1),
            "placement_bands": [{"label": "winner", "placements": [ref("ko", 1)]},
                                {"label": "runner-up", "placements": [ref("ko", 2)]},
                                *([{"label": "semifinal", "placements": [ref("ko", r) for r in range(3, len(teams) + 1)]}]
                                  if len(teams) > 2 else [])]}


def oracle_knockout(teams, opening, table):
    paths = []
    seeds = {team: i for i, team in enumerate(teams)}
    def visit(pairs, eliminated, weight):
        laws = []
        for a, b in pairs:
            if a is None or b is None:
                laws.append([(a or b, None, 1.0)])
            else:
                probability = pwin(table, a, b, 5)
                laws.append([(a, b, probability), (b, a, 1 - probability)])
        for outcomes in product(*laws):
            mass = weight
            for _, _, chance in outcomes:
                mass *= chance
            winners = [w for w, _, _ in outcomes]
            losers = sorted((l for _, l, _ in outcomes if l is not None), key=seeds.__getitem__)
            buckets = eliminated + [losers]
            if len(winners) == 1:
                ranking = winners + [team for bucket in reversed(buckets) for team in bucket]
                paths.append((mass, ranking, tuple(pairs[0])))
            else:
                visit(list(zip(winners[::2], winners[1::2])), buckets, mass)
    visit(opening, [], 1.0)
    return paths


@pytest.mark.parametrize("teams,opening", [
    (list("ABCD"), [("A", "B"), ("C", "D")]),
    (list("ABC"), [("A", None), ("B", "C")]),
])
def test_knockout_dp_matches_independent_unequal_path_oracle_with_byes(teams, opening):
    spec = knockout_spec(teams)
    table = probabilities(teams)
    exact = simulate_tournament(spec, table, integration="exact")
    paths = oracle_knockout(teams, opening, table)
    for team in teams:
        for rank in range(len(teams)):
            expected = sum(mass for mass, ranking, _ in paths if ranking[rank] == team)
            assert exact["stage_rank_prob"]["ko"]["all"][team][rank] == pytest.approx(expected)
        assert exact["champion_prob"][team] == pytest.approx(sum(m for m, r, _ in paths if r[0] == team))
        assert exact["final_prob"][team] == pytest.approx(sum(m for m, _, f in paths if team in f))
        assert sum(exact["placement_prob"][team].values()) == pytest.approx(1)
    for pair in combinations(teams, 2):
        assert exact["joint_final_prob"][pair_key(pair)] == pytest.approx(
            sum(m for m, _, f in paths if set(f) == set(pair)))
    assert exact["expected_series"] == len(teams) - 1
    assert exact["hits"] is None
    assert exact["simulations"] is None
    assert exact["integration"]["iid"] is False


def test_gsl_formats_qualifier_joint_matches_independent_five_match_oracle():
    teams = list("ABCD")
    table = probabilities(teams)
    spec = {"version": 1, "id": "gsl", "teams": teams,
            "stages": [{"id": "gsl", "kind": "gsl", "entrants": teams,
                        "best_of": 1, "decider_best_of": 3, "advance": 2}],
            "champion": ref("gsl", 1)}
    expected = dict.fromkeys(teams, 0.0)
    joint = {}
    ranks = {team: [0.0] * 4 for team in teams}
    for outcomes in product((False, True), repeat=5):
        mass = 1.0
        def match(a, b, bo, choice):
            nonlocal mass
            probability = pwin(table, a, b, bo)
            mass *= probability if choice else 1 - probability
            return (a, b) if choice else (b, a)
        a = match("A", "D", 1, outcomes[0])
        b = match("B", "C", 1, outcomes[1])
        first, candidate = match(a[0], b[0], 3, outcomes[2])
        survivor, fourth = match(a[1], b[1], 3, outcomes[3])
        second, third = match(candidate, survivor, 3, outcomes[4])
        ranking = [first, second, third, fourth]
        expected[first] += mass
        key = pair_key((first, second))
        joint[key] = joint.get(key, 0.0) + mass
        for rank, team in enumerate(ranking):
            ranks[team][rank] += mass
    exact = simulate_tournament(spec, table, integration="exact")
    assert exact["champion_prob"] == pytest.approx(expected)
    assert exact["joint_advance_prob"]["gsl"] == pytest.approx(joint)
    for team in teams:
        assert exact["stage_rank_prob"]["gsl"]["all"][team] == pytest.approx(ranks[team])
    assert exact["final_prob"] is None  # A group winner is not an invented grand final.
    assert exact["expected_series"] == pytest.approx(5)


def test_no_reset_double_elimination_rematch_is_one_final():
    teams = ["A", "B"]
    spec = {"version": 1, "id": "de", "teams": teams,
            "stages": [{"id": "de", "kind": "double_elimination", "entrants": teams,
                        "best_of": 5, "final_reset": "single_final"}], "champion": ref("de", 1)}
    result = simulate_tournament(spec, {("A", "B", 5): 0.73}, integration="exact")
    assert result["champion_prob"]["A"] == pytest.approx(0.73)
    assert result["expected_series"] == pytest.approx(2)
    assert result["joint_final_prob"][pair_key(teams)] == pytest.approx(1)
    spec["stages"][0]["final_reset"] = "if_lower_wins"
    reset = simulate_tournament(spec, {("A", "B", 5): 0.73}, integration="exact")
    assert reset["champion_prob"]["A"] == pytest.approx(0.73 ** 2 + 2 * 0.73 ** 2 * 0.27)
    assert reset["expected_series"] == pytest.approx(2 + 2 * 0.73 * 0.27)


def test_hybrid_deterministic_swiss_prefix_reports_bounded_moments():
    teams = list("ABCD")
    spec = {"version": 1, "id": "tiny-hybrid", "teams": teams,
            "stages": [{"id": "swiss", "kind": "swiss", "entrants": teams,
                        "best_of": 1, "rounds": 1, "initial_pairings": [["A", "B"], ["C", "D"]],
                        "tiebreakers": ["wins", "seed"], "advance": 2},
                       {"id": "ko", "kind": "single_elimination", "best_of": 5,
                        "entrants": [ref("swiss", 1), ref("swiss", 2)]}], "champion": ref("ko", 1)}
    table = probabilities(teams)
    table["A", "B", 1] = table["C", "D", 1] = 1.0
    result = simulate_tournament(spec, table, simulations=7, seed=6, integration="hybrid")
    assert result["champion_prob"]["A"] == pytest.approx(pwin(table, "A", "C", 5))
    assert result["final_prob"] == pytest.approx({"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.0})
    leaf = result["integration"]["prefix_moments"]["champion_prob"]["A"]
    assert leaf["n"] == 7
    assert leaf["sum"] == pytest.approx(7 * result["champion_prob"]["A"])
    assert leaf["sum_squares"] == pytest.approx(7 * result["champion_prob"]["A"] ** 2)
    assert result["integration"]["effective_sampling_unit"] == "independent_prefix"
    assert result["hits"] is None
    with pytest.raises(UnsupportedIntegrationError, match="adaptive Swiss"):
        simulate_tournament(spec, table, integration="exact")


def test_exact_finite_strength_target_is_uniform_mixture_not_iid_samples():
    spec = knockout_spec(["A", "B"])
    table = {("A", "B", 5): 0.5}
    draws = {"A": [0.0, 2.0], "B": [0.0, 0.0]}
    result = simulate_tournament(spec, table, simulations=2, team_strength_draws=draws, integration="exact")
    import math
    assert result["champion_prob"]["A"] == pytest.approx((0.5 + 1 / (1 + math.exp(-2))) / 2)
    assert result["integration"]["latent_target"] == "uniform finite-vector mixture"
    assert result["integration"]["iid"] is False


@pytest.mark.parametrize("extra", [{"state": {}}, {"observables": []}])
def test_explicit_exact_rejects_unsupported_state_even_if_empty(extra):
    with pytest.raises(UnsupportedIntegrationError, match="checkpoint"):
        simulate_tournament(knockout_spec(list("ABCD")), probabilities(list("ABCD")), integration="exact", **extra)


def test_exact_rejects_random_tie_order_without_silent_sampling():
    spec = knockout_spec(list("ABCD"))
    spec["stages"][0]["tiebreakers"] = ["random"]
    with pytest.raises(UnsupportedIntegrationError, match="random draws/ties"):
        simulate_tournament(spec, probabilities(list("ABCD")), integration="exact")

def test_fixed_graph_mixed_formats_and_tied_official_band_matches_path_oracle():
    teams = list("ABCD")
    table = probabilities(teams)
    stage = {"id": "graph", "kind": "graph", "entrants": teams, "best_of": 5,
             "matches": [{"id": "left", "a": "A", "b": "B", "best_of": 1},
                         {"id": "right", "a": "C", "b": "D", "best_of": 3},
                         {"id": "final", "a": {"winner": "left"}, "b": {"winner": "right"}}],
             "ranking": [{"winner": "final"}, {"loser": "final"},
                         {"loser": "left"}, {"loser": "right"}]}
    spec = {"version": 1, "id": "mixed-graph", "teams": teams, "stages": [stage],
            "champion": ref("graph", 1), "final": {"stage": "graph", "round": "final"},
            "placement_bands": [{"label": "first", "placements": [ref("graph", 1)]},
                                {"label": "second", "placements": [ref("graph", 2)]},
                                {"label": "third-fourth", "placements": [ref("graph", 3), ref("graph", 4)]}]}
    expected = dict.fromkeys(teams, 0.0)
    finalists = {}
    tied = dict.fromkeys(teams, 0.0)
    for left_wins, right_wins, final_wins in product((False, True), repeat=3):
        a, left_loser = ("A", "B") if left_wins else ("B", "A")
        b, right_loser = ("C", "D") if right_wins else ("D", "C")
        mass = (table["A", "B", 1] if left_wins else 1 - table["A", "B", 1])
        mass *= table["C", "D", 3] if right_wins else 1 - table["C", "D", 3]
        mass *= pwin(table, a, b, 5) if final_wins else 1 - pwin(table, a, b, 5)
        expected[a if final_wins else b] += mass
        key = pair_key((a, b))
        finalists[key] = finalists.get(key, 0.0) + mass
        tied[left_loser] += mass
        tied[right_loser] += mass
    result = simulate_tournament(spec, table, integration="exact")
    assert result["champion_prob"] == pytest.approx(expected)
    for pair, value in finalists.items():
        assert result["joint_final_prob"][pair] == pytest.approx(value)
    for team in teams:
        assert result["placement_prob"][team]["third-fourth"] == pytest.approx(tied[team])
    assert pair_key(("A", "B")) in result["structural_zero_categories"]["joint_final"]
    assert result["joint_final_prob"][pair_key(("A", "B"))] == 0

def test_tiny_hybrid_and_exact_agree_on_fixed_deterministic_gsl_prefix():
    teams = list("ABCD")
    spec = {"version": 1, "id": "gsl-ko", "teams": teams,
            "stages": [{"id": "group", "kind": "gsl", "entrants": teams,
                        "best_of": 1, "decider_best_of": 1, "advance": 2},
                       {"id": "ko", "kind": "single_elimination", "best_of": 5,
                        "entrants": [ref("group", 1), ref("group", 2)]}],
            "champion": ref("ko", 1)}
    table = probabilities(teams)
    table.update({(a, b, 1): 1.0 for a, b in combinations(teams, 2)})
    exact = simulate_tournament(spec, table, integration="exact")
    hybrid = simulate_tournament(spec, table, integration="hybrid", simulations=9)
    assert hybrid["champion_prob"] == pytest.approx(exact["champion_prob"])
    assert hybrid["final_prob"] == pytest.approx(exact["final_prob"])
    assert hybrid["joint_final_prob"] == pytest.approx(exact["joint_final_prob"])
    assert hybrid["advance_prob"]["group"] == pytest.approx(exact["advance_prob"]["group"])
    assert hybrid["expected_series"] == pytest.approx(exact["expected_series"])
    assert exact["integration"]["kind"] == "exact"
    assert hybrid["integration"]["kind"] == "hybrid"

def test_knockout_without_declared_champion_does_not_invent_final_target():
    spec = knockout_spec(list("ABCD"))
    del spec["champion"]
    result = simulate_tournament(spec, probabilities(spec["teams"]), integration="exact")
    assert result["champion_prob"] is None
    assert result["final_prob"] is None
    assert result["joint_final_prob"] is None
    assert result["expected_series"] == pytest.approx(3)


def test_hybrid_prefix_champion_keeps_only_its_declared_finalists():
    teams = list("ABCD")
    spec = {"version": 1, "id": "prefix-champion", "teams": teams,
            "stages": [{"id": "title", "kind": "single_elimination", "entrants": teams,
                        "best_of": 1, "seeding": "ordered"},
                       {"id": "extra", "kind": "single_elimination", "best_of": 5,
                        "entrants": [ref("title", 1), ref("title", 2)]}],
            "champion": ref("title", 1)}
    table = probabilities(teams)
    table.update({(a, b, 1): 1.0 for a, b in combinations(teams, 2)})
    hybrid = simulate_tournament(spec, table, integration="hybrid", simulations=7)
    assert hybrid["champion_prob"] == pytest.approx({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0})
    assert hybrid["final_prob"] == pytest.approx({"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.0})
    assert sum(hybrid["joint_final_prob"].values()) == pytest.approx(1)
    assert hybrid["expected_series"] == pytest.approx(4)


def test_full_rank_dp_rejects_unbounded_round_loser_products_explicitly():
    teams = [f"T{i}" for i in range(17)]
    table = {(a, b, 5): 0.5 for a, b in combinations(teams, 2)}
    with pytest.raises(UnsupportedIntegrationError, match="at most 16"):
        simulate_tournament(knockout_spec(teams), table, integration="exact")

def test_four_team_round_robin_exact_matches_64_path_standings_and_qualifier_oracle():
    teams = list("ABCD")
    fixtures = list(combinations(teams, 2))
    table = probabilities(teams)
    spec = {"version": 1, "id": "rr-oracle", "teams": teams,
            "stages": [{"id": "group", "kind": "round_robin", "entrants": teams,
                        "best_of": 1, "cycles": 1, "advance": 2,
                        "tiebreakers": ["wins", "head_to_head", "seed"]}],
            "champion": ref("group", 1)}
    ranks = {team: [0.0] * 4 for team in teams}
    qualifiers = {}
    champion = dict.fromkeys(teams, 0.0)
    for choices in product((False, True), repeat=6):
        mass = 1.0
        winners = []
        wins = dict.fromkeys(teams, 0)
        for (a, b), first_wins in zip(fixtures, choices):
            probability = table[a, b, 1]
            mass *= probability if first_wins else 1 - probability
            winner, loser = (a, b) if first_wins else (b, a)
            winners.append((winner, loser))
            wins[winner] += 1
        ranking = []
        for total in sorted(set(wins.values()), reverse=True):
            tied = [team for team in teams if wins[team] == total]
            head_to_head = {team: sum(w == team for w, l in winners if w in tied and l in tied)
                            for team in tied}
            ranking.extend(sorted(tied, key=lambda team: (-head_to_head[team], teams.index(team))))
        champion[ranking[0]] += mass
        key = pair_key(ranking[:2])
        qualifiers[key] = qualifiers.get(key, 0.0) + mass
        for rank, team in enumerate(ranking):
            ranks[team][rank] += mass
    result = simulate_tournament(spec, table, integration="exact")
    assert result["champion_prob"] == pytest.approx(champion)
    assert result["joint_advance_prob"]["group"] == pytest.approx(qualifiers)
    for team in teams:
        assert result["stage_rank_prob"]["group"]["all"][team] == pytest.approx(ranks[team])
        assert result["advance_prob"]["group"][team] == pytest.approx(sum(ranks[team][:2]))
    assert result["expected_series"] == pytest.approx(6)
    assert result["final_prob"] is None


def test_grouped_round_robin_feeding_knockout_retains_joint_qualifier_correlation():
    teams = list("ABCD")
    left = {"stage": "groups", "group": "left", "rank": 1}
    right = {"stage": "groups", "group": "right", "rank": 1}
    spec = {"version": 1, "id": "groups-ko", "teams": teams,
            "stages": [{"id": "groups", "kind": "round_robin", "groups": {"left": ["A", "B"], "right": ["C", "D"]},
                        "best_of": 1, "advance": 1, "tiebreakers": ["wins", "seed"]},
                       {"id": "ko", "kind": "single_elimination", "entrants": [left, right], "best_of": 5}],
            "champion": ref("ko", 1)}
    table = probabilities(teams)
    champion = dict.fromkeys(teams, 0.0)
    joint = {}
    for left_wins, right_wins in product((False, True), repeat=2):
        a = "A" if left_wins else "B"
        b = "C" if right_wins else "D"
        mass = table["A", "B", 1] if left_wins else 1 - table["A", "B", 1]
        mass *= table["C", "D", 1] if right_wins else 1 - table["C", "D", 1]
        joint[pair_key((a, b))] = mass
        probability = pwin(table, a, b, 5)
        champion[a] += mass * probability
        champion[b] += mass * (1 - probability)
    result = simulate_tournament(spec, table, integration="exact")
    assert result["champion_prob"] == pytest.approx(champion)
    assert result["joint_advance_prob"]["groups"] == pytest.approx(joint)
    for key, expected in joint.items():
        assert result["joint_final_prob"][key] == pytest.approx(expected)
    assert result["expected_series"] == pytest.approx(3)
    assert result["stage_rank_prob"]["groups"]["left"]["C"] == [0, 0]
    assert result["stage_rank_resolved"]["groups"]["left"]["C"] is True


def test_round_robin_exact_declared_regular_schedule_work_bound_is_explicit():
    teams = [f"T{i}" for i in range(7)]
    spec = {"version": 1, "id": "too-large-rr", "teams": teams,
            "stages": [{"id": "group", "kind": "round_robin", "entrants": teams,
                        "best_of": 1, "tiebreakers": ["wins", "seed"]}]}
    table = {(a, b, 1): 0.5 for a, b in combinations(teams, 2)}
    with pytest.raises(UnsupportedIntegrationError, match="20-series"):
        simulate_tournament(spec, table, integration="exact")


def test_round_robin_map_standings_are_explicitly_unsupported_even_for_bo1():
    teams = list("ABCD")
    spec = {"version": 1, "id": "map-rr", "teams": teams,
            "stages": [{"id": "group", "kind": "round_robin", "entrants": teams, "best_of": 1,
                        "tiebreakers": ["wins", "map_diff", "seed"]}]}
    with pytest.raises(UnsupportedIntegrationError, match="score-dependent"):
        simulate_tournament(spec, probabilities(teams), integration="exact")

def test_exact_finite_vector_total_work_limit_rejects_large_mixture_before_enumeration():
    teams = [f"T{i}" for i in range(16)]
    table = {(a, b, 5): 0.5 for a, b in combinations(teams, 2)}
    draws = {team: [1.0 if team == teams[0] else 0.0] * 10_000 for team in teams}
    with pytest.raises(UnsupportedIntegrationError, match="finite-mixture work"):
        simulate_tournament(knockout_spec(teams), table, simulations=10_000,
                            team_strength_draws=draws, integration="exact")


@pytest.mark.parametrize("controller", [
    {"draw": {"policy": "uniform"}},
    {"opponent_choice": "lowest_seed"},
])
def test_swiss_quarterfinal_draw_rejects_a_competing_pairing_controller(controller):
    teams = [f"T{i}" for i in range(16)]
    spec = {"version": 1, "id": "competing-quarterfinal-draw", "teams": teams,
            "stages": [{"id": "swiss", "kind": "swiss", "entrants": teams,
                        "best_of": 1, "wins_to_advance": 3, "losses_to_eliminate": 3,
                        "advance": 8, "tiebreakers": ["seed"]},
                       {"id": "ko", "kind": "single_elimination",
                        "entrants": [ref("swiss", rank) for rank in range(1, 9)],
                        "best_of": 5, "swiss_quarterfinal_from": "swiss", **controller}],
            "champion": ref("ko", 1)}
    table = {(a, b, bo): 0.5 for a, b in combinations(teams, 2) for bo in (1, 3, 5)}
    with pytest.raises(ValueError):
        simulate_tournament(spec, table, simulations=1)


def test_swiss_quarterfinal_supply_rejects_eliminated_rank_without_placement_bands():
    teams = [f"T{i}" for i in range(16)]
    supplied = [ref("swiss", rank) for rank in range(1, 9)]
    supplied[4] = ref("swiss", 9)
    spec = {"version": 1, "id": "invalid-qualified-supply", "teams": teams,
            "stages": [{"id": "swiss", "kind": "swiss", "entrants": teams,
                        "best_of": 1, "wins_to_advance": 3, "losses_to_eliminate": 3,
                        "advance": 8, "tiebreakers": ["seed"]},
                       {"id": "ko", "kind": "single_elimination", "entrants": supplied,
                        "best_of": 5, "swiss_quarterfinal_from": "swiss"}],
            "champion": ref("ko", 1)}
    table = {(a, b, bo): 0.5 for a, b in combinations(teams, 2) for bo in (1, 3, 5)}
    with pytest.raises(ValueError, match="eight qualifiers"):
        simulate_tournament(spec, table, integration="hybrid", simulations=2)


def test_monte_carlo_swiss_opening_rank_references_resolve_group_results_each_rollout():
    teams = [f"T{i}" for i in range(16)]
    groups = {f"G{i}": teams[4 * i:4 * i + 4] for i in range(4)}
    entrants = [{"stage": "groups", "group": group, "rank": rank}
                for group in groups for rank in range(1, 5)]
    pairs = [[entrants[i], entrants[i + 1]] for i in range(0, 16, 2)]
    spec = {"version": 1, "id": "groups-swiss", "teams": teams,
            "stages": [{"id": "groups", "kind": "round_robin", "groups": groups,
                        "best_of": 1, "tiebreakers": ["wins", "seed"]},
                       {"id": "swiss", "kind": "swiss", "entrants": entrants,
                        "best_of": 1, "rounds": 1, "initial_pairings": pairs,
                        "tiebreakers": ["wins", "seed"]}]}
    table = {(a, b, 1): 0.5 for a, b in combinations(teams, 2)}
    result = simulate_tournament(spec, table, simulations=2, seed=29)
    assert result["expected_series"] == pytest.approx(32)
    group_games = [row for row in result["trace"] if row["stage"] == "groups"]
    swiss_games = [row for row in result["trace"] if row["stage"] == "swiss"]
    expected_pairs = []
    for group, field in groups.items():
        wins = {team: sum(row["winner"] == team for row in group_games if row["group"] == group)
                for team in field}
        ranking = sorted(field, key=lambda team: (-wins[team], field.index(team)))
        expected_pairs.extend([(ranking[0], ranking[1]), (ranking[2], ranking[3])])
    assert [(row["team_a"], row["team_b"]) for row in swiss_games] == expected_pairs
    for team in teams:
        assert sum(result["stage_rank_prob"]["swiss"]["all"][team]) == pytest.approx(1)
        assert sum(result["stage_records"]["swiss"][team].values()) == 1
