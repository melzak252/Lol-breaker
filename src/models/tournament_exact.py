"""Rule-reusing finite integration and Rao--Blackwellized knockout suffixes.

Exact graphs enumerate winner choices, never score paths. Fixed knockouts use
subtree dynamic programming; hybrid draws its adaptive prefix only once per unit.
"""
from __future__ import annotations

from itertools import combinations, product
import json
import random

from src.models.tournament_formats import _Run, _groups, _draw_laws, _structural_support


class UnsupportedIntegrationError(ValueError):
    """The requested target cannot be integrated under this backend's contract."""


def _key(teams):
    return json.dumps(sorted(teams), separators=(",", ":"), ensure_ascii=False)


def _empty(spec):
    teams = spec["teams"]
    return {
        "champion_prob": dict.fromkeys(teams, 0.0) if spec.get("champion") else None,
        "final_prob": None,
        "joint_final_prob": None,
        "stage_rank_prob": {
            stage["id"]: {group: {team: [0.0] * len(refs) for team in teams}
                          for group, refs in _groups(stage).items()}
            for stage in spec["stages"]},
        "advance_prob": {stage["id"]: dict.fromkeys(teams, 0.0)
                         for stage in spec["stages"] if "advance" in stage},
        "joint_advance_prob": {stage["id"]: {} for stage in spec["stages"] if "advance" in stage},
        "placement_categories": [band["label"] for band in spec.get("placement_bands", [])],
        "placement_prob": ({team: {band["label"]: 0.0 for band in spec["placement_bands"]}
                            for team in teams} if spec.get("placement_bands") else None),
        "expected_series": 0.0,
        "future_expected_series": 0.0,
    }


def _add(target, source, weight):
    """Accumulate a distribution tree, retaining explicitly declared zero cells."""
    for key, value in source.items():
        if key == "placement_categories" or value is None:
            continue
        if isinstance(value, dict):
            if target.get(key) is None:
                target[key] = {}
            _add(target.setdefault(key, {}), value, weight)
        elif isinstance(value, list):
            vector = target.setdefault(key, [0.0] * len(value))
            for i, probability in enumerate(value):
                vector[i] += weight * probability
        else:
            target[key] = target.get(key, 0.0) + weight * value


def _run_distribution(spec, run):
    result = _empty(spec)
    if result["champion_prob"] is not None:
        result["champion_prob"][run.resolve(spec["champion"])] = 1.0
    if run.finalists:
        result["final_prob"] = dict.fromkeys(spec["teams"], 0.0)
        for team in run.finalists:
            result["final_prob"][team] = 1.0
        result["joint_final_prob"] = {_key(run.finalists): 1.0}
    for stage in spec["stages"]:
        identity = stage["id"]
        for group, ranking in run.rankings[identity].items():
            for rank, team in enumerate(ranking):
                result["stage_rank_prob"][identity][group][team][rank] = 1.0
                if identity in result["advance_prob"] and rank < stage["advance"]:
                    result["advance_prob"][identity][team] = 1.0
        if identity in result["advance_prob"]:
            qualifiers = [team for ranking in run.rankings[identity].values()
                          for team in ranking[:stage["advance"]]]
            result["joint_advance_prob"][identity][_key(qualifiers)] = 1.0
    for band in spec.get("placement_bands", []):
        for ref in band["placements"]:
            result["placement_prob"][run.resolve(ref)][band["label"]] += 1.0
    result["expected_series"] = float(run.series)
    result["future_expected_series"] = float(run.future_series)
    return result


class _Branch(Exception):
    def __init__(self, probability):
        self.probability = float(probability)


class _FixedRandom:
    def __init__(self, first_wins):
        self.first_wins = first_wins

    def random(self):
        return 0.0 if self.first_wins else 1.0


class _EnumeratedRun(_Run):
    def __init__(self, spec, tables, decisions, strength=None):
        super().__init__(spec, tables, {}, random.Random(0), False, strength=strength)
        self.decisions = decisions

    def match(self, a, b, round_number, group="all", best_of=None, weight=1):
        if a is not None and b is not None:
            p = float(self.tables[best_of or self.stage["best_of"]][self.index[a], self.index[b]])
            if self.strength is not None and 0 < p < 1:
                import math
                z = math.log(p) - math.log1p(-p) + self.strength[self.index[a]] - self.strength[self.index[b]]
                p = 1 / (1 + math.exp(-float(z))) if z >= 0 else math.exp(float(z)) / (1 + math.exp(float(z)))
            if self.series == len(self.decisions):
                raise _Branch(p)
            self.rng = _FixedRandom(self.decisions[self.series])
        return super().match(a, b, round_number, group, best_of, weight)


def _enumerate(spec, tables, strength=None):
    pending = [((), 1.0)]
    while pending:
        decisions, weight = pending.pop()
        run = _EnumeratedRun(spec, tables, decisions, strength)
        try:
            run.run()
        except _Branch as branch:
            if branch.probability > 0:
                pending.append((decisions + (True,), weight * branch.probability))
            if branch.probability < 1:
                pending.append((decisions + (False,), weight * (1 - branch.probability)))
        else:
            yield weight, run


def _knockout_dp(spec, stage, teams, pairs, tables):
    """Independent disjoint subtree laws plus joint loser laws within each round.

    Winner DP is quadratic in entrant count. Rank laws integrate the independently
    distributed losers at a round (not complete tournament paths); seed tie order
    is the existing routing convention. No best-of probability transformation.
    """
    result = _empty({**spec, "stages": [stage], "placement_bands": []})
    rows = result["stage_rank_prob"][stage["id"]]["all"]
    index = {team: i for i, team in enumerate(spec["teams"])}
    seeds = {team: i for i, team in enumerate(teams)}
    table = tables[stage["best_of"]]
    nodes = [({a: 1.0} if a is not None else {}, {b: 1.0} if b is not None else {}) for a, b in pairs]
    champions = None
    champion_target = spec.get("champion")
    declared_final = bool(champion_target and champion_target.get("stage") == stage["id"]
                          and champion_target.get("group") == "all" and champion_target.get("rank") == 1)
    while nodes:
        next_winners, loser_laws = [], []
        finalists = declared_final and len(nodes) == 1 and bool(nodes[0][0]) and bool(nodes[0][1])
        if finalists:
            result["final_prob"] = dict.fromkeys(spec["teams"], 0.0)
            result["joint_final_prob"] = {}
        for left, right in nodes:
            if not left or not right:
                next_winners.append(left or right)
                continue
            result["expected_series"] += 1.0
            winners, losers = {}, {}
            for a, pa in left.items():
                for b, pb in right.items():
                    joint = pa * pb
                    p = float(table[index[a], index[b]])
                    winners[a] = winners.get(a, 0.0) + joint * p
                    winners[b] = winners.get(b, 0.0) + joint * (1 - p)
                    losers[b] = losers.get(b, 0.0) + joint * p
                    losers[a] = losers.get(a, 0.0) + joint * (1 - p)
                    if finalists:
                        pair = _key((a, b))
                        result["joint_final_prob"][pair] = result["joint_final_prob"].get(pair, 0.0) + joint
                        result["final_prob"][a] += joint
                        result["final_prob"][b] += joint
            next_winners.append(winners)
            loser_laws.append(losers)
        # Same-round subtrees are disjoint, hence their loser laws are independent.
        offset = len(next_winners)
        for choices in product(*(tuple(law.items()) for law in loser_laws)):
            mass = 1.0
            for _, probability in choices:
                mass *= probability
            for rank, team in enumerate(sorted((t for t, _ in choices), key=seeds.__getitem__), offset):
                rows[team][rank] += mass
        if len(next_winners) == 1:
            champions = next_winners[0]
            break
        nodes = list(zip(next_winners[::2], next_winners[1::2]))
    if champions is None:  # One entrant, no played series.
        champions = {teams[0]: 1.0}
    for team, probability in champions.items():
        rows[team][0] += probability
    champion = spec.get("champion")
    if champion and champion["stage"] == stage["id"]:
        rank = champion["rank"] - 1
        result["champion_prob"] = {team: vector[rank] for team, vector in rows.items()}
    else:
        result["champion_prob"] = None
    result["future_expected_series"] = result["expected_series"]
    return result


def _placement_from_ranks(spec, result):
    if not spec.get("placement_bands"):
        return
    result["placement_prob"] = {team: dict.fromkeys(result["placement_categories"], 0.0)
                                for team in spec["teams"]}
    for band in spec["placement_bands"]:
        for ref in band["placements"]:
            if "group" not in ref:
                raise UnsupportedIntegrationError("Rank-DP placements require explicit group references")
            rows = result["stage_rank_prob"][ref["stage"]][ref["group"]]
            for team, ranks in rows.items():
                result["placement_prob"][team][band["label"]] += ranks[ref["rank"] - 1]


def _moments(target, distribution):
    for name in ("champion_prob", "final_prob", "advance_prob", "placement_prob", "joint_final_prob", "joint_advance_prob"):
        value = distribution.get(name)
        if value is None:
            continue
        def record(destination, tree):
            for key, item in tree.items():
                if isinstance(item, dict):
                    record(destination.setdefault(key, {}), item)
                else:
                    leaf = destination.setdefault(key, {"sum": 0.0, "sum_squares": 0.0})
                    leaf["sum"] += item
                    leaf["sum_squares"] += item * item
        record(target.setdefault(name, {}), value)


def _finish_moments(tree, n):
    for value in tree.values():
        if "sum_squares" in value:
            value["n"] = n
        else:
            _finish_moments(value, n)


def _series_bound(stage):
    groups = _groups(stage)
    n = sum(len(refs) for refs in groups.values())
    if stage["kind"] == "round_robin":
        if "schedule" in stage:
            return len(stage["schedule"])
        same = sum(len(refs) * (len(refs) - 1) // 2 for refs in groups.values())
        all_pairs = n * (n - 1) // 2
        cross = stage.get("cross_group", "none")
        pairs = same if cross == "none" else all_pairs - same if cross == "only" else all_pairs
        return pairs * stage.get("cycles", 1)
    if stage["kind"] == "graph":
        return len(stage["matches"])
    if stage["kind"] == "gsl":
        return 5
    if stage["kind"] == "double_elimination":
        return 2 * n - 1
    return n - 1


def integrate_tournament(spec, tables, *, simulations, seed, integration, scores,
                         conditional_scores, strength, state, observables):
    if state is not None or observables is not None:
        raise UnsupportedIntegrationError("Exact/hybrid checkpoint and declared observable state is unsupported; use monte_carlo explicitly")
    stages = spec["stages"]
    for stage in stages:
        if set(stage.get("tiebreakers", [])) & {"map_diff", "map_wins", "tiebreak_match"}:
            raise UnsupportedIntegrationError("Exact/hybrid score-dependent or tiebreak-match standings are unsupported")
    fixed_kinds = {"single_elimination", "double_elimination", "gsl", "graph", "gauntlet", "round_robin"}
    def fixed(stage):
        return (stage["kind"] in fixed_kinds and not any(key in stage for key in
                ("draw", "swiss_quarterfinal_from"))
                and "random" not in stage.get("tiebreakers", [])
                and "random" not in stage.get("group_tiebreakers", []))
    suffix = stages[-1]
    dp_suffix = (suffix["kind"] == "single_elimination" and not suffix.get("reseed")
                 and "opponent_choice" not in suffix and "advance" not in suffix
                 and suffix.get("tiebreakers", ["seed"]) == ["seed"])
    if integration == "exact" and not all(fixed(stage) for stage in stages):
        raise UnsupportedIntegrationError("Exact integration supports fixed finite knockout/GSL/graph/round-robin routes, not adaptive Swiss or random draws/ties")
    use_dp = dp_suffix and (integration == "hybrid" or len(stages) == 1) and strength is None
    if use_dp and len(suffix.get("entrants", [])) > 16:
        raise UnsupportedIntegrationError("Full knockout rank integration supports at most 16 entrants; round loser products are not an unbounded quadratic algorithm")
    if integration == "hybrid" and not use_dp:
        if all(fixed(stage) for stage in stages):
            # A completely finite route needs no sampled prefix at all.
            integration = "exact"
        else:
            raise UnsupportedIntegrationError("Hybrid requires a fixed single-elimination suffix with seed tie order")
    if integration == "hybrid" and strength is not None:
        raise UnsupportedIntegrationError("Hybrid finite latent vectors require an explicitly supported mixture sampling design")
    if not use_dp:
        # Keep enumeration explicitly bounded instead of silently switching to MC.
        bound = sum(_series_bound(stage) for stage in stages)
        if bound > 20:
            raise UnsupportedIntegrationError("Exact finite route exceeds the supported 20-series enumeration bound")
        vector_count = 1 if strength is None else len(strength)
        if vector_count * (1 << bound) > (1 << 20):
            raise UnsupportedIntegrationError("Exact finite-mixture work exceeds 1048576 worst-case weighted paths across all latent vectors")
    result = _empty(spec)
    rng = random.Random(int(seed))
    moments = {}
    if use_dp:
        count = simulations if integration == "hybrid" else 1
        champion_target = spec.get("champion")
        prefix_target = champion_target if champion_target and champion_target["stage"] != suffix["id"] else None
        prefix_spec = {**spec, "stages": stages[:-1], "champion": prefix_target, "placement_bands": []}
        for iteration in range(count):
            prefix = _Run(prefix_spec, tables, scores, rng, False,
                          conditional_scores=conditional_scores).run()
            prefix.stage = suffix
            refs = suffix.get("entrants", [])
            teams = [prefix.resolve(ref) for ref in refs]
            prefix.seed_order = {team: i for i, team in enumerate(teams)}
            source_groups = {team: ref.get("group") for team, ref in zip(teams, refs) if isinstance(ref, dict)}
            pairs = prefix.first_pairs(teams, source_groups)
            local = _run_distribution(prefix_spec, prefix)
            local["placement_categories"] = result["placement_categories"]
            _add(local, _knockout_dp(spec, suffix, teams, pairs, tables), 1.0)
            if spec.get("champion") and spec["champion"]["stage"] != suffix["id"]:
                local["champion_prob"] = dict.fromkeys(spec["teams"], 0.0)
                local["champion_prob"][prefix.resolve(spec["champion"])] = 1.0
            _placement_from_ranks(spec, local)
            _add(result, local, 1 / count)
            if integration == "hybrid":
                _moments(moments, local)
    else:
        vectors = [None] if strength is None else strength
        for vector in vectors:
            for weight, run in _enumerate(spec, tables, vector):
                _add(result, _run_distribution(spec, run), weight / len(vectors))
    # Diagnostic traces are illustrative only; never raw hit counts or proof of support.
    diagnostic = _Run(spec, tables, scores, random.Random(int(seed)), True,
                      conditional_scores=conditional_scores).run()
    support, possible_ref = _structural_support(spec, diagnostic)
    possible_ranks = {stage: {group: {team: [rank for rank, members in enumerate(ranks, 1) if team in members]
                                    for team in spec["teams"]} for group, ranks in groups.items()}
                      for stage, groups in support.items()}
    possible_placements = {team: [] for team in spec["teams"]}
    for band in spec.get("placement_bands", []):
        members = set().union(*(possible_ref(ref) for ref in band["placements"]))
        for team in members:
            possible_placements[team].append(band["label"])
    zero_champion = sorted(set(spec["teams"]) - possible_ref(spec["champion"])) if spec.get("champion") else []
    zero_final = []
    zero_joint = []
    if diagnostic.final_support is not None:
        zero_joint = [_key(pair) for pair in combinations(sorted(spec["teams"]), 2)
                      if pair not in diagnostic.final_support]
        candidates = set().union(*(set(pair) for pair in diagnostic.final_support))
        zero_final = sorted(set(spec["teams"]) - candidates)
        # Keep complete finite pair vectors, including structurally absent pairs.
        for pair in combinations(sorted(spec["teams"]), 2):
            result["joint_final_prob"].setdefault(_key(pair), 0.0)
    result.update({
        "simulations": simulations if integration == "hybrid" else None, "seed": seed,
        "probability_unit": "series_win", "hits": None,
        "completed_series": 0, "trace": diagnostic.trace, "stage_records": diagnostic.records,
        "trace_scope": "illustrative_sample_only", "stage_records_scope": "illustrative_sample_only",
        "known_pairings": diagnostic.known_pairings,
        "known_pairings_scope": "structurally_fixed_participants_only",
        "certified_placement_bands": [], "stage_rank_possible": possible_ranks,
        "possible_placement_categories": possible_placements if spec.get("placement_bands") else None,
        "stage_rank_resolved": {
            stage: {group: {team: not ranks for team, ranks in members.items()}
                    for group, members in groups.items()}
            for stage, groups in possible_ranks.items()},
        "structural_zero_categories": {"champion": zero_champion, "final": zero_final, "joint_final": zero_joint},
        "state_qualification": "strict_timestamped",
        "target_resolved": {"champion": False, "final": False,
                            "champion_by_team": {team: team in zero_champion for team in spec["teams"]},
                            "final_by_team": {team: team in zero_final for team in spec["teams"]},
                            "placement_by_team": {team: len(possible_placements[team]) == 1
                                                  for team in spec["teams"]},
                            "advance": dict.fromkeys(result["advance_prob"], False),
                            "advance_by_team": {
                                stage: {team: not any(possible_ranks[stage][group][team]
                                                     for group in possible_ranks[stage])
                                        for team in spec["teams"]}
                                for stage in result["advance_prob"]}},
        "integration": {"kind": integration,
                        "effective_sampling_unit": "independent_prefix" if integration == "hybrid" else "deterministic_probability_mass",
                        "fixed_n": simulations if integration == "hybrid" else None,
                        "stopping_rule": "fixed_n" if integration == "hybrid" else "not_sampled",
                        "iid": integration == "hybrid", "draw_laws": _draw_laws(spec),
                        "algorithm": ("knockout_subtree_dp_and_bounded_round_loser_products"
                                      if use_dp else "finite_winner_path_enumeration"),
                        "full_rank_dp_max_entrants": 16 if use_dp else None,
                        "enumeration_max_weighted_paths": 1 << 20 if not use_dp else None,
                        "approximation": ("sampled adaptive prefix and bracket draw; fixed knockout suffix integrated exactly"
                                          if integration == "hybrid" else "floating-point finite integration; no sampling error"),
                        "latent_target": "uniform finite-vector mixture" if strength is not None else "fixed series law"},
    })
    if integration == "hybrid":
        _finish_moments(moments, simulations)
        result["integration"]["prefix_moments"] = moments
    return result
