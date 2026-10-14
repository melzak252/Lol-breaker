"""Multistage tournaments and timestamp-validated continuation with SERIES probabilities.

The format schema is deliberately closed: unsupported rules fail rather than becoming
implicit coin flips. Draws sample uniformly over complete feasible perfect matchings
(Swiss: minimum total record gap first). A completion-count dynamic program prevents
sequential greedy dead ends. No map probability is inferred from a series probability.

Elimination entrants are ordered seeds. Teams eliminated in the same round are ranked
by original seed unless explicit standings tiebreakers are provided. Graphs instead
supply every placement explicitly, conserving entrant/winner/loser tokens. Provenance
metadata is not a point-in-time certification and never changes simulation behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from functools import lru_cache
from itertools import combinations
import math
import json
from numbers import Real
import random
from types import SimpleNamespace

import bisect
import numpy as np

from src.models.frozen_tournament import _integer, _probability_tables


_KINDS = {"round_robin", "gsl", "swiss", "single_elimination",
          "double_elimination", "gauntlet", "graph", "pick_and_play"}
_COMMON = {"id", "kind", "entrants", "groups", "best_of", "advance", "tiebreakers"}
_FIELDS = {
    "round_robin": {"cycles", "cross_group", "carry_from", "carry_policy", "cycle_weights",
                    "schedule", "group_tiebreakers"},
    "gsl": {"decider_best_of"},
    "swiss": {"wins_to_advance", "losses_to_eliminate", "decider_best_of",
              "no_rematches", "draw", "pairing_policy", "rounds", "initial_pairings", "rematch_fallback"},
    "single_elimination": {"seeding", "byes", "reseed", "draw", "opponent_choice", "swiss_quarterfinal_from"},
    "double_elimination": {"seeding", "byes", "final_reset"},
    "gauntlet": set(),
    "graph": {"matches", "ranking", "seed_order"},
    "pick_and_play": {"rounds", "initial_pairings", "choice_policy", "no_rematches"},
}
_FINAL_TIES = {"seed", "random", "tiebreak_match"}
_METRICS = {"wins", "losses", "head_to_head", "map_diff", "map_wins"}


def _closed(value, allowed, context):
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{context}: unknown fields/rules {sorted(unknown)}")


def _positive(value, context):
    if not _integer(value) or value < 1:
        raise ValueError(f"{context}: expected a positive integer")


def _best_of(value):
    if not _integer(value, (1, 3, 5)):
        raise ValueError("best_of must be 1, 3, or 5")


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("Series weight must be finite and positive")


def _token(ref):
    if isinstance(ref, str):
        return ("team", ref)
    if isinstance(ref, Mapping) and all(isinstance(k, str) and isinstance(v, (str, int)) for k, v in ref.items()):
        return tuple(sorted(ref.items()))
    raise ValueError("Entrant must be exact team ID or rank reference")


def _groups(stage):
    return stage.get("groups", {"all": stage.get("entrants", [])})


def _validate_disjoint_refs(refs, earlier):
    """Prove disjoint supplies without discarding shared-ancestor rank constraints."""
    cache = {}

    def sources(ref):
        groups = _groups(earlier[ref["stage"]])
        selected = [groups[ref["group"]]] if "group" in ref else groups.values()
        return [source for group in selected for source in group]

    def disjoint(first, second):
        key = (_token(first), _token(second))
        if key in cache:
            return cache[key]
        if isinstance(first, str) and isinstance(second, str):
            result = first != second
        elif first == second:
            result = False
        elif (isinstance(first, Mapping) and isinstance(second, Mapping)
              and first["stage"] == second["stage"]
              and (first["rank"] != second["rank"]
                   or any(selector in first and selector in second
                          and first[selector] != second[selector]
                          for selector in ("group", "group_rank")))):
            result = True
        else:
            # Expand one side at a time: two GSL groups can descend from disjoint
            # ranks of the same league even though both can contain any team ID.
            result = (
                isinstance(first, Mapping)
                and all(disjoint(source, second) for source in sources(first))
            ) or (
                isinstance(second, Mapping)
                and all(disjoint(first, source) for source in sources(second))
            )
        cache[key] = result
        return result

    for i, first in enumerate(refs):
        for second in refs[i + 1:]:
            if not disjoint(first, second):
                raise ValueError("Stage entrant supplies may overlap; use disjoint source ranks")


def required_best_ofs(spec):
    """Return every series unit requested, including implicit Swiss Bo3 deciders."""
    formats = set()
    for stage in spec.get("stages", []):
        formats.add(stage.get("best_of"))
        if stage.get("kind") == "swiss" and "rounds" not in stage:
            formats.add(stage.get("decider_best_of", 3))
        elif "decider_best_of" in stage:
            formats.add(stage["decider_best_of"])
        for node in stage.get("matches", []) + stage.get("schedule", []):
            formats.add(node.get("best_of", stage.get("best_of")))
    for value in formats:
        _best_of(value)
    return sorted(formats)


def _validate_ref(ref, teams, earlier):
    if isinstance(ref, str):
        if ref not in teams:
            raise ValueError(f"Unknown entrant {ref!r}")
        return
    _closed(ref, {"stage", "group", "group_rank", "rank"}, "rank reference")
    if set(ref) not in ({"stage", "group", "rank"}, {"stage", "group_rank", "rank"}):
        raise ValueError("Rank reference needs stage, group or group_rank, and rank")
    if not isinstance(ref["stage"], str) or ref["stage"] not in earlier:
        raise ValueError("Rank references must name an earlier stage")
    _positive(ref["rank"], "rank")
    source = earlier[ref["stage"]]
    groups = _groups(source)
    if "group_rank" in ref:
        _positive(ref["group_rank"], "group_rank")
        if ref["group_rank"] > len(groups) or not source.get("group_tiebreakers"):
            raise ValueError("group_rank requires source group_tiebreakers and a valid group rank")
        if any(ref["rank"] > len(entrants) for entrants in groups.values()):
            raise ValueError("Rank exceeds a possible selected group size")
    elif not isinstance(ref["group"], str) or ref["group"] not in groups or ref["rank"] > len(groups[ref["group"]]):
        raise ValueError("Unknown group or out-of-range rank")


def _ties(criteria, required=False, group=False):
    if criteria is None and not required:
        return
    metrics = {"wins"} if group else _METRICS
    finals = {"seed", "random"} if group else _FINAL_TIES
    if (not isinstance(criteria, list) or not criteria or
            any(not isinstance(c, str) for c in criteria) or len(set(criteria)) != len(criteria) or
            criteria[-1] not in finals or any(c not in metrics for c in criteria[:-1])):
        raise ValueError("Tiebreakers need known metrics and an explicit final tie resolution")


def _validate_draw(draw, teams, refs):
    _closed(draw, {"policy", "avoid_same", "labels", "no_rematches", "opposite_groups"}, "draw")
    if draw.get("policy") != "uniform":
        raise ValueError("Only exact uniform feasible draws are supported")
    for field in ("no_rematches", "opposite_groups"):
        if field in draw and not isinstance(draw[field], bool):
            raise ValueError(f"draw.{field} must be boolean")
    labels = draw.get("labels", {})
    avoid = draw.get("avoid_same", [])
    if not isinstance(avoid, list) or any(not isinstance(label, str) or not label for label in avoid):
        raise ValueError("avoid_same must list label names")
    if not isinstance(labels, Mapping) or set(labels) - set(teams):
        raise ValueError("Draw labels must name global teams")
    for team in teams:
        entry = labels.get(team, {})
        if not isinstance(entry, Mapping) or any(not isinstance(entry.get(label), str) for label in avoid):
            raise ValueError("Draw labels must cover every global team and requested label")
    if draw.get("opposite_groups") and any(not isinstance(ref, Mapping) or "group" not in ref for ref in refs):
        raise ValueError("opposite_groups needs explicit group rank references for all entrants")


def _graph_source_token(slot):
    if isinstance(slot, Mapping) and set(slot) in ({"winner"}, {"loser"}):
        outcome, node = next(iter(slot.items()))
        if not isinstance(node, str):
            raise ValueError("Graph outcome must reference a node ID")
        return outcome, node
    return _token(slot)


def _seeded_pool(slot):
    _closed(slot, {"seeded", "rank"}, "seeded graph supply")
    sources = slot.get("seeded")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("Seeded graph supply needs at least two non-nested sources")
    _positive(slot.get("rank"), "seeded supply rank")
    if slot["rank"] > len(sources):
        raise ValueError("Seeded supply rank exceeds pool size")
    tokens = frozenset(_graph_source_token(source) for source in sources)
    if len(tokens) != len(sources):
        raise ValueError("Seeded pool cannot duplicate participant supplies")
    return tokens


def _validate_graph(stage, refs):
    nodes = stage.get("matches")
    ranking = stage.get("ranking")
    if not isinstance(nodes, list) or not isinstance(ranking, list):
        raise ValueError("Graph needs a matches list and complete ranking")
    if stage.get("seed_order", "stage") not in ("stage", "tournament"):
        raise ValueError("Graph seed_order must be stage or tournament")
    available = {_token(ref) for ref in refs}
    known, pools = set(), set()

    def consume(slot, allow_bye=False):
        if slot is None and allow_bye:
            return False
        if isinstance(slot, Mapping) and "seeded" in slot:
            pool = _seeded_pool(slot)
            if pool not in pools:
                for source in slot["seeded"]:
                    consume(source)
                available.update(("seeded", pool, rank) for rank in range(1, len(pool) + 1))
                pools.add(pool)
            token = ("seeded", pool, slot["rank"])
        else:
            token = _graph_source_token(slot)
            if isinstance(slot, Mapping) and set(slot) in ({"winner"}, {"loser"}):
                if token[1] not in known:
                    raise ValueError("Graph output must reference an earlier node (no cycles)")
        if token not in available:
            raise ValueError("Graph consumes missing or reused participant supply")
        available.remove(token)
        return True

    for node in nodes:
        _closed(node, {"id", "a", "b", "best_of"}, "graph match")
        if not isinstance(node.get("id"), str) or not node["id"] or node["id"] in known:
            raise ValueError("Graph match IDs must be unique nonempty strings")
        if "a" not in node or "b" not in node:
            raise ValueError("Graph needs both slots; use null for an explicit bye")
        _best_of(node.get("best_of", stage["best_of"]))
        occupied = int(consume(node["a"], True)) + int(consume(node["b"], True))
        if not occupied:
            raise ValueError("Graph match cannot have two byes")
        known.add(node["id"])
        available.add(("winner", node["id"]))
        if occupied == 2:
            available.add(("loser", node["id"]))
    for slot in ranking:
        consume(slot)
    if available or len(ranking) != len(refs):
        raise ValueError("Graph ranking must consume every remaining participant exactly once")


def _validate_metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a provenance object")
    forbidden = {"winner", "winners", "score", "scores", "score1", "score2", "score_a", "score_b",
                 "observed_winners", "observed_scores"}

    def visit(value):
        if isinstance(value, Mapping):
            if forbidden & set(value) or value.get("observed_results_used") not in (None, False):
                raise ValueError("Pre-start input cannot contain observed winners or scores")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(metadata)


def validate_tournament_spec(spec):
    """Validate structural rules/references without running any tournament draws."""
    _closed(spec, {"version", "id", "teams", "stages", "champion", "metadata", "final",
                   "placement_bands"}, "tournament")
    if "metadata" in spec:
        _validate_metadata(spec["metadata"])
    if not _integer(spec.get("version"), (1,)) or not isinstance(spec.get("id"), str) or not spec["id"]:
        raise ValueError("Tournament needs version=1 and a nonempty id")
    teams = spec.get("teams")
    if (not isinstance(teams, list) or not teams or
            any(not isinstance(t, str) or not t.strip() for t in teams) or len(set(teams)) != len(teams)):
        raise ValueError("Teams must be unique nonempty exact string identifiers")
    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Tournament needs nonempty stages")
    earlier = {}
    for stage in stages:
        if (not isinstance(stage, Mapping) or not isinstance(stage.get("kind"), str)
                or stage["kind"] not in _KINDS):
            raise ValueError("Unknown stage kind")
        kind = stage["kind"]
        _closed(stage, _COMMON | _FIELDS[kind], f"{kind} stage")
        if not isinstance(stage.get("id"), str) or not stage["id"] or stage["id"] in earlier:
            raise ValueError("Stage IDs must be unique nonempty strings")
        _best_of(stage.get("best_of"))
        if ("entrants" in stage) == ("groups" in stage):
            raise ValueError("Specify exactly one of entrants or groups")
        if "groups" in stage and kind != "round_robin":
            raise ValueError("Only round_robin supports named groups; compose other stages explicitly")
        groups = _groups(stage)
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("Groups must be a nonempty object")
        refs = []
        for name, entrants in groups.items():
            if not isinstance(name, str) or not name or not isinstance(entrants, list) or not entrants:
                raise ValueError("Each named group needs nonempty entrants")
            refs.extend(entrants)
            if "advance" in stage:
                _positive(stage["advance"], "advance")
                if stage["advance"] > len(entrants):
                    raise ValueError("advance exceeds group size")
        for ref in refs:
            _validate_ref(ref, teams, earlier)
        if len({_token(ref) for ref in refs}) != len(refs):
            raise ValueError("A stage cannot duplicate entrant references")
        _validate_disjoint_refs(refs, earlier)
        _ties(stage.get("tiebreakers"), required=kind in {"round_robin", "swiss", "pick_and_play"})
        if kind == "round_robin":
            cycles = stage.get("cycles", 1)
            if not _integer(cycles, (1, 2, 3, 4)):
                raise ValueError("cycles must be 1..4")
            if stage.get("cross_group", "none") not in {"none", "include", "only"}:
                raise ValueError("cross_group must be none, include, or only")
            if stage.get("cross_group") == "only" and len(groups) < 2:
                raise ValueError("Cross-group-only schedule needs at least two groups")
            _ties(stage.get("group_tiebreakers"), group=True)
            if "cycle_weights" in stage:
                if not isinstance(stage["cycle_weights"], list) or len(stage["cycle_weights"]) != cycles:
                    raise ValueError("cycle_weights must have one positive weight per cycle")
                for weight in stage["cycle_weights"]:
                    _weight(weight)
            if ("carry_from" in stage) != ("carry_policy" in stage):
                raise ValueError("carry_from and carry_policy must appear together")
            if "carry_from" in stage:
                if (not isinstance(stage["carry_from"], str) or stage["carry_from"] not in earlier or
                        earlier[stage["carry_from"]]["kind"] != "round_robin"):
                    raise ValueError("carry_from must name an earlier round_robin stage")
                if stage["carry_policy"] not in {"all", "qualified_only"}:
                    raise ValueError("Unknown carry_policy")
            if "schedule" in stage:
                if "cycle_weights" in stage or "cycles" in stage or "cross_group" in stage:
                    raise ValueError("Explicit schedule replaces cycles/cross_group/cycle_weights")
                if not isinstance(stage["schedule"], list) or not stage["schedule"]:
                    raise ValueError("Explicit schedule needs pre-start matches")
                tokens = {_token(ref) for ref in refs}
                for match in stage["schedule"]:
                    _closed(match, {"a", "b", "best_of", "weight"}, "scheduled match")
                    if _token(match.get("a")) not in tokens or _token(match.get("b")) not in tokens or match["a"] == match["b"]:
                        raise ValueError("Scheduled pairs must be distinct stage entrants")
                    _best_of(match.get("best_of", stage["best_of"]))
                    _weight(match.get("weight", 1))
        elif kind == "gsl":
            if len(refs) != 4:
                raise ValueError("GSL needs exactly four entrants")
            _best_of(stage.get("decider_best_of", stage["best_of"]))
        elif kind == "swiss":
            if "rounds" in stage:
                _positive(stage["rounds"], "Swiss rounds")
                if any(field in stage for field in ("wins_to_advance", "losses_to_eliminate", "decider_best_of")):
                    raise ValueError("Swiss rounds and qualification thresholds are mutually exclusive")
                if stage.get("no_rematches", True) and stage["rounds"] >= len(refs):
                    raise ValueError("Swiss rounds exceed distinct opponents available without rematches")
            else:
                for field in ("wins_to_advance", "losses_to_eliminate"):
                    _positive(stage.get(field), field)
            if len(refs) % 2:
                raise ValueError("Swiss requires an even entrant count; implicit byes are forbidden")
            _best_of(stage.get("decider_best_of", 3))
            if stage.get("pairing_policy", "minimum_record_gap") != "minimum_record_gap":
                raise ValueError("Unknown Swiss pairing_policy")
            if stage.get("rematch_fallback") not in {None, "waive_only_if_no_feasible_draw"}:
                raise ValueError("Unknown Swiss rematch fallback")
            if "initial_pairings" in stage:
                pairs = stage["initial_pairings"]
                if (not isinstance(pairs, list) or any(not isinstance(p, list) or len(p) != 2 for p in pairs)
                        or len(pairs) * 2 != len(refs)
                        or {_token(t) for p in pairs for t in p} != {_token(ref) for ref in refs}):
                    raise ValueError("Swiss opening pairs must cover exact entrants once")
        elif kind == "pick_and_play":
            _positive(stage.get("rounds"), "Pick-and-Play rounds")
            if len(refs) % 2 or stage.get("choice_policy") != "nearest_better_record":
                raise ValueError("Pick-and-Play needs even entrants and nearest_better_record choice policy")
            if stage.get("no_rematches") is not False:
                raise ValueError("Pick-and-Play requires explicit no_rematches=false scenario policy")
            if stage["tiebreakers"][-1] == "tiebreak_match":
                raise ValueError("Opponent selection cannot introduce extra tiebreak matches")
            pairings = stage.get("initial_pairings")
            if not isinstance(pairings, list):
                raise ValueError("Pick-and-Play requires explicit initial_pairings")
            paired = []
            for pair in pairings:
                _closed(pair, {"a", "b"}, "initial pairing")
                paired.extend((_token(pair.get("a")), _token(pair.get("b"))))
            if len(paired) != len(refs) or set(paired) != {_token(ref) for ref in refs}:
                raise ValueError("Initial pairings must contain every entrant exactly once")
        elif kind in {"single_elimination", "double_elimination"}:
            if stage.get("seeding", "standard") not in {"standard", "ordered"}:
                raise ValueError("Unknown elimination seeding")
            bye_count = (1 << (len(refs) - 1).bit_length()) - len(refs)
            if "byes" in stage:
                byes = stage["byes"]
                if (stage.get("seeding") != "ordered" or not isinstance(byes, list) or
                        len(byes) != bye_count or len({_token(r) for r in byes}) != len(byes) or
                        any(_token(r) not in {_token(ref) for ref in refs} for r in byes)):
                    raise ValueError("Explicit byes require ordered seeding and exactly the required distinct entrants")
            if kind == "double_elimination" and stage.get("final_reset") not in {"single_final", "if_lower_wins"}:
                raise ValueError("Double elimination requires explicit final_reset policy")
            if "swiss_quarterfinal_from" in stage:
                if "draw" in stage or stage.get("opponent_choice") is not None:
                    raise ValueError("Swiss quarterfinal draw and another draw controller cannot both control pairings")
                source = earlier.get(stage["swiss_quarterfinal_from"])
                if (kind != "single_elimination" or len(refs) != 8 or source is None
                        or source["kind"] != "swiss" or source.get("wins_to_advance") != 3
                        or source.get("losses_to_eliminate") != 3
                        or {_token(ref) for ref in refs} != {
                            _token({"stage": source["id"], "group": "all", "rank": rank})
                            for rank in range(1, 9)}):
                    raise ValueError("Swiss quarterfinal draw needs eight qualifiers from a 3-win/3-loss Swiss")
            if stage.get("opponent_choice") not in {None, "lowest_seed", "highest_seed"}:
                raise ValueError("Unknown opponent_choice behavior")
            if "draw" in stage and "opponent_choice" in stage:
                raise ValueError("draw and opponent_choice cannot both control pairings")
        elif kind == "graph":
            _validate_graph(stage, refs)
        for flag in ("no_rematches", "reseed"):
            if flag in stage and not isinstance(stage[flag], bool):
                raise ValueError(f"{flag} must be boolean")
        if "draw" in stage:
            _validate_draw(stage["draw"], teams, refs)
        earlier[stage["id"]] = stage
    if spec.get("champion") is not None:
        if not isinstance(spec["champion"], Mapping):
            raise ValueError("champion must be a rank reference or null")
        _validate_ref(spec["champion"], teams, earlier)
    if "final" in spec:
        final = spec["final"]
        _closed(final, {"stage", "round", "group"}, "final target")
        stage = earlier.get(final.get("stage"))
        if stage is None or stage["kind"] != "graph":
            raise ValueError("Explicit final target must name a graph championship node")
        node = final.get("round")
        if (final.get("group", "all") != "all" or
                stage["ranking"][:2] != [{"winner": node}, {"loser": node}] or
                spec.get("champion") != {"stage": stage["id"], "group": "all", "rank": 1}):
            raise ValueError("Final target must route the declared champion and runner-up")
    if "placement_bands" in spec:
        bands = spec["placement_bands"]
        if not isinstance(bands, list) or not bands:
            raise ValueError("placement_bands must be nonempty ordered official bands")
        refs, labels = [], set()
        for band in bands:
            _closed(band, {"label", "placements"}, "placement band")
            label, placements = band.get("label"), band.get("placements")
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError("Placement labels must be unique nonempty strings")
            if not isinstance(placements, list) or not placements:
                raise ValueError("Placement band needs rank references")
            labels.add(label)
            for ref in placements:
                if not isinstance(ref, Mapping):
                    raise ValueError("Official placements require explicit stage rank references")
                _validate_ref(ref, teams, earlier)
            refs.extend(placements)
        if len(refs) != len(teams) or len({_token(ref) for ref in refs}) != len(teams):
            raise ValueError("Placement bands must cover all original entrants once")
        _validate_disjoint_refs(refs, earlier)
    return required_best_ofs(spec)


def _perfect_matching(teams, allowed, cost, rng):
    """Uniform optimal complete matching, weighted by exact completion counts."""
    teams = tuple(teams)
    if len(teams) % 2:
        raise ValueError("No feasible pairing: odd active field (no implicit bye)")
    n = len(teams)
    edges = {(i, j): cost(teams[i], teams[j]) for i in range(n) for j in range(i + 1, n)
             if allowed(teams[i], teams[j])}
    if len(edges) == n * (n - 1) // 2 and len(set(edges.values())) <= 1:
        shuffled = list(teams)
        rng.shuffle(shuffled)
        return list(zip(shuffled[::2], shuffled[1::2]))

    # Every zero-cost component can be sampled independently when it has a perfect
    # matching: the optimum is zero and its completion count is the product of the
    # component counts. In particular a 32-team Swiss splits into record cohorts
    # instead of exploring cross-record subsets of the entire 32-team field.
    adjacency = [set() for _ in teams]
    for (i, j), edge_cost in edges.items():
        if edge_cost == 0:
            adjacency[i].add(j)
            adjacency[j].add(i)
    unseen = set(range(n))
    components = []
    while unseen:
        pending = [min(unseen)]
        unseen.remove(pending[0])
        component = []
        while pending:
            index = pending.pop()
            component.append(index)
            neighbors = adjacency[index] & unseen
            pending.extend(sorted(neighbors, reverse=True))
            unseen.difference_update(neighbors)
        components.append(sorted(component))
    if len(components) > 1 and all(len(component) % 2 == 0 for component in components):
        zero_edges = {frozenset((teams[i], teams[j])) for (i, j), value in edges.items() if value == 0}
        factored_pairs = []
        try:
            for component in components:
                factored_pairs.extend(_perfect_matching(
                    [teams[index] for index in component],
                    lambda a, b: frozenset((a, b)) in zero_edges,
                    lambda a, b: 0, rng,
                ))
        except ValueError:
            # A zero-cost component need not have a perfect matching. Preserve all
            # constraints and fall back to the global minimum-cost optimization;
            # cross-record matches are then necessary, not a relaxed draw rule.
            pass
        else:
            return factored_pairs

    @lru_cache(None)
    def completions(mask):
        if not mask:
            return 0, 1
        bit = mask & -mask
        i = bit.bit_length() - 1
        rest = mask ^ bit
        best, count = math.inf, 0
        for j in range(i + 1, n):
            if rest & (1 << j) and (i, j) in edges:
                subcost, subcount = completions(rest ^ (1 << j))
                total = edges[i, j] + subcost
                if total < best:
                    best, count = total, subcount
                elif total == best:
                    count += subcount
        return best, count

    mask = (1 << n) - 1
    if completions(mask)[1] == 0:
        raise ValueError("No feasible complete pairing under declared constraints")
    pairs = []
    while mask:
        i = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << i)
        best, count = completions(mask)
        draw = rng.randrange(count)
        for j in range(i + 1, n):
            if not rest & (1 << j) or (i, j) not in edges:
                continue
            subcost, subcount = completions(rest ^ (1 << j))
            if edges[i, j] + subcost != best:
                continue
            if draw < subcount:
                pairs.append((teams[i], teams[j]))
                mask = rest ^ (1 << j)
                break
            draw -= subcount
    return pairs


_TRACE_FIELDS = {"stage", "group", "round", "team_a", "team_b", "best_of",
                 "winner", "loser", "score_a", "score_b", "weight"}


def _utc_timestamp(value, context):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{context}: timestamp must include UTC date and time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{context}: invalid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{context}: timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def _match_key(record):
    return (record["stage"], record["group"], record["round"],
            frozenset((record["team_a"], record["team_b"])))


class _Checkpoint:
    """Parse source facts once; route validation happens in one preflight rollout.

    Completed rows are identified by stage/group/round/unordered pair, not input
    list position. Published pairings describe a whole round (including unplayed
    pairs), and orders describe actual published random-tie resolutions.
    """

    def __init__(self, state, spec):
        _closed(state, {"version", "cutoff", "completed", "pairings", "orders", "group_orders",
                       "temporal_policy"}, "state")
        if not _integer(state.get("version"), (1,)):
            raise ValueError("State requires version=1")
        cutoff = _utc_timestamp(state.get("cutoff"), "cutoff")
        self.temporal_policy = state.get("temporal_policy", "strict_timestamped")
        if self.temporal_policy not in {
                "strict_timestamped", "prior_day_reconstruction", "daily_rollforward"}:
            raise ValueError("Unknown checkpoint temporal_policy")
        if self.temporal_policy == "daily_rollforward":
            if any((cutoff.hour, cutoff.minute, cutoff.second, cutoff.microsecond)):
                raise ValueError("Daily rollforward cutoff must be an exact UTC midnight")
        stages = {stage["id"] for stage in spec["stages"]}
        teams = set(spec["teams"])
        self.completed, self.pairings, self.orders = {}, {}, {}
        self.group_orders = {}
        self.times, self.publication_times = {}, {}
        self.stage_facts = set()
        self.round_facts = set()

        def source_times(row, event_field):
            if self.temporal_policy != "daily_rollforward" and "source_date" in row:
                raise ValueError("source_date is only valid for daily rollforward state")
            if self.temporal_policy == "prior_day_reconstruction":
                if event_field != "completed_at" or "completed_at" in row or "available_at" in row:
                    raise ValueError("Prior-day reconstruction requires completed_date, not fabricated timestamps")
                try:
                    completed_date = date.fromisoformat(row.get("completed_date", ""))
                except (TypeError, ValueError) as error:
                    raise ValueError("Reconstructed completed_date must be an ISO calendar date") from error
                if completed_date >= cutoff.date():
                    raise ValueError("Reconstructed completed_date must strictly precede cutoff UTC day")
                return completed_date, completed_date
            if self.temporal_policy == "daily_rollforward":
                if any(field in row for field in (
                        "completed_date", "completed_at", "available_at", "published_at")):
                    raise ValueError("Daily rollforward requires source_date, not timestamps")
                source_date = row.get("source_date")
                try:
                    parsed_source_date = date.fromisoformat(source_date)
                except (TypeError, ValueError) as error:
                    raise ValueError("Daily rollforward source_date must be a YYYY-MM-DD UTC calendar date") from error
                if not isinstance(source_date, str) or parsed_source_date.isoformat() != source_date:
                    raise ValueError("Daily rollforward source_date must be a YYYY-MM-DD UTC calendar date")
                if parsed_source_date >= cutoff.date():
                    raise ValueError("Daily rollforward source_date must strictly precede cutoff UTC day")
                return parsed_source_date, parsed_source_date
            event = _utc_timestamp(row.get(event_field), event_field)
            available = _utc_timestamp(row.get("available_at"), "available_at")
            if event > available or available > cutoff:
                raise ValueError("State event timestamp must precede availability and cutoff")
            return event, available

        def block(row):
            stage, group, number = row.get("stage"), row.get("group"), row.get("round")
            if (stage not in stages or not isinstance(group, str) or not group or
                    isinstance(number, bool) or not isinstance(number, (str, int))):
                raise ValueError("State needs valid stage/group/round identity")
            return stage, group, number

        for field in ("completed", "pairings", "orders", "group_orders"):
            rows = state.get(field, [] if field != "completed" else None)
            if not isinstance(rows, list):
                raise ValueError(f"State {field} must be a list")
            for row in rows:
                if field == "completed":
                    _closed(row, _TRACE_FIELDS | {
                        "completed_at", "completed_date", "available_at", "source_date"}, "completed")
                    if not _TRACE_FIELDS <= row.keys():
                        raise ValueError("Completed match needs every trace field including actual scores")
                    identity = block(row)
                    if self.temporal_policy == "strict_timestamped" and "completed_date" in row:
                        raise ValueError("Strict timestamped state does not accept reconstructed completed_date")
                    a, b = row["team_a"], row["team_b"]
                    if a not in teams or b not in teams or a == b or {row["winner"], row["loser"]} != {a, b}:
                        raise ValueError("Completed match has invalid participants/winner/loser")
                    _best_of(row["best_of"])
                    _weight(row["weight"])
                    sa, sb, needed = row["score_a"], row["score_b"], row["best_of"] // 2 + 1
                    if (not _integer(sa) or not _integer(sb) or min(sa, sb) < 0 or
                            max(sa, sb) != needed or min(sa, sb) >= needed or
                            (sa > sb) != (row["winner"] == a)):
                        raise ValueError("Completed match requires a legal actual series score")
                    key = _match_key(row)
                    if key in self.completed:
                        raise ValueError("Duplicate completed match identity")
                    self.completed[key] = {key: row[key] for key in _TRACE_FIELDS}
                    self.times[key] = source_times(row, "completed_at")
                    self.round_facts.add(identity)
                elif field == "pairings":
                    _closed(row, {
                        "stage", "group", "round", "pairs", "published_at", "available_at", "source_date"},
                            "published pairings")
                    key = block(row)
                    pairs = row.get("pairs")
                    if (not isinstance(pairs, list) or not pairs or
                            any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                                any(team not in teams for team in pair) for pair in pairs)):
                        raise ValueError("Published pairing needs exact entrant pairs")
                    paired = [team for pair in pairs for team in pair]
                    if len(set(paired)) != len(paired) or key in self.pairings:
                        raise ValueError("Published pairings duplicate participants or round")
                    self.pairings[key] = [tuple(pair) for pair in pairs]
                    self.publication_times[("pairings", key)] = source_times(row, "published_at")
                    self.round_facts.add(key)
                elif field == "group_orders":
                    _closed(row, {
                        "stage", "groups", "published_at", "available_at", "source_date"}, "published group order")
                    stage = next((stage for stage in spec["stages"] if stage["id"] == row.get("stage")), None)
                    ordered = row.get("groups")
                    if (stage is None or stage["kind"] != "round_robin" or not isinstance(ordered, list) or
                            any(not isinstance(group, str) for group in ordered) or
                            len(ordered) != len(_groups(stage)) or set(ordered) != set(_groups(stage)) or
                            stage["id"] in self.group_orders):
                        raise ValueError("Published group order must contain every stage group once")
                    self.group_orders[stage["id"]] = list(ordered)
                    self.publication_times[("group_orders", stage["id"])] = source_times(row, "published_at")
                else:
                    _closed(row, {
                        "stage", "group", "teams", "published_at", "available_at", "source_date"},
                            "published order")
                    ordered = row.get("teams")
                    if (row.get("stage") not in stages or not isinstance(row.get("group"), str) or
                            not isinstance(ordered, list) or not ordered or
                            any(team not in teams for team in ordered) or len(set(ordered)) != len(ordered)):
                        raise ValueError("Published order requires distinct exact entrants")
                    key = row["stage"], row["group"], frozenset(ordered)
                    if key in self.orders:
                        raise ValueError("Duplicate published order")
                    self.orders[key] = list(ordered)
                    self.publication_times[("orders", key)] = source_times(row, "published_at")
                self.stage_facts.add(row["stage"])

def _observable_nodes(spec):
    """Declared block capacities, including bye slots; never infer support from hits."""
    blocks = {}
    for stage in spec["stages"]:
        groups = _groups(stage)
        n = sum(map(len, groups.values()))
        kind, sid, bo = stage["kind"], stage["id"], stage["best_of"]

        def add(round_number, count=1, group="all", formats=None):
            if count:
                blocks[sid, group, round_number] = (count, formats or {bo})

        if kind == "graph":
            for node in stage["matches"]:
                add(node["id"], formats={node.get("best_of", bo)})
        elif kind == "gsl":
            for name in ("opening-1", "opening-2", "winners", "losers", "decider"):
                add(name, formats={bo if name.startswith("opening") else stage.get("decider_best_of", bo)})
        elif kind == "gauntlet":
            for number in range(1, n):
                add(number)
        elif kind in {"swiss", "pick_and_play"}:
            rounds = stage.get("rounds") or stage["wins_to_advance"] + stage["losses_to_eliminate"] - 1
            formats = set(required_best_ofs({"stages": [stage]}))
            for number in range(1, rounds + 1):
                add(number, n // 2, formats=formats)
        elif kind == "round_robin":
            membership = {_token(ref): group for group, refs in groups.items() for ref in refs}
            if "schedule" in stage:
                for number, match in enumerate(stage["schedule"], 1):
                    ga, gb = membership[_token(match["a"])], membership[_token(match["b"])]
                    add(number, group=ga if ga == gb else "cross_group",
                        formats={match.get("best_of", bo)})
            else:
                cross = stage.get("cross_group", "none")
                for number in range(1, stage.get("cycles", 1) + 1):
                    if cross != "only":
                        for group, refs in groups.items():
                            add(number, len(refs) * (len(refs) - 1) // 2, group)
                    if cross != "none":
                        add(number, sum(len(a) * len(b) for a, b in combinations(groups.values(), 2)), "cross_group")
        elif n > 1:
            size = 1 << (n - 1).bit_length()
            if kind == "single_elimination":
                number = 1
                while size > 1:
                    add(number, size // 2)
                    number, size = number + 1, size // 2
            else:
                upper_pairs, lower, number = size // 2, 0, 1
                while upper_pairs:
                    add(f"upper-{number}", upper_pairs)
                    drops = n - upper_pairs if number == 1 else upper_pairs
                    upper = upper_pairs
                    if not lower:
                        lower = drops
                    else:
                        pairs = min(lower, drops)
                        add(f"lower-drop-{number}", pairs)
                        lower = max(lower, drops)
                    if upper == 1:
                        if lower > 1:
                            add(f"lower-final-{number}", lower - 1)
                        add("grand-final")
                        if stage["final_reset"] == "if_lower_wins":
                            add("grand-final-reset")
                        break
                    target = upper // 2
                    add(f"lower-{number}", max(0, lower - target))
                    lower = min(lower, target)
                    upper_pairs, number = upper // 2, number + 1
    return blocks


def _prepare_observables(observables, spec):
    """Compile a closed finite manifest; category coverage is checked on every draw."""
    if observables is None:
        return ()
    if not isinstance(observables, list):
        raise ValueError("observables must be a list")
    kinds = {
        "future_series_count": {"scope"}, "future_map_count": {"scope"},
        "upset_count": {"scope", "reference_probabilities", "threshold"},
        "repeat_encounter_count": {"scope"},
        "node_matchup": {"node", "absent_category"},
        "node_score": {"node", "participants", "absent_category"},
        "champion_path": {"source", "paths"},
    }
    stages = {stage["id"]: stage for stage in spec["stages"]}
    blocks = _observable_nodes(spec) if any(
        isinstance(row, Mapping) and isinstance(row.get("kind"), str) and
        row["kind"] in {"node_matchup", "node_score", "champion_path"}
        for row in observables) else {}
    teams, ids, compiled = set(spec["teams"]), set(), []

    def node_key(node):
        _closed(node, {"stage", "group", "round", "match"}, "observable node")
        sid, group, number = node.get("stage"), node.get("group", "all"), node.get("round")
        if (not isinstance(sid, str) or not isinstance(group, str) or
                not (isinstance(number, str) and number or _integer(number))):
            raise ValueError("Observable node requires exact stage, group and round")
        block = sid, group, number
        if block not in blocks:
            raise ValueError("Observable node is not a declared tournament block")
        capacity, formats = blocks[block]
        index = node.get("match", 1)
        if "match" not in node and capacity != 1:
            raise ValueError("Observable node needs an explicit match index within this block")
        if not _integer(index) or not 1 <= index <= capacity:
            raise ValueError("Observable node match index exceeds declared block capacity")
        return (*block, index), formats

    for row in observables:
        if not isinstance(row, Mapping) or not isinstance(row.get("kind"), str) or row["kind"] not in kinds:
            raise ValueError("Unknown observable kind")
        kind = row["kind"]
        _closed(row, {"id", "kind", "categories"} | kinds[kind], "observable")
        identity, categories = row.get("id"), row.get("categories")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Observable IDs must be unique nonempty strings")
        if (not isinstance(categories, list) or not categories or
                any(not isinstance(c, str) or not c for c in categories) or
                len(set(categories)) != len(categories)):
            raise ValueError("Observable categories must be unique nonempty strings")
        ids.add(identity)
        item = {"id": identity, "kind": kind, "categories": tuple(categories)}
        if kind.endswith("_count"):
            if any(not c.isascii() or not c.isdecimal() or str(int(c)) != c for c in categories):
                raise ValueError("Observable count categories must be canonical nonnegative integers")
            scope = row.get("scope", {})
            _closed(scope, {"stage", "group", "period"}, "observable scope")
            sid, group, period = scope.get("stage"), scope.get("group"), scope.get("period", "future")
            if ((sid is not None and (not isinstance(sid, str) or sid not in stages)) or
                    (group is not None and (not isinstance(group, str) or sid is None or
                     group not in _groups(stages[sid]) and not (
                         group == "cross_group" and stages[sid]["kind"] == "round_robin" and
                         len(_groups(stages[sid])) > 1 and ("schedule" in stages[sid] or
                         stages[sid].get("cross_group", "none") != "none")))) or
                    period not in ("future", "all") or
                    kind.startswith("future_") and period != "future"):
                raise ValueError("Invalid observable scope")
            item.update(stage=sid, group=group, period=period)
            if kind == "upset_count":
                reference = row.get("reference_probabilities")
                if not isinstance(reference, Mapping):
                    raise ValueError("Upset observable requires fixed reference_probabilities")
                formats = set(required_best_ofs(spec if sid is None else {"stages": [stages[sid]]}))
                for key in reference:
                    if not isinstance(key, tuple) or len(key) != 3:
                        raise ValueError("Observable reference keys must be (team_a, team_b, best_of)")
                    _best_of(key[2])
                    formats.add(key[2])
                bracket = SimpleNamespace(teams=spec["teams"], matches={
                    bo: SimpleNamespace(best_of=bo) for bo in formats})
                threshold = row.get("threshold", 0.5)
                if (isinstance(threshold, bool) or not isinstance(threshold, Real) or
                        not math.isfinite(threshold) or not 0 < threshold <= 0.5):
                    raise ValueError("Observable upset threshold must be in (0, .5]")
                item.update(reference=_probability_tables(bracket, dict(reference)), threshold=threshold)
        elif kind in {"node_matchup", "node_score"}:
            key, formats = node_key(row.get("node"))
            item["node"] = key
            absent = row.get("absent_category")
            if "absent_category" in row and (not isinstance(absent, str) or absent not in categories):
                raise ValueError("Observable absent_category must be declared in categories")
            stage = stages[key[0]]
            optional = stage["kind"] == "double_elimination" and key[2] == "grand-final-reset"
            if stage["kind"] == "graph":
                declared = next(node for node in stage["matches"] if node["id"] == key[2])
                optional |= declared["a"] is None or declared["b"] is None
            if optional and absent is None:
                raise ValueError("Observable conditional/bye node requires an explicit absent_category")
            item["absent"] = absent
            if kind == "node_matchup":
                for category in categories:
                    try:
                        pair = json.loads(category)
                    except (ValueError, TypeError):
                        if category == absent:
                            continue
                        raise ValueError("Observable matchup categories must be canonical JSON pairs") from None
                    if category == absent:
                        if isinstance(pair, list):
                            raise ValueError("Observable absence must not alias a matchup category")
                        continue
                    if (not isinstance(pair, list) or len(pair) != 2 or
                            any(not isinstance(t, str) or t not in teams for t in pair) or
                            pair[0] == pair[1] or
                            json.dumps(sorted(pair), separators=(",", ":"), ensure_ascii=False) != category):
                        raise ValueError("Observable matchup categories must be sorted distinct entrant pairs")
            else:
                participants = row.get("participants")
                if (not isinstance(participants, list) or len(participants) != 2 or
                        any(not isinstance(t, str) or t not in teams for t in participants) or
                        participants[0] == participants[1]):
                    raise ValueError("Observable score requires two distinct oriented participants")
                legal = {f"{a}-{b}" for bo in formats for losing in range(bo // 2 + 1)
                         for a, b in ((bo // 2 + 1, losing), (losing, bo // 2 + 1))}
                if any(c not in legal and c != absent for c in categories):
                    raise ValueError("Observable score category is not a legal oriented score")
                if absent in legal:
                    raise ValueError("Observable absence must not alias an oriented score")
                item["participants"] = tuple(participants)
        else:
            if not spec.get("champion"):
                raise ValueError("Observable champion_path requires a declared champion")
            source, paths = row.get("source"), row.get("paths")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Observable champion_path requires an explicit source reference")
            if not isinstance(paths, Mapping) or set(paths) != set(categories):
                raise ValueError("Observable paths must define every declared category")
            item["paths"] = {}
            for category, predicates in paths.items():
                if not isinstance(predicates, list) or not predicates:
                    raise ValueError("Observable path must have explicit node/outcome predicates")
                compiled_path = []
                for predicate in predicates:
                    _closed(predicate, {"node", "outcome"}, "observable path predicate")
                    outcome = predicate.get("outcome")
                    if outcome not in ("winner", "loser", "participant", "absent"):
                        raise ValueError("Observable path outcome must be winner, loser, participant or absent")
                    key, _ = node_key(predicate.get("node"))
                    compiled_path.append((key, outcome))
                item["paths"][category] = tuple(compiled_path)
        compiled.append(item)
    return tuple(compiled)


class _ObservableRun:
    """Online counters and only predeclared node records, bounded per rollout."""

    def __init__(self, manifest, checkpoint):
        self.manifest = manifest
        self.counts = {row["id"]: 0 for row in manifest if row["kind"].endswith("_count")}
        self.nodes = {row["node"]: None for row in manifest if "node" in row}
        for row in manifest:
            for path in row.get("paths", {}).values():
                for key, _ in path:
                    self.nodes[key] = None
        self.blocks = dict.fromkeys((key[:3] for key in self.nodes), 0)
        self.repeats = {}
        for row in manifest:
            if row["kind"] == "repeat_encounter_count":
                self.repeats[row["id"]] = {
                    frozenset((record["team_a"], record["team_b"]))
                    for record in (checkpoint.completed.values() if checkpoint and row["period"] == "future" else ())
                    if self.in_scope(row, record)}

    @staticmethod
    def in_scope(row, record):
        return ((row["stage"] is None or row["stage"] == record["stage"]) and
                (row["group"] is None or row["group"] == record["group"]))

    def slot(self, stage, group, number):
        block = stage, group, number
        if block not in self.blocks:
            return None
        self.blocks[block] += 1
        key = (*block, self.blocks[block])
        return key if key in self.nodes else None

    def observe(self, record, completed, node, index):
        if node is not None:
            self.nodes[node] = record
        for row in self.manifest:
            kind, identity = row["kind"], row["id"]
            if not kind.endswith("_count") or not self.in_scope(row, record):
                continue
            if completed and row["period"] == "future":
                continue
            if kind == "future_series_count":
                value = 1
            elif kind == "future_map_count":
                if record["score_a"] is None or record["score_b"] is None:
                    raise ValueError("Observable future map counts require separate score distributions")
                value = record["score_a"] + record["score_b"]
            elif kind == "upset_count":
                p = row["reference"][record["best_of"]][index[record["winner"]], index[record["loser"]]]
                value = int(p < row["threshold"])
            else:
                pair = frozenset((record["team_a"], record["team_b"]))
                seen = self.repeats[identity]
                value = int(pair in seen)
                seen.add(pair)
            self.counts[identity] += value

    def finish(self, run, hits):
        for row in self.manifest:
            kind, identity = row["kind"], row["id"]
            if kind.endswith("_count"):
                category = str(self.counts[identity])
            elif kind in {"node_matchup", "node_score"}:
                record = self.nodes[row["node"]]
                category = row["absent"]
                if record is not None:
                    pair = record["team_a"], record["team_b"]
                    if kind == "node_matchup":
                        category = json.dumps(sorted(pair), separators=(",", ":"), ensure_ascii=False)
                    elif set(pair) == set(row["participants"]):
                        a, b = record["score_a"], record["score_b"]
                        if a is None or b is None:
                            raise ValueError("Observable oriented scores require separate score distributions")
                        if pair[0] != row["participants"][0]:
                            a, b = b, a
                        category = f"{a}-{b}"
            else:
                champion = run.resolve(run.spec["champion"])
                matching = []
                for label, path in row["paths"].items():
                    for key, outcome in path:
                        record = self.nodes[key]
                        if outcome == "absent":
                            satisfied = record is None
                        elif record is None:
                            satisfied = False
                        elif outcome == "participant":
                            satisfied = champion in (record["team_a"], record["team_b"])
                        else:
                            satisfied = record[outcome] == champion
                        if not satisfied:
                            break
                    else:
                        matching.append(label)
                if len(matching) != 1:
                    raise ValueError("Observable champion paths must partition outcomes into exactly one category")
                category = matching[0]
            if category not in hits[identity]:
                raise ValueError(f"Observable {identity!r} produced undeclared category {category!r}; "
                                 "declare exhaustive categories and legal absence explicitly")
            hits[identity][category] += 1


def _conditional_scores(value, formats):
    if value is None:
        return None
    _closed(value, {"probability_bins", "distributions"}, "conditional score model")
    bins = value.get("probability_bins")
    if (not isinstance(bins, (list, tuple)) or len(bins) < 2 or
            any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in bins) or
            bins[0] != 0 or bins[-1] != 1 or any(a >= b for a, b in zip(bins, bins[1:]))):
        raise ValueError("Conditional probability bins must strictly partition [0, 1]")
    supplied = value.get("distributions")
    if not isinstance(supplied, Mapping):
        raise ValueError("Conditional score distributions must be a mapping")
    distributions = {}
    for key, pmfs in supplied.items():
        if not _integer(key, (1, 3, 5)) and key not in ("1", "3", "5"):
            raise ValueError("Conditional score best_of must be 1, 3 or 5")
        bo = int(key)
        if bo in distributions or not isinstance(pmfs, (list, tuple)) or len(pmfs) != len(bins) - 1:
            raise ValueError("Conditional score model requires complete unique bin support")
        for pmf in pmfs:
            if (not isinstance(pmf, (list, tuple)) or len(pmf) != bo // 2 + 1 or
                    any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or p < 0 for p in pmf) or
                    not math.isclose(sum(pmf), 1, rel_tol=0, abs_tol=1e-9)):
                raise ValueError("Invalid conditional losing-map score distribution")
        distributions[bo] = pmfs
    if any(bo != 1 and bo not in distributions for bo in formats):
        raise ValueError("Conditional score model lacks required best_of support; no pooled fallback")
    return bins, distributions


def _strength_vectors(value, teams, simulations, *, preserve_zero=False, disabled=False):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(teams):
        raise ValueError("Strength draws must name every exact entrant and no other teams")
    vectors = []
    for team in teams:
        values = np.asarray(value[team])
        if (values.shape != (simulations,) or values.dtype.kind not in "iuf" or
                not np.isfinite(values).all() or
                np.any(np.abs(values.astype(np.longdouble)) > np.finfo(float).max)):
            raise ValueError("Each team strength must be a finite numeric vector of length simulations")
        vectors.append(values)
    if disabled:
        return None
    # Long doubles keep subtraction of two finite extreme draws finite.
    strength = np.column_stack(vectors).astype(np.longdouble, copy=False)
    return strength if preserve_zero or np.any(strength) else None


class _Run:
    def __init__(self, spec, tables, scores, rng, trace, checkpoint=None, validate_state=False, observables=(),
                 strength=None, conditional_scores=None, copula_rho=None, copula_probits=None,
                 copula_transform=None):
        self.spec, self.tables, self.scores, self.rng = spec, tables, scores, rng
        self.strength, self.conditional_scores = strength, conditional_scores
        self.copula_rho, self.copula_probits = copula_rho, copula_probits
        self.copula_transform = copula_transform
        self.index = {team: index for index, team in enumerate(spec["teams"])}
        self.rankings, self.group_orders, self.history, self.records = {}, {}, {}, {}
        self.played = set()
        self.trace = [] if trace else None
        self.series = 0
        self.stage = None
        self.log = []
        self.seed_order = {}
        self.checkpoint, self.validate_state = checkpoint, validate_state
        self.future_series = 0
        self.used_completed, self.used_pairings, self.used_orders = set(), set(), set()
        self.uncertain, self.stage_uncertain = set(), {}
        self.completed_swiss = set()
        self.used_group_orders = set()
        self.team_available = {}
        self.finalists, self.final_resolved = None, False
        self.certified_bands = []
        self.stage_future_start = 0
        self.random_order = False
        self.block_pairs = {}
        self.final_stage = None
        self.known_finalists = set()
        self.stage_available, self.input_available = {}, {}
        self.entrants_known = True
        self.unknown_blocks = set()
        self.known_pairings = []
        self.collect_facts = trace or validate_state
        # Abstract supply sets follow the first trace, not its sampled winners.
        # Unknown draws/reseeding widen them; only observed results narrow them.
        self.possible_teams = None
        self.final_support = None
        self.observables = _ObservableRun(observables, checkpoint) if observables else None

    def resolve(self, ref):
        if isinstance(ref, str):
            return ref
        if (self.validate_state and self.stage and ref["stage"] != self.stage["id"] and
                self.stage["id"] in self.checkpoint.stage_facts and self.stage_uncertain.get(ref["stage"])):
            raise ValueError("Completed history depends on an unresolved earlier stage route")
        group = ref.get("group")
        if group is None:
            group = self.group_orders[ref["stage"]][ref["group_rank"] - 1]
        team = self.rankings[ref["stage"]][group][ref["rank"] - 1]
        if self.validate_state and ref["stage"] in self.stage_available:
            source_time = self.stage_available[ref["stage"]]
            self.team_available[team] = max(self.team_available.get(team, source_time), source_time)
        return team

    def match(self, a, b, round_number, group="all", best_of=None, weight=1):
        node = self.observables.slot(self.stage["id"], group, round_number) if self.observables else None
        if a is None:
            return b, None
        if b is None:
            return a, None
        if a == b:
            raise ValueError("Resolved participant occurs twice in one match")
        bo = best_of or self.stage["best_of"]
        key = (self.stage["id"], group, round_number, frozenset((a, b)))
        block = self.stage["id"], group, round_number
        if self.validate_state:
            self.block_pairs.setdefault(block, set()).add(frozenset((a, b)))
            publication = self.checkpoint.publication_times.get(("pairings", block))
            if publication:
                if ({a, b} & self.uncertain and self.stage["kind"] != "round_robin") or any(
                        self.team_available.get(t, publication[0]) > publication[0] for t in (a, b)):
                    raise ValueError("Published pairing depends on unresolved or unavailable ancestor route")
        record = self.checkpoint.completed.get(key) if self.checkpoint else None
        completed = record is not None
        independent = self.stage["kind"] == "round_robin" and not str(round_number).startswith("tiebreak-")
        known = ((independent and self.entrants_known) or not ({a, b} & self.uncertain))
        if self.collect_facts and known and block not in self.unknown_blocks:
            self.known_pairings.append({
                "stage": self.stage["id"], "group": group, "round": round_number,
                "team_a": a, "team_b": b, "best_of": bo, "weight": weight, "completed": record is not None,
                "winner": record["winner"] if record is not None else None,
            })
        if record is not None:
            if self.validate_state:
                if key in self.used_completed or record["best_of"] != bo or record["weight"] != weight:
                    raise ValueError("Completed history has inconsistent match format, weight or route")
                if not independent and ({a, b} & self.uncertain):
                    raise ValueError("Completed history has an unresolved ancestor route")
                completed_at, available_at = self.checkpoint.times[key]
                dependencies = self.input_available if independent else self.team_available
                if any(dependencies.get(t, completed_at) > completed_at for t in (a, b)):
                    raise ValueError("Completed result precedes ancestor availability timestamp")
                block = (self.stage["id"], group, round_number)
                publication = self.checkpoint.publication_times.get(("pairings", block))
                if publication and publication[1] > completed_at:
                    raise ValueError("Historical pairing was not available before completed match")
                for team in (a, b):
                    self.team_available[team] = max(self.team_available.get(team, available_at), available_at)
            self.used_completed.add(key)
            winner, loser = record["winner"], record["loser"]
            if self.collect_facts:
                self.possible_teams[winner], self.possible_teams[loser] = {winner}, {loser}
        else:
            p = self.tables[bo][self.index[a], self.index[b]]
            if self.strength is not None and (self.copula_transform is not None or 0 < p < 1):
                delta = self.strength[self.index[a]] - self.strength[self.index[b]]
                if self.copula_transform is not None:
                    p = self.copula_transform(
                        self.copula_probits[bo][self.index[a], self.index[b]],
                        delta / math.sqrt(2), self.copula_rho)
                elif delta != 0:
                    z = math.log(p) - math.log1p(-p) + delta
                    p = 1 / (1 + math.exp(-float(z))) if z >= 0 else math.exp(float(z)) / (1 + math.exp(float(z)))
            winner, loser = (a, b) if self.rng.random() < p else (b, a)
            score_a = score_b = None
            if bo == 1 or bo in self.scores or self.conditional_scores is not None:
                pmf = self.scores.get(bo)
                if bo != 1 and self.conditional_scores is not None:
                    bins, distributions = self.conditional_scores
                    winner_p = (self.tables[bo][self.index[winner], self.index[loser]]
                                if self.strength is None or self.copula_rho is not None
                                else p if winner == a else 1 - p)
                    bin_index = min(bisect.bisect_right(bins, winner_p) - 1, len(bins) - 2)
                    pmf = distributions[bo][bin_index]
                losing_maps = 0 if bo == 1 else self.rng.choices(range(bo // 2 + 1), weights=pmf, k=1)[0]
                score_a, score_b = (bo // 2 + 1, losing_maps) if winner == a else (losing_maps, bo // 2 + 1)
            record = {"stage": self.stage["id"], "group": group, "round": round_number,
                      "team_a": a, "team_b": b, "best_of": bo, "winner": winner, "loser": loser,
                      "score_a": score_a, "score_b": score_b, "weight": weight}
            self.future_series += 1
            self.uncertain.update((a, b))
            if self.collect_facts:
                possible = (set(self.spec["teams"]) if block in self.unknown_blocks else
                            self.possible_teams[a] | self.possible_teams[b])
                self.possible_teams[a] = self.possible_teams[b] = possible
        if self.observables:
            self.observables.observe(record, completed, node, self.index)
        self.log.append(record)
        if self.trace is not None:
            self.trace.append(record)
        self.series += 1
        self.played.add(frozenset((a, b)))
        return winner, loser

    def published_pairs(self, teams, round_number, group="all", expected=None, required=False,
                        allowed=None, cost=None):
        if self.checkpoint is None:
            return None
        key = self.stage["id"], group, round_number
        pairs = self.checkpoint.pairings.get(key)
        if pairs is None:
            if required and key in self.checkpoint.round_facts:
                raise ValueError("Historical draw requires a complete published pairing record")
            return None
        if self.validate_state:
            if key in self.used_pairings or set(teams) != {team for pair in pairs for team in pair}:
                raise ValueError("Published pairing has impossible entrant route")
            if set(teams) & self.uncertain:
                raise ValueError("Published pairing depends on unresolved ancestor results")
            published_at = self.checkpoint.publication_times[("pairings", key)][0]
            if any(self.team_available.get(t, published_at) > published_at for t in teams):
                raise ValueError("Published pairing precedes ancestor availability timestamp")
            if expected is not None and {frozenset(pair) for pair in pairs} != {frozenset(pair) for pair in expected}:
                raise ValueError("Published pairing contradicts fixed bracket/choice route")
            if allowed is not None and any(not allowed(a, b) for a, b in pairs):
                raise ValueError("Published pairing violates draw or rematch constraints")
            if cost is not None:
                optimum = _perfect_matching(teams, allowed, cost, random.Random(0))
                if sum(cost(a, b) for a, b in pairs) != sum(cost(a, b) for a, b in optimum):
                    raise ValueError("Published Swiss pairing violates minimum record gap")
        self.used_pairings.add(key)
        return pairs

    def fixed_pairs(self, pairs, round_number, group="all", required=False, choice=False):
        actual = [(a, b) for a, b in pairs if a is not None and b is not None]
        published = self.published_pairs([team for pair in actual for team in pair],
                                         round_number, group, expected=None if choice else actual, required=required)
        if published is None:
            if required:
                self.unknown_blocks.add((self.stage["id"], group, round_number))
            return pairs
        if choice:
            byes = [(a, b) for a, b in pairs if a is None or b is None]
            return byes + sorted(published, key=lambda pair: min(self.seed_order[t] for t in pair))
        # Keep structural bracket slots and byes; publication order is not seeding.
        by_pair = {frozenset(pair): pair for pair in published}
        return [by_pair[frozenset((a, b))] if a is not None and b is not None else (a, b)
                for a, b in pairs]

    def final_participants(self, a, b):
        champion = self.spec.get("champion")
        if (champion and champion.get("stage") == self.stage["id"] and
                champion.get("group") == "all" and champion.get("rank") == 1):
            self.finalists = (a, b)
            self.final_resolved = not ({a, b} & self.uncertain)
            self.final_stage = self.stage["id"]
            self.known_finalists = {a, b} - self.uncertain
            if self.collect_facts:
                self.final_support = {
                    tuple(sorted((first, second)))
                    for first in self.possible_teams[a] for second in self.possible_teams[b]
                    if first != second}
            if self.collect_facts and self.known_finalists:
                self.certified_bands.append((self.stage["id"], "all", {1, 2}, self.known_finalists))

    def round_boundary(self, round_number):
        if (self.validate_state and
                (self.stage["id"], "all", round_number) in self.checkpoint.round_facts and
                self.future_series != self.stage_future_start):
            raise ValueError("Completed/published round requires all earlier round results")

    @staticmethod
    def stats(teams, records):
        result = {team: {"wins": 0, "losses": 0, "map_diff": 0, "map_wins": 0} for team in teams}
        for record in records:
            if record["winner"] in result:
                result[record["winner"]]["wins"] += record["weight"]
            if record["loser"] in result:
                result[record["loser"]]["losses"] += record["weight"]
            if record["score_a"] is not None:
                for team, won, lost in ((record["team_a"], record["score_a"], record["score_b"]),
                                        (record["team_b"], record["score_b"], record["score_a"])):
                    if team in result:
                        result[team]["map_wins"] += won
                        result[team]["map_diff"] += won - lost
        return result

    def certify_elimination(self, losers, remaining, group="all"):
        if not self.collect_facts:
            return
        known = set(losers) - self.uncertain
        if known:
            self.certified_bands.append((self.stage["id"], group,
                                         set(range(remaining + 1, remaining + len(losers) + 1)), known))

    def order(self, teams, records, criteria=None, group="all"):
        # Published random-tie resolutions are addressed by their complete tied set.
        criteria = criteria or ["seed"]
        if set(criteria or []) & {"map_diff", "map_wins"} and any(
                record["score_a"] is None or record["score_b"] is None for record in records):
            raise ValueError("Map standings require separate score distributions for future contributing series")
        stats = self.stats(teams, records)
        if self.validate_state and self.stage["kind"] == "round_robin" and set(teams) & self.uncertain and any(
                stage == self.stage["id"] and name == group and str(number).startswith("tiebreak-")
                for stage, name, number in self.checkpoint.round_facts):
            raise ValueError("Completed tiebreak history requires all relevant standings results")

        def sort_tie(tied, remaining):
            if len(tied) < 2:
                return tied
            rule = remaining[0]
            if rule == "seed":
                return sorted(tied, key=self.seed_order.__getitem__)
            if rule == "random":
                key = self.stage["id"], group, frozenset(tied)
                published = self.checkpoint.orders.get(key) if self.checkpoint else None
                if published is not None:
                    if self.validate_state:
                        if key in self.used_orders:
                            raise ValueError("Repeated random-tie identity needs an unambiguous published order")
                        published_at = self.checkpoint.publication_times[("orders", key)][0]
                        if (set(tied) & self.uncertain or self.future_series != self.stage_future_start or
                                any(self.team_available.get(t, published_at) > published_at for t in tied)):
                            raise ValueError("Published tie order depends on unresolved or unavailable history")
                    self.used_orders.add(key)
                    return list(published)
                self.random_order = True
                self.uncertain.update(tied)
                result = list(tied)
                self.rng.shuffle(result)
                return result
            if rule == "tiebreak_match":
                return self.knockout(sorted(tied, key=self.seed_order.__getitem__), group=group, tie=True)
            values = stats
            if rule == "head_to_head":
                subset = set(tied)
                values = self.stats(tied, [r for r in records if r["team_a"] in subset and r["team_b"] in subset])
            key = "wins" if rule == "head_to_head" else rule
            buckets = {}
            for team in tied:
                buckets.setdefault(values[team][key], []).append(team)
            return [team for value in sorted(buckets, reverse=rule != "losses")
                    for team in sort_tie(buckets[value], remaining[1:])]

        return sort_tie(list(teams), criteria)

    def draw(self, teams, draw, source_groups, no_rematches=False, records=None, round_number=1):
        labels = draw.get("labels", {})

        def allowed(a, b):
            return (not ((no_rematches or draw.get("no_rematches")) and frozenset((a, b)) in self.played)
                    and not (draw.get("opposite_groups") and source_groups[a] == source_groups[b])
                    and all(labels[a][key] != labels[b][key] for key in draw.get("avoid_same", [])))

        def cost(a, b):
            return (abs(records[a]["wins"] - records[b]["wins"]) +
                    abs(records[a]["losses"] - records[b]["losses"])) if records else 0

        published = self.published_pairs(teams, round_number, required=True, allowed=allowed,
                                         cost=cost if records else None)
        if published is not None:
            return published
        self.unknown_blocks.add((self.stage["id"], "all", round_number))
        return _perfect_matching(teams, allowed, cost, self.rng)

    def first_pairs(self, teams, source_groups=None, tie=False):
        if not tie and "swiss_quarterfinal_from" in self.stage:
            records = self.records[self.stage["swiss_quarterfinal_from"]]
            unbeaten = [t for t in teams if records[t] == {"wins": 3, "losses": 0}]
            three_two = [t for t in teams if records[t] == {"wins": 3, "losses": 2}]
            if len(unbeaten) != 2 or len(three_two) != 3 or len(teams) != 8:
                raise ValueError("Swiss quarterfinal draw has inconsistent terminal record supplies")

            def allowed(a, b):
                if a in unbeaten:
                    return b in three_two
                if b in unbeaten:
                    return a in three_two
                return True

            published = self.published_pairs(teams, 1, required=True, allowed=allowed)
            if published is not None:
                if self.validate_state and any(
                        sum(a in unbeaten or b in unbeaten for a, b in published[start:start + 2]) != 1
                        for start in (0, 2)):
                    raise ValueError("Published quarterfinal draw must put 3-0 teams in opposite halves")
                return published
            self.unknown_blocks.add((self.stage["id"], "all", 1))
            self.rng.shuffle(unbeaten)
            self.rng.shuffle(three_two)
            opponents = three_two[:2]
            remaining = [t for t in teams if t not in unbeaten + opponents]
            self.rng.shuffle(remaining)
            # Slots 0/1 and 2/3 form the two semifinal halves.
            pairs = [(unbeaten[0], opponents[0]), tuple(remaining[:2]),
                     (unbeaten[1], opponents[1]), tuple(remaining[2:])]
            if self.rng.randrange(2):
                pairs[0], pairs[1] = pairs[1], pairs[0]
            if self.rng.randrange(2):
                pairs[2], pairs[3] = pairs[3], pairs[2]
            return pairs
        size = 1 << (len(teams) - 1).bit_length()
        bye_count = size - len(teams)
        if not tie and "byes" in self.stage:
            byes = [self.resolve(ref) for ref in self.stage["byes"]]
        else:
            byes = teams[:bye_count]
        if not tie and ("draw" in self.stage or "opponent_choice" in self.stage or self.stage.get("seeding") == "ordered"):
            active = [team for team in teams if team not in byes]
            if "draw" in self.stage:
                pairs = self.draw(active, self.stage["draw"], source_groups or {})
            elif "opponent_choice" in self.stage:
                pairs = self.chosen_pairs(active)
            else:
                pairs = list(zip(active[::2], active[1::2]))
            return [(team, None) for team in byes] + pairs
        seeds = [1]
        while len(seeds) < size:
            ceiling = len(seeds) * 2 + 1
            seeds = [slot for seed in seeds for slot in (seed, ceiling - seed)]
        slots = [teams[seed - 1] if seed <= len(teams) else None for seed in seeds]
        return list(zip(slots[::2], slots[1::2]))

    def chosen_pairs(self, teams):
        remaining = sorted(teams, key=self.seed_order.__getitem__)
        pairs = []
        while remaining:
            chooser = remaining.pop(0)
            opponent = remaining.pop(-1 if self.stage.get("opponent_choice") == "lowest_seed" else 0)
            pairs.append((chooser, opponent))
        return pairs

    def knockout(self, teams, source_groups=None, group="all", tie=False):
        if len(teams) == 1:
            return list(teams)
        pairs = self.first_pairs(teams, source_groups, tie)
        eliminated = []
        round_number = 1
        while pairs:
            if not tie and (self.stage.get("reseed") or "opponent_choice" in self.stage):
                self.round_boundary(round_number)
                if self.future_series != self.stage_future_start:
                    self.unknown_blocks.add((self.stage["id"], group, round_number))
            identity = f"tiebreak-{round_number}" if tie else round_number
            if not (not tie and round_number == 1 and
                    ("draw" in self.stage or "swiss_quarterfinal_from" in self.stage)):
                pairs = self.fixed_pairs(pairs, identity, group,
                                         required=not tie and "opponent_choice" in self.stage,
                                         choice=not tie and "opponent_choice" in self.stage)
            if not tie and len(pairs) == 1 and all(t is not None for t in pairs[0]):
                self.final_participants(*pairs[0])
            outcomes = [self.match(a, b, f"tiebreak-{round_number}" if tie else round_number, group)
                        for a, b in pairs]
            survivors = [winner for winner, _ in outcomes]
            losers = [loser for _, loser in outcomes if loser is not None]
            if not tie:
                self.certify_elimination(losers, len(survivors), group)
            eliminated.append(sorted(losers, key=self.seed_order.__getitem__) if tie else
                              self.order(losers, self.log, self.stage.get("tiebreakers"), group))
            if len(survivors) == 1:
                return survivors + [team for group_losers in reversed(eliminated) for team in group_losers]
            if not tie and "opponent_choice" in self.stage:
                pairs = self.chosen_pairs(survivors)
            elif not tie and self.stage.get("reseed"):
                survivors.sort(key=self.seed_order.__getitem__)
                pairs = list(zip(survivors[:len(survivors) // 2], reversed(survivors[len(survivors) // 2:])))
            else:
                pairs = list(zip(survivors[::2], survivors[1::2]))
            round_number += 1

    def double_elimination(self, teams):
        if len(teams) == 1:
            return teams
        upper_pairs = self.first_pairs(teams)
        lower, eliminated = [], []
        round_number = 1
        while upper_pairs:
            outcomes = [self.match(a, b, f"upper-{round_number}") for a, b in upper_pairs]
            upper = [w for w, _ in outcomes]
            drops = [loser for _, loser in outcomes if loser is not None]
            if not lower:
                lower = drops
            else:
                # Bracket-slot reversal crosses upper drops into the lower field.
                drops.reverse()
                pairs = list(zip(lower, drops))
                surplus = lower[len(pairs):] + drops[len(pairs):]
                outcomes = [self.match(a, b, f"lower-drop-{round_number}") for a, b in pairs]
                lower = [w for w, _ in outcomes] + surplus
                eliminated.append([loser for _, loser in outcomes])
                self.certify_elimination(eliminated[-1], len(teams) - sum(map(len, eliminated)))
            if len(upper) == 1:
                while len(lower) > 1:
                    outcomes = [self.match(a, b, f"lower-final-{round_number}") for a, b in zip(lower[::2], lower[1::2])]
                    surplus = lower[-1:] if len(lower) % 2 else []
                    lower = [w for w, _ in outcomes] + surplus
                    eliminated.append([loser for _, loser in outcomes])
                    self.certify_elimination(eliminated[-1], len(teams) - sum(map(len, eliminated)))
                self.final_participants(upper[0], lower[0])
                winner, loser = self.match(upper[0], lower[0], "grand-final")
                if winner == lower[0] and self.stage["final_reset"] == "if_lower_wins":
                    winner, loser = self.match(upper[0], lower[0], "grand-final-reset")
                return [winner, loser] + [t for bucket in reversed(eliminated)
                                          for t in self.order(bucket, self.log, self.stage.get("tiebreakers"))]
            target = len(upper) // 2
            while len(lower) > target:
                # Sparse first rounds can leave one unmatched lower seed; it receives
                # an explicit structural lower bye, never an invented played match.
                pairs_needed = min(len(lower) // 2, len(lower) - target)
                outcomes = [self.match(lower[2*i], lower[2*i+1], f"lower-{round_number}") for i in range(pairs_needed)]
                lower = [w for w, _ in outcomes] + lower[2*pairs_needed:]
                eliminated.append([loser for _, loser in outcomes])
                self.certify_elimination(eliminated[-1], len(teams) - sum(map(len, eliminated)))
            upper_pairs = list(zip(upper[::2], upper[1::2]))
            round_number += 1

    def round_robin(self, groups):
        all_teams = [team for teams in groups.values() for team in teams]
        membership = {team: group for group, teams in groups.items() for team in teams}
        carried = []
        if "carry_from" in self.stage:
            field = set(all_teams)
            carried = [r for r in self.history[self.stage["carry_from"]]
                       if (r["team_a"] in field or r["team_b"] in field) and
                       (self.stage["carry_policy"] == "all" or {r["team_a"], r["team_b"]} <= field)]
        if "schedule" in self.stage:
            for i, match in enumerate(self.stage["schedule"], 1):
                a, b = self.resolve(match["a"]), self.resolve(match["b"])
                self.match(a, b, i, membership[a] if membership[a] == membership[b] else "cross_group",
                           match.get("best_of"), match.get("weight", 1))
        else:
            cross = self.stage.get("cross_group", "none")
            for cycle in range(self.stage.get("cycles", 1)):
                weight = self.stage.get("cycle_weights", [1] * self.stage.get("cycles", 1))[cycle]
                for a, b in combinations(all_teams, 2):
                    same = membership[a] == membership[b]
                    if (same and cross != "only") or (not same and cross != "none"):
                        self.match(a, b, cycle + 1, membership[a] if same else "cross_group", weight=weight)
        records = carried + self.log
        rankings = {group: self.order(teams, records, self.stage["tiebreakers"], group) for group, teams in groups.items()}
        # Classification tiebreaks resolve placement, not regular-season win totals.
        self.records[self.stage["id"]] = {t: {k: v for k, v in s.items() if k in {"wins", "losses"}}
                                          for t, s in self.stats(all_teams, records).items()}
        self.history[self.stage["id"]] = records
        if "group_tiebreakers" in self.stage:
            totals = {group: sum(self.records[self.stage["id"]][team]["wins"] for team in teams)
                      for group, teams in groups.items()}
            names = list(groups)
            if self.stage["group_tiebreakers"][-1] == "random":
                published = self.checkpoint.group_orders.get(self.stage["id"]) if self.checkpoint else None
                if published is not None:
                    if self.validate_state:
                        published_at = self.checkpoint.publication_times[("group_orders", self.stage["id"])][0]
                        if (self.future_series != self.stage_future_start or
                                any(self.team_available.get(t, published_at) > published_at for t in all_teams)):
                            raise ValueError("Published group order depends on unresolved or unavailable history")
                        if "wins" in self.stage["group_tiebreakers"] and any(
                                totals[a] < totals[b] for a, b in zip(published, published[1:])):
                            raise ValueError("Published group order contradicts group win totals")
                    names = list(published)
                    self.used_group_orders.add(self.stage["id"])
                else:
                    self.rng.shuffle(names)
                    if "wins" not in self.stage["group_tiebreakers"] or len(set(totals.values())) != len(totals):
                        self.random_order = True
            if "wins" in self.stage["group_tiebreakers"]:
                names.sort(key=totals.__getitem__, reverse=True)
            self.group_orders[self.stage["id"]] = names
        return rankings

    def swiss(self, teams, source_groups):
        records = {team: {"wins": 0, "losses": 0} for team in teams}
        fixed_rounds = self.stage.get("rounds")
        win_limit = self.stage.get("wins_to_advance")
        loss_limit = self.stage.get("losses_to_eliminate")
        rounds = fixed_rounds or win_limit + loss_limit - 1
        active = list(teams)
        for round_number in range(1, rounds + 1):
            if not active:
                break
            self.round_boundary(round_number)
            draw = self.stage.get("draw", {}) if round_number == 1 else {}
            if round_number == 1 and "initial_pairings" in self.stage:
                pairs = [tuple(self.resolve(ref) for ref in pair) for pair in self.stage["initial_pairings"]]
            elif round_number > 1 and self.stage.get("rematch_fallback") == "waive_only_if_no_feasible_draw":
                def allowed(a, b):
                    return (records[a] == records[b] and frozenset((a, b)) not in self.played)

                pairs = self.published_pairs(active, round_number, required=True, allowed=allowed)
                if pairs is None:
                    self.unknown_blocks.add((self.stage["id"], "all", round_number))
                    # Only an unknown draw may waive rematches, and only when
                    # no same-record non-rematch matching exists.
                    pairs = []
                    buckets = {}
                    for team in active:
                        buckets.setdefault((records[team]["wins"], records[team]["losses"]), []).append(team)
                    for bucket in buckets.values():
                        try:
                            pairs.extend(_perfect_matching(bucket, allowed, lambda a, b: 0, self.rng))
                        except ValueError as exc:
                            if "No feasible" not in str(exc):
                                raise
                            pairs.extend(_perfect_matching(bucket, lambda a, b: True, lambda a, b: 0, self.rng))
            else:
                pairs = self.draw(active, draw, source_groups, self.stage.get("no_rematches", True),
                                  records, round_number)
            for a, b in pairs:
                decider = fixed_rounds is None and any(
                    records[t]["wins"] == win_limit - 1 or records[t]["losses"] == loss_limit - 1 for t in (a, b))
                bo = self.stage.get("decider_best_of", 3) if decider else self.stage["best_of"]
                winner, loser = self.match(a, b, round_number, best_of=bo)
                records[winner]["wins"] += 1
                records[loser]["losses"] += 1
            if fixed_rounds is None:
                active = [t for t in active if records[t]["wins"] < win_limit and records[t]["losses"] < loss_limit]
        qualified = list(teams) if fixed_rounds else [t for t in teams if records[t]["wins"] == win_limit]
        eliminated = [] if fixed_rounds else [t for t in teams if records[t]["losses"] == loss_limit]
        if self.collect_facts and fixed_rounds is None and "advance" in self.stage:
            # Threshold exits are facts even while the rest of the field is live.
            # Certify only the broad declared qualification bands: future outcomes
            # or random within-record tiebreaks may still change individual ranks.
            for field, ranks in (
                    (qualified, range(1, self.stage["advance"] + 1)),
                    (eliminated, range(self.stage["advance"] + 1, len(teams) + 1))):
                known = set(field) - self.uncertain
                if known:
                    self.certified_bands.append((self.stage["id"], "all", set(ranks), known))
        ranking = []
        for field in (qualified, eliminated):
            buckets = {}
            for team in field:
                buckets.setdefault((records[team]["wins"], -records[team]["losses"]), []).append(team)
            for key in sorted(buckets, reverse=True):
                ranking.extend(self.order(buckets[key], self.log, self.stage["tiebreakers"]))
        if fixed_rounds is None and "advance" in self.stage and len(qualified) != self.stage["advance"]:
            raise ValueError("Swiss advance count disagrees with dynamically qualified terminal records")
        if (fixed_rounds is None and self.entrants_known and
                self.future_series == self.stage_future_start):
            self.completed_swiss.add(self.stage["id"])
        self.records[self.stage["id"]] = records
        return ranking

    def pick_and_play(self, teams):
        pairs = [(self.resolve(pair["a"]), self.resolve(pair["b"])) for pair in self.stage["initial_pairings"]]
        for round_number in range(1, self.stage["rounds"] + 1):
            self.round_boundary(round_number)
            if round_number > 1:
                remaining = self.order(teams, self.log, self.stage["tiebreakers"])
                records = self.stats(teams, self.log)
                pairs = []
                while remaining:
                    chooser = remaining.pop()
                    # Freeze the whole round's pairing before any new result.
                    index = next((i for i in range(len(remaining) - 1, -1, -1)
                                  if records[remaining[i]]["wins"] > records[chooser]["wins"]),
                                 len(remaining) - 1)
                    pairs.append((chooser, remaining.pop(index)))
            pairs = self.fixed_pairs(pairs, round_number, required=round_number > 1)
            for a, b in pairs:
                self.match(a, b, round_number)
        return self.order(teams, self.log, self.stage["tiebreakers"])

    def graph(self):
        outputs, pools = {}, {}
        seed_order = self.index if self.stage.get("seed_order") == "tournament" else self.seed_order

        def resolve(slot):
            if slot is None:
                return None
            if isinstance(slot, Mapping) and "seeded" in slot:
                pool = frozenset(_graph_source_token(source) for source in slot["seeded"])
                if pool not in pools:
                    pools[pool] = sorted((resolve(source) for source in slot["seeded"]), key=seed_order.__getitem__)
                    if set(pools[pool]) & self.uncertain:
                        self.uncertain.update(pools[pool])
                        if self.collect_facts:
                            possible = set().union(*(self.possible_teams[team] for team in pools[pool]))
                            for team in pools[pool]:
                                self.possible_teams[team] = possible
                return pools[pool][slot["rank"] - 1]
            if isinstance(slot, Mapping) and set(slot) in ({"winner"}, {"loser"}):
                outcome, node = next(iter(slot.items()))
                return outputs[node][0 if outcome == "winner" else 1]
            return self.resolve(slot)

        for node in self.stage["matches"]:
            a, b = resolve(node["a"]), resolve(node["b"])
            if self.spec.get("final", {}).get("stage") == self.stage["id"] and self.spec["final"]["round"] == node["id"]:
                if a is None or b is None:
                    raise ValueError("Declared final must have two actual participants")
                self.final_participants(a, b)
            outputs[node["id"]] = self.match(a, b, node["id"], best_of=node.get("best_of"))
        ranking = [resolve(slot) for slot in self.stage["ranking"]]
        for rank, team in enumerate(ranking, 1):
            if self.collect_facts and team not in self.uncertain:
                self.certified_bands.append((self.stage["id"], "all", {rank}, {team}))
        return ranking

    def run(self):
        for stage in self.spec["stages"]:
            self.stage, self.log = stage, []
            self.uncertain = set()
            self.random_order = False
            self.stage_future_start = self.future_series
            ref_groups = _groups(stage)
            refs = [ref for entrants in ref_groups.values() for ref in entrants]
            source = stage.get("swiss_quarterfinal_from")
            known_swiss_supply = (
                source in self.completed_swiss and
                {_token(ref) for ref in refs} == {
                    _token({"stage": source, "group": "all", "rank": rank}) for rank in range(1, 9)})
            if known_swiss_supply:
                # Consume the complete qualifier SET, not any individual Swiss
                # rank. Random within-record orders remain unresolved upstream.
                groups = {"all": self.rankings[source]["all"][:8]}
                if self.validate_state and source in self.stage_available:
                    available = self.stage_available[source]
                    for team in groups["all"]:
                        self.team_available[team] = max(self.team_available.get(team, available), available)
            else:
                groups = {name: [self.resolve(ref) for ref in entrants]
                          for name, entrants in ref_groups.items()}
            teams = [team for entrants in groups.values() for team in entrants]
            if len(set(teams)) != len(teams):
                raise ValueError("Dynamic rank references resolve to duplicate stage entrants")
            self.seed_order = {team: i for i, team in enumerate(teams)}
            source_groups = {team: ref.get("group") for team, ref in zip(teams, refs) if isinstance(ref, Mapping)}
            upstream_uncertain = not known_swiss_supply and any(
                isinstance(ref, Mapping) and self.stage_uncertain.get(ref["stage"]) for ref in refs)
            if "carry_from" in stage:
                upstream_uncertain |= self.stage_uncertain[stage["carry_from"]]
                if self.validate_state and upstream_uncertain and stage["id"] in self.checkpoint.stage_facts:
                    raise ValueError("Completed carryover stage requires complete earlier results")
            if upstream_uncertain:
                self.uncertain.update(teams)
            self.entrants_known = not upstream_uncertain
            if self.collect_facts:
                self.possible_teams = {
                    team: set(self.spec["teams"]) if upstream_uncertain else {team} for team in teams}
            self.input_available = {team: self.team_available[team] for team in teams if team in self.team_available}
            kind = stage["kind"]
            if kind == "round_robin":
                ranks = self.round_robin(groups)
            else:
                if kind == "single_elimination":
                    ranking = self.knockout(teams, source_groups)
                elif kind == "double_elimination":
                    ranking = self.double_elimination(teams)
                elif kind == "gauntlet":
                    winner, eliminated = teams[-1], []
                    for i, challenger in enumerate(reversed(teams[:-1]), 1):
                        if i == len(teams) - 1:
                            self.final_participants(challenger, winner)
                        winner, loser = self.match(challenger, winner, i)
                        eliminated.append(loser)
                        self.certify_elimination([loser], len(teams) - i)
                    ranking = [winner] + list(reversed(eliminated))
                elif kind == "gsl":
                    a, b = self.match(teams[0], teams[3], "opening-1"), self.match(teams[1], teams[2], "opening-2")
                    bo = stage.get("decider_best_of", stage["best_of"])
                    first, candidate = self.match(a[0], b[0], "winners", best_of=bo)
                    survivor, fourth = self.match(a[1], b[1], "losers", best_of=bo)
                    second, third = self.match(candidate, survivor, "decider", best_of=bo)
                    ranking = [first, second, third, fourth]
                    for rank, team in enumerate(ranking, 1):
                        if self.collect_facts and team not in self.uncertain:
                            self.certified_bands.append((stage["id"], "all", {rank}, {team}))
                elif kind == "swiss":
                    ranking = self.swiss(teams, source_groups)
                elif kind == "pick_and_play":
                    ranking = self.pick_and_play(teams)
                else:
                    ranking = self.graph()
                ranks = {"all": ranking}
                self.history[stage["id"]] = list(self.log)
                if stage["id"] not in self.records:
                    self.records[stage["id"]] = {t: {k: v for k, v in s.items() if k in {"wins", "losses"}}
                                                  for t, s in self.stats(teams, self.log).items()}
            for name, ranking in ranks.items():
                if len(ranking) != len(groups[name]) or set(ranking) != set(groups[name]):
                    raise ValueError("Stage ranking violates participant conservation")
            self.rankings[stage["id"]] = ranks
            if self.validate_state:
                times = [self.team_available[team] for team in teams if team in self.team_available]
                times.extend(available for (field, key), (_, available) in self.checkpoint.publication_times.items()
                             if (key if field == "group_orders" else key[0]) == stage["id"])
                if times:
                    self.stage_available[stage["id"]] = max(times)
            self.stage_uncertain[stage["id"]] = (upstream_uncertain or self.random_order or
                                                self.future_series != self.stage_future_start)
            if self.collect_facts and not self.stage_uncertain[stage["id"]]:
                for group, ranking in ranks.items():
                    for rank, team in enumerate(ranking, 1):
                        self.certified_bands.append((stage["id"], group, {rank}, {team}))
        if self.validate_state:
            if self.used_completed != self.checkpoint.completed.keys():
                raise ValueError("Completed history does not match a coherent tournament route")
            for key, pairs in self.checkpoint.pairings.items():
                if self.block_pairs.get(key) != {frozenset(pair) for pair in pairs}:
                    raise ValueError("Published pairing does not match a coherent tournament route")
            if self.used_orders != self.checkpoint.orders.keys():
                raise ValueError("Published order does not match an actual random tie")
            if self.used_group_orders != self.checkpoint.group_orders.keys():
                raise ValueError("Published group order does not match a random group tiebreak")
        return self


def _structural_support(spec, run):
    """Conservative rank supports propagated through declared supplies, not MC hits."""
    supports = {}

    def resolve(ref):
        if isinstance(ref, str):
            return {ref}
        groups = supports[ref["stage"]]
        if "group" in ref:
            selected = [ref["group"]]
        elif not run.stage_uncertain[ref["stage"]]:
            selected = [run.group_orders[ref["stage"]][ref["group_rank"] - 1]]
        else:
            selected = groups
        return set().union(*(groups[group][ref["rank"] - 1] for group in selected))

    certificates = {}
    for stage, group, ranks, members in run.certified_bands:
        certificates.setdefault((stage, group), []).append((ranks, members))
    for stage in spec["stages"]:
        groups = {}
        for group, refs in _groups(stage).items():
            entrants = set().union(*(resolve(ref) for ref in refs))
            ranks = [set(entrants) for _ in refs]
            for positions, members in certificates.get((stage["id"], group), []):
                for position, candidates in enumerate(ranks, 1):
                    if position not in positions:
                        candidates.difference_update(members)
                    elif len(positions) == len(members):
                        candidates.intersection_update(members)
            if any(not candidates for candidates in ranks):
                raise ValueError("Structural placement facts contradict declared stage supplies")
            groups[group] = ranks
        supports[stage["id"]] = groups
    return supports, resolve

def _draw_laws(spec):
    """Describe implemented scenario laws, not an authenticated official lottery."""
    laws = {}
    for stage in spec["stages"]:
        descriptions = []
        if "initial_pairings" in stage:
            descriptions.append("fixed supplied opening draw; availability/authenticity follows caller scenario provenance")
        if stage["kind"] == "swiss" or "draw" in stage:
            descriptions.append("uniform optimal feasible complete matching; official sequential lottery unverified")
        if "swiss_quarterfinal_from" in stage:
            descriptions.append("scenario: shuffled record pools, 3-0 versus 3-2, opposite halves; official lottery unverified")
        if "random" in stage.get("tiebreakers", []) or "random" in stage.get("group_tiebreakers", []):
            descriptions.append("uniform permutation of tied entries")
        if not descriptions:
            descriptions.append("deterministic declared bracket/seed route; no lottery")
        laws[stage["id"]] = descriptions
    return laws



def _joint_target_design(spec, has_finalists):
    """Declare categorical target sizes from validated topology, not sampled keys."""
    return {
        "teams": list(spec["teams"]),
        "finalist_count": 2 if has_finalists else None,
        "advance_counts": {
            stage["id"]: stage["advance"] * len(_groups(stage))
            for stage in spec["stages"] if "advance" in stage
        },
    }


def simulate_tournament(spec, probabilities, *, simulations=1000, seed=81, score_distributions=None,
                        state=None, observables=None, team_strength_draws=None,
                        conditional_score_distributions=None, integration="monte_carlo", copula_rho=None):
    """Simulate complete series, freezing available checkpoint facts when supplied.

    Probability keys are (team_a, team_b, best_of); one orientation per pair. Extra
    Bo1/3/5 units may be supplied but all global pairs for every required unit must
    exist. Optional scores are sampled conditional on the winner from a separate
    losing-map PMF. All inputs are read-only. Invalid/infeasible rules raise ValueError.

    team_strength_draws maps every entrant to finite SERIES-logit shifts of length
    simulations. One vector is reused for all appearances in each rollout; draws
    are supplied externally and consume no outcome/draw RNG. Zero draws preserve
    the default path exactly and p=0/1 remain absorbing. At checkpoints the caller
    must disclose an earlier-fitted/reconditioned posterior inference policy:
    forcing observed results does NOT condition latent draws on that evidence.

    copula_rho in [0, .5] explicitly selects the shared-state conditional probit
    kernel, with supplied standardized Gaussian team states rather than logit
    shifts. Prior Gaussian integration preserves the base margins; arbitrary
    supplied vectors do not establish that prior or an iid sampling design.
    Positive rho requires explicit states, pre-event Monte Carlo, and no checkpoint.
    Zero rho preserves the original outcome stream even for nonzero supplied states.
    Conditional losing-map bins stay at the original winner probability in this
    mode, preserving the prior pair scoreline law rather than changing two factors.

    conditional_score_distributions is {"probability_bins": [0, ..., 1],
    "distributions": {best_of: [losing-map PMF per bin]}}. Bins refer to the
    sampled winner's predicted probability, left-inclusive and last right-inclusive.
    Every required Bo3/5 and bin must be fitted explicitly; no pooled fallback.
    It is mutually exclusive with score_distributions and never changes p.

    State v1 requires UTC cutoff and completed trace rows augmented by completed_at
    and available_at. Actual completed scores are mandatory, independent of the
    future score PMF or current model likelihood. Optional pairings and orders
    contain published_at/available_at. Historical random draws and opponent choices
    require complete published round pairings; no historical draw is resampled.
    A Swiss quarterfinal draw may consume a fully completed eight-team qualifier
    set without published within-record tie orders; individual ranks stay unresolved.
    ``trace`` and ``stage_records`` illustrate only the first simulation. Raw hits,
    finalist pairs and declared official placement bands aggregate every rollout.

    Optional observables declare id, kind and exhaustive string categories. Count
    kinds are future_series_count, future_map_count, upset_count and
    repeat_encounter_count. Their scope accepts stage/group and period (future by
    default; all only for upset/repeat). Counts ignore standings weights/carryover.
    Repeats count encounters after the first within scope, including completed
    history as the reference for future encounters. Upsets require a fixed
    reference_probabilities series table and count winner probabilities strictly
    below threshold (default .5); reference ties are never upsets.

    node_matchup uses sorted compact JSON pairs; node_score uses participants in
    declared orientation and legal "wins-losses" categories. A node is
    {stage, group="all", round, match=1}, with explicit 1-based match index required
    for multi-slot blocks (bye slots count). absent_category must be declared
    whenever the node or specified score matchup may be absent; no conditioning
    away non-reach. champion_path requires a source reference and paths mapping
    each category to node/outcome predicates (winner, loser, participant, absent)
    relative to the eventual champion. Paths must form an exhaustive partition.
    Source references are caller evidence, not availability certification.

    Aggregates retain every declared category including zero hits. Observable
    structural-zero lists are deliberately empty: unsampled is not impossible.
    Score-dependent observables reject unknown scores instead of assuming sweeps.
    """
    if integration not in {"monte_carlo", "exact", "hybrid"}:
        raise ValueError("integration must be monte_carlo, exact, or hybrid")
    formats = validate_tournament_spec(spec)
    _positive(simulations, "simulations")
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    if not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must be a mapping")
    supplied_formats = set(formats)
    for key in probabilities:
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("Probability keys must be (team_a, team_b, best_of)")
        _best_of(key[2])
        supplied_formats.add(key[2])
    bracket = SimpleNamespace(teams=spec["teams"], matches={bo: SimpleNamespace(best_of=bo) for bo in supplied_formats})
    tables = _probability_tables(bracket, dict(probabilities))
    copula_probits = copula_transform = None
    if copula_rho is not None:
        if (isinstance(copula_rho, bool) or not isinstance(copula_rho, Real)
                or not math.isfinite(copula_rho) or not 0 <= copula_rho <= .5):
            raise ValueError("copula_rho must be finite and in [0, .5]")
        copula_rho = float(copula_rho)
        if copula_rho > 0:
            if integration != "monte_carlo":
                from src.models.tournament_exact import UnsupportedIntegrationError
                raise UnsupportedIntegrationError("Positive copula rho requires explicit pre-event monte_carlo integration")
            if state is not None:
                raise ValueError("Positive copula rho requires posterior conditioning before checkpoint forecasts")
            if team_strength_draws is None:
                raise ValueError("Positive copula rho requires explicit standardized team draws")
            from scipy.special import ndtri
            from src.models.tournament_copula import _probit_copula_probability
            # One authoritative probit per pair: reinverting a rounded 1-p can
            # manufacture certainty or amplify tail-rounding under latent shifts.
            copula_probits = {bo: np.zeros_like(table) for bo, table in tables.items()}
            index = {team: i for i, team in enumerate(spec["teams"])}
            tails = {}
            for (a, b, bo), value in probabilities.items():
                i, j = index[a], index[b]
                pair = (bo, min(i, j), max(i, j))
                if pair in tails and value >= tails[pair]:
                    continue
                tails[pair] = value
                z = ndtri(float(value))
                copula_probits[bo][i, j] = z
                copula_probits[bo][j, i] = -z
            copula_transform = _probit_copula_probability
    strength = _strength_vectors(
        team_strength_draws, spec["teams"], simulations,
        preserve_zero=copula_rho is not None and copula_rho > 0,
        disabled=copula_rho == 0)
    conditional_scores = _conditional_scores(conditional_score_distributions, formats)
    if conditional_scores is not None and score_distributions is not None:
        raise ValueError("Pooled and conditional score models are mutually exclusive")
    scores = {} if score_distributions is None else score_distributions
    if not isinstance(scores, Mapping):
        raise ValueError("score_distributions must be a mapping")
    for bo, pmf in scores.items():
        _best_of(bo)
        if (not isinstance(pmf, (list, tuple)) or len(pmf) != bo // 2 + 1 or
                any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or p < 0 for p in pmf) or
                not math.isclose(sum(pmf), 1.0, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Invalid conditional losing-map score distribution")
    contributing_formats = {}
    for stage in spec["stages"]:
        stage_formats = set(required_best_ofs({"stages": [stage]}))
        if "carry_from" in stage:
            stage_formats.update(contributing_formats[stage["carry_from"]])
        contributing_formats[stage["id"]] = stage_formats
        if state is None and set(stage.get("tiebreakers", [])) & {"map_diff", "map_wins"}:
            if any(bo != 1 and bo not in scores and conditional_scores is None for bo in stage_formats):
                raise ValueError("Map standings require separate score distributions for contributing series formats")
    if integration != "monte_carlo":
        from src.models.tournament_exact import integrate_tournament
        result = integrate_tournament(
            spec, tables, simulations=simulations, seed=seed, integration=integration,
            scores=scores, conditional_scores=conditional_scores,
            strength=strength, state=state, observables=observables)
        result["integration"]["joint_target_design"] = _joint_target_design(
            spec, result["final_prob"] is not None)
        return result
    checkpoint = _Checkpoint(state, spec) if state is not None else None
    if checkpoint and not (checkpoint.completed or checkpoint.pairings or checkpoint.orders or checkpoint.group_orders):
        checkpoint = None
    manifest = _prepare_observables(observables, spec)
    observable_hits = {row["id"]: dict.fromkeys(row["categories"], 0) for row in manifest}
    if checkpoint:
        # Validate routes once with an independent stream; never reject/sample a
        # past result according to its likelihood under the new probability table.
        _Run(spec, tables, scores, random.Random(int(seed)), False, checkpoint, True,
             conditional_scores=conditional_scores).run()
    teams = spec["teams"]
    counts = {stage["id"]: {group: {team: [0] * len(refs) for team in teams}
                            for group, refs in _groups(stage).items()} for stage in spec["stages"]}
    advance = {stage["id"]: dict.fromkeys(teams, 0) for stage in spec["stages"] if "advance" in stage}
    champion = dict.fromkeys(teams, 0) if spec.get("champion") is not None else None
    finalist_counts = dict.fromkeys(teams, 0)
    joint_final = {}
    joint_advance = {stage: {} for stage in advance}
    placement_categories = [band["label"] for band in spec.get("placement_bands", [])]
    placement_counts = {team: dict.fromkeys(placement_categories, 0) for team in teams}
    future_series = 0
    rng = random.Random(int(seed))
    total_series, first = 0, None
    for iteration in range(simulations):
        run = _Run(spec, tables, scores, rng, iteration == 0, checkpoint, observables=manifest,
                   strength=None if strength is None else strength[iteration],
                   conditional_scores=conditional_scores, copula_rho=copula_rho,
                   copula_probits=copula_probits, copula_transform=copula_transform).run()
        if run.observables:
            run.observables.finish(run, observable_hits)
        if first is None:
            first = run
        total_series += run.series
        future_series += run.future_series
        if run.finalists is not None:
            for team in run.finalists:
                finalist_counts[team] += 1
            pair = json.dumps(sorted(run.finalists), separators=(",", ":"), ensure_ascii=False)
            joint_final[pair] = joint_final.get(pair, 0) + 1
        placed = set()
        for band in spec.get("placement_bands", []):
            for ref in band["placements"]:
                team = run.resolve(ref)
                if team in placed:
                    raise ValueError("Official placement bands violate participant conservation")
                placed.add(team)
                placement_counts[team][band["label"]] += 1
        if champion is not None:
            champion[run.resolve(spec["champion"])] += 1
        for stage in spec["stages"]:
            for group, ranking in run.rankings[stage["id"]].items():
                for rank, team in enumerate(ranking):
                    counts[stage["id"]][group][team][rank] += 1
                    if stage["id"] in advance and rank < stage["advance"]:
                        advance[stage["id"]][team] += 1
            if stage["id"] in advance:
                qualified = sorted(team for ranking in run.rankings[stage["id"]].values()
                                   for team in ranking[:stage["advance"]])
                key = json.dumps(qualified, separators=(",", ":"), ensure_ascii=False)
                rows = joint_advance[stage["id"]]
                rows[key] = rows.get(key, 0) + 1
    certified = first.certified_bands
    support, possible_ref = _structural_support(spec, first)
    stage_rank_possible = {
        stage: {group: {team: [rank for rank, candidates in enumerate(ranks, 1) if team in candidates]
                        for team in teams} for group, ranks in groups.items()}
        for stage, groups in support.items()}
    possible_placements = {team: [] for team in teams}
    for band in spec.get("placement_bands", []):
        candidates = set().union(*(possible_ref(ref) for ref in band["placements"]))
        for team in candidates:
            possible_placements[team].append(band["label"])
    stage_rank_resolved = {
        stage: {group: dict.fromkeys(teams, not first.stage_uncertain[stage]) for group in groups}
        for stage, groups in first.rankings.items()}
    for stage, group, ranks, members in certified:
        if len(ranks) == 1:
            for team in members:
                stage_rank_resolved[stage][group][team] = True
    champion_ref = spec.get("champion")
    champion_resolved = bool(champion_ref and not first.stage_uncertain[champion_ref["stage"]])
    if champion_ref and "group" in champion_ref:
        champion_resolved |= any(stage == champion_ref["stage"] and group == champion_ref["group"] and
                                 ranks == {champion_ref["rank"]} and len(members) == 1
                                 for stage, group, ranks, members in certified)
    zero_champion, zero_final = set(), set()
    if champion_ref and "group" in champion_ref:
        for stage, group, ranks, members in certified:
            if stage == champion_ref["stage"] and group == champion_ref["group"]:
                if champion_ref["rank"] not in ranks:
                    zero_champion.update(members)
                if first.final_stage == stage and min(ranks) > 2:
                    zero_final.update(members)
    if champion_ref:
        zero_champion.update(set(teams) - possible_ref(champion_ref))
    if first.final_stage is not None:
        finalists = set().union(*support[first.final_stage]["all"][:2])
        zero_final.update(set(teams) - finalists)
    advance_resolved = {stage: dict.fromkeys(teams, not first.stage_uncertain[stage]) for stage in advance}
    for stage in spec["stages"]:
        if stage["id"] in advance:
            guaranteed = set()
            for group, refs in _groups(stage).items():
                positions = support[stage["id"]][group]
                candidates = set().union(*positions)
                if len(candidates) == len(positions):
                    guaranteed.update(candidates)
                for ref in refs:
                    entrants = possible_ref(ref)
                    if len(entrants) == 1:
                        guaranteed.update(entrants)
            for source, group, ranks, members in certified:
                if source == stage["id"]:
                    guaranteed.update(members)
            qualification = set().union(*(candidates for positions in support[stage["id"]].values()
                                           for candidates in positions[:stage["advance"]]))
            nonqualification = set().union(*(candidates for positions in support[stage["id"]].values()
                                              for candidates in positions[stage["advance"]:]))
            for team in teams:
                if team not in qualification or (team in guaranteed and team not in nonqualification):
                    advance_resolved[stage["id"]][team] = True
            for source, group, ranks, members in certified:
                if source == stage["id"] and (max(ranks) <= stage["advance"] or min(ranks) > stage["advance"]):
                    for team in members:
                        advance_resolved[source][team] = True
    placement_resolved = dict.fromkeys(teams, False)
    for band in spec.get("placement_bands", []):
        by_group = {}
        for ref in band["placements"]:
            if "group" in ref:
                by_group.setdefault((ref["stage"], ref["group"]), set()).add(ref["rank"])
        for stage, group, ranks, members in certified:
            if ranks <= by_group.get((stage, group), set()):
                for team in members:
                    placement_resolved[team] = True
    zero_joint_final = [
        json.dumps(pair, separators=(",", ":"), ensure_ascii=False)
        for pair in combinations(sorted(teams), 2)
        if first.final_support is not None and (
            pair not in first.final_support or set(pair).intersection(zero_final))]
    result = {
        "champion_prob": None if champion is None else {team: value / simulations for team, value in champion.items()},
        "stage_rank_prob": {stage: {group: {team: [value / simulations for value in vector]
                                            for team, vector in rows.items()} for group, rows in groups.items()}
                            for stage, groups in counts.items()},
        "advance_prob": {stage: {team: value / simulations for team, value in rows.items()} for stage, rows in advance.items()},
        "expected_series": total_series / simulations, "trace": first.trace,
        "stage_records": first.records, "simulations": simulations, "seed": seed, "probability_unit": "series_win",
        "trace_scope": "first_simulation_only", "stage_records_scope": "first_simulation_only",
        "future_expected_series": future_series / simulations,
        "completed_series": len(checkpoint.completed) if checkpoint else 0,
        "final_prob": {team: value / simulations for team, value in finalist_counts.items()} if first.finalists else None,
        "joint_final_prob": {pair: value / simulations for pair, value in joint_final.items()} if first.finalists else None,
        "joint_advance_prob": {stage: {key: value / simulations for key, value in rows.items()}
                               for stage, rows in joint_advance.items()},
        "placement_categories": placement_categories,
        "placement_prob": ({team: {label: value / simulations for label, value in rows.items()}
                             for team, rows in placement_counts.items()} if placement_categories else None),
        "hits": {"champion": champion, "stage_rank": counts, "advance": advance,
                 "final": finalist_counts if first.finalists else None, "joint_final": joint_final,
                 "joint_advance": joint_advance, "placement": placement_counts if placement_categories else None},
        "target_resolved": {
            "champion": champion_resolved,
            "champion_by_team": {team: champion_resolved or team in zero_champion for team in teams},
            "final": first.final_resolved,
            "final_by_team": {team: first.final_resolved or team in zero_final or team in first.known_finalists
                              for team in teams},
            "placement_by_team": placement_resolved,
            "advance": {stage: all(rows.values()) for stage, rows in advance_resolved.items()},
            "advance_by_team": advance_resolved,
        },
        "stage_rank_resolved": stage_rank_resolved,
        "certified_placement_bands": [
            {"stage": stage, "group": group, "ranks": sorted(ranks), "teams": sorted(members)}
            for stage, group, ranks, members in certified],
        "known_pairings": first.known_pairings,
        "known_pairings_scope": "structurally_fixed_participants_only",
        "stage_rank_possible": stage_rank_possible,
        "possible_placement_categories": possible_placements if placement_categories else None,
        "structural_zero_categories": {"champion": sorted(zero_champion), "final": sorted(zero_final),
                                      "joint_final": zero_joint_final},
        "state_qualification": (
            "daily_reconstructed_not_availability_certified"
            if state and state.get("temporal_policy") == "daily_rollforward"
            else "reconstructed_not_availability_certified"
            if state and state.get("temporal_policy") == "prior_day_reconstruction"
            else "strict_timestamped"),
        "integration": {
            "kind": "monte_carlo",
            "effective_sampling_unit": ("independent_complete_path" if strength is None
                                        else "finite_latent_vector_conditional_path"),
            "fixed_n": simulations, "stopping_rule": "fixed_n", "iid": strength is None,
            "approximation": "sampled complete tournament paths",
            "draw_laws": _draw_laws(spec),
            "joint_target_design": _joint_target_design(spec, bool(first.finalists)),
            "copula_rho": copula_rho,
            "probability_link": ("conditional_probit" if copula_rho is not None
                                 else "series_logit_offset" if strength is not None
                                 else "fixed_series_probability"),
        },
    }
    if observables is not None:
        result["observable_prob"] = {
            identity: {category: count / simulations for category, count in rows.items()}
            for identity, rows in observable_hits.items()}
        result["hits"]["observable"] = observable_hits
        result["structural_zero_categories"]["observable"] = {identity: [] for identity in observable_hits}
    return result
