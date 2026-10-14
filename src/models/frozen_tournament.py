"""Pre-start fixed-graph research simulation using complete SERIES probabilities.

This path neither reads live results nor infers map probabilities. Double elimination
means the supplied unconditional graph, with a single championship series and no
conditional bracket reset. Team and node identifiers are used exactly as supplied.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import heapq
from numbers import Integral, Real

import numpy as np
from scipy.special import expit, logit

from betting_app.services.tournament_service import TournamentBracket


def _integer(value: object, allowed: tuple[int, ...] | None = None) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and (allowed is None or value in allowed)
    )


def validate_bracket(bracket: TournamentBracket) -> list[str]:
    """Reject unsafe or partially played graphs; return a stable topological order.

    Slot supplies form a conserved participant flow: a seed occurs once, each slot
    has one supplier, and each match produces one winner and (unless a bye) one
    distinct loser. Routing either output at most once preserves that invariant
    through the DAG, including upper/lower-bracket rematches.
    """
    if bracket.format not in {"single_elimination", "double_elimination"}:
        raise ValueError("Only fixed single_elimination or double_elimination graphs are supported")
    if not bracket.teams or any(not isinstance(team, str) or not team.strip() for team in bracket.teams):
        raise ValueError("Entrant names must be nonempty strings")
    if len(set(bracket.teams)) != len(bracket.teams):
        raise ValueError("Entrant names must be unique exact identifiers")
    if not bracket.matches:
        raise ValueError("The bracket must contain a championship match")

    teams = set(bracket.teams)
    seed_counts: Counter[str] = Counter()
    supplies: dict[tuple[str, int], tuple[str, str]] = {}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in bracket.matches}
    indegree = dict.fromkeys(bracket.matches, 0)
    for node_id, node in bracket.matches.items():
        if not isinstance(node_id, str) or not node_id.strip() or node_id != node.id:
            raise ValueError("Every nonempty match key must equal its node.id")
        if not _integer(node.best_of, (1, 3, 5)):
            raise ValueError(f"{node_id}: best_of must be Bo1, Bo3 or Bo5")
        if node.winner is not None or node.score1 is not None or node.score2 is not None:
            raise ValueError(f"{node_id}: frozen forecasts require no winners or scores")
        for slot, team in ((1, node.team1), (2, node.team2)):
            if team is not None:
                if not isinstance(team, str) or team not in teams:
                    raise ValueError(f"{node_id}: unknown entrant in slot {slot}")
                seed_counts[team] += 1
        for outcome, target, slot in (
            ("winner", node.next_match_winner_id, node.next_match_winner_slot),
            ("loser", node.next_match_loser_id, node.next_match_loser_slot),
        ):
            if not _integer(slot, (1, 2)):
                raise ValueError(f"{node_id}: target slots must be 1 or 2")
            if target is None:
                continue
            if not isinstance(target, str) or target not in bracket.matches:
                raise ValueError(f"{node_id}: dangling {outcome} target")
            if target == node_id:
                raise ValueError(f"{node_id}: self-links are forbidden")
            if outcome == "loser" and bracket.format == "single_elimination":
                raise ValueError("Single elimination cannot route losers to another match")
            key = (target, slot)
            if key in supplies:
                raise ValueError(f"{target}: duplicate incoming supply for slot {slot}")
            supplies[key] = (node_id, outcome)
            outgoing[node_id].append(target)
            indegree[target] += 1

    ready = [node_id for node_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != len(bracket.matches):
        raise ValueError("Bracket graph contains a cycle")

    if any(seed_counts[team] != 1 for team in bracket.teams):
        raise ValueError("Each entrant must be seeded exactly once")
    terminals = [node_id for node_id in order if bracket.matches[node_id].next_match_winner_id is None]
    if len(terminals) != 1 or bracket.matches[terminals[0]].next_match_loser_id is not None:
        raise ValueError("The graph must have exactly one terminal championship node")
    championship = terminals[0]
    reaches_championship = {championship}
    for node_id in reversed(order):
        if bracket.matches[node_id].next_match_winner_id in reaches_championship:
            reaches_championship.add(node_id)
    if len(reaches_championship) != len(order):
        raise ValueError("Every match must reach the championship through winner routes")

    # Upper-bracket entrants have zero losses; lower-bracket entrants have one.
    # A loser edge after a possible second loss would implement an unsafe reset
    # (or a third-life format), neither expressible by this fixed-graph contract.
    output_losses: dict[tuple[str, str], set[int]] = {}
    for node_id in order:
        node = bracket.matches[node_id]
        occupied = 0
        has_seed = False
        losses: set[int] = set()
        for slot, team in ((1, node.team1), (2, node.team2)):
            incoming = supplies.get((node_id, slot))
            if team is not None and incoming is not None:
                raise ValueError(f"{node_id}: seed and incoming edge overwrite slot {slot}")
            if team is not None:
                occupied += 1
                has_seed = True
                losses.add(0)
            elif incoming is not None:
                occupied += 1
                losses.update(output_losses[incoming])
        if occupied != 2:
            if occupied != 1 or not has_seed or node.next_match_loser_id is not None:
                raise ValueError(f"{node_id}: missing participant supply; only explicit seeded byes are supported")
        if node.next_match_loser_id is not None and max(losses) >= 1:
            raise ValueError(f"{node_id}: conditional resets or routing twice-defeated entrants are unsupported")
        output_losses[(node_id, "winner")] = losses
        output_losses[(node_id, "loser")] = {loss + 1 for loss in losses}
    return order


def _probability_tables(
    bracket: TournamentBracket, probabilities: dict[tuple[str, str, int], float]
) -> dict[int, np.ndarray]:
    team_index = {team: index for index, team in enumerate(bracket.teams)}
    formats = {node.best_of for node in bracket.matches.values()}
    tables = {best_of: np.full((len(team_index), len(team_index)), np.nan) for best_of in formats}
    seen: set[tuple[int, int, int]] = set()
    for key, probability in probabilities.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("Probability keys must be (team1, team2, best_of) tuples")
        first, second, best_of = key
        if (not isinstance(first, str) or not isinstance(second, str)
                or first not in team_index or second not in team_index or first == second):
            raise ValueError(f"Unknown or self-paired teams in probability key: {key!r}")
        if not _integer(best_of, (1, 3, 5)) or best_of not in formats:
            raise ValueError(f"Unknown series format in probability key: {key!r}")
        if (not isinstance(probability, Real) or isinstance(probability, (bool, np.bool_))
                or not np.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError(f"Series probability must be finite and within [0, 1]: {key!r}")
        first_index, second_index = team_index[first], team_index[second]
        pair = (min(first_index, second_index), max(first_index, second_index), best_of)
        if pair in seen:
            raise ValueError(f"Duplicate pair or reverse probability entry: {key!r}")
        seen.add(pair)
        tables[best_of][first_index, second_index] = probability
        tables[best_of][second_index, first_index] = 1.0 - probability
    required = len(team_index) * (len(team_index) - 1) // 2 * len(formats)
    if len(seen) != required:
        raise ValueError("Missing series probabilities: every unordered entrant pair and graph best_of is required")
    return tables


def simulate_frozen_bracket(
    bracket: TournamentBracket,
    probabilities: dict[tuple[str, str, int], float],
    simulations: int = 20000,
    seed: int = 89,
    *,
    team_strength_draws: Mapping[str, np.ndarray] | None = None,
) -> dict:
    """Simulate a complete unplayed graph, without modifying it or global RNG.

    ``probabilities[(a, b, best_of)]`` is P(a wins the complete series), not a
    map-win probability. Supply exactly one orientation for every pair/format;
    the reverse is its complement. Node reach counts participation, including
    explicit byes; final reach counts participation in the terminal node.

    Optional draws are additive complete-series logit residual strengths: one
    finite vector per exact entrant, with one draw per simulated tournament.
    The same draw is reused for every appearance, including rematches. Sampling
    and posterior provenance belong to the caller. No map/series conversion is
    performed. Zero draws preserve the fixed-probability path bit for bit.
    """
    if not _integer(simulations) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    simulations, seed = int(simulations), int(seed)
    order = validate_bracket(bracket)
    tables = _probability_tables(bracket, probabilities)
    generator = np.random.default_rng(seed)
    team_index = {team: index for index, team in enumerate(bracket.teams)}
    team_count = len(team_index)
    strength = None
    logit_tables = None
    if team_strength_draws is not None:
        if not isinstance(team_strength_draws, Mapping) or set(team_strength_draws) != set(team_index):
            raise ValueError("Strength draws must name every exact entrant and no other teams")
        vectors = []
        for team in bracket.teams:
            values = np.asarray(team_strength_draws[team])
            if (values.shape != (simulations,) or values.dtype.kind not in "iuf"
                    or not np.isfinite(values).all()
                    or np.any(np.abs(values.astype(np.longdouble)) > np.finfo(float).max)):
                raise ValueError("Each team strength must be a finite numeric vector of length simulations")
            vectors.append(values)
        strength = np.column_stack(vectors).astype(float, copy=False)
        if not np.any(strength):
            strength = None
        else:
            logit_tables = {best_of: logit(table) for best_of, table in tables.items()}
    simulation_index = np.arange(simulations) if strength is not None else None
    pending: dict[tuple[str, int], np.ndarray] = {}
    node_reach: dict[str, dict[str, float]] = {}
    champion_counts: np.ndarray | None = None
    championship: str | None = None

    for node_id in order:
        node = bracket.matches[node_id]
        participants: list[np.ndarray] = []
        for slot, team in ((1, node.team1), (2, node.team2)):
            if team is not None:
                participants.append(np.full(simulations, team_index[team], dtype=np.intp))
            else:
                incoming = pending.pop((node_id, slot), None)
                if incoming is not None:
                    participants.append(incoming)
        reach_counts = np.bincount(participants[0], minlength=team_count)
        loser: np.ndarray | None = None
        if len(participants) == 1:
            winner = participants[0]
        else:
            first, second = participants
            if np.any(first == second):
                raise RuntimeError(f"{node_id}: participant conservation violated")
            reach_counts += np.bincount(second, minlength=team_count)
            probability = tables[node.best_of][first, second]
            if strength is not None:
                # Long double avoids overflow subtracting two valid extreme
                # finite draws. Endpoints remain structural certainties.
                delta = (strength[simulation_index, first].astype(np.longdouble)
                         - strength[simulation_index, second].astype(np.longdouble))
                logits = logit_tables[node.best_of][first, second].astype(np.longdouble) + delta
                probability = expit(np.clip(logits, -1e300, 1e300).astype(float))
                probability = np.where(tables[node.best_of][first, second] == 0, 0, probability)
                probability = np.where(tables[node.best_of][first, second] == 1, 1, probability)
                probability = np.where(delta == 0, tables[node.best_of][first, second], probability)
            first_wins = generator.random(simulations) < probability
            winner = np.where(first_wins, first, second)
            if node.next_match_loser_id is not None:
                loser = np.where(first_wins, second, first)
        node_reach[node_id] = {
            team: int(reach_counts[index]) / simulations for team, index in team_index.items()
        }
        if node.next_match_winner_id is None:
            championship = node_id
            champion_counts = np.bincount(winner, minlength=team_count)
        else:
            pending[(node.next_match_winner_id, node.next_match_winner_slot)] = winner
        if node.next_match_loser_id is not None:
            if loser is None:
                raise RuntimeError(f"{node_id}: a bye cannot produce a loser")
            pending[(node.next_match_loser_id, node.next_match_loser_slot)] = loser

    if pending or champion_counts is None or int(champion_counts.sum()) != simulations:
        raise RuntimeError("Bracket failed to conserve participants and produce one champion per simulation")
    return {
        "champion_prob": {
            team: int(champion_counts[index]) / simulations for team, index in team_index.items()
        },
        "final_prob": node_reach[championship],
        "node_reach_prob": node_reach,
        "championship_node_id": championship,
        "simulations": simulations,
        "seed": seed,
        "probability_unit": "series_win",
    }
