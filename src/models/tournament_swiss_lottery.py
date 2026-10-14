"""Riot World Championship Swiss Stage Sequential Draw Lottery Engine.

Implements the official ball-by-ball sequential draw lottery algorithm conforming to
the League of Legends World Championship Rulebook (§6.3.2–6.3.4):
- Pool-by-pool ball draws (0-0, 1-0, 0-1, 2-0, 1-1, 0-2, 2-1, 1-2, 2-2).
- Slot pairing: Match 1 Slot 1, Match 1 Slot 2, Match 2 Slot 1, Match 2 Slot 2, etc.
- Constraints:
  - Round 1: No intra-region pairings (when region metadata is provided).
  - Rounds 2–5: No rematches (teams cannot face an opponent previously played in Swiss).
- Slide-slot mechanics:
  If a drawn team creates an illegal pairing (intra-region in R1 or rematch in R2–R5)
  or mathematically forces a future deadlock, it slides to the next available match slot
  where legally permitted.
- Backtracking / minimal waiver:
  If an impossible state is reached where no legal permutation exists, the engine backtracks
  and applies minimal rematch waiver as permitted by official rules.
- 100% deterministic reproducibility when passing a seeded `random.Random(seed)`.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import random

__all__ = [
    "SwissMatchup",
    "draw_swiss_pool_lottery",
    "simulate_swiss_round_draws",
]


@dataclass
class SwissMatchup:
    """Represents a paired match in the Swiss Stage lottery.

    Attributes
    ----------
    team_a : str
        First drawn/assigned team in the match (Match Slot 1).
    team_b : str
        Second drawn/assigned team in the match (Match Slot 2).
    pool : str
        Record pool label (e.g., '0-0', '1-0', '0-1', '2-1', etc.).
    round_num : int
        Current round of the Swiss Stage (1 to 5).
    best_of : int
        Series length (1 for standard rounds, 3 for promotion/elimination deciders).
    """

    team_a: str
    team_b: str
    pool: str
    round_num: int
    best_of: int


def _is_pairing_legal(
    team_a: str,
    team_b: str,
    round_num: int,
    past_matchups: set[frozenset[str]],
    team_regions: dict[str, str] | None = None,
) -> bool:
    """Check whether pairing team_a and team_b is legally permitted under official rules."""
    if team_a == team_b:
        return False

    if round_num == 1:
        if team_regions is not None:
            region_a = team_regions.get(team_a)
            region_b = team_regions.get(team_b)
            if region_a and region_b and region_a == region_b:
                return False
    else:
        pair_set = frozenset({team_a, team_b})
        if pair_set in past_matchups or (team_a, team_b) in past_matchups or (team_b, team_a) in past_matchups:
            return False

    return True


def _can_complete_matches(
    matches: list[list[str | None]],
    remaining_pot: list[str],
    round_num: int,
    past_matchups: set[frozenset[str]],
    team_regions: dict[str, str] | None,
    budget: int,
) -> bool:
    """Forward-checking evaluator for feasible match completions within violation budget.

    Determines whether remaining teams in `remaining_pot` can legally populate all open
    slots in `matches` without exceeding the allowable violation budget (minimal waiver).
    """
    half_filled: list[str] = []
    existing_violations = 0

    for m in matches:
        t0, t1 = m[0], m[1]
        if t0 is not None and t1 is not None:
            if not _is_pairing_legal(t0, t1, round_num, past_matchups, team_regions):
                existing_violations += 1
        elif t0 is not None and t1 is None:
            half_filled.append(t0)
        elif t0 is None and t1 is not None:
            half_filled.append(t1)

    if existing_violations > budget:
        return False

    rem_budget = budget - existing_violations

    @functools.lru_cache(maxsize=None)
    def _solve(f_tuple: tuple[str, ...], r_tuple: tuple[str, ...], curr_budget: int) -> bool:
        if not f_tuple and not r_tuple:
            return True

        if f_tuple:
            u = f_tuple[0]
            rest_f = f_tuple[1:]
            for i, v in enumerate(r_tuple):
                viol = 0 if _is_pairing_legal(u, v, round_num, past_matchups, team_regions) else 1
                if viol <= curr_budget:
                    rest_r = r_tuple[:i] + r_tuple[i + 1:]
                    if _solve(rest_f, rest_r, curr_budget - viol):
                        return True
            return False
        else:
            u = r_tuple[0]
            rest_r = r_tuple[1:]
            for i, v in enumerate(rest_r):
                viol = 0 if _is_pairing_legal(u, v, round_num, past_matchups, team_regions) else 1
                if viol <= curr_budget:
                    next_r = rest_r[:i] + rest_r[i + 1:]
                    if _solve((), next_r, curr_budget - viol):
                        return True
            return False

    return _solve(tuple(sorted(half_filled)), tuple(sorted(remaining_pot)), rem_budget)


def draw_swiss_pool_lottery(
    teams: list[str],
    pool_name: str,
    round_num: int,
    past_matchups: set[frozenset[str]],
    team_regions: dict[str, str] | None = None,
    best_of: int = 1,
    rng: random.Random | None = None,
) -> list[SwissMatchup]:
    """Execute Riot's official ball-by-ball sequential draw lottery for a single Swiss pool.

    Parameters
    ----------
    teams : list[str]
        List of team names competing in this pool.
    pool_name : str
        Label representing the pool (e.g. '0-0', '1-0', '0-1', '2-1', etc.).
    round_num : int
        Current Swiss Stage round number (1 to 5).
    past_matchups : set[frozenset[str]]
        Set of pairs of teams that faced each other in earlier Swiss rounds.
    team_regions : dict[str, str] | None, optional
        Mapping from team name to regional affiliation (e.g. 'LCK', 'LPL', 'LEC', 'LCS').
        Enforced in Round 1 to prevent intra-regional pairings.
    best_of : int, optional
        Number of games per match series, by default 1.
    rng : random.Random | None, optional
        Deterministic random generator. If None, a new default instance is used.

    Returns
    -------
    list[SwissMatchup]
        Ordered list of drawn pairings with assigned match slots.

    Raises
    ------
    ValueError
        If the number of teams in the pool is not even.
    """
    if rng is None:
        rng = random.Random()

    if not teams:
        return []

    if len(teams) % 2 != 0:
        raise ValueError(f"Pool '{pool_name}' has an odd number of teams ({len(teams)}); cannot pair.")

    num_matches = len(teams) // 2
    all_slots = [(m, s) for m in range(num_matches) for s in (0, 1)]

    # Compute minimal waiver budget if zero-conflict matching is impossible
    empty_matches: list[list[str | None]] = [[None, None] for _ in range(num_matches)]
    budget = 0
    while budget <= num_matches:
        if _can_complete_matches(empty_matches, teams, round_num, past_matchups, team_regions, budget):
            break
        budget += 1

    matches: list[list[str | None]] = [[None, None] for _ in range(num_matches)]
    pot = list(teams)

    def _draw_recursive(rem_pot: list[str], current_budget: int) -> bool:
        if not rem_pot:
            return True

        # Target slot: earliest empty slot in standard sequence
        target_idx = next(i for i, (m, s) in enumerate(all_slots) if matches[m][s] is None)

        idx = rng.randrange(len(rem_pot))
        drawn_team = rem_pot.pop(idx)

        # Slide-slot mechanic: test candidate slots starting from target_idx
        for m, s in all_slots[target_idx:]:
            if matches[m][s] is not None:
                continue

            opp = matches[m][1 - s]
            if opp is not None:
                viol = 0 if _is_pairing_legal(drawn_team, opp, round_num, past_matchups, team_regions) else 1
            else:
                viol = 0

            if viol > current_budget:
                continue

            matches[m][s] = drawn_team
            if _can_complete_matches(matches, rem_pot, round_num, past_matchups, team_regions, current_budget):
                if _draw_recursive(rem_pot, current_budget):
                    return True

            matches[m][s] = None

        # Backtrack
        rem_pot.insert(idx, drawn_team)
        return False

    success = _draw_recursive(pot, budget)
    if not success:
        # Minimal rematch waiver relaxation if an impossible state is reached
        while not success and budget < num_matches:
            budget += 1
            success = _draw_recursive(pot, budget)

    result: list[SwissMatchup] = []
    for m in range(num_matches):
        team_a = matches[m][0]
        team_b = matches[m][1]
        assert team_a is not None and team_b is not None
        result.append(
            SwissMatchup(
                team_a=team_a,
                team_b=team_b,
                pool=pool_name,
                round_num=round_num,
                best_of=best_of,
            )
        )

    return result


def simulate_swiss_round_draws(
    active_records: dict[str, tuple[int, int]],
    past_matchups: set[frozenset[str]],
    round_num: int,
    team_regions: dict[str, str] | None = None,
    rng: random.Random | None = None,
) -> list[SwissMatchup]:
    """Simulate complete lottery draws across all active record pools for a Swiss round.

    Pools are drawn in standard descending record order (e.g. 1-0 before 0-1 in Round 2;
    2-0, 1-1, 0-2 in Round 3; 2-1, 1-2 in Round 4; 2-2 in Round 5).
    Series lengths follow Riot's official rules: Best-of-3 for qualification and elimination
    series (teams at 2 wins or 2 losses), Best-of-1 for all other series.

    Parameters
    ----------
    active_records : dict[str, tuple[int, int]]
        Mapping of active team names to their current (wins, losses) records.
        Teams that have already qualified (3 wins) or been eliminated (3 losses)
        are excluded if provided.
    past_matchups : set[frozenset[str]]
        Set of past head-to-head pairings played during earlier Swiss rounds.
    round_num : int
        Current Swiss Stage round number (1 to 5).
    team_regions : dict[str, str] | None, optional
        Regional metadata per team, enforcing intra-regional separation in Round 1.
    rng : random.Random | None, optional
        Seeded RNG for deterministic simulation. If None, a new instance is used.

    Returns
    -------
    list[SwissMatchup]
        Complete list of matchups across all active pools for this Swiss round.
    """
    if rng is None:
        rng = random.Random()

    # Filter out teams that have already finished (3 wins or 3 losses in 16-team Swiss)
    active = {
        team: record
        for team, record in active_records.items()
        if not (record[0] >= 3 or record[1] >= 3)
    }

    if not active:
        return []

    pools: dict[tuple[int, int], list[str]] = {}
    for team, record in active.items():
        pools.setdefault(record, []).append(team)

    # Sort pools: higher wins first, then fewer losses
    sorted_records = sorted(pools.keys(), key=lambda r: (-r[0], r[1]))
    all_matchups: list[SwissMatchup] = []

    for record in sorted_records:
        pool_teams = sorted(pools[record])
        pool_name = f"{record[0]}-{record[1]}"
        # Qualification (2 wins) or elimination (2 losses) deciders are Bo3; others Bo1
        best_of = 3 if (record[0] >= 2 or record[1] >= 2) else 1

        matchups = draw_swiss_pool_lottery(
            teams=pool_teams,
            pool_name=pool_name,
            round_num=round_num,
            past_matchups=past_matchups,
            team_regions=team_regions,
            best_of=best_of,
            rng=rng,
        )
        all_matchups.extend(matchups)

    return all_matchups
