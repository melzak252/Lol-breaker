"""Research-only, strictly prior-day relational features for supplied ten-player rosters.

Roster availability is NOT established here: callers may be using retrospective
first-map rosters. Residuals are descriptive historical associations, not causal
pair effects. No target outcome, draft, learned identity vocabulary, or target
roster ever updates history. Accepted historical maps are the only updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations, groupby
from math import exp, log, log1p

import numpy as np
import pandas as pd

TAU_DAYS = 180.0
PSEUDO_GAMES = 20.0
ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
ROLE_PAIRS = tuple(combinations(range(5), 2))
NODE_NAMES = [f"role_{role}" for role in ROLES] + [
    "winrate",
    "exposure",
    "log_prior_maps",
    "inactivity",
    "known",
]
EDGE_NAMES = [
    "teammate",
    "opponent",
    "exposure",
    "log_prior_maps",
    "inactivity",
    "known",
    "residual",
]
HISTORY_NAMES = EDGE_NAMES[2:]
LINEUP_NAMES = HISTORY_NAMES[:-1]
DIRECT_NAMES = (
    [f"player_{role}_{name}" for role in ROLES for name in NODE_NAMES[5:]]
    + [
        f"pair_{ROLES[i]}_{ROLES[j]}_{name}"
        for i, j in ROLE_PAIRS
        for name in HISTORY_NAMES
    ]
    + [
        f"opponent_{a}_{b}_{name}"
        for a in ROLES
        for b in ROLES
        for name in HISTORY_NAMES
    ]
    + [f"lineup_{name}" for name in LINEUP_NAMES]
    + ["player_coverage", "pair_coverage", "opponent_coverage", "newcomer_fraction"]
)
COVERAGE_NAMES = [
    "min_prior_maps",
    "min_pair_prior_maps",
    "min_prior_maps_a",
    "min_prior_maps_b",
    "min_pair_prior_maps_a",
    "min_pair_prior_maps_b",
    "prior_lineup_maps_a",
    "prior_lineup_maps_b",
    "player_coverage_a",
    "player_coverage_b",
    "pair_coverage_a",
    "pair_coverage_b",
    "opponent_coverage_a",
    "opponent_coverage_b",
    "newcomer_flag",
]


def history_integrity_checks():
    """Run tiny executable guards; no external data/model/files are required.

    Checks actual output tensors, not source text. Whole-side reversal preserves
    roles and permutes graph axes while negating every direct feature. Within a
    side each role occurs once, so other within-side permutations are *not*
    role-preserving: their historical channels are equivariant, while positional
    role one-hots must be reassigned. The latter distinction is checked too.
    """
    roster = [f"p{i}" for i in range(10)]
    changed = roster.copy()
    changed[0] = "substitute"
    events = [
        {"date": "2024-01-01", "game_id": "past", "players": roster, "y": 1},
        {"date": "2024-01-02", "game_id": "same-1", "players": changed, "y": 0},
        {"date": "2024-01-02", "game_id": "same-2", "players": roster, "y": 1},
        {"date": "2024-02-01", "game_id": "future", "players": roster, "y": 1},
    ]

    def target(day, players=roster):
        return pd.DataFrame(
            [{"date": day, "players": players, "golgg_match_id": "target"}]
        )

    def equal_features(a, b):
        for key in ("nodes", "edges", "direct"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        pd.testing.assert_frame_equal(a["coverage"], b["coverage"])

    original = build_graph_history(target("2024-01-02"), events)
    mutated = [dict(event) for event in events]
    for event in mutated[1:]:
        event["y"] = 1 - event["y"]
        event["players"] = [f"new-{event['game_id']}-{i}" for i in range(10)]
    equal_features(original, build_graph_history(target("2024-01-02"), mutated))
    equal_features(original, build_graph_history(target("2024-01-02"), events[:1]))
    equal_features(
        original, build_graph_history(target("2024-01-02"), list(reversed(events)))
    )

    later = build_graph_history(target("2024-01-03"), events)
    past_mutation = [dict(event) for event in events]
    past_mutation[0]["y"] = 0
    changed_later = build_graph_history(target("2024-01-03"), past_mutation)
    assert not np.array_equal(later["nodes"], changed_later["nodes"])
    assert not np.array_equal(later["edges"], changed_later["edges"])
    assert later["coverage"].loc[0, "min_prior_maps"] == 2

    # Same-day expectations are frozen, even when preceding map outcomes change.
    other_first = [dict(event) for event in events]
    other_first[1]["y"] = 1
    other = build_graph_history(target("2024-01-03"), other_first)
    # p0 was absent from same-1: its pairs' residual expectations cannot change.
    np.testing.assert_array_equal(later["edges"][0, 0, 1:5], other["edges"][0, 0, 1:5])

    swap = np.r_[5:10, 0:5]
    swapped = build_graph_history(
        target("2024-01-03", [roster[i] for i in swap]), events
    )
    np.testing.assert_array_equal(swapped["nodes"], later["nodes"][:, swap])
    np.testing.assert_array_equal(swapped["edges"], later["edges"][:, swap][:, :, swap])
    np.testing.assert_array_equal(swapped["direct"], -later["direct"])
    # Reversing every historical event orientation leaves historical state intact.
    reversed_history = [
        dict(e, players=e["players"][5:] + e["players"][:5], y=1 - e["y"])
        for e in events
    ]
    reverse_result = build_graph_history(target("2024-01-03"), reversed_history)
    for key in ("nodes", "edges", "direct"):
        np.testing.assert_allclose(reverse_result[key], later[key], rtol=0, atol=1e-7)

    perm = np.array([2, 0, 4, 1, 3, 7, 5, 9, 6, 8])
    permuted = build_graph_history(
        target("2024-01-03", [roster[i] for i in perm]), events
    )
    np.testing.assert_array_equal(
        permuted["nodes"][:, :, 5:], later["nodes"][:, perm, 5:]
    )
    np.testing.assert_array_equal(permuted["nodes"][:, :, :5], later["nodes"][:, :, :5])
    np.testing.assert_array_equal(
        permuted["edges"], later["edges"][:, perm][:, :, perm]
    )
    np.testing.assert_array_equal(permuted["direct"][:, -8:], later["direct"][:, -8:])

    # Targets do not become events, including a previous-day hypothetical roster.
    extra_targets = pd.concat(
        [target("2024-01-01", changed), target("2024-01-03")], ignore_index=True
    )
    extra = build_graph_history(extra_targets, events)
    for key in ("nodes", "edges", "direct"):
        np.testing.assert_array_equal(extra[key][1:], later[key])
    unseen = build_graph_history(target("2023-01-01"), events)
    assert unseen["coverage"].loc[0, "newcomer_flag"]
    assert unseen["coverage"].loc[0, "min_prior_maps"] == 0
    np.testing.assert_array_equal(
        unseen["nodes"][0, :, 5], np.full(10, 0.5, dtype=np.float32)
    )
    assert not unseen["nodes"][0, :, 9].any()
    assert not unseen["edges"][0, :, :, 5].any()
    for result in (original, later, swapped, unseen):
        for key in ("nodes", "edges", "direct"):
            assert result[key].dtype == np.float32 and np.isfinite(result[key]).all()
        assert not result["edges"][0, np.arange(10), np.arange(10)].any()
    return {"passed": True, "tau_days": TAU_DAYS, "pseudo_games": PSEUDO_GAMES}


@dataclass(slots=True)
class _State:
    mass: float = 0.0
    total: float = 0.0
    maps: int = 0
    last: int = 0

    def decayed(self, day):
        factor = exp(-(day - self.last) / TAU_DAYS)
        return self.mass * factor, self.total * factor

    def update(self, day, value):
        self.mass, self.total = self.decayed(day)
        self.mass += 1.0
        self.total += value
        self.maps += 1
        self.last = day


def _date(value):
    if isinstance(value, datetime):
        if value.time().isoformat() != "00:00:00" or value.tzinfo is not None:
            raise ValueError(f"expected timezone-free calendar date, got {value!r}")
        value = value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError(f"expected ISO calendar date, got {value!r}")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"expected YYYY-MM-DD date, got {value!r}")
    return parsed


def _roster(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 10:
        raise ValueError("players must contain exactly ten role-ordered identities")
    if any(not isinstance(p, str) or not p.strip() for p in value):
        raise ValueError(
            "player identities must be nonempty strings; missing identities cannot be imputed"
        )
    result = tuple(value)
    if len(set(result)) != 10:
        raise ValueError(
            "ten distinct player identities are required for each historical/target roster"
        )
    return result


def _key(a, b):
    return (a, b) if a < b else (b, a)


def _history(state, day):
    """Bounded exposure, raw-map log scale, inactivity, known, shrunk total."""
    if state is None:
        return (0.0, 0.0, 1.0, 0.0, 0.0)
    mass, total = state.decayed(day)
    age = day - state.last
    logged = log1p(state.maps)
    return (
        mass / (mass + PSEUDO_GAMES),
        logged / (1.0 + logged),
        age / (age + TAU_DAYS),
        1.0,
        total / (mass + PSEUDO_GAMES),
    )


def _ability(state, day):
    if state is None:
        return 0.5
    mass, wins = state.decayed(day)
    return (wins + 0.5 * PSEUDO_GAMES) / (mass + PSEUDO_GAMES)


def _probability(log_odds):
    return 1.0 / (1.0 + exp(-log_odds))


def _update(table, key, day, value):
    state = table.get(key)
    if state is None:
        state = table[key] = _State(last=day)
    state.update(day, value)


def build_graph_history(targets, events):
    """Return prior-day graph tensors and odd side-difference control features.

    ``targets`` must have a reset RangeIndex and columns date, players (TOP,
    JUNGLE,MID,ADC,SUPPORT for A then B), golgg_match_id. ``events`` are accepted
    map dictionaries with date, unique game_id, players and binary winner-A y.
    Every supplied event is validated, including future events; none is silently
    discarded or repaired. Explicitly unseen entities use zero exposure, known=0,
    inactivity=1, winrate=.5 and residual=0, rather than hidden missing imputation.

    State uses exponential decay exp(-days/180), shrinkage of 20 pseudo-games.
    Individual winrates have a .5 prior. Pair/opponent residual means have a zero
    prior. Pair expected wins use mean member log-odds minus opposing-team mean
    individual log-odds; directed opponent expected wins use individual log-odds
    differences. ALL such expectations use state frozen before that day's maps.
    Raw map counts and last-seen dates are retained alongside decayed counts.
    Lineups are unordered five-player sets; their exposure is outcome-free.

    Keys: nodes float32[N,10,10]; edges float32[N,10,10,7], diagonal zero;
    direct float32[N,D]; direct_names/node_names/edge_names lists; coverage DataFrame
    length N (raw prior-map counts/known fractions/newcomer flag); players Unicode
    [N,10]; dates Unicode[N]. The first two edge channels are teammate/opponent
    masks. Other channels share scales across those relation types. Direct
    features cover every node history channel, all ten teammate role-pairs, all
    25 opponent role-pairs, lineup exposure and missing coverage as A-minus-B.
    These are summaries of the same history, not lossless graph encodings.
    Node role one-hots encode supplied positions, never an inferred player role.
    """
    required = {"date", "players", "golgg_match_id"}
    if not isinstance(targets, pd.DataFrame) or not required.issubset(targets.columns):
        raise ValueError(f"targets must be a DataFrame containing {sorted(required)}")
    if not isinstance(targets.index, pd.RangeIndex) or not targets.index.equals(
        pd.RangeIndex(len(targets))
    ):
        raise ValueError("targets must have a reset RangeIndex")
    target_rows = []
    for row in targets.itertuples(index=False):
        if pd.isna(row.golgg_match_id) or not str(row.golgg_match_id).strip():
            raise ValueError("every target requires golgg_match_id")
        target_rows.append((_date(row.date), _roster(row.players)))
    history = []
    seen = set()
    for event in events:
        if not isinstance(event, dict) or not {
            "date",
            "game_id",
            "players",
            "y",
        }.issubset(event):
            raise ValueError("every event requires date, game_id, players and y")
        game_id = event["game_id"]
        if not isinstance(game_id, str) or not game_id.strip() or game_id in seen:
            raise ValueError(
                f"historical game_id must be a unique nonempty string: {game_id!r}"
            )
        if not isinstance(event["y"], (int, np.integer)) or event["y"] not in (0, 1):
            raise ValueError(f"event {game_id}: y must be integer 0 or 1")
        seen.add(game_id)
        history.append(
            (
                _date(event["date"]).toordinal(),
                game_id,
                _roster(event["players"]),
                int(event["y"]),
            )
        )
    history.sort(key=lambda row: (row[0], row[1]))
    n = len(target_rows)
    nodes = np.zeros((n, 10, len(NODE_NAMES)), dtype=np.float32)
    edges = np.zeros((n, 10, 10, len(EDGE_NAMES)), dtype=np.float32)
    direct = np.zeros((n, len(DIRECT_NAMES)), dtype=np.float32)
    coverage = [None] * n
    players, pairs, opponents, lineups = {}, {}, {}, {}
    position = 0

    def update_day(period, day):
        # Collect only this historical date's players, not a future vocabulary.
        identities = {p for event in period for p in event[2]}
        abilities = {p: _ability(players.get(p), day) for p in identities}
        odds = {p: log(v / (1.0 - v)) for p, v in abilities.items()}
        for _, _, roster, y in period:
            a, b = roster[:5], roster[5:]
            for side, rivals, outcome in ((a, b, y), (b, a, 1 - y)):
                rival_odds = sum(odds[p] for p in rivals) / 5.0
                for p in side:
                    _update(players, p, day, outcome)
                for p, q in combinations(side, 2):
                    expected = _probability((odds[p] + odds[q]) / 2.0 - rival_odds)
                    _update(pairs, _key(p, q), day, outcome - expected)
                _update(lineups, tuple(sorted(side)), day, 0.0)
            for p in a:
                for q in b:
                    residual = y - _probability(odds[p] - odds[q])
                    _update(opponents, (p, q), day, residual)
                    _update(opponents, (q, p), day, -residual)

    for day, grouped in groupby(
        sorted(range(n), key=lambda i: target_rows[i][0]),
        key=lambda i: target_rows[i][0],
    ):
        ordinal = day.toordinal()
        while position < len(history) and history[position][0] < ordinal:
            start = position
            event_day = history[position][0]
            while position < len(history) and history[position][0] == event_day:
                position += 1
            update_day(history[start:position], event_day)
        # Pure snapshots: no state changes depend on number or order of targets.
        for idx in grouped:
            roster = target_rows[idx][1]
            for i, p in enumerate(roster):
                state = players.get(p)
                values = _history(state, ordinal)
                nodes[idx, i, i % 5] = 1.0
                nodes[idx, i, 5:] = (_ability(state, ordinal), *values[:4])
                for j, q in enumerate(roster):
                    if i == j:
                        continue
                    teammate = i // 5 == j // 5
                    relation = (
                        pairs.get(_key(p, q)) if teammate else opponents.get((p, q))
                    )
                    edges[idx, i, j] = (
                        float(teammate),
                        float(not teammate),
                        *_history(relation, ordinal),
                    )
            side_rows = []
            for offset in (0, 5):
                own = roster[offset : offset + 5]
                opposite = 5 - offset
                player_states = [players.get(p) for p in own]
                pair_states = [pairs.get(_key(own[i], own[j])) for i, j in ROLE_PAIRS]
                lineup_state = lineups.get(tuple(sorted(own)))
                player_known = sum(s is not None for s in player_states) / 5.0
                pair_known = sum(s is not None for s in pair_states) / 10.0
                opponent_known = float(
                    edges[idx, offset : offset + 5, opposite : opposite + 5, 5].mean()
                )
                # Flatten pair vectors without copying the full tensor.
                flattened = list(nodes[idx, offset : offset + 5, 5:].ravel())
                for i, j in ROLE_PAIRS:
                    flattened.extend(edges[idx, offset + i, offset + j, 2:])
                for i in range(5):
                    for j in range(5):
                        flattened.extend(edges[idx, offset + i, opposite + j, 2:])
                flattened.extend(_history(lineup_state, ordinal)[:4])
                flattened.extend(
                    (player_known, pair_known, opponent_known, 1.0 - player_known)
                )
                side_rows.append(
                    (
                        np.asarray(flattened, dtype=np.float32),
                        {
                            "min_prior_maps": min(
                                s.maps if s else 0 for s in player_states
                            ),
                            "min_pair_prior_maps": min(
                                s.maps if s else 0 for s in pair_states
                            ),
                            "prior_lineup_maps": lineup_state.maps
                            if lineup_state
                            else 0,
                            "player_coverage": player_known,
                            "pair_coverage": pair_known,
                            "opponent_coverage": opponent_known,
                        },
                    )
                )
            direct[idx] = side_rows[0][0] - side_rows[1][0]
            a, b = side_rows[0][1], side_rows[1][1]
            coverage[idx] = {
                "min_prior_maps": min(a["min_prior_maps"], b["min_prior_maps"]),
                "min_pair_prior_maps": min(
                    a["min_pair_prior_maps"], b["min_pair_prior_maps"]
                ),
                **{f"{key}_a": value for key, value in a.items()},
                **{f"{key}_b": value for key, value in b.items()},
                "newcomer_flag": a["min_prior_maps"] == 0 or b["min_prior_maps"] == 0,
            }
    for name, tensor in (("nodes", nodes), ("edges", edges), ("direct", direct)):
        if not np.isfinite(tensor).all():
            raise ValueError(
                f"non-finite {name}: invalid historical state; no values imputed"
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "direct": direct,
        "direct_names": list(DIRECT_NAMES),
        "node_names": list(NODE_NAMES),
        "edge_names": list(EDGE_NAMES),
        "coverage": pd.DataFrame(coverage, columns=COVERAGE_NAMES),
        "players": np.asarray([row[1] for row in target_rows], dtype=str).reshape(
            n, 10
        ),
        "dates": np.asarray([row[0].isoformat() for row in target_rows], dtype="U10"),
    }
