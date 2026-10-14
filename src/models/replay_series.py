"""Independent EXP-083 replay: date-only evidence becomes usable next date.

No timezone or within-day order is invented. Stable source IDs, including academy
IDs, are kept distinct. Glicko uses daily periods: ``gap - 1`` wholly inactive
calendar days inflate RD and the active update supplies the final period. States
and opponent priors are frozen for the entire date, including repeated teams.

Player strength is the mean of five independent marginal ratings; its RD is
sqrt(sum(RD**2))/5. This is NOT a joint team/player posterior: teammates' correlated
errors and causal contributions are unidentified. Actual complete map lineups
receive updates against the opposing actual lineup's frozen prior; forecast
lineups never receive an absent player's outcome. A missing forecast lineup has
an explicit single aggregate N(1500,350**2) structural prior, not dummy players.

History is the last up-to-20 maps (short window: five). If its boundary cuts an
unordered date, every map on that date receives the same fractional weight.
Optional statistics are averaged only over observed values. Their difference is
neutral when either side has no measurements; odd/even missing-fraction features
encode BOTH side masks (A = c + d/2, B = c - d/2). Missing history has win prior
0.5, residual prior 0 and explicit history/roster masks, never measured zeros.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.models.competition_tiers import classify_competition
from src.ratings.glicko2_core import (
    Glicko2Observation,
    Glicko2State,
    expected_score,
    inflate,
    update,
)

from src.utils.golgg_schema import _normalize_team_name

_STATS = ("kills", "gold", "towers", "nashors")
_PAIRED = (
    "team_rd",
    "team_rest_log1p",
    "team_series_log1p",
    "history_games",
    "history_missing",
    "roster_known",
    "roster_ambiguous",
    "roster_age_log1p",
    "player_rd",
    "player_rest_log1p",
    "player_series_log1p",
    "map_win_w20",
    "residual_w20",
    "map_win_short_long",
    "residual_short_long",
)
ODD_FEATURES = (
    "d_team_glicko_logit",
    "d_player_glicko_logit",
    *(f"d_{name}" for name in _PAIRED),
    *(
        f"d_{stat}_{suffix}"
        for stat in _STATS
        for suffix in ("w20", "short_long", "missing", "short_missing")
    ),
    "d_team_glicko_bo3",
    "d_team_glicko_bo5",
    "d_player_glicko_bo3",
    "d_player_glicko_bo5",
)
EVEN_FEATURES = (
    *(f"c_{name}" for name in _PAIRED),
    *(
        f"c_{stat}_{suffix}"
        for stat in _STATS
        for suffix in ("missing", "short_missing")
    ),
    "c_bo3",
    "c_bo5",
)
_METADATA = (
    "golgg_match_id",
    "date",
    "team1_id",
    "team2_id",
    "team1_name",
    "team2_name",
    "best_of",
    "y_true",
    "tournament",
    "competition_tier",
    "roster_min_prior_series",
    "feature_history_max_date",
    "score_a",
    "score_b",
    "map_prob_team",
    "map_prob_player",
)


@dataclass(slots=True)
class _Map:
    game_id: str
    a: str
    b: str
    score: int
    stats_a: tuple[float | None, ...]
    stats_b: tuple[float | None, ...]
    roster_a: tuple[str, ...] | None
    roster_b: tuple[str, ...] | None


@dataclass(slots=True)
class _Series:
    match_id: str
    day: date
    a: str
    b: str
    name_a: str
    name_b: str
    tournament: str
    best_of: int
    score_a: int
    score_b: int
    maps: list[_Map]


@dataclass(slots=True)
class _Entity:
    state: Glicko2State = field(default_factory=Glicko2State)
    last: date | None = None
    series: int = 0


@dataclass(slots=True)
class _Team(_Entity):
    roster: tuple[str, ...] | None = None
    roster_date: date | None = None
    ambiguous: bool = False
    # Each day stores outcome, frozen residual, and optional measurements.
    history: deque = field(default_factory=deque)


def _id(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() in {"nan", "none", "null"}
    ):
        return None
    return value.strip()


def _integer(value: Any) -> int | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _date(value: Any) -> date | None:
    try:
        if isinstance(value, (int, float, bool, np.number)):
            return None
        parsed = pd.Timestamp(value)
        return None if pd.isna(parsed) else parsed.date()
    except (ValueError, TypeError, OverflowError):
        return None


def _stats(value: Any, counters: Counter) -> tuple[float | None, ...]:
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        payload = None
    if not isinstance(payload, dict):
        payload = {}
    result = []
    for name in _STATS:
        raw = payload.get(name)
        try:
            number = (
                float(raw)
                if raw is not None and not isinstance(raw, bool)
                else math.nan
            )
        except (ValueError, TypeError, OverflowError):
            number = math.nan
        if not math.isfinite(number) or number < 0:
            counters[f"missing_stat_{name}_map_sides"] += 1
            result.append(None)
        else:
            result.append(number / 1000.0 if name == "gold" else number)
    return tuple(result)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _recover_identity(
    row: dict, source_maps: list[dict], a: str | None, b: str | None
) -> tuple[str | None, str | None]:
    """Use exact normalized names on every map, never score or map position."""
    raw_names = [row[f"team{side}_name"] for side in (1, 2)]
    if not all(isinstance(name, str) for name in raw_names):
        return a, b
    names = [_normalize_team_name(name) for name in raw_names]
    if not all(names) or names[0] == names[1] or not source_maps:
        return a, b
    inferred: list[set[str]] = [set(), set()]
    for game in source_maps:
        mapping: dict[str, set[str]] = defaultdict(set)
        for side in (1, 2):
            raw_name = game.get(f"team{side}_name")
            name = _normalize_team_name(raw_name) if isinstance(raw_name, str) else ""
            tid = _id(game.get(f"team{side}_id"))
            if name and tid:
                mapping[name].add(tid)
        for position, name in enumerate(names):
            if len(mapping.get(name, ())) != 1:
                return a, b
            inferred[position].update(mapping[name])
    if any(len(ids) != 1 for ids in inferred):
        return a, b
    recovered_a, recovered_b = (next(iter(ids)) for ids in inferred)
    if (
        recovered_a == recovered_b
        or (a is not None and a != recovered_a)
        or (b is not None and b != recovered_b)
    ):
        return a, b
    return recovered_a, recovered_b


def _prepare(
    matches: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame, audit: dict
) -> list[_Series]:
    _require(
        matches,
        (
            "match_id",
            "date",
            "tournament_name",
            "team1_id",
            "team2_id",
            "team1_name",
            "team2_name",
            "team1_score",
            "team2_score",
            "team1_win",
            "team2_win",
            "best_of",
        ),
        "matches",
    )
    _require(
        games,
        (
            "game_id",
            "match_id",
            "date",
            "team1_id",
            "team2_id",
            "team1_win",
            "team2_win",
        ),
        "games",
    )
    _require(players, ("game_id", "match_id", "team_id", "player_id"), "players")
    counts = audit["counts"]
    match_records = matches.to_dict("records")
    match_ids = Counter(_id(row["match_id"]) for row in match_records)
    game_records = games.to_dict("records")
    game_ids = Counter(_id(row["game_id"]) for row in game_records)
    maps_by_match: dict[str, list[dict]] = defaultdict(list)
    for row in game_records:
        mid = _id(row["match_id"])
        if mid is None or mid not in match_ids:
            counts["orphan_game_rows"] += 1
        else:
            maps_by_match[mid].append(row)
    participants: dict[str, list[tuple]] = defaultdict(list)
    for gid, mid, team, player in players[
        ["game_id", "match_id", "team_id", "player_id"]
    ].itertuples(index=False, name=None):
        gid = _id(gid)
        if gid is None or gid not in game_ids:
            counts["orphan_player_rows"] += 1
        else:
            participants[gid].append((_id(mid), _id(team), _id(player)))
    valid = []
    for row in match_records:
        mid, a, b = (_id(row[key]) for key in ("match_id", "team1_id", "team2_id"))
        source_maps = maps_by_match.get(mid, [])
        if a is None or b is None:
            a, b = _recover_identity(row, source_maps, a, b)
            if a is not None and b is not None:
                counts["recovered_series_identity"] += 1
        day, bo = _date(row["date"]), _integer(row["best_of"])
        sa, sb, wa, wb = (
            _integer(row[key])
            for key in ("team1_score", "team2_score", "team1_win", "team2_win")
        )
        reason = None
        if mid is None or a is None or b is None or a == b:
            reason = "invalid_match_identity"
        elif match_ids[mid] != 1:
            reason = "duplicate_match_id"
        elif day is None:
            reason = "invalid_match_date"
        elif bo not in (1, 3, 5):
            reason = "unsupported_best_of"
        elif (
            sa is None
            or sb is None
            or min(sa, sb) < 0
            or max(sa, sb) != bo // 2 + 1
            or min(sa, sb) >= bo // 2 + 1
        ):
            reason = "nonterminal_score"
        elif (wa, wb) != (int(sa > sb), int(sb > sa)):
            reason = "contradictory_match_winner"
        if reason is None and len(source_maps) != sa + sb:
            reason = "game_count_mismatch"
        aligned = []
        if reason is None:
            for game in sorted(source_maps, key=lambda item: str(item["game_id"])):
                gid, ga, gb = (
                    _id(game[key]) for key in ("game_id", "team1_id", "team2_id")
                )
                gwin, gloss = _integer(game["team1_win"]), _integer(game["team2_win"])
                if gid is None or game_ids[gid] != 1:
                    reason = "invalid_or_duplicate_game_id"
                elif ga is None or gb is None or ga == gb or {ga, gb} != {a, b}:
                    reason = "contradictory_game_identity"
                elif _date(game["date"]) != day:
                    reason = "game_date_mismatch"
                elif (gwin, gloss) not in ((0, 1), (1, 0)):
                    reason = "invalid_game_winner"
                if reason is not None:
                    break
                aligned.append((game, gid, ga, gwin if ga == a else gloss))
            if reason is None and sum(item[3] for item in aligned) != sa:
                reason = "game_score_mismatch"
        if reason is not None:
            audit["exclusions"][reason] += 1
            audit["excluded_series"].append({"match_id": mid, "reason": reason})
            continue
        parsed_maps = []
        for game, gid, ga, outcome in aligned:
            by_team: dict[str, list[str]] = defaultdict(list)
            bad_identity = False
            for pmid, team, pid in participants.get(gid, ()):
                if pmid != mid or team not in (a, b) or pid is None:
                    bad_identity = True
                    counts["contradictory_player_rows"] += 1
                else:
                    by_team[team].append(pid)
            rosters = []
            shared = bool(set(by_team[a]) & set(by_team[b]))
            if shared:
                counts["cross_side_player_maps"] += 1
            for team in (a, b):
                ids = by_team[team]
                if bad_identity or shared or len(ids) != 5 or len(set(ids)) != 5:
                    rosters.append(None)
                    counts["incomplete_or_invalid_roster_map_sides"] += 1
                else:
                    rosters.append(tuple(sorted(ids)))
            left = "team1_stats_json" if ga == a else "team2_stats_json"
            right = "team2_stats_json" if ga == a else "team1_stats_json"
            parsed_maps.append(
                _Map(
                    gid,
                    a,
                    b,
                    outcome,
                    _stats(game.get(left), counts),
                    _stats(game.get(right), counts),
                    rosters[0],
                    rosters[1],
                )
            )
        valid.append(
            _Series(
                mid,
                day,
                a,
                b,
                str(row["team1_name"]),
                str(row["team2_name"]),
                str(row["tournament_name"]),
                bo,
                sa,
                sb,
                parsed_maps,
            )
        )
    return sorted(valid, key=lambda item: (item.day, item.match_id))


def _prior(entity: _Entity, day: date) -> Glicko2State:
    return (
        entity.state
        if entity.last is None
        else inflate(entity.state, max(0, (day - entity.last).days - 1))
    )


def _aggregate(states: list[Glicko2State]) -> Glicko2State:
    if not states:
        return Glicko2State()
    n = len(states)
    return Glicko2State(
        math.fsum(s.rating for s in states) / n,
        math.sqrt(math.fsum(s.rd * s.rd for s in states)) / n,
        math.fsum(s.volatility for s in states) / n,
    )


def _window(history: deque, size: int) -> tuple[list[float | None], list[float], float]:
    sums = [[] for _ in range(6)]
    weights = [0.0] * 6
    remaining = float(size)
    for _, observations in reversed(history):
        weight = min(1.0, remaining / len(observations))
        for observation in observations:
            for j, value in enumerate(observation):
                if value is not None:
                    sums[j].append(weight * value)
                    weights[j] += weight
        remaining -= weight * len(observations)
        if remaining <= 1e-12:
            break
    total = size - remaining
    means = [
        math.fsum(parts) / weight if weight else None
        for parts, weight in zip(sums, weights)
    ]
    missing = [1.0 - weight / total if total else 1.0 for weight in weights]
    return means, missing, total


def _side(
    team: _Team,
    prior: Glicko2State,
    day: date,
    players: dict[str, _Entity],
    player_prior,
) -> tuple[dict, Glicko2State, int, list[date]]:
    roster = team.roster or ()
    members = [players[pid] for pid in roster]
    aggregate = _aggregate([player_prior(pid) for pid in roster])
    long, missing, total = _window(team.history, 20)
    short, short_missing, _ = _window(team.history, 5)
    win, residual = (long[0] if total else 0.5), (long[1] if total else 0.0)
    side = dict(
        team_rd=prior.rd / 350.0,
        team_rest_log1p=math.log1p((day - team.last).days) if team.last else 0.0,
        team_series_log1p=math.log1p(team.series),
        history_games=total,
        history_missing=float(not total),
        roster_known=float(bool(roster)),
        roster_ambiguous=float(team.ambiguous),
        roster_age_log1p=(
            math.log1p((day - team.roster_date).days) if team.roster_date else 0.0
        ),
        player_rd=aggregate.rd / 350.0,
        player_rest_log1p=(
            math.fsum(
                math.log1p((day - p.last).days) if p.last else 0.0 for p in members
            )
            / len(members)
            if members
            else 0.0
        ),
        player_series_log1p=(
            math.fsum(math.log1p(p.series) for p in members) / len(members)
            if members
            else 0.0
        ),
        map_win_w20=win,
        residual_w20=residual,
        map_win_short_long=short[0] - win if total else 0.0,
        residual_short_long=short[1] - residual if total else 0.0,
    )
    for j, stat in enumerate(_STATS, 2):
        side[f"{stat}_w20"] = long[j]
        side[f"{stat}_short_long"] = (
            short[j] - long[j] if short[j] is not None and long[j] is not None else None
        )
        side[f"{stat}_missing"] = missing[j]
        side[f"{stat}_short_missing"] = short_missing[j]
    dates = [p.last for p in members if p.last is not None]
    if team.last is not None:
        dates.append(team.last)
    return side, aggregate, min((p.series for p in members), default=0), dates


def _probability(a: Glicko2State, b: Glicko2State) -> float:
    return expected_score(a.rating, a.rd, b.rating, b.rd)


def build_replay_features(
    matches: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Return sorted eligible series snapshots and a JSON-serializable audit.

    Invalid series never update history. Invalid/incomplete map lineups do not
    exclude otherwise valid team predictions; both actual lineups must be valid
    for player updates. A day with any unknown or differing lineup retains the
    previous unambiguous lineup and sets ambiguity. Terminal scores and aligned
    map totals are checked, but no map chronology is inferred from numeric IDs;
    the input has no independently verified within-series order/forfeit status.
    """
    audit = dict(
        schema_version="exp083-day-frozen-v1",
        input_matches=len(matches),
        input_games=len(games),
        input_players=len(players),
        counts=Counter(),
        exclusions=Counter(),
        excluded_series=[],
        odd_features=list(ODD_FEATURES),
        even_features=list(EVEN_FEATURES),
        policies={
            "availability": "strictly earlier source calendar date; timezone unspecified",
            "glicko": "default pure primitives; one-day periods; frozen opponents; gap-1 inactive days",
            "player_model": "independent marginal mean and sqrt(sum(rd^2))/5; not a joint posterior",
            "roster": "last prior-date unambiguous five IDs; retain old roster on ambiguous days",
            "history": "W20/W5; fractional equal weighting of unordered boundary-date maps",
            "optional_statistics": "observed-only means; neutral unsupported difference; side masks encoded by d difference/c mean",
            "cold_start": "Glicko 1500/350/.06; unknown roster aggregate same prior; win=.5 residual=0 and explicit masks",
            "terminal_validation": "planned BO1/3/5, terminal score and map identity/count/win agreement; map order and forfeit status unavailable",
        },
    )
    series = _prepare(matches, games, players, audit)
    teams: dict[str, _Team] = defaultdict(_Team)
    people: dict[str, _Entity] = defaultdict(_Entity)
    days: dict[date, list[_Series]] = defaultdict(list)
    for item in series:
        days[item.day].append(item)
    rows = []
    for day, fixtures in days.items():
        team_priors: dict[str, Glicko2State] = {}
        player_priors: dict[str, Glicko2State] = {}

        def team_prior(tid):
            if tid not in team_priors:
                team_priors[tid] = _prior(teams[tid], day)
            return team_priors[tid]

        def player_prior(pid):
            if pid not in player_priors:
                player_priors[pid] = _prior(people[pid], day)
            return player_priors[pid]

        for item in fixtures:
            a, b = team_prior(item.a), team_prior(item.b)
            left, pa, ea, da = _side(teams[item.a], a, day, people, player_prior)
            right, pb, eb, db = _side(teams[item.b], b, day, people, player_prior)
            pt, pp = _probability(a, b), _probability(pa, pb)
            row = dict(
                golgg_match_id=item.match_id,
                date=day.isoformat(),
                team1_id=item.a,
                team2_id=item.b,
                team1_name=item.name_a,
                team2_name=item.name_b,
                best_of=item.best_of,
                y_true=int(item.score_a > item.score_b),
                tournament=item.tournament,
                competition_tier=classify_competition(item.tournament, day).tier.value,
                roster_min_prior_series=min(ea, eb),
                feature_history_max_date=max(da + db).isoformat() if da or db else None,
                score_a=item.score_a,
                score_b=item.score_b,
                map_prob_team=pt,
                map_prob_player=pp,
                d_team_glicko_logit=math.log(pt) - math.log1p(-pt),
                d_player_glicko_logit=math.log(pp) - math.log1p(-pp),
            )
            for name in _PAIRED:
                row[f"d_{name}"] = left[name] - right[name]
                row[f"c_{name}"] = (left[name] + right[name]) / 2.0
            for stat in _STATS:
                for suffix in ("w20", "short_long", "missing", "short_missing"):
                    name = f"{stat}_{suffix}"
                    lvalue, rvalue = left[name], right[name]
                    row[f"d_{name}"] = (
                        lvalue - rvalue
                        if lvalue is not None and rvalue is not None
                        else 0.0
                    )
                    if suffix in ("missing", "short_missing"):
                        row[f"c_{name}"] = (lvalue + rvalue) / 2.0
            for bo in (3, 5):
                row[f"c_bo{bo}"] = float(item.best_of == bo)
                for kind in ("team", "player"):
                    row[f"d_{kind}_glicko_bo{bo}"] = (
                        row[f"d_{kind}_glicko_logit"] * row[f"c_bo{bo}"]
                    )
            rows.append(row)
        team_observations: dict[str, list] = defaultdict(list)
        player_observations: dict[str, list] = defaultdict(list)
        team_series: dict[str, set] = defaultdict(set)
        player_series: dict[str, set] = defaultdict(set)
        histories: dict[str, list] = defaultdict(list)
        rosters: dict[str, set] = defaultdict(set)
        for item in fixtures:
            for game in item.maps:
                a, b = team_prior(game.a), team_prior(game.b)
                probability = _probability(a, b)
                for tid, opponent, outcome, stats, roster in (
                    (game.a, b, game.score, game.stats_a, game.roster_a),
                    (game.b, a, 1 - game.score, game.stats_b, game.roster_b),
                ):
                    team_observations[tid].append(
                        Glicko2Observation(opponent.rating, opponent.rd, outcome)
                    )
                    team_series[tid].add(item.match_id)
                    expected = probability if tid == game.a else 1 - probability
                    histories[tid].append((float(outcome), outcome - expected, *stats))
                    rosters[tid].add(roster)
                    # Track trustworthy actual appearances even if the opponent
                    # roster is missing: appearance is known, rating evidence is not.
                    if roster is not None:
                        for pid in roster:
                            player_prior(pid)
                            player_series[pid].add(item.match_id)
                if game.roster_a is None or game.roster_b is None:
                    audit["counts"]["maps_without_player_update"] += 1
                    continue
                pa = _aggregate([player_prior(pid) for pid in game.roster_a])
                pb = _aggregate([player_prior(pid) for pid in game.roster_b])
                for roster, opponent, outcome in (
                    (game.roster_a, pb, game.score),
                    (game.roster_b, pa, 1 - game.score),
                ):
                    for pid in roster:
                        player_observations[pid].append(
                            Glicko2Observation(opponent.rating, opponent.rd, outcome)
                        )
                audit["counts"]["maps_with_player_update"] += 1
        for tid in sorted(team_observations):
            team = teams[tid]
            team.state = update(team_priors[tid], team_observations[tid])
            team.last = day
            team.series += len(team_series[tid])
            # Stable numeric aggregation, never an invented within-day order.
            observations = sorted(
                histories[tid],
                key=lambda values: tuple(-math.inf if v is None else v for v in values),
            )
            team.history.append((day, observations))
            while (
                len(team.history) > 1
                and sum(len(values) for _, values in team.history)
                - len(team.history[0][1])
                >= 20
            ):
                team.history.popleft()
            candidates = rosters[tid]
            if len(candidates) == 1 and None not in candidates:
                team.roster = next(iter(candidates))
                team.roster_date = day
                team.ambiguous = False
            else:
                team.ambiguous = True
                audit["counts"]["ambiguous_roster_team_dates"] += 1
        for pid in sorted(player_series):
            person = people[pid]
            observations = player_observations.get(pid, [])
            # Without usable rating evidence retain the elapsed uncertainty,
            # including this active day, rather than silently resetting its clock.
            person.state = (
                update(player_priors[pid], observations)
                if observations
                else inflate(player_priors[pid], 1)
            )
            person.last = day
            person.series += len(player_series[pid])
    frame = pd.DataFrame(rows, columns=(*_METADATA, *ODD_FEATURES, *EVEN_FEATURES))
    if not np.isfinite(
        frame[[*ODD_FEATURES, *EVEN_FEATURES]].to_numpy(dtype=float)
    ).all():
        raise ArithmeticError("replay generated non-finite predictors")
    audit["emitted_series"] = len(frame)
    audit["excluded_series_count"] = len(matches) - len(frame)
    audit["counts"] = dict(sorted(audit["counts"].items()))
    audit["exclusions"] = dict(sorted(audit["exclusions"].items()))
    audit["excluded_series"].sort(
        key=lambda item: (str(item["match_id"]), item["reason"])
    )
    return frame, audit
