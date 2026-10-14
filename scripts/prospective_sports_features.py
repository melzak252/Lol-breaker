"""Reconstruct odds-free canonical79 inputs from strictly prior source days.

The shared chronological replay also generates new EXP081 training snapshots.
Source calendar dates are not publication timestamps: this is a conservative
prior-completed-day research protocol, never a captured prospective forecast.
Frozen fitted artifacts retain their old, incompatible feature semantics.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import copy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import importlib
from itertools import combinations, groupby
import math
from typing import Any, Iterable

import pandas as pd

from scripts.build_siamese_research_dataset import _complete_game, _history_helpers
from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS
from src.utils import golgg_schema as schema

FEATURE_VERSION = "canonical79-chronological-priorday-v2"
MODEL_VERSION = "exp081-chronological-priorday-v2"
FEATURE_CONTRACT = {
    "version": FEATURE_VERSION,
    "history": "complete series released after final source game day; numeric series ID then source game order",
    "updates": "six rating families; simultaneous Glicko opponent updates; exact per-game player IDs",
    "decay": "Glicko full seven-day periods since last completed-series day; nonmutating cutoff snapshot",
    "w20": "last twenty released games, including valid draws; missing measurements occupy their real positions",
    "rosters": "explicit five-player IDs with prior appearances; training uses previous-observed roster only",
    "cutoff": "strictly prior completed calendar day; same-day outcomes never enter any daily snapshot",
    "algebra": "ratings-w20-symmetric-series-v1 canonical79; direct calibrated series output",
}

# Native46-only supplement; the canonical79 contract and defaults are unchanged.
NATIVE_FIELDS = ("player_tm_sigma_avg1", "player_tm_sigma_avg2")
NATIVE_SUPPLEMENT = {
    "version": "native46-tm-sigma-priorday-v1",
    "fields": list(NATIVE_FIELDS),
    "semantics": "arithmetic mean TM player sigma from identical prior-day state and explicit prior-known roster; no decay or imputation",
}


@dataclass(slots=True)
class _Game:
    day: date
    players1: list[str]
    players2: list[str]
    score: int
    history1: dict[str, float] | None
    history2: dict[str, float] | None


@dataclass(slots=True)
class _Series:
    order: tuple[int, int | str]
    team1: str
    team2: str
    games: list[_Game]
    identifier: str
    start_day: date
    best_of: int
    target: int | None
    is_draw: bool


def _identity(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"missing or unusable {context}: {value!r}")
    result = str(value)
    if not result or result != result.strip() or result.lower() in {"none", "null", "nan"}:
        raise ValueError(f"missing or unusable {context}: {value!r}")
    return result


def _order(identifier: str) -> tuple[int, int | str]:
    # Real GOL.GG IDs are numeric. Explicit fixture IDs have a lexical fallback.
    return (0, int(identifier)) if identifier.isascii() and identifier.isdecimal() else (1, identifier)


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"invalid {context}: {value!r}")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"invalid {context}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"invalid {context}: {value!r}")
    return result


def _flag(value: Any, context: str) -> bool:
    if not isinstance(value, (bool, int)) or value not in (0, 1):
        raise ValueError(f"missing or invalid {context}: {value!r}")
    return bool(value)


def _roster(values: Any, context: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError(f"{context} must contain exactly five explicit player IDs")
    result = sorted(_identity(value, context) for value in values)
    if len(set(result)) != 5:
        raise ValueError(f"{context} contains duplicate player IDs")
    return result


def _game_roster(game: dict, side: str) -> list[str]:
    payload = game.get(f"{side}_players")
    if not isinstance(payload, dict) or len(payload) != 5:
        raise ValueError(f"{side} game roster must contain five explicit players")
    ids = []
    for player in payload.values():
        if not isinstance(player, dict):
            raise ValueError("invalid game player payload")
        # Do not let the schema helper substitute a display name for an ID.
        ids.append(_identity(schema.first_present(player, ("player_id", "id")), "game player ID"))
    return _roster(ids, f"{side} game roster")


def _history_row(game: dict, team: str, helpers: Any) -> dict[str, float] | None:
    side = "t1" if str(game["t1_id"]) == team else "t2"
    # _complete_game assumes mapping-shaped nested stats; malformed/missing
    # measurements are missing history, not zero-valued helper fallbacks.
    if not isinstance(game.get(f"{side}_stats"), dict) or any(
        not isinstance(player.get("stats"), dict)
        for player in game[f"{side}_players"].values()
    ):
        return None
    if not _complete_game(game, team):
        return None
    compact: dict[str, deque] = {}
    helpers.update_team_history(compact, team, game, 20)
    row = compact[team][0]
    return row if all(math.isfinite(value) for value in row.values()) else None


def _game_days(raw: dict, series_day: date) -> list[date]:
    games = raw.get("games")
    if not isinstance(games, list) or not games or any(not isinstance(game, dict) for game in games):
        raise ValueError("a historical series must contain completed game objects")
    try:
        days = [date.fromisoformat(game["date"]) if "date" in game else series_day for game in games]
    except (TypeError, ValueError) as exc:
        raise ValueError("historical game has no usable calendar date") from exc
    if days[0] < series_day or any(later < earlier for earlier, later in zip(days, days[1:])):
        raise ValueError("game dates contradict the declared series chronology")
    return days


def _series_team_ids(raw: dict) -> tuple[str, str]:
    """Resolve duplicate match metadata only from consistent exact game IDs.

    With both IDs absent the schema contract uses first-game team order, not
    display names, winner labels, scores, or blue/red side. Directional header
    results are checked only when an explicit match ID grounds their orientation.
    Otherwise only orientation-independent score/count/winner claims are checked.
    """
    declared = []
    for side in (1, 2):
        values = {
            _identity(raw[key], f"match team {side} ID")
            for key in (f"tid_{side}", f"t{side}_id")
            if raw.get(key) is not None
        }
        if len(values) > 1:
            raise ValueError(f"conflicting match team {side} ID aliases")
        declared.append(next(iter(values), None))
    pair = None
    for game in raw["games"]:
        sides = tuple(_identity(game.get(f"t{side}_id"), f"game {game.get('game_id')} team {side} ID") for side in (1, 2))
        if sides[0] == sides[1]:
            raise ValueError(f"game {game.get('game_id')} must have two distinct exact team IDs")
        if pair is None:
            pair = set(sides)
        elif set(sides) != pair:
            raise ValueError(f"game {game.get('game_id')} team IDs contradict earlier games")
    if pair is None:
        raise ValueError("a historical series must contain completed game objects")
    if any(team is not None and team not in pair for team in declared):
        raise ValueError("declared match team IDs contradict exact game IDs")
    # These helpers are safe only after validating every game and explicit alias.
    t1, t2 = schema.team1_id(raw), schema.team2_id(raw)
    if t1 == t2 or {t1, t2} != pair:
        raise ValueError("a historical series must have two distinct teams")
    return t1, t2


def _compact_series(raw: dict, order: tuple, helpers: Any, game_ids: set, game_days: list[date]) -> _Series:
    t1, t2 = _series_team_ids(raw)
    bon = _integer(schema.first_present(raw, ("BoN", "best_of")), "historical best_of")
    if bon < 1:
        raise ValueError("historical best_of must be positive")
    raw_games = raw["games"]
    if raw.get("games_played") is not None and _integer(raw["games_played"], "games_played") != len(raw_games):
        raise ValueError("declared games_played disagrees with actual game count")
    is_draw = _flag(raw.get("draw", False), "series draw flag")
    raw_initial = raw.get("initial_wins")
    if raw_initial is not None:
        if not isinstance(raw_initial, dict):
            raise ValueError("invalid initial map advantage")
        for k, v in raw_initial.items():
            if not isinstance(v, int) or v < 0:
                raise ValueError("invalid initial map advantage")
    initial = schema.initial_wins(raw)
    for tid in initial:
        if tid not in (t1, t2):
            raise ValueError(f"initial_wins team ID {tid} not in series teams ({t1}, {t2})")
    adv1 = initial.get(t1, 0)
    adv2 = initial.get(t2, 0)
    target = bon // 2 + 1 if bon != 2 else bon
    if adv1 < 0 or adv2 < 0 or (adv1 > 0 and adv2 > 0) or (bon != 2 and (adv1 >= target or adv2 >= target)):
        raise ValueError("invalid initial map advantage")
    compact_games = []
    score1 = score2 = 0
    for game, game_day in zip(raw_games, game_days, strict=True):
        gid = _order(_identity(game.get("game_id"), "game ID"))
        if gid in game_ids:
            raise ValueError("duplicate prior game ID")
        game_ids.add(gid)
        sides = (_identity(game.get("t1_id"), "game team 1 ID"), _identity(game.get("t2_id"), "game team 2 ID"))
        if set(sides) != {t1, t2}:
            raise ValueError("game team IDs do not match the historical series")
        win1 = _flag(game.get("t1_win"), "game team 1 result")
        win2 = _flag(game.get("t2_win"), "game team 2 result")
        if win1 == win2 or _flag(game.get("draw", False), "game draw flag"):
            raise ValueError("historical game has no unique winner")
        p1, p2 = _game_roster(game, "t1"), _game_roster(game, "t2")
        if set(p1).intersection(p2):
            raise ValueError("historical opponents share a player ID")
        if sides[0] != t1:
            p1, p2 = p2, p1
        score = schema.game_score_for_match_team1(raw, game)
        if max(score1 + adv1, score2 + adv2) >= target:
            raise ValueError("historical series contains a game after its deciding win")
        score1 += score
        score2 += 1 - score
        compact_games.append(_Game(game_day, p1, p2, score, _history_row(game, t1, helpers), _history_row(game, t2, helpers)))
    total1 = score1 + adv1
    total2 = score2 + adv2
    if is_draw:
        if len(compact_games) != bon or total1 != total2:
            raise ValueError("historical drawn series is incomplete or inconsistent with best_of")
    elif len(compact_games) > bon or max(total1, total2) != target:
        raise ValueError("historical series is incomplete or inconsistent with best_of")
    oriented = any(schema.first_present(raw, (f"tid_{side}", f"t{side}_id")) is not None for side in (1, 2))
    declared_scores = []
    for keys, expected in ((('score_1', 't1_score'), score1), (('score_2', 't2_score'), score2)):
        values = {_integer(raw[key], key) for key in keys if raw.get(key) is not None}
        if len(values) > 1:
            raise ValueError("conflicting historical series score aliases")
        value = next(iter(values), None)
        declared_scores.append(value)
        if value is not None and (value != expected if oriented else value not in (score1, score2)):
            raise ValueError("declared historical series score disagrees with games")
    if all(value is not None for value in declared_scores) and sorted(declared_scores) != sorted((score1, score2)):
        raise ValueError("declared historical series score totals disagree with games")
    declared_winners = []
    for key, expected in (("t1_win", total1 > total2), ("t2_win", total2 > total1)):
        value = _flag(raw[key], key) if raw.get(key) is not None else None
        declared_winners.append(value)
        if value is not None and ((oriented and value != expected) or (is_draw and value)):
            raise ValueError("declared historical series winner disagrees with games")
    if all(value is not None for value in declared_winners) and sum(declared_winners) != int(not is_draw):
        raise ValueError("declared historical series has no consistent winner")
    target_label = None if is_draw or bon not in (1, 3, 5) or any(initial.values()) else int(total1 > total2)
    return _Series(order, t1, t2, compact_games, str(raw["match_id"]),
                   date.fromisoformat(raw["date"]), bon, target_label, is_draw)


def _prepare_history(matches: Iterable[dict], cutoff: date) -> tuple[list[_Series], dict]:
    """Validate consumed history only; never inspect cutoff-day/future outcomes."""
    helpers = _history_helpers()
    series, match_ids, game_ids = [], set(), set()
    excluded = Counter(current_day=0, future_day=0)
    normalized = raw_count = 0
    for raw in matches:
        raw_count += 1
        if not isinstance(raw, dict):
            raise ValueError("historical source rows must be objects with dates")
        try:
            day = date.fromisoformat(raw["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("historical source row has no usable calendar date") from exc
        if day >= cutoff:
            excluded["current_day" if day == cutoff else "future_day"] += 1
            continue
        days = _game_days(raw, day)
        if days[-1] >= cutoff:
            excluded["current_day" if days[-1] == cutoff else "future_day"] += 1
            continue
        mid = _order(_identity(raw.get("match_id"), "match ID"))
        if mid in match_ids:
            raise ValueError("duplicate prior match ID")
        match_ids.add(mid)
        try:
            item = _compact_series(raw, mid, helpers, game_ids, days)
        except ValueError as exc:
            raise ValueError(f"historical series {raw['match_id']}: {exc}") from exc
        series.append(item)
        normalized += any(schema.first_present(raw, (f"tid_{side}", f"t{side}_id")) is None for side in (1, 2))
    series.sort(key=lambda item: (item.games[-1].day, item.order))
    return series, {
        "raw_series": raw_count, "used_series": len(series), "exclusions": dict(excluded),
        "normalization": {
            "series_with_game_derived_team_ids": normalized,
            "identity_policy": "consistent exact map IDs only; no names or academy alias merging",
            "draw_policy": "fully validated drawn series contribute state but never binary labels",
        },
    }


def _announced_rosters(rosters: dict) -> dict[str, list[str]]:
    if not isinstance(rosters, dict) or len(rosters) < 2:
        raise ValueError("at least two announced team rosters are required")
    announced, players = {}, set()
    for team, ids in rosters.items():
        team = _identity(team, "announced team ID")
        roster = _roster(ids, f"announced roster for {team}")
        if team in announced:
            raise ValueError("duplicate announced team ID")
        if players.intersection(roster):
            raise ValueError("announced teams share a player ID")
        announced[team] = roster
        players.update(roster)
    return announced


class ChronologicalFeatureState:
    """One shared rating/W20 transition implementation for training and inference.

    A multi-day series is released as a unit on its final source day. Earlier
    maps cannot be retroactively inserted into prior snapshots when it completes.
    This intentional delay is conservative, not invented intraday availability.
    """

    def __init__(self):
        self.manager = importlib.import_module("scripts.05_ratingi_baseline.03_generate_ratings").create_rating_manager()
        self.helpers = _history_helpers()
        self.histories = defaultdict(lambda: deque(maxlen=20))
        self.team_series, self.team_games = Counter(), Counter()
        self.player_series, self.player_games = Counter(), Counter()
        self.latest_rosters = {}
        self.first_day = self.last_day = None
        self.last_order = None

    def consume(self, item: _Series) -> None:
        day = item.games[-1].day
        key = (day, item.order)
        if self.last_order is not None and key <= self.last_order:
            raise ValueError("state transitions require unique chronological completion-day/series order")
        self.last_order = key
        self.first_day = item.games[0].day if self.first_day is None else min(self.first_day, item.games[0].day)
        self.last_day = day
        self.team_series.update((item.team1, item.team2))
        self.player_series.update({p for game in item.games for p in (*game.players1, *game.players2)})
        for index, game in enumerate(item.games):
            self.manager.update_before_match(item.team1, item.team2, game.players1, game.players2, day)
            self.manager.update_after_game(item.team1, item.team2, game.players1, game.players2, game.score, 1-game.score)
            self.manager.update_after_match(item.team1, item.team2, game.players1, game.players2, [game.score])
            self.histories[item.team1].append(game.history1)
            self.histories[item.team2].append(game.history2)
            self.team_games.update((item.team1, item.team2))
            self.player_games.update((*game.players1, *game.players2))
            for team, roster in ((item.team1, game.players1), (item.team2, game.players2)):
                observed_key = (game.day, item.order, index)
                if team not in self.latest_rosters or observed_key > self.latest_rosters[team][0]:
                    self.latest_rosters[team] = (observed_key, roster)

    def pairs(self, rosters: dict, cutoff: date, best_ofs: Iterable[int], *,
              include_native: bool = False) -> pd.DataFrame:
        announced = _announced_rosters(rosters)
        if not isinstance(cutoff, date) or isinstance(cutoff, datetime):
            raise ValueError("cutoff must be a calendar date")
        formats = list(best_ofs)
        if not formats or any(type(bo) is not int or bo not in (1, 3, 5) for bo in formats):
            raise ValueError("requested best_ofs must contain only 1, 3, 5")
        if self.last_day is None:
            raise ValueError("no usable strictly prior historical series")
        if self.last_day >= cutoff:
            raise ValueError("state contains cutoff-day or future outcomes")
        teams, rolling = sorted(announced), {}
        for team in teams:
            if not self.team_series[team]:
                raise ValueError(f"announced team {team!r} has no prior history")
            unknown = [p for p in announced[team] if not self.player_games[p]]
            if unknown:
                raise ValueError(f"announced team {team!r} has unobserved player IDs: {unknown}")
            if not self.histories[team] or any(row is None for row in self.histories[team]):
                raise ValueError(f"announced team {team!r} has incomplete recent W20 history")
            rolling[team] = self.helpers.average_history(self.histories[team])
        # Snapshot only Glicko's queried objects. Never decay the persistent state
        # or reset last-played dates for hypothetical pairings/training targets.
        glicko, restored = self.manager.systems["gl"], []
        try:
            for ratings, dates, ids in (
                (glicko.team_ratings, glicko.team_last_played, teams),
                (glicko.player_ratings, glicko.player_last_played,
                 sorted({p for roster in announced.values() for p in roster})),
            ):
                for identity in ids:
                    original = ratings[identity]
                    restored.append((ratings, identity, original))
                    ratings[identity] = copy(original)
                    glicko.apply_time_decay(ratings[identity], dates[identity], cutoff)
            output = []
            max_at = datetime.combine(self.last_day + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat()
            for a, b in combinations(teams, 2):
                snapshot = self.manager.predict_match(a, b, announced[a], announced[b])
                for side, team in ((1, a), (2, b)):
                    snapshot[f"days_since_last_{side}"] = (cutoff-self.manager.last_match_date[team]).days
                    snapshot.update({f"t{side}_rolling_{k}": v for k, v in rolling[team].items()})
                values = {field: float(snapshot[field]) for field in sorted(_REQUIRED_BASE_FIELDS)}
                if include_native:
                    tm = self.manager.systems["tm"]
                    for side, team in ((1, a), (2, b)):
                        values[f"player_tm_sigma_avg{side}"] = float(
                            sum(tm.get_player_rating(p).sigma for p in announced[team]) / len(announced[team]))
                    if any(values[field] <= 0 for field in NATIVE_FIELDS):
                        raise ValueError("native TM sigma must be positive")
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("nonfinite sports features")
                if any(not 0 <= values[field] <= 1 for field in _PROBABILITY_FIELDS):
                    raise ValueError("rating probabilities must be bounded in [0, 1]")
                for bo in sorted(set(formats)):
                    output.append(dict(team_a=a, team_b=b, best_of=bo, feature_history_max_at=max_at, **values))
            return pd.DataFrame(output)
        finally:
            for ratings, identity, original in restored:
                ratings[identity] = original

    def audit(self, cutoff: date) -> dict:
        return {
            "feature_version": FEATURE_VERSION, "feature_contract": dict(FEATURE_CONTRACT),
            "used_games": sum(self.team_games.values()) // 2,
            "source_date_min": self.first_day.isoformat() if self.first_day else None,
            "source_date_max": self.last_day.isoformat() if self.last_day else None,
            "cutoff": cutoff.isoformat(), "window_games": 20,
            "feature_history_max_at": datetime.combine(self.last_day + timedelta(days=1), time.min, tzinfo=timezone.utc).isoformat() if self.last_day else None,
            "history_timestamp_semantics": "exclusive UTC next-midnight upper bound of final consumed source day; not source availability",
            "team_history_counts": {t: {"series": self.team_series[t], "games": self.team_games[t], "w20_games": len(self.histories[t])} for t in sorted(self.team_series)},
            "player_history_counts": {p: {"series": self.player_series[p], "games": self.player_games[p]} for p in sorted(self.player_series)},
            "compatibility_limitations": [
                "only separately trained chronological-v2 models share these semantics; frozen/old corrected folds remain incompatible",
                "source calendar dates do not certify availability; explicit roster IDs do not certify announcement time",
            ],
            "intentional_legacy_changes": list(FEATURE_CONTRACT.values()),
        }


def build_pair_features(
    matches: Iterable[dict], rosters: dict[str, list[str]], cutoff: date,
    best_ofs: Iterable[int], *, include_native: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build every hypothetical pair/Bo from the shared strictly prior-day state."""
    if not isinstance(cutoff, date) or isinstance(cutoff, datetime):
        raise ValueError("cutoff must be a calendar date")
    series, audit = _prepare_history(matches, cutoff)
    state = ChronologicalFeatureState()
    for item in series:
        state.consume(item)
    frame = state.pairs(rosters, cutoff, best_ofs, include_native=include_native)
    return frame, {**audit, **state.audit(cutoff)}


def build_chronological_training_rows(matches: Iterable[dict]) -> tuple[pd.DataFrame, dict]:
    """Linear replay with prior-observed roster proxies, never target-map rosters.

    Missing feature history excludes a labeled target with a reason. Invalid
    source transitions are fatal, not silently removed from the rating history.
    """
    series, audit = _prepare_history(matches, date.max)
    state, position, rows, excluded = ChronologicalFeatureState(), 0, [], []
    targets = sorted(series, key=lambda item: (item.start_day, item.order))
    for day, grouped in groupby(targets, key=lambda item: item.start_day):
        while position < len(series) and series[position].games[-1].day < day:
            state.consume(series[position])
            position += 1
        for item in grouped:
            if item.target is None:
                excluded.append({"golgg_match_id": item.identifier, "reason": "state_only_nonbinary_or_initial_advantage"})
                continue
            teams = sorted((item.team1, item.team2))
            if any(team not in state.latest_rosters for team in teams):
                excluded.append({"golgg_match_id": item.identifier, "reason": "no_prior_observed_roster"})
                continue
            rosters = {team: state.latest_rosters[team][1] for team in teams}
            try:
                row = state.pairs(rosters, day, [item.best_of]).iloc[0].to_dict()
            except ValueError as exc:
                excluded.append({"golgg_match_id": item.identifier, "reason": str(exc)})
                continue
            row.update(
                golgg_match_id=item.identifier, date=day.isoformat(),
                result_day=item.games[-1].day.isoformat(), team1_id=teams[0], team2_id=teams[1],
                BoN=item.best_of, y_true=item.target if item.team1 == teams[0] else 1-item.target,
            )
            rows.append(row)
    return pd.DataFrame(rows), {
        **audit, "feature_version": FEATURE_VERSION, "feature_contract": dict(FEATURE_CONTRACT),
        "roster_policy": "previous_observed_roster_proxy", "included_rows": len(rows),
        "state_only_series": sum(item.target is None for item in series),
        "validated_maps": sum(len(item.games) for item in series),
        "drawn_series": sum(item.is_draw for item in series),
        "history_series_consumed": position,
        "source_date_min": min((item.start_day for item in series), default=None).isoformat() if series else None,
        "source_date_max": series[-1].games[-1].day.isoformat() if series else None,
        "target_exclusions": excluded, "qualification": "chronologically_reconstructed_research_only",
        "availability_certified": False,
    }
