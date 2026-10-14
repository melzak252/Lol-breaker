"""Chronological raw-history features and independent Gaussian BT controls.

This module intentionally depends only on the raw JSON stream and GOL.GG
identity helpers. It does not load an existing model, feature bank, or rating
snapshot. Target lineups are retrospective first-map lineups; the prior-roster
scenario is sourced exclusively from released historical maps.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from scripts.export_prospective_features import iter_matches
from src.utils import golgg_schema as schema

ROLE_ORDER = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
SOURCE_STATS = ("csm", "gpm", "dpm", "vspm", "kp%", "dmg%", "gd@15", "csd@15", "xpd@15")
PLAYER_FEATURE_NAMES = (
    "slow_rating_mean", "slow_rating_sd", "fast_rating_mean", "fast_rating_sd",
    "log_appearance_count", "log_days_since_last_appearance",
    "ewma10_opponent_strength", "ewma30_opponent_strength",
    "ewma10_win_residual", "ewma30_win_residual",
    *(f"paired_{name}_diff_ewma" for name in SOURCE_STATS),
    *(f"paired_{name}_observation_count" for name in SOURCE_STATS),
)
TEAM_FEATURE_NAMES = (
    "slow_rating_mean", "slow_rating_sd", "fast_rating_mean", "fast_rating_sd",
    "log_map_count", "log_days_since_last_map",
    "ewma10_win_residual", "ewma30_win_residual",
    "prior_lineup_overlap_fraction", "prior_lineup_known",
    "player_pair_coplay_mean_count", "player_pair_coplay_min_count",
)

_PROCESS_SLOW = 0.002
_PROCESS_FAST = 0.02
_VARIANCE_CAP = 4.0
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(128)
_HERMITE_WEIGHTS = _HERMITE_WEIGHTS / math.sqrt(math.pi)
_STAT_INDEX = {name: i for i, name in enumerate(SOURCE_STATS)}
_ROLE_ALIASES = {
    "top": "TOP", "toplane": "TOP",
    "jungle": "JUNGLE", "jungler": "JUNGLE", "jg": "JUNGLE", "jng": "JUNGLE",
    "mid": "MID", "middle": "MID", "midlane": "MID",
    "adc": "ADC", "bot": "ADC", "bottom": "ADC", "adcarry": "ADC", "marksman": "ADC",
    "support": "SUPPORT", "sup": "SUPPORT",
}


def _identity(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"missing or unusable {context}: {value!r}")
    result = str(value)
    if not result or result != result.strip() or result.lower() in {"none", "null", "nan"}:
        raise ValueError(f"missing or unusable {context}: {value!r}")
    return result


def _day(value: Any, context: str) -> date:
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError
        text = value.strip()
        try:
            return date.fromisoformat(text)
        except ValueError:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an ISO calendar date: {value!r}") from exc


def _integer(value: Any, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid {context}: {value!r}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid {context}: {value!r}") from exc
    else:
        raise ValueError(f"invalid {context}: {value!r}")
    if result < (1 if positive else 0):
        raise ValueError(f"invalid {context}: {value!r}")
    return result


def _flag(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"missing or invalid {context}: {value!r}")


def _aliased_id(raw: Mapping[str, Any], keys: tuple[str, ...], context: str) -> str | None:
    values = {_identity(raw[key], context) for key in keys if raw.get(key) is not None}
    if len(values) > 1:
        raise ValueError(f"conflicting {context} aliases")
    return next(iter(values), None)


def _team_ids(raw: Mapping[str, Any], games: list[dict[str, Any]]) -> tuple[str, str, str]:
    header_a = _aliased_id(raw, ("tid_1", "t1_id"), "match team 1 ID")
    header_b = _aliased_id(raw, ("tid_2", "t2_id"), "match team 2 ID")
    if header_a is not None and header_b is not None and header_a == header_b:
        raise ValueError("a historical series must have two distinct exact team IDs")
    first = (
        _identity(games[0].get("t1_id"), "first-map team 1 ID"),
        _identity(games[0].get("t2_id"), "first-map team 2 ID"),
    )
    if first[0] == first[1]:
        raise ValueError("a historical series must have two distinct exact team IDs")
    pair = set(first)
    for game in games:
        left = _identity(game.get("t1_id"), f"game {game.get('game_id')} team 1 ID")
        right = _identity(game.get("t2_id"), f"game {game.get('game_id')} team 2 ID")
        if left == right or {left, right} != pair:
            raise ValueError(f"game {game.get('game_id')} team IDs contradict the series")
    for declared in (header_a, header_b):
        if declared is not None and declared not in pair:
            raise ValueError("declared match team IDs contradict exact game IDs")
    a = header_a or (header_b and next(team for team in pair if team != header_b)) or first[0]
    b = header_b or (header_a and next(team for team in pair if team != header_a)) or first[1]
    if a == b or {a, b} != pair:
        raise ValueError("a historical series must have two distinct teams")
    provenance = "match_header_ids" if header_a is not None or header_b is not None else "first_map_ids"
    return a, b, provenance


def _canonical_role(value: Any) -> str | None:
    key = "".join(ch for ch in str(value).lower() if ch.isalnum())
    return _ROLE_ALIASES.get(key)


def _player_id(player: Any, context: str) -> str:
    if not isinstance(player, Mapping):
        raise ValueError(f"{context} must be a player object")
    values = {_identity(player[key], f"{context} ID") for key in ("player_id", "id") if player.get(key) is not None}
    if len(values) != 1:
        raise ValueError(f"{context} must have one unambiguous explicit player ID")
    return next(iter(values))


def _roster_payload(game: Mapping[str, Any], side: str) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    payload = game.get(f"{side}_players")
    if not isinstance(payload, Mapping) or len(payload) != 5:
        raise ValueError(f"{side} game roster must contain five role-labelled player objects")
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw_role, player in payload.items():
        role = _canonical_role(raw_role)
        if role is None and isinstance(player, Mapping):
            role = _canonical_role(player.get("role", player.get("position", "")))
        if role is None or role in by_role:
            raise ValueError(f"{side} game roster has an unknown or duplicate role: {raw_role!r}")
        if not isinstance(player, Mapping):
            raise ValueError(f"{side} game roster player for {role} must be an object")
        by_role[role] = player
    if set(by_role) != set(ROLE_ORDER):
        raise ValueError(f"{side} game roster must cover roles {ROLE_ORDER}")
    players = tuple(by_role[role] for role in ROLE_ORDER)
    ids = tuple(_player_id(player, f"{side} {role}") for role, player in zip(ROLE_ORDER, players, strict=True))
    if len(set(ids)) != 5:
        raise ValueError(f"{side} game roster contains duplicate player IDs")
    return ids, players


def _stat_key(value: Any) -> str:
    key = "".join(ch for ch in str(value).lower() if ch.isalnum())
    aliases = {
        "killparticipation": "kp", "killparticipationpct": "kp", "kpct": "kp",
        "damageshare": "dmg", "damagesharepct": "dmg", "damagepercent": "dmg", "dmgpct": "dmg",
        "xpat15": "xpd15", "xpdiffat15": "xpd15", "experiencediffat15": "xpd15",
    }
    return aliases.get(key, key)


def _number(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float, np.number)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip().replace(",", "").replace("%", "")
        if text.lower() in {"", "-", "--", "n/a", "na", "null", "none", "nan", "?"}:
            return math.nan
        try:
            result = float(text)
        except ValueError:
            return math.nan
    else:
        return math.nan
    return result if math.isfinite(result) else math.nan


def _source_stats(players: tuple[Mapping[str, Any], ...]) -> tuple[tuple[float, ...], ...]:
    output = []
    for player in players:
        stats = player.get("stats")
        indexed: dict[str, Any] = {}
        if isinstance(stats, Mapping):
            for key, value in stats.items():
                indexed.setdefault(_stat_key(key), value)
        output.append(tuple(_number(indexed.get(_stat_key(name))) for name in SOURCE_STATS))
    return tuple(output)


def _optional_integer_alias(raw: Mapping[str, Any], keys: tuple[str, ...], context: str) -> int | None:
    values = {_integer(raw[key], key) for key in keys if raw.get(key) is not None}
    if len(values) > 1:
        raise ValueError(f"conflicting {context} aliases")
    return next(iter(values), None)


@dataclass(frozen=True, slots=True)
class _Map:
    game_id: str
    roster_a: tuple[str, ...]
    roster_b: tuple[str, ...]
    stats_a: tuple[tuple[float, ...], ...]
    stats_b: tuple[tuple[float, ...], ...]
    y_a: int


@dataclass(frozen=True, slots=True)
class _Series:
    source_index: int
    match_id: str
    day: date
    source_header_day: date
    release_day: date
    delayed_release: bool
    pre_series_date_corrected: bool
    best_of: int
    team_a: str
    team_b: str
    roster_a: tuple[str, ...]
    roster_b: tuple[str, ...]
    tournament: str | None
    y: int | None
    source_provenance: str
    maps: tuple[_Map, ...]


def _initial_advantage(raw: Mapping[str, Any], team_a: str, team_b: str) -> None:
    value = raw.get("initial_wins")
    if value is not None:
        if not isinstance(value, Mapping):
            raise ValueError("invalid initial map advantage")
        for team, wins in value.items():
            team_id = _identity(team, "initial_wins team ID")
            amount = _integer(wins, "initial_wins")
            if team_id not in (team_a, team_b):
                raise ValueError(f"initial_wins team ID {team_id} not in series teams")
            if amount:
                raise ValueError("map-advantage series are unsupported")
    for key in ("map_advantage", "map_advantages", "initial_map_advantage"):
        advantage = raw.get(key)
        if advantage in (None, 0, "0", {}, []):
            continue
        if isinstance(advantage, Mapping) and all(_integer(v, key) == 0 for v in advantage.values()):
            continue
        raise ValueError("map-advantage series are unsupported")


def _normalize_series(
    raw: Mapping[str, Any], source_index: int, match_ids: set[str], game_ids: set[str],
    release_overrides: Mapping[str, date], audit_counts: dict[str, Any],
) -> _Series:
    if not isinstance(raw, Mapping):
        raise ValueError("historical source rows must be objects")
    match_id = _identity(raw.get("match_id"), "match ID")
    if match_id in match_ids:
        raise ValueError(f"duplicate source match ID {match_id}")
    match_ids.add(match_id)
    try:
        series_day = _day(raw.get("date"), f"series {match_id} date")
        best_values = {
            _integer(raw[key], "best_of", positive=True)
            for key in ("BoN", "best_of") if raw.get(key) is not None
        }
        if len(best_values) != 1:
            raise ValueError("missing or conflicting positive best_of")
        best_of = next(iter(best_values))
        raw_games = raw.get("games")
        if not isinstance(raw_games, list) or not raw_games or any(not isinstance(game, Mapping) for game in raw_games):
            raise ValueError("a historical series must contain completed game objects")
        if len(raw_games) > best_of:
            raise ValueError("historical series has more maps than its best_of")
        team_a, team_b, identity_source = _team_ids(raw, raw_games)
        _initial_advantage(raw, team_a, team_b)
        raw_game_days = [
            _day(game.get("date", series_day.isoformat()), f"series {match_id} map date")
            for game in raw_games
        ]
        if any(b < a for a, b in zip(raw_game_days, raw_game_days[1:])):
            raise ValueError("map dates contradict declared series chronology")
        max_map_day = max(raw_game_days)
        audit_counts["max_source_header_day"] = max(
            series_day, audit_counts["max_source_header_day"] or series_day,
        )
        audit_counts["max_source_map_day"] = max(
            max_map_day, audit_counts["max_source_map_day"] or max_map_day,
        )
        forecast_day = min(series_day, min(raw_game_days))
        pre_series_date_corrected = forecast_day < series_day
        if pre_series_date_corrected:
            audit_counts["pre_series_date_corrections"].append(match_id)
        override = release_overrides.get(match_id, series_day)
        source_max_day = max(series_day, max_map_day)
        delayed_release = override > source_max_day
        release_day = max(source_max_day, override)
        wins_a = wins_b = 0
        compact_maps: list[_Map] = []
        first_rosters: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        explicit_header = any(raw.get(key) is not None for key in ("tid_1", "t1_id", "tid_2", "t2_id"))
        for game, map_day in zip(raw_games, raw_game_days, strict=True):
            game_id = _identity(game.get("game_id"), f"series {match_id} map ID")
            if game_id in game_ids:
                raise ValueError(f"duplicate source map ID {game_id}")
            game_ids.add(game_id)
            if game.get("match_id") is not None and _identity(game["match_id"], "map match ID") != match_id:
                raise ValueError(f"map {game_id} has a conflicting match ID")
            if game.get("source_best_of") is not None and _integer(game["source_best_of"], "source_best_of", positive=True) != best_of:
                raise ValueError(f"map {game_id} source_best_of contradicts series")
            side_ids = (
                _identity(game.get("t1_id"), f"map {game_id} team 1 ID"),
                _identity(game.get("t2_id"), f"map {game_id} team 2 ID"),
            )
            win_1 = _flag(game.get("t1_win"), f"map {game_id} team 1 result")
            win_2 = _flag(game.get("t2_win"), f"map {game_id} team 2 result")
            if win_1 == win_2 or (game.get("draw") is not None and _flag(game["draw"], f"map {game_id} draw flag")):
                raise ValueError(f"map {game_id} has no unique winner")
            roster_1, players_1 = _roster_payload(game, "t1")
            roster_2, players_2 = _roster_payload(game, "t2")
            if set(roster_1).intersection(roster_2):
                raise ValueError(f"map {game_id} opposing teams share a player ID")
            if side_ids[0] == team_a:
                roster_a, roster_b = roster_1, roster_2
                players_a, players_b = players_1, players_2
                y_a = int(win_1)
            else:
                roster_a, roster_b = roster_2, roster_1
                players_a, players_b = players_2, players_1
                y_a = int(win_2)
            if first_rosters is None:
                first_rosters = roster_a, roster_b
            if max(wins_a, wins_b) >= best_of // 2 + 1:
                raise ValueError(f"series {match_id} contains a map after its deciding win")
            wins_a += y_a
            wins_b += 1 - y_a
            stats_a = _source_stats(players_a)
            stats_b = _source_stats(players_b)
            compact_maps.append(_Map(game_id, roster_a, roster_b, stats_a, stats_b, y_a))
            for stat_index, stat_name in enumerate(SOURCE_STATS):
                for player_stats in (*stats_a, *stats_b):
                    if not math.isfinite(player_stats[stat_index]):
                        audit_counts["stat_missing"][stat_name] += 1
                    audit_counts["stat_seen"][stat_name] += 1
            audit_counts["player_ids"].update((*roster_a, *roster_b))
        if first_rosters is None:
            raise ValueError("a historical series has no first-map roster")
        if raw.get("games_played") is not None and _integer(raw["games_played"], "games_played") != len(raw_games):
            raise ValueError("declared games_played disagrees with actual map count")
        threshold = best_of // 2 + 1
        decisive = max(wins_a, wins_b) == threshold
        tied_full = len(raw_games) == best_of and wins_a == wins_b
        if not decisive and not tied_full:
            raise ValueError("historical series is incomplete or inconsistent with best_of")
        actual_draw = tied_full and not decisive
        if raw.get("draw") is not None and _flag(raw["draw"], "series draw flag") != actual_draw:
            raise ValueError("declared series draw flag disagrees with maps")
        declared_score_1 = _optional_integer_alias(raw, ("score_1", "t1_score"), "series score")
        declared_score_2 = _optional_integer_alias(raw, ("score_2", "t2_score"), "series score")
        if explicit_header:
            if declared_score_1 is not None and declared_score_1 != wins_a:
                raise ValueError("declared series score disagrees with maps")
            if declared_score_2 is not None and declared_score_2 != wins_b:
                raise ValueError("declared series score disagrees with maps")
        elif declared_score_1 is not None and declared_score_2 is not None:
            if sorted((declared_score_1, declared_score_2)) != sorted((wins_a, wins_b)):
                raise ValueError("declared series scores disagree with maps")
        elif declared_score_1 is not None and declared_score_1 not in (wins_a, wins_b):
            raise ValueError("declared series score disagrees with maps")
        elif declared_score_2 is not None and declared_score_2 not in (wins_a, wins_b):
            raise ValueError("declared series score disagrees with maps")
        declared_wins = []
        for key, expected in (("t1_win", wins_a > wins_b), ("t2_win", wins_b > wins_a)):
            declared_wins.append(_flag(raw[key], key) if raw.get(key) is not None else None)
            if explicit_header and declared_wins[-1] is not None and declared_wins[-1] != expected:
                raise ValueError("declared series winner disagrees with maps")
        if all(value is not None for value in declared_wins) and sum(declared_wins) != int(not actual_draw):
            raise ValueError("declared series winners are inconsistent")
        for key, value in (("source_series", 1), ("source_maps", len(compact_maps))):
            audit_counts[key] += value
        audit_counts["identity_provenance"][identity_source] += 1
        tournament_raw = schema.match_tournament(raw)
        tournament = str(tournament_raw) if tournament_raw not in (None, "") else None
        y = int(wins_a > wins_b) if best_of in (1, 3, 5) else None
        return _Series(
            source_index, match_id, forecast_day, series_day, release_day,
            delayed_release, pre_series_date_corrected, best_of, team_a, team_b,
            first_rosters[0], first_rosters[1], tournament, y,
            f"team_ids={identity_source}; maps=explicit_game_ids", tuple(compact_maps),
        )
    except ValueError as exc:
        if str(exc).startswith(f"series {match_id}:"):
            raise
        raise ValueError(f"series {match_id}: {exc}") from exc


def _validate_releases(release_days: Mapping[str, str]) -> dict[str, date]:
    if not isinstance(release_days, Mapping):
        raise ValueError("release_days must be a mapping of match IDs to ISO dates")
    output: dict[str, date] = {}
    for key, value in release_days.items():
        match_id = _identity(key, "release-day match ID")
        if match_id in output:
            raise ValueError(f"duplicate release-day match ID {match_id}")
        output[match_id] = _day(value, f"release day for match {match_id}")
    return output




def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _series_margin(log_odds: float, best_of: int) -> float:
    """Odd BO1/3/5 majority margin, evaluated without subtracting near-equal CDFs."""
    margin = math.tanh(log_odds / 2.0)
    if best_of == 1:
        return margin
    squared = margin * margin
    if best_of == 3:
        return margin * (3.0 - squared) / 2.0
    return margin * (15.0 + squared * (3.0 * squared - 10.0)) / 8.0


def _integrated_series_probability(mean: float, variance: float, best_of: int) -> float:
    scale = math.sqrt(2.0 * max(0.0, variance))
    # Integrate the odd margin: symmetric nodes cancel exactly for equal strength.
    margin = math.fsum(
        float(weight) * _series_margin(mean + scale * float(node), best_of)
        for node, weight in zip(_HERMITE_NODES, _HERMITE_WEIGHTS, strict=True)
    )
    return min(1.0, max(0.0, 0.5 + 0.5 * margin))


def _ewma(previous: float | None, observations: list[float], window: int) -> float | None:
    if not observations:
        return previous
    current = math.fsum(observations) / len(observations)
    base_alpha = 2.0 / (window + 1.0)
    batch_alpha = 1.0 - (1.0 - base_alpha) ** len(observations)
    return current if previous is None else batch_alpha * current + (1.0 - batch_alpha) * previous


def _history_player() -> dict[str, Any]:
    return {
        "appearances": 0, "last_day": None,
        "opponent10": None, "opponent30": None,
        "residual10": None, "residual30": None,
        "paired_ewma": {}, "paired_count": {},
    }


def _history_team() -> dict[str, Any]:
    return {"maps": 0, "last_day": None, "residual10": None, "residual30": None}


@dataclass(slots=True)
class ReplayState:
    """Serializable posterior/history snapshot with pure, future-only queries."""

    team_slow: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    team_fast: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    player_slow: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    player_fast: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    team_history: dict[str, dict[str, Any]] = field(default_factory=dict)
    player_history: dict[str, dict[str, Any]] = field(default_factory=dict)
    team_lineups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    last_release_day: str | None = None

    def _query_date(self, day: Any) -> date:
        target = _day(day, "forecast day")
        if self.last_release_day is not None and target <= date.fromisoformat(self.last_release_day):
            raise ValueError(
                f"forecast day {target.isoformat()} must be after consumed history {self.last_release_day}"
            )
        return target

    @staticmethod
    def _query_roster(values: Any, context: str) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple)) or len(values) != 5:
            raise ValueError(f"{context} must contain exactly five role-ordered player IDs")
        roster = tuple(_identity(value, f"{context} player ID") for value in values)
        if len(set(roster)) != 5:
            raise ValueError(f"{context} contains duplicate player IDs")
        return roster

    def features(
        self, team_a: str | None, roster_a: tuple[str, ...],
        team_b: str | None, roster_b: tuple[str, ...], day: str, best_of: int,
    ) -> dict[str, Any]:
        """Return features for a future matchup without mutating this snapshot."""
        target = self._query_date(day)
        if isinstance(best_of, bool) or best_of not in (1, 3, 5):
            raise ValueError(f"unsupported forecast best_of {best_of!r}; supported values are 1, 3, and 5")
        a = self._query_roster(roster_a, "roster_a")
        b = self._query_roster(roster_b, "roster_b")
        if set(a).intersection(b):
            raise ValueError("opposing forecast rosters overlap")
        team_a = None if team_a is None else _identity(team_a, "team_a ID")
        team_b = None if team_b is None else _identity(team_b, "team_b ID")
        if team_a is not None and team_a == team_b:
            raise ValueError("forecast teams must be distinct")
        return self._features_for_rosters(team_a, a, team_b, b, target, best_of)

    def _features_for_rosters(
        self, team_a: str | None, roster_a: tuple[str, ...] | None,
        team_b: str | None, roster_b: tuple[str, ...] | None, target: date, best_of: int,
    ) -> dict[str, Any]:
        players = np.empty((2, 5, len(PLAYER_FEATURE_NAMES)), dtype=np.float32)
        players[0] = _player_team_features(self, roster_a, target)
        players[1] = _player_team_features(self, roster_b, target)
        teams = np.empty((2, len(TEAM_FEATURE_NAMES)), dtype=np.float32)
        teams[0] = _team_features(self, team_a, roster_a, target)
        teams[1] = _team_features(self, team_b, roster_b, target)
        context = np.zeros(3, dtype=np.float32)
        context[{1: 0, 3: 1, 5: 2}[best_of]] = 1.0
        baselines = np.empty(2, dtype=np.float64)
        for column, slow_map in enumerate((self.team_slow, self.player_slow)):
            if column == 0:
                mean_a, var_a = _prior_filter(slow_map, team_a, target, _PROCESS_SLOW, 1.0)
                mean_b, var_b = _prior_filter(slow_map, team_b, target, _PROCESS_SLOW, 1.0)
                mean, variance = mean_a - mean_b, var_a + var_b
            else:
                mean_a, var_a = _player_team_strength(self, roster_a, target, _PROCESS_SLOW)
                mean_b, var_b = _player_team_strength(self, roster_b, target, _PROCESS_SLOW)
                mean, variance = mean_a - mean_b, var_a + var_b
            baselines[column] = _integrated_series_probability(mean, variance, best_of)
        return {
            "players": players,
            "teams": teams,
            "context": context,
            "baseline_series": baselines,
            "max_release_day": self.last_release_day,
        }


def _prior_filter(
    values: Mapping[str, tuple[float, float, str]], key: str | None,
    day: date, process: float, initial_var: float,
) -> tuple[float, float]:
    if key is None:
        return 0.0, initial_var
    return _prior_filter_impl(values, key, day, process, initial_var)


def _prior_filter_impl(
    values: Mapping[str, tuple[float, float, str]], key: str,
    day: date, process: float, initial_var: float,
) -> tuple[float, float]:
    entry = values.get(key)
    if entry is None:
        return 0.0, initial_var
    mean, variance, last_day = entry
    elapsed = max(0, (day - date.fromisoformat(last_day)).days)
    return mean, min(_VARIANCE_CAP, variance + process * elapsed)


def _player_team_strength(state: ReplayState, roster: tuple[str, ...] | None, day: date, process: float) -> tuple[float, float]:
    ids = roster if roster is not None else (None,) * 5
    values = [_prior_filter(state.player_slow, player, day, process, 4.0) for player in ids]
    return math.fsum(mean for mean, _ in values) / 5.0, math.fsum(var for _, var in values) / 25.0


def _player_team_features(state: ReplayState, roster: tuple[str, ...] | None, day: date) -> np.ndarray:
    rows = np.empty((5, len(PLAYER_FEATURE_NAMES)), dtype=np.float32)
    for role_index, player in enumerate(roster if roster is not None else (None,) * 5):
        if player is None:
            slow_mean, slow_var = 0.0, 4.0
            fast_mean, fast_var = 0.0, 4.0
            history = None
        else:
            slow_mean, slow_var = _prior_filter(state.player_slow, player, day, _PROCESS_SLOW, 4.0)
            fast_mean, fast_var = _prior_filter(state.player_fast, player, day, _PROCESS_FAST, 4.0)
            history = state.player_history.get(player)
        values = [
            slow_mean, math.sqrt(slow_var), fast_mean, math.sqrt(fast_var),
            math.log1p(history["appearances"]) if history else 0.0,
            math.log1p(max(0, (day - date.fromisoformat(history["last_day"])).days)) if history and history["last_day"] else math.nan,
            history["opponent10"] if history and history["opponent10"] is not None else math.nan,
            history["opponent30"] if history and history["opponent30"] is not None else math.nan,
            history["residual10"] if history and history["residual10"] is not None else math.nan,
            history["residual30"] if history and history["residual30"] is not None else math.nan,
        ]
        values.extend(
            history["paired_ewma"].get(name, math.nan) if history else math.nan
            for name in SOURCE_STATS
        )
        values.extend(
            float(history["paired_count"].get(name, 0)) if history else 0.0
            for name in SOURCE_STATS
        )
        rows[role_index] = _finite_float32(values)
    return rows


def _team_features(state: ReplayState, team: str | None, roster: tuple[str, ...] | None, day: date) -> np.ndarray:
    slow_mean, slow_var = _prior_filter(state.team_slow, team, day, _PROCESS_SLOW, 1.0)
    fast_mean, fast_var = _prior_filter(state.team_fast, team, day, _PROCESS_FAST, 1.0)
    history = state.team_history.get(team) if team is not None else None
    prior = state.team_lineups.get(team) if team is not None else None
    overlap = (
        len(set(roster).intersection(prior)) / 5.0
        if roster is not None and prior is not None else math.nan
    )
    pair_support = []
    if roster is not None:
        for left in range(5):
            for right in range(left + 1, 5):
                pair_support.append(state.pair_counts.get(tuple(sorted((roster[left], roster[right]))), 0))
    pair_mean = math.fsum(pair_support) / len(pair_support) if pair_support else 0.0
    pair_min = min(pair_support) if pair_support else 0.0
    values = (
        slow_mean, math.sqrt(slow_var), fast_mean, math.sqrt(fast_var),
        math.log1p(history["maps"]) if history else 0.0,
        math.log1p(max(0, (day - date.fromisoformat(history["last_day"])).days))
        if history and history["last_day"] else math.nan,
        history["residual10"] if history and history["residual10"] is not None else math.nan,
        history["residual30"] if history and history["residual30"] is not None else math.nan,
        overlap, float(prior is not None), pair_mean, float(pair_min),
    )
    return _finite_float32(values)


def _finite_float32(values: Any) -> np.ndarray:
    limit = np.finfo(np.float32).max
    clean = [float(v) if math.isfinite(float(v)) and abs(float(v)) <= limit else math.nan for v in values]
    return np.asarray(clean, dtype=np.float32)



def _add_contribution(bucket: dict[str, tuple[list[float], list[float]]], key: str, gradient: float, curvature: float) -> None:
    if key not in bucket:
        bucket[key] = ([], [])
    bucket[key][0].append(gradient)
    bucket[key][1].append(curvature)




def _update_filter(
    target: dict[str, tuple[float, float, str]],
    priors: Mapping[str, tuple[float, float]],
    contributions: Mapping[str, tuple[list[float], list[float]]],
    day: date,
) -> None:
    day_text = day.isoformat()
    for entity, (gradients, curvatures) in contributions.items():
        mean, variance = priors[entity]
        gradient = math.fsum(gradients)
        curvature = math.fsum(curvatures)
        posterior_variance = 1.0 / (1.0 / variance + curvature)
        posterior_mean = mean + posterior_variance * gradient
        target[entity] = (posterior_mean, min(_VARIANCE_CAP, posterior_variance), day_text)


def _consume_release_day(state: ReplayState, records: list[_Series], day: date) -> None:
    team_entities = {team for record in records for game in record.maps for team in (record.team_a, record.team_b)}
    player_entities = {player for record in records for game in record.maps for player in (*game.roster_a, *game.roster_b)}
    team_slow_prior = {key: _prior_filter_impl(state.team_slow, key, day, _PROCESS_SLOW, 1.0) for key in team_entities}
    team_fast_prior = {key: _prior_filter_impl(state.team_fast, key, day, _PROCESS_FAST, 1.0) for key in team_entities}
    player_slow_prior = {key: _prior_filter_impl(state.player_slow, key, day, _PROCESS_SLOW, 4.0) for key in player_entities}
    player_fast_prior = {key: _prior_filter_impl(state.player_fast, key, day, _PROCESS_FAST, 4.0) for key in player_entities}
    team_contrib_slow: dict[str, tuple[list[float], list[float]]] = {}
    team_contrib_fast: dict[str, tuple[list[float], list[float]]] = {}
    player_contrib_slow: dict[str, tuple[list[float], list[float]]] = {}
    player_contrib_fast: dict[str, tuple[list[float], list[float]]] = {}
    team_observations: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"residual": []})
    player_observations: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"opponent": [], "residual": [], "paired": defaultdict(list), "appearances": 0}
    )
    lineup_observations: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    pair_increments: Counter[tuple[str, str]] = Counter()

    for record in records:
        for game in record.maps:
            team_entities_for_map = (record.team_a, record.team_b)
            team_coefficients = (1.0, -1.0)
            team_mean_slow = team_slow_prior[record.team_a][0] - team_slow_prior[record.team_b][0]
            team_mean_fast = team_fast_prior[record.team_a][0] - team_fast_prior[record.team_b][0]
            p_team_slow = _sigmoid(team_mean_slow)
            p_team_fast = _sigmoid(team_mean_fast)
            player_entities_for_map = (*game.roster_a, *game.roster_b)
            player_coefficients = (0.2,) * 5 + (-0.2,) * 5
            player_mean_slow = math.fsum(
                coeff * player_slow_prior[player][0]
                for player, coeff in zip(player_entities_for_map, player_coefficients, strict=True)
            )
            player_mean_fast = math.fsum(
                coeff * player_fast_prior[player][0]
                for player, coeff in zip(player_entities_for_map, player_coefficients, strict=True)
            )
            p_player_slow = _sigmoid(player_mean_slow)
            p_player_fast = _sigmoid(player_mean_fast)
            outcome = float(game.y_a)
            for entity, coeff in zip(team_entities_for_map, team_coefficients, strict=True):
                _add_contribution(team_contrib_slow, entity, coeff * (outcome - p_team_slow), coeff * coeff * p_team_slow * (1.0 - p_team_slow))
                _add_contribution(team_contrib_fast, entity, coeff * (outcome - p_team_fast), coeff * coeff * p_team_fast * (1.0 - p_team_fast))
            for player, coeff in zip(player_entities_for_map, player_coefficients, strict=True):
                _add_contribution(player_contrib_slow, player, coeff * (outcome - p_player_slow), coeff * coeff * p_player_slow * (1.0 - p_player_slow))
                _add_contribution(player_contrib_fast, player, coeff * (outcome - p_player_fast), coeff * coeff * p_player_fast * (1.0 - p_player_fast))

            player_strength_a = math.fsum(player_slow_prior[p][0] for p in game.roster_a) / 5.0
            player_strength_b = math.fsum(player_slow_prior[p][0] for p in game.roster_b) / 5.0
            residual = outcome - p_team_slow
            team_observations[record.team_a]["residual"].append(residual)
            team_observations[record.team_b]["residual"].append(-residual)
            lineup_observations[record.team_a][game.roster_a] += 1
            lineup_observations[record.team_b][game.roster_b] += 1
            for side_roster, side_stats, opp_strength, win_residual in (
                (game.roster_a, game.stats_a, player_strength_b, game.y_a - p_player_slow),
                (game.roster_b, game.stats_b, player_strength_a, (1 - game.y_a) - (1.0 - p_player_slow)),
            ):
                for role_index, player in enumerate(side_roster):
                    observation = player_observations[player]
                    observation["appearances"] += 1
                    observation["opponent"].append(opp_strength)
                    observation["residual"].append(win_residual)
                for role_index, player in enumerate(side_roster):
                    opposing_stats = game.stats_b[role_index] if side_roster is game.roster_a else game.stats_a[role_index]
                    for stat_index, stat_name in enumerate(SOURCE_STATS):
                        own = side_stats[role_index][stat_index]
                        opposing = opposing_stats[stat_index]
                        if math.isfinite(own) and math.isfinite(opposing):
                            observation = player_observations[player]
                            observation["paired"][stat_name].append(own - opposing)
            # Each same-side pair appeared together once in this map.
            for roster in (game.roster_a, game.roster_b):
                for left in range(5):
                    for right in range(left + 1, 5):
                        pair_increments[tuple(sorted((roster[left], roster[right])))] += 1

    _update_filter(state.team_slow, team_slow_prior, team_contrib_slow, day)
    _update_filter(state.team_fast, team_fast_prior, team_contrib_fast, day)
    _update_filter(state.player_slow, player_slow_prior, player_contrib_slow, day)
    _update_filter(state.player_fast, player_fast_prior, player_contrib_fast, day)

    for team, observation in team_observations.items():
        history = state.team_history.setdefault(team, _history_team())
        history["maps"] += len(observation["residual"])
        history["last_day"] = day.isoformat()
        history["residual10"] = _ewma(history["residual10"], observation["residual"], 10)
        history["residual30"] = _ewma(history["residual30"], observation["residual"], 30)
    for team, counts in lineup_observations.items():
        state.team_lineups[team] = min(counts, key=lambda roster: (-counts[roster], roster))
    for player, observation in player_observations.items():
        history = state.player_history.setdefault(player, _history_player())
        history["appearances"] += observation["appearances"]
        history["last_day"] = day.isoformat()
        history["opponent10"] = _ewma(history["opponent10"], observation["opponent"], 10)
        history["opponent30"] = _ewma(history["opponent30"], observation["opponent"], 30)
        history["residual10"] = _ewma(history["residual10"], observation["residual"], 10)
        history["residual30"] = _ewma(history["residual30"], observation["residual"], 30)
        for stat_name, values in observation["paired"].items():
            history["paired_ewma"][stat_name] = _ewma(history["paired_ewma"].get(stat_name), values, 10)
            history["paired_count"][stat_name] = history["paired_count"].get(stat_name, 0) + len(values)
    for pair, increment in pair_increments.items():
        state.pair_counts[pair] = state.pair_counts.get(pair, 0) + int(round(increment))
    state.last_release_day = day.isoformat()


def _consume_batches(state: ReplayState, release_batches: list[tuple[date, list[_Series]]], position: int, cutoff: date) -> int:
    while position < len(release_batches) and release_batches[position][0] < cutoff:
        day, records = release_batches[position]
        _consume_release_day(state, records, day)
        position += 1
    return position


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_features(source: Path, release_days: Mapping[str, str], output_dir: Path) -> dict[str, str]:
    """Normalize, audit, replay, and persist the fixed from-scratch feature set."""
    source = Path(source)
    output_dir = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"historical match source does not exist: {source}")
    corrections = _validate_releases(release_days)
    source_hash = _hash_file(source)
    match_ids: set[str] = set()
    game_ids: set[str] = set()
    counts: dict[str, Any] = {
        "source_series": 0, "source_maps": 0,
        "player_ids": set(), "identity_provenance": Counter(),
        "stat_seen": Counter(), "stat_missing": Counter(),
        "max_source_header_day": None, "max_source_map_day": None,
        "pre_series_date_corrections": [],
    }
    series: list[_Series] = []
    for source_index, raw in enumerate(iter_matches(source)):
        series.append(_normalize_series(raw, source_index, match_ids, game_ids, corrections, counts))
    series.sort(key=lambda item: (item.release_day, item.match_id))
    by_release: list[tuple[date, list[_Series]]] = []
    for item in series:
        if not by_release or by_release[-1][0] != item.release_day:
            by_release.append((item.release_day, []))
        by_release[-1][1].append(item)

    targets = [item for item in series if item.best_of in (1, 3, 5) and item.y is not None]
    targets.sort(key=lambda item: item.source_index)
    target_index = {item.match_id: index for index, item in enumerate(targets)}
    targets_by_day: dict[date, list[_Series]] = defaultdict(list)
    for item in targets:
        targets_by_day[item.day].append(item)
    n_rows = len(targets)
    players = np.empty((n_rows, 2, 5, len(PLAYER_FEATURE_NAMES)), dtype=np.float32)
    teams = np.empty((n_rows, 2, len(TEAM_FEATURE_NAMES)), dtype=np.float32)
    context = np.empty((n_rows, 3), dtype=np.float32)
    baseline = np.empty((n_rows, 2), dtype=np.float64)
    prior_players = np.empty_like(players)
    prior_teams = np.empty_like(teams)
    prior_baseline = np.empty_like(baseline)
    max_history: dict[str, str | None] = {}
    replay = ReplayState()
    release_position = 0
    for forecast_day in sorted(targets_by_day):
        release_position = _consume_batches(replay, by_release, release_position, forecast_day)
        for item in targets_by_day[forecast_day]:
            index = target_index[item.match_id]
            current = replay.features(item.team_a, item.roster_a, item.team_b, item.roster_b, forecast_day.isoformat(), item.best_of)
            prior_roster_a = replay.team_lineups.get(item.team_a)
            prior_roster_b = replay.team_lineups.get(item.team_b)
            prior = replay._features_for_rosters(
                item.team_a, prior_roster_a, item.team_b, prior_roster_b, forecast_day, item.best_of,
            )
            players[index], teams[index], context[index], baseline[index] = (
                current["players"], current["teams"], current["context"], current["baseline_series"],
            )
            prior_players[index], prior_teams[index], prior_baseline[index] = (
                prior["players"], prior["teams"], prior["baseline_series"],
            )
            max_history[item.match_id] = replay.last_release_day
    while release_position < len(by_release):
        day, batch = by_release[release_position]
        _consume_release_day(replay, batch, day)
        release_position += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.parquet"
    arrays_path = output_dir / "arrays.npz"
    state_path = output_dir / "state.joblib"
    audit_path = output_dir / "audit.json"
    metadata_rows = []
    source_year_bo = Counter()
    target_year_bo = Counter()
    exclusions = Counter()
    delayed_releases = []
    for item in series:
        source_year_bo[(item.source_header_day.year, item.best_of)] += 1
        if item.delayed_release:
            delayed_releases.append(item.match_id)
        if item.best_of == 2:
            exclusions["all_bo2_series"] += 1
        elif item.best_of not in (1, 3, 5):
            exclusions["unsupported_target_best_of"] += 1
        else:
            target_year_bo[(item.source_header_day.year, item.best_of)] += 1
    for item in targets:
        metadata_rows.append({
            "match_id": item.match_id,
            "date": item.day.isoformat(),
            "release_day": item.release_day.isoformat(),
            "best_of": item.best_of,
            "y": int(item.y),
            "team_a": item.team_a,
            "team_b": item.team_b,
            "roster_a": list(item.roster_a),
            "roster_b": list(item.roster_b),
            "tournament": item.tournament,
            "source_provenance": item.source_provenance,
            "max_history_release_day": max_history[item.match_id],
        })
    metadata_columns = (
        "match_id", "date", "release_day", "best_of", "y", "team_a", "team_b",
        "roster_a", "roster_b", "tournament", "source_provenance", "max_history_release_day",
    )
    metadata = pd.DataFrame(metadata_rows, columns=metadata_columns)
    metadata.to_parquet(metadata_path, index=False)
    np.savez_compressed(
        arrays_path,
        players=players, teams=teams, context=context, baseline_series=baseline,
        prior_players=prior_players, prior_teams=prior_teams, prior_baseline_series=prior_baseline,
    )
    joblib.dump(replay, state_path, compress=3)
    observed_player_ids = len(counts["player_ids"])
    max_target_history = max((value for value in max_history.values() if value is not None), default=None)
    source_year_counts: dict[str, dict[str, int]] = defaultdict(dict)
    target_year_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for (year, best_of), count in sorted(source_year_bo.items()):
        source_year_counts[str(year)][str(best_of)] = count
    for (year, best_of), count in sorted(target_year_bo.items()):
        target_year_counts[str(year)][str(best_of)] = count
    audit = {
        "source_path": str(source.resolve()),
        "source_sha256": source_hash,
        "exact_counts": {"series": len(series), "maps": counts["source_maps"], "unique_player_ids": observed_player_ids},
        "identity_provenance_counts": dict(counts["identity_provenance"]),
        "target_rows": len(targets),
        "target_exclusions": dict(exclusions),
        "per_year_best_of_source_series": dict(source_year_counts),
        "per_year_best_of_binary_targets": dict(target_year_counts),
        "stat_missingness": {
            name: {"seen": int(counts["stat_seen"][name]), "missing": int(counts["stat_missing"][name])}
            for name in SOURCE_STATS
        },
        "chronology": {
            "release_policy": "max(series date, every map date, supplied corrected release day)",
            "forecast_policy": "consume only whole series with effective release day strictly before forecast date",
            "max_source_release_day": max((item.release_day for item in series), default=None).isoformat()
            if series else None,
            "max_target_history_release_day": max_target_history,
            "max_source_header_day": counts["max_source_header_day"].isoformat()
            if counts["max_source_header_day"] else None,
            "max_source_map_day": counts["max_source_map_day"].isoformat()
            if counts["max_source_map_day"] else None,
            "pre_series_header_map_date_correction_count": len(counts["pre_series_date_corrections"]),
            "pre_series_header_map_date_correction_match_ids": counts["pre_series_date_corrections"],
            "target_day_policy": "earliest of series header and all recorded map dates; correction count/IDs recorded; effective release remains max(series header, every map date, supplied correction)",
            "state_as_of": replay.last_release_day,
            "delayed_release_count": len(delayed_releases),
            "delayed_release_match_ids": delayed_releases,
            "unused_release_override_count": len(set(corrections) - match_ids),
            "strict_inequality_verified": all(
                value is None or value < item.day.isoformat()
                for item in targets for value in (max_history[item.match_id],)
            ),
        },
        "feature_names": {
            "role_order": list(ROLE_ORDER),
            "player": list(PLAYER_FEATURE_NAMES),
            "team": list(TEAM_FEATURE_NAMES),
            "source_stat_differences": list(SOURCE_STATS),
        },
        "feature_definitions": {
            "dynamic_filters": "independent diagonal Gaussian Bradley-Terry filters; slow q=.002/day, fast q=.02/day; cap variance at 4; team prior variance 1, player prior variance 4",
            "forecast_day_correction": "if a map date precedes the series header date, forecast date is moved to the earliest of the header and map dates; raw header remains in the release-day maximum",
            "player_experience": "log1p historical map appearances and log1p calendar days since last released appearance",
            "player_ewma": "10- and 30-appearance daily-batched EWMAs of opposing roster slow strength and per-map win residual",
            "paired_source_statistics": "same-role own-minus-opponent raw source-field differences; 10-appearance daily-batched EWMA and valid paired observation count",
            "team_history": "log1p released maps, inactivity, daily-batched residual EWMAs, overlap with latest released-day modal lineup, and historical roster-pair co-play support",
            "prior_roster_policy": "latest strictly prior released day modal lineup by organization, lexicographic tuple tie-break; no target-lineup fallback",
            "baseline_series": "24-node Gauss-Hermite expectation of the best-of win polynomial over Gaussian map-logit uncertainty; not a polynomial of a projected map probability",
            "covariance_limitation": "diagonal Gaussian approximation ignores posterior covariance among teams and players",
        },
        "output_sha256": {
            "metadata.parquet": _hash_file(metadata_path),
            "arrays.npz": _hash_file(arrays_path),
            "state.joblib": _hash_file(state_path),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "metadata_path": str(metadata_path),
        "arrays_path": str(arrays_path),
        "state_path": str(state_path),
        "audit_path": str(audit_path),
    }
