"""Compatibility helpers for GOL.GG match payload schemas.

The historical project scripts were written against an older flattened
``golgg_matches.json`` schema using keys such as ``name_1``, ``tid_1`` and
``players_1``. The current dataset uses keys such as ``sname_t1``, ``t1_id``
and stores rosters inside per-game payloads. This module centralizes schema
access so pipeline scripts can remain stable across both formats.
"""

from __future__ import annotations

import re
from typing import Any


def first_present(match: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Return the first non-missing value for a set of alternative keys.

    Args:
        match: Match dictionary from ``golgg_matches.json``.
        keys: Candidate keys ordered by preference.
        default: Value returned when no candidate key is present.

    Returns:
        First present and non-``None`` value, or ``default``.
    """

    for key in keys:
        value = match.get(key)
        if value is not None:
            return value
    return default


def _normalize_team_name(name: Any) -> str:
    """Normalize a team name for side inference across abbreviated labels.

    Args:
        name: Raw team name or abbreviation.

    Returns:
        Lowercase alphanumeric representation used only for conservative exact
        comparisons after removing punctuation and whitespace.
    """

    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _first_game(match: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first valid game payload when present."""

    match_games = games(match)
    return match_games[0] if match_games else None


def _infer_missing_team_id(
    match: dict[str, Any], missing_side: int, known_id: Any | None
) -> str:
    """Infer a missing match-level team identifier from the first game.

    Some historical records in the newer JSON export have one or both
    match-level identifiers missing, while per-game identifiers are populated.
    When one side is known, the missing side must be the opposite first-game
    side. When neither side is known, the function falls back to the same side
    in the first game.

    Args:
        match: Match dictionary from ``golgg_matches.json``.
        missing_side: ``1`` for match team 1, ``2`` for match team 2.
        known_id: Identifier of the opposite match side, if available.

    Returns:
        Inferred identifier as string, or an empty string when inference is not
        possible.
    """

    game = _first_game(match)
    if game is None:
        return ""

    game_t1 = str(game.get("t1_id") or "")
    game_t2 = str(game.get("t2_id") or "")
    known = str(known_id or "")
    if known and known == game_t1:
        return game_t2
    if known and known == game_t2:
        return game_t1
    return game_t1 if missing_side == 1 else game_t2


def _game_name_for_team_id(match: dict[str, Any], target_team_id: str) -> str | None:
    """Return first-game display name for a team identifier."""

    target = str(target_team_id or "")
    for game in games(match):
        if str(game.get("t1_id") or "") == target:
            return str(game.get("t1_name") or "")
        if str(game.get("t2_id") or "") == target:
            return str(game.get("t2_name") or "")
    return None


def match_tournament(match: dict[str, Any]) -> Any:
    """Return tournament name from old or new GOL.GG schema."""

    return first_present(match, ("tournament", "tournament_name"))


def team1_name(match: dict[str, Any]) -> Any:
    """Return team-1 display name from old or new GOL.GG schema."""

    inferred_name = _game_name_for_team_id(match, team1_id(match))
    return inferred_name or first_present(match, ("name_1", "sname_t1"))


def team2_name(match: dict[str, Any]) -> Any:
    """Return team-2 display name from old or new GOL.GG schema."""

    inferred_name = _game_name_for_team_id(match, team2_id(match))
    return inferred_name or first_present(match, ("name_2", "sname_t2"))


def team1_id(match: dict[str, Any]) -> str:
    """Return team-1 identifier from old or new GOL.GG schema."""

    value = first_present(match, ("tid_1", "t1_id"), "")
    if value:
        return str(value)
    return _infer_missing_team_id(
        match,
        missing_side=1,
        known_id=first_present(match, ("tid_2", "t2_id")),
    )


def team2_id(match: dict[str, Any]) -> str:
    """Return team-2 identifier from old or new GOL.GG schema."""

    value = first_present(match, ("tid_2", "t2_id"), "")
    if value:
        return str(value)
    return _infer_missing_team_id(
        match,
        missing_side=2,
        known_id=first_present(match, ("tid_1", "t1_id")),
    )


def score1(match: dict[str, Any]) -> int:
    """Return team-1 match score from old or new GOL.GG schema."""

    adv = initial_wins(match).get(team1_id(match), 0)
    match_games = games(match)
    if match_games:
        return int(sum(game_score_for_match_team1(match, game) for game in match_games)) + adv
    return int(first_present(match, ("score_1", "t1_score"), 0) or 0) + adv


def score2(match: dict[str, Any]) -> int:
    """Return team-2 match score from old or new GOL.GG schema."""

    adv = initial_wins(match).get(team2_id(match), 0)
    match_games = games(match)
    if match_games:
        played_1 = sum(game_score_for_match_team1(match, game) for game in match_games)
        return int(len(match_games) - played_1) + adv
    return int(first_present(match, ("score_2", "t2_score"), 0) or 0) + adv

def best_of(match: dict[str, Any]) -> int | None:
    """Return declared best-of value from old or new GOL.GG schema."""

    value = first_present(match, ("BoN", "best_of"))
    return int(value) if value is not None else None


def initial_wins(match: dict[str, Any]) -> dict[str, int]:
    """Return map advantage / initial wins granted by stage rule before play."""

    value = match.get("initial_wins")
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items() if int(v) > 0}
    return {}

def games(match: dict[str, Any]) -> list[dict[str, Any]]:
    """Return list of per-game payloads from a match dictionary."""

    raw_games = match.get("games") or []
    if not isinstance(raw_games, list):
        return []
    return [game for game in raw_games if isinstance(game, dict)]


def _players_from_game_side(game: dict[str, Any], side_key: str) -> list[str]:
    """Extract player identifiers from a game side payload.

    Args:
        game: Per-game dictionary.
        side_key: Either ``t1_players`` or ``t2_players``.

    Returns:
        Player identifiers as strings. Player names are used as a fallback when
        identifiers are missing.
    """

    players_payload = game.get(side_key) or {}
    if not isinstance(players_payload, dict):
        return []

    player_ids: list[str] = []
    for player_data in players_payload.values():
        if not isinstance(player_data, dict):
            continue
        player_id = player_data.get("player_id") or player_data.get("id")
        player_name = player_data.get("player_name") or player_data.get("name")
        player_ids.append(str(player_id or player_name or ""))
    return [player_id for player_id in player_ids if player_id]


def players_for_team(match: dict[str, Any], target_team_id: str) -> list[str]:
    """Return roster for a target team using top-level or per-game schema.

    Args:
        match: Match dictionary from ``golgg_matches.json``.
        target_team_id: Team identifier whose roster should be returned.

    Returns:
        List of player identifiers as strings.
    """

    target = str(target_team_id)
    if target == team1_id(match):
        old_players = match.get("players_1")
    elif target == team2_id(match):
        old_players = match.get("players_2")
    else:
        old_players = None

    if isinstance(old_players, list) and old_players:
        return [str(player) for player in old_players]

    for game in games(match):
        if str(game.get("t1_id")) == target:
            players = _players_from_game_side(game, "t1_players")
        elif str(game.get("t2_id")) == target:
            players = _players_from_game_side(game, "t2_players")
        else:
            players = []
        if players:
            return players
    return []


def players1(match: dict[str, Any]) -> list[str]:
    """Return team-1 player identifiers from old or new GOL.GG schema."""

    return players_for_team(match, team1_id(match))


def players2(match: dict[str, Any]) -> list[str]:
    """Return team-2 player identifiers from old or new GOL.GG schema."""

    return players_for_team(match, team2_id(match))


def game_score_for_match_team1(match: dict[str, Any], game: dict[str, Any]) -> int:
    """Return binary game result from the perspective of match team 1.

    Args:
        match: Match-level payload.
        game: Per-game payload.

    Returns:
        ``1`` when match team 1 won the game, otherwise ``0``.
    """

    if str(game.get("t1_id")) == team1_id(match):
        return int(bool(game.get("t1_win")))
    if str(game.get("t2_id")) == team1_id(match):
        return int(bool(game.get("t2_win")))
    return int(bool(game.get("t1_win")))


def normalized_match(match: dict[str, Any]) -> dict[str, Any]:
    """Build a compact compatibility view of a match.

    Args:
        match: Match dictionary in any supported GOL.GG schema.

    Returns:
        Dictionary exposing the historical field names expected by older
        scripts, plus the original ``games`` payload.
    """

    return {
        "match_id": str(match.get("match_id")),
        "date": match.get("date"),
        "tournament": match_tournament(match),
        "name_1": team1_name(match),
        "name_2": team2_name(match),
        "tid_1": team1_id(match),
        "tid_2": team2_id(match),
        "score_1": score1(match),
        "score_2": score2(match),
        "t1_win": bool(match.get("t1_win", score1(match) > score2(match))),
        "t2_win": bool(match.get("t2_win", score2(match) > score1(match))),
        "draw": bool(match.get("draw", score1(match) == score2(match))),
        "BoN": best_of(match),
        "players_1": players1(match),
        "players_2": players2(match),
        "games": games(match),
    }
