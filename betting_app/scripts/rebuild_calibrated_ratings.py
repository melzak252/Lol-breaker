#!/usr/bin/env python3
"""Build the versioned player Glicko-2 rating snapshot with family calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from betting_app.core.db import connect, transaction  # noqa: E402
from betting_app.core.matching import normalize_team_name, similarity  # noqa: E402
from src.models.competition_tiers import (  # noqa: E402
    CompetitionIdentity,
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)
from src.ratings.family_calibrated_glicko2 import (  # noqa: E402
    FamilyCalibratedGlicko2,
    RatingEvent,
)

RATINGS_VERSION = "player-glicko2-family-v1"
RATING_SYSTEM = "gl2f"
DEFAULT_SOURCE = "exp075-successor"
UNKNOWN_AFFILIATION = "unknown"
SYSTEM_PARAMETERS: dict[str, Any] = {
    "glicko2": {
        "initial_rating": 1500.0,
        "initial_rd": 350.0,
        "initial_volatility": 0.06,
        "tau": 0.5,
        "convergence_tolerance": 1e-6,
    },
    "family_calibration": {
        "initial_tier_deviation": 100.0,
        "initial_family_deviation": 150.0,
        "bridge_process_deviation_per_30_days": 1.0,
        "family_location_constraint": "mean_total_family_offset_zero",
        "bridge_evidence": "one_majority_outcome_per_cross_family_event",
        "competition_prestige_weight": False,
    },
    "rating_period": "complete_calendar_date",
}


@dataclass(frozen=True)
class RosterPlayer:
    player_id: str
    display_name: str
    role: str | None


@dataclass(frozen=True)
class LoadedMatch:
    event_id: str
    event_date: date
    tournament_name: str
    competition: CompetitionIdentity
    team_a_id: str
    team_b_id: str
    team_a_name: str
    team_b_name: str
    players_a: tuple[RosterPlayer, ...]
    players_b: tuple[RosterPlayer, ...]
    scores: tuple[int, ...]


@dataclass(frozen=True)
class DomesticAffiliation:
    family: str
    tier: str
    source_date: str
    team_id: str

    def to_state(self) -> dict[str, str]:
        return {
            "family": self.family,
            "tier": self.tier,
            "source_date": self.source_date,
            "team_id": self.team_id,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "DomesticAffiliation":
        return cls(
            family=str(state["family"]),
            tier=str(state["tier"]),
            source_date=str(state["source_date"]),
            team_id=str(state["team_id"]),
        )


@dataclass
class RebuildMetadata:
    team_names: dict[str, str] = field(default_factory=dict)
    player_names: dict[str, str] = field(default_factory=dict)
    player_team_ids: dict[str, str] = field(default_factory=dict)
    player_roles: dict[str, str | None] = field(default_factory=dict)
    team_rosters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    team_affiliations: dict[str, DomesticAffiliation] = field(default_factory=dict)
    player_affiliations: dict[str, DomesticAffiliation] = field(default_factory=dict)
    team_games: Counter[str] = field(default_factory=Counter)
    player_games: Counter[str] = field(default_factory=Counter)
    team_last_activity: dict[str, str] = field(default_factory=dict)
    player_last_activity: dict[str, str] = field(default_factory=dict)
    matches_processed: int = 0
    games_processed: int = 0
    bridge_unknown_affiliation: int = 0

    def to_state(self) -> dict[str, Any]:
        return {
            "team_names": dict(sorted(self.team_names.items())),
            "player_names": dict(sorted(self.player_names.items())),
            "player_team_ids": dict(sorted(self.player_team_ids.items())),
            "player_roles": dict(sorted(self.player_roles.items())),
            "team_rosters": {
                key: list(self.team_rosters[key]) for key in sorted(self.team_rosters)
            },
            "team_affiliations": {
                key: self.team_affiliations[key].to_state()
                for key in sorted(self.team_affiliations)
            },
            "player_affiliations": {
                key: self.player_affiliations[key].to_state()
                for key in sorted(self.player_affiliations)
            },
            "team_games": {key: int(self.team_games[key]) for key in sorted(self.team_games)},
            "player_games": {
                key: int(self.player_games[key]) for key in sorted(self.player_games)
            },
            "team_last_activity": dict(sorted(self.team_last_activity.items())),
            "player_last_activity": dict(sorted(self.player_last_activity.items())),
            "matches_processed": int(self.matches_processed),
            "games_processed": int(self.games_processed),
            "bridge_unknown_affiliation": int(self.bridge_unknown_affiliation),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RebuildMetadata":
        def strings(name: str) -> dict[str, str]:
            value = state.get(name, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid calibrated rebuild metadata field {name!r}")
            return {str(key): str(item) for key, item in value.items()}

        roles_value = state.get("player_roles", {})
        rosters_value = state.get("team_rosters", {})
        team_affiliations_value = state.get("team_affiliations", {})
        player_affiliations_value = state.get("player_affiliations", {})
        if not all(
            isinstance(value, Mapping)
            for value in (
                roles_value,
                rosters_value,
                team_affiliations_value,
                player_affiliations_value,
            )
        ):
            raise ValueError("invalid calibrated rebuild metadata mappings")
        return cls(
            team_names=strings("team_names"),
            player_names=strings("player_names"),
            player_team_ids=strings("player_team_ids"),
            player_roles={
                str(key): None if value is None else str(value)
                for key, value in roles_value.items()
            },
            team_rosters={
                str(key): tuple(str(player_id) for player_id in value)
                for key, value in rosters_value.items()
            },
            team_affiliations={
                str(key): DomesticAffiliation.from_state(value)
                for key, value in team_affiliations_value.items()
            },
            player_affiliations={
                str(key): DomesticAffiliation.from_state(value)
                for key, value in player_affiliations_value.items()
            },
            team_games=Counter(
                {key: int(value) for key, value in strings("team_games").items()}
            ),
            player_games=Counter(
                {key: int(value) for key, value in strings("player_games").items()}
            ),
            team_last_activity=strings("team_last_activity"),
            player_last_activity=strings("player_last_activity"),
            matches_processed=int(state.get("matches_processed", 0)),
            games_processed=int(state.get("games_processed", 0)),
            bridge_unknown_affiliation=int(state.get("bridge_unknown_affiliation", 0)),
        )


@dataclass(frozen=True)
class _RunStart:
    run_id: int
    previous: dict[str, Any] | None


def stable_team_id(team_id: Any, team_name: Any) -> str:
    if team_id is not None and str(team_id).strip():
        return str(team_id).strip()
    normalized = normalize_team_name(str(team_name or ""))
    if not normalized:
        raise ValueError("rating match has no usable team identifier or name")
    return f"name:{normalized}"


def _natural_id_key(value: str) -> tuple[int, int | str, str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text), text)
    return (1, text, text)


def _query_filter(from_date: str | None, until_date: str | None, alias: str) -> tuple[str, list[Any]]:
    calendar_date = f"SUBSTR({alias}.date, 1, 10)"
    clauses = [f"COALESCE({alias}.draw, 0) = 0", f"{alias}.date IS NOT NULL"]
    params: list[Any] = []
    if from_date is not None:
        clauses.append(f"{calendar_date} >= ?")
        params.append(from_date)
    if until_date is not None:
        clauses.append(f"{calendar_date} <= ?")
        params.append(until_date)
    return " AND ".join(clauses), params


def load_matches(*, from_date: str | None = None, until_date: str | None = None) -> list[LoadedMatch]:
    """Load complete non-draw series and their first-game rosters."""

    match_filter, params = _query_filter(from_date, until_date, "m")
    with connect() as connection:
        match_rows = connection.execute(
            f"""
            SELECT m.match_id, m.date, m.tournament_name,
                   m.team1_id, m.team2_id, m.team1_name, m.team2_name
            FROM golgg_matches m
            WHERE {match_filter}
            """,
            params,
        ).fetchall()
        game_rows = connection.execute(
            f"""
            SELECT g.game_id, g.match_id, g.team1_id, g.team2_id,
                   g.team1_name, g.team2_name, g.team1_win, g.team2_win, g.draw
            FROM golgg_games g
            JOIN golgg_matches m ON m.match_id = g.match_id
            WHERE {match_filter}
            """,
            params,
        ).fetchall()
        roster_rows = connection.execute(
            f"""
            SELECT g.match_id, gp.game_id, gp.team_id, gp.team_name,
                   gp.player_id, gp.player_name, gp.role
            FROM golgg_game_players gp
            JOIN golgg_games g ON g.game_id = gp.game_id
            JOIN golgg_matches m ON m.match_id = g.match_id
            WHERE {match_filter}
              AND NOT EXISTS (
                    SELECT 1
                    FROM golgg_games earlier
                    WHERE earlier.match_id = g.match_id
                      AND (
                            LENGTH(earlier.game_id) < LENGTH(g.game_id)
                         OR (LENGTH(earlier.game_id) = LENGTH(g.game_id)
                             AND earlier.game_id < g.game_id)
                      )
              )
            """,
            params,
        ).fetchall()

    games_by_match: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in game_rows:
        games_by_match[str(row["match_id"])].append(dict(row))
    for games in games_by_match.values():
        games.sort(key=lambda item: _natural_id_key(str(item["game_id"])))

    rosters: defaultdict[str, list[tuple[str, str, RosterPlayer]]] = defaultdict(list)
    for row in roster_rows:
        player_id = str(row.get("player_id") or row.get("player_name") or "").strip()
        source_team_id = str(row.get("team_id") or "").strip()
        normalized_team_name = normalize_team_name(str(row.get("team_name") or ""))
        if not player_id or (not source_team_id and not normalized_team_name):
            continue
        rosters[str(row["match_id"])].append(
            (
                source_team_id,
                normalized_team_name,
                RosterPlayer(
                    player_id=player_id,
                    display_name=str(row.get("player_name") or player_id),
                    role=None if row.get("role") is None else str(row["role"]),
                ),
            )
        )

    matches: list[LoadedMatch] = []
    for row in match_rows:
        event_id = str(row["match_id"])
        event_date = date.fromisoformat(str(row["date"])[:10])
        competition = classify_competition(row.get("tournament_name"), event_date)
        if (
            competition.family == UNKNOWN_AFFILIATION
            or competition.tier is CompetitionTier.UNKNOWN
            or competition.scope is CompetitionScope.UNKNOWN
        ):
            raise ValueError(
                "unknown competition classification for "
                f"match {event_id}: {row.get('tournament_name')!r} "
                f"({competition.matched_rule})"
            )

        team_a_id = stable_team_id(row.get("team1_id"), row.get("team1_name"))
        team_b_id = stable_team_id(row.get("team2_id"), row.get("team2_name"))
        if team_a_id == team_b_id:
            raise ValueError(f"rating match {event_id} has the same team on both sides")
        team_a_name = str(row.get("team1_name") or team_a_id)
        team_b_name = str(row.get("team2_name") or team_b_id)
        players_a = _roster_for_side(
            rosters.get(event_id, ()), team_a_id, team_a_name
        )
        players_b = _roster_for_side(
            rosters.get(event_id, ()), team_b_id, team_b_name
        )
        if (not players_a or not players_b) and games_by_match.get(event_id):
            g0 = games_by_match[event_id][0]
            gt1_name = str(g0.get("team1_name") or "")
            gt2_name = str(g0.get("team2_name") or "")
            gt1_id = str(g0.get("team1_id") or "")
            gt2_id = str(g0.get("team2_id") or "")
            if gt1_id and gt2_id:
                s_same = similarity(team_a_name, gt1_name) + similarity(team_b_name, gt2_name)
                s_swap = similarity(team_a_name, gt2_name) + similarity(team_b_name, gt1_name)
                if s_same >= s_swap:
                    alt_a_id, alt_a_name = stable_team_id(gt1_id, gt1_name), gt1_name
                    alt_b_id, alt_b_name = stable_team_id(gt2_id, gt2_name), gt2_name
                else:
                    alt_a_id, alt_a_name = stable_team_id(gt2_id, gt2_name), gt2_name
                    alt_b_id, alt_b_name = stable_team_id(gt1_id, gt1_name), gt1_name
                pa_alt = _roster_for_side(rosters.get(event_id, ()), alt_a_id, alt_a_name)
                pb_alt = _roster_for_side(rosters.get(event_id, ()), alt_b_id, alt_b_name)
                if pa_alt and pb_alt:
                    team_a_id, team_a_name, players_a = alt_a_id, alt_a_name, pa_alt
                    team_b_id, team_b_name, players_b = alt_b_id, alt_b_name, pb_alt
        if not players_a or not players_b:
            continue
        scores = _scores_for_match(
            event_id,
            team_a_id,
            team_a_name,
            games_by_match.get(event_id, ()),
        )
        if not scores or sum(scores) * 2 == len(scores):
            continue
        matches.append(
            LoadedMatch(
                event_id=event_id,
                event_date=event_date,
                tournament_name=str(row.get("tournament_name") or ""),
                competition=competition,
                team_a_id=team_a_id,
                team_b_id=team_b_id,
                team_a_name=team_a_name,
                team_b_name=team_b_name,
                players_a=players_a,
                players_b=players_b,
                scores=scores,
            )
        )
    matches.sort(key=lambda item: (item.event_date, _natural_id_key(item.event_id)))
    return matches


def _roster_for_side(
    roster: Sequence[tuple[str, str, RosterPlayer]],
    team_id: str,
    team_name: str,
) -> tuple[RosterPlayer, ...]:
    normalized_team_name = normalize_team_name(team_name)
    return _deduplicate_roster(
        tuple(
            player
            for source_team_id, source_team_name, player in roster
            if source_team_id == team_id
            or (normalized_team_name and source_team_name == normalized_team_name)
        )
    )


def _deduplicate_roster(players: Sequence[RosterPlayer]) -> tuple[RosterPlayer, ...]:
    by_id: dict[str, RosterPlayer] = {}
    for player in sorted(players, key=lambda item: (item.player_id, item.display_name, item.role or "")):
        by_id.setdefault(player.player_id, player)
    return tuple(by_id[player_id] for player_id in sorted(by_id))


def _scores_for_match(
    event_id: str,
    team_a_id: str,
    team_a_name: str,
    games: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    scores: list[int] = []
    normalized_a = normalize_team_name(team_a_name)
    for game in games:
        if bool(game.get("draw")):
            return ()
        team1_win = game.get("team1_win")
        team2_win = game.get("team2_win")
        if team1_win is None or team2_win is None or bool(team1_win) == bool(team2_win):
            return ()
        game_team1 = stable_team_id(game.get("team1_id"), game.get("team1_name"))
        game_team2 = stable_team_id(game.get("team2_id"), game.get("team2_name"))
        if game_team1 == team_a_id or normalize_team_name(str(game.get("team1_name") or "")) == normalized_a:
            scores.append(int(bool(team1_win)))
        elif game_team2 == team_a_id or normalize_team_name(str(game.get("team2_name") or "")) == normalized_a:
            scores.append(int(bool(team2_win)))
        else:
            raise ValueError(f"game {game.get('game_id')} is not aligned with match {event_id}")
    return tuple(scores)


def process_matches(
    engine: FamilyCalibratedGlicko2,
    metadata: RebuildMetadata,
    matches: Sequence[LoadedMatch],
) -> dict[str, Any] | None:
    """Apply complete calendar-date periods and checkpoint before the final one."""

    periods = [
        tuple(group) for _, group in groupby(matches, key=lambda item: item.event_date)
    ]
    checkpoint: dict[str, Any] | None = None
    for period_index, period in enumerate(periods):
        if period_index == len(periods) - 1:
            checkpoint = {
                "before_date": period[0].event_date.isoformat(),
                "state": engine.to_state(),
                "metadata": metadata.to_state(),
            }
        team_candidates = _domestic_team_candidates(period)
        events: list[RatingEvent] = []
        for match in period:
            if match.competition.scope is CompetitionScope.DOMESTIC:
                affiliation_a = _single_affiliation(team_candidates.get(match.team_a_id, ()))
                affiliation_b = _single_affiliation(team_candidates.get(match.team_b_id, ()))
            else:
                affiliation_a = metadata.team_affiliations.get(match.team_a_id)
                affiliation_b = metadata.team_affiliations.get(match.team_b_id)
                if affiliation_a is None or affiliation_b is None:
                    metadata.bridge_unknown_affiliation += 1
                    affiliation_a = None
                    affiliation_b = None
            events.append(
                RatingEvent(
                    event_id=match.event_id,
                    event_date=match.event_date,
                    team_a_id=match.team_a_id,
                    team_b_id=match.team_b_id,
                    players_a=tuple(player.player_id for player in match.players_a),
                    players_b=tuple(player.player_id for player in match.players_b),
                    family_a=(affiliation_a.family if affiliation_a else UNKNOWN_AFFILIATION),
                    family_b=(affiliation_b.family if affiliation_b else UNKNOWN_AFFILIATION),
                    tier_a=(affiliation_a.tier if affiliation_a else CompetitionTier.UNKNOWN.value),
                    tier_b=(affiliation_b.tier if affiliation_b else CompetitionTier.UNKNOWN.value),
                    scores=match.scores,
                )
            )
        engine.process_period(events)
        _apply_period_metadata(metadata, period, team_candidates)
    return checkpoint


def _domestic_team_candidates(
    matches: Sequence[LoadedMatch],
) -> defaultdict[str, list[DomesticAffiliation]]:
    candidates: defaultdict[str, list[DomesticAffiliation]] = defaultdict(list)
    for match in matches:
        if match.competition.scope is not CompetitionScope.DOMESTIC:
            continue
        for team_id in (match.team_a_id, match.team_b_id):
            candidates[team_id].append(
                DomesticAffiliation(
                    family=match.competition.family,
                    tier=match.competition.tier.value,
                    source_date=match.event_date.isoformat(),
                    team_id=team_id,
                )
            )
    return candidates


def _single_affiliation(
    candidates: Sequence[DomesticAffiliation],
) -> DomesticAffiliation | None:
    destinations = {(item.family, item.tier) for item in candidates}
    if len(destinations) != 1:
        return None
    return sorted(candidates, key=lambda item: (item.family, item.tier, item.team_id))[0]


def _apply_period_metadata(
    metadata: RebuildMetadata,
    matches: Sequence[LoadedMatch],
    team_candidates: Mapping[str, Sequence[DomesticAffiliation]],
) -> None:
    player_domestic: defaultdict[str, list[DomesticAffiliation]] = defaultdict(list)
    team_observations: defaultdict[str, list[tuple[tuple[int, int | str, str], str, tuple[RosterPlayer, ...]]]] = defaultdict(list)
    player_observations: defaultdict[str, list[tuple[tuple[int, int | str, str], str, str, str | None]]] = defaultdict(list)

    for match in matches:
        match_key = _natural_id_key(match.event_id)
        for team_id, team_name, players in (
            (match.team_a_id, match.team_a_name, match.players_a),
            (match.team_b_id, match.team_b_name, match.players_b),
        ):
            team_observations[team_id].append((match_key, team_name, players))
            metadata.team_games[team_id] += len(match.scores)
            metadata.team_last_activity[team_id] = match.event_date.isoformat()
            for player in players:
                player_observations[player.player_id].append(
                    (match_key, player.display_name, team_id, player.role)
                )
                metadata.player_games[player.player_id] += len(match.scores)
                metadata.player_last_activity[player.player_id] = match.event_date.isoformat()
                if match.competition.scope is CompetitionScope.DOMESTIC:
                    player_domestic[player.player_id].append(
                        DomesticAffiliation(
                            family=match.competition.family,
                            tier=match.competition.tier.value,
                            source_date=match.event_date.isoformat(),
                            team_id=team_id,
                        )
                    )

    for team_id, observations in sorted(team_observations.items()):
        _, team_name, players = max(observations, key=lambda item: (item[0], item[1]))
        metadata.team_names[team_id] = team_name
        metadata.team_rosters[team_id] = tuple(player.player_id for player in players)
    for player_id, observations in sorted(player_observations.items()):
        _, display_name, team_id, role = max(
            observations, key=lambda item: (item[0], item[1], item[2], item[3] or "")
        )
        metadata.player_names[player_id] = display_name
        metadata.player_team_ids[player_id] = team_id
        metadata.player_roles[player_id] = role
    for team_id, candidates in sorted(team_candidates.items()):
        affiliation = _single_affiliation(candidates)
        if affiliation is not None:
            metadata.team_affiliations[team_id] = affiliation
    for player_id, candidates in sorted(player_domestic.items()):
        affiliation = _single_affiliation(candidates)
        if affiliation is not None:
            metadata.player_affiliations[player_id] = affiliation

    metadata.matches_processed += len(matches)
    metadata.games_processed += sum(len(match.scores) for match in matches)


def _system_payload(
    engine: FamilyCalibratedGlicko2,
    metadata: RebuildMetadata,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        RATING_SYSTEM: {
            "version": RATINGS_VERSION,
            "parameters": SYSTEM_PARAMETERS,
            "state": engine.to_state(),
            "metadata": metadata.to_state(),
            "checkpoint": dict(checkpoint),
        }
    }


def _load_incremental_state(
    version: str,
) -> tuple[FamilyCalibratedGlicko2, RebuildMetadata, str]:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT data_cutoff_at, systems_json
            FROM rating_runs
            WHERE ratings_version = ? AND status = 'completed'
            """,
            (version,),
        ).fetchone()
    if row is None:
        raise ValueError(f"no completed calibrated snapshot exists for incremental version {version!r}")
    if not row.get("data_cutoff_at"):
        raise ValueError(f"completed calibrated snapshot {version!r} has no cutoff")
    cutoff = str(row["data_cutoff_at"])[:10]
    try:
        payload = json.loads(str(row["systems_json"]))
        system = payload[RATING_SYSTEM]
        if system["version"] != RATINGS_VERSION:
            raise ValueError("unexpected calibrated engine version")
        checkpoint = system["checkpoint"]
        if checkpoint["before_date"] != cutoff:
            raise ValueError("calibrated checkpoint does not match snapshot cutoff")
        engine = FamilyCalibratedGlicko2.from_state(checkpoint["state"])
        metadata = RebuildMetadata.from_state(checkpoint["metadata"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid calibrated engine state for {version!r}") from error
    return engine, metadata, cutoff


def _start_run(version: str, source: str) -> _RunStart:
    with transaction() as connection:
        previous = connection.execute(
            """
            SELECT id, ratings_version, source, data_cutoff_at, started_at, finished_at,
                   status, systems_json, matches_processed, games_processed,
                   players_processed, error
            FROM rating_runs
            WHERE ratings_version = ?
            """,
            (version,),
        ).fetchone()
        if previous is not None:
            connection.execute(
                """
                UPDATE rating_runs
                SET source = ?, status = 'running', error = NULL,
                    started_at = CURRENT_TIMESTAMP, finished_at = NULL
                WHERE id = ?
                """,
                (source, previous["id"]),
            )
            return _RunStart(int(previous["id"]), dict(previous))
        inserted = connection.execute(
            """
            INSERT INTO rating_runs(ratings_version, source, status, systems_json, started_at)
            VALUES (?, ?, 'running', NULL, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (version, source),
        ).fetchone()
        if inserted is None:
            raise RuntimeError("rating run insertion did not return an id")
        return _RunStart(int(inserted["id"]), None)


def _record_failure(start: _RunStart, error: Exception) -> None:
    with transaction() as connection:
        if start.previous is not None and start.previous.get("status") == "completed":
            previous = start.previous
            connection.execute(
                """
                UPDATE rating_runs
                SET source = ?, data_cutoff_at = ?, started_at = ?, finished_at = ?,
                    status = ?, systems_json = ?, matches_processed = ?,
                    games_processed = ?, players_processed = ?, error = ?
                WHERE id = ?
                """,
                (
                    previous.get("source"),
                    previous.get("data_cutoff_at"),
                    previous.get("started_at"),
                    previous.get("finished_at"),
                    previous.get("status"),
                    previous.get("systems_json"),
                    previous.get("matches_processed", 0),
                    previous.get("games_processed", 0),
                    previous.get("players_processed", 0),
                    previous.get("error"),
                    start.run_id,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE rating_runs
                SET status = 'failed', finished_at = CURRENT_TIMESTAMP,
                    systems_json = NULL, error = ?
                WHERE id = ?
                """,
                (str(error), start.run_id),
            )


def _location_state(
    engine: FamilyCalibratedGlicko2,
    family_name: str,
    tier_name: str,
) -> tuple[float, float, float, float]:
    location = engine.get_family_location(family_name, tier_name)
    if family_name == tier_name == UNKNOWN_AFFILIATION:
        return 0.0, 0.0, float(location.mean), float(location.variance)
    family = engine.get_family_state(family_name)
    tier = engine.get_tier_state(tier_name)
    return (
        float(family.mean),
        float(tier.mean),
        float(location.mean),
        float(location.variance),
    )


def materialize_entity_rows(
    *,
    engine: FamilyCalibratedGlicko2,
    metadata: RebuildMetadata,
    run_id: int,
    version: str,
    snapshot_at: str,
    rating_system: str = RATING_SYSTEM,
    competition_calibration: str | None = None,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    player_ids = frozenset(engine.player_ids)
    for player_id in sorted(player_ids):
        engine_affiliation = engine.get_player_affiliation(player_id)
        family, tier = engine_affiliation or (
            UNKNOWN_AFFILIATION,
            CompetitionTier.UNKNOWN.value,
        )
        domestic_provenance = metadata.player_affiliations.get(player_id)
        ranking = engine.get_player_ranking(player_id)
        family_residual, tier_offset, offset, location_variance = _location_state(
            engine, family, tier
        )
        team_id = metadata.player_team_ids.get(player_id)
        team_name = metadata.team_names.get(team_id, team_id) if team_id else None
        state = {
            "player_id": player_id,
            "raw_rating": float(ranking.raw_rating),
            "raw_rd": float(ranking.raw_rd),
            "volatility": float(ranking.volatility),
            "family": family,
            "tier": tier,
            "family_residual": float(family_residual),
            "tier_offset": float(tier_offset),
            "offset": float(offset),
            "location_variance": float(location_variance),
            "domestic_affiliation": (
                domestic_provenance.to_state() if domestic_provenance else None
            ),
            "team_id": team_id,
            "last_activity": metadata.player_last_activity.get(player_id),
        }
        if competition_calibration:
            state["competition_calibration"] = competition_calibration
        rows.append(
            (
                run_id,
                version,
                snapshot_at,
                "player",
                metadata.player_names.get(player_id, player_id),
                player_id,
                team_name,
                metadata.player_roles.get(player_id),
                rating_system,
                float(ranking.rating),
                float(ranking.rd),
                float(ranking.volatility),
                int(metadata.player_games[player_id]),
                metadata.player_last_activity.get(player_id),
                json.dumps(state, ensure_ascii=False, sort_keys=True),
            )
        )
    team_rows: dict[str, tuple[Any, ...]] = {}
    team_priorities: dict[str, tuple[str, str]] = {}


    for team_id in sorted(metadata.team_rosters):
        roster = tuple(
            player_id
            for player_id in metadata.team_rosters[team_id]
            if player_id in player_ids
        )
        if not roster:
            continue
        projected_players = [
            engine.get_player_ranking(player_id) for player_id in roster
        ]
        raw_rating = (
            sum(float(item.raw_rating) for item in projected_players)
            / len(projected_players)
        )
        raw_variance = (
            sum(float(item.raw_rd) ** 2 for item in projected_players)
            / len(projected_players) ** 2
        )
        affiliation = metadata.team_affiliations.get(team_id)
        family = affiliation.family if affiliation else UNKNOWN_AFFILIATION
        tier = affiliation.tier if affiliation else CompetitionTier.UNKNOWN.value
        family_residual, tier_offset, offset, location_variance = _location_state(
            engine, family, tier
        )
        rating_value = raw_rating + offset
        rd = math.sqrt(raw_variance + location_variance)
        volatility = (
            sum(float(item.volatility) for item in projected_players)
            / len(projected_players)
        )
        team_name = metadata.team_names.get(team_id, team_id)
        normalized_team_name = normalize_team_name(team_name)
        state = {
            "team_id": team_id,
            "roster": list(roster),
            "raw_rating": float(raw_rating),
            "raw_rd": float(math.sqrt(raw_variance)),
            "volatility": float(volatility),
            "family": family,
            "tier": tier,
            "family_residual": float(family_residual),
            "tier_offset": float(tier_offset),
            "offset": float(offset),
            "location_variance": float(location_variance),
            "last_activity": metadata.team_last_activity.get(team_id),
        }
        if competition_calibration:
            state["competition_calibration"] = competition_calibration
        team_row = (
            run_id,
            version,
            snapshot_at,
            "team",
            team_name,
            normalized_team_name,
            team_name,
            None,
            rating_system,
            float(rating_value),
            float(rd),
            float(volatility),
            int(metadata.team_games[team_id]),
            metadata.team_last_activity.get(team_id),
            json.dumps(state, ensure_ascii=False, sort_keys=True),
        )
        priority = (metadata.team_last_activity.get(team_id, ""), team_id)
        if priority > team_priorities.get(normalized_team_name, ("", "")):
            team_priorities[normalized_team_name] = priority
            team_rows[normalized_team_name] = team_row
    rows.extend(team_rows[key] for key in sorted(team_rows))
    return rows


def _commit_snapshot(
    *,
    start: _RunStart,
    version: str,
    source: str,
    cutoff: str | None,
    rows: Sequence[tuple[Any, ...]],
    systems_json: str,
    metadata: RebuildMetadata,
    players_processed: int,
) -> None:
    insert_sql = """
        INSERT INTO entity_ratings(
            rating_run_id, ratings_version, snapshot_at, entity_type, entity_name,
            normalized_entity_name, team_name, role, rating_system, rating_value,
            rd, sigma, games_played, last_match_at, state_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with transaction() as connection:
        connection.execute("DELETE FROM entity_ratings WHERE ratings_version = ?", (version,))
        connection.executemany(insert_sql, list(rows))
        connection.execute(
            """
            UPDATE rating_runs
            SET source = ?, data_cutoff_at = ?, finished_at = CURRENT_TIMESTAMP,
                status = 'completed', systems_json = ?, matches_processed = ?,
                games_processed = ?, players_processed = ?, error = NULL
            WHERE id = ?
            """,
            (
                source,
                cutoff,
                systems_json,
                metadata.matches_processed,
                metadata.games_processed,
                players_processed,
                start.run_id,
            ),
        )


def rebuild_calibrated_ratings(
    *,
    version: str = RATINGS_VERSION,
    source: str = DEFAULT_SOURCE,
    mode: str = "full",
    until_date: str | None = None,
) -> dict[str, Any]:
    """Compute and atomically replace one calibrated-rating version snapshot."""

    if mode not in {"full", "incremental"}:
        raise ValueError(f"unsupported calibrated rebuild mode {mode!r}")
    if until_date is not None:
        date.fromisoformat(until_date)

    if mode == "incremental":
        engine, metadata, previous_cutoff = _load_incremental_state(version)
        from_date = previous_cutoff
    else:
        engine = FamilyCalibratedGlicko2()
        metadata = RebuildMetadata()
        from_date = None

    start = _start_run(version, source)
    try:
        matches = load_matches(from_date=from_date, until_date=until_date)
        if not matches:
            raise ValueError(f"{mode} calibrated rebuild found no eligible matches")
        checkpoint = process_matches(engine, metadata, matches)
        if checkpoint is None:
            raise RuntimeError("calibrated rebuild did not produce a cutoff checkpoint")
        cutoff = max(match.event_date for match in matches).isoformat()
        rows = materialize_entity_rows(
            engine=engine,
            metadata=metadata,
            run_id=start.run_id,
            version=version,
            snapshot_at=cutoff,
        )
        systems_json = json.dumps(
            _system_payload(engine, metadata, checkpoint),
            ensure_ascii=False,
            sort_keys=True,
        )
        _commit_snapshot(
            start=start,
            version=version,
            source=source,
            cutoff=cutoff,
            rows=rows,
            systems_json=systems_json,
            metadata=metadata,
            players_processed=len(tuple(engine.player_ids)),
        )
    except Exception as error:
        _record_failure(start, error)
        raise

    return {
        "version": version,
        "mode": mode,
        "matches": len(matches),
        "matches_total": metadata.matches_processed,
        "games": sum(len(match.scores) for match in matches),
        "games_total": metadata.games_processed,
        "players": len(tuple(engine.player_ids)),
        "entities": len(rows),
        "rows": len(rows),
        "data_cutoff_at": cutoff,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "incremental"), default="full")
    parser.add_argument("--until", dest="until_date", help="Include complete dates through YYYY-MM-DD.")
    parser.add_argument("--ratings-version", default=RATINGS_VERSION)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)

    stats = rebuild_calibrated_ratings(
        version=args.ratings_version,
        source=args.source,
        mode=args.mode,
        until_date=args.until_date,
    )
    print(
        f"{'Updated' if args.mode == 'incremental' else 'Rebuilt'} calibrated ratings:",
        f"version={stats['version']}",
        f"matches={stats['matches']}",
        f"games={stats['games']}",
        f"players={stats['players']}",
        f"rows={stats['rows']}",
        f"cutoff={stats['data_cutoff_at']}",
    )


if __name__ == "__main__":
    main()
