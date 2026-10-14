"""ENC 2027 roster selection and published-format Monte Carlo simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from betting_app.core.db import get_session
from betting_app.services.enc_rosters import (
    ENC_PARTICIPANTS,
    ENC_PUBLISHED_ROLE_PLAYERS,
    ENC_ROSTER_SOURCE_URL,
)
from betting_app.services.rating_contract import OPERATIONAL_RATINGS_VERSION
from betting_app.services.c0_inference import predict_c0

STANDARD_ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")

# Leaguepedia names that resolve to more than one current player-rating entity.
# These ids identify the player linked from the published national roster.
ROSTER_PLAYER_IDS = {
    "Knight": "1270",
    "Kaze": "5009",
    "Lost": "1123",
    "Neo": "1126",
}


@dataclass(frozen=True)
class EncTeam:
    nation: str
    entry_stage: str
    organization_id: str
    roster: tuple[str, ...]
    roster_rating: float | None = None

class EncConfigurationError(ValueError):
    """The published participant list cannot yet produce a valid simulation."""


def _rating_snapshot(session: Session) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    run = session.execute(
        text(
            """
            SELECT ratings_version, data_cutoff_at
            FROM rating_runs
            WHERE status = 'completed'
            ORDER BY
                CASE WHEN ratings_version = :operational_version THEN 0 ELSE 1 END,
                finished_at DESC NULLS LAST,
                id DESC
            LIMIT 1
            """
        ),
        {"operational_version": OPERATIONAL_RATINGS_VERSION},
    ).mappings().first()
    if run is None:
        return None, []

    rows = session.execute(
        text(
            """
            SELECT entity_name, normalized_entity_name, role, rating_value, games_played
            FROM entity_ratings
            WHERE ratings_version = :ratings_version
              AND entity_type = 'player'
              AND rating_system = 'gl'
              AND rating_value IS NOT NULL
            """
        ),
        {"ratings_version": run["ratings_version"]},
    ).mappings().all()
    return dict(run), [dict(row) for row in rows]


def _load_native_player_ids(session: Session | None) -> dict[str, str]:
    """Resolve published handles to unique source IDs in raw GOL.GG history."""
    if session is None:
        return {}
    names = sorted({
        name.casefold()
        for role_map in ENC_PUBLISHED_ROLE_PLAYERS.values()
        for players in role_map.values()
        for name in players
    })
    bind_names = {f"name_{index}": name for index, name in enumerate(names)}
    placeholders = ", ".join(f":{key}" for key in bind_names)
    rows = session.execute(
        text(
            f"""
            SELECT player_name, player_id
            FROM golgg_game_players
            WHERE lower(player_name) IN ({placeholders})
              AND player_id IS NOT NULL
              AND TRIM(player_id) != ''
            """
        ),
        bind_names,
    ).mappings().all()
    ids_by_name: dict[str, set[str]] = {}
    for row in rows:
        ids_by_name.setdefault(str(row["player_name"]).casefold(), set()).add(str(row["player_id"]))
    return {
        name: next(iter(player_ids))
        for name, player_ids in ids_by_name.items()
        if len(player_ids) == 1
    }


def _select_lineup(
    nation: str,
    ratings_by_name: Mapping[str, list[dict[str, Any]]],
    native_player_ids: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    role_players = ENC_PUBLISHED_ROLE_PLAYERS.get(nation, {})
    selected: list[dict[str, Any]] = []
    missing_roles: list[str] = []
    for role in STANDARD_ROLES:
        eligible_players = role_players.get(role, ())
        chosen = next(
            ((name, native_player_ids[name.casefold()]) for name in eligible_players if name.casefold() in native_player_ids),
            None,
        )
        if chosen is None:
            missing_roles.append(role)
            continue
        player_name, source_player_id = chosen
        rating_rows = ratings_by_name.get(player_name.casefold(), [])
        rating_row = next(
            (
                row for row in rating_rows
                if str(row.get("normalized_entity_name")) == ROSTER_PLAYER_IDS.get(player_name, source_player_id)
            ),
            None,
        )
        selected.append(
            {
                "role": role,
                "player": player_name,
                "normalized_player_id": source_player_id,
                "rating": round(float(rating_row["rating_value"]), 1) if rating_row else None,
                "games_played": int(rating_row.get("games_played") or 0) if rating_row else 0,
                "rating_source": "gl" if rating_row else None,
            }
        )
    return selected, missing_roles


def build_enc_configuration(
    *,
    session: Session | None = None,
    rating_run: Mapping[str, Any] | None = None,
    rating_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build published-roster metadata without inventing missing C0 identities."""
    if rating_rows is None:
        if session is None:
            with get_session() as owned_session:
                return build_enc_configuration(session=owned_session)
        loaded_run, loaded_rows = _rating_snapshot(session)
        rating_run, rating_rows = loaded_run, loaded_rows

    ratings_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rating_rows or ():
        name = str(row.get("entity_name") or "")
        if name:
            ratings_by_name.setdefault(name.casefold(), []).append(dict(row))

    teams: list[dict[str, Any]] = []
    native_player_ids = _load_native_player_ids(session)
    incomplete_nations: list[str] = []
    for participant in ENC_PARTICIPANTS:
        selected, missing_roles = _select_lineup(participant["nation"], ratings_by_name, native_player_ids)
        ready = not missing_roles and len(selected) == 5
        if not ready:
            incomplete_nations.append(participant["nation"])
        teams.append(
            {
                "nation": participant["nation"],
                "entry_stage": participant["entry_stage"],
                "organization_id": f"national:{participant['nation'].casefold().replace(' ', '-')}",
                "ranking": participant["ranking"],
                "source_roster": list(participant["players"]),
                "selected_roster": selected,
                "roster_ids": [str(player["normalized_player_id"]) for player in selected],
                "missing_roles": missing_roles,
                "selection_status": "ready" if ready else "incomplete",
                "roster_rating": round(sum(player["rating"] for player in selected) / len(selected), 1)
                if ready and all(player.get("rating") is not None for player in selected)
                else None,
            }
        )

    teams.append(
        {
            "nation": "Solidarity Slot",
            "organization_id": "national:solidarity-slot",
            "entry_stage": "play_in",
            "ranking": None,
            "source_roster": [],
            "selected_roster": [],
            "roster_ids": [],
            "missing_roles": list(STANDARD_ROLES),
            "selection_status": "unavailable",
            "roster_rating": None,
        }
    )
    issues: list[str] = []
    if incomplete_nations:
        issues.append("Missing exact published-role C0 player IDs for: " + ", ".join(incomplete_nations) + ".")
    issues.append("Solidarity Slot has no published player roster or C0 player IDs.")
    return {
        "tournament_id": "enc_2027",
        "tournament_name": "Esports Nations Cup 2027",
        "source_url": ENC_ROSTER_SOURCE_URL,
        "format": enc_format(),
        "ratings_version": str(rating_run["ratings_version"]) if rating_run else None,
        "data_cutoff_at": str(rating_run["data_cutoff_at"]) if rating_run else None,
        "probability_model": "Causal-C0",
        "probability_model_version": "c0-native-2026-w32-e12-v1",
        "organization_policy": "synthetic_national_identity_no_organization",
        "teams": teams,
        "simulation_ready": not issues,
        "blocking_issues": issues,
    }


def enc_format() -> dict[str, Any]:
    """The published structure; draw and tie-break procedures remain unannounced."""
    return {
        "participants": 32,
        "invited": 16,
        "direct_group_stage": 8,
        "online_qualifiers": 14,
        "wildcards": ["Solidarity Slot", "Host GCC"],
        "play_in": "24 teams, 4 groups of 6, double round robin Bo1; top 2 advance",
        "group_stage": "16 teams, 4 groups of 4, single round robin Bo3; top 2 advance",
        "playoffs": "single elimination; quarterfinals and semifinals Bo3, final Bo5",
        "draw_and_tiebreak_policy": "Group draws, playoff bracket draws, and tied group positions are shuffled uniformly because the published page does not specify them.",
    }


class EncSimulator:
    """Simulate published ENC stages using native C0 series probabilities."""

    def __init__(self, teams: Sequence[EncTeam]):
        self.teams = tuple(teams)
        self._by_nation = {team.nation: team for team in teams}
        self.decision_at = datetime.now(timezone.utc)
        self.roster_basis = "explicit_scenario_roster"
        self._probabilities: dict[tuple[Any, ...], float] = {}
        self._native_diagnostics: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._validate_participants()

    @classmethod
    def from_configuration(cls, configuration: Mapping[str, Any]) -> "EncSimulator":
        if not configuration.get("simulation_ready"):
            issues = "; ".join(str(issue) for issue in configuration.get("blocking_issues", []))
            raise EncConfigurationError(issues or "ENC C0 configuration is unavailable.")
        simulator = cls([
            EncTeam(
                nation=str(team["nation"]),
                entry_stage=str(team["entry_stage"]),
                organization_id=str(team["organization_id"]),
                roster=tuple(str(player_id) for player_id in team["roster_ids"]),
                roster_rating=float(team["roster_rating"]) if team.get("roster_rating") is not None else None,
            )
            for team in configuration["teams"]
        ])
        simulator.roster_basis = "configured_role_selection"
        return simulator

    def _validate_participants(self) -> None:
        if len(self.teams) != 32:
            raise ValueError("ENC requires exactly 32 national teams.")
        if len({team.nation for team in self.teams}) != 32:
            raise ValueError("Each ENC nation must occupy exactly one slot.")
        if any(not team.organization_id or len(team.roster) != 5 or len(set(team.roster)) != 5 for team in self.teams):
            raise EncConfigurationError("Every ENC C0 participant requires a synthetic organization identity and five exact player IDs.")
        direct = [team for team in self.teams if team.entry_stage == "group_stage"]
        play_in = [team for team in self.teams if team.entry_stage == "play_in"]
        if len(direct) != 8 or len(play_in) != 24:
            raise ValueError("ENC requires 8 direct Group Stage teams and 24 Play-In teams.")

    def _winner(self, nation_a: str, nation_b: str, best_of: int) -> str:
        if type(best_of) is not int or best_of not in {1, 3, 5}:
            raise ValueError("ENC series best_of must be one of 1, 3, 5.")
        team_a, team_b = self._by_nation[nation_a], self._by_nation[nation_b]
        origin = self.decision_at.isoformat()
        key = (
            team_a.organization_id, team_a.roster, team_b.organization_id, team_b.roster,
            best_of, origin, "enc-2027", "no_organization", "c0-native-2026-w32-e12-v1",
        )
        if key not in self._probabilities:
            result = predict_c0({
                "canonical": {"id": None},
                "c0_request": {
                    "team1_id": team_a.organization_id,
                    "team2_id": team_b.organization_id,
                    "roster_a": list(team_a.roster),
                    "roster_b": list(team_b.roster),
                    "best_of": best_of,
                    "decision_at": origin,
                    "competition_context": "enc-2027",
                    "start_at": None,
                    "mode": "no_organization",
                },
            })
            probability = float(result.prob_a)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise EncConfigurationError("Native C0 returned an invalid ENC series probability.")
            self._native_diagnostics[key] = {
                "team1_id": team_a.organization_id,
                "team2_id": team_b.organization_id,
                "best_of": best_of,
                "roster_provenance": {
                    "basis": {nation_a: self.roster_basis, nation_b: self.roster_basis},
                    "submitted_roster_a": list(team_a.roster),
                    "submitted_roster_b": list(team_b.roster),
                    "published_at": None,
                    "authenticated_prior_announcement": False,
                },
                "diagnostics": dict(getattr(result, "diagnostics", {}) or {}),
            }
            self._probabilities[key] = probability
        return nation_a if random.random() < self._probabilities[key] else nation_b

    def _rank_group(self, teams: Sequence[str], *, best_of: int, repeat: int) -> list[str]:
        wins = {team: 0 for team in teams}
        for _ in range(repeat):
            for index, nation_a in enumerate(teams):
                for nation_b in teams[index + 1 :]:
                    wins[self._winner(nation_a, nation_b, best_of)] += 1
        shuffled = list(teams)
        random.shuffle(shuffled)
        return sorted(shuffled, key=lambda nation: wins[nation], reverse=True)

    def simulate(self, n_simulations: int) -> dict[str, Any]:
        if type(n_simulations) is not int or n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer.")
        direct = [team.nation for team in self.teams if team.entry_stage == "group_stage"]
        play_in = [team.nation for team in self.teams if team.entry_stage == "play_in"]
        group_stage_counts = {team.nation: n_simulations if team.nation in direct else 0 for team in self.teams}
        playoff_counts = {team.nation: 0 for team in self.teams}
        top4_counts = {team.nation: 0 for team in self.teams}
        top2_counts = {team.nation: 0 for team in self.teams}
        champion_counts = {team.nation: 0 for team in self.teams}

        for _ in range(n_simulations):
            shuffled_play_in = play_in[:]
            random.shuffle(shuffled_play_in)
            play_in_advancers: list[str] = []
            for index in range(0, 24, 6):
                play_in_advancers.extend(
                    self._rank_group(shuffled_play_in[index : index + 6], best_of=1, repeat=2)[:2]
                )
            for nation in play_in_advancers:
                group_stage_counts[nation] += 1

            group_stage_teams = [*direct, *play_in_advancers]
            random.shuffle(group_stage_teams)
            playoff_teams: list[str] = []
            for index in range(0, 16, 4):
                playoff_teams.extend(
                    self._rank_group(group_stage_teams[index : index + 4], best_of=3, repeat=1)[:2]
                )
            for nation in playoff_teams:
                playoff_counts[nation] += 1

            random.shuffle(playoff_teams)
            semifinalists = [
                self._winner(playoff_teams[index], playoff_teams[index + 1], best_of=3)
                for index in range(0, 8, 2)
            ]
            for nation in semifinalists:
                top4_counts[nation] += 1
            finalists = [
                self._winner(semifinalists[index], semifinalists[index + 1], best_of=3)
                for index in range(0, 4, 2)
            ]
            for nation in finalists:
                top2_counts[nation] += 1
            champion_counts[self._winner(finalists[0], finalists[1], best_of=5)] += 1

        standings = [
            {
                "nation": team.nation,
                "entry_stage": team.entry_stage,
                "roster_rating": round(team.roster_rating, 1) if team.roster_rating is not None else None,
                "group_stage_prob": round(group_stage_counts[team.nation] / n_simulations, 4),
                "playoff_prob": round(playoff_counts[team.nation] / n_simulations, 4),
                "top4_prob": round(top4_counts[team.nation] / n_simulations, 4),
                "top2_prob": round(top2_counts[team.nation] / n_simulations, 4),
                "champion_prob": round(champion_counts[team.nation] / n_simulations, 4),
            }
            for team in self.teams
        ]
        standings.sort(key=lambda team: team["champion_prob"], reverse=True)
        return {
            "tournament_id": "enc_2027",
            "tournament_name": "Esports Nations Cup 2027",
            "simulations": n_simulations,
            "format": enc_format(),
            "probability_model": "Causal-C0",
            "probability_model_version": "c0-native-2026-w32-e12-v1",
            "probability_unit": "series",
            "decision_at": self.decision_at.isoformat(),
            "mode": "no_organization",
            "organization_policy": "synthetic_national_identity_no_organization",
            "roster_basis": {team.nation: self.roster_basis for team in self.teams},
            "roster_publication_status": "unknown",
            "authenticated_prior_announcement": False,
            "eligibility_live": 0,
            "point_in_time_certified": False,
            "simulation_scope": {
                "mode": "hypothetical_configured_roster_scenario",
                "pre_event_forecast": False,
            },
            "native_prediction_provenance": list(self._native_diagnostics.values()),
            "standings": standings,
        }
