"""Tests for the manually configured Worlds Play-In, Swiss, and knockout simulator."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from betting_app.services import c0_inference
from betting_app.services.canonical_match_service import canonical_team_key
from betting_app.services.tournament_service import WorldsSimulator, WorldsTeam


def _native_identity_maps(teams: list[WorldsTeam]) -> tuple[dict[str, str], dict[str, list[str]]]:
    team_ids = {team.name: f"native-team:{canonical_team_key(team.name)}" for team in teams}
    rosters = {
        team.name: [f"native-player:{canonical_team_key(team.name)}:{role}" for role in range(5)]
        for team in teams
    }
    return team_ids, rosters


@pytest.fixture
def controlled_c0(monkeypatch):
    """Keep the real simulator/input path while controlling only the C0 inference boundary."""
    def predict(features):
        request = features["c0_request"]
        team1_id = request["team1_id"]
        team2_id = request["team2_id"]
        favorite_id = f"native-team:{canonical_team_key('Play-In Favorite')}"
        probability = 0.8 if team1_id == favorite_id else 0.2 if team2_id == favorite_id else 0.5
        return SimpleNamespace(prob_a=probability, diagnostics={})

    monkeypatch.setattr(c0_inference, "predict_c0", predict)


def _all_teams() -> list[WorldsTeam]:
    return [*_direct_teams(), *_play_in_teams()]


def _configured_simulator() -> WorldsSimulator:
    team_ids, team_rosters = _native_identity_maps(_all_teams())
    return WorldsSimulator(team_ids=team_ids, team_rosters=team_rosters, seed=17)


def _direct_teams() -> list[WorldsTeam]:
    pools = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    return [
        WorldsTeam(name=f"Swiss Team {index}", region="LCK" if index < 4 else "LEC", pool=pool)
        for index, pool in enumerate(pools, start=1)
    ]


def _play_in_teams() -> list[WorldsTeam]:
    return [
        WorldsTeam(name="Play-In Favorite", region="LPL"),
        WorldsTeam(name="Play-In Team 2", region="LCP"),
        WorldsTeam(name="Play-In Team 3", region="CBLOL"),
        WorldsTeam(name="Play-In Team 4", region="EMEA Masters"),
    ]


def test_worlds_simulator_runs_play_in_swiss_and_knockout(controlled_c0) -> None:
    simulator = _configured_simulator()

    result = simulator.simulate_worlds(
        direct_teams=_direct_teams(),
        play_in_teams=_play_in_teams(),
        play_in_winner_pool=4,
        n_simulations=200,
    )

    assert result["format"] == "play_in_double_elimination_bo5_swiss_and_knockout"
    assert len(result["standings"]) == 19
    assert round(sum(standing["champion_prob"] for standing in result["standings"]), 1) == 1.0
    assert round(sum(standing["top8_swiss_prob"] for standing in result["standings"]), 1) == 8.0

    standings = {standing["team"]: standing for standing in result["standings"]}
    assert standings["Swiss Team 1"]["play_in_qualifier_prob"] == 1.0
    assert standings["Play-In Favorite"]["play_in_qualifier_prob"] > 0.5


def test_worlds_api_requires_manual_participants(client: TestClient, monkeypatch, controlled_c0) -> None:
    team_ids, team_rosters = _native_identity_maps(_all_teams())
    monkeypatch.setattr(
        "betting_app.api.routers.tournaments.WorldsSimulator",
        lambda **kwargs: WorldsSimulator(
            team_ids=team_ids,
            team_rosters=team_rosters,
            seed=17,
            **kwargs,
        ),
    )
    direct_teams = [
        {"team": team.name, "region": team.region, "pool": team.pool}
        for team in _direct_teams()
    ]
    play_in_teams = [{"team": team.name, "region": team.region} for team in _play_in_teams()]

    missing_roster = client.post("/tournaments/worlds/simulate", json={"simulations": 100})
    assert missing_roster.status_code == 422

    response = client.post(
        "/tournaments/worlds/simulate",
        json={
            "simulations": 150,
            "direct_teams": direct_teams,
            "play_in_teams": play_in_teams,
            "play_in_winner_pool": 4,
        },
    )
    assert response.status_code == 200
    assert len(response.json()["standings"]) == 19


def test_worlds_rejects_an_unbalanced_swiss_pool(client: TestClient) -> None:
    direct_teams = [
        {"team": team.name, "region": team.region, "pool": 1}
        for team in _direct_teams()
    ]
    response = client.post(
        "/tournaments/worlds/simulate",
        json={
            "direct_teams": direct_teams,
            "play_in_teams": [{"team": team.name, "region": team.region} for team in _play_in_teams()],
            "play_in_winner_pool": 4,
        },
    )
    assert response.status_code == 422
    assert "Direct Swiss slots" in response.json()["detail"]
