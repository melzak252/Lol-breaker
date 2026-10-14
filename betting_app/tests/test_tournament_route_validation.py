"""Invalid tournament state is a client error, not a fabricated forecast or HTTP 500."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from betting_app.api.routers import tournaments
from betting_app.services import tournament_service as tournament_core
from betting_app.services.tournament_service import (
    BracketMatchNode,
    TournamentBracket,
    TournamentSimulator,
)


@pytest.mark.parametrize("method,path", [
    ("GET", "/tournaments/invalid"),
    ("POST", "/tournaments/invalid/sync"),
    ("POST", "/tournaments/invalid/simulate"),
])
def test_invalid_bracket_returns_422_without_forecasts(monkeypatch, method, path):
    bracket = TournamentBracket(
        id="invalid", name="Invalid cycle", region="fixture", format="single_elimination",
        teams=["A", "B"],
        matches={"final": BracketMatchNode(
            id="final", name="Final", round_name="Final", bracket_section="final",
            best_of=3, team1="A", team2="B", next_match_winner_id="final",
        )},
    )
    simulator = TournamentSimulator()
    monkeypatch.setattr(tournaments, "SUPPORTED_BRACKETS", {"invalid": lambda: bracket})
    monkeypatch.setattr(tournaments, "TournamentSimulator", lambda: simulator)
    monkeypatch.setattr(tournaments, "LiquipediaBracketService", lambda: SimpleNamespace(
        sync_bracket=lambda *args, **kwargs: {"bracket": bracket, "source": "fixture"},
    ))
    app = FastAPI()
    app.include_router(tournaments.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.request(method, path, **({"json": {}} if method == "POST" else {}))
    assert response.status_code == 422
    assert "standings" not in response.json()


def test_unavailable_native_roster_source_returns_500_without_forecasts(monkeypatch):
    bracket = TournamentBracket(
        id="offline", name="Offline native roster source", region="fixture", format="single_elimination",
        teams=["Alpha", "Beta"],
        matches={"final": BracketMatchNode(
            id="final", name="Final", round_name="Final", bracket_section="final",
            best_of=3, team1="Alpha", team2="Beta",
        )},
    )

    def unavailable_native_source():
        raise ConnectionError("private-native-roster-diagnostic")

    monkeypatch.setattr(tournament_core, "canonical_team_key", str.lower)
    monkeypatch.setattr(tournament_core, "get_session", unavailable_native_source)
    monkeypatch.setattr(tournaments, "SUPPORTED_BRACKETS", {"offline": lambda: bracket})
    monkeypatch.setattr(tournaments, "LiquipediaBracketService", lambda: SimpleNamespace(
        sync_bracket=lambda *args, **kwargs: {"bracket": bracket, "source": "fixture"},
    ))
    app = FastAPI()
    app.include_router(tournaments.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/tournaments/offline")
    assert response.status_code == 500
    assert "standings" not in response.text
    assert "private-native-roster-diagnostic" not in response.text

@pytest.mark.parametrize("path", ["/tournaments/local/simulate", "/tournaments/local/recalculate"])
def test_simulation_does_not_collect_online_sources(monkeypatch, tmp_path, path):
    from datetime import UTC, datetime
    from unittest.mock import Mock
    from betting_app.services import c0_inference, tournament_cache_service
    from betting_app.services.liquipedia_bracket_service import LiquipediaBracketService
    import betting_app.services.liquipedia_bracket_service as bracket_source

    def builder():
        return TournamentBracket(
            id="local", name="Local", region="fixture", format="single_elimination",
            teams=["Alpha", "Beta"],
            matches={"final": BracketMatchNode(
                id="final", name="Final", round_name="Final", bracket_section="final",
                best_of=3, team1="Alpha", team2="Beta",
            )},
        )

    def simulator_factory(**kwargs):
        return TournamentSimulator(
            team_ids={"Alpha": "native-a", "Beta": "native-b"},
            team_rosters={"Alpha": [f"a{i}" for i in range(5)], "Beta": [f"b{i}" for i in range(5)]},
            decision_at=datetime(2026, 10, 5, tzinfo=UTC), **kwargs,
        )

    monkeypatch.setattr(tournament_core, "canonical_team_key", str.lower)
    monkeypatch.setattr(tournaments, "SUPPORTED_BRACKETS", {"local": builder})
    monkeypatch.setattr(tournament_cache_service, "SUPPORTED_BRACKETS", {"local": builder})
    monkeypatch.setattr(bracket_source, "SUPPORTED_BRACKETS", {"local": builder})
    monkeypatch.setattr(bracket_source, "canonical_team_key", str.lower)
    monkeypatch.setitem(bracket_source.TOURNAMENT_METADATA, "local", {
        "name": "Local", "fandom_overview": "Local", "liquipedia_page": "Local",
    })
    monkeypatch.setattr(tournaments, "TournamentSimulator", simulator_factory)
    monkeypatch.setattr(tournament_cache_service, "TournamentSimulator", simulator_factory)
    monkeypatch.setattr(c0_inference, "predict_c0", lambda _features: SimpleNamespace(prob_a=0.6, diagnostics={}))
    monkeypatch.setattr(tournament_cache_service, "get_c0_artifact_identity", lambda: {"artifact_sha256": "fixture"})
    monkeypatch.setattr(tournament_cache_service, "CACHE_DIR", tmp_path / "simulations")
    service_factory = lambda: LiquipediaBracketService(cache_dir=tmp_path / "brackets")
    monkeypatch.setattr(tournaments, "LiquipediaBracketService", service_factory)
    monkeypatch.setattr(tournament_cache_service, "LiquipediaBracketService", service_factory)
    network = Mock(side_effect=AssertionError("ordinary simulation attempted collection"))
    monkeypatch.setattr("urllib.request.urlopen", network)
    app = FastAPI()
    app.include_router(tournaments.router)
    with TestClient(app) as client:
        response = client.post(path, json={"simulations": 100})
        if path.endswith("/recalculate"):
            imported = client.post("/tournaments/local/sync", json={
                "source": "liquipedia",
                "raw_content": "{{Bracket|r1m1team1=Alpha|r1m1team2=Beta|r1m1score1=0|r1m1score2=2|r1m1win2=1\n}}",
            })
            assert imported.status_code == 200, imported.text
            refreshed = client.get("/tournaments/local?simulations=100")
            assert refreshed.status_code == 200, refreshed.text
            beta = next(row for row in refreshed.json()["standings"] if row["team"] == "Beta")
            assert beta["champion_prob"] == 1.0
    assert response.status_code == 200, response.text
    assert network.call_count == 0
    assert response.json()["simulations"] == 100
    assert sum(team["champion_prob"] for team in response.json()["standings"]) == 1.0
