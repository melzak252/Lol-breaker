from datetime import UTC, datetime, timedelta

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from betting_app.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_simulate_matchup_endpoint_basic(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    forwarded_ids = []
    def fake_build_features(match: dict, **kwargs):
        forwarded_ids.append((
            match.get("team_a_golgg_id"), match.get("team_b_golgg_id"),
            match.get("native_team_a_id"), match.get("native_team_b_id"),
        ))
        result = {
            "status": "ready_player",
            "features": {
                "mapping": {
                    "team_a_golgg_name": "T1",
                    "team_b_golgg_name": "Gen.G",
                    "team_a_confidence": 1.0,
                    "team_b_confidence": 1.0,
                    "team_a_source": "exact",
                    "team_b_source": "exact",
                },
                "ratings": {
                    "probabilities": {system: 0.65 for system in ("elo", "gl", "ts", "os", "pl", "tm", "consensus")},
                    "team_a": {"gl": {"rating_value": 1850.0, "rd": 70.0}},
                    "team_b": {"gl": {"rating_value": 1700.0, "rd": 75.0}},
                },
                "player_ratings": {
                    "probabilities": {system: 0.68 for system in ("elo", "gl", "ts", "os", "pl", "tm", "consensus")},
                    "team_a_roster": {
                        "team_name": "T1",
                        "players": [{"player_id": "t1-top", "player_name": "Zeus", "role": "TOP"}],
                    },
                    "team_b_roster": {
                        "team_name": "Gen.G",
                        "players": [{"player_id": "geng-top", "player_name": "Kiin", "role": "TOP"}],
                    },
                    "team_a": {
                        "gl": {
                            "avg_rating_value": 1820.0,
                            "avg_rd": 65.0,
                            "max_rating_value": 1825.0,
                            "players_with_rating": 1,
                            "players": [{
                                "normalized_entity_name": "t1-top",
                                "entity_name": "ZEUS",
                                "rating_value": 1825.0,
                                "rd": 65.0,
                                "games_played": 20,
                            }],
                        }
                    },
                    "team_b": {"gl": {"avg_rating_value": 1710.0, "avg_rd": 68.0,
                                       "max_rating_value": 1710.0, "players": [{"rd": 68.0}]}},
                },
                "w20": {
                    "probability": 0.60,
                    "team_a": {"team_name": "T1", "win_rate": 0.75},
                    "team_b": {"team_name": "Gen.G", "win_rate": 0.65},
                },
            },
        }
        features = result["features"]
        team_a_id = match.get("native_team_a_id")
        team_b_id = match.get("native_team_b_id")
        features["c0_request"] = {
            "team1_id": str(team_a_id) if team_a_id is not None else "no_organization:target:T1",
            "team2_id": str(team_b_id) if team_b_id is not None else "no_organization:target:Gen.G",
            "roster_a": ["a1", "a2", "a3", "a4", "a5"],
            "roster_b": ["b1", "b2", "b3", "b4", "b5"],
            "best_of": match["best_of"],
            "decision_at": match["_c0_decision_at"],
            "start_at": match["start_time_normalized"],
            "mode": "full" if team_a_id is not None and team_b_id is not None else "no_organization",
        }
        for side, elo in (("team_a", 1800.0), ("team_b", 1700.0)):
            features["ratings"][side]["elo"] = {"rating_value": elo}
            features["player_ratings"][side]["elo"] = {"min_rating_value": elo - 100}
            for system, value, sigma in (("ts", 25.0, 3.0), ("os", 25.0, 3.0),
                                         ("pl", 0.0, 0.3), ("tm", 0.0, 0.3)):
                features["ratings"][side][system] = {"rating_value": value, "sigma": sigma}
                features["player_ratings"][side][system] = {"players": [{"sigma": sigma}]}
            features["w20"][side].update({
                "avg_kills": 15.0, "avg_deaths": 13.0, "avg_gd15": 0.0,
                "avg_dpm": 2000.0, "avg_vspm": 8.0, "avg_towers": 6.0,
                "avg_nashors": 1.0, "avg_gold": 60000.0, "avg_game_duration": 32.0,
            })
        return result

    monkeypatch.setattr("betting_app.services.upcoming_inference_service.build_features_for_match", fake_build_features)
    requests = []

    def fake_c0_predict(features, model_spec=None):
        requests.append(features["c0_request"])
        return SimpleNamespace(
            model_name="Causal-C0",
            model_version="c0-native-2026-w32-e12-v1",
            feature_version="native-c0-history16-v1",
            best_of=features["c0_request"]["best_of"],
            prob_a=0.67,
            prob_b=0.33,
            map_prob_a=None,
            map_prob_b=None,
            epistemic_sigma_z=None,
            p_low_a=None,
            p_low_b=None,
            diagnostics={"market_features_used": False, "side_symmetric": True},
        )

    monkeypatch.setattr(
        "betting_app.api.routers.matches.PredictionEngine.predict_from_features",
        fake_c0_predict,
    )

    # Test Bo1
    resp = client.post(
        "/matches/matchup",
        json={"team_a_name": "T1", "team_b_name": "Gen.G", "best_of": 1},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["team_a_name"] == "T1"
    assert data["team_b_name"] == "Gen.G"
    assert data["best_of"] == 1
    assert data["map_prob_a"] is None
    assert data["map_prob_b"] is None
    assert data["team_comparison"]["team_a"] == {
        "canonical_name": "T1",
        "golgg_name": "T1",
        "confidence": 1.0,
        "source": "exact",
    }
    roster_a_player = data["roster_a"]["players"][0]
    assert roster_a_player["player_id"] == "t1-top"
    assert roster_a_player["player_name"] == "Zeus"
    assert roster_a_player["role"] == "TOP"
    assert roster_a_player["glicko_rating"] == 1825.0
    assert roster_a_player["glicko_rd"] == 65.0
    assert roster_a_player["games_played"] == 20
    assert data["roster_b"]["players"][0]["player_name"] == "Kiin"
    assert data["roster_b"]["players"][0]["glicko_rating"] is None

    # Test Bo3
    resp3 = client.post(
        "/matches/matchup",
        json={"team_a_name": "T1", "team_b_name": "Gen.G", "best_of": 3},
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["best_of"] == 3
    assert data3["map_prob_a"] is None
    assert data3["map_prob_b"] is None
    assert 0.0 <= data3["series_prob_a"] <= 1.0
    assert data3["series_prob_a"] + data3["series_prob_b"] == pytest.approx(1.0)
    identified = client.post(
        "/matches/matchup",
        json={
            "team_a_name": "T1", "team_b_name": "Gen.G", "best_of": 3,
            "native_team_a_id": "2817", "native_team_b_id": "2818",
        },
    )
    assert identified.status_code == 200
    assert [(request["team1_id"], request["team2_id"], request["mode"]) for request in requests] == [
        ("no_organization:target:T1", "no_organization:target:Gen.G", "no_organization"),
        ("no_organization:target:T1", "no_organization:target:Gen.G", "no_organization"),
        ("2817", "2818", "full"),
    ]
    assert [request["best_of"] for request in requests] == [1, 3, 3]
    assert forwarded_ids == [
        (None, None, None, None), (None, None, None, None),
        (None, None, "2817", "2818"),
    ]
    assert all("odds" not in request for request in requests)
    decision_at = datetime.fromisoformat(requests[0]["decision_at"])
    start_at = datetime.fromisoformat(requests[0]["start_at"])
    assert start_at > decision_at


def test_native_c0_matchup_rejects_partial_rosters_before_engine(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    monkeypatch.setattr(
        "betting_app.services.upcoming_inference_service.build_features_for_match",
        lambda *args, **kwargs: {"status": "partial", "missing": ["c0_team_a_id"], "features": {}},
    )

    def forbidden_engine(*args, **kwargs):
        raise AssertionError("prediction engine must not run for invalid native C0 input")

    monkeypatch.setattr(
        "betting_app.api.routers.matches.PredictionEngine.predict_from_features",
        forbidden_engine,
    )
    response = client.post(
        "/matches/matchup",
        json={"team_a_name": "T1", "team_b_name": "Gen.G", "best_of": 3},
    )
    assert response.status_code == 422
    assert "c0_team_a_id" in response.json()["detail"]["missing"]


def test_synthetic_matchup_feature_build_does_not_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    from betting_app.services import upcoming_inference_service as inference

    persisted: list[dict] = []
    roster = {"players": [
        {"player_id": f"player-{index}", "player_name": f"Player {index}"}
        for index in range(5)
    ]}

    monkeypatch.setattr(inference, "golgg_name_from_id", lambda team_id: f"Team {team_id}")
    monkeypatch.setattr(
        inference, "native_golgg_team_id",
        lambda team_id, **kwargs: {1: "2817", 2: "2818"}.get(int(team_id)),
    )
    monkeypatch.setattr(inference, "load_team_ratings", lambda *_: {})
    monkeypatch.setattr(inference, "load_regional_adjustment", lambda *_: None)
    monkeypatch.setattr(inference, "rating_probabilities", lambda *_: {})
    monkeypatch.setattr(inference, "regional_adjustment_state", lambda *_: {})
    monkeypatch.setattr(inference, "load_w20", lambda *_: {"win_rate": 0.5})
    monkeypatch.setattr(inference, "load_last_roster", lambda *_: roster)
    monkeypatch.setattr(inference, "load_roster_player_ratings", lambda *_, **kwargs: {})
    monkeypatch.setattr(inference, "player_rating_probabilities", lambda *_: {})
    monkeypatch.setattr(inference, "latest_data_cutoff", lambda *_: None)
    monkeypatch.setattr(
        inference,
        "upsert_upcoming_features",
        lambda **kwargs: persisted.append(kwargs),
    )

    result = inference.build_features_for_match(
        {
            "id": 0,
            "team_a_name": "T1",
            "team_b_name": "Gen.G",
            "team_a_golgg_id": 1,
            "team_b_golgg_id": 2,
            "start_time_normalized": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "best_of": 3,
        },
        feature_version="test",
        ratings_version="test",
        w20_version="test",
        min_mapping_confidence=0.5,
        persist=False,
    )

    assert result["canonical_match_id"] == 0
    request = result["features"]["c0_request"]
    assert request["team1_id"] == "2817"
    assert request["team2_id"] == "2818"
    assert request["roster_a"] == [f"player-{index}" for index in range(5)]
    assert request["roster_b"] == [f"player-{index}" for index in range(5)]
    assert datetime.fromisoformat(request["decision_at"]).utcoffset() is not None
    assert datetime.fromisoformat(request["start_at"]).utcoffset() is not None
    assert request["best_of"] == 3
    assert "odds" not in request
    assert persisted == []


def test_player_ratings_preserve_roster_names_and_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    from betting_app.services import upcoming_inference_service as inference

    def fake_query_df(sql: str, params: tuple[object, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            [{
                "rating_system": "gl",
                "entity_name": "Zeus",
                "normalized_entity_name": "golgg-123",
                "rating_value": 1825.0,
                "rd": 65.0,
                "sigma": 0.06,
                "games_played": 20,
                "last_match_at": "2026-09-01T12:00:00+00:00",
            }]
        )

    monkeypatch.setattr(inference, "query_df", fake_query_df)

    ratings = inference.load_roster_player_ratings(
        {
            "players": [{
                "player_id": "Zeus",
                "player_name": "Choi Hyeon-jun",
                "role": "TOP",
            }]
        },
        "test",
    )


    assert ratings["gl"]["players"][0]["player_id"] == "Zeus"
    assert ratings["gl"]["players"][0]["player_name"] == "Choi Hyeon-jun"
    assert ratings["gl"]["players"][0]["role"] == "TOP"
