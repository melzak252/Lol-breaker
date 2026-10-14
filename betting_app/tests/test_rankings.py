"""Behavioral tests for the team and player rankings API."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import text

from betting_app.core.db import get_session


def _seed_completed_rankings() -> None:
    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO rating_runs(
                    id, ratings_version, data_cutoff_at, started_at, finished_at,
                    status, matches_processed, games_processed, players_processed
                ) VALUES (
                    1, 'ratings-old', '2026-08-01T00:00:00+00:00',
                    '2026-08-01T00:00:00+00:00', '2026-08-01T00:05:00+00:00',
                    'completed', 20, 40, 100
                ), (
                    2, 'ratings-current', '2026-09-01T00:00:00+00:00',
                    '2026-09-01T00:00:00+00:00', '2026-09-01T00:05:00+00:00',
                    'completed', 30, 60, 150
                ), (
                    3, 'ratings-running', '2026-09-02T00:00:00+00:00',
                    '2026-09-02T00:00:00+00:00', NULL,
                    'running', 0, 0, 0
                )
                ON CONFLICT (id) DO NOTHING
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO entity_ratings(
                    rating_run_id, ratings_version, snapshot_at, entity_type,
                    entity_name, normalized_entity_name, team_name, role,
                    rating_system, rating_value, rd, sigma, games_played, last_match_at
                ) VALUES
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'Gen.G', 'gen g', NULL, NULL, 'elo', 1710.5, NULL, NULL, 45, '2026-08-31'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'T1', 't1', NULL, NULL, 'elo', 1690.25, NULL, NULL, 52, '2026-08-30'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'Bilibili Gaming', 'bilibili gaming', NULL, NULL, 'elo', 1650.0, NULL, NULL, 40, '2026-08-29'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'Rookie Team', 'rookie team', NULL, NULL, 'elo', 1800.0, NULL, NULL, 2, '2026-08-29'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'Gen.G', 'gen g', NULL, NULL, 'ts', 28.4, NULL, 4.1, 45, '2026-08-31'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'T1', 't1', NULL, NULL, 'ts', 29.0, NULL, 3.9, 52, '2026-08-30'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'Bilibili Gaming', 'bilibili gaming', NULL, NULL, 'ts', 27.0, NULL, 4.4, 40, '2026-08-29'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'Rookie Team', 'rookie team', NULL, NULL, 'ts', 31.0, NULL, 5.2, 2, '2026-08-29'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'SK Telecom T1', 'sk telecom t1', NULL, NULL, 'elo', 1900.0, NULL, NULL, 90, '2019-11-03'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'SK Telecom T1', 'sk telecom t1', NULL, NULL, 'ts', 32.0, NULL, 3.0, 90, '2019-11-03'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'T1 Esports Academy', 't1 academy', NULL, NULL, 'elo', 1850.0, NULL, NULL, 70, '2026-08-20'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'team',
                     'T1 Esports Academy', 't1 academy', NULL, NULL, 'ts', 31.0, NULL, 3.5, 70, '2026-08-20'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'player',
                     'Faker', 'faker-id', 'T1', 'MID', 'elo', 1640.0, NULL, NULL, 110, '2026-08-30'),
                    (2, 'ratings-current', '2026-09-01T00:05:00+00:00', 'player',
                     'Chovy', 'chovy-id', 'Gen.G', 'MID', 'elo', 1665.0, NULL, NULL, 105, '2026-08-31')
                """
            )
        )
        session.commit()


def test_rankings_empty_without_completed_run(client: TestClient) -> None:
    response = client.get("/rankings")

    assert response.status_code == 200
    assert response.json() == {
        "entity_type": "team",
        "rating_system": "unified",
        "ratings_version": None,
        "active_since": None,
        "squad_scope": "major",
        "data_cutoff_at": None,
        "snapshot_at": None,
        "total": 0,
        "available_rating_systems": [],
        "rankings": [],
    }


def test_team_rankings_use_latest_completed_snapshot_and_filters(client: TestClient) -> None:
    _seed_completed_rankings()

    response = client.get("/rankings", params={"entity_type": "team", "rating_system": "elo", "min_games": 10})

    assert response.status_code == 200
    data = response.json()
    assert data["ratings_version"] == "ratings-current"
    assert data["data_cutoff_at"] == "2026-09-01T00:00:00+00:00"
    assert data["available_rating_systems"] == ["elo", "ts"]
    assert data["total"] == 3
    assert [(row["rank"], row["entity_name"]) for row in data["rankings"]] == [
        (1, "Gen.G"),
        (2, "T1"),
        (3, "Bilibili Gaming"),
    ]
    assert data["rankings"][0]["rating_value"] == 1710.5
    assert data["rankings"][0]["system_count"] == 1


def test_player_rankings_support_name_search(client: TestClient) -> None:
    _seed_completed_rankings()

    response = client.get(
        "/rankings",
        params={"entity_type": "player", "rating_system": "elo", "search": "Fak", "min_games": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["rankings"][0]["entity_name"] == "Faker"
    assert data["rankings"][0]["team_name"] == "T1"
    assert data["rankings"][0]["role"] == "MID"


def test_unified_rankings_average_system_percentiles_and_preserve_global_rank(client: TestClient) -> None:
    _seed_completed_rankings()

    response = client.get(
        "/rankings",
        params={"entity_type": "team", "rating_system": "unified", "search": "Gen", "min_games": 10},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["rankings"][0]["entity_name"] == "Gen.G"
    assert data["rankings"][0]["rank"] == 2
    assert data["rankings"][0]["rating_value"] == 75.0
    assert data["rankings"][0]["system_count"] == 2


def test_default_cohort_excludes_stale_and_explicit_development_squads(client: TestClient) -> None:
    _seed_completed_rankings()

    default_response = client.get(
        "/rankings",
        params={"entity_type": "team", "rating_system": "unified", "search": "T1", "min_games": 10},
    )
    development_response = client.get(
        "/rankings",
        params={
            "entity_type": "team",
            "rating_system": "unified",
            "search": "T1",
            "min_games": 10,
            "squad_scope": "development",
        },
    )
    historical_response = client.get(
        "/rankings",
        params={
            "entity_type": "team",
            "rating_system": "unified",
            "search": "T1",
            "min_games": 10,
            "squad_scope": "all",
            "active_within_months": 0,
        },
    )

    assert default_response.status_code == 200
    assert default_response.json()["active_since"] == "2026-03-01"
    assert [row["entity_name"] for row in default_response.json()["rankings"]] == ["T1"]
    assert [row["entity_name"] for row in development_response.json()["rankings"]] == ["T1 Esports Academy"]
    assert {row["entity_name"] for row in historical_response.json()["rankings"]} == {
        "T1",
        "SK Telecom T1",
        "T1 Esports Academy",
    }


def test_rankings_prefer_regional_snapshot_and_expose_region(client: TestClient) -> None:
    _seed_completed_rankings()
    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO rating_runs(
                    id, ratings_version, data_cutoff_at, started_at, finished_at,
                    status, matches_processed, games_processed, players_processed
                ) VALUES (
                    4, 'ratings-v2', '2026-07-01T00:00:00+00:00',
                    '2026-07-01T00:00:00+00:00', '2026-07-01T00:05:00+00:00',
                    'completed', 40, 80, 200
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO entity_ratings(
                    rating_run_id, ratings_version, snapshot_at, entity_type,
                    entity_name, normalized_entity_name, team_name, role,
                    rating_system, rating_value, rd, sigma, games_played,
                    last_match_at, state_json
                ) VALUES (
                    4, 'ratings-v2', '2026-07-01T00:05:00+00:00', 'team',
                    'Gen.G', 'gen g', 'Gen.G', NULL, 'gl', 1725.0, 80.0, 0.06,
                    50, '2026-06-30',
                    '{"family":"LCK","tier":"major","offset": 25.0,"location_variance": 100.0}'
                )
                """
            )
        )
        session.commit()

    response = client.get(
        "/rankings",
        params={"entity_type": "team", "rating_system": "unified", "min_games": 10},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ratings_version"] == "ratings-v2"
    assert data["available_rating_systems"] == ["gl"]
    assert data["rankings"][0]["region_family"] == "LCK"
    assert data["rankings"][0]["region_tier"] == "major"
    assert data["rankings"][0]["regional_offset"] == 25.0
    assert data["rankings"][0]["regional_uncertainty"] == 10.0


def test_rankings_scope_major_vs_regional_academy(client: TestClient) -> None:
    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO rating_runs(
                    id, ratings_version, data_cutoff_at, started_at, finished_at,
                    status, matches_processed, games_processed, players_processed
                ) VALUES (
                    10, 'ratings-v2', '2026-09-01T00:00:00+00:00',
                    '2026-09-01T00:00:00+00:00', '2026-09-01T00:05:00+00:00',
                    'completed', 50, 100, 250
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO entity_ratings(
                    rating_run_id, ratings_version, snapshot_at, entity_type,
                    entity_name, normalized_entity_name, team_name, role,
                    rating_system, rating_value, rd, sigma, games_played,
                    last_match_at, state_json
                ) VALUES
                    (10, 'ratings-v2', '2026-09-01T00:05:00+00:00', 'team',
                     'T1', 't1', 'T1', NULL, 'gl', 1850.0, 70.0, 0.05, 50, '2026-08-30',
                     '{"family":"LCK","tier":"major","offset": 20.0,"location_variance": 50.0}'),
                    (10, 'ratings-v2', '2026-09-01T00:05:00+00:00', 'team',
                     'Gen.G', 'gen g', 'Gen.G', NULL, 'gl', 1880.0, 65.0, 0.05, 45, '2026-08-31',
                     '{"family":"LCK","tier":"major","offset": 20.0,"location_variance": 50.0}'),
                    (10, 'ratings-v2', '2026-09-01T00:05:00+00:00', 'team',
                     'Karmine Corp Blue', 'karmine corp blue', 'Karmine Corp Blue', NULL, 'gl', 1650.0, 75.0, 0.06, 40, '2026-08-25',
                     '{"family":"LFL","tier":"regional","offset": -10.0,"location_variance": 80.0}'),
                    (10, 'ratings-v2', '2026-09-01T00:05:00+00:00', 'team',
                     'T1 Esports Academy', 't1 academy', 'T1 Esports Academy', NULL, 'gl', 1600.0, 80.0, 0.06, 35, '2026-08-20',
                     '{"family":"LCK CL","tier":"development","offset": -30.0,"location_variance": 100.0}')
                """
            )
        )
        session.commit()

    # Major (default)
    major_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "gl", "squad_scope": "major"})
    assert major_resp.status_code == 200
    assert [r["entity_name"] for r in major_resp.json()["rankings"]] == ["Gen.G", "T1"]

    # Regional & Academy
    reg_acad_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "gl", "squad_scope": "regional_academy"})
    assert reg_acad_resp.status_code == 200
    assert [r["entity_name"] for r in reg_acad_resp.json()["rankings"]] == ["Karmine Corp Blue", "T1 Esports Academy"]

    # Regional only
    reg_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "gl", "squad_scope": "regional"})
    assert reg_resp.status_code == 200
    assert [r["entity_name"] for r in reg_resp.json()["rankings"]] == ["Karmine Corp Blue"]

    # Development only
    dev_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "gl", "squad_scope": "development"})
    assert dev_resp.status_code == 200
    assert [r["entity_name"] for r in dev_resp.json()["rankings"]] == ["T1 Esports Academy"]

    # All
    all_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "gl", "squad_scope": "all"})
    assert all_resp.status_code == 200
    assert len(all_resp.json()["rankings"]) == 4
    assert {r["entity_name"] for r in all_resp.json()["rankings"]} == {
        "Gen.G", "T1", "Karmine Corp Blue", "T1 Esports Academy"
    }


def test_unified_rankings_with_gl_calibration_and_weights(client: TestClient) -> None:
    import json

    with get_session() as session:
        session.execute(text("DELETE FROM entity_ratings"))
        session.execute(text("DELETE FROM rating_runs"))
        session.execute(
            text(
                """
                INSERT INTO rating_runs (
                    id, ratings_version, data_cutoff_at, started_at, finished_at,
                    status, matches_processed, games_processed, players_processed
                ) VALUES (
                    1, 'ratings-v3', '2026-09-01T00:00:00+00:00',
                    '2026-09-01T00:00:00+00:00', '2026-09-01T00:05:00+00:00',
                    'completed', 10, 20, 50
                )
                """
            )
        )

        # Gen.G (Major LCK): high GL, high raw Elo
        geng_state = json.dumps({"offset": 2200.0, "location_variance": 50000.0, "tier": "major"})
        # T1 (Major LCK): high GL, good raw Elo
        t1_state = json.dumps({"offset": 2150.0, "location_variance": 50000.0, "tier": "major"})
        # Galions (Regional): low GL, but inflated raw Elo from stomping weak regional opponents
        galions_state = json.dumps({"offset": 1500.0, "location_variance": 10000.0, "tier": "regional"})

        session.execute(
            text(
                """
                INSERT INTO entity_ratings(
                    rating_run_id, ratings_version, snapshot_at, entity_type,
                    entity_name, normalized_entity_name, team_name, role,
                    rating_system, rating_value, rd, sigma, games_played, last_match_at, state_json
                ) VALUES
                    (1, 'ratings-v3', '2026-09-01T00:05:00+00:00', 'team',
                     'Gen.G', 'gen g', NULL, NULL, 'gl', 2230.0, 35.0, NULL, 30, '2026-08-31', :geng_state),
                    (1, 'ratings-v3', '2026-09-01T00:05:00+00:00', 'team',
                     'T1', 't1', NULL, NULL, 'gl', 2160.0, 38.0, NULL, 30, '2026-08-31', :t1_state),
                    (1, 'ratings-v3', '2026-09-01T00:05:00+00:00', 'team',
                     'Galions', 'galions', NULL, NULL, 'gl', 1600.0, 45.0, NULL, 30, '2026-08-31', :galions_state),

                    (1, 'ratings-v3', '2026-09-01T00:05:00+00:00', 'team',
                     'Gen.G', 'gen g', NULL, NULL, 'elo', 2020.0, NULL, NULL, 30, '2026-08-31', :geng_state),
                    (1, 'ratings-v3', '2026-09-01T00:05:00+00:00', 'team',
                     'T1', 't1', NULL, NULL, 'elo', 1980.0, NULL, NULL, 30, '2026-08-31', :t1_state),
                    (1, 'ratings-v3', '2026-09-01T00:05:00+00:00', 'team',
                     'Galions', 'galions', NULL, NULL, 'elo', 2050.0, NULL, NULL, 30, '2026-08-31', :galions_state)
                """
            ),
            {"geng_state": geng_state, "t1_state": t1_state, "galions_state": galions_state},
        )
        session.commit()

    # 1. Under raw Elo, Galions is #1 because 2050 > 2020 > 1980
    elo_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "elo", "squad_scope": "all"})
    assert elo_resp.status_code == 200
    elo_rankings = elo_resp.json()["rankings"]
    assert elo_rankings[0]["entity_name"] == "Galions"
    assert elo_rankings[1]["entity_name"] == "Gen.G"
    assert elo_rankings[2]["entity_name"] == "T1"

    # 2. Under Unified (calibrated consensus with 80% GL weight + regional offset on Elo),
    # Gen.G and T1 must be ranked ahead of Galions
    unified_resp = client.get("/rankings", params={"entity_type": "team", "rating_system": "unified", "squad_scope": "all"})
    assert unified_resp.status_code == 200
    unified_rankings = unified_resp.json()["rankings"]
    assert [r["entity_name"] for r in unified_rankings] == ["Gen.G", "T1", "Galions"]
    assert unified_rankings[0]["rank"] == 1
    assert unified_rankings[1]["rank"] == 2
    assert unified_rankings[2]["rank"] == 3
    # Check system_count is 2
    assert all(r["system_count"] == 2 for r in unified_rankings)
