"""Tests for player search, profile, rating trajectory, and comparison endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import text

from betting_app.api.schemas import H2HSummary, PlayerProfileDetail
from betting_app.core.db import get_session
from betting_app.services.player_comparison_service import (
    calculate_model_verdict,
    get_head_to_head_summary,
    get_player_profile,
    search_players,
)


@pytest.fixture(autouse=True)
def seed_test_player_data(client: TestClient):
    """Seed minimal test data for players, ratings, and matches."""
    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO rating_runs (
                    id, ratings_version, status, games_processed, players_processed
                ) VALUES (999, 'test-player-version', 'completed', 50, 10)
                ON CONFLICT (id) DO NOTHING;
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO entity_ratings (
                    ratings_version, entity_type, entity_name, normalized_entity_name,
                    team_name, role, rating_system, rating_value, rd, sigma, games_played, last_match_at
                ) VALUES
                ('test-player-version', 'player', 'AlphaPlayer', '101', 'Team One', 'MID', 'elo', 2100.0, NULL, NULL, 50, '2026-08-01'),
                ('test-player-version', 'player', 'AlphaPlayer', '101', 'Team One', 'MID', 'gl', 1800.0, 60.0, NULL, 50, '2026-08-01'),
                ('test-player-version', 'player', 'AlphaPlayer', '101', 'Team One', 'MID', 'ts', 32.0, NULL, 3.5, 50, '2026-08-01'),
                ('test-player-version', 'player', 'BetaPlayer', '102', 'Team Two', 'MID', 'elo', 1900.0, NULL, NULL, 40, '2026-08-01'),
                ('test-player-version', 'player', 'BetaPlayer', '102', 'Team Two', 'MID', 'gl', 1700.0, 65.0, NULL, 40, '2026-08-01'),
                ('test-player-version', 'player', 'BetaPlayer', '102', 'Team Two', 'MID', 'ts', 28.0, NULL, 4.0, 40, '2026-08-01')
                ON CONFLICT DO NOTHING;
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO golgg_matches (match_id, date, team1_name, team2_name)
                VALUES ('test-match-1', '2026-07-20', 'Team One', 'Team Two')
                ON CONFLICT DO NOTHING;
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO golgg_games (game_id, match_id, date, team1_win, team2_win)
                VALUES
                ('test-game-1', 'test-match-1', '2026-07-20', 1, 0),
                ('test-game-2', 'test-match-1', '2026-07-20', 0, 1)
                ON CONFLICT DO NOTHING;
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO golgg_game_players (game_id, match_id, player_id, player_name, team_name, side, champion_name, role)
                VALUES
                ('test-game-1', 'test-match-1', '101', 'AlphaPlayer', 'Team One', 't1', 'Ahri', 'MID'),
                ('test-game-1', 'test-match-1', '102', 'BetaPlayer', 'Team Two', 't2', 'Azir', 'MID'),
                ('test-game-2', 'test-match-1', '101', 'AlphaPlayer', 'Team One', 't1', 'Orianna', 'MID'),
                ('test-game-2', 'test-match-1', '102', 'BetaPlayer', 'Team Two', 't2', 'Syndra', 'MID')
                ON CONFLICT DO NOTHING;
                """
            )
        )
        session.commit()
    yield


def test_search_players(client: TestClient):
    """Verify player search returns relevant entries sorted by relevance and games."""
    with get_session() as session:
        results = search_players(session, "Alpha")
        assert len(results) >= 1
        assert any(p.player_name == "AlphaPlayer" for p in results)
        first = next(p for p in results if p.player_name == "AlphaPlayer")
        assert first.player_id == "101"
        assert first.current_elo == 2100.0


def test_get_player_profile(client: TestClient):
    """Verify player profile loads multi-system ratings and career games."""
    with get_session() as session:
        profile = get_player_profile(session, "101")
        assert profile is not None
        assert profile.player_name == "AlphaPlayer"
        assert profile.role == "MID"
        assert "elo" in profile.ratings
        assert profile.ratings["elo"].rating_value == 2100.0
        assert "gl" in profile.ratings
        assert profile.ratings["gl"].rating_value == 1800.0


def test_h2h_summary(client: TestClient):
    """Verify head-to-head match records between two players."""
    with get_session() as session:
        h2h = get_head_to_head_summary(session, "101", "102")
        assert h2h.total_games >= 2
        assert h2h.wins_a >= 1
        assert h2h.wins_b >= 1
        assert len(h2h.recent_games) >= 2

def test_calculate_model_verdict_favors_higher_rating():
    """Verify verdict selects the player with systematically higher ratings."""
    prof_a = PlayerProfileDetail(
        player_id="101",
        player_name="AlphaPlayer",
        games_played=50,
        career_wins=35,
        career_losses=15,
        career_win_rate=0.7,
        ratings={
            "elo": {"system": "elo", "rating_value": 2200.0},
            "gl": {"system": "gl", "rating_value": 1900.0, "rd": 60.0},
            "ts": {"system": "ts", "rating_value": 35.0, "sigma": 3.0},
        },
    )
    prof_b = PlayerProfileDetail(
        player_id="102",
        player_name="BetaPlayer",
        games_played=40,
        career_wins=20,
        career_losses=20,
        career_win_rate=0.5,
        ratings={
            "elo": {"system": "elo", "rating_value": 1800.0},
            "gl": {"system": "gl", "rating_value": 1600.0, "rd": 60.0},
            "ts": {"system": "ts", "rating_value": 26.0, "sigma": 3.0},
        },
    )
    h2h = H2HSummary(total_games=2, wins_a=1, wins_b=1, win_rate_a=0.5, win_rate_b=0.5)

    verdict = calculate_model_verdict(prof_a, prof_b, h2h)
    assert verdict.better_player == "a"
    assert verdict.better_player_id == "101"
    assert verdict.win_probability_a > 0.65
    assert verdict.systems_favor_a == 3
    assert verdict.systems_favor_b == 0
    assert "AlphaPlayer" in verdict.summary_pl


def test_api_player_endpoints(client: TestClient):
    """Verify FastAPI routes for player search, profile, and comparison."""
    # 1. Search
    res_search = client.get("/players/search?query=Alpha")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert isinstance(search_data, list)
    assert any(p["player_name"] == "AlphaPlayer" for p in search_data)

    # 2. Profile
    res_profile = client.get("/players/101")
    assert res_profile.status_code == 200
    prof_data = res_profile.json()
    assert prof_data["player_name"] == "AlphaPlayer"
    assert "elo" in prof_data["ratings"]

    # 3. History
    res_hist = client.get("/players/101/history")
    assert res_hist.status_code == 200
    assert isinstance(res_hist.json(), list)

    # 4. Comparison
    res_comp = client.get("/players/compare?player_a=101&player_b=102")
    assert res_comp.status_code == 200
    comp_data = res_comp.json()
    assert comp_data["player_a"]["player_id"] == "101"
    assert comp_data["player_b"]["player_id"] == "102"
    assert comp_data["verdict"]["better_player"] in ("a", "b", "tied")
    assert "summary_pl" in comp_data["verdict"]
    assert "total_games" in comp_data["h2h"]
