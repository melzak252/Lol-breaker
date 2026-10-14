from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from betting_app.core.db import connect, dispose_engine, init_db, transaction
from betting_app.scripts.rebuild_calibrated_ratings import load_matches
from betting_app.scripts.repair_golgg_match_teams import repair_golgg_matches
from betting_app.services.golgg_import_service import import_golgg_batch


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    database = tmp_path / "test-repair-golgg.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    dispose_engine()
    init_db()
    try:
        yield
    finally:
        dispose_engine()


def test_repair_golgg_matches_populates_missing_team_ids(temp_db: None) -> None:
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO golgg_matches(
                match_id, date, tournament_name, team1_name, team2_name,
                team1_id, team2_id, team1_score, team2_score, team1_win, team2_win, draw, games_played
            ) VALUES ('82208', '2026-08-29', 'LEC 2026 Summer Season', 'KC', 'SK Gaming', '', '', 2, 1, 1, 0, 0, 3)
            """
        )
        connection.execute(
            """
            INSERT INTO golgg_games(
                game_id, match_id, date, tournament_name, team1_name, team2_name,
                team1_id, team2_id, team1_win, team2_win, draw
            ) VALUES ('1001', '82208', '2026-08-29', 'LEC 2026 Summer Season', 'SK Gaming', 'Karmine Corp', '2903', '2899', 0, 1, 0)
            """
        )

    stats = repair_golgg_matches(dry_run=False)
    assert stats["checked"] == 1
    assert stats["repaired"] == 1

    with connect() as connection:
        row = connection.execute(
            "SELECT team1_name, team1_id, team2_name, team2_id FROM golgg_matches WHERE match_id = '82208'"
        ).fetchone()
        assert row is not None
        # Swapped alignment check: KC matches Karmine Corp, SK matches SK Gaming
        assert row["team1_name"] == "Karmine Corp"
        assert row["team1_id"] == "2899"
        assert row["team2_name"] == "SK Gaming"
        assert row["team2_id"] == "2903"


def test_import_golgg_batch_preserves_full_names_on_conflict(temp_db: None) -> None:
    # 1. First import with full game data
    batch_with_games = [
        {
            "match_id": "82208",
            "date": "2026-08-29",
            "tournament_name": "LEC 2026 Summer Season",
            "sname_t1": "KC",
            "sname_t2": "SK",
            "games_played": 1,
            "games": [
                {
                    "game_id": "1001",
                    "t1_name": "Karmine Corp",
                    "t2_name": "SK Gaming",
                    "t1_id": "2899",
                    "t2_id": "2903",
                    "team1_win": 1,
                    "team2_win": 0,
                }
            ],
        }
    ]
    import_golgg_batch(batch_with_games)

    with connect() as connection:
        row = connection.execute(
            "SELECT team1_name, team1_id, team2_name, team2_id FROM golgg_matches WHERE match_id = '82208'"
        ).fetchone()
        assert row["team1_name"] == "Karmine Corp"
        assert row["team1_id"] == "2899"
        assert row["team2_name"] == "SK Gaming"
        assert row["team2_id"] == "2903"

    # 2. Subsequent stub import without games / without team IDs must NOT overwrite full data
    stub_batch = [
        {
            "match_id": "82208",
            "date": "2026-08-29",
            "tournament_name": "LEC 2026 Summer Season",
            "sname_t1": "KC",
            "sname_t2": "SK",
            "games_played": 1,
        }
    ]
    import_golgg_batch(stub_batch)

    with connect() as connection:
        row = connection.execute(
            "SELECT team1_name, team1_id, team2_name, team2_id FROM golgg_matches WHERE match_id = '82208'"
        ).fetchone()
        assert row["team1_name"] == "Karmine Corp"
        assert row["team1_id"] == "2899"
        assert row["team2_name"] == "SK Gaming"
        assert row["team2_id"] == "2903"


def test_load_matches_resolves_teams_from_games_fallback(temp_db: None) -> None:
    # Match row has abbreviated names and empty team IDs
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO golgg_matches(
                match_id, date, tournament_name, team1_name, team2_name,
                team1_id, team2_id, team1_score, team2_score, team1_win, team2_win, draw, games_played
            ) VALUES ('82208', '2026-08-29', 'LEC 2026 Summer Season', 'KC', 'SK', '', '', 1, 0, 1, 0, 0, 1)
            """
        )
        connection.execute(
            """
            INSERT INTO golgg_games(
                game_id, match_id, date, tournament_name, team1_name, team2_name,
                team1_id, team2_id, team1_win, team2_win, draw
            ) VALUES ('1001', '82208', '2026-08-29', 'LEC 2026 Summer Season', 'Karmine Corp', 'SK Gaming', '2899', '2903', 1, 0, 0)
            """
        )
        # Players attached to team_id 2899 and 2903
        connection.execute(
            """
                INSERT INTO golgg_game_players(game_id, match_id, team_id, team_name, player_id, player_name, role, side)
                VALUES
                    ('1001', '82208', '2899', 'Karmine Corp', '3406', 'Yike', 'JUNGLE', 'blue'),
                    ('1001', '82208', '2903', 'SK Gaming', '9999', 'Opponent', 'JUNGLE', 'red')
            """
        )

    loaded = load_matches(from_date="2026-08-01")
    assert len(loaded) == 1
    assert loaded[0].event_id == "82208"
    assert loaded[0].team_a_name == "Karmine Corp"
    assert loaded[0].team_b_name == "SK Gaming"
    assert any(p.display_name == "Yike" for p in loaded[0].players_a)
