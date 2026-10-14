from __future__ import annotations

import os
import sys

import pytest

from betting_app.core.db import dispose_engine, init_db, transaction
from betting_app.scripts import rematch_canonical_matches
from betting_app.scripts.rematch_canonical_matches import deduplicate_odds_for_canonical_update


@pytest.fixture()
def temp_db(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'rematch.sqlite3'}"
    dispose_engine()
    init_db()
    try:
        yield
    finally:
        os.environ.pop("DATABASE_URL", None)
        dispose_engine()


def _insert_canonical_match(connection, match_id: int) -> None:
    connection.execute(
        """
        INSERT INTO canonical_matches(
            id, canonical_key, team_a_name, team_b_name,
            normalized_team_a, normalized_team_b, start_time_normalized, league
        ) VALUES (?, ?, 'Team A', 'Team B', 'team a', 'team b', '2026-07-23T10:00:00+00:00', 'LCK')
        """,
        (match_id, f"canonical-{match_id}"),
    )


def test_deduplicate_odds_for_canonical_update_removes_rows_that_would_violate_unique_index(temp_db):
    with transaction() as connection:
        _insert_canonical_match(connection, 100)
        connection.execute(
            """
            INSERT INTO odds_snapshots(
                id, bookmaker_id, match_id, canonical_match_id, market_type,
                raw_team_a, raw_team_b, odds_a, odds_b, scraped_at
            ) VALUES
              (1, 1, 10, 100, 'match_winner', 'Team A', 'Team B', 2.0, 1.8, '2026-07-23T02:55:00+00:00'),
              (2, 1, 11, NULL, 'match_winner', 'Team A', 'Team B', 2.1, 1.7, '2026-07-23T02:55:00+00:00')
            """
        )

        removed = deduplicate_odds_for_canonical_update(connection, match_id=11, canonical_match_id=100)
        connection.execute(
            "UPDATE odds_snapshots SET canonical_match_id = ? WHERE match_id = ?",
            (100, 11),
        )

        rows = connection.execute(
            "SELECT id, canonical_match_id FROM odds_snapshots ORDER BY id"
        ).fetchall()

    assert removed == 1
    assert rows == [{"id": 1, "canonical_match_id": 100}]


def test_deduplicate_odds_for_canonical_update_keeps_highest_id_inside_same_match(temp_db):
    with transaction() as connection:
        _insert_canonical_match(connection, 100)
        connection.execute(
            """
            INSERT INTO odds_snapshots(
                id, bookmaker_id, match_id, canonical_match_id, market_type,
                raw_team_a, raw_team_b, odds_a, odds_b, scraped_at
            ) VALUES
              (1, 1, 11, NULL, 'match_winner', 'Team A', 'Team B', 2.0, 1.8, '2026-07-23T02:55:00+00:00'),
              (2, 1, 11, NULL, 'match_winner', 'Team A', 'Team B', 2.1, 1.7, '2026-07-23T02:55:00+00:00')
            """
        )

        removed = deduplicate_odds_for_canonical_update(connection, match_id=11, canonical_match_id=100)
        connection.execute(
            "UPDATE odds_snapshots SET canonical_match_id = ? WHERE match_id = ?",
            (100, 11),
        )
        rows = connection.execute(
            "SELECT id, canonical_match_id FROM odds_snapshots ORDER BY id"
        ).fetchall()

    assert removed == 1
    assert rows == [{"id": 2, "canonical_match_id": 100}]

def test_main_no_overview_suppresses_canonical_match_report(monkeypatch, capsys):
    monkeypatch.setattr(rematch_canonical_matches, "init_db", lambda: None)
    monkeypatch.setattr(rematch_canonical_matches, "rematch_odds_snapshots", lambda: 3)
    monkeypatch.setattr(
        rematch_canonical_matches,
        "canonical_match_overview",
        lambda limit: pytest.fail("overview must not be queried"),
    )
    monkeypatch.setattr(sys, "argv", ["rematch_canonical_matches.py", "--no-overview"])

    rematch_canonical_matches.main()

    assert capsys.readouterr().out == "Rematched odds snapshots: 3\n"
