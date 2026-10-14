"""Regression tests for native-C0-ready active-team suggestions."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import text

from betting_app.core.db import get_session


def test_active_teams_requires_exact_five_player_roster_not_gl_ratings(
    client: TestClient,
) -> None:
    session = get_session()
    ready_name = "C0 Ready Main"
    partial_name = "C0 Partial Academy"
    try:
        session.execute(
            text(
                """
                INSERT INTO golgg_teams(team_name, team_id, normalized_name)
                VALUES (:name, :team_id, :normalized)
                """
            ),
            [
                {"name": ready_name, "team_id": 9182736, "normalized": "c0 ready main"},
                {"name": partial_name, "team_id": 9182737, "normalized": "c0 partial academy"},
            ],
        )
        for team_id, team_name, count in (
            ("9182736", ready_name, 5),
            ("9182737", partial_name, 4),
        ):
            normalized = team_name.lower()
            for index in range(count):
                session.execute(
                    text(
                        """
                        INSERT INTO team_current_roster_players(
                            team_id, team_name, normalized_team_name, player_id,
                            player_name, role, source
                        ) VALUES (
                            :team_id, :team_name, :normalized, :player_id,
                            :player_name, :role, 'test'
                        )
                        """
                    ),
                    {
                        "team_id": team_id,
                        "team_name": team_name,
                        "normalized": normalized,
                        "player_id": f"{team_id}-player-{index}",
                        "player_name": f"Player {index}",
                        "role": ("TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT")[index],
                    },
                )
        session.commit()
    finally:
        session.close()

    response = client.get("/matches/active-teams")
    assert response.status_code == 200
    teams = response.json()["teams"]
    ready = next(team for team in teams if team["name"] == ready_name)
    assert ready["team_row_id"] is not None
    assert ready["native_team_id"] == "9182736"
    assert str(ready["team_row_id"]) != ready["native_team_id"]
    assert ready["rating"] is None
    assert not any(team["name"] == partial_name for team in teams)
