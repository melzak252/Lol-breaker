from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from betting_app.core.db import connect, dispose_engine, init_db, transaction
from betting_app.scripts.rebuild_calibrated_ratings import (
    DEFAULT_SOURCE,
    RATING_SYSTEM,
    rebuild_calibrated_ratings,
)
from betting_app.scripts.rebuild_regional_ratings import (
    PUBLIC_SYSTEMS,
    RATINGS_VERSION,
    rebuild_regional_ratings,
)


@pytest.fixture
def calibrated_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    database = tmp_path / "calibrated-ratings.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    dispose_engine()
    init_db()
    try:
        yield
    finally:
        dispose_engine()


def _insert_match(
    *,
    match_id: str,
    match_date: str,
    tournament: str,
    team_a: tuple[str, str],
    team_b: tuple[str, str],
    roster_a: Sequence[tuple[str, str]],
    roster_b: Sequence[tuple[str, str]],
    scores: Sequence[int],
) -> None:
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO golgg_matches(
                match_id, date, tournament_name, team1_id, team2_id,
                team1_name, team2_name, team1_score, team2_score,
                team1_win, team2_win, draw, games_played
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                match_id,
                match_date,
                tournament,
                team_a[0],
                team_b[0],
                team_a[1],
                team_b[1],
                sum(scores),
                len(scores) - sum(scores),
                int(sum(scores) * 2 > len(scores)),
                int(sum(scores) * 2 < len(scores)),
                len(scores),
            ),
        )
        for game_number, score in enumerate(scores, start=1):
            game_id = f"{match_id}{game_number:02d}"
            connection.execute(
                """
                INSERT INTO golgg_games(
                    game_id, match_id, date, tournament_name,
                    team1_id, team2_id, team1_name, team2_name,
                    team1_win, team2_win, draw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    game_id,
                    match_id,
                    match_date,
                    tournament,
                    team_a[0],
                    team_b[0],
                    team_a[1],
                    team_b[1],
                    int(score),
                    1 - int(score),
                ),
            )
            if game_number != 1:
                continue
            for side, team, roster in (
                ("1", team_a, roster_a),
                ("2", team_b, roster_b),
            ):
                for role, (player_id, player_name) in zip(
                    ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT"), roster, strict=False
                ):
                    connection.execute(
                        """
                        INSERT INTO golgg_game_players(
                            game_id, match_id, team_id, team_name, side,
                            role, player_id, player_name
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            game_id,
                            match_id,
                            team[0],
                            team[1],
                            side,
                            role,
                            player_id,
                            player_name,
                        ),
                    )


def _insert_history(*, include_bridge: bool = True) -> None:
    _insert_match(
        match_id="100",
        match_date="2024-01-01",
        tournament="LCK 2024 Spring",
        team_a=("kr-a", "Korea Alpha"),
        team_b=("kr-b", "Korea Beta"),
        roster_a=(("k1", "K One"), ("k2", "K Two")),
        roster_b=(("k3", "K Three"), ("k4", "K Four")),
        scores=(1, 1),
    )
    _insert_match(
        match_id="101",
        match_date="2024-01-01",
        tournament="LEC 2024 Spring",
        team_a=("eu-a", "Europe Alpha"),
        team_b=("eu-b", "Europe Beta"),
        roster_a=(("e1", "E One"), ("e2", "E Two")),
        roster_b=(("e3", "E Three"), ("e4", "E Four")),
        scores=(1, 1),
    )
    if include_bridge:
        _insert_match(
            match_id="200",
            match_date="2024-02-01",
            tournament="Worlds 2024 Main Event",
            team_a=("kr-a", "Korea Alpha"),
            team_b=("eu-a", "Europe Alpha"),
            roster_a=(("k1", "K One Prime"), ("k2", "K Two")),
            roster_b=(("e1", "E One"), ("e2", "E Two")),
            scores=(1, 1),
        )


def _insert_late_cutoff_match() -> None:
    _insert_match(
        match_id="201",
        match_date="2024-02-01",
        tournament="Worlds 2024 Main Event",
        team_a=("kr-b", "Korea Beta"),
        team_b=("eu-b", "Europe Beta"),
        roster_a=(("k3", "K Three"), ("k4", "K Four")),
        roster_b=(("e3", "E Three"), ("e4", "E Four")),
        scores=(0, 0),
    )


def _rating_rows(version: str) -> list[dict[str, object]]:
    with connect() as connection:
        return connection.execute(
            """
            SELECT entity_type, entity_name, normalized_entity_name, team_name, role,
                   rating_system, rating_value, rd, sigma, games_played,
                   last_match_at, state_json
            FROM entity_ratings
            WHERE ratings_version = ?
            ORDER BY entity_type, normalized_entity_name
            """,
            (version,),
        ).fetchall()


def _run(version: str) -> dict[str, object]:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT ratings_version, source, data_cutoff_at, started_at, finished_at,
                   status, systems_json, matches_processed, games_processed,
                   players_processed, error
            FROM rating_runs
            WHERE ratings_version = ?
            """,
            (version,),
        ).fetchone()
    assert row is not None
    return row


def test_full_rebuild_materializes_only_real_players_and_teams_with_metadata(
    calibrated_db: None,
) -> None:
    _insert_history()

    stats = rebuild_calibrated_ratings(version="calibrated-full", mode="full")

    assert stats == {
        "version": "calibrated-full",
        "mode": "full",
        "matches": 3,
        "matches_total": 3,
        "games": 6,
        "games_total": 6,
        "players": 8,
        "entities": 12,
        "rows": 12,
        "data_cutoff_at": "2024-02-01",
    }
    rows = _rating_rows("calibrated-full")
    assert len(rows) == 12
    assert {row["entity_type"] for row in rows} == {"player", "team"}
    assert {row["rating_system"] for row in rows} == {RATING_SYSTEM}
    assert "unknown" not in {row["normalized_entity_name"] for row in rows}
    assert not any("dummy" in str(row["entity_name"]).casefold() for row in rows)

    players = {
        str(row["normalized_entity_name"]): row
        for row in rows
        if row["entity_type"] == "player"
    }
    assert set(players) == {"k1", "k2", "k3", "k4", "e1", "e2", "e3", "e4"}
    assert players["k1"]["entity_name"] == "K One Prime"
    assert players["k1"]["team_name"] == "Korea Alpha"
    assert float(players["k1"]["rating_value"]) > float(players["k3"]["rating_value"])
    k1_state = json.loads(str(players["k1"]["state_json"]))
    assert isinstance(k1_state["raw_rating"], float)
    assert isinstance(k1_state["raw_rd"], float)
    assert k1_state["family"] == "LCK"
    assert k1_state["tier"] == "major"
    assert k1_state["last_activity"] == "2024-02-01"

    run = _run("calibrated-full")
    assert run["source"] == DEFAULT_SOURCE
    assert run["status"] == "completed"
    assert run["data_cutoff_at"] == "2024-02-01"
    assert run["matches_processed"] == 3
    assert run["games_processed"] == 6
    assert run["players_processed"] == 8
    system = json.loads(str(run["systems_json"]))[RATING_SYSTEM]
    assert system["version"] == "player-glicko2-family-v1"
    assert system["parameters"]["rating_period"] == "complete_calendar_date"
    assert system["parameters"]["family_calibration"]["competition_prestige_weight"] is False
    assert system["state"]
    assert system["metadata"]["team_affiliations"]["kr-a"]["family"] == "LCK"
    assert system["checkpoint"]["before_date"] == "2024-02-01"


def test_incremental_replays_cutoff_date_and_matches_full_state_with_late_data(
    calibrated_db: None,
) -> None:
    _insert_history()
    rebuild_calibrated_ratings(version="stepped", mode="full")
    _insert_late_cutoff_match()
    rebuild_calibrated_ratings(version="one-pass", mode="full")

    incremental = rebuild_calibrated_ratings(
        version="stepped", mode="incremental", until_date="2024-02-01"
    )

    assert incremental["matches"] == 2
    assert incremental["matches_total"] == 4
    assert incremental["data_cutoff_at"] == "2024-02-01"
    one_pass_system = json.loads(str(_run("one-pass")["systems_json"]))
    stepped_system = json.loads(str(_run("stepped")["systems_json"]))
    assert stepped_system == one_pass_system
    assert _rating_rows("stepped") == _rating_rows("one-pass")


def test_unknown_competition_failure_preserves_completed_snapshot(
    calibrated_db: None,
) -> None:
    _insert_history(include_bridge=False)
    rebuild_calibrated_ratings(version="safe", mode="full")
    before_rows = _rating_rows("safe")
    before_run = _run("safe")
    _insert_match(
        match_id="300",
        match_date="2024-03-01",
        tournament="Unmapped Invitational ZZZ",
        team_a=("kr-a", "Korea Alpha"),
        team_b=("eu-a", "Europe Alpha"),
        roster_a=(("k1", "K One"), ("k2", "K Two")),
        roster_b=(("e1", "E One"), ("e2", "E Two")),
        scores=(1, 1),
    )

    with pytest.raises(ValueError, match="unknown competition classification"):
        rebuild_calibrated_ratings(version="safe", mode="incremental")

    assert _rating_rows("safe") == before_rows
    assert _run("safe") == before_run


def test_full_rebuild_rejects_empty_history(calibrated_db: None) -> None:
    with pytest.raises(ValueError, match="no eligible matches"):
        rebuild_calibrated_ratings(version="empty", mode="full")

    assert _rating_rows("empty") == []
    assert _run("empty")["status"] == "failed"


def test_team_snapshot_uses_player_uncertainty_projected_to_cutoff(
    calibrated_db: None,
) -> None:
    _insert_history(include_bridge=False)
    _insert_match(
        match_id="400",
        match_date="2024-04-01",
        tournament="LEC 2024 Spring",
        team_a=("eu-a", "Europe Alpha"),
        team_b=("eu-b", "Europe Beta"),
        roster_a=(("e1", "E One"), ("e2", "E Two")),
        roster_b=(("e3", "E Three"), ("e4", "E Four")),
        scores=(1, 1),
    )

    rebuild_calibrated_ratings(version="projected-team", mode="full")
    rows = _rating_rows("projected-team")
    player_states = {
        str(row["normalized_entity_name"]): json.loads(str(row["state_json"]))
        for row in rows
        if row["entity_type"] == "player"
    }
    team_row = next(
        row
        for row in rows
        if row["entity_type"] == "team"
        and json.loads(str(row["state_json"]))["team_id"] == "kr-a"
    )
    team_state = json.loads(str(team_row["state_json"]))
    expected_raw_rd = math.sqrt(
        player_states["k1"]["raw_rd"] ** 2
        + player_states["k2"]["raw_rd"] ** 2
    ) / 2

    assert team_state["raw_rd"] == pytest.approx(expected_raw_rd)


def test_regional_rebuild_materializes_one_cohort_for_all_public_systems(
    calibrated_db: None,
) -> None:
    _insert_history()

    stats = rebuild_regional_ratings()

    assert stats == {
        "version": RATINGS_VERSION,
        "mode": "full",
        "matches": 3,
        "games": 6,
        "players": 8,
        "entities": 12,
        "rows": 72,
        "data_cutoff_at": "2024-02-01",
    }
    rows = _rating_rows(RATINGS_VERSION)
    assert {str(row["rating_system"]) for row in rows} == set(PUBLIC_SYSTEMS)
    for system in PUBLIC_SYSTEMS:
        system_rows = [row for row in rows if row["rating_system"] == system]
        assert len(system_rows) == 12
        assert {
            (str(row["entity_type"]), str(row["normalized_entity_name"]))
            for row in system_rows
        } == {
            (str(row["entity_type"]), str(row["normalized_entity_name"]))
            for row in rows
            if row["rating_system"] == "gl"
        }

    regional_state = json.loads(
        str(
            next(
                row["state_json"]
                for row in rows
                if row["rating_system"] == "gl"
                and row["entity_type"] == "team"
                and row["normalized_entity_name"] == "korea alpha"
            )
        )
    )
    assert regional_state["competition_calibration"] == "family-calibrated-glicko2-v1"
    run = _run(RATINGS_VERSION)
    payload = json.loads(str(run["systems_json"]))
    assert payload["contract_version"] == RATINGS_VERSION
    assert payload["public_systems"] == list(PUBLIC_SYSTEMS)
    assert payload["gl"]["engine"] == "family-calibrated-glicko2-v1"


def test_regional_rebuild_deduplicates_renamed_team_across_every_system(
    calibrated_db: None,
) -> None:
    _insert_history()
    _insert_match(
        match_id="300",
        match_date="2024-03-01",
        tournament="LCK 2024 Spring",
        team_a=("kr-a-renamed", "Korea Alpha"),
        team_b=("kr-b", "Korea Beta"),
        roster_a=(("k1", "K One"), ("k2", "K Two")),
        roster_b=(("k3", "K Three"), ("k4", "K Four")),
        scores=(1, 1),
    )

    stats = rebuild_regional_ratings()

    assert stats["entities"] == 12
    assert stats["rows"] == 72
    rows = _rating_rows(RATINGS_VERSION)
    expected = {
        (str(row["entity_type"]), str(row["normalized_entity_name"]))
        for row in rows
        if row["rating_system"] == "gl"
    }
    for system in PUBLIC_SYSTEMS:
        current = [
            (str(row["entity_type"]), str(row["normalized_entity_name"]))
            for row in rows
            if row["rating_system"] == system
        ]
        assert len(current) == len(expected)
        assert set(current) == expected
