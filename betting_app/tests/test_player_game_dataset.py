from __future__ import annotations

import json
import math

import pandas as pd

from betting_app.ml.training.player_game_dataset import (
    PlayerGameDatasetConfig,
    build_player_game_dataset,
)


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "g1",
        "match_id": "m1",
        "date": "2025-01-01",
        "tournament_name": "LCK 2025",
        "patch": "15.1",
        "team1_id": "t1id",
        "team2_id": "t2id",
        "team1_name": "Team A",
        "team2_name": "Team B",
        "game_team1_win": 1,
        "game_team2_win": 0,
        "team1_side": "Blue",
        "team2_side": "Red",
        "game_duration": 1800,
        "team1_stats_json": json.dumps({"kills": 12, "towers": 8, "dragons": 3, "nashors": 1, "gold": 62000}),
        "team2_stats_json": json.dumps({"kills": 7, "towers": 3, "dragons": 1, "nashors": 0, "gold": 54000}),
        "team1_score": 2,
        "team2_score": 1,
        "match_team1_win": 1,
        "match_team2_win": 0,
        "best_of": 3,
        "team_id": "t1id",
        "team_name": "Team A",
        "side": "t1",
        "role": "MID",
        "player_id": "p1",
        "player_name": "Midlaner",
        "champion_id": "103",
        "champion_name": "Ahri",
        "stats_json": json.dumps(
            {
                "kills": 5,
                "deaths": 1,
                "assists": 8,
                "cs": 280,
                "csm": 9.3,
                "golds": 14500,
                "gpm": 483,
                "gold%": 0.23,
                "total_damage_to_champion": 23000,
                "dpm": 767,
                "dmg%": 0.31,
                "kp%": 0.81,
                "gd@15": 500,
                "csd@15": 12,
                "xpd@15": 300,
                "lvld@15": 1,
            }
        ),
    }
    row.update(overrides)
    return row


def test_build_player_game_dataset_parses_stats_and_targets() -> None:
    dataset = build_player_game_dataset(pd.DataFrame([_raw_row()]))

    assert dataset.metadata["rows"] == 1
    assert dataset.metadata["distinct_players"] == 1
    assert dataset.metadata["distinct_champions"] == 1
    assert "stat_kills" in dataset.feature_names
    assert "champion_id" in dataset.categorical_names
    row = dataset.frame.iloc[0]
    assert row["role_index"] == 2.0
    assert row["side_blue"] == 1.0
    assert row["patch_major"] == 15.0
    assert row["patch_minor"] == 1.0
    assert row["game_win"] == 1.0
    assert row["match_win"] == 1.0
    assert row["stat_kills"] == 5.0
    assert row["team_gold_diff"] == 8000.0
    assert row["team_kill_diff"] == 5.0


def test_build_player_game_dataset_handles_t2_orientation() -> None:
    dataset = build_player_game_dataset(
        pd.DataFrame([
            _raw_row(
                team_id="t2id",
                team_name="Team B",
                side="t2",
                role="ADC",
                game_team1_win=0,
                game_team2_win=1,
                match_team1_win=0,
                match_team2_win=1,
            )
        ])
    )
    row = dataset.frame.iloc[0]
    assert row["role_index"] == 3.0
    assert row["side_blue"] == 0.0
    assert row["game_win"] == 1.0
    assert row["match_win"] == 1.0
    assert row["team_gold_diff"] == -8000.0
    assert row["team_kill_diff"] == -5.0


def test_build_player_game_dataset_skips_rows_without_core_stats() -> None:
    dataset = build_player_game_dataset(pd.DataFrame([_raw_row(stats_json=json.dumps({"vision_score": 10}))]))

    assert dataset.frame.empty
    assert dataset.metadata["rows"] == 0
    assert dataset.metadata["skipped"] == {"missing_core_stats": 1}


def test_build_player_game_dataset_can_keep_sparse_rows_for_autoencoder() -> None:
    dataset = build_player_game_dataset(
        pd.DataFrame([_raw_row(stats_json=json.dumps({"vision_score": 10}))]),
        PlayerGameDatasetConfig(require_core_stats=False),
    )

    assert dataset.metadata["rows"] == 1
    row = dataset.frame.iloc[0]
    assert math.isnan(row["stat_kills"])
    assert row["stat_vision_score"] == 10.0
