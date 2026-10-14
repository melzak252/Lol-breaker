"""Regression contracts for historical feature reconstruction, without a database."""

from copy import deepcopy

import pandas as pd
import pytest

from scripts.build_siamese_research_dataset import build_dataset
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS


def match(mid, day, winner=True, kills=10):
    players = {
        str(i): {
            "player_id": str(i),
            "stats": {
                "kills": kills / 5,
                "deaths": 2,
                "dpm": 400,
                "vspm": 2,
                "gd@15": 10,
            },
        }
        for i in range(5)
    }
    opponent = deepcopy(players)
    for i, player in enumerate(opponent.values()):
        player["player_id"] = str(i + 5)
    return {
        "match_id": str(mid),
        "date": day,
        "best_of": 1,
        "t1_id": "a",
        "t2_id": "b",
        "t1_score": int(winner),
        "t2_score": int(not winner),
        "t1_win": winner,
        "t2_win": not winner,
        "tournament_name": "LCK 2024",
        "games": [
            {
                "game_id": str(mid),
                "t1_id": "a",
                "t2_id": "b",
                "t1_win": winner,
                "t2_win": not winner,
                "t1_players": players,
                "t2_players": opponent,
                "t1_stats": {"towers": 5, "nashors": 1, "gold": 50000},
                "t2_stats": {"towers": 4, "nashors": 0, "gold": 45000},
                "game_duration": 1800,
            }
        ],
    }


def rated(matches):
    rows = []
    for m in matches:
        row = {
            name: (
                0.6
                if name
                in {
                    f"{scope}_{s}"
                    for scope in ("team", "player")
                    for s in ("elo", "gl", "ts", "os", "pl", "tm")
                }
                else 1.0
            )
            for name in _REQUIRED_BASE_FIELDS
            if "rolling" not in name
        }
        row.update(
            golgg_match_id=m["match_id"],
            date=m["date"],
            team1_id="a",
            team2_id="b",
            BoN=1,
            y_true=int(m["t1_win"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_current_result_cannot_change_own_features_and_same_day_is_excluded():
    matches = [
        match(1, "2023-01-01"),
        match(2, "2023-01-02", kills=20),
        match(3, "2023-01-03"),
        match(4, "2023-01-03"),
        match(5, "2023-01-04"),
    ]
    data, audit = build_dataset(rated(matches), matches)
    assert data.golgg_match_id.tolist() == ["2", "5"]
    assert data.iloc[0].t1_rolling_kills == 10
    assert audit["exclusions"]["same_day_participant"] == 2
    changed = deepcopy(matches)
    changed[1] = match(2, "2023-01-02", winner=False, kills=100)
    other, _ = build_dataset(rated(changed), changed)
    columns = [c for c in data if "rolling" in c]
    pd.testing.assert_series_equal(data.iloc[0][columns], other.iloc[0][columns])
    assert other.iloc[1].t1_rolling_kills != data.iloc[1].t1_rolling_kills


def test_incomplete_stat_is_not_fabricated_as_zero():
    matches = [match(1, "2023-01-01"), match(2, "2023-01-02")]
    matches[0]["games"][0]["t1_players"]["0"]["stats"]["dpm"] = None
    data, audit = build_dataset(rated(matches), matches)
    assert data.empty
    assert audit["exclusions"]["incomplete_w20_history"] == 2


def test_reversed_rating_sides_are_not_silently_joined():
    matches = [match(1, "2023-01-01"), match(2, "2023-01-02")]
    ratings = rated(matches)
    ratings.loc[1, ["team1_id", "team2_id"]] = ["b", "a"]
    data, audit = build_dataset(ratings, matches)
    assert data.empty
    assert audit["exclusions"]["rating_metadata_mismatch"] == 1


def test_duplicate_ids_are_rejected():
    matches = [match(1, "2023-01-01")]
    with pytest.raises(ValueError, match="duplicate"):
        build_dataset(rated(matches * 2), matches)
