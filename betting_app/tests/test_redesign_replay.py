"""Event-date and side contracts for the independent EXP-083 replay."""

import numpy as np
import pandas as pd

from src.models.replay_series import build_replay_features


def source_fixture():
    matches, games, players = [], [], []
    for number, day in enumerate(("2023-01-01", "2023-01-03", "2023-01-05"), 1):
        matches.append(
            dict(
                match_id=str(number),
                date=day,
                tournament_name="LCK 2023",
                team1_id="a",
                team2_id="b",
                team1_name="Alpha",
                team2_name="Beta",
                team1_score=1,
                team2_score=0,
                team1_win=1,
                team2_win=0,
                best_of=1,
            )
        )
        games.append(
            dict(
                game_id=str(number),
                match_id=str(number),
                date=day,
                team1_id="a",
                team2_id="b",
                team1_win=1,
                team2_win=0,
                game_duration=30,
                team1_stats_json='{"kills": 15, "gold": 55000, "towers": 8, "nashors": 1}',
                team2_stats_json='{"kills": 5, "gold": 45000, "towers": 2, "nashors": 0}',
            )
        )
        for team in ("a", "b"):
            for slot in range(5):
                players.append(
                    dict(
                        game_id=str(number),
                        match_id=str(number),
                        team_id=team,
                        player_id=f"{team}{slot}",
                        player_name=f"{team}{slot}",
                        role=str(slot),
                    )
                )
    return pd.DataFrame(matches), pd.DataFrame(games), pd.DataFrame(players)


def feature_values(frame):
    return frame[[name for name in frame if name.startswith(("d_", "c_"))]].to_numpy(
        float
    )


def test_current_result_statistics_and_roster_cannot_change_current_features():
    matches, games, players = source_fixture()
    expected, _ = build_replay_features(matches, games, players)
    matches.loc[1, ["team1_score", "team2_score", "team1_win", "team2_win"]] = [
        0,
        1,
        0,
        1,
    ]
    games.loc[1, ["team1_win", "team2_win"]] = [0, 1]
    games.loc[1, "team1_stats_json"] = (
        '{"kills": 99, "gold": 99000, "towers": 0, "nashors": 0}'
    )
    players.loc[players.game_id == "2", "player_id"] += "-new"
    players.loc[players.game_id == "2", "player_name"] += "-new"
    actual, _ = build_replay_features(matches, games, players)
    np.testing.assert_allclose(feature_values(actual)[:2], feature_values(expected)[:2])
    assert not np.allclose(feature_values(actual)[2], feature_values(expected)[2])
    assert actual.loc[1, "y_true"] == 0


def test_same_day_inputs_are_frozen_and_permutation_invariant():
    matches, games, players = source_fixture()
    matches.loc[1, "date"] = matches.loc[0, "date"]
    games.loc[1, "date"] = games.loc[0, "date"]
    original, _ = build_replay_features(matches, games, players)
    permuted, _ = build_replay_features(
        matches.iloc[::-1], games.iloc[::-1], players.iloc[::-1]
    )
    original = original.sort_values("golgg_match_id")
    permuted = permuted.sort_values("golgg_match_id")
    np.testing.assert_allclose(feature_values(original), feature_values(permuted))
    np.testing.assert_allclose(
        feature_values(original)[:1], feature_values(original)[1:2]
    )


def test_raw_side_swap_negates_only_odd_evidence():
    matches, games, players = source_fixture()
    original, _ = build_replay_features(matches, games, players)
    for frame in (matches, games):
        for left in list(frame):
            if left.startswith("team1_"):
                right = left.replace("team1_", "team2_", 1)
                if right in frame:
                    frame[left], frame[right] = frame[right].copy(), frame[left].copy()
    swapped, _ = build_replay_features(matches, games, players)
    odd = [name for name in original if name.startswith("d_")]
    even = [name for name in original if name.startswith("c_")]
    np.testing.assert_allclose(original[odd], -swapped[odd], atol=1e-12)
    np.testing.assert_allclose(original[even], swapped[even], atol=1e-12)
    np.testing.assert_array_equal(original.y_true, 1 - swapped.y_true)


def test_new_teams_remain_eligible_without_realized_roster():
    matches, games, players = source_fixture()
    frame, audit = build_replay_features(matches, games, players.iloc[0:0])
    assert frame.golgg_match_id.tolist() == ["1", "2", "3"]
    assert np.isfinite(feature_values(frame)).all()
    assert frame.roster_min_prior_series.eq(0).all()


def test_reversed_map_orientation_keeps_series_features_and_history():
    matches, games, players = source_fixture()
    expected, _ = build_replay_features(matches, games, players)
    for left in list(games):
        if left.startswith("team1_"):
            right = left.replace("team1_", "team2_", 1)
            if right in games:
                games.loc[0, left], games.loc[0, right] = (
                    games.loc[0, right],
                    games.loc[0, left],
                )
    actual, audit = build_replay_features(matches, games, players)
    pd.testing.assert_frame_equal(actual, expected)
    assert audit["excluded_series_count"] == 0


def test_ambiguous_prior_date_retains_old_roster_without_id_order():
    matches, games, players = source_fixture()
    extra_match = matches.iloc[[1]].copy()
    extra_game = games.iloc[[1]].copy()
    extra_players = players.loc[players.game_id == "2"].copy()
    extra_match["match_id"] = "unordered"
    extra_game[["game_id", "match_id"]] = "unordered"
    extra_players[["game_id", "match_id"]] = "unordered"
    extra_players.loc[extra_players.team_id == "a", "player_id"] += "-sub"
    matches = pd.concat([matches, extra_match], ignore_index=True)
    games = pd.concat([games, extra_game], ignore_index=True)
    players = pd.concat([players, extra_players], ignore_index=True)
    frame, audit = build_replay_features(matches, games, players)
    reversed_frame, _ = build_replay_features(
        matches.iloc[::-1], games.iloc[::-1], players.iloc[::-1]
    )
    pd.testing.assert_frame_equal(frame, reversed_frame)
    latest = frame.loc[frame.golgg_match_id == "3"].iloc[0]
    assert latest.d_roster_ambiguous == 1
    assert latest.c_roster_ambiguous == 0.5
    np.testing.assert_allclose(
        latest.d_roster_age_log1p, np.log1p(4) - np.log1p(2), atol=1e-12, rtol=0
    )
    assert latest.roster_min_prior_series == 2
    assert audit["counts"]["ambiguous_roster_team_dates"] == 1


def test_missing_series_ids_recover_from_names_not_first_game_position():
    matches, games, players = source_fixture()
    expected, _ = build_replay_features(matches, games, players)
    games["team1_name"], games["team2_name"] = "Alpha", "Beta"
    matches[["team1_id", "team2_id"]] = None
    for left in list(games):
        if left.startswith("team1_"):
            right = left.replace("team1_", "team2_", 1)
            if right in games:
                games[left], games[right] = games[right].copy(), games[left].copy()
    actual, audit = build_replay_features(matches, games, players)
    pd.testing.assert_frame_equal(actual, expected)
    assert audit["counts"]["recovered_series_identity"] == 3
    games["team1_name"] = "Unrelated"
    excluded, audit = build_replay_features(matches, games, players)
    assert excluded.empty
    assert audit["exclusions"]["invalid_match_identity"] == 3


def test_missing_measurements_have_side_masks_not_zero_statistics():
    matches, games, players = source_fixture()
    games.loc[0, "team1_stats_json"] = '{"kills": null, "gold": 55000}'
    frame, _ = build_replay_features(matches, games, players)
    next_day = frame.loc[frame.golgg_match_id == "2"].iloc[0]
    assert next_day.d_kills_w20 == 0
    assert next_day.d_kills_missing == 1
    assert next_day.c_kills_missing == 0.5
    assert next_day.d_gold_w20 == 10


def test_invalid_terminal_series_cannot_update_later_history():
    matches, games, players = source_fixture()
    matches.loc[1, "team1_score"] = 2
    actual, audit = build_replay_features(matches, games, players)
    expected, _ = build_replay_features(
        matches.drop(index=1),
        games.loc[games.match_id != "2"],
        players.loc[players.match_id != "2"],
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert audit["exclusions"]["nonterminal_score"] == 1


def test_departed_forecast_players_do_not_receive_replacement_outcomes():
    matches, games, players = source_fixture()
    expected, _ = build_replay_features(
        matches.drop(index=1),
        games.loc[games.game_id != "2"],
        players.loc[players.game_id != "2"],
    )
    extra_match = matches.iloc[[1]].copy()
    extra_game = games.iloc[[1]].copy()
    extra_players = players.loc[players.game_id == "2"].copy()
    extra_match["match_id"] = "other-roster"
    extra_game[["game_id", "match_id"]] = "other-roster"
    extra_players[["game_id", "match_id"]] = "other-roster"
    extra_players["player_id"] += "-replacement2"
    players.loc[players.game_id == "2", "player_id"] += "-replacement1"
    actual, _ = build_replay_features(
        pd.concat([matches, extra_match], ignore_index=True),
        pd.concat([games, extra_game], ignore_index=True),
        pd.concat([players, extra_players], ignore_index=True),
    )
    latest = actual.loc[actual.golgg_match_id == "3"].iloc[0]
    reference = expected.loc[expected.golgg_match_id == "3"].iloc[0]
    assert latest.c_roster_ambiguous == 1
    assert latest.roster_min_prior_series == 1
    assert latest.map_prob_player == reference.map_prob_player
    assert latest.map_prob_team != reference.map_prob_team
