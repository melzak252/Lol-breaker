"""Offline repair must preserve results, history and unresolved source evidence."""
from copy import deepcopy
import json

import pytest

from betting_app.tests.test_siamese_research_dataset import match
from scripts.prospective_sports_features import _history_helpers
from scripts.repair_golgg_archive import repair_record, repair_archive


@pytest.fixture(scope="module")
def helpers():
    return _history_helpers()


def reversed_header():
    raw = match(1, "2024-01-01")
    raw.update(sname_t1="Alpha", sname_t2="Beta", won="Alpha", lost="Beta",
               t1_score=0, t2_score=1, t1_win=False, t2_win=True, score="0 - 1")
    raw["games"][0].update(t1_name="Alpha", t2_name="Beta")
    return raw


def test_corroborated_header_repair_preserves_every_game_and_is_idempotent(helpers):
    raw = reversed_header()
    original = deepcopy(raw)
    fixed, evidence, issues = repair_record(raw, helpers)
    assert (fixed["t1_score"], fixed["t2_score"], fixed["t1_win"], fixed["t2_win"]) == (1, 0, True, False)
    # The source result-cell string is raw provenance, not link-ordered scores.
    assert fixed["score"] == original["score"]
    assert raw == original
    assert fixed["games"] == original["games"]
    assert evidence["winner_id"] == "a"
    assert not issues
    again, evidence, issues = repair_record(fixed, helpers)
    assert again == fixed and evidence is None and not issues


def test_score_repair_tracks_exact_team_ids_across_blue_red_rotation(helpers):
    raw = reversed_header()
    raw.update(best_of=3, t2_score=2, score="0 - 2")
    game = deepcopy(raw["games"][0])
    game["game_id"] = "2"
    for field in ("id", "name", "win", "players", "stats"):
        game[f"t1_{field}"], game[f"t2_{field}"] = game[f"t2_{field}"], game[f"t1_{field}"]
    raw["games"].append(game)
    fixed, evidence, issues = repair_record(raw, helpers)
    assert (fixed["t1_score"], fixed["t2_score"]) == (2, 0)
    assert fixed["games"] == raw["games"]
    assert evidence["winner_id"] == "a" and not issues


def test_disagreeing_winner_text_cannot_authorize_a_label_repair(helpers):
    raw = reversed_header()
    raw["won"] = "Beta"
    fixed, evidence, issues = repair_record(raw, helpers)
    assert fixed == raw and evidence is None
    assert issues


def test_incomplete_maps_are_not_relabelled_as_a_clean_sweep(helpers):
    raw = reversed_header()
    raw.update(best_of=3, games_played=3, t1_score=1, t2_score=2)
    game = deepcopy(raw["games"][0])
    game["game_id"] = "2"
    raw["games"].append(game)
    fixed, evidence, issues = repair_record(raw, helpers)
    assert fixed == raw and evidence is None
    assert issues


def test_missing_measurements_remain_missing_after_metadata_repair(helpers):
    raw = reversed_header()
    raw["games"][0]["t1_players"]["0"]["stats"]["vspm"] = None
    fixed, evidence, issues = repair_record(raw, helpers)
    assert fixed["t1_win"] is True
    assert fixed["games"][0]["t1_players"]["0"]["stats"]["vspm"] is None
    assert "missing_required_w20_stats" in issues


@pytest.mark.parametrize("bad_games", [None, 42])
def test_archive_keeps_unresolved_records_and_never_overwrites_source(tmp_path, bad_games):
    source = tmp_path / "history.json"
    bad = reversed_header()
    bad["match_id"] = "2"
    bad["games"] = bad_games
    records = [reversed_header(), bad]
    source.write_text(json.dumps(records))
    original = source.read_bytes()
    output = tmp_path / "repair"
    report = repair_archive(source, output)
    assert source.read_bytes() == original
    saved = json.loads((output / "matches.json").read_text())
    assert len(saved) == 2 and saved[1] == bad
    assert saved[0]["t1_win"] is True
    assert report["repaired_series"] == 1
    assert report["unresolved_series"] == 1
    assert report["status"] == "blocked_unresolved_history"
    with pytest.raises(FileExistsError):
        repair_archive(source, output)
