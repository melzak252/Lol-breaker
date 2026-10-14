"""Frozen raw meta ingestion: causal clocks and explicit provenance."""
from copy import deepcopy

import numpy as np
import pytest

from src.features.c0_meta_features import STATS
from src.features.c0_meta_source import parse_records, read_raw_meta_source


def record(identity="1", day="2023-12-20", champion="A", patch=None):
    return dict(match_id=identity, tournament_name=" LCK ", patch=patch,
                games=[dict(date=day, t1_players={"top": dict(player_id=1, champion_id=champion,
                            stats={"kills": 0, "deaths": None, "assists": True, "csm": float("nan")})},
                            t2_players={"support": dict(player_id=None, champion_id="B", stats={})})])


@pytest.mark.parametrize("champion", [0, "unknown"])
def test_explicit_unknown_champion_keeps_observed_mass_separate_from_absence(champion):
    from src.features.c0_meta_features import build_meta_features

    raw = record(champion=champion)
    raw["games"][0]["t2_players"]["support"]["champion_id"] = None
    source = parse_records([raw], {})
    batch = build_meta_features(source.observations, [
        dict(row_id="target", cutoff_day="2024-01-01", competition="LCK",
             roster_a=["1", None, None, None, None], roster_b=[None] * 5)
    ], champion_ids=source.champion_ids)
    assert batch.arrays["player_champions"][0, 0, 0, 0] == 1
    assert batch.arrays["player_stats"][0, 0, 0, 37] == 1
    assert batch.arrays["meta_champions"][0, 0, 4, 0] == 0
    assert source.champion_ids == ()
    assert source.audit["missing_champion_id_count"] == 1
    assert source.audit["unknown_champion_id_count"] == 1


def test_vocabulary_uses_both_clocks_including_records_outside_retained_window():
    old = record("old", "2020-01-01", "ancient")
    late = record("late", "2020-01-02", "late-only")
    future = record("future", "2024-01-01", "future-only")
    unknown = record("unknown", "2023-12-19", "unknown")
    result = parse_records([old, late, record(), future, unknown], {"late": "2024-01-01"})
    assert result.champion_ids == ("A", "B", "ancient")
    assert {o["series_id"] for o in result.observations} == {"1", "future", "unknown"}
    assert result.audit["source_series_count"] == 5
    assert result.audit["source_map_count"] == 5
    assert result.audit["source_observation_count"] == 10
    assert result.audit["retained_observation_count"] == 6


def test_release_floor_is_original_last_map_and_does_not_shift_played_clock():
    raw = record()
    raw["games"].append(deepcopy(raw["games"][0]))
    raw["games"][1]["date"] = "2024-01-03"
    result = parse_records([raw], {"1": "2023-12-21"})
    assert [o["played_day"] for o in result.observations] == ["2023-12-20"] * 2 + ["2024-01-03"] * 2
    assert {o["release_day"] for o in result.observations} == {"2024-01-03"}
    assert result.champion_ids == ()
    assert result.series.iloc[0]["effective_label_release_day"] == "2024-01-03"


def test_explicit_first_map_patch_scope_fallback_unknown_and_conflicts():
    raw = record(patch="13.23")
    raw["games"][0]["patch"] = "13.24"
    raw["games"].append(deepcopy(raw["games"][0]))
    raw["games"][1]["patch"] = "14.1"
    fallback = record("fallback", patch="13.22")
    unknown = record("unknown", patch="unknown")
    result = parse_records([raw, fallback, unknown], {})
    catalog = result.series.set_index("golgg_match_id")
    assert catalog.loc["1", "patch"] == "13.24"
    assert catalog.loc["1", "patch_source"] == "game.patch"
    assert catalog.loc["fallback", "patch_source"] == "raw.patch"
    assert not isinstance(catalog.loc["unknown", "patch"], str)
    assert [o["patch"] for o in result.observations[:4]] == ["13.24", "13.24", "14.1", "14.1"]
    assert result.audit["patch"]["game_match_conflict_count"] == 2
    assert result.audit["patch"]["within_series_conflict_count"] == 1
    assert result.audit["patch"]["unknown_map_count"] == 1


def test_stat_missingness_preserves_numeric_zero_and_missing_player_global_records():
    result = parse_records([record()], {})
    first, missing_player = result.observations
    assert first["values12"].dtype == np.float32
    assert first["observed12"].dtype == np.bool_
    assert first["values12"].shape == (len(STATS),)
    assert first["observed12"].tolist() == [True] + [False] * 11
    assert not first["values12"].any()
    assert first["player_id"] == "1"
    assert first["side"] == "t1" and first["map_index"] == 1
    assert missing_player["player_id"] is None
    assert result.audit["missing_player_id_count"] == 1


@pytest.mark.parametrize("kind", ["series", "within_side", "across_sides"])
def test_duplicate_source_and_roster_identities_raise(kind):
    raw = record()
    records = [raw]
    if kind == "series":
        records.append(deepcopy(raw))
    elif kind == "within_side":
        raw["games"][0]["t1_players"]["mid"] = dict(player_id="1", stats={})
    else:
        raw["games"][0]["t2_players"]["support"]["player_id"] = "1"
    with pytest.raises(ValueError):
        parse_records(records, {})


def test_real_loader_rejects_unpinned_reader_before_import(tmp_path):
    reader = tmp_path / "reader.py"
    reader.write_text("raise RuntimeError('must not import')\n")
    with pytest.raises(ValueError):
        read_raw_meta_source(tmp_path / "raw.json", tmp_path / "release.json", reader)


@pytest.mark.parametrize("kind", ["side", "player", "stats", "too_many"])
def test_malformed_roster_payload_is_not_silently_filtered(kind):
    raw = record()
    game = raw["games"][0]
    if kind == "side":
        game["t1_players"] = None
    elif kind == "player":
        game["t1_players"]["top"] = None
    elif kind == "stats":
        game["t1_players"]["top"]["stats"] = [1, 2]
    else:
        game["t1_players"] = {str(i): dict(player_id=i, stats={}) for i in range(6)}
    with pytest.raises(ValueError):
        parse_records([raw], {})
