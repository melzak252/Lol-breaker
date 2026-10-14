"""Consumer contracts for history-only champion data (no fits or live data)."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.features.champion_history_data import prepare_history_data

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")


def source_series(identity, day, *, best_of=1, map_days=None, champions=None):
    map_days = map_days or [day]
    games = []
    for index, map_day in enumerate(map_days):
        game = dict(game_id=f"{identity}-{index}", match_id=identity, date=map_day,
                    t1_id="A", t2_id="B", t1_win=True, t2_win=False,
                    game_duration=1800, t1_stats={"kills": 10}, t2_stats={"kills": 8})
        for side, prefix in (("t1", "a"), ("t2", "b")):
            game[side + "_players"] = {
                role: {"player_id": prefix + str(slot), "champion_id":
                       (str(slot + 110) if side == "t2" else champions[slot] if champions else str(slot + 10)),
                       "stats": {"kills": slot, "deaths": 1, "assists": 3,
                                 "gpm": 350 + slot}}
                for slot, role in enumerate(ROLES)
            }
        games.append(game)
    return dict(match_id=identity, date=day, best_of=best_of, t1_id="A", t2_id="B",
                tournament_name="Exact Competition", games=games)


def base_frame(records):
    return pd.DataFrame([dict(match_id=r["match_id"], date=r["date"],
                              release_day=max(g["date"] for g in r["games"]),
                              best_of=r["best_of"], team_a="A", team_b="B",
                              roster_a=["a" + str(i) for i in range(5)],
                              roster_b=["b" + str(i) for i in range(5)],
                              tournament=r["tournament_name"]) for r in records])


def build(tmp_path, records, base=None, **kwargs):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(records))
    return prepare_history_data(path, base_frame(records) if base is None else base,
                                rules={"bindings": []}, **kwargs)


def assert_inputs_equal(left, right, row):
    for key in ("history", "champions", "roles", "mask", "players", "role_probs",
                "context", "patches", "competitions"):
        np.testing.assert_array_equal(left[key][row], right[key][row], err_msg=key)


def test_target_fields_are_not_inputs_and_later_maps_share_one_snapshot(tmp_path):
    old = source_series("old", "2025-01-01")
    target = source_series("target", "2025-01-03", best_of=3,
                           map_days=["2025-01-03", "2025-01-04"])
    original = [old, target]
    base = base_frame(original)
    _, maps, arrays, _, _ = build(tmp_path, original, base)
    changed = deepcopy(original)
    for game in changed[1]["games"]:
        game["patch"] = "99.99"
        game["game_duration"] = 7000
        game["t1_stats"]["kills"] = 200
        game["t1_players"]["TOP"]["champion_id"] = "NEW_TARGET_ONLY"
        game["t1_players"]["TOP"]["stats"]["gpm"] = 9000
        game["t1_players"]["TOP"]["role"] = "SUPPORT"
    _, changed_maps, changed_arrays, _, _ = build(tmp_path, changed, base)
    assert_inputs_equal(arrays, changed_arrays, 1)
    assert len(maps.loc[maps.row_index == 1]) == 2
    assert changed_maps.loc[changed_maps.row_index == 1, "kills"].tolist() == [208, 208]
    assert arrays["mask"][1].sum() == 10


def test_series_release_is_atomic_and_same_day_unavailable(tmp_path):
    delayed = source_series("delayed", "2025-01-01", best_of=3,
                            map_days=["2025-01-01", "2025-01-04"])
    middle = source_series("middle", "2025-01-02")
    same_day = source_series("same", "2025-01-04")
    after = source_series("after", "2025-01-05")
    _, _, arrays, _, _ = build(tmp_path, [delayed, middle, same_day, after])
    assert not arrays["mask"][1].any()
    assert arrays["mask"][2].sum() == 10  # only middle, not delayed or same
    assert arrays["mask"][3].sum() == 40


def test_future_source_mutation_cannot_change_prior_inputs(tmp_path):
    records = [source_series("a", "2025-01-01"), source_series("b", "2025-01-02"),
               source_series("future", "2026-02-01")]
    base = base_frame(records)
    original = build(tmp_path, records, base)[2]
    changed = deepcopy(records)
    changed[-1]["games"][0]["t1_players"]["TOP"]["stats"]["kills"] = 99
    changed[-1]["games"][0]["t1_players"]["TOP"]["champion_id"] = "FUTURE"
    altered = build(tmp_path, changed, base)[2]
    assert_inputs_equal(original, altered, 1)


def test_roster_permutation_and_team_swap_preserve_identity_alignment(tmp_path):
    records = [source_series("a", "2025-01-01"), source_series("b", "2025-01-02")]
    base = base_frame(records)
    arrays = build(tmp_path, records, base)[2]
    permutation = [4, 2, 0, 3, 1]
    permuted = base.copy(deep=True)
    permuted.at[1, "roster_a"] = [base.at[1, "roster_a"][i] for i in permutation]
    changed = build(tmp_path, records, permuted)[2]
    for key in ("history", "champions", "roles", "mask", "players", "role_probs"):
        np.testing.assert_array_equal(changed[key][1, 0], arrays[key][1, 0, permutation])
    swapped = base.copy(deep=True)
    for a, b in (("team_a", "team_b"), ("roster_a", "roster_b")):
        swapped[a], swapped[b] = base[b], base[a]
    series, _, swapped_arrays, _, _ = build(tmp_path, records, swapped)
    np.testing.assert_array_equal(swapped_arrays["history"], arrays["history"][:, ::-1])
    assert series.score_index.tolist() == [3., 3.]


def test_cold_unknown_player_is_not_champion_missing_or_unseen(tmp_path):
    record = source_series("a", "2026-01-01")
    base = base_frame([record])
    base.at[0, "roster_a"] = [None] * 5
    _, _, arrays, schema, _ = build(tmp_path, [record], base)
    assert not arrays["mask"].any()
    assert np.isnan(arrays["history"]).all()
    np.testing.assert_array_equal(arrays["role_probs"][0, 0, :, 0], np.ones(5))
    known = schema["player_feature_names"].index("known_player_identity")
    assert not arrays["players"][0, 0, :, known].any()


def test_missing_champion_does_not_remove_stats_or_independent_labels(tmp_path):
    old = source_series("a", "2025-01-01", champions=[None, "0", "12", "13", "14"])
    target = source_series("b", "2025-01-02")
    target["games"][0]["t1_players"]["TOP"]["champion_id"] = None
    series, maps, arrays, schema, audit = build(tmp_path, [old, target])
    assert arrays["champions"][1, 0, 0, -1] == 0
    assert arrays["champions"][1, 0, 1, -1] == 0
    assert arrays["champion_status"][1, 0, 0, -1] == 1
    assert arrays["champion_status"][1, 0, 1, -1] == 2
    assert arrays["history"][1, 0, 0, -1, schema["history_feature_names"].index("kills")] == 0
    assert arrays["mask"][1, 0, 0, -1]
    assert maps.kills.notna().all() and series.score_index.notna().all()
    assert audit["canonical_series"] == 2


def test_vocabulary_is_closed_train_and_unseen_is_observed(tmp_path):
    records = [source_series("train", "2025-06-28"),
               source_series("cal", "2025-07-02", champions=["999", "998", "997", "996", "995"]),
               source_series("assess", "2026-01-02")]
    _, _, arrays, schema, _ = build(tmp_path, records)
    assert "999" not in schema["champion_vocabulary"]
    assert arrays["champions"][2, 0, 0, -1] == 1
    assert arrays["mask"][2, 0, 0, -1]


def test_bo2_is_history_but_not_a_terminal_score_target(tmp_path):
    old = source_series("bo2", "2025-01-01", best_of=2,
                        map_days=["2025-01-01", "2025-01-01"])
    old["games"][1].update(t1_win=False, t2_win=True)
    target = source_series("target", "2025-01-02")
    _, _, arrays, _, audit = build(tmp_path, [old, target], base_frame([target]))
    assert arrays["mask"][0].sum() == 20
    assert audit["source_formats"]["2"] == 1


def test_map_task_absences_remain_independent_and_remakes_named(tmp_path):
    records = [source_series("a", "2025-01-01"), source_series("b", "2025-01-02")]
    records[0]["games"][0]["game_duration"] = None
    records[1]["games"][0]["remake"] = True
    series, maps, _, _, audit = build(tmp_path, records)
    assert maps.at[0, "kills"] == 18 and pd.isna(maps.at[0, "duration_minutes"])
    assert "duration_missing" in maps.at[0, "duration_reason"]
    assert pd.isna(maps.at[1, "kills"]) and "remake" in maps.at[1, "kills_reason"]
    assert pd.isna(series.at[1, "score_index"])
    assert sum(audit["canonical_map_task_reasons"]["kills"].values()) == len(maps)


def test_initial_advantage_is_named_and_does_not_remove_prop_targets(tmp_path):
    record = source_series("a", "2025-01-01")
    record["initial_wins"] = {"A": 1}
    series, maps, _, _, _ = build(tmp_path, [record])
    assert pd.isna(series.at[0, "score_index"])
    assert series.at[0, "score_reason"] == "map_advantage"
    assert maps.at[0, "kills"] == 18


def test_exact_rule_binding_does_not_infer_from_substring_or_date(tmp_path):
    record = source_series("a", "2025-01-01")
    path = tmp_path / "source.json"
    path.write_text(json.dumps([record]))
    rules = {"bindings": [{"competition_edition_phase": "Exact Competition Playoffs",
                           "draft_mode": "HARD_FEARLESS", "evidence_key": "official"}]}
    series, _, _, _, _ = prepare_history_data(path, base_frame([record]), rules=rules)
    assert series.at[0, "draft_mode"] == "UNKNOWN"


def test_last_history_length_and_source_order_are_deterministic(tmp_path):
    records = [source_series(str(i), f"2025-01-{i + 1:02d}") for i in range(20)]
    base = base_frame([records[-1]])
    normal = build(tmp_path, records, base, history_length=16)[2]
    reversed_ = build(tmp_path, records[::-1], base, history_length=16)[2]
    assert_inputs_equal(normal, reversed_, 0)
    assert normal["mask"].sum() == 160


def test_conflicting_explicit_player_identity_is_rejected(tmp_path):
    record = source_series("a", "2025-01-01")
    record["games"][0]["t1_players"]["TOP"]["id"] = "different"
    with pytest.raises(ValueError, match="conflicting.*player"):
        build(tmp_path, [record])


def test_train_vocabulary_requires_entire_series_release_before_boundary(tmp_path):
    delayed = source_series("delayed", "2025-06-30", best_of=3,
                            map_days=["2025-06-30", "2025-07-02"],
                            champions=["DELAYED", "D2", "D3", "D4", "D5"])
    target = source_series("target", "2025-07-03")
    _, _, arrays, schema, _ = build(tmp_path, [delayed, target], base_frame([target]))
    assert "DELAYED" not in schema["champion_vocabulary"]
    assert arrays["champions"][0, 0, 0, -1] == 1
    assert arrays["mask"][0].sum() == 20


def test_historical_patch_and_competition_are_closed_train_embeddings(tmp_path):
    records = [source_series("train", "2025-06-28"),
               source_series("cal", "2025-07-02"),
               source_series("assess", "2026-01-02")]
    records[0]["patch"] = "25.10"
    records[1]["patch"] = "25.99"
    records[1]["tournament_name"] = "New Edition"
    _, _, arrays, schema, _ = build(tmp_path, records)
    assert schema["patch_vocabulary"] == {"25.10": 2}
    assert "New Edition" not in schema["competition_vocabulary"]
    assert arrays["patches"][2, 0, 0, -1] == 1
    assert arrays["competitions"][2, 0, 0, -1] == 1
    assert schema["champion_numeric_channels"] == []
    assert "champion_status" not in schema["model_input_keys"]

def test_ambiguous_duplicate_champions_mask_only_champion_evidence_not_statistics(tmp_path):
    old = source_series("old", "2025-01-01")
    old["games"][0]["t2_players"]["TOP"]["champion_id"] = old["games"][0]["t1_players"]["TOP"]["champion_id"]
    target = source_series("target", "2025-01-02")
    series, maps, arrays, schema, audit = build(tmp_path, [old, target])
    assert arrays["champions"][1, 0, 0, -1] == 0
    assert arrays["champions"][1, 1, 0, -1] == 0
    assert arrays["champion_status"][1, 0, 0, -1] == 2
    assert arrays["champions"][1, 0, 1, -1] >= 2
    np.testing.assert_allclose(arrays["history"][1, :, 0, -1, schema["history_feature_names"].index("kills")], [0, 0])
    assert maps.kills.notna().all() and series.score_index.notna().all()
    assert audit["ambiguous_duplicate_champion_maps"] == 1
