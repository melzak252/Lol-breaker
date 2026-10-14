"""Outcome-independent identity contracts for production market comparisons."""

import pandas as pd
import pytest

from src.analysis.production_market_identity import build_identity_tables


def identity_inputs():
    matches = pd.DataFrame([{
        "canonical_match_id": 7, "team_a_name": "Alpha", "team_b_name": "Beta",
        "normalized_team_a": "alpha", "normalized_team_b": "beta",
        "start_time_normalized": "2026-06-01T18:00:00+00:00", "best_of": 3,
        "winner_side": "team_a",
    }])
    mappings = pd.DataFrame([{"canonical_match_id": 7, "golgg_match_id": "42"}])
    odds = pd.DataFrame([{
        "id": 11, "canonical_match_id": 7, "bookmaker": "book",
        "raw_team_a": "Beta", "raw_team_b": "Alpha",
    }])
    aliases = pd.DataFrame(columns=[
        "is_active", "is_blocked", "review_status", "source_system", "league_pattern",
        "tournament_pattern", "valid_from", "valid_to", "normalized_name", "alias",
        "raw_name", "normalized_alias",
    ])
    sources = {"fixture_source": [{
        "match_id": "42", "t1_id": "1", "t2_id": "2", "t1_name": None, "t2_name": None,
        "date": "2026-06-01", "best_of": 3,
        "games": [{"t1_id": "2", "t2_id": "1", "t1_name": "Beta", "t2_name": "Alpha"}],
    }]}
    predictions = pd.DataFrame([{
        "golgg_match_id": "42", "team1_id": "1", "team2_id": "2",
        "date": "2026-06-01", "best_of": 3, "y": 1, "p_full": 0.7,
    }])
    return matches, mappings, odds, aliases, sources, predictions


def test_first_map_identity_is_atomic_and_quotes_align_once():
    identities, quotes, _ = build_identity_tables(*identity_inputs())
    assert identities.iloc[0].identity_status == "verified"
    assert bool(identities.iloc[0].canonical_a_is_team1)
    assert quotes.iloc[0].quote_identity_status == "verified"
    assert not bool(quotes.iloc[0].raw_a_is_team1)
    assert not bool(quotes.iloc[0].raw_a_is_canonical_a)


def test_result_changes_cannot_change_identity_orientation():
    inputs = identity_inputs()
    identities, quotes, _ = build_identity_tables(*inputs)
    inputs[0].loc[0, "winner_side"] = "team_b"
    inputs[-1].loc[0, "y"] = 0
    changed_identities, changed_quotes, _ = build_identity_tables(*inputs)
    pd.testing.assert_frame_equal(identities, changed_identities)
    pd.testing.assert_frame_equal(quotes, changed_quotes)


def test_book_scoped_alias_cannot_resolve_global_match_identity():
    inputs = list(identity_inputs())
    inputs[0].loc[0, ["team_a_name", "normalized_team_a"]] = ["ShortName", "shortname"]
    inputs[3] = pd.DataFrame([{
        "is_active": 1, "is_blocked": 0, "review_status": "approved", "source_system": "specific_book",
        "league_pattern": None, "tournament_pattern": None, "valid_from": None, "valid_to": None,
        "normalized_name": "Alpha", "alias": "ShortName", "raw_name": None, "normalized_alias": None,
    }])
    identities, _, _ = build_identity_tables(*inputs)
    assert identities.iloc[0].identity_status != "verified"


def test_approved_global_alias_that_connects_both_sides_is_ambiguous():
    inputs = list(identity_inputs())
    inputs[3] = pd.DataFrame([{
        "is_active": 1, "is_blocked": 0, "review_status": "approved",
        "normalized_name": "Alpha", "alias": "Beta",
    }])
    identities, _, _ = build_identity_tables(*inputs)
    assert identities.iloc[0].identity_status == "ambiguous_alias_identity"


def test_incomplete_source_tuple_is_not_repaired_from_header_or_prediction():
    inputs = list(identity_inputs())
    inputs[4]["fixture_source"][0]["games"] = [{
        "t1_id": "2", "t2_id": "1", "t1_name": "Beta", "t2_name": None,
    }]
    identities, quotes, _ = build_identity_tables(*inputs)
    assert identities.iloc[0].identity_status == "missing_or_duplicate_source_id"
    assert quotes.iloc[0].quote_identity_status == "match_identity_unresolved"


def test_duplicate_source_candidates_are_not_selected_by_order():
    inputs = list(identity_inputs())
    inputs[4]["fixture_source"].append({
        "match_id": "42", "t1_id": "1", "t2_id": "2",
        "t1_name": "Alpha", "t2_name": "Beta", "date": "2026-06-01", "best_of": 3,
    })
    identities, _, _ = build_identity_tables(*inputs)
    assert identities.iloc[0].identity_status == "missing_or_duplicate_source_id"


def test_reversed_source_tuple_is_reoriented_with_forecast_ids_and_retained():
    identities, quotes, _ = build_identity_tables(*identity_inputs())
    identity = identities.iloc[0]
    assert (identity.team1_id, identity.team2_id) == ("1", "2")
    assert (identity.source_team1_id, identity.source_team2_id) == ("2", "1")
    assert (identity.golgg_team1, identity.golgg_team2) == ("Alpha", "Beta")
    assert bool(identity.source_order_reversed_to_prediction)
    assert bool(identity.canonical_a_is_team1)
    assert bool(quotes.iloc[0].raw_a_is_team1) is False


def test_duplicate_quote_primary_key_is_rejected_without_collapsing_quotes():
    inputs = list(identity_inputs())
    inputs[2] = pd.concat([inputs[2], inputs[2]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate quote primary key"):
        build_identity_tables(*inputs)


def test_mixed_numeric_and_string_source_ids_roundtrip_through_parquet(tmp_path):
    first = list(identity_inputs())
    second = list(identity_inputs())
    for index in (0, 1, 2):
        second[index]["canonical_match_id"] = 8
    second[1]["golgg_match_id"] = "43"
    second[2]["id"] = 12
    second[-1]["golgg_match_id"] = "43"
    second[4]["fixture_source"][0]["match_id"] = "43"
    second[4]["fixture_source"][0]["games"][0].update(t1_id=2, t2_id=1)
    for index in (0, 1, 2, 5):
        first[index] = pd.concat([first[index], second[index]], ignore_index=True)
    first[4]["fixture_source"].extend(second[4]["fixture_source"])

    identities, quotes, _ = build_identity_tables(*first)
    identities.to_parquet(tmp_path / "identities.parquet", index=False)
    quotes.to_parquet(tmp_path / "quotes.parquet", index=False)

    restored = pd.read_parquet(tmp_path / "identities.parquet")
    assert restored.identity_status.tolist() == ["verified", "verified"]
    assert restored.team1_id.tolist() == ["1", "1"]
    assert restored.canonical_a_is_team1.tolist() == [True, True]
