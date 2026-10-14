"""Observable quote-time and paired-cohort benchmark contracts."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.production_market_benchmark import (
    market_consensus,
    paired_cluster_interval,
    run_production_market_benchmark,
    select_market_quotes,
    validate_fixture_cohort,
)


def test_cutoff_consensus_gives_each_book_one_vote_and_ignores_future():
    start = pd.Timestamp("2026-06-02T12:00:00Z")
    cutoff = start - pd.Timedelta(hours=24)
    quotes = pd.DataFrame([
        {"id": 1, "bookmaker": "A", "scraped_at": cutoff - pd.Timedelta(hours=2), "p_market_quote": 0.1},
        {"id": 2, "bookmaker": "A", "scraped_at": cutoff, "p_market_quote": 0.8},
        {"id": 3, "bookmaker": "B", "scraped_at": cutoff - pd.Timedelta(minutes=5), "p_market_quote": 0.2},
        {"id": 4, "bookmaker": "B", "scraped_at": cutoff + pd.Timedelta(seconds=1), "p_market_quote": 0.99},
        {"id": 5, "bookmaker": "C", "scraped_at": cutoff - pd.Timedelta(hours=24, seconds=1), "p_market_quote": 0.01},
    ])
    quotes["canonical_match_id"] = 7
    quotes["start_dt"] = start
    selected = select_market_quotes(quotes, "24h")
    assert set(selected.id) == {2, 3}
    consensus = market_consensus(selected)
    assert consensus.iloc[0].p_market == pytest.approx(0.5)


def test_open_does_not_import_a_later_book_and_close_has_no_old_fallback():
    start = pd.Timestamp("2026-06-02T12:00:00Z")
    quotes = pd.DataFrame([
        {"id": 1, "bookmaker": "A", "scraped_at": start - pd.Timedelta(days=3), "p_market_quote": 0.3},
        {"id": 2, "bookmaker": "B", "scraped_at": start - pd.Timedelta(days=2), "p_market_quote": 0.9},
        {"id": 3, "bookmaker": "A", "scraped_at": start, "p_market_quote": 0.99},
    ])
    quotes["canonical_match_id"] = 7
    quotes["start_dt"] = start
    assert list(select_market_quotes(quotes, "OPEN").id) == [1]
    assert select_market_quotes(quotes, "CLOSE").empty


def test_cluster_weighting_distinguishes_matches_from_months():
    result = paired_cluster_interval(np.array([-1.0, -1.0, -1.0, 1.0]), ["2026-05"] * 3 + ["2026-06"])
    assert result["delta"] == pytest.approx(-0.5)
    assert result["equal_block_delta"] == pytest.approx(0.0)
    assert result["blocks"] == 2


def test_outcome_conflict_is_quarantined_without_flipping_sides():
    matches = pd.DataFrame([{
        "canonical_match_id": 7, "start_time_normalized": "2026-06-01T18:00:00Z",
        "best_of": 3, "winner_side": "team_b", "league": "league",
    }])
    identities = pd.DataFrame([{
        "canonical_match_id": 7, "golgg_match_id": "42", "identity_status": "verified",
        "canonical_a_is_team1": True, "prediction_team_ids_match": True,
        "prediction_date_matches_source": True, "prediction_best_of_matches_source": True,
    }])
    predictions = pd.DataFrame([{
        "golgg_match_id": "42", "team1_id": "1", "team2_id": "2", "date": "2026-06-01",
        "best_of": 3, "y": 1, "p_full": 0.7, "feature_source_max_day": "2026-05-30",
    }])
    audit = validate_fixture_cohort(matches, identities, predictions)
    assert audit.iloc[0].exclusion_reason == "outcome_conflict_after_identity_alignment"
    assert bool(audit.iloc[0].canonical_a_is_team1)


def _valid_cohort_inputs(winner_side="team_a"):
    matches = pd.DataFrame([{
        "canonical_match_id": 7, "start_time_normalized": "2026-06-01T18:00:00Z",
        "best_of": 3, "winner_side": winner_side, "league": "league",
    }])
    identities = pd.DataFrame([{
        "canonical_match_id": 7, "golgg_match_id": "42", "identity_status": "verified",
        "canonical_a_is_team1": True, "prediction_team_ids_match": True,
        "prediction_date_matches_source": True, "prediction_best_of_matches_source": True,
    }])
    predictions = pd.DataFrame([{
        "golgg_match_id": "42", "team1_id": "1", "team2_id": "2", "date": "2026-06-01",
        "best_of": 3, "y": 1, "p_full": 0.7, "feature_source_max_day": "2026-05-30",
    }])
    return matches, identities, predictions


def test_naive_fixture_start_is_quarantined():
    matches, identities, predictions = _valid_cohort_inputs()
    matches.loc[0, "start_time_normalized"] = "2026-06-01T18:00:00"
    audit = validate_fixture_cohort(matches, identities, predictions)
    assert audit.iloc[0].exclusion_reason == "missing_or_naive_start_time"


def test_canonical_best_of_must_match_the_forecast_series():
    matches, identities, predictions = _valid_cohort_inputs()
    matches.loc[0, "best_of"] = 5
    audit = validate_fixture_cohort(matches, identities, predictions)
    assert audit.iloc[0].exclusion_reason == "best_of_mismatch"


def test_missing_winner_is_quarantined_before_outcome_comparison():
    matches, identities, predictions = _valid_cohort_inputs()
    matches.loc[0, "winner_side"] = None
    audit = validate_fixture_cohort(matches, identities, predictions)
    assert audit.iloc[0].exclusion_reason == "missing_winner"


def test_reversed_identity_accepts_float_labels_with_arrow_team_names():
    matches, identities, predictions = _valid_cohort_inputs(winner_side="team_b")
    matches["winner_side"] = matches["winner_side"].astype("string[pyarrow]")
    identities["canonical_a_is_team1"] = False
    predictions["y"] = predictions["y"].astype(float)
    audit = validate_fixture_cohort(matches, identities, predictions)
    assert audit.iloc[0].exclusion_reason == ""


def test_duplicate_golgg_fixture_ids_are_quarantined_for_every_fixture():
    matches, identities, predictions = _valid_cohort_inputs()
    second_match = matches.iloc[0].copy()
    second_match["canonical_match_id"] = 8
    matches = pd.concat([matches, second_match.to_frame().T], ignore_index=True)
    second_identity = identities.iloc[0].copy()
    second_identity["canonical_match_id"] = 8
    identities = pd.concat([identities, second_identity.to_frame().T], ignore_index=True)
    audit = validate_fixture_cohort(matches, identities, predictions)
    assert audit.exclusion_reason.tolist() == ["duplicate_golgg_fixture"] * 2


def test_empty_horizon_preserves_selected_quote_schema():
    quotes = pd.DataFrame(columns=[
        "id", "canonical_match_id", "bookmaker", "scraped_at", "start_dt", "p_market_quote",
    ])
    selected = select_market_quotes(quotes, "24h")
    assert selected.empty
    assert "cutoff" in selected.columns


def test_close_includes_exact_two_hour_boundary_but_not_start_time():
    start = pd.Timestamp("2026-06-02T12:00:00Z")
    quotes = pd.DataFrame([
        {"id": 1, "bookmaker": "A", "scraped_at": start - pd.Timedelta(hours=2), "p_market_quote": 0.4},
        {"id": 2, "bookmaker": "B", "scraped_at": start, "p_market_quote": 0.7},
    ])
    quotes["canonical_match_id"] = 7
    quotes["start_dt"] = start
    assert list(select_market_quotes(quotes, "CLOSE").id) == [1]


def test_naive_quote_timestamp_is_not_treated_as_utc():
    quotes = pd.DataFrame([{
        "id": 1, "canonical_match_id": 7, "bookmaker": "A",
        "scraped_at": pd.Timestamp("2026-06-02T10:00:00"),
        "start_dt": pd.Timestamp("2026-06-02T12:00:00Z"), "p_market_quote": 0.4,
    }])
    assert select_market_quotes(quotes, "CLOSE").empty


def test_quote_selector_rejects_invalid_prices_even_with_a_probability():
    start = pd.Timestamp("2026-06-02T12:00:00Z")
    quotes = pd.DataFrame([{
        "id": 1, "canonical_match_id": 7, "bookmaker": "A",
        "scraped_at": start - pd.Timedelta(hours=1), "start_dt": start,
        "odds_a": 1.0, "odds_b": 2.0, "p_market_quote": 0.25,
    }])
    assert select_market_quotes(quotes, "CLOSE").empty


def test_complete_snapshot_run_scores_quote_horizons_and_calendar_blocks(tmp_path):
    raw_dir, identity_dir = tmp_path / "raw", tmp_path / "identities"
    raw_dir.mkdir()
    identity_dir.mkdir()
    matches, identities, forecasts = _valid_cohort_inputs()
    forecast_path = tmp_path / "forecasts.parquet"
    forecasts.to_parquet(forecast_path, index=False)
    quotes = pd.DataFrame([
        {"id": 1, "scraped_at": "2026-05-29T18:00:00Z", "odds_a": 2.0, "odds_b": 3.0},
        {"id": 2, "scraped_at": "2026-05-31T18:00:00Z", "odds_a": 1.5, "odds_b": 4.5},
        {"id": 3, "scraped_at": "2026-06-01T17:00:00Z", "odds_a": 1.1, "odds_b": 9.9},
    ])
    quotes["canonical_match_id"] = 7
    quotes["bookmaker"] = "book"
    quotes["market_type"] = "match_winner"
    quotes["is_live"] = False
    alignment = quotes[["id", "canonical_match_id", "bookmaker"]].copy()
    alignment["quote_identity_status"] = "verified"
    alignment["raw_a_is_canonical_a"] = True
    alignment["raw_a_is_team1"] = True
    raw_frames = {
        "matches": matches,
        "odds": quotes,
        "mappings": pd.DataFrame([{"canonical_match_id": 7, "golgg_match_id": "42"}]),
        "aliases": pd.DataFrame(columns=["id"]),
    }
    for name, frame in raw_frames.items():
        frame.to_parquet(raw_dir / f"{name}.parquet", index=False)
    identities.to_parquet(identity_dir / "match_identity.parquet", index=False)
    alignment.to_parquet(identity_dir / "quote_alignment.parquet", index=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    protocol_path = raw_dir / "protocol.json"
    protocol_path.write_text(json.dumps({
        "primary_model": {"column": "p_full", "sha256": digest(forecast_path)},
    }))
    snapshot_path = raw_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(quotes.to_dict("records")))
    (raw_dir / "receipt.json").write_text(json.dumps({
        "files": {name: {"sha256": digest(raw_dir / f"{name}.parquet")} for name in raw_frames},
        "snapshot_file": snapshot_path.name,
        "snapshot_sha256": digest(snapshot_path),
        "protocol_sha256": digest(protocol_path),
    }))
    source_paths = [forecast_path, *(raw_dir / f"{name}.parquet" for name in raw_frames)]
    (identity_dir / "identity_receipt.json").write_text(json.dumps({
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "output_sha256": {
            name: digest(identity_dir / name)
            for name in ("match_identity.parquet", "quote_alignment.parquet")
        },
    }))

    output_dir = tmp_path / "report"
    run_production_market_benchmark(raw_dir, identity_dir, forecast_path, output_dir)
    metrics = pd.read_csv(output_dir / "metrics.csv")
    primary = metrics.loc[metrics.cohort.eq("all_horizon_quote_cohort")]
    market = primary.loc[primary.model.eq("p_market")].set_index("horizon")
    assert market.loc["OPEN", "log_loss"] == pytest.approx(-np.log(0.6))
    assert market.loc["24h", "log_loss"] == pytest.approx(-np.log(0.75))
    assert market.loc["CLOSE", "log_loss"] == pytest.approx(-np.log(0.9))
    assert primary.loc[primary.model.eq("p_a0"), "log_loss"].to_numpy() == pytest.approx(
        np.repeat(-np.log(0.7), 4)
    )
