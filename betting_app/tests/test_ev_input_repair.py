from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.ev_input_repair import attach_repaired_predictions, selected_reliability


def test_repaired_predictions_preserve_identity_order_and_unavailable_rows():
    opportunities = pd.DataFrame([
        {"row_id": "second", "cohort": "oddspapi_common", "fixture_id": "f2", "horizon": "T-1h", "p": .999,
         "available": True, "mapping_available": True, "history_available": True, "quote_available": True,
         "exclusion_reason": None, "odds_a": 1.8, "odds_b": 2.2, "provider_team1_is_a": False},
        {"row_id": "first", "cohort": "oddspapi_common", "fixture_id": "f1", "horizon": "T-1h", "p": .6,
         "available": True, "mapping_available": True, "history_available": True, "quote_available": True,
         "exclusion_reason": None, "odds_a": 3., "odds_b": 1.4, "provider_team1_is_a": False},
    ])
    predictions = pd.DataFrame([
        {"cohort": "oddspapi_common", "fixture_id": "f1", "horizon": "T-1h", "p_golgg": .3, "status": "available", "missing_reason": None},
        {"cohort": "oddspapi_common", "fixture_id": "f2", "horizon": "T-1h", "p_golgg": None, "status": "unavailable", "missing_reason": "unresolved_roster"},
    ])
    result = attach_repaired_predictions(opportunities, predictions)
    assert result.row_id.tolist() == ["second", "first"]
    assert result.available.tolist() == [False, True]
    assert pd.isna(result.loc[0, "p"])
    assert result.loc[1, "p"] == .3  # Already canonical A/B; no second reversal.
    assert result.odds_a.tolist() == [1.8, 3.]
    assert opportunities.p.tolist() == [.999, .6]
    with pytest.raises(ValueError):
        attach_repaired_predictions(opportunities, pd.concat([predictions, predictions.iloc[[0]]]))


def test_empty_selected_band_cannot_establish_ev_reliability():
    universe = pd.DataFrame({"row_id": ["jan", "feb"], "date": ["2026-01-01", "2026-02-01"]})
    selected = pd.DataFrame(columns=["row_id", "ev", "p", "odds", "won"])
    result = selected_reliability(universe, selected, tax_rate=.12)
    assert result["count"] == 0
    assert result["mean_predicted_ev"] is None
    assert result["mean_realized_return"] is None
    assert result["reliable"] is False
    assert result["ev_overstatement_interval"]["ci_lower"] is None


def test_ev_band_bootstrap_retains_months_without_selected_bets():
    universe = pd.DataFrame({
        "row_id": ["jan", "feb", "mar"],
        "date": [None, "2026-02-01", "2026-03-01"],
        "provider_day": ["2026-01-01", "2026-02-01", "2026-03-01"],
    })
    selected = pd.DataFrame({"row_id": ["feb"], "ev": [.075], "p": [.5], "odds": [2.4431818181818183], "won": [False]})
    result = selected_reliability(universe, selected, tax_rate=.12)
    assert result["count"] == 1
    assert result["mean_predicted_ev"] == pytest.approx(.075)
    assert result["mean_realized_return"] == -1.
    assert result["universe_months"] == 3
    assert result["reliable"] is False
    assert result["ev_overstatement_interval"]["ci_lower"] is None


def test_missing_history_is_cold_start_but_malformed_or_future_history_is_not():
    fixtures = ["empty-history", "malformed-history", "future-history"]
    opportunities = pd.DataFrame([
        {"row_id": fixture, "cohort": "oddspapi_common", "fixture_id": fixture,
         "horizon": "T-1h", "p": .6, "mapping_available": True,
         "history_available": False, "quote_available": True,
         "available": False, "exclusion_reason": "old_history_gap"}
        for fixture in fixtures
    ])
    predictions = pd.DataFrame([
        {"cohort": "oddspapi_common", "fixture_id": fixture, "horizon": "T-1h",
         "p_golgg": .6, "status": "available", "missing_reason": None,
         "cutoff_day": "2026-02-01", "feature_source_max_day": source}
        for fixture, source in zip(fixtures, [None, "not-a-date", "2026-02-01"])
    ])
    result = attach_repaired_predictions(opportunities, predictions)
    assert result.available.tolist() == [True, False, False]
    assert pd.isna(result.loc[0, "exclusion_reason"])
    assert result.loc[1, "exclusion_reason"] == "invalid_or_nonprior_feature_clock"
    assert result.loc[2, "exclusion_reason"] == "invalid_or_nonprior_feature_clock"
