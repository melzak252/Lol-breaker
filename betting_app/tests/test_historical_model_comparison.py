"""Behavioral coverage for exact native-C0 historical model selection."""

from __future__ import annotations

import math
import pytest

from betting_app.api.routers import timing
from betting_app.core.models import C0_NATIVE, get_active_hybrid
from betting_app.services.rating_contract import OPERATIONAL_BACKFILL_MODEL_VERSION


def test_historical_comparison_uses_exact_c0_and_hybrid_cohorts(monkeypatch) -> None:
    exp039_rows = [
        {"canonical_match_id": 1, "prob_a": 0.70, "winner_side": "team_a"},
        {"canonical_match_id": 2, "prob_a": 0.40, "winner_side": "team_b"},
    ]
    archived_rows = [
        {"canonical_match_id": 1, "prob_a": 0.99, "winner_side": "team_b"},
    ]
    current_rows = [
        {"canonical_match_id": 1, "prob_a": 0.80, "winner_side": "team_a"},
        {"canonical_match_id": 2, "prob_a": 0.30, "winner_side": "team_b"},
        {"canonical_match_id": 3, "prob_a": 0.55, "winner_side": "team_a"},
    ]

    expected = {
        (C0_NATIVE.name, C0_NATIVE.version, C0_NATIVE.feature_version): current_rows,
        (
            get_active_hybrid().hybrid_model_name,
            get_active_hybrid().hybrid_model_version,
            C0_NATIVE.feature_version,
        ): current_rows,
        (
            timing.OPERATIONAL_MODEL_NAME,
            OPERATIONAL_BACKFILL_MODEL_VERSION,
            timing.OPERATIONAL_BACKFILL_FEATURE_VERSION,
        ): current_rows,
        (
            timing.THESIS_MODEL_NAME,
            "exp-039",
            timing.ANALYSIS_FEATURES_VERSION,
        ): exp039_rows,
        (C0_NATIVE.name, "c0-native-archived", C0_NATIVE.feature_version): archived_rows,
        (
            get_active_hybrid().hybrid_model_name,
            "Hybrid-Operational-Market-archived",
            C0_NATIVE.feature_version,
        ): archived_rows,
    }

    def fake_query_df(_db, sql: str, params: dict):
        if "model_version" not in params:
            return []
        assert "cp.model_name = :model_name" in sql
        assert "cp.model_version = :model_version" in sql
        assert "cp.features_version = :features_version" in sql
        assert "cp.predicted_at::timestamptz" in sql
        identity = (
            params["model_name"],
            params["model_version"],
            params["features_version"],
        )
        return expected.get(identity, [])

    monkeypatch.setattr(timing, "query_df", fake_query_df)
    result = timing.historical_model_comparison(db=object())

    assert result["target_model_key"] == "c0"
    assert result["common_cohort"]["target_model_key"] == "c0"
    assert result["common_cohort"]["n_matches"] == 2
    assert result["common_cohort"]["target_minus_exp039_logloss"] < 0
    assert result["common_cohort"]["operational_minus_exp039_logloss"] < 0
    assert [model["key"] for model in result["models"]] == [
        "c0", "operational_hybrid", "operational_regional", "exp039"
    ]
    assert result["models"][0]["temporal_eligible_matches"] == 3
    assert result["models"][0]["avg_logloss"] == pytest.approx(
        timing._metric_summary([1, 0, 1], [0.80, 0.30, 0.55])["avg_logloss"]
    )
    assert result["models"][1]["model_name"] == get_active_hybrid().hybrid_model_name
    assert result["models"][1]["model_version"] == get_active_hybrid().hybrid_model_version
    assert result["models"][3]["temporal_eligible_matches"] == 2

    result_op = timing.historical_model_comparison(
        db=object(), target_model="operational_regional"
    )
    assert result_op["target_model_key"] == "operational_regional"
    assert result_op["common_cohort"]["operational_minus_exp039_logloss"] < 0


def test_missing_c0_coverage_is_not_filled_from_legacy_predictions(monkeypatch) -> None:
    def fake_query_df(_db, _sql: str, params: dict):
        if "model_version" not in params:
            return []
        if params.get("model_version") == "exp-039":
            return [{"canonical_match_id": 1, "prob_a": 0.70, "winner_side": "team_a"}]
        if params.get("model_version") == get_active_hybrid().hybrid_model_version:
            return [{"canonical_match_id": 1, "prob_a": 0.80, "winner_side": "team_a"}]
        return []

    monkeypatch.setattr(timing, "query_df", fake_query_df)
    result = timing.historical_model_comparison(db=object())

    assert result["target_model_key"] == "c0"
    assert result["common_cohort"]["n_matches"] == 0
    assert result["common_cohort"]["target_model"] is None


def test_model_profitability_audit_selects_c0_identity(monkeypatch) -> None:
    def fake_query_df(_db, _sql: str, params: dict):
        if "mname" in params:
            assert params.get("mname") == C0_NATIVE.name
            assert params.get("mver") == C0_NATIVE.version
            return [{
                "canonical_match_id": 1,
                "prob_a": 0.65,
                "prob_b": 0.35,
                "diagnostics_json": None,
            }]
        if "cutoff" in params:
            return [{
                "id": 1,
                "team_a_name": "T1",
                "team_b_name": "Gen.G",
                "start_time_normalized": "2026-06-01T10:00:00+00:00",
                "winner_side": "team_a",
                "best_of": 3,
                "league": "LCK",
            }]
        if "mid_0" in params:
            return [{
                "canonical_match_id": 1,
                "odds_a": 2.20,
                "odds_b": 1.65,
                "scraped_at": "2026-06-01T08:00:00+00:00",
                "bookmaker": "sts",
            }]
        return []

    monkeypatch.setattr(timing, "query_df", fake_query_df)
    result = timing.model_profitability_audit(model_key="c0", db=object())

    assert result["model_key"] == "c0"
    assert result["model_title"] == "Native C0"
    assert result["overall"]["total_bets"] >= 0
    assert "avg_clv_pct" in result["overall"]



def test_model_clv_query_binds_exact_active_c0_hybrid_identity(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_query_df(_db, sql: str, params: dict):
        captured["sql"] = sql
        captured["params"] = params
        return []

    monkeypatch.setattr(timing, "query_df", fake_query_df)
    monkeypatch.setattr(timing, "_read_model_analysis_cache", lambda *_a, **_k: None)
    result = timing.model_clv_by_horizon(refresh=True, db=object())

    active = get_active_hybrid()
    assert result["total_entries"] == 0
    assert ":op_hybrid_model" in captured["sql"]
    assert captured["params"]["op_hybrid_model"] == active.hybrid_model_name
    assert captured["params"]["op_hybrid_version"] == active.hybrid_model_version

def test_dynamic_active_hybrid_bins_use_registered_logit_shrinkage(monkeypatch) -> None:
    match_start = "2026-06-01T10:00:00+00:00"
    predictions = [
        {"canonical_match_id": 1, "thesis_prob_a": 0.8, "winner_side": "team_a"},
        {"canonical_match_id": 2, "thesis_prob_a": 0.2, "winner_side": "team_b"},
    ]
    snapshots = [
        {
            "canonical_match_id": match_id,
            "scraped_at": "2026-06-01T09:00:00+00:00",
            "bookmaker_id": 1,
            "odds_a": odds_a,
            "odds_b": odds_b,
            "raw_team_a": "T1",
            "raw_team_b": "GEN",
        }
        for match_id, odds_a, odds_b in ((1, 2.0, 3.0), (2, 3.0, 2.0))
    ]
    metadata = [
        {
            "id": match_id,
            "start_time_normalized": match_start,
            "winner_side": winner_side,
            "normalized_team_a": "T1",
            "normalized_team_b": "GEN",
        }
        for match_id, winner_side in ((1, "team_a"), (2, "team_b"))
    ]

    def fake_query_df(_db, sql: str, params: dict):
        if "FROM canonical_predictions cp" in sql:
            assert params["base_model"] == C0_NATIVE.name
            assert params["base_version"] == C0_NATIVE.version
            assert params["features_version"] == C0_NATIVE.feature_version
            return predictions
        if "FROM odds_snapshots os" in sql:
            return snapshots
        if "FROM canonical_matches" in sql:
            return metadata
        raise AssertionError("Unexpected query in dynamic hybrid calculation")

    monkeypatch.setattr(timing, "query_df", fake_query_df)
    bins = timing._compute_hybrid_bins_dynamic(
        db=object(),
        cutoff="2026-05-01T00:00:00+00:00",
        model_name=get_active_hybrid().hybrid_model_name,
        model_version=get_active_hybrid().hybrid_model_version,
        bin_defs=[("0-2h", 0, 2)],
        min_matches=2,
        base_model=C0_NATIVE.name,
        base_version=C0_NATIVE.version,
        features_version=C0_NATIVE.feature_version,
    )

    # Market probabilities are 0.6 and 0.4. Active C0 uses alpha=.5,
    # T=1 and logit shrinkage, yielding sigmoid((logit(.8)+logit(.6))/2).
    expected_probability = 1.0 / (1.0 + math.exp(-(
        math.log(0.8 / 0.2) + math.log(0.6 / 0.4)
    ) / 2.0))
    assert len(bins) == 1
    assert bins[0]["match_count"] == 2
    assert bins[0]["avg_logloss"] == pytest.approx(-math.log(expected_probability), abs=5e-5)
