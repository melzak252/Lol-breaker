"""Behavioral contracts for event-time financial simulation, not execution.

F1 response and UI contract:
response and ledger ``uncertainty_basis`` distinguish mean, measured_bounds,
assumed_research_haircut and missing_bounds; ``uncertainty_reason`` names the
evidence or missing prerequisite. The API has no assumed-haircut policy: that
identity belongs to explicitly authorized research evaluation, never a native
C0 fallback. ``result_status`` distinguishes computed from no_eligible_bets.
``evaluation_kind=historical_quote_simulation`` does not certify recorded
issuance/execution, even for timestamp-valid live-scope rows. Absent historical
diagnostics have ``historical_diagnostics_status=unknown``. Valid no-action
balances, turnover and PnL remain numeric; conditional ROI/hit rate stay null.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from betting_app.api.routers import financial
from betting_app.core.models import BAYESIAN_SHRUNK_HYBRID, C0_NATIVE


def _prediction(
    match_id: int,
    *,
    start: str,
    result_available_at: str,
    cutoff: str | None = "2026-01-01T10:00:00+00:00",
    diagnostics_json: str | None = None,
) -> dict[str, Any]:
    return {
        "canonical_match_id": match_id,
        "team_a_name": f"A{match_id}",
        "team_b_name": f"B{match_id}",
        "league": "Test League",
        "start_time_normalized": start,
        "result_recorded_at": result_available_at,
        "winner_side": "team_a" if match_id == 1 else "team_b",
        "prob_a": 0.70,
        "prob_b": 0.30,
        "predicted_at": "2026-01-01T11:00:00+00:00",
        "data_cutoff_at": cutoff,
        "prediction_id": match_id,
        "diagnostics_json": diagnostics_json,
    }


def _snapshot(match_id: int, scraped_at: str) -> dict[str, Any]:
    return {
        "canonical_match_id": match_id,
        "bookmaker_id": 1,
        "bookmaker": "TestBook",
        "raw_team_a": f"A{match_id}",
        "raw_team_b": f"B{match_id}",
        "odds_a": 2.0,
        "odds_b": 2.0,
        "scraped_at": scraped_at,
    }


def _stub_query(
    predictions: list[dict[str, Any]], odds: list[dict[str, Any]]
) -> Callable[..., list[dict[str, Any]]]:
    def query(_db: object, statement: str, _params: dict[str, Any]):
        return predictions if "canonical_predictions p" in statement else odds

    return query


def test_financial_api_reserves_overlapping_stakes(monkeypatch, client: TestClient):
    predictions = [
        _prediction(1, start="2026-01-01T18:00:00+00:00", result_available_at="2026-01-01T22:00:00+00:00"),
        _prediction(2, start="2026-01-01T19:00:00+00:00", result_available_at="2026-01-01T23:00:00+00:00"),
    ]
    odds = [
        _snapshot(1, "2026-01-01T12:00:00+00:00"),
        _snapshot(2, "2026-01-01T13:00:00+00:00"),
    ]
    monkeypatch.setattr(financial, "query_df", _stub_query(predictions, odds))

    response = client.get(
        "/financial/analysis",
        params={
            "model_name": BAYESIAN_SHRUNK_HYBRID.name,
            "model_version": BAYESIAN_SHRUNK_HYBRID.version,
            "staking_mode": "fixed",
            "fixed_stake": 700,
            "initial_bankroll": 1000,
            "days_back": 730,
            "min_ev": 0,
            "use_conformal_gating": False,
            "data_scope": "live",
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert [entry["stake"] for entry in result["ledger"]] == [700, 300]
    assert result["max_open_bets"] == 2
    assert result["max_open_stake"] == 1000
    # Match two loses; no settlement before its placement may fund another 700 PLN stake.
    assert result["final_bankroll"] == 1232


def test_financial_live_scope_excludes_missing_cutoff(monkeypatch):
    predictions = [
        _prediction(
            1,
            start="2026-01-01T18:00:00+00:00",
            result_available_at="2026-01-01T22:00:00+00:00",
            cutoff=None,
        )
    ]
    monkeypatch.setattr(financial, "query_df", _stub_query(predictions, []))

    result = financial.financial_analysis(
        model_name=C0_NATIVE.name,
        model_version=C0_NATIVE.version,
        data_scope="live",
        days_back=730,
        db=object(),
    )

    assert result.total_bets == 0
    assert result.temporal_exclusions["missing_data_cutoff_at"] == 1


def test_c0_default_conformal_gate_excludes_missing_bounds(monkeypatch):
    predictions = [
        _prediction(
            1,
            start="2026-01-01T18:00:00+00:00",
            result_available_at="2026-01-01T22:00:00+00:00",
        )
    ]
    monkeypatch.setattr(
        financial,
        "query_df",
        _stub_query(predictions, [_snapshot(1, "2026-01-01T12:00:00+00:00")]),
    )

    result = financial.financial_analysis(
        model_name=C0_NATIVE.name,
        model_version=C0_NATIVE.version,
        data_scope="live",
        days_back=730,
        db=object(),
    )

    assert result.total_bets == 0
    assert result.temporal_exclusions["conformal_bounds_unavailable"] == 1


def test_c0_conformal_gate_uses_only_persisted_venn_abers_bounds(monkeypatch):
    predictions = [
        _prediction(
            1,
            start="2026-01-01T18:00:00+00:00",
            result_available_at="2026-01-01T22:00:00+00:00",
            diagnostics_json=(
                '{"conformal":{"method":"venn_abers","p_lower_a":0.60,'
                '"p_upper_a":0.64}}'
            ),
        )
    ]
    monkeypatch.setattr(
        financial,
        "query_df",
        _stub_query(predictions, [_snapshot(1, "2026-01-01T12:00:00+00:00")]),
    )

    result = financial.financial_analysis(
        model_name=C0_NATIVE.name,
        model_version=C0_NATIVE.version,
        data_scope="live",
        days_back=730,
        db=object(),
    )

    assert result.total_bets == 1
    assert result.ledger[0].target_prob == 0.60

def test_stored_c0_hybrid_is_not_blended_with_market_again(monkeypatch):
    predictions = [
        _prediction(
            1,
            start="2026-01-01T18:00:00+00:00",
            result_available_at="2026-01-01T22:00:00+00:00",
            diagnostics_json=None,
        )
    ]
    monkeypatch.setattr(
        financial,
        "query_df",
        _stub_query(predictions, [_snapshot(1, "2026-01-01T12:00:00+00:00")]),
    )

    result = financial.financial_analysis(
        model_name=BAYESIAN_SHRUNK_HYBRID.name,
        model_version=BAYESIAN_SHRUNK_HYBRID.version,
        data_scope="live",
        days_back=730,
        use_conformal_gating=False,
        db=object(),
    )

    assert result.total_bets == 1
    assert result.ledger[0].model_prob == 0.70
    assert result.ledger[0].target_prob == 0.70

def test_financial_api_empty_sqlite_cohort_is_valid(client: TestClient):
    response = client.get("/financial/analysis")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["data_scope"] == "historical"
    assert payload["odds_mode"] == "mid"
    assert payload["total_bets"] == 0


@pytest.fixture
def financial_api_client():
    """Exercise real HTTP serialization without application lifespan or any DB."""
    app = FastAPI()
    app.include_router(financial.router)
    app.dependency_overrides[financial.get_db] = lambda: object()
    with TestClient(app) as api_client:
        yield api_client


@pytest.fixture
def saved_financial_forecast(tmp_path, monkeypatch):
    class FixtureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 2, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(financial, "datetime", FixtureDateTime)
    prediction = _prediction(
        1,
        start="2026-01-01T18:00:00+00:00",
        result_available_at="2026-01-01T22:00:00+00:00",
    )
    prediction.update({"prob_a": 0.65, "prob_b": 0.35})
    path = tmp_path / "saved_c0_forecast_and_quote.json"
    path.write_text(json.dumps({
        "prediction": prediction,
        "odds": _snapshot(1, "2026-01-01T12:00:00+00:00"),
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "gated,bounds,basis,reason,bets,target_prob",
    [
        pytest.param(
            False, None, "mean", "uncertainty_not_requested", 1, 0.65,
            id="mean",
        ),
        pytest.param(
            True, (0.62, 0.66), "measured_bounds", "persisted_venn_abers",
            1, 0.62, id="bounded",
        ),
        pytest.param(
            True, None, "missing_bounds", "conformal_bounds_unavailable",
            0, None, id="missing",
        ),
    ],
)
def test_financial_api_saved_forecast_reports_uncertainty_and_support(
    monkeypatch, financial_api_client, saved_financial_forecast,
    gated, bounds, basis, reason, bets, target_prob,
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    prediction = dict(fixture["prediction"])
    if bounds is not None:
        prediction["diagnostics_json"] = json.dumps({
            "conformal": {
                "method": "venn_abers",
                "p_lower_a": bounds[0],
                "p_upper_a": bounds[1],
            },
        })
    monkeypatch.setattr(
        financial, "query_df", _stub_query([prediction], [fixture["odds"]]),
    )

    response = financial_api_client.get("/financial/analysis", params={
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "days_back": 30,
        "data_scope": "live",
        "use_conformal_gating": gated,
        "staking_mode": "kelly",
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_bets"] == bets
    assert payload["uncertainty_basis"] == basis
    assert payload["uncertainty_reason"] == reason
    assert payload["result_status"] == ("computed" if bets else "no_eligible_bets")
    assert payload["evaluation_kind"] == "historical_quote_simulation"
    assert payload["historical_diagnostics_status"] == (
        "available" if bounds is not None else "unknown"
    )
    assert payload["temporal_exclusions"]["conformal_bounds_unavailable"] == (not bets)
    # Missing evidence is not measured negative EV; exclusions must not double-count.
    assert payload["temporal_exclusions"]["no_positive_ev"] == 0
    if bets:
        entry = payload["ledger"][0]
        assert entry["model_prob"] == 0.65
        assert entry["target_prob"] == target_prob
        assert entry["uncertainty_basis"] == basis
        assert entry["uncertainty_reason"] == reason
        assert entry["entry_odds"] == 2.0
        assert entry["ev"] == pytest.approx(target_prob * 2.0 * 0.88 - 1)
        full_kelly = (target_prob * 1.76 - 1) / 0.76
        expected_stake = min(1000 * 0.05, 1000 * full_kelly * 0.25)
        assert entry["stake"] == pytest.approx(expected_stake)
        assert entry["profit"] == pytest.approx(expected_stake * 0.76)
        assert payload["roi"] == pytest.approx(0.76)
    else:
        assert payload["ledger"] == []
        assert payload["roi"] is None
        assert payload["hit_rate"] is None
        assert payload["max_drawdown_pct"] is None
        assert payload["total_staked"] == 0
        assert payload["total_profit"] == 0
        assert payload["final_bankroll"] == payload["initial_bankroll"] == 1000
        for bucket in payload["horizon_buckets"]:
            assert bucket["bets"] == 0
            assert bucket["roi"] is None
            assert bucket["avg_ev"] is None
    assert json.loads(saved_financial_forecast.read_text(encoding="utf-8")) == fixture


def test_financial_missing_conformal_bounds_are_not_negative_ev(
    monkeypatch, saved_financial_forecast
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        financial, "query_df",
        _stub_query([fixture["prediction"]], [fixture["odds"]]),
    )

    result = financial.financial_analysis(
        model_name=C0_NATIVE.name,
        model_version=C0_NATIVE.version,
        data_scope="live",
        days_back=30,
        db=object(),
    )

    assert result.total_bets == 0
    assert result.temporal_exclusions["conformal_bounds_unavailable"] == 1
    assert result.temporal_exclusions["no_positive_ev"] == 0
    assert result.roi is None
    assert result.hit_rate is None


def test_financial_historical_missing_diagnostics_remain_unknown(
    monkeypatch, financial_api_client, saved_financial_forecast
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    # An old quote predates forecast issuance: usable for research, not execution.
    odds = dict(fixture["odds"], scraped_at="2026-01-01T09:00:00+00:00")
    monkeypatch.setattr(
        financial, "query_df", _stub_query([fixture["prediction"]], [odds]),
    )
    response = financial_api_client.get("/financial/analysis", params={
        "model_name": BAYESIAN_SHRUNK_HYBRID.name,
        "model_version": BAYESIAN_SHRUNK_HYBRID.version,
        "days_back": 30,
        "data_scope": "historical",
        "use_conformal_gating": False,
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_bets"] == 1
    assert payload["historical_diagnostics_status"] == "unknown"
    assert payload["evaluation_kind"] == "historical_quote_simulation"
    assert payload["uncertainty_basis"] == "mean"
    # Selection may recompute EV but may neither reblend the stored hybrid nor
    # retroactively certify absent uncertainty/safety/issuance diagnostics.
    entry = payload["ledger"][0]
    assert entry["model_prob"] == 0.65
    assert entry["target_prob"] == 0.65
    assert entry["entry_scraped_at"] == odds["scraped_at"]
    assert entry["ev"] == pytest.approx(0.144)


def test_financial_api_no_forecast_support_has_undefined_conditional_results(
    monkeypatch, financial_api_client
):
    monkeypatch.setattr(financial, "query_df", _stub_query([], []))

    response = financial_api_client.get("/financial/analysis")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_bets"] == payload["total_matches"] == 0
    assert payload["total_staked"] == payload["total_profit"] == 0
    assert payload["initial_bankroll"] == payload["final_bankroll"] == 1000
    assert payload["roi"] is None
    assert payload["hit_rate"] is None
    assert payload["max_drawdown_pct"] is None
    assert payload["historical_diagnostics_status"] == "unknown"
    assert payload["result_status"] == "no_eligible_bets"
    assert payload["evaluation_kind"] == "historical_quote_simulation"


@pytest.mark.parametrize("diagnostics", [None, "", "{}", "[]", "not-json"])
def test_financial_missing_or_unreadable_historical_diagnostics_are_unknown(
    monkeypatch, financial_api_client, saved_financial_forecast, diagnostics,
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    prediction = dict(fixture["prediction"], diagnostics_json=diagnostics)
    monkeypatch.setattr(
        financial, "query_df", _stub_query([prediction], [fixture["odds"]]),
    )

    response = financial_api_client.get("/financial/analysis", params={
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "days_back": 30,
        "use_conformal_gating": False,
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["historical_diagnostics_status"] == "unknown"
    assert payload["evaluation_kind"] == "historical_quote_simulation"
    assert payload["uncertainty_basis"] == "mean"
    assert payload["ledger"][0]["target_prob"] == 0.65
    assert payload["ledger"][0]["uncertainty_basis"] == "mean"


@pytest.mark.parametrize(
    "odds,lower,upper,exclusion",
    [
        pytest.param(2.0, 0.50, 0.70, "conformal_bounds_too_wide", id="wide"),
        pytest.param(1.5, 0.62, 0.66, "no_positive_ev", id="bounded-negative-ev"),
    ],
)
def test_financial_measured_rejections_are_distinct_and_not_calibration_success(
    monkeypatch, financial_api_client, saved_financial_forecast,
    odds, lower, upper, exclusion,
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    prediction = dict(fixture["prediction"], diagnostics_json=json.dumps({
        "conformal": {
            "method": "venn_abers",
            "p_lower_a": lower,
            "p_upper_a": upper,
        },
    }))
    snapshot = dict(fixture["odds"], odds_a=odds, odds_b=odds)
    monkeypatch.setattr(
        financial, "query_df", _stub_query([prediction], [snapshot]),
    )

    response = financial_api_client.get("/financial/analysis", params={
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "days_back": 30,
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["uncertainty_basis"] == "measured_bounds"
    assert payload["uncertainty_reason"] == "persisted_venn_abers"
    assert payload["historical_diagnostics_status"] == "available"
    assert payload["result_status"] == "no_eligible_bets"
    assert payload["temporal_exclusions"][exclusion] == 1
    other = (
        "no_positive_ev"
        if exclusion == "conformal_bounds_too_wide"
        else "conformal_bounds_too_wide"
    )
    assert payload["temporal_exclusions"][other] == 0
    assert payload["temporal_exclusions"]["conformal_bounds_unavailable"] == 0
    assert payload["total_staked"] == payload["total_profit"] == 0
    assert payload["final_bankroll"] == payload["initial_bankroll"] == 1000
    assert payload["roi"] is payload["hit_rate"] is payload["max_drawdown_pct"] is None


def test_financial_hybrid_does_not_reuse_native_conformal_bounds(
    monkeypatch, financial_api_client, saved_financial_forecast,
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    prediction = dict(fixture["prediction"], diagnostics_json=json.dumps({
        "conformal": {
            "method": "venn_abers",
            "p_lower_a": 0.62,
            "p_upper_a": 0.66,
        },
    }))
    monkeypatch.setattr(
        financial, "query_df", _stub_query([prediction], [fixture["odds"]]),
    )

    response = financial_api_client.get("/financial/analysis", params={
        "model_name": BAYESIAN_SHRUNK_HYBRID.name,
        "model_version": BAYESIAN_SHRUNK_HYBRID.version,
        "days_back": 30,
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["uncertainty_basis"] == "missing_bounds"
    assert payload["uncertainty_reason"] == "conformal_bounds_unavailable"
    assert payload["historical_diagnostics_status"] == "available"
    assert payload["result_status"] == "no_eligible_bets"
    assert payload["ledger"] == []
    assert payload["temporal_exclusions"]["conformal_bounds_unavailable"] == 1
    assert payload["temporal_exclusions"]["no_positive_ev"] == 0


def test_financial_exhausted_reserved_capital_is_a_cash_exclusion(
    monkeypatch, financial_api_client, saved_financial_forecast,
):
    fixture = json.loads(saved_financial_forecast.read_text(encoding="utf-8"))
    predictions = [
        fixture["prediction"],
        _prediction(
            2,
            start="2026-01-01T19:00:00+00:00",
            result_available_at="2026-01-01T23:00:00+00:00",
        ),
        _prediction(
            3,
            start="2026-01-01T20:00:00+00:00",
            result_available_at="2026-01-01T23:30:00+00:00",
        ),
    ]
    odds = [
        _snapshot(1, "2026-01-01T12:00:00+00:00"),
        _snapshot(2, "2026-01-01T13:00:00+00:00"),
        _snapshot(3, "2026-01-01T14:00:00+00:00"),
    ]
    monkeypatch.setattr(financial, "query_df", _stub_query(predictions, odds))

    response = financial_api_client.get("/financial/analysis", params={
        "model_name": C0_NATIVE.name,
        "model_version": C0_NATIVE.version,
        "days_back": 30,
        "data_scope": "live",
        "use_conformal_gating": False,
        "staking_mode": "fixed",
        "fixed_stake": 700,
    })

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [entry["stake"] for entry in payload["ledger"]] == [700, 300]
    assert payload["temporal_exclusions"]["insufficient_available_balance"] == 1
    assert payload["temporal_exclusions"]["no_positive_ev"] == 0
    assert payload["max_open_stake"] == 1000
    assert payload["total_staked"] == 1000
    assert payload["final_bankroll"] == 1232
    assert payload["result_status"] == "computed"
