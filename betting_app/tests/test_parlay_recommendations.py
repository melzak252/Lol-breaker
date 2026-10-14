"""Unit and integration tests for Tax-Amortized Parlay Recommendations (IDEA-019)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from betting_app.api.main import app
from betting_app.api.schemas import ParlayLeg
from betting_app.models.base import Base
from betting_app.services.parlay_service import (
    calculate_parlay_metrics,
    find_parlay_recommendations,
    POLISH_TAX_RATE,
)


def test_calculate_parlay_metrics():
    """Verify exact compounding, 12% tax amortization bonus, and Kelly sizing."""
    leg1 = ParlayLeg(
        canonical_match_id=101,
        match_name="T1 vs Gen.G",
        league="LCK",
        side="a",
        team_name="T1",
        opponent_name="Gen.G",
        odds=1.70,
        model_prob=0.70,
        single_ev=0.70 * 1.70 * 0.88 - 1.0,  # +4.72%
    )
    leg2 = ParlayLeg(
        canonical_match_id=102,
        match_name="G2 vs Fnatic",
        league="LEC",
        side="a",
        team_name="G2",
        opponent_name="Fnatic",
        odds=1.80,
        model_prob=0.65,
        single_ev=0.65 * 1.80 * 0.88 - 1.0,  # +2.96%
    )

    metrics = calculate_parlay_metrics(leg1, leg2, tax_rate=0.12, bankroll=100.0)

    # Combined odds: 1.70 * 1.80 = 3.06
    assert metrics["combined_odds"] == pytest.approx(3.06, abs=1e-3)
    # Effective odds: 3.06 * 0.88 = 2.6928
    assert metrics["effective_odds"] == pytest.approx(2.693, abs=1e-2)
    # Joint prob: 0.70 * 0.65 = 0.455
    assert metrics["joint_prob"] == pytest.approx(0.455, abs=1e-4)

    # Net EV: 0.455 * 2.6928 - 1 = +22.52%
    expected_ev = 0.455 * 3.06 * 0.88 - 1.0
    assert metrics["ev"] == pytest.approx(expected_ev, abs=1e-3)
    assert metrics["ev"] > 0.20  # Over 20% net EV after tax!

    # Tax amortization gain: parlay EV vs avg single EV
    avg_single = (leg1.single_ev + leg2.single_ev) / 2.0
    expected_gain = expected_ev - avg_single
    assert metrics["tax_amortization_gain"] == pytest.approx(expected_gain, abs=1e-3)
    assert metrics["tax_amortization_gain"] > 0.15  # Massive gain over paying tax twice

    # Kelly sizing:
    assert metrics["quarter_kelly"] > 0.0
    assert 5.0 <= metrics["suggested_stake"] <= 50.0


@pytest.fixture
def valid_parlay_rows(monkeypatch):
    """Persist real quote contexts and hybrid/base lineage on a fixed UTC clock."""
    from betting_app.core.models.engine import blend_market_probability
    from betting_app.models.bookmaker import Bookmaker
    from betting_app.models.match import CanonicalMatch
    from betting_app.models.prediction import CanonicalPrediction, ModelArtifact
    from betting_app.services import parlay_service
    from betting_app.services.prematch_quote_service import (
        load_prematch_quotes,
        serialize_market_context,
    )

    now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    clock = {"now": now}

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"].astimezone(tz) if tz else clock["now"].replace(tzinfo=None)

    monkeypatch.setattr(parlay_service, "datetime", FixedDateTime)
    hybrid = parlay_service.ACTIVE_HYBRID
    base = hybrid.base_model
    params = {
        "base_model_name": base.name,
        "base_model_version": base.version,
        "alpha": hybrid.alpha,
        "temperature": hybrid.temperature,
        "blending_mode": hybrid.blending_mode,
    }
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    try:
        with sessionmaker(bind=engine)() as session:
            session.add_all([Bookmaker(id=1, name="superbet"), Bookmaker(id=2, name="sts")])
            session.add(ModelArtifact(
                id=1, model_name=hybrid.hybrid_model_name,
                model_version=hybrid.hybrid_model_version,
                model_params_json=json.dumps(params),
                feature_schema_json=json.dumps({"base_model": f"{base.name}/{base.version}"}),
            ))
            fixtures = [
                (1, "T1", "Gen.G", "LCK", 1, 1.65),
                (2, "G2", "Fnatic", "LEC", 1, 1.75),
                # Shares a participant with each of the two admissible legs.
                (3, "T1", "G2", "International", 1, 1.60),
                # Disjoint participants, but no second leg offered at this bookmaker.
                (4, "MAD Lions", "BDS", "LEC", 2, 1.70),
            ]
            for mid, team_a, team_b, league, book, odds_a in fixtures:
                session.add(CanonicalMatch(
                    id=mid, canonical_key=f"parlay_{mid}",
                    team_a_name=team_a, team_b_name=team_b,
                    normalized_team_a=team_a.lower(), normalized_team_b=team_b.lower(),
                    league=league, status="upcoming",
                    start_time_normalized=(now + timedelta(hours=2)).isoformat(),
                ))
            session.flush()
            for mid, team_a, team_b, _, book, odds_a in fixtures:
                # Raw SQL preserves the explicit offset that SQLite's DateTime
                # bind processor otherwise drops; production rejects naive clocks.
                session.execute(text("""
                    INSERT INTO odds_snapshots
                    (id, canonical_match_id, bookmaker_id, market_type, is_live,
                     raw_team_a, raw_team_b, odds_a, odds_b, scraped_at)
                    VALUES (:id, :id, :book, 'match_winner', 0,
                            :team_a, :team_b, :odds_a, 3.5, :clock)
                """), {
                    "id": mid, "book": book, "team_a": team_a, "team_b": team_b,
                    "odds_a": odds_a, "clock": (now - timedelta(seconds=30)).isoformat(),
                })
            origin = now - timedelta(seconds=10)
            contexts = load_prematch_quotes([1, 2, 3, 4], decision_at=origin, session=session)
            forecasts = {}
            for mid, *_ in fixtures:
                base_cutoff = (origin - timedelta(minutes=1)).isoformat()
                session.add(CanonicalPrediction(
                    id=mid, canonical_match_id=mid, model_name=base.name,
                    model_version=base.version, prob_a=0.95, prob_b=0.05,
                    data_cutoff_at=base_cutoff, predicted_at=origin.isoformat(),
                    prediction_status="active",
                ))
                context = contexts[mid]
                assert context.book_count == 1
                probability = blend_market_probability(
                    0.95, context.prob_a, alpha=hybrid.alpha,
                    temperature=hybrid.temperature, blending_mode=hybrid.blending_mode,
                )
                saved = serialize_market_context(
                    context, model_name=hybrid.hybrid_model_name,
                    model_version=hybrid.hybrid_model_version,
                    base_model_name=base.name, base_model_version=base.version,
                )
                forecast = CanonicalPrediction(
                    id=100 + mid, canonical_match_id=mid, model_artifact_id=1,
                    model_name=hybrid.hybrid_model_name,
                    model_version=hybrid.hybrid_model_version,
                    prob_a=probability, prob_b=1 - probability,
                    data_cutoff_at=origin.isoformat(), predicted_at=now.isoformat(),
                    prediction_status="active",
                    diagnostics_json=json.dumps({
                        **params, "base_prediction_id": mid,
                        "base_data_cutoff_at": base_cutoff, "market_context": saved,
                    }),
                )
                session.add(forecast)
                forecasts[mid] = forecast
            session.commit()
            yield session, clock, forecasts
    finally:
        engine.dispose()


def test_valid_bound_parlays_require_same_bookmaker_and_disjoint_participants(valid_parlay_rows):
    session, _, _ = valid_parlay_rows
    response = find_parlay_recommendations(session, bankroll=100.0)

    assert response.count == 1
    assert response.top_parlay == response.parlays[0]
    top = response.top_parlay
    assert top.bookmaker == "superbet"
    assert {leg.canonical_match_id for leg in top.legs} == {1, 2}
    participants = [
        {leg.team_name.lower(), leg.opponent_name.lower()} for leg in top.legs
    ]
    assert participants[0].isdisjoint(participants[1])
    assert top.combined_odds == pytest.approx(1.65 * 1.75, abs=1e-3)
    assert top.joint_prob == pytest.approx(
        top.legs[0].model_prob * top.legs[1].model_prob, abs=1e-4,
    )
    assert top.ev == pytest.approx(
        top.joint_prob * top.combined_odds * (1 - POLISH_TAX_RATE) - 1, abs=1e-4,
    )
    assert top.ev > 0
    assert "założeniu niezależności" in top.summary_pl
    assert "nie potwierdza" in response.explanation
    filtered = find_parlay_recommendations(session, bookmaker="STS")
    assert filtered.count == 0
    assert filtered.top_parlay is None


@pytest.mark.parametrize("invalid_contract", [
    "past_start", "expired_quotes", "changed_quote", "unbound_forecast",
    "missing_base_row", "wrong_base_match", "hybrid_parameter_mismatch",
    "hybrid_identity_mismatch", "future_prediction", "missing_cutoff",
])
def test_parlays_withhold_invalid_forecast_or_market_contract(valid_parlay_rows, invalid_contract):
    session, clock, forecasts = valid_parlay_rows
    forecast = forecasts[2]
    diagnostics = json.loads(forecast.diagnostics_json)
    if invalid_contract == "past_start":
        from betting_app.models.match import CanonicalMatch
        session.get(CanonicalMatch, 2).start_time_normalized = (
            clock["now"] - timedelta(seconds=1)
        ).isoformat()
    elif invalid_contract == "expired_quotes":
        clock["now"] += timedelta(seconds=301)
    elif invalid_contract == "changed_quote":
        session.execute(text("UPDATE odds_snapshots SET odds_a=1.8 WHERE id=2"))
    elif invalid_contract == "unbound_forecast":
        diagnostics.pop("market_context")
    elif invalid_contract == "missing_base_row":
        diagnostics["base_prediction_id"] = 9999
    elif invalid_contract == "wrong_base_match":
        diagnostics["base_prediction_id"] = 1
    elif invalid_contract == "hybrid_parameter_mismatch":
        diagnostics["alpha"] = 0.9
    elif invalid_contract == "hybrid_identity_mismatch":
        diagnostics["market_context"]["model_version"] = "unbound-version"
    elif invalid_contract == "future_prediction":
        forecast.predicted_at = (clock["now"] + timedelta(seconds=1)).isoformat()
    elif invalid_contract == "missing_cutoff":
        forecast.data_cutoff_at = None
    forecast.diagnostics_json = json.dumps(diagnostics)
    session.commit()

    response = find_parlay_recommendations(session)
    assert response.count == 0
    assert response.parlays == []
    assert response.top_parlay is None


def test_api_parlays_endpoint(client: TestClient):
    """Test GET /matches/recommendations/parlays HTTP contract."""
    response = client.get("/matches/recommendations/parlays")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "parlays" in data
    assert "tax_rate" in data
    assert data["tax_rate"] == 0.12
    assert "explanation" in data
