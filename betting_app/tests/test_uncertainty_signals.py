"""Native mean-probability EV consumers over isolated PostgreSQL rows and quotes."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from betting_app.core.db import get_session
from betting_app.core.models import C0_NATIVE
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import OddsSnapshot
from betting_app.models.prediction import CanonicalPrediction, ModelEvSignal
from betting_app.services import upcoming_inference_service as service


@pytest.fixture
def signal_clock(client, monkeypatch):
    clock = {"now": datetime(2026, 10, 2, 12, tzinfo=UTC)}

    class SignalDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = clock["now"]
            return current.astimezone(tz) if tz is not None else current.replace(tzinfo=None)

    monkeypatch.setattr(service, "datetime", SignalDatetime)
    return clock


def _seed_signal_inputs(clock):
    now = clock["now"]
    with get_session() as session:
        match = CanonicalMatch(
            canonical_key="native-mean-signals",
            team_a_name="Alpha", team_b_name="Beta",
            normalized_team_a="alpha", normalized_team_b="beta",
            start_time_normalized=(now + timedelta(hours=1)).isoformat(),
            best_of=3, league="Unmapped invitational", status="upcoming",
        )
        session.add(match)
        session.flush()
        predictions = [
            CanonicalPrediction(
                canonical_match_id=match.id,
                model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
                features_version=C0_NATIVE.feature_version,
                ratings_version=C0_NATIVE.ratings_version,
                predicted_at=(now - timedelta(minutes=age)).isoformat(),
                data_cutoff_at=(now - timedelta(days=1)).isoformat(),
                prob_a=prob_a, prob_b=1.0 - prob_a,
                prediction_status="active", diagnostics_json="{}",
            )
            for age, prob_a in ((3, 0.3), (1, 0.52))
        ]
        quotes = [
            OddsSnapshot(
                canonical_match_id=match.id, bookmaker_id=1,
                market_type="match_winner", raw_team_a="Alpha", raw_team_b="Beta",
                odds_a=odds, odds_b=odds, is_live=0,
                scraped_at=now + offset,
            )
            for offset, odds in (
                (timedelta(minutes=-2), 2.4),
                (timedelta(seconds=-30), 2.6),
                (timedelta(seconds=30), 6.0),
            )
        ]
        session.add_all([*predictions, *quotes])
        session.flush()
        ids = {"match": match.id, "prediction": predictions[-1].id, "quote": quotes[1].id}
        session.commit()
    return ids


def test_c0_ev_uses_each_sides_mean_probability(signal_clock):
    ids = _seed_signal_inputs(signal_clock)

    signals = service.generate_model_ev_signals(bankroll=100)

    assert {signal["side"] for signal in signals} == {"a", "b"}
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, ids["prediction"])
        quote = session.get(OddsSnapshot, ids["quote"])
        saved = session.scalars(select(ModelEvSignal).where(ModelEvSignal.status == "new")).all()
        assert len(saved) == 2
        for signal in signals:
            side = signal["side"]
            mean = getattr(prediction, f"prob_{side}")
            odds = getattr(quote, f"odds_{side}")
            assert signal["canonical_match_id"] == ids["match"]
            assert signal["odds_snapshot_id"] == quote.id
            assert signal["quote_scraped_at"] == quote.scraped_at.isoformat()
            assert signal["evaluated_at"] == signal_clock["now"].isoformat()
            assert signal["model_prob"] == mean
            assert signal["prob_for_ev"] == mean
            assert signal["prob_conservative"] is None
            assert signal["uncertainty_basis"] == "mean"
            assert signal["ev"] == pytest.approx(mean * odds * 0.88 - 1)
            persisted = next(row for row in saved if row.side == side)
            assert persisted.canonical_prediction_id == prediction.id
            assert persisted.odds_snapshot_id == quote.id
            assert persisted.model_prob == pytest.approx(mean)
            assert persisted.ev == pytest.approx(signal["ev"])
            context = json.loads(persisted.market_context_json)
            assert context["decision_at"] == signal_clock["now"].isoformat()
            assert [source["odds_snapshot_id"] for source in context["sources"]] == [quote.id]
        assert json.loads(prediction.diagnostics_json) == {}


def test_swapping_c0_probability_sides_swaps_ev_sides(signal_clock):
    ids = _seed_signal_inputs(signal_clock)
    forward = {signal["side"]: signal for signal in service.generate_model_ev_signals()}
    assert set(forward) == {"a", "b"}
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, ids["prediction"])
        prediction.prob_a, prediction.prob_b = prediction.prob_b, prediction.prob_a
        session.commit()

    reversed_signals = {
        signal["side"]: signal for signal in service.generate_model_ev_signals()
    }

    assert set(reversed_signals) == {"a", "b"}
    for side, opposite in (("a", "b"), ("b", "a")):
        assert reversed_signals[side]["model_prob"] == forward[opposite]["model_prob"]
        assert reversed_signals[side]["ev"] == pytest.approx(forward[opposite]["ev"])
        assert reversed_signals[side]["prob_conservative"] is None
    with get_session() as session:
        saved = session.scalars(select(ModelEvSignal)).all()
        assert sum(row.status == "stale" for row in saved) == 2
        assert sum(row.status == "new" for row in saved) == 2


@pytest.mark.parametrize("clock_error", ["unknown_cutoff", "future_prediction", "cutoff_after_prediction"])
def test_c0_signals_refuse_predictions_without_actionable_clocks(signal_clock, clock_error):
    ids = _seed_signal_inputs(signal_clock)
    with get_session() as session:
        prediction = session.get(CanonicalPrediction, ids["prediction"])
        if clock_error == "unknown_cutoff":
            prediction.data_cutoff_at = None
        elif clock_error == "future_prediction":
            prediction.predicted_at = (signal_clock["now"] + timedelta(seconds=1)).isoformat()
        else:
            prediction.data_cutoff_at = signal_clock["now"].isoformat()
        session.commit()

    assert service.generate_model_ev_signals() == []
    with get_session() as session:
        assert session.scalars(select(ModelEvSignal)).all() == []
