"""Unit tests for fixed-horizon quote selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from betting_app.ml.backtesting.horizons import HorizonSpec, select_horizon_quotes
from betting_app.ml.backtesting.types import HistoricalPrediction, MatchLabel, OddsQuote


def dt(hour: int) -> datetime:
    return datetime(2026, 1, 1, hour, 0, tzinfo=UTC)


def test_horizon_selection_uses_only_fresh_post_prediction_quotes() -> None:
    prediction = HistoricalPrediction(
        canonical_match_id=1,
        model_name="candidate",
        model_version="v1",
        prob_a=0.6,
        prob_b=0.4,
        data_cutoff_at=dt(1),
        predicted_at=dt(2),
    )
    label = MatchLabel(canonical_match_id=1, winner_side="a", start_time=dt(18))
    horizon = HorizonSpec("t6h", timedelta(hours=6), timedelta(hours=1))
    quotes = [
        OddsQuote(1, 1, "book-a", 2.0, 2.0, dt(11), 10),
        OddsQuote(1, 1, "book-a", 1.9, 2.1, dt(12), 11),
        OddsQuote(1, 2, "book-b", 2.0, 2.0, dt(11) + timedelta(minutes=30), 12),
        OddsQuote(1, 3, "book-c", 2.0, 2.0, dt(13), 13),
        OddsQuote(1, 4, "book-d", 2.0, 2.0, dt(2), 14),
    ]

    selected = select_horizon_quotes(prediction, label, quotes, horizon)

    assert [quote.odds_snapshot_id for quote in selected] == [11, 12]


def test_horizon_selection_rejects_predictions_after_target_horizon() -> None:
    prediction = HistoricalPrediction(
        canonical_match_id=1,
        model_name="candidate",
        model_version="v1",
        prob_a=0.6,
        prob_b=0.4,
        data_cutoff_at=dt(13),
        predicted_at=dt(13),
    )
    label = MatchLabel(canonical_match_id=1, winner_side="a", start_time=dt(18))
    horizon = HorizonSpec("t6h", timedelta(hours=6), timedelta(hours=1))

    assert select_horizon_quotes(
        prediction,
        label,
        [OddsQuote(1, 1, "book-a", 2.0, 2.0, dt(12), 10)],
        horizon,
    ) == []


def test_horizon_model_defs_include_operational_and_exclude_polluted_features() -> None:
    import inspect
    from betting_app.api.routers import timing

    src = inspect.getsource(timing)
    # Ensure the polluted features version string is never used as a query filter
    assert "cp.features_version = 'thesis-exp039'" not in src
    assert "cp.features_version = :features_version OR" not in src


def test_model_clv_supports_operational_models() -> None:
    from betting_app.api.routers import timing

    assert hasattr(timing, "OPERATIONAL_HYBRID_MODEL_NAME")
    assert timing.OPERATIONAL_HYBRID_MODEL_NAME == "Hybrid-Operational-Market"
    assert timing.OPERATIONAL_MODEL_NAME == "Operational-PlayerTeamRatings-W20"
