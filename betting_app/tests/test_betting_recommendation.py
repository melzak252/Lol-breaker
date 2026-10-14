"""Consumer contracts for match recommendations; no database required."""

import pytest
from datetime import UTC, datetime

from betting_app.api.routers.matches import _build_match_recommendation
from betting_app.api.schemas import BookmakerOddsRow
from betting_app.services.market_service import kelly_fraction


def recommendation(**changes):
    params = dict(
        team_a_name="Alpha",
        team_b_name="Beta",
        has_unmapped_teams=False,
        odds_rows=[
            BookmakerOddsRow(
                bookmaker="sts", canonical_odds_a=1.6, canonical_odds_b=2.4,
                odds_snapshot_id=1, quote_scraped_at=datetime.now(UTC).isoformat(), quote_status="actionable",
            )
        ],
        hybrid_prob_a=0.45,
        hybrid_prob_b=0.55,
        pure_prob_a=0.4,
        pure_prob_b=0.6,
        hybrid_diagnostics={},
    )
    params.update(changes)
    return _build_match_recommendation(**params)


def test_unmapped_or_missing_odds_cannot_recommend():
    assert not recommendation(has_unmapped_teams=True).has_value
    assert not recommendation(odds_rows=[]).has_value


def test_qualified_side_uses_mean_probability_for_ev_and_stake():
    rec = recommendation()
    assert rec.has_value and rec.side == "b"
    assert rec.hybrid_prob == 0.55
    assert rec.ev == pytest.approx(0.55 * 2.4 * 0.88 - 1, abs=1e-4)
    assert rec.quarter_kelly == pytest.approx(
        kelly_fraction(0.55, 2.4, 0.12) / 4, abs=1e-4
    )


def test_missing_hybrid_prediction_cannot_be_replaced_by_pure_prediction():
    rec = recommendation(
        hybrid_diagnostics=None,
        diagnostics={},
    )
    assert not rec.has_value


def test_tier1_mean_market_gap_still_blocks_recommendation():
    rec = recommendation(
        league="LCK 2026",
        hybrid_prob_a=0.2,
        hybrid_prob_b=0.8,
    )
    assert not rec.has_value


def test_side_with_larger_mean_ev_is_recommended():
    rec = recommendation(
        hybrid_prob_a=0.55,
        hybrid_prob_b=0.45,
        odds_rows=[
            BookmakerOddsRow(
                bookmaker="sts", canonical_odds_a=2.6, canonical_odds_b=3.0,
                odds_snapshot_id=2, quote_scraped_at=datetime.now(UTC).isoformat(), quote_status="actionable",
            )
        ],
        hybrid_diagnostics={},
    )
    assert rec.has_value and rec.side == "a"
    assert rec.ev == pytest.approx(0.55 * 2.6 * 0.88 - 1)

