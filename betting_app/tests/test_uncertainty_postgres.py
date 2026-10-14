"""Isolated PostgreSQL regressions for active C0 prediction consumers."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from betting_app.core.db import get_session
from betting_app.core.models import get_active_hybrid
from betting_app.services import upcoming_inference_service as service


def seed_prediction(*, hybrid=False):
    now = datetime.now(UTC)
    diagnostics = {}
    name = service.DEFAULT_HYBRID_MODEL_NAME if hybrid else service.DEFAULT_MODEL_NAME
    version = (
        get_active_hybrid().hybrid_model_version if hybrid else service.DEFAULT_MODEL_VERSION
    )
    with get_session() as session:
        session.execute(
            text("""INSERT INTO canonical_matches
            (id, canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b,
             start_time_normalized, league, status, match_confidence)
            VALUES (1, 'uncertainty', 'Alpha', 'Beta', 'alpha', 'beta', :start, 'Unmapped invitational', 'upcoming', 1)
        """),
            {"start": (now + timedelta(days=1)).isoformat()},
        )
        session.execute(
            text("""INSERT INTO odds_snapshots
            (id, canonical_match_id, bookmaker_id, market_type, is_live, raw_team_a, raw_team_b, odds_a, odds_b, scraped_at)
            VALUES (1, 1, 2, 'match_winner', 0, 'Alpha', 'Beta', 3.5, 3.5, :now)
        """),
            {"now": now.isoformat()},
        )
        session.execute(
            text("""INSERT INTO upcoming_matches
            (bookmaker_id, bookmaker_match_key, canonical_match_id, raw_team_a, raw_team_b,
             normalized_team_a, normalized_team_b, last_seen_at)
            VALUES (2, 'uncertainty', 1, 'Alpha', 'Beta', 'alpha', 'beta', :now)
        """),
            {"now": now.isoformat()},
        )
        session.execute(
            text("""INSERT INTO canonical_predictions
            (canonical_match_id, model_name, model_version, predicted_at, data_cutoff_at,
             prob_a, prob_b, diagnostics_json, prediction_status)
            VALUES (1, :name, :version, :predicted, :cutoff, .6, .4, :diag, 'active')
        """),
            {
                "name": name,
                "version": version,
                "predicted": (now - timedelta(minutes=1)).isoformat(),
                "cutoff": (now - timedelta(minutes=2)).isoformat(),
                "diag": json.dumps(diagnostics),
            },
        )
        session.commit()
    return name, version


def test_wallet_lists_only_active_hybrid_version_and_preserves_minimum_ev(client):
    seed_prediction(hybrid=True)
    old_version = "exp081-siamese-series-v1-a0.50-t1.00"
    with get_session() as session:
        legacy_prediction_id = session.execute(
            text("""INSERT INTO canonical_predictions
            (canonical_match_id, model_name, model_version, predicted_at, data_cutoff_at,
             prob_a, prob_b, diagnostics_json, prediction_status)
            VALUES (1, :name, :version, :predicted, :cutoff, .7, .3, '{}', 'active')
            RETURNING id"""),
            {
                "name": service.DEFAULT_HYBRID_MODEL_NAME,
                "version": old_version,
                "predicted": datetime.now(UTC).isoformat(),
                "cutoff": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            },
        ).scalar_one()
        current_prediction_id = session.execute(
            text("""SELECT id FROM canonical_predictions
            WHERE canonical_match_id=1 AND model_name=:name AND model_version=:version"""),
            {
                "name": service.DEFAULT_HYBRID_MODEL_NAME,
                "version": get_active_hybrid().hybrid_model_version,
            },
        ).scalar_one()
        stale_prediction_id = session.execute(
            text("""INSERT INTO canonical_predictions
            (canonical_match_id, model_name, model_version, predicted_at, data_cutoff_at,
             prob_a, prob_b, diagnostics_json, prediction_status)
            VALUES (1, :name, :version, :predicted, :cutoff, .8, .2, '{}', 'stale')
            RETURNING id"""),
            {
                "name": service.DEFAULT_HYBRID_MODEL_NAME,
                "version": get_active_hybrid().hybrid_model_version,
                "predicted": (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
                "cutoff": (datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
            },
        ).scalar_one()
        for prediction_id, ev in (
            (legacy_prediction_id, 0.50),
            (current_prediction_id, 0.20),
            (stale_prediction_id, 0.80),
        ):
            session.execute(
                text("""INSERT INTO model_ev_signals
                (canonical_match_id, canonical_prediction_id, odds_snapshot_id, bookmaker_id,
                 side, odds, model_prob, market_prob, ev, tax_rate, stake_suggestion, status)
                VALUES (1, :prediction_id, 1, 2, 'a', 3.5, .6, .3, :ev, .12, 1, 'new')"""),
                {"prediction_id": prediction_id, "ev": ev},
            )
        session.commit()

    from betting_app.services.wallet_service import latest_model_ev_signals

    signals = latest_model_ev_signals(min_ev=0.15)
    assert signals.model_version.tolist() == [get_active_hybrid().hybrid_model_version]
    assert signals.ev.tolist() == pytest.approx([0.20])


