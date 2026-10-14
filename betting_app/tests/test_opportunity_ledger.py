"""Append-only no-forecast discovery and origin-decision ledger contracts."""
from datetime import datetime, timezone
import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from betting_app.models.prediction import OpportunityEvent
from betting_app.services.forecast_evidence import append_opportunity_event, load_opportunity_events


UTC = timezone.utc
ORIGIN = datetime(2026, 10, 2, 4, 0, tzinfo=UTC)


@pytest.fixture
def session():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url or "_test" not in url:
        pytest.skip("requires an explicitly isolated TEST_DATABASE_URL")
    engine = create_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection, join_transaction_mode="create_savepoint") as current:
            yield current
        transaction.rollback()
    engine.dispose()


def _payload(status, reasons=(), **extra):
    return {
        "status": status,
        "rejection_reasons": list(reasons),
        "clocks": {"discovered_at": "2026-10-02T03:00:00+00:00", "scheduled_start_at": None},
        "model_availability": {"available": False, "reason": "not_attempted"},
        "inference_eligible": False,
        "forward_admitted": False,
        "feature_availability": {"available": False, "reason": "not_available"},
        "information_status": "local_fixture",
        "source_authentication_status": "local_fixture",
        "source_capture_ids": [],
        **extra,
    }


def _append(session, *, event_key=None, fixture_id="provider-fixture-1", origin_at=None,
            event_type="discovery", primary_decision=False, parent_event_key=None,
            payload=None, protocol_id="forward-test", provider_id="fixture-provider",
            forecast_evidence_key=None):
    return append_opportunity_event(
        session,
        event_key=event_key or str(uuid4()),
        protocol_id=protocol_id,
        provider_id=provider_id,
        fixture_id=fixture_id,
        origin_at=origin_at,
        event_type=event_type,
        primary_decision=primary_decision,
        parent_event_key=parent_event_key,
        payload=payload or _payload("discovered"),
        forecast_evidence_key=forecast_evidence_key,
    )


def test_discovery_preserves_unknown_fixture_and_every_rejection_without_forecast(session):
    unknown_id, known_id = f"unknown-{uuid4()}", f"known-{uuid4()}"
    discovered = _append(
        session, fixture_id=unknown_id, payload=_payload("pending", ["unknown_provider_id"]),
    )
    due = _append(
        session, fixture_id=known_id, origin_at=ORIGIN, event_type="primary_decision",
        primary_decision=True,
        payload=_payload("rejected", ["withdrawn_roster", "participant_binding_absent", "latest_quote_invalid"]),
    )

    assert discovered.origin_at is None and not discovered.primary_decision
    assert due.forecast_evidence_key is None
    assert load_opportunity_events(
        session, protocol_id="forward-test", provider_id="fixture-provider", fixture_id=unknown_id,
    )[0].payload["rejection_reasons"] == ["unknown_provider_id"]
    assert load_opportunity_events(
        session, protocol_id="forward-test", provider_id="fixture-provider", fixture_id=known_id,
    )[0].payload["rejection_reasons"] == [
        "withdrawn_roster", "participant_binding_absent", "latest_quote_invalid",
    ]


def test_event_idempotency_conflicts_corrections_and_primary_uniqueness(session):
    fixture_id = f"fixture-{uuid4()}"
    discovery_key = str(uuid4())
    first = _append(
        session, event_key=discovery_key, fixture_id=fixture_id,
        payload=_payload("discovered", schedule_revision=1),
    )
    same = _append(
        session, event_key=discovery_key, fixture_id=fixture_id,
        payload=_payload("discovered", schedule_revision=1),
    )
    assert first.id == same.id
    with pytest.raises(ValueError, match="event_key"):
        _append(
            session, event_key=discovery_key, fixture_id=fixture_id,
            payload=_payload("discovered", schedule_revision=2),
        )

    revised = _append(
        session, fixture_id=fixture_id, event_type="schedule_revision", parent_event_key=discovery_key,
        payload=_payload("schedule_revised", schedule_revision=2,
                         scheduled_start_at="2026-10-02T05:00:00+00:00"),
    )
    decision_key = str(uuid4())
    decision = _append(
        session, event_key=decision_key, fixture_id=fixture_id, origin_at=ORIGIN,
        event_type="primary_decision", primary_decision=True,
        payload=_payload("pending", ["not_yet_due"]),
    )
    correction = _append(
        session, fixture_id=fixture_id, origin_at=ORIGIN, event_type="correction",
        parent_event_key=decision_key, payload=_payload("correction", ["corrected_schedule_metadata"]),
    )
    assert revised.parent_event_key == discovery_key
    assert correction.parent_event_key == decision_key
    assert decision.primary_decision and correction.primary_decision is False
    assert [event.event_type for event in load_opportunity_events(
        session, protocol_id="forward-test", provider_id="fixture-provider", fixture_id=fixture_id,
    )] == ["discovery", "schedule_revision", "primary_decision", "correction"]

    duplicate_primary_key = str(uuid4())
    with pytest.raises(IntegrityError), session.begin_nested():
        _append(
            session, event_key=duplicate_primary_key, fixture_id=fixture_id, origin_at=ORIGIN,
            event_type="primary_decision", primary_decision=True,
            payload=_payload("rejected", ["duplicate_primary_attempt"]),
        )

    with pytest.raises(ValueError, match="forward_admitted"):
        _append(
            session,
            fixture_id=fixture_id,
            payload=_payload("forecasted", forward_admitted=True),
        )


def test_event_insert_joins_caller_rollback_and_database_denies_update_delete(session):
    fixture_id = f"fixture-{uuid4()}"
    rolled_back_key = str(uuid4())
    with pytest.raises(RuntimeError), session.begin_nested():
        _append(session, event_key=rolled_back_key, fixture_id=fixture_id)
        raise RuntimeError("cancel opportunity transaction")
    assert session.execute(
        select(OpportunityEvent).where(OpportunityEvent.event_key == rolled_back_key)
    ).scalar_one_or_none() is None

    row = _append(session, fixture_id=fixture_id, payload=_payload("pending", ["late_discovery"]))
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(
            text("UPDATE opportunity_events SET payload_json = '{}' WHERE event_key = :key"),
            {"key": row.event_key},
        )
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(
            text("DELETE FROM opportunity_events WHERE event_key = :key"),
            {"key": row.event_key},
        )
    assert load_opportunity_events(
        session, protocol_id="forward-test", provider_id="fixture-provider", fixture_id=fixture_id,
    )[0].payload["rejection_reasons"] == ["late_discovery"]


@pytest.mark.parametrize("event_type", ["schedule_revision", "correction"])
def test_revision_events_require_an_existing_parent(session, event_type):
    with pytest.raises(ValueError, match="parent"):
        _append(
            session, fixture_id=f"unlinked-{uuid4()}", event_type=event_type,
            payload=_payload("revised"),
        )
