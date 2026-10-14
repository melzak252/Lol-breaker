"""Plan11 contracts against an EXISTING, explicitly owned PostgreSQL schema.

Provision each schema path separately before running this module. No fixture calls
create_all, migrations, truncation, collectors or notification transports. Set
PLAN11_SCHEMA_DATABASE_URL, DATABASE_URL and TEST_DATABASE_URL to the same owned
URL because the shared conftest assigns DATABASE_URL at import. Every write rolls
back, including commits made by the actual services under test.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import os
from uuid import uuid4

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Numeric, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from betting_app.api.deps import get_db
from betting_app.api.routers import alerts, props
from betting_app.models.alerts import ValueAlertLog
from betting_app.models.prediction import ModelPropPrediction
from betting_app.services.prop_odds_service import get_saved_prop_prediction


# Only application-critical columns: extra tables/indexes (including Timescale
# internals) are deliberately not treated as discrepancies.
REQUIRED_COLUMNS = {
    "alert_configs": {
        "id", "is_enabled", "min_ev", "min_odds", "max_odds", "cooldown_hours",
        "discord_enabled", "discord_webhook_url", "telegram_enabled",
        "telegram_bot_token", "telegram_chat_id", "updated_at",
    },
    "value_alert_logs": {
        "id", "created_at", "canonical_match_id", "match_label", "league",
        "match_start_at", "side", "team_name", "bookmaker_name", "odds",
        "model_prob", "market_prob", "ev", "suggested_stake", "channels",
        "status", "message_payload", "error_message",
    },
    "prop_odds_snapshots": {
        "id", "bookmaker_id", "canonical_match_id", "market_type", "map_number",
        "line", "target_team", "raw_team_a", "raw_team_b", "odds_over",
        "odds_under", "prob_over_novig", "prob_under_novig", "margin",
        "raw_market_name", "scraped_at", "is_live", "source_url", "outcomes_payload",
    },
    "model_prop_predictions": {
        "id", "canonical_match_id", "map_number", "model_name", "model_version",
        "predicted_at", "expected_total_kills", "mu_team_a", "mu_team_b",
        "expected_spread", "league_avg_kills", "pace_category", "distribution_json", "signals_json",
    },
    "odds_snapshots": {
        "id", "bookmaker_id", "canonical_match_id", "market_type", "odds_a", "odds_b",
        "is_live", "scraped_at", "source_url", "offer_url", "raw_payload",
    },
    "canonical_predictions": {
        "id", "canonical_match_id", "model_name", "model_version", "predicted_at",
        "prob_a", "prob_b", "prediction_status", "diagnostics_json",
    },
    "model_ev_signals": {
        "id", "canonical_match_id", "canonical_prediction_id", "odds_snapshot_id",
        "bookmaker_id", "side", "odds", "model_prob", "market_prob", "ev", "tax_rate",
        "stake_suggestion", "status", "evaluated_at", "market_context_json",
    },
    "bookmaker_accounts": {
        "id", "bookmaker_id", "account_name", "currency", "opening_balance",
        "current_balance", "is_active", "created_at", "updated_at",
    },
    "bets": {
        "id", "bookmaker_account_id", "canonical_match_id", "placed_at", "side", "stake",
        "taken_odds", "status", "profit", "settled_at", "match_start_time", "tax_rate", "source",
    },
    "bookmaker_wallet_transactions": {
        "id", "bookmaker_account_id", "bet_id", "transaction_time",
        "transaction_type", "amount", "balance_after",
    },
    "forecast_evidence_snapshots": {"id", "evidence_key", "content_sha256", "payload_json", "recorded_at"},
    "source_captures": {
        "id", "capture_id", "source_id", "sha256", "content_base64", "body_uri",
        "first_observed_at", "request_started_at", "response_received_at", "recorded_at",
    },
    "opportunity_events": {
        "id", "event_key", "protocol_id", "provider_id", "fixture_id", "origin_at",
        "primary_decision", "parent_event_key", "forecast_evidence_key", "recorded_at",
    },
    "canonical_teams": {"id", "canonical_name", "normalized_name", "squad_type", "created_at"},
    "canonical_competitions": {"id", "canonical_name", "normalized_name", "family"},
    "source_team_identities": {
        "id", "canonical_team_id", "source_system", "source_team_id", "source_name",
        "normalized_source_name", "competition_scope", "valid_from", "valid_to",
        "confidence", "review_status",
    },
    "source_competition_identities": {
        "id", "canonical_competition_id", "source_system", "source_competition_id",
        "source_name", "normalized_source_name",
    },
    "mapping_review_decisions": {
        "id", "canonical_match_id", "old_golgg_match_id", "new_golgg_match_id",
        "decision", "reason", "operator", "evidence_json", "decided_at",
    },
    "match_roster_overrides": {"id", "canonical_match_id", "team_side", "roster_json", "updated_at"},
    "team_current_roster_players": {
        "id", "team_id", "team_name", "normalized_team_name", "player_id", "player_name",
        "role", "source", "source_match_id", "source_game_id", "source_match_date", "updated_at",
    },
    "oddspapi_fixture_mappings": {
        "id", "fixture_id", "canonical_match_id", "sport_id", "league", "provider_team_1",
        "provider_team_2", "provider_team_1_is_a", "start_time", "has_odds", "last_synced_at",
    },
    "oddspapi_request_logs": {"id", "endpoint", "fixture_id", "status_code", "response_time_ms", "created_at"},
    "team_aliases": {
        "id", "normalized_name", "alias", "source", "source_system", "normalized_alias",
        "league_pattern", "tournament_pattern", "valid_from", "valid_to",
        "confidence", "is_active", "is_blocked", "review_status",
    },
}


@pytest.fixture
def schema_session():
    explicit = os.environ.get("PLAN11_SCHEMA_DATABASE_URL")
    url = explicit or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("requires an explicitly owned Plan11 PostgreSQL URL")
    parsed = make_url(url)
    # Without the explicit override, refuse conftest's configured default DB.
    if not explicit and not (parsed.database or "").startswith("plan11_"):
        pytest.skip("set PLAN11_SCHEMA_DATABASE_URL to the owned schema DB")
    if parsed.get_backend_name() != "postgresql" or parsed.host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail("Plan11 schema exercises require owned loopback PostgreSQL")
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            outer = connection.begin()
            try:
                with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
                    yield session
            finally:
                outer.rollback()
    finally:
        engine.dispose()


@pytest.fixture
def schema_client(schema_session, monkeypatch):
    external_calls = []

    def forbid_delivery(*args, **kwargs):
        external_calls.append((args, kwargs))
        raise AssertionError("external HTTP/notification delivery is forbidden")

    monkeypatch.setattr(requests.sessions.Session, "request", forbid_delivery)
    for name in ("DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(props, "_CACHED_TRACKER", None)
    # Tracker SQL must use the same existing owned schema, not a cached configured
    # application engine. This replaces only connection ownership, not results.
    import betting_app.core.db as core_db
    monkeypatch.setattr(core_db, "get_session", lambda: Session(
        bind=schema_session.get_bind(), join_transaction_mode="create_savepoint"
    ))
    app = FastAPI()
    app.include_router(alerts.router)
    app.include_router(props.router)

    def existing_schema_db():
        yield schema_session

    app.dependency_overrides[get_db] = existing_schema_db
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    assert external_calls == []


def _bookmaker(session):
    # Other suites seed explicit IDs without advancing the sequence. Allocate an
    # owned row ID from existing data instead of consuming that unrelated state.
    return session.execute(text("""
        INSERT INTO bookmakers(id, name, is_active)
        VALUES ((SELECT COALESCE(MAX(id), 0) + 1000 FROM bookmakers), :name, '1')
        RETURNING id
    """), {"name": f"plan11-{uuid4().hex}"}).scalar_one()


def _match(session):
    return session.execute(text("""
        INSERT INTO canonical_matches
        (id, canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b,
         league, status, match_confidence, best_of)
        VALUES ((SELECT COALESCE(MAX(id), 0) + 1000 FROM canonical_matches),
                :key, 'Plan11 Alpha', 'Plan11 Beta', 'plan11 alpha', 'plan11 beta',
                'LEC', 'upcoming', 1.0, 1) RETURNING id
    """), {"key": f"plan11-{uuid4().hex}"}).scalar_one()


def _account(session, bookmaker_id):
    return session.execute(text("""
        INSERT INTO bookmaker_accounts
        (id, bookmaker_id, account_name, currency, opening_balance, current_balance,
         is_active, created_at, updated_at)
        VALUES ((SELECT COALESCE(MAX(id), 0) + 1000 FROM bookmaker_accounts),
                :bookmaker, :name, 'PLN', 100.00, 100.00, '1', NOW(), NOW()) RETURNING id
    """), {"bookmaker": bookmaker_id, "name": f"plan11-{uuid4().hex}"}).scalar_one()


def test_bookmaker_active_flag_matches_integer_application_contract(schema_session):
    row_id = _bookmaker(schema_session)
    active = schema_session.execute(text(
        "SELECT is_active FROM bookmakers WHERE id = :id"
    ), {"id": row_id}).scalar_one()
    assert type(active) is int, "ORM/services use integer is_active = 1, not PostgreSQL boolean"
    assert active == 1


@pytest.mark.parametrize("confidence", [0.625, 0.90000001])
def test_match_identity_confidence_retains_fractional_threshold_value(schema_session, confidence):
    match_id = _match(schema_session)
    restored = schema_session.execute(text(
        "UPDATE canonical_matches SET match_confidence = :confidence "
        "WHERE id = :id RETURNING match_confidence"
    ), {"confidence": confidence, "id": match_id}).scalar_one()
    assert restored == confidence, "identity matching thresholds must not see a rounded confidence"


@pytest.mark.parametrize("table", REQUIRED_COLUMNS)
def test_required_operational_columns_exist_without_schema_repair(schema_session, table):
    inspector = inspect(schema_session.get_bind())
    assert inspector.has_table(table), f"missing application table: {table}"
    actual = {column["name"] for column in inspector.get_columns(table)}
    missing = REQUIRED_COLUMNS[table] - actual
    assert not missing, f"{table}: missing required columns {sorted(missing)}"


def test_disabled_alert_configuration_check_and_history_use_existing_schema(schema_session, schema_client):
    response = schema_client.post("/alerts/config", json={
        "is_enabled": False, "discord_enabled": False, "telegram_enabled": False,
        "min_ev": 0.05, "min_odds": 1.25, "max_odds": 12.0, "cooldown_hours": 6.0,
    })
    assert response.status_code == 200, response.text
    assert response.json()["is_enabled"] is False
    log = ValueAlertLog(
        created_at=datetime.now(timezone.utc).isoformat(), match_label="Plan11 local fixture",
        channels="simulation", status="simulated", message_payload='{"information_status":"local_fixture"}',
    )
    schema_session.add(log)
    schema_session.flush()
    response = schema_client.post("/alerts/check?dry_run=false")
    assert response.status_code == 200, response.text
    assert response.json()["dispatched"] == 0
    assert response.json()["failed"] == 0
    # Dry run still traverses the real signal/prediction/feature SQL, but the
    # channels stay disabled even if the database contains an eligible signal.
    response = schema_client.post("/alerts/check?dry_run=true")
    assert response.status_code == 200, response.text
    assert response.json()["failed"] == 0
    response = schema_client.get("/alerts/history?status=simulated")
    assert response.status_code == 200, response.text
    assert any(entry["id"] == log.id and entry["channels"] == "simulation"
               for entry in response.json()["alerts"])


def test_prop_analysis_persists_real_prediction_and_returns_missing_match_404(schema_session, schema_client):
    match_id = _match(schema_session)
    bookmaker_id = _bookmaker(schema_session)
    # Exercise the real ORM query and persistence against the provisioned schema.
    schema_session.execute(text("""
        INSERT INTO prop_odds_snapshots
        (bookmaker_id, canonical_match_id, market_type, map_number, line,
         odds_over, odds_under, scraped_at, is_live)
        VALUES (:bookmaker, :match, 'total_kills', 1, 26.5, 1.90, 1.90, NOW(), 0)
    """), {"bookmaker": bookmaker_id, "match": match_id})
    response = schema_client.get(f"/matches/{match_id}/props/analysis?persist=true")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["canonical_match_id"] == match_id
    assert len(payload["evaluated_lines"]) == 1
    line = payload["evaluated_lines"][0]
    assert line["model_prob_over"] + line["model_prob_under"] == pytest.approx(1.0, abs=0.0002)
    saved = schema_session.get(ModelPropPrediction, payload["saved_prediction_id"])
    assert saved is not None and saved.canonical_match_id == match_id
    assert saved.expected_total_kills > 0
    expectations = payload["model_expectations"]
    assert saved.model_name == "kill_spread_distribution"
    assert saved.model_version == "prop-v1.0"
    assert saved.map_number == 1
    assert saved.mu_team_a == pytest.approx(expectations["mu_team_a"], abs=0.005)
    assert saved.mu_team_b == pytest.approx(expectations["mu_team_b"], abs=0.005)
    persisted = get_saved_prop_prediction(match_id, db_session=schema_session)
    assert persisted is not None and persisted["id"] == saved.id
    assert persisted["distribution"] == expectations["distribution"]
    assert persisted["signals"] == payload["signals"]
    assert saved.predicted_at.tzinfo is not None
    missing_id = schema_session.execute(text("SELECT COALESCE(MAX(id), 0) + 1 FROM canonical_matches")).scalar_one()
    for method, endpoint in (("get", "analysis"), ("post", "predict")):
        response = getattr(schema_client, method)(f"/matches/{missing_id}/props/{endpoint}")
        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"Canonical match id={missing_id} not found"


def test_prop_tracker_consumes_existing_golgg_team1_win_history(schema_session, schema_client):
    fixture_id = f"plan11-{uuid4().hex}"
    schema_session.execute(text("INSERT INTO golgg_matches(match_id) VALUES (:id)"), {"id": fixture_id})
    schema_session.execute(text("""
        INSERT INTO golgg_games
        (game_id, match_id, date, tournament_name, team1_name, team2_name,
         team1_stats_json, team2_stats_json, game_duration, team1_win)
        VALUES (:id, :id, '2026-01-01', 'LEC', 'Plan11 Alpha', 'Plan11 Beta',
                :stats_a, :stats_b, 1800, 1)
    """), {"id": fixture_id, "stats_a": '{"kills":20}', "stats_b": '{"kills":10}'})
    tracker = props.get_cached_pace_tracker()
    form = tracker.get_team_recent_form("Plan11 Alpha")
    assert form.sample_size == 1, "existing historical row must not silently become empty prior history"
    assert form.recent_games[0].kills_for == 20
    assert form.recent_games[0].kills_against == 10
    assert form.recent_games[0].won is True
    other_form = tracker.get_team_recent_form("Plan11 Beta")
    assert other_form.sample_size == 1
    assert other_form.recent_games[0].kills_for == 10
    assert other_form.recent_games[0].kills_against == 20
    assert other_form.recent_games[0].won is False


def test_prop_tracker_sql_failure_propagates_without_caching_prior(schema_session, schema_client):
    match_id = _match(schema_session)
    # A transaction-local schema lookup failure exercises actual PostgreSQL SQL
    # without altering tables or retaining a broken search_path.
    with pytest.raises(DBAPIError):
        with schema_session.begin_nested():
            schema_session.execute(text("SET LOCAL search_path TO pg_catalog"))
            props.get_cached_pace_tracker()
    assert props._CACHED_TRACKER is None
    savepoint = schema_session.begin_nested()
    try:
        schema_session.execute(text("SET LOCAL search_path TO pg_catalog"))
        response = schema_client.get(f"/matches/{match_id}/props/analysis")
        assert response.status_code == 500, response.text
        assert props._CACHED_TRACKER is None
    finally:
        savepoint.rollback()
    tracker = props.get_cached_pace_tracker()
    assert tracker.get_team_recent_form(f"unseen-{uuid4().hex}").sample_size == 0


@pytest.mark.parametrize("method,endpoint", [("get", "analysis"), ("post", "predict")])
def test_prop_unrelated_value_error_is_server_failure(schema_session, schema_client, monkeypatch, method, endpoint):
    match_id = _match(schema_session)

    def invalid_model_result(*args, **kwargs):
        raise ValueError("invalid statistical prediction")

    monkeypatch.setattr(props, "evaluate_match_props", invalid_model_result)
    response = getattr(schema_client, method)(f"/matches/{match_id}/props/{endpoint}")
    assert response.status_code == 500, response.text


def test_wallet_sql_defaults_are_usable_without_orm_insert_defaults(schema_session):
    bookmaker_id = _bookmaker(schema_session)
    row = schema_session.execute(text("""
        INSERT INTO bookmaker_accounts(bookmaker_id, account_name)
        VALUES (:bookmaker, :name)
        RETURNING currency, opening_balance, current_balance, is_active, created_at, updated_at
    """), {"bookmaker": bookmaker_id, "name": f"plan11-{uuid4().hex}"}).mappings().one()
    assert row["currency"] == "PLN"
    assert row["opening_balance"] == row["current_balance"] == Decimal("0.00")
    assert row["is_active"] == 1
    assert row["created_at"] is not None and row["updated_at"] is not None


@pytest.mark.parametrize("table,column,precision,scale", [
    ("bookmaker_accounts", "current_balance", 12, 2),
    ("bets", "stake", 12, 2), ("bets", "taken_odds", 8, 4),
    ("bookmaker_wallet_transactions", "amount", 12, 2),
])
def test_money_columns_retain_numeric_precision(schema_session, table, column, precision, scale):
    definition = next(item for item in inspect(schema_session.get_bind()).get_columns(table) if item["name"] == column)
    assert isinstance(definition["type"], Numeric)
    assert (definition["type"].precision, definition["type"].scale) == (precision, scale)


def test_wallet_rejects_negative_balance_duplicate_account_and_orphan_reference(schema_session):
    bookmaker_id = _bookmaker(schema_session)
    account_id = _account(schema_session, bookmaker_id)
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(text("UPDATE bookmaker_accounts SET current_balance = -0.01 WHERE id = :id"), {"id": account_id})
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(text("""
            INSERT INTO bookmaker_accounts
            (bookmaker_id, account_name, currency, opening_balance, current_balance, is_active, created_at, updated_at)
            SELECT bookmaker_id, account_name, currency, opening_balance, current_balance, is_active, created_at, updated_at
            FROM bookmaker_accounts WHERE id = :id
        """), {"id": account_id})
    orphan = schema_session.execute(text("SELECT MAX(id) + 1 FROM bookmakers")).scalar_one()
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        _account(schema_session, orphan)


@pytest.mark.parametrize("column", ["created_at", "updated_at"])
def test_wallet_clock_round_trip_preserves_utc_instant(schema_session, column):
    account_id = _account(schema_session, _bookmaker(schema_session))
    instant = datetime.fromisoformat("2026-01-02T10:11:12+00:00")
    schema_session.execute(text("SET LOCAL TIME ZONE 'Europe/Warsaw'"))
    # column is drawn only from the fixed parameter list above.
    schema_session.execute(text(f"UPDATE bookmaker_accounts SET {column} = :instant WHERE id = :id"), {"instant": instant, "id": account_id})
    restored = schema_session.execute(text(f"SELECT {column} FROM bookmaker_accounts WHERE id = :id"), {"id": account_id}).scalar_one()
    assert restored.tzinfo is not None, f"bookmaker_accounts.{column} loses timezone information"
    assert restored.astimezone(timezone.utc) == instant


def test_bet_and_ledger_sql_defaults_preserve_money_and_aware_clocks(schema_session):
    account_id = _account(schema_session, _bookmaker(schema_session))
    bet = schema_session.execute(text("""
        INSERT INTO bets(bookmaker_account_id, side, stake, taken_odds)
        VALUES (:account, 'a', 1.23, 1.8765)
        RETURNING id, stake, taken_odds, status, profit, tax_rate, source, placed_at
    """), {"account": account_id}).mappings().one()
    assert bet["stake"] == Decimal("1.23")
    assert bet["taken_odds"] == Decimal("1.8765")
    assert (bet["status"], bet["profit"], bet["tax_rate"], bet["source"]) == (
        "open", Decimal("0.00"), Decimal("0.12"), "manual"
    )
    assert bet["placed_at"].tzinfo is not None
    ledger = schema_session.execute(text("""
        INSERT INTO bookmaker_wallet_transactions
        (bookmaker_account_id, bet_id, transaction_type, amount, balance_after)
        VALUES (:account, :bet, 'stake', -1.23, 98.77)
        RETURNING amount, balance_after, transaction_time
    """), {"account": account_id, "bet": bet["id"]}).mappings().one()
    assert (ledger["amount"], ledger["balance_after"]) == (Decimal("-1.23"), Decimal("98.77"))
    assert ledger["transaction_time"].tzinfo is not None


def test_bet_side_constraint_rejects_unknown_selection(schema_session):
    # Supply every baseline NOT NULL value so only the side check is exercised.
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(text("""
            INSERT INTO bets(placed_at, side, stake, taken_odds, status, profit, tax_rate, source)
            VALUES (NOW(), 'x', 1.23, 1.8765, 'open', 0.00, 0.12, 'manual')
        """))


def test_quote_snapshot_uniqueness_is_enforced_by_existing_schema(schema_session):
    params = {"bookmaker": _bookmaker(schema_session), "match": _match(schema_session),
              "clock": datetime.now(timezone.utc)}
    statement = text("""
        INSERT INTO odds_snapshots
        (bookmaker_id, canonical_match_id, market_type, scraped_at, odds_a, odds_b)
        VALUES (:bookmaker, :match, 'match_winner', :clock, 1.8, 2.2)
    """)
    schema_session.execute(statement, params)
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(statement, params)


def test_quote_clock_round_trip_preserves_aware_instant_and_fractional_prices(schema_session):
    instant = datetime.fromisoformat("2026-01-02T10:11:12+00:00")
    row = schema_session.execute(text("""
        INSERT INTO odds_snapshots
        (bookmaker_id, canonical_match_id, market_type, scraped_at, odds_a, odds_b)
        VALUES (:bookmaker, :match, 'match_winner', :clock, 1.8765, 2.2345)
        RETURNING scraped_at, odds_a, odds_b, is_live
    """), {"bookmaker": _bookmaker(schema_session), "match": _match(schema_session),
           "clock": instant}).mappings().one()
    assert isinstance(row["scraped_at"], datetime), "quote receipt must be a PostgreSQL aware timestamp, not text"
    assert row["scraped_at"].tzinfo is not None
    assert row["scraped_at"].astimezone(timezone.utc) == instant
    assert row["odds_a"] == pytest.approx(1.8765)
    assert row["odds_b"] == pytest.approx(2.2345)
    assert row["is_live"] is None, "missing prematch evidence must not default to known prematch"


@pytest.mark.parametrize("table", ["forecast_evidence_snapshots", "source_captures", "opportunity_events"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_existing_evidence_tables_reject_mutation(schema_session, table, operation):
    key = f"plan11-{uuid4().hex}"
    sha = hashlib.sha256(b"{}").hexdigest()
    statements = {
        "forecast_evidence_snapshots": """
            INSERT INTO forecast_evidence_snapshots(evidence_key, content_sha256, payload_json)
            VALUES (:key, :sha, '{}') RETURNING id
        """,
        "source_captures": """
            INSERT INTO source_captures
            (capture_id, source_id, evidence_uri, body_representation, source_class,
             availability_basis, attempt_status, failure_reason, request_metadata_json,
             response_metadata_json, source_authentication_status)
            VALUES (:key, 'plan11-local', 'fixture://plan11', 'unavailable', 'local_fixture',
                    'local_fixture', 'failed', 'local test, no acquisition', '{}', '{}', 'local_fixture') RETURNING id
        """,
        "opportunity_events": """
            INSERT INTO opportunity_events
            (event_key, protocol_id, provider_id, fixture_id, event_type, payload_json, payload_sha256)
            VALUES (:key, 'plan11-local', 'fixture', :key, 'discovery', '{}', :sha) RETURNING id
        """,
    }
    row_id = schema_session.execute(text(statements[table]), {"key": key, "sha": sha}).scalar_one()
    mutation = (f"UPDATE {table} SET id = id WHERE id = :id" if operation == "UPDATE"
                else f"DELETE FROM {table} WHERE id = :id")
    with pytest.raises(DBAPIError, match="append-only"), schema_session.begin_nested():
        schema_session.execute(text(mutation), {"id": row_id})
    assert schema_session.execute(text(f"SELECT COUNT(*) FROM {table} WHERE id = :id"), {"id": row_id}).scalar_one() == 1


def test_evidence_uniqueness_foreign_key_and_primary_origin_constraints(schema_session):
    key = f"plan11-{uuid4().hex}"
    sha = hashlib.sha256(b"{}").hexdigest()
    evidence_insert = text("""
        INSERT INTO forecast_evidence_snapshots(evidence_key, content_sha256, payload_json)
        VALUES (:key, :sha, '{}')
    """)
    schema_session.execute(evidence_insert, {"key": key, "sha": sha})
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(evidence_insert, {"key": key, "sha": sha})
    event_insert = text("""
        INSERT INTO opportunity_events
        (event_key, protocol_id, provider_id, fixture_id, origin_at, event_type,
         primary_decision, forecast_evidence_key, payload_json, payload_sha256)
        VALUES (:event, 'plan11-local', 'fixture', :fixture, :origin, 'primary_decision',
                TRUE, :evidence, '{}', :sha)
    """)
    params = {"event": key, "fixture": key, "origin": datetime.now(timezone.utc), "evidence": key, "sha": sha}
    schema_session.execute(event_insert, params)
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(event_insert, dict(params, event=f"{key}-duplicate"))
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(event_insert, dict(params, event=f"{key}-orphan", fixture=f"{key}-other", evidence="absent-plan11-evidence"))
    with pytest.raises(IntegrityError), schema_session.begin_nested():
        schema_session.execute(event_insert, dict(params, event=f"{key}-no-origin", fixture=f"{key}-no-origin", origin=None))
