"""Unit and integration tests for Value Bet alerting service and API endpoints."""

from __future__ import annotations


import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from betting_app.models.base import Base
from betting_app.models.alerts import AlertConfig, ValueAlertLog
from betting_app.models.bookmaker import Bookmaker
from betting_app.services.alert_service import (
    build_discord_embed,
    build_telegram_html,
    get_or_create_alert_config,
    update_alert_config,
    is_alert_on_cooldown,
    send_test_alert,
    send_discord_notification,
    send_telegram_notification,
)


@pytest.fixture
def sqlite_session():
    """Create an isolated in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Seed bookmaker
    bm = Bookmaker(id=4, name="superbet", base_url="https://superbet.pl/")
    session.add(bm)
    session.commit()

    yield session
    session.close()


def test_build_discord_embed_format():
    signal = {
        "match_label": "G2 Esports vs Fnatic",
        "league": "LEC 2026",
        "team_name": "G2 Esports",
        "side": "a",
        "bookmaker_name": "Superbet",
        "odds": 1.95,
        "ev": 0.082,
        "model_prob": 0.62,
        "market_prob": 0.51,
        "suggested_stake": 3.4,
        "match_start_at": "2026-09-06 18:00 UTC",
    }
    embed = build_discord_embed(signal)

    assert "VALUE BET: G2 Esports vs Fnatic" in embed["title"]
    assert embed["color"] == 0x38BDF8  # Cyan for EV < 10%
    field_names = [f["name"] for f in embed["fields"]]
    assert "🎯 Typowana Drużyna" in field_names
    assert "💰 EV (po podatku 12%)" in field_names

    # High EV should be green
    signal_high = dict(signal, ev=0.14)
    embed_high = build_discord_embed(signal_high)
    assert embed_high["color"] == 0x10B981


def test_build_telegram_html_format():
    signal = {
        "match_label": "Gen.G vs T1",
        "league": "LCK 2026",
        "team_name": "Gen.G",
        "side": "a",
        "bookmaker_name": "STS",
        "odds": 2.10,
        "ev": 0.115,
        "model_prob": 0.58,
        "market_prob": 0.46,
        "suggested_stake": 4.5,
        "match_start_at": "2026-09-07 10:00 UTC",
    }
    html_text = build_telegram_html(signal)

    assert "<b>VALUE BET ALERT</b>" in html_text
    assert "Gen.G vs T1" in html_text
    assert "+11.5%" in html_text
    assert "<b>2.10</b>" in html_text
    assert "4.5% bankrolla" in html_text


def test_get_or_create_and_update_alert_config(sqlite_session):
    cfg = get_or_create_alert_config(sqlite_session)
    assert cfg.id == 1
    assert cfg.is_enabled is True
    assert cfg.min_ev == 0.05
    assert cfg.cooldown_hours == 6.0

    # Update configuration
    updated = update_alert_config(
        sqlite_session,
        is_enabled=False,
        min_ev=0.075,
        min_odds=1.40,
        discord_webhook_url="https://discord.com/api/webhooks/123/abc",
        telegram_chat_id="-100987654321",
    )
    assert updated.is_enabled is False
    assert updated.min_ev == 0.075
    assert updated.min_odds == 1.40
    assert updated.discord_webhook_url == "https://discord.com/api/webhooks/123/abc"
    assert updated.telegram_chat_id == "-100987654321"


def test_is_alert_on_cooldown_logic(sqlite_session):
    now_iso = datetime.now(timezone.utc).isoformat()
    log = ValueAlertLog(
        created_at=now_iso,
        canonical_match_id=101,
        match_label="T1 vs KT",
        side="a",
        team_name="T1",
        bookmaker_name="Superbet",
        odds=1.85,
        ev=0.06,
        status="sent",
    )
    sqlite_session.add(log)
    sqlite_session.commit()

    # Same match, bookmaker, side, similar odds -> should be on cooldown
    assert is_alert_on_cooldown(sqlite_session, 101, "Superbet", "a", 1.85, 0.06, 6.0) is True

    # Different bookmaker -> not on cooldown
    assert is_alert_on_cooldown(sqlite_session, 101, "STS", "a", 1.85, 0.06, 6.0) is False

    # Odds improved significantly (+0.06) -> cooldown bypass!
    assert is_alert_on_cooldown(sqlite_session, 101, "Superbet", "a", 1.91, 0.06, 6.0) is False

    # EV jumped by 3 percentage points (+0.03) -> cooldown bypass!
    assert is_alert_on_cooldown(sqlite_session, 101, "Superbet", "a", 1.85, 0.09, 6.0) is False


@pytest.mark.parametrize(
    "diagnostics, expected_ev",
    [
        ({"uncertainty_required": False}, 0.1088),
        ({
            "uncertainty_required": True,
            "p_low_a": 0.40,
            "p_low_b": 0.56,
            "epistemic_sigma_z": 0.15,
            "rating_disagreement": 0.0,
            "uncertainty_basis": "measured_bounds",
        }, 0.056),
        ({"uncertainty_required": True}, None),
    ],
    ids=["native_mean", "provided_bounds", "missing_required_bounds"],
)
def test_alerts_use_selected_ev_score_and_withhold_missing_bounds(monkeypatch, diagnostics, expected_ev):
    from betting_app.core.models import C0_NATIVE
    from betting_app.services import alert_service as alerts
    from betting_app.services.prematch_quote_service import (
        PrematchMarketContext, PrematchQuote, serialize_market_context,
    )

    now = datetime.now(timezone.utc)
    quote = PrematchQuote(1, 201, 4, "superbet", now - timedelta(seconds=10),
                          "Alpha", "Beta", 3.0, 2.0, None, None)
    context = PrematchMarketContext(
        201, "alpha", "beta", "LCK", "upcoming", now + timedelta(hours=8),
        now, (quote,), 0.4, (),
    )
    row = dict(
        signal_id=401, prediction_id=301, prediction_match_id=201,
        canonical_prediction_id=301, canonical_match_id=201,
        bookmaker_id=4, odds_snapshot_id=1, side="a", odds=3.0,
        model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
        prob_a=0.42, prob_b=0.58, model_prob=0.42, market_prob=context.prob_a,
        ev=9.0, status="new", evaluated_at=now, prediction_status="active",
        predicted_at=now - timedelta(seconds=20),
        data_cutoff_at=now - timedelta(minutes=1),
        start_time_normalized=context.start_at.isoformat(), league="LCK",
        team_a_name="Alpha", team_b_name="Beta", bookmaker_name="superbet",
        diagnostics_json=diagnostics,
        market_context_json=serialize_market_context(
            context, model_name=C0_NATIVE.name, model_version=C0_NATIVE.version,
        ),
    )
    cfg = SimpleNamespace(
        is_enabled=True, min_ev=0.05, min_odds=1.25, max_odds=12.0,
        cooldown_hours=6.0, discord_enabled=False, discord_webhook_url=None,
        telegram_enabled=False, telegram_bot_token=None, telegram_chat_id=None,
    )
    db = MagicMock()
    monkeypatch.setattr(alerts, "get_or_create_alert_config", lambda db: cfg)
    monkeypatch.setattr(alerts, "query_df", lambda *args: [row])
    monkeypatch.setattr(alerts, "load_prematch_quotes", lambda *args, **kwargs: {201: context})
    monkeypatch.setattr(alerts, "is_alert_on_cooldown", lambda *args: False)
    with patch.object(alerts, "send_discord_notification") as discord, patch.object(
        alerts, "send_telegram_notification"
    ) as telegram:
        result = alerts.scan_and_dispatch_ev_alerts(db)
        discord.assert_not_called()
        telegram.assert_not_called()

    if expected_ev is None:
        assert result["dispatched"] == 0
        assert result["alerts"] == []
        db.add.assert_not_called()
    else:
        assert result["dispatched"] == 1
        assert result["alerts"][0]["status"] == "simulated"
        assert result["alerts"][0]["ev"] == pytest.approx(expected_ev, abs=1e-12)
        assert json.loads(db.add.call_args.args[0].message_payload)["ev"] == pytest.approx(expected_ev)


def test_send_discord_notification_mock():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 204
        ok, err = send_discord_notification("https://discord.com/api/webhooks/test", {"title": "Test"})
        assert ok is True
        assert err is None

        # Failure case
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        ok2, err2 = send_discord_notification("https://discord.com/api/webhooks/test", {"title": "Test"})
        assert ok2 is False
        assert "Discord HTTP 400" in err2


def test_send_telegram_notification_mock():
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        ok, err = send_telegram_notification("token123", "chat123", "<b>Hello</b>")
        assert ok is True
        assert err is None

        # Failure case
        mock_post.return_value.status_code = 403
        mock_post.return_value.text = "Forbidden"
        ok2, err2 = send_telegram_notification("token123", "chat123", "<b>Hello</b>")
        assert ok2 is False
        assert "Telegram HTTP 403" in err2


def test_api_alerts_endpoints(client: TestClient):
    # 1. GET /api/alerts/config
    # 1. GET /alerts/config
    resp = client.get("/alerts/config")
    assert resp.status_code == 200
    cfg = resp.json()
    assert "is_enabled" in cfg
    assert "min_ev" in cfg
    assert "cooldown_hours" in cfg

    # 2. POST /alerts/config
    update_resp = client.post(
        "/alerts/config",
        json={
            "min_ev": 0.065,
            "min_odds": 1.35,
            "cooldown_hours": 4.0,
            "discord_webhook_url": "https://discord.com/api/webhooks/mock123456",
        },
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["min_ev"] == 0.065
    assert updated["min_odds"] == 1.35
    assert updated["cooldown_hours"] == 4.0
    assert updated["discord_configured"] is True
    # Secret must be masked!
    assert "mock123456" not in updated["discord_webhook_url_masked"]
    assert updated["discord_webhook_url_masked"].endswith("456")

    # 3. POST /alerts/check?dry_run=true
    check_resp = client.post("/alerts/check?dry_run=true")
    assert check_resp.status_code == 200
    check_data = check_resp.json()
    assert "dispatched" in check_data
    assert "skipped" in check_data

    # 4. POST /alerts/test
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 204
        test_resp = client.post("/alerts/test?channel=discord")
        assert test_resp.status_code == 200
        assert "results" in test_resp.json()

    # 5. GET /alerts/history
    hist_resp = client.get("/alerts/history")
    assert hist_resp.status_code == 200
    hist = hist_resp.json()
    assert "total" in hist
    assert isinstance(hist["alerts"], list)
    assert hist["total"] >= 1

def test_alert_conviction_and_roster_warning():
    signal = {
        "match_label": "Karmine Corp vs Vitality",
        "league": "LEC 2026",
        "team_name": "Vitality",
        "side": "b",
        "bookmaker_name": "Fortuna",
        "odds": 2.45,
        "ev": 0.085,
        "model_prob": 0.52,
        "market_prob": 0.44,
        "market_edge": 0.078,
        "suggested_stake": 2.5,
        "match_start_at": "2026-09-08 17:00 UTC",
        "conviction": "ROSTER_WARNING",
        "conviction_badge": "⚠️ SKŁAD NIEZWERYFIKOWANY",
        "is_full_roster": False,
    }
    embed = build_discord_embed(signal)
    assert embed["color"] == 0xF59E0B  # Amber/Orange for roster warning
    assert "⚠️ SKŁAD NIEZWERYFIKOWANY" in embed["title"]
    assert any("Skład częściowy" in f["value"] for f in embed["fields"])

    html = build_telegram_html(signal)
    assert "Przewaga nad rynkiem" in html
    assert "Skład częściowy" in html
