"""Router: /api/alerts — Value Bet alerting, notifications & settings."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from betting_app.api.deps import get_db
from betting_app.api.schemas import (
    AlertCheckResponse,
    AlertConfigResponse,
    AlertConfigUpdateRequest,
    AlertHistoryResponse,
    AlertLogEntry,
    AlertTestResponse,
)
from betting_app.models.alerts import ValueAlertLog
from betting_app.services.alert_service import (
    get_or_create_alert_config,
    scan_and_dispatch_ev_alerts,
    send_test_alert,
    update_alert_config,
)

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _mask_url_or_token(val: Optional[str]) -> Optional[str]:
    """Mask a webhook URL or secret token to protect credentials."""
    if not val or len(val) < 8:
        return None
    return f"{val[:6]}...{val[-4:]}"


def _build_config_response(cfg) -> AlertConfigResponse:
    env_discord = os.getenv("DISCORD_WEBHOOK_URL")
    env_tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    env_tg_chat = os.getenv("TELEGRAM_CHAT_ID")

    effective_discord = cfg.discord_webhook_url or env_discord
    effective_tg_token = cfg.telegram_bot_token or env_tg_token
    effective_tg_chat = cfg.telegram_chat_id or env_tg_chat

    return AlertConfigResponse(
        is_enabled=cfg.is_enabled,
        min_ev=cfg.min_ev,
        min_odds=cfg.min_odds,
        max_odds=cfg.max_odds,
        cooldown_hours=cfg.cooldown_hours,
        discord_enabled=cfg.discord_enabled,
        discord_configured=bool(effective_discord),
        discord_webhook_url_masked=_mask_url_or_token(effective_discord),
        telegram_enabled=cfg.telegram_enabled,
        telegram_configured=bool(effective_tg_token and effective_tg_chat),
        telegram_bot_token_masked=_mask_url_or_token(effective_tg_token),
        telegram_chat_id_masked=_mask_url_or_token(effective_tg_chat),
        updated_at=cfg.updated_at,
    )


@router.get("/config", response_model=AlertConfigResponse)
def get_alert_configuration(db: Session = Depends(get_db)):
    """Retrieve current alerting configuration and channel status."""
    cfg = get_or_create_alert_config(db)
    return _build_config_response(cfg)


@router.post("/config", response_model=AlertConfigResponse)
def update_alert_configuration(payload: AlertConfigUpdateRequest, db: Session = Depends(get_db)):
    """Update alerting thresholds, channels, or webhook credentials."""
    cfg = update_alert_config(
        db,
        is_enabled=payload.is_enabled,
        min_ev=payload.min_ev,
        min_odds=payload.min_odds,
        max_odds=payload.max_odds,
        cooldown_hours=payload.cooldown_hours,
        discord_enabled=payload.discord_enabled,
        discord_webhook_url=payload.discord_webhook_url,
        telegram_enabled=payload.telegram_enabled,
        telegram_bot_token=payload.telegram_bot_token,
        telegram_chat_id=payload.telegram_chat_id,
    )
    return _build_config_response(cfg)


@router.get("/history", response_model=AlertHistoryResponse)
def get_alert_history(
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieve audit history of dispatched Value Bet alerts."""
    query = db.query(ValueAlertLog)
    if status:
        query = query.filter(ValueAlertLog.status == status)

    logs = query.order_by(ValueAlertLog.created_at.desc()).limit(limit).all()

    entries = [
        AlertLogEntry(
            id=log.id,
            created_at=log.created_at,
            canonical_match_id=log.canonical_match_id,
            match_label=log.match_label,
            league=log.league,
            match_start_at=log.match_start_at,
            side=log.side,
            team_name=log.team_name,
            bookmaker_name=log.bookmaker_name,
            odds=log.odds,
            model_prob=log.model_prob,
            market_prob=log.market_prob,
            ev=log.ev,
            suggested_stake=log.suggested_stake,
            channels=log.channels,
            status=log.status,
            error_message=log.error_message,
        )
        for log in logs
    ]
    return AlertHistoryResponse(total=len(entries), alerts=entries)


@router.post("/check", response_model=AlertCheckResponse)
def check_and_dispatch(dry_run: bool = Query(False), db: Session = Depends(get_db)):
    """Manually trigger Value Bet scanning and notification dispatch."""
    res = scan_and_dispatch_ev_alerts(db, force_dry_run=dry_run)
    return AlertCheckResponse(
        dispatched=res["dispatched"],
        skipped=res["skipped"],
        failed=res["failed"],
        message=res["message"],
        alerts=res["alerts"],
    )


@router.post("/test", response_model=AlertTestResponse)
def trigger_test_alert(channel: str = Query("both"), db: Session = Depends(get_db)):
    """Send a test alert to verify Discord / Telegram webhook connectivity."""
    results = send_test_alert(db, channel=channel)
    return AlertTestResponse(results=results)
