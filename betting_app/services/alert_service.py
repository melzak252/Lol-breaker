"""Service for automated Value Bet alerting via Discord Webhooks and Telegram."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text
from sqlalchemy.orm import Session
from betting_app.models.alerts import AlertConfig, ValueAlertLog
from betting_app.api.deps import query_df, query_one
from betting_app.core.config import load_config
from betting_app.core.models import get_active_hybrid
from betting_app.services.prematch_quote_service import load_prematch_quotes, parse_aware_utc, signal_is_actionable
from betting_app.services.bet_qualification_service import qualify_prediction_sides
from betting_app.services.market_service import expected_value

ACTIVE_HYBRID = get_active_hybrid()

logger = logging.getLogger(__name__)

DEFAULT_MIN_EV = 0.05
DEFAULT_MIN_ODDS = 1.25
DEFAULT_MAX_ODDS = 12.0
DEFAULT_COOLDOWN_HOURS = 6.0


def get_or_create_alert_config(db: Session) -> AlertConfig:
    """Get persistent alert configuration from DB or initialize from environment defaults."""
    cfg = db.query(AlertConfig).filter(AlertConfig.id == 1).first()
    if not cfg:
        app_cfg = load_config()
        cfg = AlertConfig(
            id=1,
            is_enabled=True,
            min_ev=float(os.getenv("ALERT_MIN_EV", str(DEFAULT_MIN_EV))),
            min_odds=float(os.getenv("ALERT_MIN_ODDS", str(DEFAULT_MIN_ODDS))),
            max_odds=float(os.getenv("ALERT_MAX_ODDS", str(DEFAULT_MAX_ODDS))),
            cooldown_hours=float(os.getenv("ALERT_COOLDOWN_HOURS", str(DEFAULT_COOLDOWN_HOURS))),
            discord_enabled=True,
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
            telegram_enabled=True,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        db.add(cfg)
        db.commit()
        db.refresh(cfg)
    return cfg


def update_alert_config(
    db: Session,
    *,
    is_enabled: Optional[bool] = None,
    min_ev: Optional[float] = None,
    min_odds: Optional[float] = None,
    max_odds: Optional[float] = None,
    cooldown_hours: Optional[float] = None,
    discord_enabled: Optional[bool] = None,
    discord_webhook_url: Optional[str] = None,
    telegram_enabled: Optional[bool] = None,
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
) -> AlertConfig:
    """Update alert configuration parameters."""
    cfg = get_or_create_alert_config(db)
    if is_enabled is not None:
        cfg.is_enabled = is_enabled
    if min_ev is not None:
        cfg.min_ev = min_ev
    if min_odds is not None:
        cfg.min_odds = min_odds
    if max_odds is not None:
        cfg.max_odds = max_odds
    if cooldown_hours is not None:
        cfg.cooldown_hours = cooldown_hours
    if discord_enabled is not None:
        cfg.discord_enabled = discord_enabled
    if discord_webhook_url is not None:
        cfg.discord_webhook_url = discord_webhook_url.strip() if discord_webhook_url else None
    if telegram_enabled is not None:
        cfg.telegram_enabled = telegram_enabled
    if telegram_bot_token is not None:
        cfg.telegram_bot_token = telegram_bot_token.strip() if telegram_bot_token else None
    if telegram_chat_id is not None:
        cfg.telegram_chat_id = telegram_chat_id.strip() if telegram_chat_id else None

    cfg.updated_at = datetime.now(timezone.utc).isoformat()
    db.commit()
    db.refresh(cfg)
    return cfg


def build_discord_embed(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Format a Discord Rich Embed payload for a Value Bet."""
    match_label = signal.get("match_label", "League of Legends Match")
    league = signal.get("league") or "LoL Esports"
    team_name = signal.get("team_name", "Unknown Team")
    side = (signal.get("side") or "").upper()
    bookmaker = signal.get("bookmaker_name", "Bukmacher")
    odds = float(signal.get("odds") or 0.0)
    ev = float(signal.get("ev") or 0.0)
    model_prob = float(signal.get("model_prob") or 0.0)
    market_prob = float(signal.get("market_prob") or 0.0)
    stake_pct = float(signal.get("suggested_stake") or 0.0)
    match_start = signal.get("match_start_at") or "Wkrótce"
    conviction = signal.get("conviction", "HIGH_CONVICTION")
    conviction_badge = signal.get("conviction_badge") or ("🔥 WYSOKA PEWNOŚĆ" if ev >= 0.10 else "⚡ SPEKULACYJNY")
    market_edge = float(signal.get("market_edge") or 0.0)
    is_full_roster = signal.get("is_full_roster", True)

    # Color: Bright Green for strong EV (>=10%), Cyan for 5-10%
    if ev >= 0.10:
        color = 0x10B981
    elif conviction == "ROSTER_WARNING":
        color = 0xF59E0B
    else:
        color = 0x38BDF8

    edge_str = f"+{market_edge * 100:.1f}%" if market_edge >= 0 else f"{market_edge * 100:.1f}% (pod prąd rynku)"
    roster_str = "✅ Pełny skład (5/5 zweryfikowany)" if is_full_roster else "⚠️ Skład częściowy / niepotwierdzony"

    fields = [
        {"name": "🎯 Typowana Drużyna", "value": f"**{team_name}** ({side}) • `{conviction_badge}`", "inline": True},
        {"name": "🏢 Bukmacher", "value": f"**{bookmaker}**", "inline": True},
        {"name": "📈 Kurs i Przewaga", "value": f"**{odds:.2f}** (Edge: `{edge_str}`)", "inline": True},
        {"name": "💰 EV (po podatku 12%)", "value": f"**+{ev * 100:.1f}%**", "inline": True},
        {"name": "📊 Model vs Rynek", "value": f"Model: `{model_prob * 100:.1f}%`\nRynek: `{market_prob * 100:.1f}%`", "inline": True},
        {"name": "💵 Sugerowana Stawka", "value": f"`{stake_pct:.1f}%` bankrolla", "inline": True},
        {"name": "👥 Weryfikacja Składu", "value": roster_str, "inline": True},
        {"name": "⏰ Początek Meczu", "value": f"{match_start}", "inline": False},
    ]

    return {
        "title": f"🔥 VALUE BET: {match_label} [{conviction_badge}]",
        "description": f"Zidentyfikowano dodatnią wartość oczekiwaną w lidze **{league}**.",
        "color": color,
        "fields": fields,
        "footer": {"text": "LoL-Breaker • LoL Value Bet Intelligence"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_telegram_html(signal: Dict[str, Any]) -> str:
    """Format a Telegram HTML message for a Value Bet."""
    match_label = signal.get("match_label", "League of Legends Match")
    league = signal.get("league") or "LoL Esports"
    team_name = signal.get("team_name", "Unknown Team")
    side = (signal.get("side") or "").upper()
    bookmaker = signal.get("bookmaker_name", "Bukmacher")
    odds = float(signal.get("odds") or 0.0)
    ev = float(signal.get("ev") or 0.0)
    model_prob = float(signal.get("model_prob") or 0.0)
    market_prob = float(signal.get("market_prob") or 0.0)
    stake_pct = float(signal.get("suggested_stake") or 0.0)
    match_start = signal.get("match_start_at") or "Wkrótce"
    conviction_badge = signal.get("conviction_badge") or "🔥 VALUE BET"
    market_edge = float(signal.get("market_edge") or 0.0)
    is_full_roster = signal.get("is_full_roster", True)

    edge_str = f"(Przewaga nad rynkiem: <b>+{market_edge * 100:.1f}%</b>)" if market_edge >= 0 else f"(⚠️ Pod prąd rynku: <b>{market_edge * 100:.1f}%</b>)"
    roster_str = "✅ Pełny skład (5/5)" if is_full_roster else "⚠️ <b>Skład częściowy/niepotwierdzony!</b>"

    return (
        f"🔥 <b>VALUE BET ALERT</b> [{conviction_badge}]\n\n"
        f"🏆 <b>Mecz:</b> {match_label} (<i>{league}</i>)\n"
        f"🎯 <b>Typ:</b> <b>{team_name}</b> [Strona {side}]\n"
        f"🏢 <b>Bukmacher:</b> <b>{bookmaker}</b>\n"
        f"📈 <b>Kurs:</b> <b>{odds:.2f}</b> {edge_str}\n"
        f"💰 <b>EV (po podatku 12%):</b> <b>+{ev * 100:.1f}%</b>\n"
        f"👥 <b>Skład:</b> {roster_str}\n"
        f"📊 <b>Prawdopodobieństwo:</b> Model: {model_prob * 100:.1f}% | Rynek: {market_prob * 100:.1f}%\n"
        f"💵 <b>Sugerowana stawka (Kelly):</b> {stake_pct:.1f}% bankrolla\n"
        f"⏰ <b>Start meczu:</b> {match_start}\n\n"
        f"⚡ <i>LoL-Breaker Intelligence</i>"
    )

def send_discord_notification(webhook_url: str, embed: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Send a notification to Discord webhook. Returns (success, error_message)."""
    if not webhook_url:
        return False, "Brak skonfigurowanego URL Discord Webhook"
    try:
        payload = {
            "username": "LoL-Breaker Alerts",
            "avatar_url": "https://raw.githubusercontent.com/FortAwesome/Font-Awesome/master/svgs/solid/fire.svg",
            "embeds": [embed],
        }
        res = requests.post(webhook_url, json=payload, timeout=8)
        if 200 <= res.status_code < 300:
            return True, None
        return False, f"Discord HTTP {res.status_code}: {res.text[:200]}"
    except Exception as e:
        logger.warning("Failed to send Discord webhook alert: %s", e)
        return False, str(e)


def send_telegram_notification(token: str, chat_id: str, html_text: str) -> tuple[bool, Optional[str]]:
    """Send a notification to Telegram chat. Returns (success, error_message)."""
    if not token or not chat_id:
        return False, "Brak skonfigurowanego tokenu lub Chat ID Telegrama"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        res = requests.post(url, json=payload, timeout=8)
        if 200 <= res.status_code < 300:
            return True, None
        return False, f"Telegram HTTP {res.status_code}: {res.text[:200]}"
    except Exception as e:
        logger.warning("Failed to send Telegram alert: %s", e)
        return False, str(e)


def is_alert_on_cooldown(
    db: Session,
    canonical_match_id: Optional[int],
    bookmaker_name: Optional[str],
    side: Optional[str],
    new_odds: float,
    new_ev: float,
    cooldown_hours: float,
) -> bool:
    """Check whether an alert for this exact match, bookmaker, and side was sent recently."""
    if not canonical_match_id or not bookmaker_name or not side:
        return False

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)).isoformat()

    row = query_one(
        db,
        """
        SELECT odds, ev
        FROM value_alert_logs
        WHERE canonical_match_id = :mid
          AND bookmaker_name = :bname
          AND side = :side
          AND status IN ('sent', 'simulated')
          AND created_at >= :cutoff
        ORDER BY created_at DESC
        LIMIT 1
        """,
        {"mid": canonical_match_id, "bname": bookmaker_name, "side": side, "cutoff": cutoff},
    )
    if not row:
        return False

    old_odds = float(row.get("odds") or 0.0)
    old_ev = float(row.get("ev") or 0.0)

    # If odds increased by at least 0.05 or EV grew by at least 2.0 percentage points, allow re-alerting!
    if new_odds >= old_odds + 0.05 or new_ev >= old_ev + 0.02:
        return False

    return True


def scan_and_dispatch_ev_alerts(db: Session, force_dry_run: bool = False) -> Dict[str, Any]:
    """Scan the database for valid upcoming EV+ signals and dispatch notifications."""
    cfg = get_or_create_alert_config(db)
    if not cfg.is_enabled and not force_dry_run:
        return {
            "dispatched": 0,
            "skipped": 0,
            "failed": 0,
            "alerts": [],
            "message": "Powiadomienia Value Bet są wyłączone w konfiguracji.",
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    min_ev = cfg.min_ev
    min_odds = cfg.min_odds
    max_odds = cfg.max_odds or 999.0
    cooldown_hours = cfg.cooldown_hours

    # Fetch active new EV signals for matches starting in the future
    sql = """
        SELECT
            mes.*, mes.id AS signal_id,
            cp.id AS prediction_id, cp.canonical_match_id AS prediction_match_id,
            cp.model_name, cp.model_version, cp.predicted_at, cp.data_cutoff_at,
            cp.prediction_status, cp.prob_a, cp.prob_b,
            cm.team_a_name,
            cm.team_b_name,
            cm.league,
            cm.start_time_normalized,
            b.name AS bookmaker_name,
            umf.feature_status,
            cp.diagnostics_json
        FROM model_ev_signals mes
        JOIN canonical_matches cm ON cm.id = mes.canonical_match_id
        JOIN bookmakers b ON b.id = mes.bookmaker_id
        JOIN canonical_predictions cp ON cp.id = mes.canonical_prediction_id
          AND cp.model_name = :model_name
          AND cp.model_version = :model_version
          AND cp.prediction_status = 'active'
        LEFT JOIN (
            SELECT canonical_match_id, MAX(feature_status) AS feature_status
            FROM upcoming_match_features
            GROUP BY canonical_match_id
        ) umf ON umf.canonical_match_id = mes.canonical_match_id
        WHERE mes.status = 'new'
          AND mes.odds >= :min_odds
          AND mes.odds <= :max_odds
        ORDER BY mes.ev DESC
    """

    rows = query_df(
        db,
        sql,
        {
            "min_ev": min_ev,
            "min_odds": min_odds,
            "max_odds": max_odds,
            "now_iso": now_iso,
            "model_name": ACTIVE_HYBRID.hybrid_model_name,
            "model_version": ACTIVE_HYBRID.hybrid_model_version,
        },
    )
    decision_at = datetime.now(timezone.utc)
    contexts = load_prematch_quotes(sorted({int(r["canonical_match_id"]) for r in rows}), decision_at=decision_at, session=db)
    admitted_rows = []
    for row in rows:
        current = contexts[int(row["canonical_match_id"])]
        evaluated = parse_aware_utc(row.get("evaluated_at"))
        if evaluated is None:
            continue
        evaluation = load_prematch_quotes([current.canonical_match_id], decision_at=evaluated, session=db)[current.canonical_match_id]
        prediction = dict(row, id=row["prediction_id"], canonical_match_id=row["prediction_match_id"])
        if not signal_is_actionable(row, prediction, evaluation_context=evaluation, current_context=current, session=db)[0]:
            continue
        side = row.get("side")
        quote = next((quote for quote in current.quotes
                      if quote.odds_snapshot_id == row["odds_snapshot_id"]
                      and quote.bookmaker_id == row["bookmaker_id"]), None)
        if quote is None:
            continue
        choices = qualify_prediction_sides(
            prob_a=row["prob_a"], prob_b=row["prob_b"], diagnostics=row.get("diagnostics_json"),
            model_name=row["model_name"], odds_a=quote.odds_a, odds_b=quote.odds_b,
            market_prob_a=current.prob_a,
            league=row.get("league"), match_date=row.get("start_time_normalized"))
        if not choices[side][0]:
            continue
        raw_ev = expected_value(choices[side][2]["prob_for_ev"], row["odds"], .12)
        if raw_ev < min_ev:
            continue
        row["ev"] = raw_ev
        admitted_rows.append(row)
    rows = sorted(admitted_rows, key=lambda row: (-row["ev"], row["signal_id"]))[:25]

    dispatched = 0
    skipped = 0
    failed = 0
    alert_results = []

    # Check which channels are active
    discord_url = cfg.discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    tg_token = cfg.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = cfg.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")

    can_discord = bool(cfg.discord_enabled and discord_url)
    can_telegram = bool(cfg.telegram_enabled and tg_token and tg_chat)

    for row in rows:
        side = str(row.get("side") or "").lower()
        team_a = row.get("team_a_name") or "Team A"
        team_b = row.get("team_b_name") or "Team B"
        team_name = team_a if side == "a" else team_b
        match_label = f"{team_a} vs {team_b}"
        bookmaker_name = row.get("bookmaker_name") or "Bukmacher"
        odds = float(row.get("odds") or 0.0)
        ev = float(row.get("ev") or 0.0)
        model_prob = float(row.get("model_prob") or 0.0)
        market_prob = float(row.get("market_prob") or 0.0)
        cm_id = int(row.get("canonical_match_id") or 0) or None

        # Check cooldown
        if is_alert_on_cooldown(db, cm_id, bookmaker_name, side, odds, ev, cooldown_hours):
            skipped += 1
            continue

        # Market edge vs consensus line
        market_edge = (odds * market_prob) - 1.0 if (odds > 0 and market_prob > 0) else 0.0

        # Roster verification status
        feature_status = str(row.get("feature_status") or "").lower()
        is_full_roster = feature_status in {"ready_player", "ready"}

        # Conformal risk / Epistemic uncertainty check from diagnostics
        p_low = None
        p_low_ev = None
        diag_raw = row.get("diagnostics_json")
        if diag_raw:
            try:
                diag = json.loads(diag_raw) if isinstance(diag_raw, str) else diag_raw
                p_low_key = f"prob_risk_adjusted_p_low_{side}"
                if p_low_key in diag:
                    p_low = float(diag[p_low_key])
                elif f"res_p_low_{side}" in diag:
                    p_low = float(diag[f"res_p_low_{side}"])
                if p_low is not None and 0 < p_low < 1:
                    p_low_ev = p_low * odds * 0.88 - 1.0
            except Exception:
                pass

        # Conviction classification:
        # 1. HIGH CONVICTION: Verified full roster AND (market edge > 0 or conformal lower-bound EV > 0)
        # 2. ROSTER WARNING: Unverified or partial roster (high operational hazard)
        # 3. SPECULATIVE: Full roster, but betting against market steam (market_edge < 0 and p_low_ev <= 0)
        if is_full_roster and (market_edge >= 0.0 or (p_low_ev is not None and p_low_ev > 0)):
            conviction = "HIGH_CONVICTION"
            conviction_badge = "🔥 WYSOKA PEWNOŚĆ"
        elif feature_status in {"partial", "missing_ratings", "unknown", ""}:
            conviction = "ROSTER_WARNING"
            conviction_badge = "⚠️ SKŁAD NIEZWERYFIKOWANY"
        else:
            conviction = "SPECULATIVE"
            conviction_badge = "⚡ SPEKULACYJNY"

        raw_stake = float(row.get("stake_suggestion") or 0.0)
        # Apply defensive scaling: halve suggested stake if not high conviction
        suggested_stake = round(raw_stake * 0.5, 2) if conviction != "HIGH_CONVICTION" else raw_stake

        signal_data = {
            "canonical_match_id": cm_id,
            "match_label": match_label,
            "league": row.get("league"),
            "match_start_at": row.get("start_time_normalized"),
            "side": side,
            "team_name": team_name,
            "bookmaker_name": bookmaker_name,
            "odds": odds,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "market_edge": market_edge,
            "ev": ev,
            "suggested_stake": suggested_stake,
            "conviction": conviction,
            "conviction_badge": conviction_badge,
            "is_full_roster": is_full_roster,
            "p_low": p_low,
            "p_low_ev": p_low_ev,
        }

        channel_names = []
        errors = []

        if force_dry_run or (not can_discord and not can_telegram):
            # Simulated mode
            status = "simulated"
            channel_names.append("simulation")
            dispatched += 1
        else:
            # Real send
            if can_discord and discord_url:
                embed = build_discord_embed(signal_data)
                ok, err = send_discord_notification(discord_url, embed)
                if ok:
                    channel_names.append("discord")
                else:
                    errors.append(f"Discord: {err}")

            if can_telegram and tg_token and tg_chat:
                html_msg = build_telegram_html(signal_data)
                ok, err = send_telegram_notification(tg_token, tg_chat, html_msg)
                if ok:
                    channel_names.append("telegram")
                else:
                    errors.append(f"Telegram: {err}")

            if channel_names:
                status = "sent"
                dispatched += 1
            else:
                status = "failed"
                failed += 1

        # Audit log entry
        log_entry = ValueAlertLog(
            created_at=datetime.now(timezone.utc).isoformat(),
            canonical_match_id=cm_id,
            match_label=match_label,
            league=row.get("league"),
            match_start_at=row.get("start_time_normalized"),
            side=side,
            team_name=team_name,
            bookmaker_name=bookmaker_name,
            odds=odds,
            model_prob=row.get("model_prob"),
            market_prob=row.get("market_prob"),
            ev=ev,
            suggested_stake=row.get("stake_suggestion"),
            channels=",".join(channel_names) if channel_names else "none",
            status=status,
            message_payload=json.dumps(signal_data),
            error_message="; ".join(errors) if errors else None,
        )
        db.add(log_entry)
        alert_results.append({
            "match": match_label,
            "team": team_name,
            "bookmaker": bookmaker_name,
            "odds": odds,
            "ev": ev,
            "channels": channel_names,
            "status": status,
        })

    db.commit()
    return {
        "dispatched": dispatched,
        "skipped": skipped,
        "failed": failed,
        "alerts": alert_results,
        "message": f"Przeskanowano sygnały: wysłano {dispatched}, pominięto (cooldown) {skipped}, błędów {failed}.",
    }


def send_test_alert(db: Session, channel: str = "both") -> Dict[str, Any]:
    """Send a test Value Bet notification to verify webhook/bot connectivity."""
    cfg = get_or_create_alert_config(db)

    test_signal = {
        "match_label": "Bilibili Gaming vs Anyone's Legend",
        "league": "LPL 2026 Split 3",
        "side": "b",
        "team_name": "Anyone's Legend",
        "bookmaker_name": "Superbet",
        "odds": 2.45,
        "model_prob": 0.485,
        "market_prob": 0.390,
        "ev": 0.076,  # +7.6% EV
        "suggested_stake": 2.5,
        "match_start_at": (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M UTC"),
    }

    results = {}
    discord_url = cfg.discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    tg_token = cfg.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = cfg.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if channel in {"both", "discord"}:
        if discord_url:
            embed = build_discord_embed(test_signal)
            embed["title"] = "🧪 [TEST] " + embed["title"]
            ok, err = send_discord_notification(discord_url, embed)
            results["discord"] = {"ok": ok, "error": err}
        else:
            results["discord"] = {"ok": False, "error": "Brak skonfigurowanego URL Discord Webhook"}

    if channel in {"both", "telegram"}:
        if tg_token and tg_chat:
            html_msg = "🧪 <b>[POWIADOMIENIE TESTOWE]</b>\n\n" + build_telegram_html(test_signal)
            ok, err = send_telegram_notification(tg_token, tg_chat, html_msg)
            results["telegram"] = {"ok": ok, "error": err}
        else:
            results["telegram"] = {"ok": False, "error": "Brak skonfigurowanego tokenu lub Chat ID Telegrama"}

    # Log the test
    log_entry = ValueAlertLog(
        created_at=datetime.now(timezone.utc).isoformat(),
        match_label="[TEST] Bilibili Gaming vs Anyone's Legend",
        league="LPL 2026 Split 3",
        side="b",
        team_name="Anyone's Legend",
        bookmaker_name="Superbet",
        odds=2.45,
        model_prob=0.485,
        market_prob=0.390,
        ev=0.076,
        suggested_stake=2.5,
        channels=channel,
        status="test",
        message_payload=json.dumps(test_signal),
        error_message=str(results),
    )
    db.add(log_entry)
    db.commit()

    return results
