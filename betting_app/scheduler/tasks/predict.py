"""Prediction pipeline tasks."""

import logging
from datetime import UTC, datetime

from .scrape import _run_module

logger = logging.getLogger(__name__)


def rematch_canonical() -> bool:
    """Rematch scraped matches to canonical matches."""
    logger.info("Rematching canonical matches")
    return _run_module(
        "betting_app.scripts.rematch_canonical_matches",
        args=["--no-overview"],
        timeout=300,
    )


def sync_liquipedia_daily() -> dict:
    """Collect ticker formats once; roster verification has its own daily job."""
    logger.info("Running daily Liquipedia ticker sync")
    ok = _run_module(
        "betting_app.scripts.sync_liquipedia_bon",
        args=["--limit", "60"],
        timeout=300,
    )
    return {"success": ok}


def dispatch_value_alerts() -> dict:
    """Scan upcoming EV+ signals and dispatch notifications to Discord / Telegram."""
    logger.info("Scanning and dispatching Value Bet alerts")
    from betting_app.core.db import get_session
    from betting_app.services.alert_service import scan_and_dispatch_ev_alerts

    with get_session() as session:
        result = scan_and_dispatch_ev_alerts(session)
    logger.info(
        "Alert dispatch finished: %d dispatched, %d skipped, %d failed",
        result.get("dispatched", 0),
        result.get("skipped", 0),
        result.get("failed", 0),
    )
    return result

def run_prediction_pipeline() -> dict:
    """Run the prediction pipeline:
    1. Rematch canonical matches
    2. Build features + predict + hybrid + EV signals
    """
    logger.info("Starting prediction pipeline")
    start = datetime.now(UTC)
    
    rematch_ok = rematch_canonical()
    alerts_dispatched = None
    if rematch_ok:
        predict_ok = _run_module(
            "betting_app.scripts.run_upcoming_prediction_pipeline",
            args=["--include-partial", "--operational-hybrid"],
            timeout=900,
        )
        if predict_ok:
            try:
                alerts_dispatched = dispatch_value_alerts()
                logger.info("Auto-dispatched value alerts: %s", alerts_dispatched)
            except Exception as alert_err:
                logger.warning("Failed to auto-dispatch value alerts after prediction: %s", alert_err)
    else:
        logger.error("Skipping prediction because canonical rematching failed")
        predict_ok = False

    steps = {
        "rematch": rematch_ok,
        "predict": predict_ok,
    }
    duration = (datetime.now(UTC) - start).total_seconds()
    all_ok = all(steps.values())

    logger.info(f"Prediction pipeline: {'OK' if all_ok else 'PARTIAL'} ({duration:.1f}s)")

    return {
        "success": all_ok,
        "steps": steps,
        "duration_s": duration,
        "alerts": alerts_dispatched,
        "error": "Canonical rematching failed" if not rematch_ok else None,
    }
