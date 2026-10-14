"""Scheduled tasks for OddsPapi fixture discovery and horizon odds collection."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from betting_app.services.oddspapi_service import (
    OddsPapiClient,
    fetch_pinnacle_horizon_odds,
    sync_oddspapi_fixtures,
)

logger = logging.getLogger(__name__)


def sync_fixtures_task() -> dict[str, Any]:
    """Sync upcoming LoL fixtures from OddsPapi to local mappings.

    Scheduled every 3 days. Consumes exactly 1 request.
    """
    logger.info("Starting OddsPapi fixture sync task")
    start = datetime.now(UTC)
    client = OddsPapiClient()
    if not client.is_configured():
        logger.info("Skipping OddsPapi fixture sync: ODDSPAPI_API_KEY not configured")
        return {"status": "skipped", "reason": "ODDSPAPI_API_KEY not configured"}

    result = sync_oddspapi_fixtures(client=client)
    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"OddsPapi fixture sync finished in {duration:.1f}s: {result}")
    return {"result": result, "duration_s": duration}


def fetch_horizon_odds_task(
    target_horizon_hours: float = 6.0,
    tolerance_hours: float = 1.0,
    max_requests: int = 2,
) -> dict[str, Any]:
    """Fetch Pinnacle pre-match odds around the target horizon window (default T−6h).

    Scheduled periodically. Strictly respects daily and monthly quota boundaries.
    """
    logger.info(f"Starting OddsPapi horizon odds fetch (T−{target_horizon_hours}h)")
    start = datetime.now(UTC)
    client = OddsPapiClient()
    if not client.is_configured():
        logger.info("Skipping OddsPapi horizon odds fetch: ODDSPAPI_API_KEY not configured")
        return {"status": "skipped", "reason": "ODDSPAPI_API_KEY not configured"}

    result = fetch_pinnacle_horizon_odds(
        target_horizon_hours=target_horizon_hours,
        tolerance_hours=tolerance_hours,
        max_requests=max_requests,
        client=client,
    )
    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"OddsPapi horizon odds fetch finished in {duration:.1f}s: {result}")
    return {"result": result, "duration_s": duration}
