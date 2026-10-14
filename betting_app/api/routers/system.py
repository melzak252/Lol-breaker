"""Router: /api/system, /api/bookmakers, /api/automation."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from betting_app.api.deps import get_db, query_df, query_one
from betting_app.api.schemas import (
    AutomationTriggerResponse,
    BookmakerStatus,
    HealthResponse,
    SystemStatusResponse,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status(db=Depends(get_db)):
    counts = {
        r["key"]: r["value"]
        for r in query_df(
            db,
            """
            SELECT 'canonical_matches' AS key, COUNT(*) AS value FROM canonical_matches
            UNION ALL SELECT 'odds_snapshots', COUNT(*) FROM odds_snapshots
            UNION ALL SELECT 'bookmakers_latest', COUNT(DISTINCT bookmaker_id) FROM odds_snapshots
            UNION ALL SELECT 'active_predictions', COUNT(*) FROM canonical_predictions WHERE prediction_status='active'
            UNION ALL SELECT 'new_ev_signals', COUNT(*) FROM model_ev_signals WHERE status='new'
            UNION ALL SELECT 'ready_features', COUNT(*) FROM upcoming_match_features WHERE feature_status LIKE 'ready%'
            UNION ALL SELECT 'golgg_matches', COUNT(*) FROM golgg_matches
            UNION ALL SELECT 'golgg_games', COUNT(*) FROM golgg_games
            UNION ALL SELECT 'golgg_teams', COUNT(*) FROM golgg_teams
            UNION ALL SELECT 'entity_ratings', COUNT(*) FROM entity_ratings
            UNION ALL SELECT 'team_rolling_features', COUNT(*) FROM team_rolling_features
            UNION ALL SELECT 'wallets', COUNT(*) FROM bookmaker_accounts WHERE is_active=1
            UNION ALL SELECT 'bets', COUNT(*) FROM bets
            """,
        )
    }

    last_scrape = query_df(
        db,
        """
        SELECT b.name, MAX(os.scraped_at) AS last_scraped_at,
               COUNT(*) AS snapshot_count
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id=os.bookmaker_id
        GROUP BY b.name
        ORDER BY last_scraped_at DESC
        LIMIT 20
        """,
    )

    last_auto = query_df(
        db,
        """
        SELECT run_type, status, started_at, finished_at, error AS summary
        FROM automation_runs
        ORDER BY started_at DESC
        LIMIT 10
        """,
    )

    return SystemStatusResponse(
        counts=counts,
        last_scrape_runs=last_scrape,
        last_automation_runs=last_auto,
    )


@router.get("/bookmakers", response_model=list[BookmakerStatus])
def list_bookmakers(db=Depends(get_db)):
    rows = query_df(
        db,
        """
        SELECT b.id, b.name, b.base_url,
               MAX(os.scraped_at) AS last_scraped_at,
               COUNT(*) AS snapshot_count
        FROM bookmakers b
        LEFT JOIN odds_snapshots os ON os.bookmaker_id=b.id
        WHERE b.is_active='1'
        GROUP BY b.id
        ORDER BY b.name
        """,
    )
    return [
        BookmakerStatus(
            id=r["id"],
            name=r["name"],
            base_url=r.get("base_url"),
            last_scraped_at=r.get("last_scraped_at"),
            snapshot_count=r["snapshot_count"],
        )
        for r in rows
    ]


@router.post("/automation/light-cycle", response_model=AutomationTriggerResponse)
def trigger_light_cycle():
    try:
        from betting_app.scheduler.tasks import scrape, predict
        scrape.scrape_all()
        predict.run_prediction_pipeline()
        return AutomationTriggerResponse(status="completed", message="Light cycle finished")
    except Exception as exc:
        return AutomationTriggerResponse(status="error", message=f"Light cycle failed: {exc}")


@router.post("/automation/backup", response_model=AutomationTriggerResponse)
def trigger_backup():
    try:
        from betting_app.scripts import backup_sqlite as backup
        result = backup.create_backup()
        return AutomationTriggerResponse(status="completed", message=f"Backup created: {result}")
    except Exception as exc:
        return AutomationTriggerResponse(status="error", message=f"Backup failed: {exc}")
