"""Task registry - central place to define all scheduled tasks."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskDefinition:
    """Definition of a scheduled task."""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)  # Arguments to pass to func
    kwargs: dict = field(default_factory=dict)  # Keyword arguments
    cron_trigger: Optional[str] = None  # Cron expression
    interval_minutes: Optional[int] = None  # Interval in minutes
    description: str = ""
    enabled: bool = True
    lock_key: Optional[str] = None  # Cross-process mutual-exclusion group


class TaskRegistry:
    """Registry of all available tasks."""
    
    def __init__(self):
        self._tasks: dict[str, TaskDefinition] = {}
    
    def register(self, task: TaskDefinition):
        """Register a new task."""
        if task.id in self._tasks:
            logger.warning(f"Task {task.id} already registered, overwriting")
        self._tasks[task.id] = task
        logger.info(f"Registered task: {task.id} ({task.name})")
    
    def get(self, task_id: str) -> Optional[TaskDefinition]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def list_all(self) -> list[TaskDefinition]:
        """List all registered tasks."""
        return list(self._tasks.values())
    
    def list_enabled(self) -> list[TaskDefinition]:
        """List all enabled tasks."""
        return [t for t in self._tasks.values() if t.enabled]


# Global registry instance
registry = TaskRegistry()


def register_all_tasks():
    """Register all tasks in the system."""
    from .tasks import scrape, predict, maintenance, ml, oddspapi
    
    # Scrape tasks - run at :15 and :45 every hour (captures odds before :30 and :00 match starts)
    for bookmaker in scrape.BOOKMAKERS:
        registry.register(TaskDefinition(
            id=f"scrape_{bookmaker}",
            name=f"Scrape {bookmaker.title()}",
            func=scrape.scrape_bookmaker,
            args=(bookmaker,),
            cron_trigger="15,45 * * * *",  # At :15 and :45 every hour
            description=f"Scrape odds from {bookmaker}",
            enabled=True
        ))
    # Scrape prop odds for Polish bookmakers (IDEA-018)
    registry.register(TaskDefinition(
        id="scrape_prop_odds",
        name="Scrape Prop Odds (STS, Fortuna, Betclic, Superbet)",
        func=scrape.scrape_prop_odds,
        cron_trigger="25 * * * *",  # Once per hour at :25 to prevent bookmaker rate-limiting
        description="Scrape in-game proposition odds (kills, duration, objectives) from Polish bookmakers",
        enabled=True,
        lock_key="scrape_props",
    ))
    
    # Prediction pipeline - run at :20 and :50 every hour (5 min after scraping, 10 min before matches)
    registry.register(TaskDefinition(
        id="prediction_pipeline",
        name="Prediction Pipeline",
        func=predict.run_prediction_pipeline,
        cron_trigger="20,50 * * * *",  # At minute 20 and 50 every hour
        description="Run full prediction pipeline",
        enabled=True
    ))

    # Value Bet alerts dispatch - runs at :23 and :53 every hour (3 min after prediction pipeline, 7 min before matches)
    registry.register(TaskDefinition(
        id="value_alerts_dispatch",
        name="Value Bet Alerts Dispatch",
        func=predict.dispatch_value_alerts,
        cron_trigger="23,53 * * * *",  # At minute 23 and 53 every hour
        description="Scan upcoming EV+ opportunities and send Discord/Telegram notifications",
        enabled=True
    ))

    # Shadow ML inference - run at :25 and :55 every hour (after the main prediction pipeline)
    registry.register(TaskDefinition(
        id="shadow_ml_inference",
        name="Shadow ML Inference",
        func=ml.run_shadow_inference,
        cron_trigger="25,55 * * * *",  # At minute 25 and 55 every hour
        description="Run registered shadow ML models without replacing production predictions",
        enabled=True
    ))
    
    # Heavy maintenance components remain available for manual execution, while
    # the scheduled cycle enforces refresh -> ratings -> features ordering.
    registry.register(TaskDefinition(
        id="refresh_golgg",
        name="Refresh GolGG Data",
        func=maintenance.refresh_golgg,
        description="Refresh GolGG match data",
        enabled=True,
        lock_key="heavy_maintenance",
    ))

    registry.register(TaskDefinition(
        id="rebuild_ratings",
        name="Rebuild Team Ratings",
        func=maintenance.rebuild_ratings,
        description="Rebuild team Elo ratings",
        enabled=True,
        lock_key="heavy_maintenance",
    ))

    registry.register(TaskDefinition(
        id="rebuild_features",
        name="Rebuild Rolling Features",
        func=maintenance.rebuild_rolling_features,
        description="Rebuild W20 rolling features",
        enabled=True,
        lock_key="heavy_maintenance",
    ))

    registry.register(TaskDefinition(
        id="heavy_maintenance_cycle",
        name="Heavy Maintenance Cycle",
        func=maintenance.run_heavy_cycle,
        cron_trigger="40 0,6,12,18 * * *",
        description="Refresh GolGG, then rebuild ratings and rolling features",
        enabled=True,
        lock_key="heavy_maintenance",
    ))

    # Weekly ML retraining - Sunday early morning UTC
    registry.register(TaskDefinition(
        id="weekly_ml_retraining",
        name="Weekly ML Retraining",
        func=ml.run_weekly_retraining,
        cron_trigger="30 3 * * sun",  # Sunday 03:30 UTC
        description="Train/evaluate/register an EXP-039 family retrain candidate",
        enabled=True
    ))

    # Production model healthcheck - once per day after the heavy maintenance cycle
    registry.register(TaskDefinition(
        id="thesis_model_healthcheck",
        name="Thesis Model Healthcheck",
        func=ml.run_thesis_model_healthcheck,
        cron_trigger="45 4 * * *",  # Daily 04:45 UTC
        description="Read-only 90-day evaluation of the current thesis model vs market",
        enabled=True
    ))

    # Embedding components remain manually triggerable. The scheduled cycle
    # serializes both GPU-heavy refreshes and prevents accidental overlap.
    registry.register(TaskDefinition(
        id="refresh_champion_role_embeddings",
        name="Refresh Champion Role Embeddings",
        func=ml.refresh_champion_role_embeddings,
        description="Rebuild champion-role embeddings and walk-forward snapshots",
        enabled=True,
        lock_key="embedding_refresh",
    ))

    registry.register(TaskDefinition(
        id="refresh_team_context_embeddings",
        name="Refresh Team Context Embeddings",
        func=ml.refresh_team_context_embeddings,
        description="Rebuild team/opponent context embeddings and walk-forward snapshots",
        enabled=True,
        lock_key="embedding_refresh",
    ))

    registry.register(TaskDefinition(
        id="embedding_refresh_cycle",
        name="Embedding Refresh Cycle",
        func=ml.run_embedding_refresh_cycle,
        cron_trigger="50 4 * * *",
        description="Refresh champion-role and then team-context embeddings",
        enabled=True,
        lock_key="embedding_refresh",
    ))

    # Scheduler healthcheck - frequent lightweight alerting via automation_runs
    registry.register(TaskDefinition(
        id="scheduler_healthcheck",
        name="Scheduler Healthcheck",
        func=ml.run_scheduler_healthcheck,
        cron_trigger="35 * * * *",  # Hourly at :35 UTC
        description="Check recent scheduler failures and stale bookmaker odds snapshots",
        enabled=True
    ))

    # Browser artifact cleanup - guardrail for interrupted nodriver/Chromium scrapes.
    registry.register(TaskDefinition(
        id="browser_artifact_cleanup",
        name="Browser Artifact Cleanup",
        func=scrape.cleanup_browser_artifacts,
        cron_trigger="25 * * * *",  # Hourly, after inference and before healthcheck
        description="Kill stale Chromium/nodriver processes and remove scraper temp dirs",
        enabled=True
    ))
    
    # Backfill after the midnight heavy cycle has refreshed GOL.GG.
    registry.register(TaskDefinition(
        id="backfill_expired_matches",
        name="Backfill Expired Matches to GOL.GG",
        func=maintenance.backfill_expired_matches,
        cron_trigger="40 2 * * *",
        description="Map expired canonical matches to existing GOL.GG results",
        enabled=True,
        lock_key="heavy_maintenance",
    ))
    
    # Expire old matches - every hour at :00, 5 min after scraping
    registry.register(TaskDefinition(
        id="expire_matches",
        name="Expire Old Matches",
        func=maintenance.expire_matches,
        cron_trigger="0 * * * *",  # At minute 0 every hour
        kwargs={"grace_hours": 3},
        description="Mark matches as expired when start_time has passed (3h grace)",
        enabled=True
    ))
    
    # Horizon bootstrap analysis - once per day
    registry.register(TaskDefinition(
        id="horizon_bootstrap",
        name="Horizon Bootstrap Analysis",
        func=maintenance.run_horizon_bootstrap,
        cron_trigger="15 0 * * *",  # Daily, after Model Analysis cache refresh
        description="Run monthly block bootstrap analysis comparing models vs bookmaker per horizon bin",
        enabled=True
    ))

    # Model Analysis cache - once per day at midnight
    registry.register(TaskDefinition(
        id="model_analysis_cache",
        name="Model Analysis Cache Refresh",
        func=maintenance.refresh_model_analysis_cache,
        cron_trigger="0 0 * * *",
        description="Precompute cached horizon accuracy and CLV payloads for the Model Analysis page",
        enabled=True
    ))

    # Expire stale-seen matches - every hour
    # Catches cancelled/postponed matches that disappeared from scrapers
    registry.register(TaskDefinition(
        id="expire_stale_matches",
        name="Expire Stale-Seen Matches",
        func=maintenance.expire_stale_matches,
        cron_trigger="30 * * * *",  # At minute 30, every hour
        kwargs={"stale_seen_hours": 6},
        description="Mark matches as expired when not seen by scrapers for 6h",
        enabled=True
    ))
    # One ticker collector; the following job owns roster verification.
    registry.register(TaskDefinition(
        id="sync_liquipedia_daily",
        name="Daily Liquipedia Sync",
        func=predict.sync_liquipedia_daily,
        cron_trigger="0 5 * * *",  # Daily 05:00 UTC
        description="Sync Best-of formats from Liquipedia; roster collection belongs to verify_team_rosters_daily",
        enabled=True
    ))
    # Team roster verification (LoL Fandom / Liquipedia) - runs daily at 05:15 UTC
    registry.register(TaskDefinition(
        id="verify_team_rosters_daily",
        name="Team Roster Verification (Fandom / Liquipedia)",
        func=maintenance.verify_team_rosters_task,
        cron_trigger="15 5 * * *",  # Daily 05:15 UTC
        description="Verify and update active 5-man team rosters from LoL Fandom (Leaguepedia) and Liquipedia",
        enabled=True,
    ))
    # OddsPapi fixture sync - runs every 3 days (1 request per run, ~10 requests/month)
    registry.register(TaskDefinition(
        id="oddspapi_fixture_sync",
        name="OddsPapi Fixture Sync",
        func=oddspapi.sync_fixtures_task,
        cron_trigger="0 3 */3 * *",  # Every 3 days at 03:00 UTC
        description="Sync upcoming LoL fixtures from OddsPapi to local mappings (1 request per run)",
        enabled=True,
    ))
    # OddsPapi horizon odds - checks every 30 minutes for matches in T-1h window
    registry.register(TaskDefinition(
        id="oddspapi_horizon_fetch",
        name="OddsPapi Horizon Fetch",
        func=oddspapi.fetch_horizon_odds_task,
        cron_trigger="5,35 * * * *",  # Twice an hour at :05 and :35 UTC
        kwargs={"target_horizon_hours": 1.0, "tolerance_hours": 0.5, "max_requests": 2},
        description="Fetch Pinnacle pre-match odds near T-1h (strictly capped by daily/monthly budget)",
        enabled=True,
    ))
    logger.info(f"Registered {len(registry.list_all())} tasks")
