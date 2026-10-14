"""ML/retraining scheduler tasks."""

import logging
from datetime import UTC, datetime

from sqlalchemy import text

from betting_app.core.db import get_session

from betting_app.core.models import get_thesis_model
from .scrape import _run_module

logger = logging.getLogger(__name__)


def run_weekly_retraining() -> dict:
    """Run weekly retraining for the EXP-039 model family.

    The frozen thesis version ``exp-039`` remains immutable.  This scheduled
    task trains a new ``exp039-weekly-*`` artifact with the same 46-feature,
    order-symmetric, Platt-calibrated methodology and registers it as a
    candidate for inspection/promotion.
    """
    logger.info("Starting weekly ML retraining")
    start = datetime.now(UTC)

    success = _run_module(
        "betting_app.ml.pipelines.exp039_weekly_retrain",
        args=[
            "--status-on-success",
            "candidate",
            "--json",
        ],
        timeout=3600,
    )

    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Weekly ML retraining: {'OK' if success else 'FAIL'} ({duration:.1f}s)")

    return {
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }


def run_shadow_inference() -> dict:
    """Run inference for registered shadow ML models.

    This does not promote or replace the current production/thesis model. It
    only writes active `canonical_predictions` rows for models registered with
    `status='shadow'`, so they become visible in match detail/history views.
    """
    logger.info("Starting shadow ML inference")
    start = datetime.now(UTC)

    success = _run_module(
        "betting_app.ml.inference.cli",
        args=["--status", "shadow", "--json"],
        timeout=600,
    )

    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Shadow ML inference: {'OK' if success else 'FAIL'} ({duration:.1f}s)")

    return {
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }


def run_thesis_model_healthcheck() -> dict:
    """Run a read-only rolling evaluation of the current thesis model.

    The healthcheck intentionally uses ``--no-register``: it should verify
    recent model-vs-market quality and emit logs, not create/promote model
    versions.  This gives the scheduler a lightweight recurring guardrail for
    the production model without affecting inference state.
    """
    logger.info("Starting thesis model healthcheck")
    start = datetime.now(UTC)

    success = _run_module(
        "betting_app.ml.pipelines.evaluate_existing_model",
        args=[
            "--model-name",
            get_thesis_model().name,
            "--model-version",
            get_thesis_model().version,
            "--days-back",
            "90",
            "--no-register",
            "--json",
        ],
        timeout=900,
    )

    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Thesis model healthcheck: {'OK' if success else 'FAIL'} ({duration:.1f}s)")

    return {
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }


def refresh_champion_role_embeddings() -> dict:
    """Rebuild current and walk-forward champion-role embedding snapshots.

    The dashboard visualisation should reflect how champions are played as of a
    selected date, not after seeing future games.  This scheduled refresh writes
    a latest artifact plus monthly leakage-safe snapshots from 2026 onward;
    each snapshot is computed using only games before its reference date.
    """
    logger.info("Starting champion-role embedding refresh")
    start = datetime.now(UTC)

    success = _run_module(
        "betting_app.scripts.build_champion_role_embeddings",
        args=[
            "--min-date",
            "2020-01-01",
            "--walk-forward",
            "--snapshot-start",
            "2026-01-01",
            "--snapshot-frequency",
            "MS",
            "--output-dir",
            "/app/betting_app/models/ml/champion_role_embeddings/exp-056",
        ],
        timeout=1800,
    )

    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Champion-role embedding refresh: {'OK' if success else 'FAIL'} ({duration:.1f}s)")

    return {
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }


def refresh_team_context_embeddings() -> dict:
    """Rebuild current and walk-forward team/opponent context embeddings.

    These EXP-057 artifacts are leakage-safe team form/style snapshots used as
    future ``own_team_embedding`` and ``opponent_team_embedding`` context.  They
    intentionally refresh after the GOL.GG import/rating maintenance cycle and
    after champion-role embeddings.
    """
    logger.info("Starting team-context embedding refresh")
    start = datetime.now(UTC)

    success = _run_module(
        "betting_app.scripts.build_team_context_embeddings",
        args=[
            "--min-date",
            "2020-01-01",
            "--walk-forward",
            "--snapshot-start",
            "2026-01-01",
            "--snapshot-frequency",
            "MS",
            "--output-dir",
            "/app/betting_app/models/ml/team_context_embeddings/exp-057",
        ],
        timeout=1800,
    )

    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Team-context embedding refresh: {'OK' if success else 'FAIL'} ({duration:.1f}s)")

    return {
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }


def run_embedding_refresh_cycle() -> dict:
    """Refresh both embedding families sequentially to avoid GPU contention."""
    logger.info("Starting embedding refresh cycle")
    start = datetime.now(UTC)

    champion_result = refresh_champion_role_embeddings()
    team_result = refresh_team_context_embeddings()
    results = {
        "champion_role": champion_result,
        "team_context": team_result,
    }
    duration = (datetime.now(UTC) - start).total_seconds()
    all_ok = all(result.get("success", False) for result in results.values())

    logger.info(f"Embedding refresh cycle: {'OK' if all_ok else 'PARTIAL'} ({duration:.1f}s)")
    return {
        "success": all_ok,
        "results": results,
        "duration_s": duration,
    }


def run_scheduler_healthcheck() -> dict:
    """Check recent scheduler/scraper health and fail loudly on regressions.

    This is a lightweight alerting primitive: APScheduler records the run in
    ``automation_runs``.  If any critical task failed recently or a bookmaker
    has stale odds snapshots, this task returns ``success=False`` so the failure
    becomes visible in logs and the system status dashboard.
    """
    logger.info("Starting scheduler healthcheck")
    start = datetime.now(UTC)
    failures: list[str] = []

    with get_session() as session:
        failed_runs = session.execute(
            text(
                """
                SELECT f.run_type, f.status, f.started_at, f.error
                FROM automation_runs f
                WHERE f.started_at >= NOW() - INTERVAL '6 hours'
                  AND f.status = 'failed'
                  AND f.run_type IN (
                      'scrape_sts',
                      'scrape_betclic',
                      'scrape_superbet',
                      'scrape_efortuna',
                      'scrape_betfan',
                      'scrape_totalbet',
                      'scrape_lebull',
                      'prediction_pipeline',
                      'shadow_ml_inference',
                      'refresh_golgg',
                      'rebuild_ratings',
                      'rebuild_features',
                      'heavy_maintenance_cycle',
                      'embedding_refresh_cycle',
                      'refresh_champion_role_embeddings',
                      'refresh_team_context_embeddings',
                      'weekly_ml_retraining',
                      'thesis_model_healthcheck',
                      'model_analysis_cache',
                      'horizon_bootstrap',
                      'backfill_expired_matches',
                      'expire_matches',
                      'expire_stale_matches',
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM automation_runs s
                      WHERE s.run_type = f.run_type
                        AND s.status = 'completed'
                        AND s.started_at > f.started_at
                  )
                ORDER BY f.started_at DESC
                LIMIT 10
                """
            ),
        ).mappings().all()
        for row in failed_runs:
            failures.append(
                f"recent failed run: {row['run_type']} at {row['started_at']} ({row.get('error') or 'no error'})"
            )

        stale_scrapes = session.execute(
            text(
                """
                SELECT b.name, MAX(os.scraped_at) AS last_scraped_at
                FROM bookmakers b
                LEFT JOIN odds_snapshots os ON os.bookmaker_id = b.id
                WHERE b.is_active = 1
                  AND b.name IN ('sts', 'betclic', 'superbet', 'efortuna', 'betfan', 'totalbet', 'lebull')
                GROUP BY b.name
                HAVING MAX(os.scraped_at) IS NULL
                    OR MAX(os.scraped_at) < NOW() - INTERVAL '8 hours'
                ORDER BY last_scraped_at NULLS FIRST
                """
            )
        ).mappings().all()
        for row in stale_scrapes:
            failures.append(f"stale bookmaker snapshots: {row['name']} last_scraped_at={row['last_scraped_at']}")

    duration = (datetime.now(UTC) - start).total_seconds()
    success = not failures
    if success:
        logger.info(f"Scheduler healthcheck: OK ({duration:.1f}s)")
    else:
        logger.error("Scheduler healthcheck: FAIL (%s) failures=%s", f"{duration:.1f}s", failures)

    return {
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
        "failures": failures,
    }
