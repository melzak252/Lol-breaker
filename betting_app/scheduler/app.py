"""Main scheduler application using APScheduler.

Run with: python -m betting_app.scheduler.app
"""

import json
import logging
import signal
import sys
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from sqlalchemy import create_engine, inspect, text

from betting_app.core.db import database_url, get_session, is_pg
from betting_app.services.automation_service import (
    automation_run_context,
    cleanup_stale_runs,
    finish_run,
    start_run,
)

from .registry import registry, register_all_tasks

logger = logging.getLogger(__name__)

_ADVISORY_LOCK_NAMESPACE = 1162627396
_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.Lock] = {}


@contextmanager
def _task_execution_lock(lock_key: str) -> Iterator[bool]:
    """Hold a non-blocking task lock across scheduler and API processes."""

    if is_pg():
        session = get_session()
        params = {"namespace": _ADVISORY_LOCK_NAMESPACE, "lock_key": lock_key}
        try:
            # PostgreSQL advisory locks belong to a physical connection. Keep
            # this autocommit connection checked out until the task finishes.
            connection = session.connection(
                execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            acquired = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:namespace, hashtext(:lock_key))"),
                    params,
                ).scalar()
            )
            try:
                yield acquired
            finally:
                if acquired:
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:namespace, hashtext(:lock_key))"),
                        params,
                    )
        finally:
            session.close()
        return

    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.setdefault(lock_key, threading.Lock())
    acquired = lock.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


def _failure_error(result: dict) -> str:
    """Return a useful bounded error for a failed task result."""

    explicit = result.get("error")
    if explicit:
        return str(explicit)[:8000]
    return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)[:8000]


def reset_persisted_jobs() -> int:
    """Remove obsolete persisted jobs before registering the current schedule."""

    engine = create_engine(database_url())
    try:
        if not inspect(engine).has_table("apscheduler_jobs"):
            return 0
        with engine.begin() as connection:
            result = connection.execute(text("DELETE FROM apscheduler_jobs"))
            return max(int(result.rowcount or 0), 0)
    finally:
        engine.dispose()



def execute_task(
    task_id: str,
    *args,
    _trigger_source: str = "apscheduler",
    **kwargs,
):
    """Execute one registered task with locking and run tracking."""
    from .registry import registry

    task = registry.get(task_id)
    if not task:
        logger.error(f"Task {task_id} not found in registry")
        return {"success": False, "error": f"Task {task_id} not found in registry"}

    lock_key = task.lock_key or task.id
    with _task_execution_lock(lock_key) as acquired:
        if not acquired:
            reason = f"Task lock already held: {lock_key}"
            logger.warning("[%s] Skipped because lock %s is already held", task_id, lock_key)
            skipped_run_id = start_run(
                run_type=task_id,
                trigger_source=_trigger_source,
            )
            finish_run(skipped_run_id, status="skipped", error=reason)
            return {
                "success": True,
                "skipped": True,
                "reason": reason,
            }

        run_id = None
        try:
            run_id = start_run(
                run_type=task_id,
                trigger_source=_trigger_source,
            )
            logger.info(f"[{task_id}] Started automation run #{run_id}")

            with automation_run_context(run_id):
                result = task.func(*args, **kwargs)

            if isinstance(result, dict):
                success = bool(result.get("success", True))
                status = "completed" if success else "failed"
                error = _failure_error(result) if not success else None
            elif result is False:
                status = "failed"
                error = "Task returned False"
            else:
                status = "completed"
                error = None

            finish_run(run_id, status=status, error=error)
            logger.info(f"[{task_id}] Run #{run_id} finished: {status}")
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            logger.error(f"[{task_id}] Run #{run_id} failed: {error_msg}\n{tb}")
            if run_id:
                try:
                    finish_run(run_id, status="failed", error=error_msg)
                except Exception:
                    logger.error(f"[{task_id}] Failed to mark run #{run_id} as failed")
            raise


def create_scheduler() -> BlockingScheduler:
    """Create and configure the APScheduler instance."""
    
    # Job store - persist jobs in PostgreSQL (or SQLite fallback)
    jobstores = {
        "default": SQLAlchemyJobStore(url=database_url()),
    }
    
    # Executors
    executors = {
        "default": ThreadPoolExecutor(max_workers=3),
    }
    
    # Job defaults
    job_defaults = {
        "coalesce": True,       # Combine missed runs into one
        "max_instances": 1,     # Only one instance of each job at a time
        "misfire_grace_time": 300,  # 5 min grace period for missed jobs
    }
    
    scheduler = BlockingScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone="UTC",
    )
    
    return scheduler


def schedule_tasks(scheduler: BlockingScheduler):
    """Add all registered tasks to the scheduler.
    
    Each task is scheduled to call execute_task(task_id, ...).
    """
    
    for task in registry.list_enabled():
        if task.interval_minutes:
            scheduler.add_job(
                execute_task,
                trigger="interval",
                minutes=task.interval_minutes,
                id=task.id,
                name=task.name,
                args=(task.id,) + task.args,
                kwargs=task.kwargs,
                replace_existing=True,
            )
            logger.info(
                f"Scheduled: {task.id} every {task.interval_minutes}min"
            )
        elif task.cron_trigger:
            # Parse cron expression (minute hour day month day_of_week)
            parts = task.cron_trigger.split()
            if len(parts) == 5:
                scheduler.add_job(
                    execute_task,
                    trigger="cron",
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4],
                    id=task.id,
                    name=task.name,
                    args=(task.id,) + task.args,
                    kwargs=task.kwargs,
                    replace_existing=True,
                )
                logger.info(
                    f"Scheduled: {task.id} cron={task.cron_trigger}"
                )
            else:
                logger.error(f"Invalid cron for {task.id}: {task.cron_trigger}")
        else:
            logger.info(f"Manual-only task: {task.id}")


def main():
    """Main entry point for the scheduler."""
    
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("Betting App Scheduler starting")
    logger.info("Database backend: %s", "PostgreSQL" if is_pg() else "SQLite")
    logger.info("=" * 60)
    
    # Register all tasks
    register_all_tasks()
    
    # Clean up any stale 'running' runs from a previous crashed scheduler
    stale_count = cleanup_stale_runs(max_age_hours=2)
    if stale_count:
        logger.warning(f"Cleaned up {stale_count} stale 'running' automation run(s)")
    
    # Discard obsolete persisted jobs before adding the current registry. This
    # is required when a formerly independent job becomes manual-only.
    removed_jobs = reset_persisted_jobs()
    if removed_jobs:
        logger.info(f"Removed {removed_jobs} persisted scheduler job(s)")

    scheduler = create_scheduler()
    schedule_tasks(scheduler)
    
    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        scheduler.shutdown(wait=False)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Print scheduled jobs
    jobs = scheduler.get_jobs()
    logger.info(f"Total scheduled jobs: {len(jobs)}")
    for job in jobs:
        logger.info(f"  - {job.id}: trigger={job.trigger}")
    
    # Start
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
