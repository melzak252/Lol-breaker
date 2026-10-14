"""Router: /api/scheduler — task listing, status, and manual triggers."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from betting_app.api.deps import get_db, query_df
from betting_app.api.schemas import (
    SchedulerJobResponse,
    SchedulerTaskResponse,
    SchedulerTriggerResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduler", tags=["scheduler"])

# Background executor for manual task triggers
_executor = ThreadPoolExecutor(max_workers=2)
_running_tasks: dict[str, datetime] = {}
_lock = threading.Lock()


def _parse_datetime(value: object) -> datetime | None:
    """Parse database timestamp values as timezone-aware UTC datetimes."""

    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _duration_seconds(started_at: object, finished_at: object) -> float | None:
    """Calculate a run duration without dialect-specific SQL."""

    started = _parse_datetime(started_at)
    finished = _parse_datetime(finished_at)
    if started is None or finished is None:
        return None
    return (finished - started).total_seconds()


def _recent_running_task_ids(rows: list[dict], *, max_age_hours: float = 2.0) -> set[str]:
    """Return non-stale task IDs from automation rows marked running."""

    now = datetime.now(timezone.utc)
    running: set[str] = set()
    for row in rows:
        if row.get("status") != "running":
            continue
        started = _parse_datetime(row.get("started_at"))
        if started is not None and (now - started).total_seconds() <= max_age_hours * 3600:
            running.add(str(row.get("run_type")))
    return running


def _get_registry():
    """Lazy import to avoid circular imports and heavy loading at startup."""
    from betting_app.scheduler.registry import registry, register_all_tasks

    # Ensure tasks are registered (idempotent)
    if not registry.list_all():
        register_all_tasks()
    return registry


@router.get("/tasks", response_model=list[SchedulerTaskResponse])
def list_tasks():
    """List all registered task definitions."""
    reg = _get_registry()
    tasks = []
    for t in reg.list_all():
        # Derive human-readable schedule string
        if t.cron_trigger:
            schedule = t.cron_trigger
        elif t.interval_minutes:
            schedule = f"every {t.interval_minutes} min"
        else:
            schedule = "manual"

        tasks.append(
            SchedulerTaskResponse(
                id=t.id,
                task_id=t.id,
                name=t.name,
                description=t.description,
                schedule=schedule,
                interval_minutes=t.interval_minutes,
                cron_trigger=t.cron_trigger,
                enabled=t.enabled,
            )
        )
    return tasks


@router.get("/jobs", response_model=list[SchedulerJobResponse])
def list_jobs(db=Depends(get_db)):
    """List scheduled jobs from APScheduler job store + recent automation runs."""
    # Read from apscheduler_jobs table (APScheduler serializes job state here)
    jobs_raw = query_df(
        db,
        """
        SELECT id, next_run_time
        FROM apscheduler_jobs
        ORDER BY next_run_time
        """,
    )

    # Build a map of job_id -> next_run_time
    job_map = {}
    for row in jobs_raw:
        jid = row["id"]
        nrt = row.get("next_run_time")
        # APScheduler stores next_run_time as epoch float in some backends
        if isinstance(nrt, (int, float)):
            nrt = datetime.fromtimestamp(nrt, tz=timezone.utc).isoformat()
        elif isinstance(nrt, datetime):
            nrt = nrt.isoformat()
        job_map[jid] = nrt

    # Keep the latest row for every task, not an arbitrary global top-N slice.
    recent_runs = query_df(
        db,
        """
        SELECT run_type, status, started_at, finished_at, error
        FROM (
            SELECT run_type, status, started_at, finished_at, error, id,
                   ROW_NUMBER() OVER (
                       PARTITION BY run_type
                       ORDER BY started_at DESC, id DESC
                   ) AS row_num
            FROM automation_runs
        ) ranked
        WHERE row_num = 1
        """,
    )

    # Build last_run map per task type
    last_run_map: dict[str, dict] = {}
    for run in recent_runs:
        rt = run.get("run_type", "")
        if rt not in last_run_map:
            last_run_map[rt] = run

    # Merge registry tasks with job store data. Tasks sharing a lock are shown
    # as running together so manual components cannot appear safe to overlap.
    reg = _get_registry()
    running_task_ids = _recent_running_task_ids(recent_runs)
    with _lock:
        in_process_task_ids = set(_running_tasks)
    active_task_ids = running_task_ids | in_process_task_ids
    active_lock_keys = {
        (registered.lock_key or registered.id)
        for task_id in active_task_ids
        if (registered := reg.get(task_id)) is not None
    }

    results = []
    for t in reg.list_all():
        next_run = job_map.get(t.id)
        last_run_info = last_run_map.get(t.id)

        last_run_at = None
        last_run_status = None
        if last_run_info:
            la = last_run_info.get("finished_at") or last_run_info.get("started_at")
            if isinstance(la, datetime):
                last_run_at = la.isoformat()
            elif la:
                last_run_at = str(la)
            last_run_status = last_run_info.get("status")

        lock_key = t.lock_key or t.id
        is_running = t.id in active_task_ids or lock_key in active_lock_keys
        # Derive trigger string from cron/interval
        if t.cron_trigger:
            trigger = t.cron_trigger
        elif t.interval_minutes:
            trigger = f"interval:{t.interval_minutes}m"
        else:
            trigger = "manual"

        # Pending = has a next_run_time scheduled but not currently running
        pending = next_run is not None and not is_running

        results.append(
            SchedulerJobResponse(
                id=t.id,
                name=t.name,
                enabled=t.enabled,
                next_run_time=next_run,
                last_run_at=last_run_at,
                last_run_status=last_run_status,
                is_running=is_running,
                trigger=trigger,
                pending=pending,
            )
        )

    return results


@router.post("/trigger/{task_id}", response_model=SchedulerTriggerResponse)
def trigger_task(task_id: str, db=Depends(get_db)):
    """Manually trigger a task unless its cross-process lock is occupied."""
    reg = _get_registry()
    task = reg.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    running_rows = query_df(
        db,
        """
        SELECT run_type, status, started_at
        FROM automation_runs
        WHERE status = 'running'
        """,
    )
    running_task_ids = _recent_running_task_ids(running_rows)
    requested_lock_key = task.lock_key or task.id
    with _lock:
        active_task_ids = running_task_ids | set(_running_tasks)
        conflicting_task = next(
            (
                active_id
                for active_id in sorted(active_task_ids)
                if (active_task := reg.get(active_id)) is not None
                and (active_task.lock_key or active_task.id) == requested_lock_key
            ),
            None,
        )
        if conflicting_task is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Task lock '{requested_lock_key}' is held by '{conflicting_task}'",
            )
        _running_tasks[task_id] = datetime.now(timezone.utc)

    def _run():
        try:
            logger.info(f"Manual trigger: {task_id}")
            from betting_app.scheduler.app import execute_task

            result = execute_task(
                task_id,
                *task.args,
                _trigger_source="manual",
                **task.kwargs,
            )
            if isinstance(result, dict) and result.get("skipped"):
                logger.warning(f"Manual trigger skipped: {task_id}: {result.get('reason')}")
            else:
                logger.info(f"Manual trigger completed: {task_id}")
        except Exception as exc:
            logger.error(f"Manual trigger failed: {task_id}: {exc}")
        finally:
            with _lock:
                _running_tasks.pop(task_id, None)

    _executor.submit(_run)

    return SchedulerTriggerResponse(
        task_id=task_id,
        status="started",
        message=f"Task '{task_id}' triggered in background",
    )


@router.get("/runs", response_model=list[dict])
def list_recent_runs(limit: int = 20, db=Depends(get_db)):
    """List recent automation runs with cross-dialect durations."""
    rows = query_df(
        db,
        """
        SELECT id, run_type, run_type AS task_name, status, started_at,
               finished_at, error
        FROM automation_runs
        ORDER BY started_at DESC, id DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    for row in rows:
        row["duration_seconds"] = _duration_seconds(
            row.get("started_at"),
            row.get("finished_at"),
        )
    return rows


@router.get("/runs/{run_id}/commands", response_model=list[dict])
def list_run_commands(run_id: int, db=Depends(get_db)):
    """List tracked subprocesses for one automation run."""
    rows = query_df(
        db,
        """
        SELECT id, command, status, started_at, finished_at, exit_code,
               output, error,
               ROW_NUMBER() OVER (ORDER BY started_at, id) AS step_order
        FROM automation_commands
        WHERE run_id = :run_id
        ORDER BY started_at, id
        """,
        {"run_id": run_id},
    )
    for row in rows:
        row["duration_seconds"] = _duration_seconds(
            row.get("started_at"),
            row.get("finished_at"),
        )
        row["output"] = row.get("error") or row.get("output")
    return rows
