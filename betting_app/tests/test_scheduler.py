from __future__ import annotations

import itertools
import threading
from datetime import UTC, datetime

import pytest
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import create_engine, text

from betting_app.core.db import get_session
from betting_app.scheduler import app as scheduler_app
from betting_app.scheduler.registry import TaskDefinition, register_all_tasks, registry
from betting_app.scheduler.tasks import maintenance, ml, predict, scrape
from betting_app.services.automation_service import (
    automation_run_context,
    current_automation_run_id,
)
from betting_app.scripts import scheduler as legacy_scheduler


@pytest.fixture(autouse=True)
def preserve_registry():
    original = dict(registry._tasks)
    registry._tasks.clear()
    try:
        yield
    finally:
        registry._tasks.clear()
        registry._tasks.update(original)


def _next_fire(task_id: str, now: datetime) -> datetime:
    task = registry.get(task_id)
    assert task is not None
    assert task.cron_trigger is not None
    trigger = CronTrigger.from_crontab(task.cron_trigger, timezone=UTC)
    next_fire = trigger.get_next_fire_time(None, now)
    assert next_fire is not None
    return next_fire





def test_prediction_consumers_follow_scrape_and_prediction_origins() -> None:
    register_all_tasks()
    now = datetime(2026, 9, 2, 9, 12, tzinfo=UTC)
    scrape_at = _next_fire("scrape_sts", now)
    prediction_at = _next_fire("prediction_pipeline", now)
    assert now < scrape_at < prediction_at < _next_fire("value_alerts_dispatch", now)
    assert prediction_at < _next_fire("shadow_ml_inference", now)


def test_schedule_tasks_excludes_manual_components() -> None:
    register_all_tasks()
    scheduler = BlockingScheduler(timezone="UTC")

    scheduler_app.schedule_tasks(scheduler)

    scheduled_ids = {job.id for job in scheduler.get_jobs()}
    assert "heavy_maintenance_cycle" in scheduled_ids
    assert "embedding_refresh_cycle" in scheduled_ids
    assert "refresh_golgg" not in scheduled_ids
    assert "rebuild_ratings" not in scheduled_ids
    assert "rebuild_features" not in scheduled_ids
    for task_id in (
        "refresh_golgg",
        "rebuild_ratings",
        "rebuild_features",
        "refresh_champion_role_embeddings",
        "refresh_team_context_embeddings",
    ):
        task = registry.get(task_id)
        assert task is not None
        assert task.cron_trigger is None
        assert task.interval_minutes is None


def test_heavy_cycle_runs_dependencies_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def successful(step: str):
        def run() -> dict:
            calls.append(step)
            return {"success": True}

        return run

    monkeypatch.setattr(maintenance, "refresh_golgg", successful("golgg"))
    monkeypatch.setattr(maintenance, "rebuild_ratings", successful("ratings"))
    monkeypatch.setattr(maintenance, "rebuild_rolling_features", successful("features"))

    result = maintenance.run_heavy_cycle()

    assert result["success"] is True
    assert calls == ["golgg", "ratings", "features"]


def test_heavy_cycle_skips_dependents_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_golgg() -> dict:
        calls.append("golgg")
        return {"success": False, "error": "refresh failed"}

    def unexpected() -> dict:
        raise AssertionError("dependent step should not run")

    monkeypatch.setattr(maintenance, "refresh_golgg", fail_golgg)
    monkeypatch.setattr(maintenance, "rebuild_ratings", unexpected)
    monkeypatch.setattr(maintenance, "rebuild_rolling_features", unexpected)

    result = maintenance.run_heavy_cycle()

    assert result["success"] is False
    assert result["error"] == "Heavy maintenance step failed: golgg"
    assert calls == ["golgg"]
    assert result["results"]["ratings"]["skipped"] is True
    assert result["results"]["features"]["skipped"] is True


def test_heavy_cycle_skips_features_after_ratings_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def succeed_golgg() -> dict:
        calls.append("golgg")
        return {"success": True}

    def fail_ratings() -> dict:
        calls.append("ratings")
        return {"success": False, "error": "ratings failed"}

    def unexpected_features() -> dict:
        raise AssertionError("features must not run after a ratings failure")

    monkeypatch.setattr(maintenance, "refresh_golgg", succeed_golgg)
    monkeypatch.setattr(maintenance, "rebuild_ratings", fail_ratings)
    monkeypatch.setattr(
        maintenance,
        "rebuild_rolling_features",
        unexpected_features,
    )

    result = maintenance.run_heavy_cycle()

    assert result["success"] is False
    assert result["error"] == "Heavy maintenance step failed: ratings"
    assert calls == ["golgg", "ratings"]
    assert result["results"]["ratings"]["error"] == "ratings failed"
    assert result["results"]["features"] == {
        "success": False,
        "skipped": True,
        "reason": "dependency step ratings failed",
    }


def test_embedding_cycle_serializes_both_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def champion() -> dict:
        calls.append("champion")
        return {"success": True}

    def team() -> dict:
        calls.append("team")
        return {"success": True}

    monkeypatch.setattr(ml, "refresh_champion_role_embeddings", champion)
    monkeypatch.setattr(ml, "refresh_team_context_embeddings", team)

    result = ml.run_embedding_refresh_cycle()

    assert result["success"] is True
    assert calls == ["champion", "team"]


def test_prediction_pipeline_uses_fast_rematch_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str] | None, int]] = []

    def run_module(module: str, args: list[str] | None = None, timeout: int = 300) -> bool:
        calls.append((module, args, timeout))
        return True

    monkeypatch.setattr(predict, "_run_module", run_module)

    result = predict.run_prediction_pipeline()

    assert result["success"] is True
    assert calls == [
        (
            "betting_app.scripts.rematch_canonical_matches",
            ["--no-overview"],
            300,
        ),
        (
            "betting_app.scripts.run_upcoming_prediction_pipeline",
            ["--include-partial", "--operational-hybrid"],
            900,
        ),
    ]


def test_prediction_pipeline_stops_when_rematch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict, "rematch_canonical", lambda: False)

    def unexpected(*args, **kwargs):
        raise AssertionError("prediction must not run after a rematch failure")

    monkeypatch.setattr(predict, "_run_module", unexpected)

    result = predict.run_prediction_pipeline()

    assert result["success"] is False
    assert result["steps"] == {"rematch": False, "predict": False}
    assert result["error"] == "Canonical rematching failed"



def test_rebuild_ratings_invokes_full_regional_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str] | None, int]] = []

    def run_module(module: str, args: list[str] | None = None, timeout: int = 300) -> bool:
        calls.append((module, args, timeout))
        return True

    monkeypatch.setattr(maintenance, "_run_module", run_module)

    result = maintenance.rebuild_ratings()

    assert result["success"] is True
    assert calls == [
        (
            "betting_app.scripts.rebuild_regional_ratings",
            [
                "--ratings-version",
                "ratings-v2",
                "--source",
                "scheduler-regional-ratings-v2",
            ],
            3600,
        )
    ]


def test_legacy_scheduler_uses_regional_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    finished: list[tuple[int, str]] = []

    monkeypatch.setattr(
        legacy_scheduler,
        "start_run",
        lambda *args, **kwargs: 91,
    )
    monkeypatch.setattr(
        legacy_scheduler,
        "finish_run",
        lambda run_id, *, status, **kwargs: finished.append((run_id, status)),
    )
    monkeypatch.setattr(
        legacy_scheduler,
        "run_module",
        lambda module, *args, **kwargs: calls.append((module, args)) or 0,
    )

    legacy_scheduler.run_light_cycle(0.05, ["sts"])
    legacy_scheduler.run_heavy_cycle(0.05, ["sts"])

    assert (
        "betting_app.scripts.run_upcoming_prediction_pipeline",
        ("--operational-hybrid", "--min-ev", "0.05"),
    ) in calls
    assert (
        "betting_app.scripts.rebuild_regional_ratings",
        (
            "--ratings-version",
            "ratings-v2",
            "--source",
            "legacy-scheduler-regional-ratings-v2",
        ),
    ) in calls
    assert finished == [(91, "completed"), (91, "completed")]

def test_execute_task_records_trigger_context_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def task_func() -> dict:
        observed["context_run_id"] = current_automation_run_id()
        return {"success": False, "steps": {"rematch": False}}

    registry.register(TaskDefinition(id="failing_task", name="Failing", func=task_func))
    monkeypatch.setattr(scheduler_app, "is_pg", lambda: False)

    def start_run(*, run_type: str, trigger_source: str) -> int:
        observed["run_type"] = run_type
        observed["trigger_source"] = trigger_source
        return 41

    def finish_run(run_id: int, *, status: str, error: str | None = None) -> None:
        observed["finish"] = (run_id, status, error)

    monkeypatch.setattr(scheduler_app, "start_run", start_run)
    monkeypatch.setattr(scheduler_app, "finish_run", finish_run)

    result = scheduler_app.execute_task("failing_task", _trigger_source="manual")

    assert result["success"] is False
    assert observed["run_type"] == "failing_task"
    assert observed["trigger_source"] == "manual"
    assert observed["context_run_id"] == 41
    run_id, status, error = observed["finish"]
    assert run_id == 41
    assert status == "failed"
    assert '"rematch": false' in error


def test_execute_task_prevents_shared_lock_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    first_result: dict[str, object] = {}
    second_called = False
    run_ids = itertools.count(1)
    finished: list[tuple[int, str, str | None]] = []

    def first_task() -> dict:
        entered.set()
        assert release.wait(timeout=2)
        return {"success": True}

    def second_task() -> dict:
        nonlocal second_called
        second_called = True
        return {"success": True}

    registry.register(
        TaskDefinition(id="first", name="First", func=first_task, lock_key="shared")
    )
    registry.register(
        TaskDefinition(id="second", name="Second", func=second_task, lock_key="shared")
    )
    monkeypatch.setattr(scheduler_app, "is_pg", lambda: False)
    monkeypatch.setattr(
        scheduler_app,
        "start_run",
        lambda *, run_type, trigger_source: next(run_ids),
    )
    monkeypatch.setattr(
        scheduler_app,
        "finish_run",
        lambda run_id, *, status, error=None: finished.append((run_id, status, error)),
    )

    thread = threading.Thread(
        target=lambda: first_result.update(result=scheduler_app.execute_task("first")),
        daemon=True,
    )
    thread.start()
    assert entered.wait(timeout=2)
    try:
        second_result = scheduler_app.execute_task("second", _trigger_source="manual")
    finally:
        release.set()
        thread.join(timeout=2)

    assert thread.is_alive() is False
    assert first_result["result"] == {"success": True}
    assert second_result["success"] is True
    assert second_result["skipped"] is True
    assert second_called is False
    assert any(status == "skipped" and error == "Task lock already held: shared" for _, status, error in finished)


def test_run_module_records_timeout_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[int | None, list[str]]] = []
    finished: list[dict] = []
    monkeypatch.setattr(
        scrape,
        "start_command",
        lambda run_id, command: started.append((run_id, command)) or 7,
    )
    monkeypatch.setattr(
        scrape,
        "finish_command",
        lambda command_id, **kwargs: finished.append({"command_id": command_id, **kwargs}),
    )
    monkeypatch.setattr(
        scrape,
        "cleanup_browser_leftovers",
        lambda **kwargs: {"processes_killed": 0, "temp_dirs_removed": 0},
    )

    with automation_run_context(99):
        success = scrape._run_module(
            "timeit",
            args=["-n", "1", "import time; time.sleep(2)"],
            timeout=0.1,
        )

    assert success is False
    assert started[0][0] == 99
    assert finished == [
        {
            "command_id": 7,
            "returncode": 124,
            "output": "",
            "error": "Timed out after 0.1s",
        }
    ]


def test_scheduler_api_rejects_conflicting_manual_task(client) -> None:
    register_all_tasks()
    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO automation_runs(run_type, trigger_source, status, started_at)
                VALUES ('heavy_maintenance_cycle', 'apscheduler', 'running', :started_at)
                """
            ),
            {"started_at": datetime.now(UTC).isoformat()},
        )
        session.commit()

    response = client.post("/scheduler/trigger/rebuild_ratings")

    assert response.status_code == 409
    assert "heavy_maintenance" in response.json()["detail"]


def test_scheduler_command_api_returns_output_and_duration(client) -> None:
    with get_session() as session:
        session.execute(
            text(
                """
                INSERT INTO automation_runs(id, run_type, trigger_source, status, started_at, finished_at)
                VALUES (501, 'test_task', 'test', 'completed',
                        '2026-09-02T10:00:00+00:00', '2026-09-02T10:00:03+00:00')
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO automation_commands(
                    id, run_id, command, status, started_at, finished_at,
                    exit_code, output, error
                ) VALUES (
                    601, 501, 'python -m example', 'completed',
                    '2026-09-02T10:00:00+00:00', '2026-09-02T10:00:02.500000+00:00',
                    0, 'done', NULL
                )
                """
            )
        )
        session.commit()

    response = client.get("/scheduler/runs/501/commands")

    assert response.status_code == 200
    assert response.json()[0]["output"] == "done"
    assert response.json()[0]["duration_seconds"] == 2.5


def test_reset_persisted_jobs_removes_obsolete_schedule(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'jobs.sqlite3'}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE apscheduler_jobs (
                    id VARCHAR(191) PRIMARY KEY,
                    next_run_time FLOAT,
                    job_state BLOB NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO apscheduler_jobs(id, next_run_time, job_state)
                VALUES ('obsolete', 1.0, :job_state)
                """
            ),
            {"job_state": b"old"},
        )
    engine.dispose()
    monkeypatch.setattr(scheduler_app, "database_url", lambda: database_url)

    assert scheduler_app.reset_persisted_jobs() == 1

    engine = create_engine(database_url)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM apscheduler_jobs")).scalar_one() == 0
    engine.dispose()


def test_scheduler_jobs_keep_latest_status_for_every_registered_task(client) -> None:
    register_all_tasks()
    with get_session() as session:
        is_pg = session.bind.dialect.name == "postgresql" if session.bind else False
        blob_type = "BYTEA" if is_pg else "BLOB"
        session.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS apscheduler_jobs (
                    id VARCHAR(191) PRIMARY KEY,
                    next_run_time FLOAT,
                    job_state {blob_type} NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO automation_runs(
                    id, run_type, trigger_source, status, started_at, finished_at
                ) VALUES (
                    1, 'refresh_golgg', 'test', 'completed',
                    '2026-09-02T00:00:00+00:00', '2026-09-02T00:01:00+00:00'
                )
                """
            )
        )
        session.execute(
            text(
                """
                INSERT INTO automation_runs(
                    id, run_type, trigger_source, status, started_at, finished_at
                ) VALUES (
                    :id, :run_type, 'test', 'completed',
                    '2026-09-02T01:00:00+00:00', '2026-09-02T01:00:01+00:00'
                )
                """
            ),
            [
                {"id": index + 100, "run_type": f"noise_{index}"}
                for index in range(60)
            ],
        )
        session.commit()

    response = client.get("/scheduler/jobs")

    assert response.status_code == 200
    refresh_job = next(job for job in response.json() if job["id"] == "refresh_golgg")
    assert refresh_job["last_run_status"] == "completed"
    assert refresh_job["last_run_at"] == "2026-09-02T00:01:00+00:00"


def test_bootstrap_reader_and_writer_use_same_shared_directory() -> None:
    from betting_app.api.routers import bootstrap as bootstrap_api
    from betting_app.scripts import horizon_block_bootstrap

    assert bootstrap_api.BOOTSTRAP_DIR == horizon_block_bootstrap.OUTPUT_DIR
    assert bootstrap_api.BOOTSTRAP_DIR.name == "horizon_block_bootstrap"
    assert bootstrap_api.BOOTSTRAP_DIR.parent.name == "data"
