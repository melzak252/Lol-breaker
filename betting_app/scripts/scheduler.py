"""Simple scheduler for Docker/local betting automation.

The scheduler intentionally stays dependency-free.  It runs the existing CLI
modules in subprocesses, logs every command and keeps looping even when one
bookmaker fails.  Use it from docker-compose or locally.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

from betting_app.services.automation_service import finish_command, finish_run, start_command, start_run


DEFAULT_BOOKMAKERS = ("sts", "betclic", "superbet", "efortuna", "betfan", "totalbet", "lebull")
HEADLESS_BOOKMAKERS = {"betclic", "superbet", "efortuna", "betfan"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["light", "light-once", "heavy", "heavy-once"], default="light")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("BETTING_SCHEDULER_INTERVAL_SECONDS", "7200")))
    parser.add_argument("--heavy-interval-seconds", type=int, default=int(os.getenv("BETTING_HEAVY_INTERVAL_SECONDS", "21600")))
    parser.add_argument("--min-ev", type=float, default=float(os.getenv("BETTING_APP_MIN_EV", "0.05")))
    parser.add_argument("--bookmakers", default=os.getenv("BETTING_SCHEDULER_BOOKMAKERS", ",".join(DEFAULT_BOOKMAKERS)))
    parser.add_argument("--no-run-on-start", action="store_true", default=os.getenv("BETTING_SCHEDULER_RUN_ON_START", "1") in {"0", "false", "False"})
    args = parser.parse_args()

    ensure_db()
    if args.mode.endswith("once"):
        if args.mode.startswith("heavy"):
            run_heavy_cycle(args.min_ev, parse_bookmakers(args.bookmakers), trigger_source="manual")
        else:
            run_light_cycle(args.min_ev, parse_bookmakers(args.bookmakers), trigger_source="manual")
        return

    interval = args.heavy_interval_seconds if args.mode == "heavy" else args.interval_seconds
    cycle = run_heavy_cycle if args.mode == "heavy" else run_light_cycle
    if not args.no_run_on_start:
        cycle(args.min_ev, parse_bookmakers(args.bookmakers), interval_seconds=interval)
    while True:
        log(f"Sleeping {interval}s before next {args.mode} cycle")
        time.sleep(interval)
        cycle(args.min_ev, parse_bookmakers(args.bookmakers), interval_seconds=interval)


def parse_bookmakers(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def ensure_db() -> None:
    run_module("betting_app.scripts.init_db", check=False)
    run_module("betting_app.scripts.prepare_model_db", "--register-default-model", check=False)


def run_light_cycle(
    min_ev: float,
    bookmakers: list[str],
    *,
    interval_seconds: int | None = None,
    trigger_source: str = "scheduler",
) -> None:
    """Scrape odds and run canonical/features/predictions/EV."""

    run_id = start_run("light", trigger_source=trigger_source, interval_seconds=interval_seconds)
    log("Starting light cycle")
    try:
        for bookmaker in bookmakers:
            command = ["betting_app.scripts.scrape_odds", "--bookmaker", bookmaker]
            if bookmaker in HEADLESS_BOOKMAKERS:
                command.append("--headless")
            run_module(*command, check=False, automation_run_id=run_id)
        run_module("betting_app.scripts.rematch_canonical_matches", "--rebuild", check=False, automation_run_id=run_id)
        run_module(
            "betting_app.scripts.run_upcoming_prediction_pipeline",
            "--operational-hybrid",
            "--min-ev",
            str(min_ev),
            check=False,
            automation_run_id=run_id,
        )
        run_module(
            "betting_app.scripts.list_upcoming_model_predictions",
            "--positive-only",
            check=False,
            automation_run_id=run_id,
        )
        finish_run(run_id, status="completed", next_run_at=next_run_at(interval_seconds))
        log("Finished light cycle")
    except Exception as exc:  # pragma: no cover - defensive daemon guard
        finish_run(run_id, status="failed", error=str(exc), next_run_at=next_run_at(interval_seconds))
        log(f"Light cycle failed: {exc}")


def run_heavy_cycle(
    min_ev: float,
    bookmakers: list[str],
    *,
    interval_seconds: int | None = None,
    trigger_source: str = "scheduler",
) -> None:
    """Refresh GOL.GG/SQLite/regional ratings/W20, then run light cycle."""

    run_id = start_run("heavy", trigger_source=trigger_source, interval_seconds=interval_seconds)
    log("Starting heavy cycle")
    try:
        refresh_args = ["betting_app.scripts.refresh_golgg_results"]
        embedded_dir = os.getenv("EMBEDDED_RIFT_ESPORT_DIR")
        if embedded_dir:
            refresh_args.extend(["--embedded-rift-esport-dir", embedded_dir])
        run_module(*refresh_args, check=False, automation_run_id=run_id)
        run_module("betting_app.scripts.import_golgg_to_db", check=False, automation_run_id=run_id)
        run_module(
            "betting_app.scripts.rebuild_regional_ratings",
            "--ratings-version",
            "ratings-v2",
            "--source",
            "legacy-scheduler-regional-ratings-v2",
            check=False,
            automation_run_id=run_id,
        )
        run_module(
            "betting_app.scripts.rebuild_w20_features",
            "--feature-version",
            "w20-latest",
            "--window-size",
            "20",
            check=False,
            automation_run_id=run_id,
        )
        run_module(
            "betting_app.scripts.scheduler",
            "--mode",
            "light-once",
            "--min-ev",
            str(min_ev),
            "--bookmakers",
            ",".join(bookmakers),
            check=False,
            automation_run_id=run_id,
        )
        finish_run(run_id, status="completed", next_run_at=next_run_at(interval_seconds))
        log("Finished heavy cycle")
    except Exception as exc:  # pragma: no cover - defensive daemon guard
        finish_run(run_id, status="failed", error=str(exc), next_run_at=next_run_at(interval_seconds))
        log(f"Heavy cycle failed: {exc}")


def run_module(module: str, *args: str, check: bool = True, automation_run_id: int | None = None) -> int:
    command = [sys.executable, "-m", module, *args]
    log("$ " + " ".join(command))
    command_id = start_command(automation_run_id, command)
    result = subprocess.run(command, text=True)
    finish_command(command_id, returncode=int(result.returncode))
    if result.returncode != 0:
        log(f"Command failed rc={result.returncode}: {' '.join(command)}")
        if check:
            raise SystemExit(result.returncode)
    return int(result.returncode)


def next_run_at(interval_seconds: int | None) -> str | None:
    if not interval_seconds:
        return None
    return (datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=interval_seconds)).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).replace(microsecond=0).isoformat()}] {message}", flush=True)


if __name__ == "__main__":
    main()
