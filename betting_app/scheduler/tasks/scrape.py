"""Scraping tasks for all bookmakers."""

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime

from betting_app.services.automation_service import (
    current_automation_run_id,
    finish_command,
    start_command,
)
from betting_app.utils.browser_cleanup import cleanup_browser_leftovers

logger = logging.getLogger(__name__)

BOOKMAKERS = ("sts", "betclic", "superbet", "efortuna", "betfan", "totalbet", "lebull")
HEADLESS_BOOKMAKERS = {"betclic", "superbet", "efortuna", "betfan"}


def _run_module(module: str, args: list[str] | None = None, timeout: int = 300) -> bool:
    """Run a Python module as a tracked subprocess. Return True on success."""
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)

    logger.info(f"Running: {' '.join(cmd)}")
    command_id = start_command(current_automation_run_id(), cmd)
    tmp_dir = tempfile.mkdtemp(prefix="betting-subprocess-", dir="/tmp")
    env = os.environ.copy()
    # Force Chromium/NoDriver temporary profiles, caches and crash files into a
    # per-task directory that is removed even when the child process times out.
    env["TMPDIR"] = tmp_dir
    env["TEMP"] = tmp_dir
    env["TMP"] = tmp_dir
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # subprocess.run() would kill only the direct child on timeout. We
            # retain the process-group id so nested browser processes die too.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as kill_exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to kill process group for timed-out %s: %s", module, kill_exc)
            stdout, stderr = proc.communicate()
            error = f"Timed out after {timeout}s"
            if stderr:
                error = f"{error}: {stderr[-4000:]}"
            finish_command(command_id, returncode=124, output=stdout, error=error)
            logger.error(
                "Module %s timed out after %ss; killed child process group. stdout=%r stderr=%r",
                module,
                timeout,
                (stdout or "")[-500:],
                (stderr or "")[-500:],
            )
            return False
        if proc.returncode != 0:
            error = (stderr or stdout or f"Exited with return code {proc.returncode}")[-4000:]
            finish_command(
                command_id,
                returncode=int(proc.returncode),
                output=stdout,
                error=error,
            )
            logger.error("Module %s failed (rc=%s): %s", module, proc.returncode, error)
            return False
        diagnostics = stdout or ""
        if stderr:
            diagnostics = f"{diagnostics}\n[stderr]\n{stderr}".lstrip()
        finish_command(command_id, returncode=0, output=diagnostics or None)
        if stdout:
            logger.info(f"Output: {stdout[:300]}")
        return True
    except Exception as exc:
        finish_command(command_id, returncode=1, error=f"{type(exc).__name__}: {exc}")
        logger.error(f"Module {module} error: {exc}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cleanup = cleanup_browser_leftovers(min_age_seconds=900)
        if cleanup["processes_killed"] or cleanup["temp_dirs_removed"]:
            logger.warning("Cleaned stale browser leftovers after %s: %s", module, cleanup)


def scrape_bookmaker(bookmaker: str) -> dict:
    """Scrape odds from a single bookmaker.
    
    Returns dict with status info.
    """
    logger.info(f"Starting scrape for: {bookmaker}")
    start = datetime.now(UTC)
    cleanup = cleanup_browser_leftovers(min_age_seconds=900)
    if cleanup["processes_killed"] or cleanup["temp_dirs_removed"]:
        logger.warning("Cleaned stale browser leftovers before scraping %s: %s", bookmaker, cleanup)
    
    headless = "--headless" if bookmaker in HEADLESS_BOOKMAKERS else ""
    args = ["--bookmaker", bookmaker]
    if headless:
        args.append(headless)
    
    success = _run_module("betting_app.scripts.scrape_odds", args, timeout=300)
    
    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Scrape {bookmaker}: {'OK' if success else 'FAIL'} ({duration:.1f}s)")
    
    return {
        "bookmaker": bookmaker,
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }


def scrape_all() -> dict:
    """Scrape all bookmakers sequentially."""
    logger.info("Starting full scrape cycle")
    start = datetime.now(UTC)
    results = []
    
    for bk in BOOKMAKERS:
        result = scrape_bookmaker(bk)
        results.append(result)
    
    success_count = sum(1 for r in results if r["success"])
    duration = (datetime.now(UTC) - start).total_seconds()
    
    logger.info(f"Full scrape done: {success_count}/{len(BOOKMAKERS)} OK ({duration:.1f}s)")
    
    return {
        "total": len(BOOKMAKERS),
        "succeeded": success_count,
        "failed": len(BOOKMAKERS) - success_count,
        "success": success_count == len(BOOKMAKERS),
        "results": results,
        "duration_s": duration,
    }


def scrape_prop_odds() -> dict:
    """Scrape proposition odds across supported bookmakers (IDEA-018)."""
    logger.info("Starting proposition odds scraping cycle")
    start = datetime.now(UTC)
    cleanup = cleanup_browser_leftovers(min_age_seconds=900)
    if cleanup["processes_killed"] or cleanup["temp_dirs_removed"]:
        logger.warning("Cleaned stale browser leftovers before prop scraping: %s", cleanup)

    success = _run_module("betting_app.scripts.collect_prop_odds", ["--scrape-all"], timeout=600)
    duration = (datetime.now(UTC) - start).total_seconds()
    logger.info(f"Proposition odds scrape: {'OK' if success else 'FAIL'} ({duration:.1f}s)")
    return {
        "task": "scrape_prop_odds",
        "success": success,
        "duration_s": duration,
        "timestamp": start.isoformat(),
    }

def cleanup_browser_artifacts(max_age_minutes: int = 15) -> dict:
    """Remove stale browser processes/temp dirs left by interrupted scrapes."""

    cleanup = cleanup_browser_leftovers(min_age_seconds=max_age_minutes * 60)
    logger.info("Browser artifact cleanup: %s", cleanup)
    return {"success": True, **cleanup}
