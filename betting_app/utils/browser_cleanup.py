"""Helpers for cleaning up browser automation leftovers.

NoDriver/Chromium processes can outlive the Python scraper when the scheduler
hard-times out a subprocess.  This module keeps cleanup conservative: it only
kills browser-related processes older than a configured age and only removes
known scraper temp directories.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from pathlib import Path


BROWSER_PROCESS_MARKERS = (
    "chromium",
    "chrome_crashpad",
    "chrome_crashpad_handler",
    "nodriver",
    "playwright",
)

TEMP_PATTERNS = (
    "betting-subprocess-*",
    "betting-nodriver-*",
    "uc_*",
    ".org.chromium.Chromium.*",
)


def _process_age_seconds(pid: int) -> float | None:
    """Return process age in seconds using /proc, or None if unavailable."""

    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # Field 22 is process start time in clock ticks.  The process name may
        # contain spaces inside parentheses, so split after the final ")".
        after_comm = stat_text.rsplit(")", maxsplit=1)[1].strip().split()
        start_ticks = int(after_comm[19])
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return max(0.0, uptime - (start_ticks / hz))
    except Exception:
        return None


def cleanup_stale_browser_processes(*, min_age_seconds: int = 900, sig: int = signal.SIGKILL) -> int:
    """Kill stale browser automation processes in the current container.

    Only processes whose command line contains one of ``BROWSER_PROCESS_MARKERS``
    and are older than ``min_age_seconds`` are killed.  This avoids interrupting
    currently running scrapes while preventing multi-day Chromium leaks.
    """

    current_pid = os.getpid()
    killed = 0
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        cmdline_lower = cmdline.lower()
        if not any(marker in cmdline_lower for marker in BROWSER_PROCESS_MARKERS):
            continue
        age = _process_age_seconds(pid)
        if age is None or age < min_age_seconds:
            continue
        try:
            os.kill(pid, sig)
            killed += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return killed


def cleanup_stale_temp_dirs(*, base_dir: str | Path = "/tmp", min_age_seconds: int = 900) -> int:
    """Remove old scraper/browser temp directories from ``base_dir``."""

    base = Path(base_dir)
    if not base.exists():
        return 0
    now = time.time()
    removed = 0
    for pattern in TEMP_PATTERNS:
        for path in base.glob(pattern):
            try:
                age = now - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < min_age_seconds:
                continue
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
    return removed


def cleanup_browser_leftovers(*, min_age_seconds: int = 900) -> dict[str, int]:
    """Clean stale browser processes and known temporary directories."""

    return {
        "processes_killed": cleanup_stale_browser_processes(min_age_seconds=min_age_seconds),
        "temp_dirs_removed": cleanup_stale_temp_dirs(min_age_seconds=min_age_seconds),
    }
