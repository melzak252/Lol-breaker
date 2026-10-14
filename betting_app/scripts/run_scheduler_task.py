"""Run one registered scheduler task synchronously with normal tracking."""

from __future__ import annotations

import argparse
import json

from betting_app.scheduler.app import execute_task
from betting_app.scheduler.registry import register_all_tasks, registry


def main() -> None:
    """Run a named registry task and return a failing shell status on failure."""

    register_all_tasks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_id",
        choices=sorted(task.id for task in registry.list_all()),
        help="Registered scheduler task to execute.",
    )
    args = parser.parse_args()
    task = registry.get(args.task_id)
    if task is None:  # argparse choices make this defensive only.
        raise SystemExit(f"Unknown scheduler task: {args.task_id}")

    result = execute_task(
        task.id,
        *task.args,
        _trigger_source="manual-cli",
        **task.kwargs,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    if isinstance(result, dict) and not result.get("success", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
