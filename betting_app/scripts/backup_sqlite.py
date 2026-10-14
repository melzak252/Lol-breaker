"""Create a timestamped backup of the local SQLite database."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from betting_app.core.config import load_config
from betting_app.core.db import init_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", default=None, help="Default: <data-dir>/backups")
    parser.add_argument("--keep", type=int, default=30, help="How many newest backups to keep")
    args = parser.parse_args()

    db_path = init_db()
    cfg = load_config()
    backup_dir = Path(args.backup_dir) if args.backup_dir else cfg.db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = backup_dir / f"betting_app_{timestamp}.sqlite3"
    shutil.copy2(db_path, target)

    prune_backups(backup_dir, keep=args.keep)
    print(f"Created SQLite backup: {target}")


def prune_backups(backup_dir: Path, *, keep: int) -> None:
    if keep <= 0:
        return
    backups = sorted(backup_dir.glob("betting_app_*.sqlite3"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old_backup in backups[keep:]:
        old_backup.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
