#!/usr/bin/env python3
"""EXP-086 minimal retrospective champion sports extract; never operational writes.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/export_champion_research_data.py --output-dir data/artifacts/exp086-champion-meta/source-20260908
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from betting_app.core.db import database_url
from scripts.build_siamese_research_dataset import sha256

QUERY = """SELECT gp.game_id,gp.match_id,m.date AS date,g.date AS game_date,
    gp.team_id,gp.player_id,gp.role,gp.champion_id,gp.champion_name,
    g.patch AS observed_game_patch,m.patch AS observed_match_patch,
    CASE WHEN gp.team_id=g.team1_id THEN g.team1_win
         WHEN gp.team_id=g.team2_id THEN g.team2_win ELSE NULL END AS game_win
    FROM golgg_game_players gp
    JOIN golgg_games g ON g.game_id=gp.game_id AND g.match_id=gp.match_id
    JOIN golgg_matches m ON m.match_id=g.match_id
    WHERE m.date >= '2013-01-01' AND m.date <= '2026-05-11'
    ORDER BY m.date,gp.match_id,gp.game_id,gp.team_id,gp.role,gp.player_id"""


def export_data(output_dir: Path) -> dict:
    url = database_url()
    if make_url(url).get_backend_name() != "postgresql":
        raise ValueError("PostgreSQL read-only transactions required")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "contract": "exp086-minimal-sports-extract-v1",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source": "configured PostgreSQL sports tables",
        "source_available_at": None,
        "availability_caveat": "Present-day retrospective tables; no historical ingestion timestamps",
        "query": QUERY,
        "code_sha256": sha256(Path(__file__)),
    }
    engine = create_engine(
        url,
        isolation_level="REPEATABLE READ",
        connect_args={
            "connect_timeout": 10,
            "options": "-c default_transaction_read_only=on -c statement_timeout=180000",
        },
    )
    try:
        with engine.connect() as conn, conn.begin():
            readonly = conn.execute(text("SHOW transaction_read_only")).scalar_one()
            isolation = conn.execute(text("SHOW transaction_isolation")).scalar_one()
            if readonly != "on" or isolation != "repeatable read":
                raise RuntimeError("Read-only repeatable-read verification failed")
            manifest.update(
                transaction_read_only=readonly,
                transaction_isolation=isolation,
                snapshot_at=str(
                    conn.execute(text("SELECT CURRENT_TIMESTAMP")).scalar_one()
                ),
            )
            result = conn.execution_options(stream_results=True).execute(text(QUERY))
            columns = list(result.keys())
            count = 0
            path = output_dir / "champion_history.csv"
            with path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for rows in result.partitions(10000):
                    writer.writerows(rows)
                    count += len(rows)
            result.close()
            manifest.update(
                rows=count, columns=columns, sha256=sha256(path), status="completed"
            )
    except Exception as exc:
        # Do not serialize driver exception messages: these can contain connection details.
        manifest.update(status="failed", failure_type=type(exc).__name__)
        raise RuntimeError(
            f"Read-only extraction failed ({type(exc).__name__}); safe manifest retained"
        ) from None
    finally:
        engine.dispose()
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    print(json.dumps(export_data(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
