#!/usr/bin/env python3
"""Export minimal EXP-083 sports inputs, never a database dump or operational write."""

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

QUERIES = {
    "matches": """SELECT match_id,date,tournament_name,team1_id,team2_id,
        team1_name,team2_name,team1_score,team2_score,team1_win,team2_win,best_of
        FROM golgg_matches ORDER BY date,match_id""",
    "games": """SELECT game_id,match_id,date,team1_id,team2_id,team1_win,team2_win,
        team1_name,team2_name,game_duration,team1_stats_json,team2_stats_json FROM golgg_games
        ORDER BY date,match_id,game_id""",
    "players": """SELECT game_id,match_id,team_id,player_id,player_name,role
        FROM golgg_game_players ORDER BY game_id,team_id,player_id""",
}


def export_data(output_dir: Path) -> dict:
    """One consistent PostgreSQL read-only snapshot; record no connection credentials."""
    url = database_url()
    if make_url(url).get_backend_name() != "postgresql":
        raise ValueError(
            "Research extraction requires PostgreSQL read-only transactions"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    engine = create_engine(
        url,
        isolation_level="REPEATABLE READ",
        connect_args={
            "connect_timeout": 10,
            "options": "-c default_transaction_read_only=on -c statement_timeout=180000",
        },
    )
    manifest = {
        "contract": "exp083-sports-extract-v1",
        "source": "configured PostgreSQL sports tables; no user/account data",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_available_at": None,
        "availability_caveat": "Present-day retrospective tables, not historical source snapshots",
        "tables": {},
    }
    try:
        with engine.connect() as conn, conn.begin():
            assert conn.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
            manifest["snapshot_at"] = str(
                conn.execute(text("SELECT CURRENT_TIMESTAMP")).scalar_one()
            )
            for name, query in QUERIES.items():
                result = conn.execution_options(stream_results=True).execute(
                    text(query)
                )
                columns = list(result.keys())
                path = output_dir / f"{name}.csv"
                count = 0
                with path.open("x", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    for rows in result.partitions(10000):
                        writer.writerows(rows)
                        count += len(rows)
                result.close()
                manifest["tables"][name] = {
                    "rows": count,
                    "columns": columns,
                    "sha256": sha256(path),
                    "query": query,
                }
    finally:
        engine.dispose()
    manifest["code_sha256"] = sha256(Path(__file__))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    print(json.dumps(export_data(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
