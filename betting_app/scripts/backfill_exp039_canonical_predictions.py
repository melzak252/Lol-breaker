"""Backfill canonical EXP-039 predictions from the corrected EXP-060 DB backtest.

This repairs stale/mismatched historical rows in `canonical_predictions` for
`Sym-Cal LR-ElasticNet-W20-Binomial/exp-039` using the corrected probabilities
computed by `backtest_exp039_db_market.py` and stored in
`reports/exp039_db_market_backtest_v2/exp039_market_common.csv`.

The script is intentionally conservative:
- dry-run by default;
- writes only when `--apply` is passed;
- backs up existing target rows to a timestamped table before deletion;
- only touches canonical matches present in the corrected EXP-060 market-common file.

Usage:
  docker exec -w /app ensemblelegends-betting-scheduler \
    python -m betting_app.scripts.backfill_exp039_canonical_predictions --apply
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text

from betting_app.core.db import get_session
from betting_app.services.thesis_inference_service import THESIS_MODEL_NAME, THESIS_MODEL_VERSION


DEFAULT_INPUT = Path("reports/exp039_db_market_backtest_v2/exp039_market_common.csv")
DEFAULT_REPORT_DIR = Path("reports/exp039_canonical_backfill")
FEATURES_VERSION = "exp060-db-backfill-v1"
RATINGS_VERSION = "latest-full"


def _parse_start(value: str) -> datetime:
    parsed = pd.to_datetime(value, utc=True)
    return parsed.to_pydatetime()


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(input_path)
    required = {
        "canonical_match_id",
        "golgg_match_id",
        "start_time_normalized",
        "exp039_prob_team_a",
        "exp039_calibrated_prob_team1",
        "canonical_a_is_golgg_team1",
        "team_a_name",
        "team_b_name",
        "team1_name",
        "team2_name",
        "y_team_a",
        "y_team1",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        prob_a = float(rec["exp039_prob_team_a"])
        prob_a = min(max(prob_a, 1e-6), 1.0 - 1e-6)
        start_at = _parse_start(str(rec["start_time_normalized"]))
        predicted_at = start_at - timedelta(minutes=1)
        diagnostics = {
            "source": "EXP-060 corrected DB backtest",
            "source_file": str(input_path),
            "golgg_match_id": int(rec["golgg_match_id"]),
            "exp039_calibrated_prob_team1": float(rec["exp039_calibrated_prob_team1"]),
            "canonical_a_is_golgg_team1": bool(rec["canonical_a_is_golgg_team1"]),
            "team_a_name": str(rec.get("team_a_name") or ""),
            "team_b_name": str(rec.get("team_b_name") or ""),
            "golgg_team1_name": str(rec.get("team1_name") or ""),
            "golgg_team2_name": str(rec.get("team2_name") or ""),
            "y_team_a": int(rec["y_team_a"]),
            "y_team1": int(rec["y_team1"]),
            "backfill_method": "delete old exp-039 rows for target canonical ids, insert one corrected row per match",
        }
        rows.append(
            {
                "canonical_match_id": int(rec["canonical_match_id"]),
                "model_name": THESIS_MODEL_NAME,
                "model_version": THESIS_MODEL_VERSION,
                "predicted_at": predicted_at,
                "prob_a": prob_a,
                "prob_b": 1.0 - prob_a,
                "features_version": FEATURES_VERSION,
                "ratings_version": RATINGS_VERSION,
                "data_cutoff_at": predicted_at.isoformat(),
                "prediction_status": "active",
                "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
            }
        )
    # Deduplicate defensively by canonical match id, preserving first CSV row.
    dedup: dict[int, dict[str, Any]] = {}
    for row in rows:
        dedup.setdefault(row["canonical_match_id"], row)
    return list(dedup.values())


def _query_counts(db, match_ids: list[int]) -> dict[str, Any]:
    if not match_ids:
        return {"existing_target_rows": 0, "existing_target_matches": 0}
    stmt = text(
        """
        SELECT count(*) AS rows_n,
               count(DISTINCT canonical_match_id) AS matches_n,
               min(predicted_at) AS min_predicted_at,
               max(predicted_at) AS max_predicted_at
        FROM canonical_predictions
        WHERE model_name = :model_name
          AND model_version = :model_version
          AND canonical_match_id IN :match_ids
        """
    ).bindparams(bindparam("match_ids", expanding=True))
    row = db.execute(
        stmt,
        {"model_name": THESIS_MODEL_NAME, "model_version": THESIS_MODEL_VERSION, "match_ids": match_ids},
    ).mappings().one()
    return dict(row)


def run(input_path: Path, report_dir: Path, apply: bool) -> dict[str, Any]:
    rows = _load_rows(input_path)
    match_ids = [r["canonical_match_id"] for r in rows]
    report_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)

    db = get_session()
    backup_table: str | None = None
    try:
        before = _query_counts(db, match_ids)
        report: dict[str, Any] = {
            "success": False,
            "apply": apply,
            "started_at": started.isoformat(),
            "input_path": str(input_path),
            "model_name": THESIS_MODEL_NAME,
            "model_version": THESIS_MODEL_VERSION,
            "features_version": FEATURES_VERSION,
            "ratings_version": RATINGS_VERSION,
            "candidate_rows": len(rows),
            "candidate_matches": len(match_ids),
            "before": before,
        }

        if not apply:
            report["success"] = True
            report["message"] = "dry-run only; pass --apply to write database"
            return report

        ts = started.strftime("%Y%m%d_%H%M%S")
        backup_table = f"canonical_predictions_exp039_backup_{ts}"
        db.execute(
            text(
                f"""
                CREATE TABLE {backup_table} AS
                SELECT cp.*, now() AS backup_created_at
                FROM canonical_predictions cp
                WHERE cp.model_name = :model_name
                  AND cp.model_version = :model_version
                  AND cp.canonical_match_id IN :match_ids
                """
            ).bindparams(bindparam("match_ids", expanding=True)),
            {"model_name": THESIS_MODEL_NAME, "model_version": THESIS_MODEL_VERSION, "match_ids": match_ids},
        )
        deleted = db.execute(
            text(
                """
                DELETE FROM canonical_predictions
                WHERE model_name = :model_name
                  AND model_version = :model_version
                  AND canonical_match_id IN :match_ids
                """
            ).bindparams(bindparam("match_ids", expanding=True)),
            {"model_name": THESIS_MODEL_NAME, "model_version": THESIS_MODEL_VERSION, "match_ids": match_ids},
        ).rowcount
        insert_stmt = text(
            """
            INSERT INTO canonical_predictions (
                canonical_match_id, model_name, model_version, predicted_at,
                prob_a, prob_b, features_version, ratings_version, data_cutoff_at,
                prediction_status, diagnostics_json
            ) VALUES (
                :canonical_match_id, :model_name, :model_version, :predicted_at,
                :prob_a, :prob_b, :features_version, :ratings_version, :data_cutoff_at,
                :prediction_status, :diagnostics_json
            )
            """
        )
        db.execute(insert_stmt, rows)
        db.commit()
        after = _query_counts(db, match_ids)
        report.update(
            {
                "success": True,
                "backup_table": backup_table,
                "deleted_rows": int(deleted or 0),
                "inserted_rows": len(rows),
                "after": after,
            }
        )
        return report
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill corrected EXP-039 canonical_predictions rows")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--apply", action="store_true", help="write database changes; default is dry-run")
    args = parser.parse_args()

    result = run(args.input, args.report_dir, args.apply)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / ("backfill_apply_report.json" if args.apply else "backfill_dry_run_report.json")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
