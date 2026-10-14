#!/usr/bin/env python3
"""
Populate betting_app PostgreSQL DB with fusion v2 model predictions for historical matches.

Optimized version: uses batch INSERTs with commit per batch (not per row).
Uses betting_app.core.db.connect() which works with both SQLite and PostgreSQL.

Steps:
1. Register 3 fusion models in model_artifacts
2. Create canonical_matches entries from golgg_matches (batch insert)
3. Insert fusion predictions into canonical_predictions (batch insert)

Usage (inside Docker container):
    docker exec ensemblelegends-betting-api python /app/scripts/populate_fusion_predictions_pg.py [--dry-run] [--limit N]
"""

import argparse
import hashlib
import json
import unicodedata
import re
from pathlib import Path

from betting_app.core.db import connect, is_pg

# --- Config ---
PREDICTIONS_PATH = Path(__file__).parent.parent / "data" / "fusion_predictions_all.json"

STOP_WORDS = {"the", "esports", "esport", "gaming", "games", "game", "team", "club", "e", "v", "of"}

FUSION_MODELS = [
    {
        "model_name": "Fusion-v2",
        "model_version": "v1.0",
        "artifact_path": "models/fusion_v2_best.pt",
        "feature_schema_json": json.dumps({"baseline_features": 54, "embedding_dim": 192, "total": 246}),
        "model_params_json": json.dumps({"architecture": "256->128->64->1 MLP", "dropout": [0.3, 0.2, 0.1], "batch_norm": True}),
        "training_cutoff_at": "2025-01-01",
        "metrics_json": json.dumps({"test_logloss": 0.5582, "test_auc": 0.7822, "test_accuracy": 0.7131, "test_ece": 0.0217}),
        "pred_key": "fusion_v2",
    },
    {
        "model_name": "Fusion-v2-SymAug",
        "model_version": "v1.0",
        "artifact_path": "models/fusion_v2_sym_best.pt",
        "feature_schema_json": json.dumps({"baseline_features": 54, "embedding_dim": 192, "total": 246, "symmetrization": "data_augmentation_inference_averaging"}),
        "model_params_json": json.dumps({"architecture": "256->128->64->1 MLP", "dropout": [0.3, 0.2, 0.1], "batch_norm": True, "symmetrization": "swap+average"}),
        "training_cutoff_at": "2025-01-01",
        "metrics_json": json.dumps({"test_logloss": 0.5575, "test_auc": 0.7817, "test_accuracy": 0.7086, "test_ece": 0.0225}),
        "pred_key": "fusion_v2_sym",
    },
    {
        "model_name": "Fusion-v2-ArchSym",
        "model_version": "v1.0",
        "artifact_path": "models/fusion_v2_archsym_best.pt",
        "feature_schema_json": json.dumps({"baseline_features": 54, "embedding_dim": 192, "total": 246, "symmetrization": "architectural"}),
        "model_params_json": json.dumps({"architecture": "SymmetricFusionModel 256->128->64->1", "dropout": [0.3, 0.2, 0.1], "batch_norm": True, "symmetrization": "architectural_sym_logit"}),
        "training_cutoff_at": "2025-01-01",
        "metrics_json": json.dumps({"test_logloss": 0.5623, "test_auc": 0.7824, "test_accuracy": 0.7054, "test_ece": 0.0374}),
        "pred_key": "fusion_v2_archsym",
    },
]


def normalize_team_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    tokens = [t for t in name.split() if t not in STOP_WORDS]
    return " ".join(tokens)


def build_canonical_key(team_a_key: str, team_b_key: str, start_norm: str, league_norm: str) -> str:
    left, right = sorted([team_a_key, team_b_key])
    time_bucket = start_norm[:13] if start_norm else "unknown"
    base = f"{left}|{right}|{time_bucket}|{league_norm or 'unknown'}"
    digest = hashlib.sha1(base.encode()).hexdigest()[:10]
    return f"{base}|{digest}"


def normalize_league(name: str) -> str:
    if not name:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def batch_insert(conn, table, columns, rows, batch_size=5000):
    """Insert rows in batches, committing after each batch. Returns total inserted."""
    if not rows:
        return 0
    placeholders = ", ".join(["?"] * len(columns))
    col_str = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        conn.executemany(sql, batch)
        conn.commit()
        total += len(batch)
        print(f"    Inserted {total}/{len(rows)} rows into {table}...")
    return total


def main():
    parser = argparse.ArgumentParser(description="Populate fusion predictions into betting_app DB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of matches to process (0=all)")
    args = parser.parse_args()

    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    with open(PREDICTIONS_PATH) as f:
        all_predictions = json.load(f)
    print(f"  Loaded {len(all_predictions)} predictions")

    db_type = "PostgreSQL" if is_pg() else "SQLite"
    print(f"Connecting to {db_type} DB...")
    conn = connect()

    # --- Step 1: Register fusion models in model_artifacts ---
    print("\n=== Step 1: Register fusion models ===")
    model_artifact_ids = {}
    for model in FUSION_MODELS:
        existing = conn.execute(
            "SELECT id FROM model_artifacts WHERE model_name=? AND model_version=?",
            (model["model_name"], model["model_version"]),
        ).fetchone()
        if existing:
            model_artifact_ids[model["pred_key"]] = existing["id"]
            print(f"  {model['model_name']} v{model['model_version']} already registered (id={existing['id']})")
        else:
            if args.dry_run:
                print(f"  [DRY RUN] Would register {model['model_name']} v{model['model_version']}")
                model_artifact_ids[model["pred_key"]] = -1
            else:
                result = conn.execute(
                    """INSERT INTO model_artifacts 
                       (model_name, model_version, artifact_path, feature_schema_json, model_params_json, 
                        training_cutoff_at, metrics_json, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'active')""",
                    (
                        model["model_name"], model["model_version"], model["artifact_path"],
                        model["feature_schema_json"], model["model_params_json"],
                        model["training_cutoff_at"], model["metrics_json"],
                    ),
                )
                conn.commit()
                model_artifact_ids[model["pred_key"]] = result.lastrowid
                print(f"  Registered {model['model_name']} v{model['model_version']} (id={result.lastrowid})")

    # --- Step 2: Build canonical_matches from golgg_matches ---
    print("\n=== Step 2: Create canonical_matches from golgg_matches ===")

    # Get existing canonical_keys
    existing_rows = conn.execute("SELECT id, canonical_key FROM canonical_matches").fetchall()
    existing_keys = {r["canonical_key"]: r["id"] for r in existing_rows}
    print(f"  Existing canonical_keys: {len(existing_keys)}")

    # Load golgg_matches that have fusion predictions
    golgg_rows = conn.execute(
        "SELECT match_id, date, tournament_name, team1_name, team2_name, best_of, team1_win "
        "FROM golgg_matches ORDER BY date, match_id"
    ).fetchall()
    print(f"  Total golgg_matches: {len(golgg_rows)}")

    # Build list of new canonical_matches to insert
    # Also build match_id -> canonical_key mapping
    match_id_to_canonical_key = {}
    new_canonical_rows = []  # list of tuples for batch insert
    new_keys = set()

    for row in golgg_rows:
        match_id = row["match_id"]
        mid_str = str(match_id)

        if mid_str not in all_predictions:
            continue

        if args.limit > 0 and len(match_id_to_canonical_key) >= args.limit:
            break

        t1_name = row["team1_name"]
        t2_name = row["team2_name"]
        date = row["date"]
        tournament = row["tournament_name"]
        t1_win = row["team1_win"]

        norm_a = normalize_team_name(t1_name)
        norm_b = normalize_team_name(t2_name)
        league_norm = normalize_league(tournament)
        start_norm = f"{date}T00:00:00" if date else None
        canonical_key = build_canonical_key(norm_a, norm_b, start_norm, league_norm)

        match_id_to_canonical_key[mid_str] = canonical_key

        if canonical_key not in existing_keys and canonical_key not in new_keys:
            status = "completed" if t1_win is not None else "upcoming"
            new_canonical_rows.append((
                canonical_key, t1_name, t2_name, norm_a, norm_b,
                start_norm, tournament, status, 1.0,  # match_confidence
            ))
            new_keys.add(canonical_key)

    print(f"  New canonical_matches to insert: {len(new_canonical_rows)}")
    print(f"  Total matches with predictions: {len(match_id_to_canonical_key)}")

    if not args.dry_run and new_canonical_rows:
        batch_insert(
            conn, "canonical_matches",
            ["canonical_key", "team_a_name", "team_b_name", "normalized_team_a", "normalized_team_b",
             "start_time_normalized", "league", "status", "match_confidence"],
            new_canonical_rows,
            batch_size=5000,
        )

        # Refresh existing_keys with newly inserted rows
        existing_rows = conn.execute("SELECT id, canonical_key FROM canonical_matches").fetchall()
        existing_keys = {r["canonical_key"]: r["id"] for r in existing_rows}
        print(f"  canonical_matches now has {len(existing_keys)} rows")

    # Build match_id -> canonical_id mapping
    match_id_to_canonical_id = {}
    for mid_str, ckey in match_id_to_canonical_key.items():
        if ckey in existing_keys:
            match_id_to_canonical_id[mid_str] = existing_keys[ckey]
        elif args.dry_run:
            match_id_to_canonical_id[mid_str] = -1

    # --- Step 3: Insert fusion predictions ---
    print("\n=== Step 3: Insert fusion predictions ===")

    # Get existing predictions to avoid duplicates
    existing_preds = set()
    if not args.dry_run:
        rows = conn.execute(
            "SELECT canonical_match_id, model_name FROM canonical_predictions"
        ).fetchall()
        existing_preds = {(r["canonical_match_id"], r["model_name"]) for r in rows}
        print(f"  Existing predictions: {len(existing_preds)}")

    # Build all prediction rows
    prediction_rows = []
    skipped = 0

    for mid_str, pred_data in all_predictions.items():
        if mid_str not in match_id_to_canonical_id:
            skipped += 1
            continue
        canonical_id = match_id_to_canonical_id[mid_str]
        if canonical_id == -1:  # dry run
            continue

        y_true = pred_data.get("y_true")
        date = pred_data.get("date", "")

        for model in FUSION_MODELS:
            pred_key = model["pred_key"]
            prob_a = pred_data.get(pred_key)
            if prob_a is None:
                continue

            prob_b = 1.0 - prob_a

            if not args.dry_run and (canonical_id, model["model_name"]) in existing_preds:
                continue

            diagnostics = json.dumps({
                "y_true": y_true,
                "date": date,
                "match_id": mid_str,
                "player_elo_prob": pred_data.get("player_elo"),
            })

            prediction_rows.append((
                canonical_id,
                model_artifact_ids.get(pred_key),
                model["model_name"],
                model["model_version"],
                prob_a,
                prob_b,
                "fusion_v2_features",
                "v2",
                date if date else None,
                "active",
                diagnostics,
            ))

    print(f"  Predictions to insert: {len(prediction_rows)} (skipped {skipped} no canonical_match)")

    if not args.dry_run and prediction_rows:
        batch_insert(
            conn, "canonical_predictions",
            ["canonical_match_id", "model_artifact_id", "model_name", "model_version",
             "prob_a", "prob_b", "features_version", "ratings_version", "data_cutoff_at",
             "prediction_status", "diagnostics_json"],
            prediction_rows,
            batch_size=5000,
        )

    # --- Verify ---
    print("\n=== Verification ===")
    if not args.dry_run:
        cm_count = conn.execute("SELECT COUNT(*) as cnt FROM canonical_matches").fetchone()["cnt"]
        cp_count = conn.execute("SELECT COUNT(*) as cnt FROM canonical_predictions").fetchone()["cnt"]
        ma_count = conn.execute("SELECT COUNT(*) as cnt FROM model_artifacts").fetchone()["cnt"]
        print(f"  canonical_matches: {cm_count} rows")
        print(f"  canonical_predictions: {cp_count} rows")
        print(f"  model_artifacts: {ma_count} rows")

        for model in FUSION_MODELS:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM canonical_predictions WHERE model_name=?",
                (model["model_name"],),
            ).fetchone()
            print(f"    {model['model_name']}: {row['cnt']} predictions")

        print("\n  Sample predictions (2025+):")
        samples = conn.execute("""
            SELECT cm.team_a_name, cm.team_b_name, cm.start_time_normalized, 
                   cp.model_name, cp.prob_a, cp.prob_b
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cp.canonical_match_id = cm.id
            WHERE cm.start_time_normalized >= '2025'
            LIMIT 6
        """).fetchall()
        for s in samples:
            print(f"    {s['team_a_name']} vs {s['team_b_name']} ({s['start_time_normalized'][:10]}) | {s['model_name']}: p_a={s['prob_a']:.4f} p_b={s['prob_b']:.4f}")
    else:
        print("  [DRY RUN] No verification - no data was written")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
