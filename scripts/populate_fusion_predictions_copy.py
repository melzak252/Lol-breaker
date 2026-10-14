#!/usr/bin/env python3
"""
Populate betting_app PostgreSQL DB with fusion v2 model predictions using COPY for bulk inserts.

This script uses psycopg2 directly (bypassing betting_app.core.db) with COPY FROM STDIN
for orders-of-magnitude faster bulk inserts compared to batch INSERT.

Steps:
1. Register 3 fusion models in model_artifacts (if not already)
2. Create canonical_matches entries from golgg_matches (COPY bulk insert)
3. Insert fusion predictions into canonical_predictions (COPY bulk insert)

Usage (inside Docker container):
    docker exec ensemblelegends-betting-api python /app/scripts/populate_fusion_predictions_copy.py [--dry-run] [--limit N]
"""

import argparse
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

import psycopg2
import psycopg2.extras

# --- Config ---
PREDICTIONS_PATH = Path(__file__).parent.parent / "data" / "fusion_predictions_all.json"

# Direct PostgreSQL connection (bypasses betting_app.core.db wrapper)
DB_HOST = "timescaledb"
DB_PORT = 5432
DB_NAME = "betting"
DB_USER = "betting"
DB_PASS = "betting_local_password"

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


def copy_from_csv(conn, table, columns, rows, null="\\N"):
    """
    Use COPY FROM STDIN with CSV format for bulk insert.
    Rows is a list of tuples. Values are auto-converted to strings.
    None values become NULL in PostgreSQL.
    """
    if not rows:
        return 0

    col_str = ", ".join(columns)
    buf = io.StringIO()

    for row in rows:
        line_parts = []
        for val in row:
            if val is None:
                line_parts.append(null)
            elif isinstance(val, bool):
                line_parts.append("t" if val else "f")
            elif isinstance(val, float):
                # Use repr for full precision
                line_parts.append(repr(val))
            else:
                # Escape CSV: double any quotes, wrap in quotes if contains comma/newline/quote
                s = str(val).replace('"', '""')
                if "," in s or "\n" in s or '"' in s or s == null:
                    s = f'"{s}"'
                line_parts.append(s)
        buf.write(",".join(line_parts) + "\n")

    buf.seek(0)
    try:
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {table} ({col_str}) FROM STDIN WITH (FORMAT csv, NULL '{null}')",
                buf,
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error during COPY to {table}: {e}")
        raise
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Populate fusion predictions into betting_app PostgreSQL DB (fast COPY)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of matches to process (0=all)")
    args = parser.parse_args()

    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    with open(PREDICTIONS_PATH) as f:
        all_predictions = json.load(f)
    print(f"  Loaded {len(all_predictions)} predictions")

    # Connect directly to PostgreSQL using psycopg2
    print(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )
    conn.autocommit = False

    # --- Step 1: Register fusion models in model_artifacts ---
    print("\n=== Step 1: Register fusion models ===")
    model_artifact_ids = {}

    with conn.cursor() as cur:
        for model in FUSION_MODELS:
            cur.execute(
                "SELECT id FROM model_artifacts WHERE model_name = %s AND model_version = %s",
                (model["model_name"], model["model_version"]),
            )
            row = cur.fetchone()
            if row:
                model_artifact_ids[model["pred_key"]] = row[0]
                print(f"  {model['model_name']} v{model['model_version']} already registered (id={row[0]})")
            else:
                if args.dry_run:
                    print(f"  [DRY RUN] Would register {model['model_name']} v{model['model_version']}")
                    model_artifact_ids[model["pred_key"]] = -1
                else:
                    cur.execute(
                        """INSERT INTO model_artifacts 
                           (model_name, model_version, artifact_path, feature_schema_json, model_params_json, 
                            training_cutoff_at, metrics_json, status)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                           RETURNING id""",
                        (
                            model["model_name"], model["model_version"], model["artifact_path"],
                            model["feature_schema_json"], model["model_params_json"],
                            model["training_cutoff_at"], model["metrics_json"],
                        ),
                    )
                    new_id = cur.fetchone()[0]
                    conn.commit()
                    model_artifact_ids[model["pred_key"]] = new_id
                    print(f"  Registered {model['model_name']} v{model['model_version']} (id={new_id})")

    # --- Step 2: Build canonical_matches from golgg_matches ---
    print("\n=== Step 2: Create canonical_matches from golgg_matches ===")

    # Get existing canonical_keys
    with conn.cursor() as cur:
        cur.execute("SELECT id, canonical_key FROM canonical_matches")
        existing_keys = {row[1]: row[0] for row in cur.fetchall()}
    print(f"  Existing canonical_keys: {len(existing_keys)}")

    # Load golgg_matches that have fusion predictions
    with conn.cursor() as cur:
        cur.execute(
            "SELECT match_id, date, tournament_name, team1_name, team2_name, best_of, team1_win "
            "FROM golgg_matches ORDER BY date, match_id"
        )
        golgg_rows = cur.fetchall()
    print(f"  Total golgg_matches: {len(golgg_rows)}")

    # Build list of new canonical_matches to insert
    match_id_to_canonical_key = {}
    new_canonical_rows = []
    new_keys = set()

    for row in golgg_rows:
        match_id, date, tournament, t1_name, t2_name, best_of, t1_win = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
        mid_str = str(match_id)

        if mid_str not in all_predictions:
            continue

        if args.limit > 0 and len(match_id_to_canonical_key) >= args.limit:
            break

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
        print("  Using COPY for bulk insert of canonical_matches...")
        n = copy_from_csv(
            conn, "canonical_matches",
            ["canonical_key", "team_a_name", "team_b_name", "normalized_team_a", "normalized_team_b",
             "start_time_normalized", "league", "status", "match_confidence"],
            new_canonical_rows,
        )
        print(f"  Inserted {n} canonical_matches via COPY")

        # Refresh existing_keys with newly inserted rows
        with conn.cursor() as cur:
            cur.execute("SELECT id, canonical_key FROM canonical_matches")
            existing_keys = {row[1]: row[0] for row in cur.fetchall()}
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
        with conn.cursor() as cur:
            cur.execute("SELECT canonical_match_id, model_name FROM canonical_predictions")
            existing_preds = {(row[0], row[1]) for row in cur.fetchall()}
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
        print("  Using COPY for bulk insert of canonical_predictions...")
        n = copy_from_csv(
            conn, "canonical_predictions",
            ["canonical_match_id", "model_artifact_id", "model_name", "model_version",
             "prob_a", "prob_b", "features_version", "ratings_version", "data_cutoff_at",
             "prediction_status", "diagnostics_json"],
            prediction_rows,
        )
        print(f"  Inserted {n} canonical_predictions via COPY")

    # --- Verify ---
    print("\n=== Verification ===")
    if not args.dry_run:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM canonical_matches")
            cm_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM canonical_predictions")
            cp_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM model_artifacts")
            ma_count = cur.fetchone()[0]

        print(f"  canonical_matches: {cm_count} rows")
        print(f"  canonical_predictions: {cp_count} rows")
        print(f"  model_artifacts: {ma_count} rows")

        with conn.cursor() as cur:
            for model in FUSION_MODELS:
                cur.execute(
                    "SELECT COUNT(*) FROM canonical_predictions WHERE model_name = %s",
                    (model["model_name"],),
                )
                cnt = cur.fetchone()[0]
                print(f"    {model['model_name']}: {cnt} predictions")

        print("\n  Sample predictions (2025+):")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cm.team_a_name, cm.team_b_name, cm.start_time_normalized, 
                       cp.model_name, cp.prob_a, cp.prob_b
                FROM canonical_predictions cp
                JOIN canonical_matches cm ON cp.canonical_match_id = cm.id
                WHERE cm.start_time_normalized >= '2025'
                LIMIT 6
            """)
            for s in cur.fetchall():
                print(f"    {s[0]} vs {s[1]} ({s[2][:10]}) | {s[3]}: p_a={s[4]:.4f} p_b={s[5]:.4f}")
    else:
        print("  [DRY RUN] No verification - no data was written")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
