#!/usr/bin/env python3
"""
Populate betting_app DB with fusion v2 model predictions for historical matches.

Steps:
1. Register 3 fusion models in model_artifacts
2. Create canonical_matches entries from golgg_matches (for historical matches)
3. Insert fusion predictions into canonical_predictions

Usage:
    cd /home/melzak/dev/inzynierka
    /home/melzak/dev/embedded-rift/.venv/bin/python scripts/populate_fusion_predictions.py [--dry-run]
"""

import argparse
import hashlib
import json
import sqlite3
import unicodedata
import re
from datetime import datetime
from pathlib import Path

# --- Config ---
DB_PATH = Path(__file__).parent.parent / "data" / "betting_app.sqlite3"
PREDICTIONS_PATH = Path(__file__).parent.parent / "data" / "fusion_predictions_all.json"

# Stop words from betting_app/core/matching.py
STOP_WORDS = {"the", "esports", "esport", "gaming", "games", "game", "team", "club", "e", "v", "of"}

# Models to register
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
    """Replicate betting_app/core/matching.py normalize_team_name."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    tokens = [t for t in name.split() if t not in STOP_WORDS]
    return " ".join(tokens)


def build_canonical_key(team_a_key: str, team_b_key: str, start_norm: str, league_norm: str) -> str:
    """Replicate betting_app/services/canonical_match_service.py build_canonical_key."""
    left, right = sorted([team_a_key, team_b_key])
    time_bucket = start_norm[:13] if start_norm else "unknown"
    base = f"{left}|{right}|{time_bucket}|{league_norm or 'unknown'}"
    digest = hashlib.sha1(base.encode()).hexdigest()[:10]
    return f"{base}|{digest}"


def normalize_league(name: str) -> str:
    """Simple league normalization."""
    if not name:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def main():
    parser = argparse.ArgumentParser(description="Populate fusion predictions into betting_app DB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of matches to process (0=all)")
    args = parser.parse_args()

    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    with open(PREDICTIONS_PATH) as f:
        all_predictions = json.load(f)
    print(f"  Loaded {len(all_predictions)} predictions")

    print(f"Connecting to DB at {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # --- Step 1: Register fusion models in model_artifacts ---
    print("\n=== Step 1: Register fusion models ===")
    model_artifact_ids = {}
    for model in FUSION_MODELS:
        existing = conn.execute(
            "SELECT id FROM model_artifacts WHERE model_name=? AND model_version=?",
            (model["model_name"], model["model_version"]),
        ).fetchone()
        if existing:
            model_artifact_ids[model["pred_key"]] = existing[0]
            print(f"  {model['model_name']} v{model['model_version']} already registered (id={existing[0]})")
        else:
            if args.dry_run:
                print(f"  [DRY RUN] Would register {model['model_name']} v{model['model_version']}")
                model_artifact_ids[model["pred_key"]] = -1
            else:
                cursor = conn.execute(
                    """INSERT INTO model_artifacts 
                       (model_name, model_version, artifact_path, feature_schema_json, model_params_json, 
                        training_cutoff_at, metrics_json, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'active')""",
                    (
                        model["model_name"],
                        model["model_version"],
                        model["artifact_path"],
                        model["feature_schema_json"],
                        model["model_params_json"],
                        model["training_cutoff_at"],
                        model["metrics_json"],
                    ),
                )
                model_artifact_ids[model["pred_key"]] = cursor.lastrowid
                print(f"  Registered {model['model_name']} v{model['model_version']} (id={cursor.lastrowid})")
    if not args.dry_run:
        conn.commit()

    # --- Step 2: Load golgg_matches and create canonical_matches ---
    print("\n=== Step 2: Create canonical_matches from golgg_matches ===")
    
    # Get existing canonical_keys to avoid duplicates
    existing_keys = set(r[0] for r in conn.execute("SELECT canonical_key FROM canonical_matches").fetchall())
    print(f"  Existing canonical_keys: {len(existing_keys)}")

    # Load golgg_matches that have fusion predictions
    golgg_rows = conn.execute(
        "SELECT match_id, date, tournament_name, team1_name, team2_name, best_of, team1_win "
        "FROM golgg_matches ORDER BY date, match_id"
    ).fetchall()

    # Build mapping: match_id -> canonical_match_id
    match_id_to_canonical_id = {}
    new_canonical_count = 0

    for row in golgg_rows:
        match_id, date, tournament, t1_name, t2_name, best_of, t1_win = row
        mid_str = str(match_id)

        # Skip if no fusion prediction
        if mid_str not in all_predictions:
            continue

        if args.limit > 0 and len(match_id_to_canonical_id) >= args.limit:
            break

        # Build canonical key
        norm_a = normalize_team_name(t1_name)
        norm_b = normalize_team_name(t2_name)
        league_norm = normalize_league(tournament)
        start_norm = f"{date}T00:00:00" if date else None
        canonical_key = build_canonical_key(norm_a, norm_b, start_norm, league_norm)

        if canonical_key in existing_keys:
            # Already exists, get its ID
            cid = conn.execute(
                "SELECT id FROM canonical_matches WHERE canonical_key=?", (canonical_key,)
            ).fetchone()[0]
            match_id_to_canonical_id[mid_str] = cid
        else:
            # Create new canonical_match
            status = "completed" if t1_win is not None else "upcoming"
            if args.dry_run:
                match_id_to_canonical_id[mid_str] = -1
                if new_canonical_count < 3:
                    print(f"  [DRY RUN] Would insert: {t1_name} vs {t2_name} ({date}) key={canonical_key[:60]}...")
            else:
                cursor = conn.execute(
                    """INSERT INTO canonical_matches 
                       (canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b,
                        start_time_normalized, league, status, match_confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0)""",
                    (canonical_key, t1_name, t2_name, norm_a, norm_b, start_norm, tournament, status),
                )
                match_id_to_canonical_id[mid_str] = cursor.lastrowid
            existing_keys.add(canonical_key)
            new_canonical_count += 1

    print(f"  New canonical_matches to insert: {new_canonical_count}")
    print(f"  Total matches with predictions: {len(match_id_to_canonical_id)}")
    if not args.dry_run:
        conn.commit()

    # --- Step 3: Insert fusion predictions into canonical_predictions ---
    print("\n=== Step 3: Insert fusion predictions ===")

    # Get existing predictions to avoid duplicates
    existing_preds = set()
    if not args.dry_run:
        rows = conn.execute(
            "SELECT canonical_match_id, model_name FROM canonical_predictions"
        ).fetchall()
        existing_preds = {(r[0], r[1]) for r in rows}
        print(f"  Existing predictions: {len(existing_preds)}")

    total_inserted = 0
    batch_size = 5000
    batch = []

    for mid_str, pred_data in all_predictions.items():
        if mid_str not in match_id_to_canonical_id:
            continue
        if match_id_to_canonical_id[mid_str] == -1:  # dry run
            continue

        canonical_id = match_id_to_canonical_id[mid_str]
        y_true = pred_data.get("y_true")
        date = pred_data.get("date", "")

        for model in FUSION_MODELS:
            pred_key = model["pred_key"]
            prob_a = pred_data.get(pred_key)
            if prob_a is None:
                continue

            prob_b = 1.0 - prob_a

            # Check for duplicate
            if not args.dry_run and (canonical_id, model["model_name"]) in existing_preds:
                continue

            diagnostics = json.dumps({
                "y_true": y_true,
                "date": date,
                "match_id": mid_str,
                "player_elo_prob": pred_data.get("player_elo"),
            })

            if args.dry_run:
                if total_inserted < 3:
                    print(f"  [DRY RUN] Would insert: match={mid_str}, model={model['model_name']}, prob_a={prob_a:.4f}")
                total_inserted += 1
                continue

            batch.append((
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

            if len(batch) >= batch_size:
                conn.executemany(
                    """INSERT INTO canonical_predictions 
                       (canonical_match_id, model_artifact_id, model_name, model_version,
                        prob_a, prob_b, features_version, ratings_version, data_cutoff_at,
                        prediction_status, diagnostics_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                total_inserted += len(batch)
                print(f"  Inserted {total_inserted} predictions so far...")
                batch = []

    if batch and not args.dry_run:
        conn.executemany(
            """INSERT INTO canonical_predictions 
               (canonical_match_id, model_artifact_id, model_name, model_version,
                prob_a, prob_b, features_version, ratings_version, data_cutoff_at,
                prediction_status, diagnostics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )
        total_inserted += len(batch)

    if not args.dry_run:
        conn.commit()

    print(f"\n  Total predictions inserted: {total_inserted}")

    # --- Verify ---
    print("\n=== Verification ===")
    if not args.dry_run:
        cm_count = conn.execute("SELECT COUNT(*) FROM canonical_matches").fetchone()[0]
        cp_count = conn.execute("SELECT COUNT(*) FROM canonical_predictions").fetchone()[0]
        ma_count = conn.execute("SELECT COUNT(*) FROM model_artifacts").fetchone()[0]
        print(f"  canonical_matches: {cm_count} rows")
        print(f"  canonical_predictions: {cp_count} rows")
        print(f"  model_artifacts: {ma_count} rows")

        # Per-model breakdown
        for model in FUSION_MODELS:
            count = conn.execute(
                "SELECT COUNT(*) FROM canonical_predictions WHERE model_name=?",
                (model["model_name"],),
            ).fetchone()[0]
            print(f"    {model['model_name']}: {count} predictions")

        # Sample predictions
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
            print(f"    {s[0]} vs {s[1]} ({s[2][:10]}) | {s[3]}: p_a={s[4]:.4f} p_b={s[5]:.4f}")
    else:
        print("  [DRY RUN] No verification - no data was written")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
