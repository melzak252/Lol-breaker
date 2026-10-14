"""Run end-to-end A1 walkforward training, prediction generation, and candidate formatting."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.a1.architecture import A1Model, A1Config
from src.models.a1.predictor import A1Predictor
from scripts.a1.dataset_loader import A1Dataset
from scripts.a1.global_meta_aggregator import compute_14d_global_meta_matrix
from scripts.a1.train_a1 import load_research_bank, train_a1_model


def main():
    parser = argparse.ArgumentParser(description="A1 Training and Benchmark Pipeline")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # 1. Resolve research root
    research_root_file = PROJECT_ROOT / "data/research_root.txt"
    if not research_root_file.exists():
        raise FileNotFoundError("Missing data/research_root.txt")
    research_root = Path(research_root_file.read_text().strip())

    print(f"=== A1 End-to-End Walkforward Pipeline ===")
    print(f"Research Root: {research_root}")

    # 2. Load feature bank
    print("Loading feature bank...")
    meta_df, context, players, history, mask, champions = load_research_bank(research_root)
    n_total = len(meta_df)
    print(f"Loaded feature bank: {n_total} matches.")

    # 3. Load canonical benchmark targets (paired.parquet)
    paired_path = research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    paired_df = pd.read_parquet(paired_path)
    test_ids = set(paired_df["golgg_match_id"])
    print(f"Loaded canonical benchmark targets: {len(test_ids)} matches (2024–2026).")

    # Align bank indices with test_ids and train/cal split
    meta_df["idx"] = np.arange(len(meta_df))
    meta_df["date_dt"] = pd.to_datetime(meta_df["date"])

    # Partitions:
    # TEST: 2024-01-01 to 2026-09-15 (matching paired.parquet exactly)
    # CAL: 2023-01-01 to 2023-12-31
    # TRAIN: matches prior to 2023-01-01
    test_mask = meta_df["golgg_match_id"].isin(test_ids)
    test_indices = meta_df.loc[test_mask, "idx"].values

    cal_mask = (meta_df["date_dt"] >= "2023-01-01") & (meta_df["date_dt"] < "2024-01-01") & (~test_mask)
    cal_indices = meta_df.loc[cal_mask, "idx"].values

    # To ensure training stays fast while learning robust weights, sample recent train history (2020-2022)
    train_mask = (meta_df["date_dt"] >= "2020-01-01") & (meta_df["date_dt"] < "2023-01-01") & (~test_mask)
    train_indices = meta_df.loc[train_mask, "idx"].values

    print(f"Partition counts: TRAIN={len(train_indices)}, CAL={len(cal_indices)}, TEST={len(test_indices)}")
    assert len(test_indices) == 11550, f"Expected exactly 11,550 test matches, got {len(test_indices)}"

    # 4. Compute 14-day Global Meta Champion matrices
    print("Computing 14-day rolling global meta champion matrices for TRAIN, CAL, and TEST...")
    t0 = time.time()
    train_meta = compute_14d_global_meta_matrix(meta_df, history, mask, champions, train_indices)
    cal_meta = compute_14d_global_meta_matrix(meta_df, history, mask, champions, cal_indices)
    test_meta = compute_14d_global_meta_matrix(meta_df, history, mask, champions, test_indices)
    print(f"Meta champion matrices computed in {time.time() - t0:.1f}s.")

    # 5. Build PyTorch Datasets
    train_labels = meta_df.loc[train_indices, "y"].values.astype(np.float32)
    cal_labels = meta_df.loc[cal_indices, "y"].values.astype(np.float32)
    test_labels = meta_df.loc[test_indices, "y"].values.astype(np.float32)

    train_bo = meta_df.loc[train_indices, "best_of"].values.astype(int)
    cal_bo = meta_df.loc[cal_indices, "best_of"].values.astype(int)
    test_bo = meta_df.loc[test_indices, "best_of"].values.astype(int)

    train_dataset = A1Dataset(
        context=context[train_indices],
        player_ratings=players[train_indices],
        player_history=history[train_indices],
        player_mask=mask[train_indices],
        player_champions=champions[train_indices],
        meta_champions=train_meta,
        best_of=train_bo,
        labels=train_labels,
    )

    val_dataset = A1Dataset(
        context=context[cal_indices],
        player_ratings=players[cal_indices],
        player_history=history[cal_indices],
        player_mask=mask[cal_indices],
        player_champions=champions[cal_indices],
        meta_champions=cal_meta,
        best_of=cal_bo,
        labels=cal_labels,
    )

    test_dataset = A1Dataset(
        context=context[test_indices],
        player_ratings=players[test_indices],
        player_history=history[test_indices],
        player_mask=mask[test_indices],
        player_champions=champions[test_indices],
        meta_champions=test_meta,
        best_of=test_bo,
        labels=test_labels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 6. Train A1 Model
    config = A1Config()
    model = A1Model(config)
    trained_model, cal_params = train_a1_model(
        model=model,
        train_loader=train_loader,
        val_dataset=val_dataset,
        epochs=args.epochs,
        lr=args.lr,
        device="cpu",
    )

    # Save trained model weights
    save_dir = PROJECT_ROOT / "data/06_models/a1"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), save_dir / "a1_model_weights.pt")
    (save_dir / "calibration.json").write_text(json.dumps(cal_params, indent=2))
    print(f"Model saved to {save_dir}")

    # 7. Generate Test Predictions for Candidate Submission
    print("Generating calibrated test predictions for the 11,550 canonical matches...")
    predictor = A1Predictor(trained_model, calibration_params=cal_params)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            probs = predictor.predict_batch(
                context=batch["context"].numpy(),
                player_ratings=batch["player_ratings"].numpy(),
                player_history=batch["player_history"].numpy(),
                player_mask=batch["player_mask"].numpy(),
                player_champions=batch["player_champions"].numpy(),
                meta_champions=batch["meta_champions"].numpy(),
                best_of=batch["best_of"].numpy(),
            )
            all_preds.append(probs)
    test_preds = np.concatenate(all_preds)
    # Build Candidate DataFrame conforming strictly to scripts/run_model_benchmark.py requirements
    sub_df = pd.DataFrame({
        "golgg_match_id": meta_df.loc[test_indices, "golgg_match_id"].values,
        "p_a1": test_preds,
    })

    # Merge directly with paired_df columns to guarantee exact match alignment
    candidate_cols = ["golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y", "feature_source_max_day"]
    merged_sub = paired_df[candidate_cols].merge(sub_df, on="golgg_match_id", how="left")
    merged_sub["train_end"] = "2023-12-31"
    merged_sub["calibration_end"] = "2023-12-31"
    assert len(merged_sub) == 11550
    assert not merged_sub["p_a1"].isna().any(), "No missing predictions allowed!"

    out_dir = PROJECT_ROOT / "data/07_model_output/a1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "a1_canonical_predictions.parquet"
    merged_sub.to_parquet(out_file, index=False)
    print(f"Saved canonical candidate predictions ({len(merged_sub)} rows) to {out_file}")

    # Preliminary metrics on full cohort
    y_test = merged_sub["y"].values
    p_test = merged_sub["p_a1"].values
    ll = -np.mean(y_test * np.log(p_test) + (1.0 - y_test) * np.log(1.0 - p_test))
    brier = np.mean((p_test - y_test) ** 2)
    acc = np.mean((p_test >= 0.5) == y_test)
    print(f"\n=== A1 Preliminary Test Metrics (N = 11,550) ===")
    print(f"Log Loss:    {ll:.6f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"Accuracy:    {acc*100:.2f}%")


if __name__ == "__main__":
    main()
