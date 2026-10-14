#!/usr/bin/env python3
"""
Walk-Forward Pipeline for Fusion Model v2.
==========================================
Transformer yearly expanding windows + MLP/XGBoost walk-forward N=1000.

Comparison:
  - MLP  (FusionMLP:  256→128→64→1, BatchNorm+Dropout)
  - XGBoost (XGBClassifier: 100 trees, max_depth=6)

Usage:
  # Quick test (use existing embeddings, skip transformer):
  python walk_forward_pipeline.py --quick

  # Full pipeline (single year, to test):
  python walk_forward_pipeline.py --year 2025

  # Full pipeline (all years, overnight run):
  python walk_forward_pipeline.py

Author: code review + fix pipeline experiment
"""

import argparse
import json
import os
import sys
import time
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import xgboost as xgb

# ─── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/melzak/dev/inzynierka")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = DATA_DIR / "walk_forward_results"

BASELINE_CSV = DATA_DIR / "golgg_y_predicts.csv"
SEQUENCES_JSON = DATA_DIR / "transformer_team_sequences_v2.json"
EXISTING_EMBEDDINGS = DATA_DIR / "transformer_embeddings_v2.json"
EXISTING_TRANSFORMER = MODELS_DIR / "transformer_best.pt"

# ─── Hyperparams ──────────────────────────────────────────────────────────
N_CHUNK = 1000                # walk-forward chunk size
TRANSFORMER_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Transformer arch (MatchPredictor from team_transformer.py)
INPUT_DIM = 51
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.2

TRANSFORMER_EPOCHS = 30
TRANSFORMER_LR = 1e-4
TRANSFORMER_WD = 1e-5
TRANSFORMER_PATIENCE = 7  # early stopping

# MLP arch (FusionMLP)
MLP_EPOCHS = 50
MLP_LR = 1e-4
MLP_WD = 1e-5
MLP_PATIENCE = 7

BATCH_SIZE = 512
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"  [GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB]")


# ─── Model Definitions ────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """MLP matching original FusionModel classifier part."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Data Loading ─────────────────────────────────────────────────────────

def load_baseline():
    """Load and prepare baseline CSV."""
    print("→ Loading baseline CSV...")
    df = pd.read_csv(BASELINE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {len(df)} matches, {df.date.min().date()} → {df.date.max().date()}")
    return df


def load_sequences():
    """Load transformer sequences."""
    print("→ Loading sequences...")
    with open(SEQUENCES_JSON) as f:
        seqs = json.load(f)
    print(f"   {len(seqs)} sequences")
    return seqs


def load_embeddings(path, embed_dim=192):
    """Load pre-computed embeddings as DataFrame with expanded columns."""
    print(f"→ Loading embeddings from {path.name}...")
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["match_id"] = df["match_id"].astype(str)
    # Expand embedding array into columns
    embed_cols = [f"emb_{i}" for i in range(embed_dim)]
    emb_values = np.array([e["embedding"] for e in raw], dtype=np.float32)
    for i, col in enumerate(embed_cols):
        df[col] = emb_values[:, i]
    df = df.drop(columns=["embedding"])
    print(f"   {len(df)} embeddings, {embed_dim}-dim")
    return df, embed_cols


# ─── Transformer Training ─────────────────────────────────────────────────

class TransformerDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        t1 = torch.tensor(s["t1_seq"], dtype=torch.float)   # (15, 51)
        t2 = torch.tensor(s["t2_seq"], dtype=torch.float)
        y = torch.tensor([float(s["y"])], dtype=torch.float) # (1,)
        return t1, t2, y


def train_transformer(train_seqs, val_seqs):
    """Train MatchPredictor from scratch."""
    # Lazy import to avoid circular issues
    from betting_app.models.transformer.team_transformer import MatchPredictor

    model = MatchPredictor(
        input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
    ).to(DEVICE)

    train_ds = TransformerDataset(train_seqs)
    val_ds = TransformerDataset(val_seqs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRANSFORMER_LR,
                           weight_decay=TRANSFORMER_WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    print(f"   Train: {len(train_seqs)} | Val: {len(val_seqs)} | Device: {DEVICE}")

    for epoch in range(TRANSFORMER_EPOCHS):
        t0 = time.time()

        # ── Training ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for t1_batch, t2_batch, y_batch in train_loader:
            t1_batch = t1_batch.to(DEVICE, non_blocking=True)
            t2_batch = t2_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(t1_batch, t2_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(1, n_batches)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for t1_batch, t2_batch, y_batch in val_loader:
                t1_batch = t1_batch.to(DEVICE, non_blocking=True)
                t2_batch = t2_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)

                logits = model(t1_batch, t2_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                n_val_batches += 1

                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / max(1, n_val_batches)
        scheduler.step(avg_val_loss)

        # Val metrics (sample first 1000 or all)
        if len(all_probs) > 10:
            val_ll = log_loss(all_labels, np.clip(all_probs, 1e-15, 1-1e-15))
        else:
            val_ll = 0.693

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        print(f"   Epoch {epoch+1:2d}/{TRANSFORMER_EPOCHS} | "
              f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
              f"val_LL={val_ll:.4f} | [{elapsed:.1f}s]")

        if epochs_no_improve >= TRANSFORMER_PATIENCE:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


# ─── Embedding Extraction ─────────────────────────────────────────────────

def extract_embeddings(model, sequences, batch_size=256):
    """Extract 192-dim embeddings. Returns [{match_id, embedding}]."""
    print(f"→ Extracting embeddings for {len(sequences)} matches...")
    model.eval()

    results = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            seq_a_list, seq_b_list, mids = [], [], []

            for s in batch:
                seq_a_list.append(torch.tensor(s["t1_seq"], dtype=torch.float).unsqueeze(0))
                seq_b_list.append(torch.tensor(s["t2_seq"], dtype=torch.float).unsqueeze(0))
                mids.append(s["match_id"])

            seq_a = torch.cat(seq_a_list, dim=0).to(DEVICE)
            seq_b = torch.cat(seq_b_list, dim=0).to(DEVICE)

            embeddings = model.get_embedding(seq_a, seq_b)
            emb_np = embeddings.cpu().numpy()

            for j, mid in enumerate(mids):
                results.append({
                    "match_id": str(mid),
                    "embedding": emb_np[j].tolist(),
                })

            if (i // batch_size) % 20 == 0 and i > 0:
                print(f"   ... {i}/{len(sequences)} sequences processed")

    print(f"   Done: {len(results)} embeddings")
    return results


# ─── MLP Training / Prediction ────────────────────────────────────────────

class MatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlp(X_train, y_train):
    """Train FusionMLP, returns model + training time."""
    model = FusionMLP(X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)

    ds = MatchDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=True)

    t0 = time.time()
    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(MLP_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for Xb, yb in loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= MLP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    return model, train_time


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ─── Walk-Forward Core ────────────────────────────────────────────────────

def walk_forward(df_merged, embed_cols, year_boundary, n_chunk=1000):
    """
    Walk-forward MLP + XGBoost through the target year.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Merged baseline + embeddings for all matches.
    embed_cols : list[str]
        Embedding column names (emb_0 to emb_191).
    year_boundary : int
        Target year to predict (e.g. 2020 means predict matches from 2020).
    n_chunk : int
        Walk-forward chunk size.

    Returns
    -------
    pd.DataFrame with predictions per match.
    """
    target_year = year_boundary

    # Select features
    base_cols = [c for c in df_merged.columns
                 if c not in ("date", "year", "match_id", "golgg_match_id",
                              "golgg_match_id_str", "team1_name", "team2_name",
                              "y_true") and not c.startswith("emb_")]
    feature_cols = base_cols + embed_cols

    # Split: pre-target = train window, target = test window
    pre_target = df_merged[df_merged["year"] < target_year].copy()
    target_df = df_merged[df_merged["year"] == target_year].copy()
    target_df = target_df.sort_values("date").reset_index(drop=True)

    n_target = len(target_df)
    if n_target == 0:
        print(f"   No matches in year {target_year}, skipping.")
        return pd.DataFrame()

    n_chunks = max(1, math.ceil(n_target / n_chunk))
    print(f"   Pre-target: {len(pre_target)} | Target: {n_target} | "
          f"Chunks: {n_chunks} (N={n_chunk})")

    all_chunks = []
    chunk_sizes = []

    for ci in range(n_chunks):
        c_start = ci * n_chunk
        c_end = min(c_start + n_chunk, n_target)
        test_df = target_df.iloc[c_start:c_end]

        # Build training set: pre_target + already-seen target chunks
        if ci == 0:
            train_df = pre_target.copy()
        else:
            seen_target = target_df.iloc[:c_start]
            train_df = pd.concat([pre_target, seen_target], ignore_index=True)

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["y_true"].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["y_true"].values.astype(np.float32)

        n_train = len(X_train)
        n_test = len(X_test)

        if n_train < 200:
            print(f"   Chunk {ci+1}/{n_chunks}: SKIP ({n_train} train < 200)")
            continue

        # ── Standardize ──
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ── MLP ──
        t0 = time.time()
        mlp_model, mlp_train_time = train_mlp(X_train_s, y_train)
        mlp_preds = predict_mlp(mlp_model, X_test_s)
        mlp_total_time = time.time() - t0

        # Clip for metrics
        mlp_preds_c = np.clip(mlp_preds, 1e-15, 1 - 1e-15)

        try:
            mlp_ll = log_loss(y_test, mlp_preds_c)
            mlp_auc = roc_auc_score(y_test, mlp_preds) if len(np.unique(y_test)) > 1 else 0.5
            mlp_brier = brier_score_loss(y_test, mlp_preds)
        except Exception:
            mlp_ll = 0.5
            mlp_auc = 0.5
            mlp_brier = 0.25

        # ── XGBoost ──
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss",
        )
        xgb_model.fit(X_train_s, y_train)
        xgb_preds = xgb_model.predict_proba(X_test_s)[:, 1]
        xgb_total_time = time.time() - t0

        xgb_preds_c = np.clip(xgb_preds, 1e-15, 1 - 1e-15)

        try:
            xgb_ll = log_loss(y_test, xgb_preds_c)
            xgb_auc = roc_auc_score(y_test, xgb_preds) if len(np.unique(y_test)) > 1 else 0.5
            xgb_brier = brier_score_loss(y_test, xgb_preds)
        except Exception:
            xgb_ll = 0.5
            xgb_auc = 0.5
            xgb_brier = 0.25

        # ── Consensus ──
        consensus_pred = (mlp_preds + xgb_preds) / 2
        try:
            consensus_ll = log_loss(y_test, np.clip(consensus_pred, 1e-15, 1-1e-15))
        except Exception:
            consensus_ll = 0.5

        # Store
        result = test_df[["match_id", "date", "year", "team1_name", "team2_name", "y_true"]].copy()
        result["boundary_year"] = year_boundary
        result["chunk"] = ci
        result["chunk_total"] = n_chunks
        result["n_train"] = n_train
        result["mlp_pred"] = mlp_preds
        result["mlp_ll"] = mlp_ll
        result["mlp_auc"] = mlp_auc
        result["mlp_brier"] = mlp_brier
        result["xgb_pred"] = xgb_preds
        result["xgb_ll"] = xgb_ll
        result["xgb_auc"] = xgb_auc
        result["xgb_brier"] = xgb_brier
        result["consensus_pred"] = consensus_pred
        result["consensus_ll"] = consensus_ll

        all_chunks.append(result)
        chunk_sizes.append(n_test)

        print(f"   Chunk {ci+1:2d}/{n_chunks} ({n_test:4d} matches, "
              f"train={n_train:5d}) | "
              f"MLP LL={mlp_ll:.4f} AUC={mlp_auc:.4f} | "
              f"XGB LL={xgb_ll:.4f} AUC={xgb_auc:.4f} | "
              f"Con LL={consensus_ll:.4f} | "
              f"MLP {mlp_total_time:.0f}s XGB {xgb_total_time:.0f}s")

    if not all_chunks:
        return pd.DataFrame()

    results = pd.concat(all_chunks, ignore_index=True)

    # Overall metrics for this year
    overall_mlp_ll = log_loss(results["y_true"],
                              np.clip(results["mlp_pred"], 1e-15, 1-1e-15))
    overall_xgb_ll = log_loss(results["y_true"],
                              np.clip(results["xgb_pred"], 1e-15, 1-1e-15))
    overall_con_ll = log_loss(results["y_true"],
                              np.clip(results["consensus_pred"], 1e-15, 1-1e-15))

    print(f"\n   ═══════════════════════════════════════════════")
    print(f"   YEAR {target_year} RESULTS ({len(results)} matches)")
    print(f"   ═══════════════════════════════════════════════")
    print(f"   MLP       LL={overall_mlp_ll:.4f}  AUC={roc_auc_score(results['y_true'], results['mlp_pred']):.4f}")
    print(f"   XGBoost   LL={overall_xgb_ll:.4f}  AUC={roc_auc_score(results['y_true'], results['xgb_pred']):.4f}")
    print(f"   Consensus LL={overall_con_ll:.4f}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Use existing embeddings, skip transformer training")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year only (default: all years)")
    parser.add_argument("--n-chunk", type=int, default=1000,
                        help=f"Walk-forward chunk size (default: {N_CHUNK})")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR),
                        help=f"Output directory (default: {RESULTS_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_chunk = args.n_chunk

    print(f"\n{'='*60}")
    print("  WALK-FORWARD PIPELINE v2")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Load data ──
    df_baseline = load_baseline()
    sequences = load_sequences()

    years_to_process = [args.year] if args.year else TRANSFORMER_YEARS
    years_to_process = [y for y in years_to_process if y <= df_baseline["year"].max()]
    print(f"  Years to process: {years_to_process}")

    if not years_to_process:
        print("No valid years to process. Exiting.")
        return

    all_year_results = []

    for year in years_to_process:
        print(f"\n{'─'*60}")
        print(f"  YEAR BOUNDARY: {year}")
        print(f"{'─'*60}")

        # ── Transformer phase ──
        if args.quick and year == 2025:
            # For quick mode, only use existing embeddings for the current year
            # (existing transformer was trained on pre-2024)
            print("\n  → Quick mode: using existing embeddings")
            emb_path = EXISTING_EMBEDDINGS
            if not emb_path.exists():
                print(f"  ERROR: {emb_path} not found. Run without --quick first.")
                return
            emb_df, embed_cols = load_embeddings(emb_path)
            model_ckpt_path = None
        else:
            # Split sequences by year boundary
            train_seqs = [s for s in sequences
                          if int(s["date"][:4]) < year]
            val_seqs = [s for s in sequences
                        if int(s["date"][:4]) == year and
                        int(s["date"][5:7]) <= 3]  # Q1 of target year as validation

            if len(val_seqs) == 0:
                # Fallback: use last 10% of train as validation
                n_val = max(1, len(train_seqs) // 10)
                val_seqs = train_seqs[-n_val:]
                train_seqs = train_seqs[:-n_val]

            # Train Transformer
            print(f"\n  → Training Transformer (train={len(train_seqs)}, val={len(val_seqs)})")
            model = train_transformer(train_seqs, val_seqs)

            # Extract embeddings for ALL sequences
            all_seqs_for_emb = sequences  # extract for all matches, not just training
            embeddings = extract_embeddings(model, all_seqs_for_emb)

            # Save checkpoint
            ckpt_path = MODELS_DIR / f"transformer_{year}.pt"
            torch.save(model.state_dict(), ckpt_path)
            model_ckpt_path = str(ckpt_path)
            print(f"  → Model saved: {ckpt_path}")

            # Save embeddings
            emb_path = DATA_DIR / f"transformer_embeddings_{year}.json"
            with open(emb_path, "w") as f:
                json.dump(embeddings, f)
            print(f"  → Embeddings saved: {emb_path} ({len(embeddings)} matches)")

            # Load embeddings as DataFrame
            emb_df, embed_cols = load_embeddings(emb_path)

        # ── Merge phase ──
        print("\n  → Merging embeddings with baseline features...")
        df_baseline["_mid"] = df_baseline["golgg_match_id"].astype(str)
        df_merged = df_baseline.merge(emb_df, left_on="_mid", right_on="match_id",
                                      how="inner")
        print(f"     Merged: {len(df_merged)} / {len(df_baseline)} baseline "
              f"({len(emb_df)} embeddings)")

        if len(df_merged) == 0:
            print("     ERROR: No matches merged. Check match_id alignment.")
            continue

        # ── Walk-forward ──
        print(f"\n  → Walk-forward MLP + XGBoost (N={n_chunk})...")
        results = walk_forward(df_merged, embed_cols, year, n_chunk)

        if len(results) > 0:
            all_year_results.append(results)

            # Save yearly results
            yr_path = output_dir / f"predictions_{year}.csv"
            results.to_csv(yr_path, index=False)
            print(f"  → Saved: {yr_path}")

            # Save summary
            summary = {
                "year": year,
                "n_matches": len(results),
                "mlp_logloss": float(log_loss(results["y_true"],
                    np.clip(results["mlp_pred"], 1e-15, 1-1e-15))),
                "xgb_logloss": float(log_loss(results["y_true"],
                    np.clip(results["xgb_pred"], 1e-15, 1-1e-15))),
                "consensus_logloss": float(log_loss(results["y_true"],
                    np.clip(results["consensus_pred"], 1e-15, 1-1e-15))),
                "model_checkpoint": model_ckpt_path if not args.quick else None,
                "embeddings_file": str(emb_path),
                "device": DEVICE,
            }
            sum_path = output_dir / f"summary_{year}.json"
            with open(sum_path, "w") as f:
                json.dump(summary, f, indent=2)

    # ── Final summary ──
    if all_year_results:
        combined = pd.concat(all_year_results, ignore_index=True)
        combined.to_csv(output_dir / "all_predictions.csv", index=False)

        print(f"\n{'='*60}")
        print("  FINAL SUMMARY")
        print(f"{'='*60}")

        overall_summaries = []
        for year in years_to_process:
            yr = combined[combined["boundary_year"] == year]
            if len(yr) == 0:
                continue
            mlp_ll = log_loss(yr["y_true"], np.clip(yr["mlp_pred"], 1e-15, 1-1e-15))
            xgb_ll = log_loss(yr["y_true"], np.clip(yr["xgb_pred"], 1e-15, 1-1e-15))
            con_ll = log_loss(yr["y_true"], np.clip(yr["consensus_pred"], 1e-15, 1-1e-15))
            print(f"  {year}: {len(yr):5d} matches | "
                  f"MLP LL={mlp_ll:.4f} | XGB LL={xgb_ll:.4f} | Con LL={con_ll:.4f}")
            overall_summaries.append({
                "year": year, "n_matches": len(yr),
                "mlp_logloss": mlp_ll, "xgb_logloss": xgb_ll,
                "consensus_logloss": con_ll,
            })

        # Focus on 2026 predictions
        df_2026 = combined[combined["date"] >= "2026-01-01"].copy()
        if len(df_2026) > 0:
            print(f"\n  ── 2026 MATCHES ──")
            for b in sorted(df_2026["boundary_year"].unique()):
                yr26 = df_2026[df_2026["boundary_year"] == b]
                n26 = len(yr26)
                mlp26 = log_loss(yr26["y_true"], np.clip(yr26["mlp_pred"], 1e-15, 1-1e-15))
                xgb26 = log_loss(yr26["y_true"], np.clip(yr26["xgb_pred"], 1e-15, 1-1e-15))
                con26 = log_loss(yr26["y_true"], np.clip(yr26["consensus_pred"], 1e-15, 1-1e-15))
                print(f"     Embeddings from boundary {b}: {n26} matches | "
                      f"MLP={mlp26:.4f} | XGB={xgb26:.4f} | Con={con26:.4f}")

        # Save overall summary
        final_summary = {
            "device": DEVICE,
            "n_chunk": n_chunk,
            "quick_mode": args.quick,
            "years": overall_summaries,
        }
        with open(output_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)

        print(f"\n  All results → {output_dir}/")
        print()

    else:
        print("\nNo results produced. Check errors above.")


if __name__ == "__main__":
    main()
