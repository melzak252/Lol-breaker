"""Train A1DynamicModel using pretrained DynamicChampionEncoder and run canonical benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.champion_encoder.model import DynamicChampionEncoder, ChampionEncoderConfig
from src.models.a1.architecture_dynamic import A1DynamicModel, A1DynamicConfig
from scripts.a1.train_a1 import load_research_bank


class A1DynamicDataset(Dataset):
    def __init__(self, context, player_ratings, player_history, player_mask, dynamic_champions, best_of, labels=None):
        self.context = torch.as_tensor(context, dtype=torch.float32)
        self.player_ratings = torch.as_tensor(player_ratings, dtype=torch.float32)
        self.player_history = torch.as_tensor(player_history, dtype=torch.float32)
        self.player_mask = torch.as_tensor(player_mask, dtype=torch.uint8)
        self.dynamic_champions = torch.as_tensor(dynamic_champions, dtype=torch.float32)
        self.best_of = torch.as_tensor(best_of, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        item = {
            "context": self.context[idx],
            "player_ratings": self.player_ratings[idx],
            "player_history": self.player_history[idx],
            "player_mask": self.player_mask[idx],
            "dynamic_champions": self.dynamic_champions[idx],
            "best_of": self.best_of[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def compute_dynamic_embeddings_for_matches(
    encoder: DynamicChampionEncoder,
    history_arr: np.ndarray,
    mask_arr: np.ndarray,
    target_indices: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Pass historical maps for each player's champion through the pretrained encoder."""
    encoder.eval()
    n_targets = len(target_indices)
    out_dyn = np.zeros((n_targets, 2, 5, 32), dtype=np.float32)

    stat_scales = np.array(
        [10.0, 10.0, 10.0, 10.0, 500.0, 1000.0, 3.0, 1000.0, 30.0, 1000.0, 1.0, 1.0],
        dtype=np.float32,
    )

    with torch.no_grad():
        for i, idx in enumerate(target_indices):
            for side in (0, 1):
                for role_idx in range(5):
                    m = mask_arr[idx, side, role_idx]
                    valid = m == 1
                    if not np.any(valid):
                        continue

                    h = history_arr[idx, side, role_idx][valid]
                    num_maps = len(h)
                    if num_maps == 0:
                        continue

                    # Construct (1, num_maps, 16) token sequence
                    seq_16 = np.zeros((1, num_maps, 16), dtype=np.float32)
                    seq_16[0, :, :12] = np.clip(np.nan_to_num(h[:, :12].astype(np.float32)) / stat_scales, -5.0, 5.0)
                    seq_16[0, :, 12] = h[:, 24] if h.shape[1] > 24 else 0.5
                    seq_16[0, :, 13] = float(role_idx) / 4.0

                    t_seq = torch.from_numpy(seq_16).to(device)
                    t_mask = torch.ones((1, num_maps), dtype=torch.bool, device=device)

                    emb = encoder(t_seq, sequence_mask=t_mask)  # (1, 32)
                    out_dyn[i, side, role_idx] = emb.cpu().numpy()[0]

    return np.nan_to_num(out_dyn, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    research_root = Path((project_root / "data/research_root.txt").read_text().strip())
    print("=== Training A1 Dynamic Model (Zero ID Embeddings) ===")
    meta_df, context, players, history, mask, champions = load_research_bank(research_root)

    # Load canonical benchmark targets
    paired_path = research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    paired_df = pd.read_parquet(paired_path)
    test_ids = set(paired_df["golgg_match_id"])

    meta_df["idx"] = np.arange(len(meta_df))
    meta_df["date_dt"] = pd.to_datetime(meta_df["date"])

    test_mask = meta_df["golgg_match_id"].isin(test_ids)
    test_indices = meta_df.loc[test_mask, "idx"].values

    cal_mask = (meta_df["date_dt"] >= "2023-01-01") & (meta_df["date_dt"] < "2024-01-01") & (~test_mask)
    cal_indices = meta_df.loc[cal_mask, "idx"].values

    train_mask = (meta_df["date_dt"] >= "2020-01-01") & (meta_df["date_dt"] < "2023-01-01") & (~test_mask)
    train_indices = meta_df.loc[train_mask, "idx"].values

    print(f"Partitions: TRAIN={len(train_indices)}, CAL={len(cal_indices)}, TEST={len(test_indices)}")

    # Load pretrained DynamicChampionEncoder
    encoder_path = project_root / "data/06_models/champion_encoder/dynamic_champion_encoder.pt"
    encoder_cfg = ChampionEncoderConfig(map_feature_dim=16, latent_dim=32)
    encoder = DynamicChampionEncoder(encoder_cfg)
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    encoder.eval()
    print("Pretrained Dynamic Champion Encoder loaded.")

    print("Computing dynamic champion embeddings via Transformer across partitions...")
    t0 = time.time()
    train_dyn = compute_dynamic_embeddings_for_matches(encoder, history, mask, train_indices)
    cal_dyn = compute_dynamic_embeddings_for_matches(encoder, history, mask, cal_indices)
    test_dyn = compute_dynamic_embeddings_for_matches(encoder, history, mask, test_indices)
    print(f"Dynamic embeddings computed in {time.time() - t0:.1f}s.")

    # Datasets
    train_labels = meta_df.loc[train_indices, "y"].values.astype(np.float32)
    cal_labels = meta_df.loc[cal_indices, "y"].values.astype(np.float32)
    test_labels = meta_df.loc[test_indices, "y"].values.astype(np.float32)

    train_bo = meta_df.loc[train_indices, "best_of"].values.astype(int)
    cal_bo = meta_df.loc[cal_indices, "best_of"].values.astype(int)
    test_bo = meta_df.loc[test_indices, "best_of"].values.astype(int)

    train_ds = A1DynamicDataset(context[train_indices], players[train_indices], history[train_indices], mask[train_indices], train_dyn, train_bo, train_labels)
    cal_ds = A1DynamicDataset(context[cal_indices], players[cal_indices], history[cal_indices], mask[cal_indices], cal_dyn, cal_bo, cal_labels)
    test_ds = A1DynamicDataset(context[test_indices], players[test_indices], history[test_indices], mask[test_indices], test_dyn, test_bo, test_labels)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(cal_ds, batch_size=256, shuffle=False)

    # Train A1DynamicModel
    config = A1DynamicConfig(dynamic_champ_dim=32, player_hidden_dim=64)
    model = A1DynamicModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCELoss()

    print("Starting A1DynamicModel training (6 epochs)...")
    for epoch in range(1, 7):
        model.train()
        losses = []
        t_ep = time.time()
        for b in train_loader:
            optimizer.zero_grad()
            pred = model(b["context"], b["player_ratings"], b["player_history"], b["player_mask"], b["dynamic_champions"], b["best_of"])
            loss = criterion(torch.clamp(pred, 1e-6, 1.0 - 1e-6), b["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            v_preds = []
            for b in val_loader:
                vp = model(b["context"], b["player_ratings"], b["player_history"], b["player_mask"], b["dynamic_champions"], b["best_of"])
                v_preds.append(vp.numpy())
            v_preds = np.concatenate(v_preds)
            v_loss = -np.mean(cal_labels * np.log(np.clip(v_preds, 1e-6, 1-1e-6)) + (1 - cal_labels) * np.log(np.clip(1 - v_preds, 1e-6, 1-1e-6)))
        print(f"Epoch {epoch}/6 | Train Loss: {np.mean(losses):.4f} | Val Loss: {v_loss:.4f} | Time: {time.time() - t_ep:.1f}s")

    # Fit Platt scaling
    print("Fitting Platt scaling calibration...")
    v_logits = np.log(np.clip(v_preds, 1e-6, 1-1e-6) / np.clip(1 - v_preds, 1e-6, 1-1e-6))
    def platt_obj(p):
        a, b = p
        prob = 1.0 / (1.0 + np.exp(-(a * v_logits + b)))
        prob = np.clip(prob, 1e-6, 1-1e-6)
        return -np.mean(cal_labels * np.log(prob) + (1 - cal_labels) * np.log(1 - prob))

    res = minimize(platt_obj, [1.0, 0.0], method="L-BFGS-B")
    a_cal, b_cal = float(res.x[0]), float(res.x[1])
    print(f"Platt params: a={a_cal:.4f}, b={b_cal:.4f}")

    # Generate test predictions
    print("Generating predictions for full 11,550 canonical test cohort...")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    raw_test_preds = []
    with torch.no_grad():
        for b in test_loader:
            p = model(b["context"], b["player_ratings"], b["player_history"], b["player_mask"], b["dynamic_champions"], b["best_of"])
            raw_test_preds.append(p.numpy())
    raw_test_preds = np.concatenate(raw_test_preds)

    # Calibrate
    test_logits = np.log(np.clip(raw_test_preds, 1e-6, 1-1e-6) / np.clip(1 - raw_test_preds, 1e-6, 1-1e-6))
    cal_test_preds = 1.0 / (1.0 + np.exp(-(a_cal * test_logits + b_cal)))
    cal_test_preds = np.clip(cal_test_preds, 1e-6, 1.0 - 1e-6)

    # Candidate DataFrame
    sub_df = pd.DataFrame({
        "golgg_match_id": meta_df.loc[test_indices, "golgg_match_id"].values,
        "p_a1_dyn": cal_test_preds,
    })
    candidate_cols = ["golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y", "feature_source_max_day"]
    merged_sub = paired_df[candidate_cols].merge(sub_df, on="golgg_match_id", how="left")
    merged_sub["train_end"] = "2023-12-31"
    merged_sub["calibration_end"] = "2023-12-31"

    out_dir = project_root / "data/07_model_output/a1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "a1_dynamic_predictions.parquet"
    merged_sub.to_parquet(out_file, index=False)
    print(f"Saved A1 Dynamic candidate predictions ({len(merged_sub)} rows) to {out_file}")

    y_test = merged_sub["y"].values
    p_test = merged_sub["p_a1_dyn"].values
    ll = -np.mean(y_test * np.log(p_test) + (1.0 - y_test) * np.log(1.0 - p_test))
    brier = np.mean((p_test - y_test) ** 2)
    acc = np.mean((p_test >= 0.5) == y_test)
    print(f"\n=== A1 Dynamic Preliminary Metrics (N = 11,550) ===")
    print(f"Log Loss:    {ll:.6f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"Accuracy:    {acc*100:.2f}%")


if __name__ == "__main__":
    main()
