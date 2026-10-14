"""Train A1 models on historical train partitions with walkforward validation."""

from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.a1.architecture import A1Model
from scripts.a1.dataset_loader import A1Dataset


def load_research_bank(research_root: Path):
    bank_dir = research_root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    meta_df = pd.read_parquet(bank_dir / "metadata.parquet")

    with np.load(bank_dir / "arrays.npz") as z:
        players = z["players"]  # (40636, 2, 5, 21)
        team = z["team"]        # (40636, 2, 11)
        w20 = z["w20"]          # (40636, 2, 10)
        gates = z["gates"]      # (40636, 2, 2)
        core = z["core"]        # (40636, 2, 5)

    # Construct a strictly side-signed context: team, W20, gates, and core.
    team_diff = team[:, 0] - team[:, 1]  # (N, 11)
    w20_diff = w20[:, 0] - w20[:, 1]  # (N, 10)
    gate_diff = gates[:, 0] - gates[:, 1]  # (N, 2)
    core_diff = core[:, 0] - core[:, 1]  # (N, 5)
    context = np.concatenate(
        [team_diff, w20_diff, gate_diff, core_diff], axis=1
    )  # (N, 28)

    history = np.load(bank_dir / "history.npy", mmap_mode="r")
    mask = np.load(bank_dir / "mask.npy", mmap_mode="r")
    champions = np.load(bank_dir / "champions.npy", mmap_mode="r")

    return meta_df, context, players, history, mask, champions


def train_a1_model(
    model: A1Model,
    train_loader: DataLoader,
    val_dataset: A1Dataset,
    epochs: int = 8,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    device: str = "cpu",
) -> tuple[A1Model, dict]:
    """Train A1 model using AdamW and Binary Cross-Entropy."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    print(f"Starting A1 training on {device} ({epochs} epochs, lr={lr})...")
    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        t0 = time.time()

        for batch in train_loader:
            ctx = batch["context"].to(device)
            pr = batch["player_ratings"].to(device)
            ph = batch["player_history"].to(device)
            pm = batch["player_mask"].to(device)
            pc = batch["player_champions"].to(device)
            mc = batch["meta_champions"].to(device)
            bo = batch["best_of"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            pred = model(ctx, pr, ph, pm, pc, mc, bo)
            pred_clamped = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
            loss = criterion(pred_clamped, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            for batch in val_loader:
                ctx = batch["context"].to(device)
                pr = batch["player_ratings"].to(device)
                ph = batch["player_history"].to(device)
                pm = batch["player_mask"].to(device)
                pc = batch["player_champions"].to(device)
                mc = batch["meta_champions"].to(device)
                bo = batch["best_of"].to(device)
                y = batch["label"].to(device)
                pred = model(ctx, pr, ph, pm, pc, mc, bo)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())

            val_preds_arr = np.concatenate(val_preds)
            val_targets_arr = np.concatenate(val_targets)
            val_loss = float(-np.mean(val_targets_arr * np.log(np.clip(val_preds_arr, 1e-6, 1.0 - 1e-6)) + (1.0 - val_targets_arr) * np.log(np.clip(1.0 - val_preds_arr, 1e-6, 1.0 - 1e-6))))

        elapsed = time.time() - t0
        mean_tr_loss = np.mean(train_losses)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {mean_tr_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)

    # Recompute validation predictions from the restored best checkpoint.
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        for batch in val_loader:
            ctx = batch["context"].to(device)
            pr = batch["player_ratings"].to(device)
            ph = batch["player_history"].to(device)
            pm = batch["player_mask"].to(device)
            pc = batch["player_champions"].to(device)
            mc = batch["meta_champions"].to(device)
            bo = batch["best_of"].to(device)
            y = batch["label"].to(device)
            pred = model(ctx, pr, ph, pm, pc, mc, bo)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
        val_preds_arr = np.concatenate(val_preds)
        val_targets_arr = np.concatenate(val_targets)

    # Fit symmetry-preserving positive slope calibration on the selected model.
    print("Fitting positive-slope calibration on validation partition...")
    val_preds_clipped = np.clip(val_preds_arr, 1e-6, 1.0 - 1e-6)
    val_logits = np.log(val_preds_clipped / (1.0 - val_preds_clipped))

    def platt_obj(params):
        a = params[0]
        p = 1.0 / (1.0 + np.exp(-a * val_logits))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -np.mean(val_targets_arr * np.log(p) + (1.0 - val_targets_arr) * np.log(1.0 - p))

    res = minimize(platt_obj, [1.0], method="L-BFGS-B", bounds=[(1e-8, None)])
    if not res.success or not np.isfinite(res.x[0]) or res.x[0] <= 0.0:
        raise RuntimeError(f"Positive-slope calibration failed: {res.message}")
    cal_params = {"platt_a": float(res.x[0]), "platt_b": 0.0}
    print(f"Calibrated slope-only parameters: {cal_params}")

    return model, cal_params
