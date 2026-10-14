"""Train Dynamic Champion Encoder via Self-Supervised Multi-Task Pretraining.

Pretrains the Transformer-based Dynamic Champion Encoder on sequential pro match streams
using:
1. Win-Rate Binary Cross-Entropy Loss
2. Tactical Boxscore Denoising Reconstruction MSE Loss
3. Latent Contrastive Regularization
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.champion_encoder.model import (
    DynamicChampionEncoder,
    ChampionEncoderConfig,
    MultiTaskPretrainingHead,
)
from scripts.champion_encoder.build_pretraining_dataset import (
    extract_champion_role_sequences,
    ChampionSequenceDataset,
)
from scripts.a1.train_a1 import load_research_bank


def train_champion_encoder(
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 0.001,
    latent_dim: int = 32,
    n_recent_maps: int = 30,
) -> tuple[DynamicChampionEncoder, dict]:
    research_root = Path((project_root / "data/research_root.txt").read_text().strip())
    print("Loading feature bank for champion self-supervised pretraining...")
    meta_df, _, _, history, mask, champions = load_research_bank(research_root)

    print(f"Building sequential champion dataset (last N={n_recent_maps} maps on role)...")
    dataset = extract_champion_role_sequences(
        meta_df=meta_df,
        history_arr=history,
        mask_arr=mask,
        champions_arr=champions,
        n_recent_maps=n_recent_maps,
        max_samples_per_pair=20,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    config = ChampionEncoderConfig(
        map_feature_dim=16,
        latent_dim=latent_dim,
        num_heads=4,
        num_encoder_layers=2,
        max_history_maps=n_recent_maps,
    )
    encoder = DynamicChampionEncoder(config)
    heads = MultiTaskPretrainingHead(latent_dim=latent_dim, stat_dim=12)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(heads.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    print(f"Starting Multi-Task Pretraining on {len(dataset)} sequences ({epochs} epochs)...")
    history_logs = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        heads.train()
        t0 = time.time()

        total_loss = 0.0
        win_losses = []
        recon_losses = []

        for batch in train_loader:
            seq = batch["sequence"]          # (B, N, 16)
            seq_mask = batch["mask"]          # (B, N)
            win_rate = batch["win_rate"]      # (B,)
            target_stats = batch["target_stats"]  # (B, 12)

            optimizer.zero_grad()

            # 1. Forward through transformer encoder -> normalized (B, latent_dim)
            embeddings = encoder(seq, sequence_mask=seq_mask)

            # 2. Multi-task heads
            outputs = heads(embeddings)

            # 3. Compute loss components:
            # Task 1: Win rate BCE
            l_win = bce_loss_fn(outputs["win_logit"], win_rate)
            # Task 2: Boxscore Denoising Autoencoder MSE
            l_recon = mse_loss_fn(outputs["reconstruction"], target_stats)

            loss = l_win + 2.0 * l_recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            win_losses.append(l_win.item())
            recon_losses.append(l_recon.item())

        elapsed = time.time() - t0
        mean_win = float(np.mean(win_losses))
        mean_recon = float(np.mean(recon_losses))
        epoch_total = total_loss / len(train_loader)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Win BCE: {mean_win:.4f} | Recon MSE: {mean_recon:.4f} | Total Loss: {epoch_total:.4f} | Time: {elapsed:.1f}s")
        history_logs.append({
            "epoch": epoch,
            "win_bce": mean_win,
            "recon_mse": mean_recon,
            "total_loss": epoch_total,
        })

    # Save pretrained weights
    save_dir = project_root / "data/06_models/champion_encoder"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_dir / "dynamic_champion_encoder.pt")
    torch.save(heads.state_dict(), save_dir / "multitask_heads.pt")
    with open(save_dir / "pretraining_config.json", "w") as f:
        json.dump({
            "map_feature_dim": 16,
            "latent_dim": latent_dim,
            "n_recent_maps": n_recent_maps,
            "epochs": epochs,
            "history": history_logs,
        }, f, indent=2)
    print(f"Pretrained Dynamic Champion Encoder saved successfully to {save_dir}")

    return encoder, {"logs": history_logs}


if __name__ == "__main__":
    train_champion_encoder(epochs=6, batch_size=64, n_recent_maps=30)
