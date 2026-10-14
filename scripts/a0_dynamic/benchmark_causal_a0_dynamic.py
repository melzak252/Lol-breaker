"""Benchmark Causal A0 with Inductive Dynamic Champion Embeddings injected into pro history attention.

Loads the pretrained DynamicChampionEncoder (trained via self-supervised sequence learning),
computes 32-dim dynamic embeddings for all champions, injects them into Causal A0's neural
history models, and evaluates on the locked 11,550 canonical test matches (2024–2026).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
research_root = Path((project_root / "data/research_root.txt").read_text().strip())

# 1. First import local models from our repository
from src.models.champion_encoder.model import DynamicChampionEncoder, ChampionEncoderConfig
from scripts.a1.train_a1 import load_research_bank

# 2. Now extend sys.path and sys.modules so research code can be unpickled
sys.path.insert(0, str(research_root / "a0-phase-walkforward-20260914/code"))
sys.path.insert(0, str(research_root / "nonlinear-history-20260912/code"))

res_src = str(research_root / "nonlinear-history-20260912/code/src")
if "src" in sys.modules:
    if res_src not in sys.modules["src"].__path__:
        sys.modules["src"].__path__.append(res_src)
if "src.models" in sys.modules:
    res_models = str(research_root / "nonlinear-history-20260912/code/src/models")
    if res_models not in sys.modules["src.models"].__path__:
        sys.modules["src.models"].__path__.append(res_models)

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.auxiliary_player import AuxiliaryResidual
from causal_predictor import CausalA0Predictor


def build_dynamic_champion_matrix(
    encoder: DynamicChampionEncoder,
    meta_df: pd.DataFrame,
    history: np.ndarray,
    mask: np.ndarray,
    champions: np.ndarray,
) -> np.ndarray:
    """Extract pro map streams and compute 32-dim dynamic embeddings for all 256 champion IDs."""
    champ_recent_maps = defaultdict(list)
    stat_scales = np.array(
        [10.0, 10.0, 10.0, 10.0, 500.0, 1000.0, 3.0, 1000.0, 30.0, 1000.0, 1.0, 1.0],
        dtype=np.float32,
    )

    for idx in range(len(meta_df) - 1, -1, -1):
        for side in (0, 1):
            for role_idx in range(5):
                m = mask[idx, side, role_idx]
                valid = m == 1
                if not np.any(valid):
                    continue
                c_ids = champions[idx, side, role_idx][valid]
                h_sub = history[idx, side, role_idx][valid]
                for c_id, h in zip(c_ids, h_sub):
                    c_id_int = int(c_id)
                    if c_id_int == 0 or len(champ_recent_maps[c_id_int]) >= 30:
                        continue
                    stats_raw = np.nan_to_num(h[:12].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    win = float(h[24]) if len(h) > 24 else 0.5
                    token_16 = np.zeros(16, dtype=np.float32)
                    token_16[:12] = np.clip(stats_raw / stat_scales, -5.0, 5.0)
                    token_16[12] = win
                    token_16[13] = float(role_idx) / 4.0
                    champ_recent_maps[c_id_int].append(token_16)

    dynamic_matrix = np.zeros((256, 32), dtype=np.float32)
    with torch.no_grad():
        for c_id, map_list in champ_recent_maps.items():
            seq_arr = np.zeros((1, 30, 16), dtype=np.float32)
            seq_mask = np.zeros((1, 30), dtype=bool)
            for j, t in enumerate(map_list[:30]):
                seq_arr[0, j] = t
                seq_mask[0, j] = True
            t_seq = torch.from_numpy(seq_arr)
            t_mask = torch.from_numpy(seq_mask)
            emb = encoder(t_seq, sequence_mask=t_mask)
            dynamic_matrix[c_id] = emb.cpu().numpy()[0]

    return dynamic_matrix


def patch_auxiliary_residual():
    """Patch AuxiliaryResidual.forward to inject dynamic champion embeddings."""
    def forward_with_dynamic_champions(self, x, best_of_index, return_aux=False):
        h = self.player(x["players"])
        n = h.shape[0]
        mask = x["mask"].reshape(-1, 16).bool()
        values = x["history"].reshape(-1, 16, 33)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        encoded = self.history_encoder(values)

        # Inject Dynamic Champion Embeddings if enabled
        if hasattr(self, "champion") and self.champion is not None and getattr(self, "has_champions", False):
            ids = torch.where(mask, x["champions"].reshape(-1, 16), 0)
            gamma = getattr(self, "champ_gamma", 0.02)
            encoded = encoded + gamma * self.champion(ids)

        encoded = torch.cat([self.empty.expand(len(encoded), -1, -1), encoded], 1)
        valid = torch.cat([torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask], 1)
        weight = self.attention(encoded).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        out = (encoded * weight[..., None]).sum(1)
        h = h + out.reshape(n, 2, 5, 32)
        h = torch.tanh(h)
        team = torch.cat([x["context"], h.mean(2), h.std(2, unbiased=False)], -1)
        utility = self.head(team)
        z = utility[torch.arange(n), 0, best_of_index] - utility[torch.arange(n), 1, best_of_index]
        return (z, self.auxiliary_head(h)) if return_aux else z

    AuxiliaryResidual.forward = forward_with_dynamic_champions


def main():
    parser = argparse.ArgumentParser(description="Inject Dynamic Champion Embeddings into Causal A0")
    parser.add_argument("--gamma", type=float, default=0.02, help="Scaling factor for dynamic champion embedding injection")
    args = parser.parse_args()

    print(f"=== Causal A0 with Inductive Dynamic Champion Embeddings ===")
    print(f"Injection scale gamma: {args.gamma}")

    # 1. Load pretrained dynamic champion encoder
    encoder_path = project_root / "data/06_models/champion_encoder/dynamic_champion_encoder.pt"
    encoder_cfg = ChampionEncoderConfig(map_feature_dim=16, latent_dim=32, max_history_maps=30)
    encoder = DynamicChampionEncoder(encoder_cfg)
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    encoder.eval()
    print("Pretrained Dynamic Champion Encoder loaded.")

    # 2. Load feature bank
    print("Loading feature bank...")
    meta_df, _, _, history, mask, champions = load_research_bank(research_root)
    bank_dir = research_root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    with np.load(bank_dir / "arrays.npz") as z:
        raw = {k: z[k] for k in z.files}
    hist_data = {
        "history": history,
        "mask": mask,
        "champions": champions,
        "pool": np.zeros((len(meta_df), 2, 5, 4), dtype=np.float32),
    }
    # 3. Build dynamic champion embedding matrix
    print("Building dynamic embedding matrix for all 256 champion slots...")
    dynamic_matrix = build_dynamic_champion_matrix(encoder, meta_df, history, mask, champions)
    dyn_tensor = torch.from_numpy(dynamic_matrix).float()

    # 4. Patch AuxiliaryResidual
    patch_auxiliary_residual()

    # 5. Load canonical benchmark targets
    paired_path = research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    paired_df = pd.read_parquet(paired_path)
    test_ids = set(paired_df["golgg_match_id"])

    meta_df["idx"] = np.arange(len(meta_df))
    meta_df["year"] = pd.to_datetime(meta_df["date"]).dt.year
    meta_df["best_of"] = meta_df["best_of"].astype(int)

    # 6. Evaluate Causal A0 with dynamic champion embeddings year by year
    print("Evaluating Causal A0 with dynamic champion embeddings across 2024, 2025, 2026...")
    predictions_map = {}

    for year in [2024, 2025, 2026]:
        year_mask = (meta_df["year"] == year) & (meta_df["golgg_match_id"].isin(test_ids))
        year_indices = meta_df.loc[year_mask, "idx"].values
        if len(year_indices) == 0:
            continue

        print(f"Processing Year {year}: {len(year_indices)} matches...")
        t0 = time.time()
        predictor = CausalA0Predictor(year)

        # Inject dynamic champion embedding into all 3 neural models
        for nn_model in predictor.nn:
            nn_model.network.champion = nn.Embedding.from_pretrained(dyn_tensor, freeze=True, padding_idx=0)
            nn_model.network.has_champions = True
            nn_model.network.champ_gamma = args.gamma

        # Generate predictions for year batch
        raw_year = {k: v[year_indices] for k, v in raw.items()}
        hist_year = {k: v[year_indices] for k, v in hist_data.items()}
        bo_year = meta_df.loc[year_indices, "best_of"].values

        # A0 predict returns (N, 3): columns are [full, no_w20, no_organization]
        year_preds_matrix = predictor.predict(raw_year, hist_year, bo_year)
        p_full_year = year_preds_matrix[:, 0]  # full mode

        for g_id, p_val in zip(meta_df.loc[year_indices, "golgg_match_id"], p_full_year):
            predictions_map[str(g_id)] = float(p_val)

        print(f"  Year {year} completed in {time.time() - t0:.1f}s.")

    # 7. Assemble Candidate Submission DataFrame
    print("Assembling canonical candidate predictions DataFrame...")
    sub_df = pd.DataFrame({
        "golgg_match_id": paired_df["golgg_match_id"].values,
        "team1_id": paired_df["team1_id"].values,
        "team2_id": paired_df["team2_id"].values,
        "date": paired_df["date"].values,
        "result_day": paired_df["result_day"].values,
        "best_of": paired_df["best_of"].values.astype(int),
        "y": paired_df["y"].values.astype(int),
        "feature_source_max_day": paired_df["feature_source_max_day"].values,
        "train_end": "2023-12-31",
        "calibration_end": "2023-12-31",
        "p_a0_dyn": [predictions_map[str(gid)] for gid in paired_df["golgg_match_id"]],
    })

    assert len(sub_df) == 11550, f"Expected 11,550 rows, got {len(sub_df)}"
    assert not sub_df["p_a0_dyn"].isna().any(), "Found NaN predictions!"

    out_dir = project_root / "data/07_model_output/a0_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "a0_dynamic_predictions.parquet"
    sub_df.to_parquet(out_file, index=False)
    print(f"Saved candidate predictions to {out_file}")

    # Preliminary comparison vs Original Causal A0
    y_test = sub_df["y"].values
    p_orig = paired_df["p_full"].values
    p_dyn = sub_df["p_a0_dyn"].values

    ll_orig = -np.mean(y_test * np.log(p_orig) + (1.0 - y_test) * np.log(1.0 - p_orig))
    ll_dyn = -np.mean(y_test * np.log(p_dyn) + (1.0 - y_test) * np.log(1.0 - p_dyn))
    brier_orig = np.mean((p_orig - y_test) ** 2)
    brier_dyn = np.mean((p_dyn - y_test) ** 2)
    acc_orig = np.mean((p_orig >= 0.5) == y_test)
    acc_dyn = np.mean((p_dyn >= 0.5) == y_test)

    print(f"\n=======================================================")
    print(f"HEAD-TO-HEAD: CAUSAL A0 vs CAUSAL A0 + DYNAMIC CHAMPIONS")
    print(f"=======================================================")
    print(f"Matches Evaluated: N = {len(sub_df)}")
    print(f"Original Causal A0:       Log Loss = {ll_orig:.6f} | Brier = {brier_orig:.6f} | Acc = {acc_orig*100:.2f}%")
    print(f"Causal A0 + Dynamic Meta: Log Loss = {ll_dyn:.6f} | Brier = {brier_dyn:.6f} | Acc = {acc_dyn*100:.2f}%")
    print(f"Delta Log Loss (Dyn - Orig): {ll_dyn - ll_orig:+.6f}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
