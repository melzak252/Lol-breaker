"""Evaluate learned dynamic champion embedding geometry, cosine clustering, and reconstruction."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.champion_encoder.model import (
    DynamicChampionEncoder,
    ChampionEncoderConfig,
    MultiTaskPretrainingHead,
)
from scripts.champion_encoder.build_pretraining_dataset import extract_champion_role_sequences
from scripts.a1.train_a1 import load_research_bank


def evaluate_embeddings():
    research_root = Path((project_root / "data/research_root.txt").read_text().strip())
    meta_df, _, _, history, mask, champions = load_research_bank(research_root)

    print("Extracting test sequences for evaluation...")
    dataset = extract_champion_role_sequences(
        meta_df=meta_df,
        history_arr=history,
        mask_arr=mask,
        champions_arr=champions,
        n_recent_maps=30,
        max_samples_per_pair=10,
    )

    save_dir = project_root / "data/06_models/champion_encoder"
    config = ChampionEncoderConfig(map_feature_dim=16, latent_dim=32, max_history_maps=30)
    encoder = DynamicChampionEncoder(config)
    encoder.load_state_dict(torch.load(save_dir / "dynamic_champion_encoder.pt", map_location="cpu"))
    encoder.eval()

    heads = MultiTaskPretrainingHead(latent_dim=32, stat_dim=12)
    heads.load_state_dict(torch.load(save_dir / "multitask_heads.pt", map_location="cpu"))
    heads.eval()

    print("Computing latent embeddings and reconstructions across test sequences...")
    all_embeddings = []
    all_recon_errors = []
    all_roles = []

    with torch.no_grad():
        for i in range(min(len(dataset), 2000)):
            item = dataset[i]
            seq = item["sequence"].unsqueeze(0)
            m = item["mask"].unsqueeze(0)
            target_stat = item["target_stats"].unsqueeze(0)
            role = item["role"].item()

            emb = encoder(seq, sequence_mask=m)
            out = heads(emb)

            recon_err = F.mse_loss(out["reconstruction"], target_stat).item()

            all_embeddings.append(emb.squeeze(0).numpy())
            all_recon_errors.append(recon_err)
            all_roles.append(role)

    embeddings_arr = np.array(all_embeddings)
    roles_arr = np.array(all_roles)
    mean_recon_mse = float(np.mean(all_recon_errors))

    # Compute silhouette score by competitive role (Top, Jgl, Mid, Bot, Sup)
    sil_score = float(silhouette_score(embeddings_arr, roles_arr, metric="cosine"))

    # Compute inter-role vs intra-role cosine similarity
    role_centroids = []
    for r in range(5):
        r_mask = roles_arr == r
        if np.any(r_mask):
            c = np.mean(embeddings_arr[r_mask], axis=0)
            c /= np.linalg.norm(c)
            role_centroids.append(c)
        else:
            role_centroids.append(np.zeros(32))

    role_centroids = np.array(role_centroids)
    centroid_sim_matrix = (role_centroids @ role_centroids.T).tolist()

    report = {
        "num_evaluated_sequences": len(embeddings_arr),
        "embedding_dimension": 32,
        "boxscore_reconstruction_mse": mean_recon_mse,
        "cosine_silhouette_score_by_role": sil_score,
        "role_centroid_cosine_similarity": centroid_sim_matrix,
    }

    out_file = project_root / "data/06_models/champion_encoder/evaluation_geometry.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== Dynamic Champion Embedding Evaluation ===")
    print(f"Mean Boxscore Reconstruction MSE: {mean_recon_mse:.6f}")
    print(f"Cosine Silhouette Score by Role:   {sil_score:.4f} (Positive = clean role clustering)")
    print(f"Role Centroid Cosine Matrix:\n{np.array(centroid_sim_matrix).round(3)}")


if __name__ == "__main__":
    evaluate_embeddings()
