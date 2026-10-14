"""Run comprehensive ablation study on A1 architecture across the canonical 11,550 matches.

Ablations:
1. Full A1 (Team Attention + Global Meta Champions)
2. A1 without Global Meta Champions (Zero out meta)
3. A1 Mixture with Calibrated Glicko-2
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import json
import numpy as np
import pandas as pd
import torch

from src.models.a1.architecture import A1Model, A1Config
from src.models.a1.predictor import A1Predictor
from scripts.a1.train_a1 import load_research_bank
from scripts.a1.global_meta_aggregator import compute_14d_global_meta_matrix


def main():
    project_root = Path(__file__).resolve().parents[2]
    research_root = Path((project_root / "data/research_root.txt").read_text().strip())

    meta_df, context, players, history, mask, champions = load_research_bank(research_root)
    paired_df = pd.read_parquet(research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet")
    test_ids = set(paired_df["golgg_match_id"])
    meta_df["idx"] = np.arange(len(meta_df))
    test_mask = meta_df["golgg_match_id"].isin(test_ids)
    test_indices = meta_df.loc[test_mask, "idx"].values

    save_dir = project_root / "data/06_models/a1"
    config = A1Config()
    model = A1Model(config)
    model.load_state_dict(torch.load(save_dir / "a1_model_weights.pt", map_location="cpu"))
    model.eval()

    with open(save_dir / "calibration.json") as f:
        cal_params = json.load(f)

    test_meta = compute_14d_global_meta_matrix(meta_df, history, mask, champions, test_indices)

    ctx_t = context[test_indices]
    pr_t = players[test_indices]
    ph_t = history[test_indices]
    pm_t = mask[test_indices]
    pc_t = champions[test_indices]
    bo_t = meta_df.loc[test_indices, "best_of"].values.astype(int)
    y_test = meta_df.loc[test_indices, "y"].values.astype(float)

    predictor = A1Predictor(model, calibration_params=cal_params)

    # 1. Full A1
    p_full = predictor.predict_batch(ctx_t, pr_t, ph_t, pm_t, pc_t, test_meta, bo_t)
    ll_full = float(-np.mean(y_test * np.log(p_full) + (1 - y_test) * np.log(1 - p_full)))
    brier_full = float(np.mean((p_full - y_test) ** 2))
    acc_full = float(np.mean((p_full >= 0.5) == y_test))

    # 2. Ablation: No Meta Champions (Zero out meta)
    zero_meta = np.zeros_like(test_meta)
    p_no_meta = predictor.predict_batch(ctx_t, pr_t, ph_t, pm_t, pc_t, zero_meta, bo_t)
    ll_no_meta = float(-np.mean(y_test * np.log(p_no_meta) + (1 - y_test) * np.log(1 - p_no_meta)))
    brier_no_meta = float(np.mean((p_no_meta - y_test) ** 2))
    acc_no_meta = float(np.mean((p_no_meta >= 0.5) == y_test))

    # 3. Residual Mixture: A1 blended with Calibrated Glicko-2
    p_glicko = paired_df["p_glicko"].values
    p_a0 = paired_df["p_full"].values

    results = {
        "n_matches": len(test_indices),
        "full_a1": {"log_loss": ll_full, "brier": brier_full, "accuracy": acc_full},
        "no_meta_champions": {"log_loss": ll_no_meta, "brier": brier_no_meta, "accuracy": acc_no_meta, "delta_ll": ll_no_meta - ll_full},
        "glicko_mixture": {},
        "a0_mixture": {},
    }

    for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        p_blend = (1 - w) * p_glicko + w * p_full
        ll_blend = float(-np.mean(y_test * np.log(p_blend) + (1 - y_test) * np.log(1 - p_blend)))
        results["glicko_mixture"][f"w_a1_{w:.2f}"] = ll_blend

    for w in [0.0, 0.02, 0.05, 0.1, 0.2, 1.0]:
        p_blend_a0 = (1 - w) * p_a0 + w * p_full
        ll_blend_a0 = float(-np.mean(y_test * np.log(p_blend_a0) + (1 - y_test) * np.log(1 - p_blend_a0)))
        results["a0_mixture"][f"w_a1_{w:.2f}"] = ll_blend_a0

    out_file = project_root / "data/08_reporting/benchmark/a1_run_001/ablation_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved ablation results to {out_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
