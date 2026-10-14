"""Validate that thesis model artefacts still reproduce expected metrics.

This is a stability/sanity check for the historical EXP-039 prediction stream and
the model-market hybrid idea from EXP-032/033/041.  It does not retrain the model;
it verifies the saved final common sample and recomputes core metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from betting_app.core.config import PROJECT_ROOT
from betting_app.services.upcoming_inference_service import apply_temperature_probability


FINAL_SAMPLE = PROJECT_ROOT / "docs/assets/final_symmetric_calibrated_market_comparison/final_symmetric_calibrated_market_common_sample.csv"
FINAL_MODEL_COL = "prob_sym_cal_lr_elasticnet_w20_binomial"
TARGET = "y_true"
EXPECTED = {
    "final_logloss": 0.585376,
    "final_auc": 0.755055,
    "market_open_logloss": 0.604656,
    "market_close_logloss": 0.597779,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-path", type=Path, default=FINAL_SAMPLE)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    args = parser.parse_args()

    data = pd.read_csv(args.sample_path)
    required = [TARGET, FINAL_MODEL_COL, "market_open", "market_close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(f"Missing columns in {args.sample_path}: {missing}")
    data = data.dropna(subset=required).copy()

    metrics = []
    metrics.append(evaluate(data, "Sym-Cal LR-ElasticNet-W20-Binomial", FINAL_MODEL_COL))
    metrics.append(evaluate(data, "Market Open", "market_open"))
    metrics.append(evaluate(data, "Market Close", "market_close"))

    model_t = np.array([apply_temperature_probability(float(p), args.temperature) for p in data[FINAL_MODEL_COL]])
    data["hybrid"] = args.alpha * model_t + (1.0 - args.alpha) * data["market_open"].to_numpy(dtype=float)
    metrics.append(evaluate(data, f"Hybrid a={args.alpha:.2f} T={args.temperature:.2f}", "hybrid"))

    frame = pd.DataFrame(metrics).sort_values("logloss")
    print(f"Sample: {args.sample_path}")
    print(f"Rows: {len(data)}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    checks = {
        "final_logloss": float(frame.loc[frame["model"] == "Sym-Cal LR-ElasticNet-W20-Binomial", "logloss"].iloc[0]),
        "final_auc": float(frame.loc[frame["model"] == "Sym-Cal LR-ElasticNet-W20-Binomial", "auc"].iloc[0]),
        "market_open_logloss": float(frame.loc[frame["model"] == "Market Open", "logloss"].iloc[0]),
        "market_close_logloss": float(frame.loc[frame["model"] == "Market Close", "logloss"].iloc[0]),
    }
    failures = []
    for key, expected in EXPECTED.items():
        observed = checks[key]
        if abs(observed - expected) > args.tolerance:
            failures.append(f"{key}: observed={observed:.6f}, expected={expected:.6f}")
    if failures:
        raise SystemExit("Metric drift detected:\n" + "\n".join(failures))
    print("OK: EXP-039 historical metrics are within tolerance.")


def evaluate(data: pd.DataFrame, model: str, column: str) -> dict[str, float | str | int]:
    y_true = data[TARGET].astype(int).to_numpy()
    y_prob = np.clip(data[column].to_numpy(dtype=float), 0.001, 0.999)
    return {
        "model": model,
        "n": int(len(data)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


if __name__ == "__main__":
    main()
