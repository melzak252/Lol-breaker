"""Sweep EXP-039/market hybrid alpha values on EXP-060 common sample.

The hybrid formula mirrors production thesis hybrid semantics:

    p_hybrid = alpha * temperature(exp039_probability, T) + (1 - alpha) * market_probability

where alpha is the model weight and market_probability is one of opening/mid/close
no-vig market probabilities from EXP-060.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


DEFAULT_INPUT = Path("reports/exp039_db_market_backtest_v2/exp039_market_common.csv")
DEFAULT_OUTPUT_DIR = Path("reports/exp039_alpha_sweep")


def _clip(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = _clip(prob)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logit / temperature))


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    p = _clip(prob)
    return {
        "n": int(len(y_true)),
        "logloss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "auc": float(roc_auc_score(y_true, p)) if len(set(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, p >= 0.5)),
        "mean_prob": float(np.mean(p)),
        "target_rate": float(np.mean(y_true)),
    }


def run_sweep(
    input_path: Path,
    output_dir: Path,
    temperature: float,
    alpha_step: float,
) -> dict[str, Any]:
    df = pd.read_csv(input_path)
    required = {"y_team_a", "exp039_prob_team_a"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    y = df["y_team_a"].astype(int).to_numpy()
    model_raw = _clip(df["exp039_prob_team_a"].to_numpy())
    model_t = apply_temperature(model_raw, temperature)

    alpha_grid = [round(float(x), 4) for x in np.arange(0.0, 1.0 + alpha_step / 2.0, alpha_step)]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "temperature": temperature,
        "alpha_step": alpha_step,
        "n_rows": int(len(df)),
        "formula": "p_hybrid = alpha * temperature(exp039_probability, T) + (1-alpha) * market_probability",
        "alpha_definition": "alpha is EXP-039 model weight; 0.0 = market only, 1.0 = temperature-scaled EXP-039 only",
        "exp039_raw": compute_metrics(y, model_raw),
        "exp039_temperature_scaled": compute_metrics(y, model_t),
        "by_market_reference": {},
    }

    for market_key in ["open", "mid", "close"]:
        market_col = f"market_{market_key}_p_a_novig"
        if market_col not in df.columns:
            raise ValueError(f"Missing market column: {market_col}")
        sub = df[["y_team_a", "exp039_prob_team_a", market_col]].dropna().copy()
        y_sub = sub["y_team_a"].astype(int).to_numpy()
        market = _clip(sub[market_col].to_numpy())
        model_sub = apply_temperature(_clip(sub["exp039_prob_team_a"].to_numpy()), temperature)

        best_logloss: dict[str, Any] | None = None
        best_brier: dict[str, Any] | None = None
        selected: dict[str, dict[str, Any]] = {}
        selected_alphas = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.75, 1.0]

        for alpha in alpha_grid:
            p_hybrid = alpha * model_sub + (1.0 - alpha) * market
            metric = compute_metrics(y_sub, p_hybrid)
            row = {"market_reference": market_key, "alpha": alpha, **metric}
            rows.append(row)
            if best_logloss is None or metric["logloss"] < best_logloss["logloss"]:
                best_logloss = row
            if best_brier is None or metric["brier"] < best_brier["brier"]:
                best_brier = row

        for alpha in selected_alphas:
            p_hybrid = alpha * model_sub + (1.0 - alpha) * market
            selected[f"{alpha:.2f}"] = compute_metrics(y_sub, p_hybrid)

        summary["by_market_reference"][market_key] = {
            "market_only": compute_metrics(y_sub, market),
            "model_only_temperature_scaled": compute_metrics(y_sub, model_sub),
            "best_alpha_by_logloss": best_logloss,
            "best_alpha_by_brier": best_brier,
            "selected_alphas": selected,
        }

    pd.DataFrame(rows).to_csv(output_dir / "alpha_sweep.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    args = parser.parse_args()
    summary = run_sweep(args.input, args.output_dir, args.temperature, args.alpha_step)
    for market, data in summary["by_market_reference"].items():
        best = data["best_alpha_by_logloss"]
        market_only = data["market_only"]
        print(
            f"{market:>5}: market LL={market_only['logloss']:.6f} | "
            f"best alpha={best['alpha']:.2f} LL={best['logloss']:.6f} "
            f"AUC={best['auc']:.6f}"
        )


if __name__ == "__main__":
    main()
