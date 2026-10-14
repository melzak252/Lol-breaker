import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm


def odds_to_prob(o1: float, o2: float) -> float:
    """Convert two decimal odds into margin-normalized probability for team 1.

    Args:
        o1: Decimal odds for team 1.
        o2: Decimal odds for team 2.

    Returns:
        Margin-normalized implied probability for team 1, or NaN for invalid odds.
    """
    if pd.isnull(o1) or pd.isnull(o2) or o1 <= 1 or o2 <= 1:
        return np.nan

    margin = 1 / o1 + 1 / o2
    return (1 / o1) / margin


def optimize_alpha_historical(train_df: pd.DataFrame) -> float:
    """Select hybrid alpha using only historical expanding-window validation.

    Args:
        train_df: Historical rows containing model probability, market probability,
            and binary target.

    Returns:
        Alpha weight for the metamodel in the hybrid blend.
    """
    if len(train_df) < 100:
        return 0.5

    n_splits = min(5, max(2, len(train_df) // 500))
    alphas = np.linspace(0.0, 1.0, 51)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    alpha_scores = []

    for alpha in alphas:
        fold_losses = []
        for _, val_idx in tscv.split(train_df):
            val_df = train_df.iloc[val_idx]
            y_val = val_df["y_true"]
            p_val = (
                alpha * val_df["metamodel_lgbm_calibrated"]
                + (1 - alpha) * val_df["prob_avg_open"]
            )
            p_val = np.clip(p_val, 0.01, 0.99)
            fold_losses.append(log_loss(y_val, p_val))

        alpha_scores.append((float(np.mean(fold_losses)), float(alpha)))

    return min(alpha_scores, key=lambda item: item[0])[1]


def generate_hybrid_oof(train_df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Generate historical OOF-like hybrid probabilities for calibration.

    Since the hybrid is a deterministic blend rather than a fitted estimator, the
    OOF restriction is implemented by calibrating only on validation folds that
    occur after each fold's training window. The first historical segment remains
    NaN and is excluded from fitting the calibrator.

    Args:
        train_df: Historical rows sorted chronologically.
        alpha: Metamodel weight in the hybrid blend.

    Returns:
        Array of fold-valid hybrid probabilities; early unavailable rows are NaN.
    """
    n_splits = min(5, max(2, len(train_df) // 500))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.full(len(train_df), np.nan, dtype=float)

    for _, val_idx in tscv.split(train_df):
        val_df = train_df.iloc[val_idx]
        oof_preds[val_idx] = (
            alpha * val_df["metamodel_lgbm_calibrated"]
            + (1 - alpha) * val_df["prob_avg_open"]
        )

    return np.clip(oof_preds, 0.01, 0.99)


def fit_historical_calibrator(train_df: pd.DataFrame, alpha: float) -> IsotonicRegression | None:
    """Fit an isotonic calibrator using only historical fold-validation rows.

    Args:
        train_df: Historical rows sorted chronologically.
        alpha: Metamodel weight in the hybrid blend.

    Returns:
        Fitted isotonic calibrator, or None when insufficient data is available.
    """
    if len(train_df) < 100:
        return None

    oof_preds = generate_hybrid_oof(train_df, alpha)
    valid_oof = ~np.isnan(oof_preds)

    if valid_oof.sum() < 50:
        return None

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds[valid_oof], train_df["y_true"].iloc[valid_oof])
    return calibrator


def main() -> None:
    """Train and save a leakage-safe walk-forward hybrid model."""
    print("Running 07_train_hybrid_model.py (leakage-safe walk-forward)...")
    os.makedirs("docs/assets", exist_ok=True)

    print("Loading data...")
    odds_df = pd.read_csv("data/odds.csv")
    stacking_df = pd.read_csv("data/golgg_stacking_results.csv")

    odds_df["golgg_match_id"] = odds_df["golgg_match_id"].astype(str)
    stacking_df["golgg_match_id"] = stacking_df["golgg_match_id"].astype(str)

    df = pd.merge(odds_df, stacking_df, on="golgg_match_id", suffixes=("", "_stack"))
    df = df[df["t1_win"] | df["t2_win"]].copy()
    df["y_true"] = df["t1_win"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Opening odds are the only market signal available at bet-decision time.
    df["prob_avg_open"] = df.apply(
        lambda row: odds_to_prob(row["avg_open_home"], row["avg_open_away"]),
        axis=1,
    )

    features_stage3 = ["metamodel_lgbm_calibrated", "prob_avg_open"]
    target = "y_true"
    df_clean = df.dropna(subset=features_stage3 + [target]).copy()

    initial_train_mask = df_clean["date"] < datetime(2020, 1, 1)
    train_df = df_clean[initial_train_mask].copy()
    test_pool_df = df_clean[~initial_train_mask].copy()

    update_interval = 1000
    result_chunks = []

    print(f"Starting hybrid walk-forward training (interval: {update_interval} matches)...")
    for i in tqdm(range(0, len(test_pool_df), update_interval), desc="Hybrid Walk-Forward"):
        test_chunk = test_pool_df.iloc[i : i + update_interval].copy()
        if test_chunk.empty:
            break

        alpha = optimize_alpha_historical(train_df)
        calibrator = fit_historical_calibrator(train_df, alpha)

        raw_preds = (
            alpha * test_chunk["metamodel_lgbm_calibrated"]
            + (1 - alpha) * test_chunk["prob_avg_open"]
        )
        raw_preds = np.clip(raw_preds.to_numpy(), 0.01, 0.99)

        if calibrator is not None:
            calibrated_preds = calibrator.transform(raw_preds)
        else:
            calibrated_preds = raw_preds

        test_chunk["final_hybrid_raw"] = raw_preds
        test_chunk["final_hybrid_prob"] = np.clip(calibrated_preds, 0.01, 0.99)
        test_chunk["hybrid_alpha"] = alpha
        result_chunks.append(test_chunk)

        # Only after predictions are made do these rows become historical data.
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    if not result_chunks:
        raise ValueError("No walk-forward hybrid predictions were generated.")

    results_df = pd.concat(result_chunks, ignore_index=True)

    y_true = results_df[target].to_numpy()
    preds = results_df["final_hybrid_prob"].to_numpy()

    print("\n=== STAGE 3 (LEAKAGE-SAFE HYBRID) RESULTS ===")
    print(f"Log Loss:    {log_loss(y_true, preds):.5f}")
    print(f"AUC:         {roc_auc_score(y_true, preds):.5f}")
    print(f"Brier Score: {brier_score_loss(y_true, preds):.5f}")
    print(f"Mean Alpha:  {results_df['hybrid_alpha'].mean():.4f}")

    results_df.to_csv("data/golgg_final_hybrid_results.csv", index=False)
    print("\nSaved final hybrid results to golgg_final_hybrid_results.csv")
    print("07_train_hybrid_model.py completed successfully.")


if __name__ == "__main__":
    main()
