"""Optimize LightGBM hyperparameters for the odds-only metamodel sample."""

import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
def main() -> None:
    """Run Optuna hyperparameter optimization for odds-mapped matches.

    The optimization is restricted to matches present in `odds.csv`, because this
    is the operational sample used later for market comparison and financial
    simulation. The split is chronological 80/20, therefore the result is a
    diagnostic holdout experiment rather than a nested walk-forward estimate.
    """
    print(
        "Running 13_hyperparameter_optimization.py "
        "(Optuna, odds.csv only, NO team-based features)..."
    )
    os.makedirs("docs/assets", exist_ok=True)

    print("Loading data...")
    df_preds = pd.read_csv("data/golgg_y_predicts.csv")
    df_odds = pd.read_csv("data/odds.csv")

    df_preds["golgg_match_id"] = df_preds["golgg_match_id"].astype(str)
    df_odds["golgg_match_id"] = df_odds["golgg_match_id"].astype(str)

    df = pd.merge(df_preds, df_odds[["golgg_match_id"]], on="golgg_match_id")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = [
        "player_elo",
        "player_gl",
        "player_ts",
        "player_os",
        "player_pl",
        "player_tm",
        "player_elo_min1",
        "player_elo_min2",
        "player_gl_max1",
        "player_gl_max2",
        "player_gl_rd_avg1",
        "player_gl_rd_avg2",
        "player_ts_sigma_avg1",
        "player_ts_sigma_avg2",
        "player_os_sigma_avg1",
        "player_os_sigma_avg2",
        "player_pl_sigma_avg1",
        "player_pl_sigma_avg2",
        "player_tm_sigma_avg1",
        "player_tm_sigma_avg2",
    ]

    df_clean = df.dropna(subset=features + ["y_true"]).copy()
    df_clean = df_clean[df_clean["date"] >= pd.to_datetime("2020-01-01")].copy()

    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]

    x_train = train_df[features]
    y_train = train_df["y_true"]
    x_test = test_df[features]
    y_test = test_df["y_true"]

    print(f"Dataset size: {len(df_clean)} matches (with odds)")
    print(f"Train size: {len(x_train)} matches (up to {train_df['date'].max()})")
    print(f"Test size: {len(x_test)} matches (from {test_df['date'].min()})")

    def objective(trial: optuna.Trial) -> float:
        """Evaluate one LightGBM hyperparameter configuration.

        Args:
            trial: Optuna trial object.

        Returns:
            Test-set LogLoss for the sampled configuration.
        """
        max_depth = trial.suggest_int("max_depth", 2, 10)
        max_num_leaves = min(127, (2 ** max_depth) - 1)
        params = {
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 3, max_num_leaves),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(x_train, y_train)

        preds = model.predict_proba(x_test)[:, 1]
        loss = log_loss(y_test, preds)

        trial.set_user_attr("auc", roc_auc_score(y_test, preds))
        trial.set_user_attr("brier", brier_score_loss(y_test, preds))
        trial.set_user_attr("ece", calculate_ece(y_test, preds))

        return float(loss)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name="metamodel_optimization_odds",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n=== OPTUNA OPTIMIZATION FINISHED ===")
    print(f"Best Log Loss: {study.best_value:.5f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_trial = study.best_trial
    print("\nMetrics for best trial:")
    print(f"  AUC: {best_trial.user_attrs['auc']:.5f}")
    print(f"  Brier: {best_trial.user_attrs['brier']:.5f}")
    print(f"  ECE: {best_trial.user_attrs['ece']:.5f}")

    results_df = study.trials_dataframe()
    results_df.to_csv("docs/assets/metopt_optuna_results_odds.csv", index=False)

    with open("docs/assets/metamodel_best_params_optuna_odds.txt", "w", encoding="utf-8") as file:
        for key, value in study.best_params.items():
            file.write(f"{key}: {value}\n")

    print("\n13_hyperparameter_optimization.py completed.")


if __name__ == "__main__":
    main()
