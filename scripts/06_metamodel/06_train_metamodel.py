import sys
from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece



def generate_time_series_oof_predictions(model_class, model_kwargs, X_train, y_train, n_splits=5):
    """Generate leakage-safe out-of-fold predictions for time-series data.

    The first expanding-window training segment in ``TimeSeriesSplit`` cannot be
    predicted out-of-fold because no earlier data exists. Those rows remain NaN
    and must be excluded from calibration/training stages that require genuine
    OOF predictions.

    Args:
        model_class: Estimator class exposing ``fit`` and ``predict_proba``.
        model_kwargs: Keyword arguments passed to the estimator constructor.
        X_train: Chronologically ordered feature matrix.
        y_train: Chronologically ordered binary labels.
        n_splits: Number of expanding-window splits.

    Returns:
        NumPy array containing OOF probabilities; unavailable early rows are NaN.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.full(len(X_train), np.nan, dtype=float)

    for train_idx, val_idx in tscv.split(X_train):
        model = model_class(**model_kwargs)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_preds[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]

    return oof_preds

def train_and_calibrate(model_class, model_kwargs, X_train, y_train, X_test, calib_method='isotonic'):
    """Train a classifier and calibrate it using historical OOF predictions.

    Args:
        model_class: Estimator class exposing ``fit`` and ``predict_proba``.
        model_kwargs: Keyword arguments passed to the estimator constructor.
        X_train: Historical training features.
        y_train: Historical training labels.
        X_test: Future test features.
        calib_method: Calibration method: ``isotonic`` or ``platt``.

    Returns:
        Tuple of calibrated test probabilities and the fitted main estimator.
    """
    m_main = model_class(**model_kwargs)
    m_main.fit(X_train, y_train)

    oof_preds = generate_time_series_oof_predictions(
        model_class, model_kwargs, X_train, y_train, n_splits=5
    )
    valid_oof = ~np.isnan(oof_preds)

    if valid_oof.sum() == 0:
        raise ValueError("No valid historical OOF predictions available for calibration.")
        
    if calib_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(oof_preds[valid_oof], y_train.iloc[valid_oof])
    elif calib_method == 'platt':
        calibrator = LogisticRegression()
        calibrator.fit(oof_preds[valid_oof].reshape(-1, 1), y_train.iloc[valid_oof])
    else:
        raise ValueError(f"Unsupported calibration method: {calib_method}")
    
    raw_preds = m_main.predict_proba(X_test)[:, 1]
    
    if calib_method == 'isotonic':
        calibrated_preds = calibrator.transform(raw_preds)
    elif calib_method == 'platt':
        calibrated_preds = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
    
    # Clipping probabilities to avoid 100% certainty (Black Swans)
    calibrated_preds = np.clip(calibrated_preds, 0.01, 0.99)
    
    return calibrated_preds, m_main

def main():
    """Train the stacked metamodel with leakage-safe walk-forward calibration."""
    print("Running 06_train_metamodel.py (Fixing Temporal Leakage & Comparing Calibration)...")
    os.makedirs("docs/assets", exist_ok=True)
    
    print("Loading data...")
    df_preds = pd.read_csv("data/golgg_y_predicts.csv")
    df_rolling = pd.read_csv("data/golgg_rolling_stats.csv")
    
    df_preds['golgg_match_id'] = df_preds['golgg_match_id'].astype(str)
    df_rolling['golgg_match_id'] = df_rolling['golgg_match_id'].astype(str)
    
    df = pd.merge(df_preds, df_rolling, on="golgg_match_id")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features_stage1 = [
        "player_elo", "player_gl", "player_ts", "player_os",
        "player_pl", "player_tm",
        "team_elo", "team_gl", "team_ts",
        "player_gl_rd_avg1", "player_gl_rd_avg2",
        "player_ts_sigma_avg1", "player_ts_sigma_avg2",
        "days_since_last_1", "days_since_last_2", "days_diff", "BoN"
    ]

    features_stage2_context = [
        "t1_rolling_win_rate", "t2_rolling_win_rate",
        "t1_rolling_kills", "t2_rolling_kills",
        "t1_rolling_deaths", "t2_rolling_deaths",
        "t1_rolling_gd15", "t2_rolling_gd15",
        "t1_rolling_dpm", "t2_rolling_dpm",
        "t1_rolling_vspm", "t2_rolling_vspm"
    ]
    
    target = "y_true"
    df_clean = df.dropna(subset=features_stage1 + features_stage2_context + [target]).copy()

    # Modern era focus
    initial_train_mask = df_clean["date"] < datetime(2021, 1, 1)
    train_df_init = df_clean[initial_train_mask].copy()
    test_pool_df = df_clean[~initial_train_mask].copy()

    UPDATE_INTERVAL = 1000
    
    model_types = ["LGBM_Isotonic", "LGBM_Platt"]
    results = {m: {"preds": [], "y_true": [], "s1_preds": []} for m in model_types}
    
    final_model_s1_20gen = None
    final_model_s2 = None
    rng = np.random.default_rng(42)

    print(f"Starting Walk-Forward Training (Interval: {UPDATE_INTERVAL} matches)...")
    
    train_df = train_df_init.copy()
    
    # Hyperparameters from the report
    lgbm_params = {
        'max_depth': 6,
        'num_leaves': 6,
        'learning_rate': 0.024,
        'n_estimators': 486,
        'min_child_samples': 60,
        'subsample': 0.727,
        'colsample_bytree': 0.859,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'verbosity': -1,
        'random_state': 42
    }
    
    for i in tqdm(range(0, len(test_pool_df), UPDATE_INTERVAL), desc="Walk-Forward"):
        test_chunk = test_pool_df.iloc[i : i + UPDATE_INTERVAL]
        if test_chunk.empty: break

        X1_train = train_df[features_stage1].copy()
        y_train = train_df[target]
        
        # 2. 20% General Masking
        X1_20gen = X1_train.copy()
        mask_20 = rng.random(X1_20gen.shape) < 0.20
        X1_20gen[mask_20] = np.nan
        m_s1_20 = LGBMClassifier(**lgbm_params)
        m_s1_20.fit(X1_20gen, y_train)
        final_model_s1_20gen = m_s1_20
        
        # Generate true OOF Stage 1 predictions using TimeSeriesSplit.
        # Early rows without historical training data are excluded from Stage 2 training.
        oof_preds_s1 = generate_time_series_oof_predictions(
            LGBMClassifier, lgbm_params, X1_20gen, y_train, n_splits=5
        )

        train_df = train_df.copy()
        train_df.loc[:, 's1_prob'] = oof_preds_s1
        train_df_s2 = train_df.dropna(subset=['s1_prob']).copy()

        # --- STAGE 2 ---
        X2_train = train_df_s2[['s1_prob'] + features_stage2_context].copy()
        y_train_s2 = train_df_s2[target]
        X1_test = test_chunk[features_stage1]
        
        s1_20_test = m_s1_20.predict_proba(X1_test)[:, 1]
        
        X2_test_base = test_chunk[features_stage2_context].copy()
        X2_test_20 = X2_test_base.copy()
        X2_test_20.insert(0, 's1_prob', s1_20_test)

        # Isotonic
        preds_iso, final_model_s2 = train_and_calibrate(LGBMClassifier, lgbm_params, X2_train, y_train_s2, X2_test_20, calib_method='isotonic')
        results["LGBM_Isotonic"]["preds"].extend(preds_iso)
        results["LGBM_Isotonic"]["y_true"].extend(test_chunk[target])
        results["LGBM_Isotonic"]["s1_preds"].extend(s1_20_test)
        
        # Platt Scaling
        preds_platt, _ = train_and_calibrate(LGBMClassifier, lgbm_params, X2_train, y_train_s2, X2_test_20, calib_method='platt')
        results["LGBM_Platt"]["preds"].extend(preds_platt)
        results["LGBM_Platt"]["y_true"].extend(test_chunk[target])
        results["LGBM_Platt"]["s1_preds"].extend(s1_20_test)

        train_df = pd.concat([train_df, test_chunk])

    # --- EVALUATION & PLOTTING ---
    print("\n=== MODEL COMPARISON (STAGE 2) ===")
    comparison_metrics = []
    for m in model_types:
        y_t = np.array(results[m]["y_true"])
        y_p = np.array(results[m]["preds"])
        comparison_metrics.append({
            "Model": m,
            "AUC": roc_auc_score(y_t, y_p),
            "LogLoss": log_loss(y_t, y_p),
            "Brier": brier_score_loss(y_t, y_p),
            "ECE": calculate_ece(y_t, y_p)
        })
    print(pd.DataFrame(comparison_metrics).to_string(index=False))

    # Feature Importance Comparison (Stage 1)
    plt.figure(figsize=(8, 10))
    imp_20gen = pd.DataFrame(sorted(zip(final_model_s1_20gen.feature_importances_, features_stage1)), columns=['Value','Feature'])
    sns.barplot(x="Value", y="Feature", data=imp_20gen.sort_values(by="Value", ascending=False), palette="Reds_r", hue="Feature", legend=False)
    plt.title("Stage 1 Importance (20% GEN MASKING)")
    plt.tight_layout()
    plt.savefig("docs/assets/metamodel_masking_comparison.png")

    # Feature Importance (Stage 2)
    if final_model_s2 is not None:
        feature_imp = pd.DataFrame(sorted(zip(final_model_s2.feature_importances_, ['s1_prob'] + features_stage2_context)), columns=['Value','Feature'])
        plt.figure(figsize=(8, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), palette="magma", hue="Feature", legend=False)
        plt.title('Stage 2 Feature Importance')
        plt.tight_layout()
        plt.savefig("docs/assets/metamodel_feature_importance.png")

    # Calibration Plot for the best model
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(8, 8))
    
    for m in model_types:
        y_t = np.array(results[m]["y_true"])
        y_p = np.array(results[m]["preds"])
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=m)
        
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Metamodel Calibration Comparison (Isotonic vs Platt)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/metamodel_calibration.png")

    # Save results for next stage. The primary metamodel output is Isotonic,
    # because both training and calibration now use historical OOF predictions only.
    results_df = test_pool_df.copy()
    results_df["s1_prob"] = results["LGBM_Isotonic"]["s1_preds"]
    results_df["metamodel_lgbm_isotonic"] = results["LGBM_Isotonic"]["preds"]
    results_df["metamodel_lgbm_platt"] = results["LGBM_Platt"]["preds"]
    results_df["metamodel_lgbm_prob"] = results["LGBM_Isotonic"]["preds"]
    results_df["metamodel_lgbm_calibrated"] = results["LGBM_Isotonic"]["preds"]
    results_df.to_csv("data/golgg_stacking_results.csv", index=False)
    
    print("06_train_metamodel.py completed successfully.")

if __name__ == "__main__":
    main()
