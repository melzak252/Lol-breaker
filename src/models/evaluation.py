import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from catboost import CatBoostClassifier
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

def train_static_model(df_train: pd.DataFrame, df_test: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, float, float, CatBoostClassifier]:
    """Train a static CatBoost model and evaluate it."""
    cb = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, loss_function='Logloss', verbose=False)
    cb.fit(df_train[features], df_train["y_true"])

    df_test_out = df_test.copy()
    df_test_out["prob_raw"] = cb.predict_proba(df_test_out[features])[:, 1]

    # Isotonic Calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(cb.predict_proba(df_train[features])[:, 1], df_train["y_true"])
    df_test_out["prob_calibrated"] = iso.transform(df_test_out["prob_raw"])

    auc = roc_auc_score(df_test_out["y_true"], df_test_out["prob_calibrated"])
    brier = brier_score_loss(df_test_out["y_true"], df_test_out["prob_calibrated"])
    
    return df_test_out, auc, brier, cb

def train_walk_forward_model(df_preds: pd.DataFrame, df_matched: pd.DataFrame, features: List[str], retrain_frequency_days: int = 7) -> Tuple[pd.DataFrame, float, float, CatBoostClassifier]:
    """Train a walk-forward CatBoost model and evaluate it."""
    df_test_wf = df_matched.dropna(subset=features + ["y_true"]).copy()
    df_test_wf = df_test_wf.sort_values('date_dt')

    wf_raw_probs = pd.Series(index=df_test_wf.index, dtype=float)
    wf_cal_probs = pd.Series(index=df_test_wf.index, dtype=float)

    last_train_date = None
    cb_wf = None
    iso_wf = None

    for current_date in df_test_wf['date_dt'].unique():
        # Check if we need to retrain
        if last_train_date is None or (current_date - last_train_date).days >= retrain_frequency_days:
            df_train_wf = df_preds[df_preds['date_dt'] < current_date].dropna(subset=features + ["y_true"])
            
            cb_wf = CatBoostClassifier(iterations=200, depth=3, learning_rate=0.05, loss_function='Logloss', verbose=False)
            cb_wf.fit(df_train_wf[features], df_train_wf["y_true"])
            
            iso_wf = IsotonicRegression(out_of_bounds='clip')
            iso_wf.fit(cb_wf.predict_proba(df_train_wf[features])[:, 1], df_train_wf["y_true"])
            
            last_train_date = current_date

        # Predict for the current day using the latest trained model
        day_mask = df_test_wf['date_dt'] == current_date
        raw_preds = cb_wf.predict_proba(df_test_wf.loc[day_mask, features])[:, 1]
        
        wf_raw_probs.loc[day_mask] = raw_preds
        wf_cal_probs.loc[day_mask] = iso_wf.transform(raw_preds)

    df_test_wf["prob_raw"] = wf_raw_probs
    df_test_wf["prob_calibrated"] = wf_cal_probs

    auc_wf = roc_auc_score(df_test_wf["y_true"], df_test_wf["prob_calibrated"])
    brier_wf = brier_score_loss(df_test_wf["y_true"], df_test_wf["prob_calibrated"])
    
    return df_test_wf, auc_wf, brier_wf, cb_wf
