import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime
import os

def main():
    print("Running 06c_winstreak_experiment.py...")
    
    df_preds = pd.read_csv("data/golgg_y_predicts.csv")
    df_rolling = pd.read_csv("data/golgg_rolling_stats.csv")
    df_preds['golgg_match_id'] = df_preds['golgg_match_id'].astype(str)
    df_rolling['golgg_match_id'] = df_rolling['golgg_match_id'].astype(str)
    df = pd.merge(df_preds, df_rolling, on="golgg_match_id")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Features
    s1_features = ["player_gl"] # Best single system
    
    context_features = [
        "days_diff", "BoN", "t1_rolling_gd15", "t2_rolling_gd15", 
        "t1_rolling_win_rate", "t2_rolling_win_rate"
    ]
    
    # "Extended" features that might add noise
    extended_features = [
        "t1_rolling_kills", "t2_rolling_kills", 
        "t1_rolling_deaths", "t2_rolling_deaths",
        "t1_rolling_dpm", "t2_rolling_dpm"
    ]

    target = "y_true"
    df_clean = df.dropna(subset=s1_features + context_features + extended_features + [target]).copy()

    train_mask = df_clean["date"] < datetime(2023, 1, 1)
    test_mask = (df_clean["date"] >= datetime(2023, 1, 1)) & (df_clean["date"] < datetime(2025, 1, 1))
    
    X_train = df_clean[train_mask]
    y_train = df_clean[train_mask][target]
    X_test = df_clean[test_mask]
    y_test = df_clean[test_mask][target]

    results = []

    # 1. Stage 1 Only (Glicko2)
    auc_s1 = roc_auc_score(y_test, X_test["player_gl"])
    ll_s1 = log_loss(y_test, X_test["player_gl"])
    results.append({"Model": "Stage 1 (Glicko2)", "AUC": auc_s1, "LogLoss": ll_s1})

    # 2. Stage 2 Standard (Ratings + Core Context)
    m2 = LGBMClassifier(n_estimators=100, max_depth=3, verbosity=-1)
    m2.fit(X_train[s1_features + context_features], y_train)
    p2 = m2.predict_proba(X_test[s1_features + context_features])[:, 1]
    results.append({"Model": "Stage 2 Standard", "AUC": roc_auc_score(y_test, p2), "LogLoss": log_loss(y_test, p2)})

    # 3. Stage 2 Extended (Adding noisy features)
    m3 = LGBMClassifier(n_estimators=100, max_depth=3, verbosity=-1)
    m3.fit(X_train[s1_features + context_features + extended_features], y_train)
    p3 = m3.predict_proba(X_test[s1_features + context_features + extended_features])[:, 1]
    results.append({"Model": "Stage 2 Extended", "AUC": roc_auc_score(y_test, p3), "LogLoss": log_loss(y_test, p3)})

    res_df = pd.DataFrame(results)
    print("\n=== EXPERIMENT: FEATURE SELECTION ===")
    print(res_df.to_string(index=False))
    res_df.to_csv("docs/assets/metamodel_experiment_winstreak.csv", index=False)

if __name__ == "__main__":
    main()
