import joblib
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/home/melzak/dev/inzynierka")
PIPELINE_PATH = PROJECT_ROOT / "betting_app" / "models" / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"

def main():
    if not PIPELINE_PATH.exists():
        print(f"Pipeline not found at {PIPELINE_PATH}")
        return

    pipeline = joblib.load(PIPELINE_PATH)
    model = pipeline.named_steps['model']
    
    # Get feature names from metadata if possible, or use the ones from train_thesis_model.py
    OPTUNA_BASE_FEATURES = [
        "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
        "player_elo_min1", "player_elo_min2", "player_gl_max1", "player_gl_max2",
        "player_gl_rd_avg1", "player_gl_rd_avg2", "player_ts_sigma_avg1", "player_ts_sigma_avg2",
        "player_os_sigma_avg1", "player_os_sigma_avg2", "player_pl_sigma_avg1", "player_pl_sigma_avg2",
        "player_tm_sigma_avg1", "player_tm_sigma_avg2",
    ]
    ROLLING_FULL_FEATURES = [
        "t1_rolling_win_rate", "t2_rolling_win_rate", "t1_rolling_kills", "t2_rolling_kills",
        "t1_rolling_deaths", "t2_rolling_deaths", "t1_rolling_gd15", "t2_rolling_gd15",
        "t1_rolling_dpm", "t2_rolling_dpm", "t1_rolling_vspm", "t2_rolling_vspm",
        "t1_rolling_towers", "t2_rolling_towers", "t1_rolling_nashors", "t2_rolling_nashors",
        "t1_rolling_gold", "t2_rolling_gold", "t1_rolling_duration", "t2_rolling_duration",
    ]
    RANK_PROB_FEATURES = [
        "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
    ]
    binomial_features = [f"{f}_binom_series" for f in RANK_PROB_FEATURES]
    
    all_features = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + binomial_features
    
    coefs = model.coef_[0]
    
    df_coef = pd.DataFrame({
        'feature': all_features,
        'coef': coefs,
        'abs_coef': np.abs(coefs)
    }).sort_values('abs_coef', ascending=False)
    
    print("Top 20 features by absolute coefficient:")
    print(df_coef.head(20).to_string(index=False))
    
    print("\nBottom 10 features (least important):")
    print(df_coef.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
