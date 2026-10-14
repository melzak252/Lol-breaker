
import joblib
import numpy as np
from pathlib import Path

PIPELINE_PATH = Path("betting_app/models/sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib")

def main():
    pipeline = joblib.load(PIPELINE_PATH)
    scaler = pipeline.named_steps['scaler']
    
    print("Scaler Means:")
    print(scaler.mean_)
    print("\nScaler Scales:")
    print(scaler.scale_)
    
    # Map features to means/scales
    metadata_path = Path("docs/assets/final_symmetric_calibrated_market_comparison/sym_cal_lr_elasticnet_w20_binomial_metadata.json")
    import json
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    features = metadata['features']
    for f, m, s in zip(features, scaler.mean_, scaler.scale_):
        print(f"{f:25} | Mean: {m:10.4f} | Scale: {s:10.4f}")

if __name__ == "__main__":
    main()
