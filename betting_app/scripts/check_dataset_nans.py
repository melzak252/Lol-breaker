import json
import numpy as np

def check_nans(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"Checking {len(data)} samples...")
    
    # Feature names for sequence (16 features)
    seq_features = [
        "win", "side", "duration", "kills", "deaths", "gold", "towers", "dragons", "nashors", 
        "gd15", "csd15", "xpd15", "dpm", "vspm", "days_rest", "opp_elo"
    ]
    
    nan_counts_seq = {feat: 0 for feat in seq_features}
    nan_counts_static = 0
    
    for i, item in enumerate(data):
        t1 = np.array(item['t1_seq'])
        t2 = np.array(item['t2_seq'])
        
        for seq in [t1, t2]:
            if np.isnan(seq).any():
                nan_mask = np.isnan(seq)
                # Find which columns have NaNs
                cols_with_nan = np.where(nan_mask.any(axis=0))[0]
                for col_idx in cols_with_nan:
                    nan_counts_seq[seq_features[col_idx]] += 1
        
        if np.isnan(item['static_feats']).any():
            nan_counts_static += 1
            
    print("NaN counts in sequence features (per occurrence in t1 or t2):")
    for feat, count in nan_counts_seq.items():
        if count > 0:
            print(f"  {feat}: {count}")
            
    print(f"NaN counts in static features: {nan_counts_static}")

if __name__ == "__main__":
    check_nans('data/fusion_dataset_v1.json')
