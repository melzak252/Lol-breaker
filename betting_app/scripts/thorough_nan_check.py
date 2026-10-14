import json
import numpy as np
import math

def check_file(path):
    print(f"Checking {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    nan_found = False
    inf_found = False
    
    for i, m in enumerate(data):
        for key in ['t1_seq', 't2_seq', 'static_feats']:
            arr = np.array(m[key])
            if np.isnan(arr).any():
                print(f"NaN found in {key} at index {i}")
                nan_found = True
            if np.isinf(arr).any():
                print(f"Inf found in {key} at index {i}")
                inf_found = True
        if nan_found or inf_found:
            break
            
    if not nan_found and not inf_found:
        print("No NaNs or Infs found in dataset.")

if __name__ == "__main__":
    check_file('data/fusion_dataset_v1.json')
