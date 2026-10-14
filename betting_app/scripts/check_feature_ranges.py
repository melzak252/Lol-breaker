import json
import numpy as np

def check_ranges(path):
    print(f"Checking ranges in {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    seq_data = []
    static_data = []
    
    for m in data:
        seq_data.append(m['t1_seq'])
        seq_data.append(m['t2_seq'])
        static_data.append(m['static_feats'])
        
    seq_data = np.array(seq_data)
    static_data = np.array(static_data)
    
    print("\nSequence Features (16 features):")
    seq_features = [
        "win", "side", "duration", "kills", "deaths", "gold", "towers", "dragons", "nashors", 
        "gd15", "csd15", "xpd15", "dpm", "vspm", "days_rest", "opp_elo"
    ]
    for i, feat in enumerate(seq_features):
        vals = seq_data[:, :, i].flatten()
        print(f"  {feat:10}: min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")
        
    print("\nStatic Features (20 features):")
    for i in range(static_data.shape[1]):
        vals = static_data[:, i]
        print(f"  Feat {i:2}: min={np.min(vals):.4f}, max={np.max(vals):.4f}, mean={np.mean(vals):.4f}")

if __name__ == "__main__":
    check_ranges('/home/melzak/dev/inzynierka/data/fusion_dataset_v1.json')
