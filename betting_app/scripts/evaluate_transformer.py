import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import os
from sklearn.metrics import log_loss, roc_auc_score

from betting_app.models.transformer.team_transformer import MatchPredictor

class MatchSequenceDataset(Dataset):
    def __init__(self, matches: List[Dict[str, Any]], max_seq_len: int = 15):
        self.matches = matches
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        seq_a = torch.tensor(match['t1_seq'], dtype=torch.float)
        seq_b = torch.tensor(match['t2_seq'], dtype=torch.float)
        label = torch.tensor([float(match['y'])], dtype=torch.float)
        mask_a = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask_b = torch.zeros(self.max_seq_len, dtype=torch.bool)
        return seq_a, seq_b, mask_a, mask_b, label

def evaluate(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seq_a, seq_b, mask_a, mask_b, labels in loader:
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            mask_a, mask_b = mask_a.to(device), mask_b.to(device)
            
            logits = model(seq_a, seq_b, mask_a, mask_b)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    ll = log_loss(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    acc = np.mean((all_preds > 0.5) == all_labels)
    
    return ll, auc, acc

def main():
    data_path = 'data/transformer_team_sequences_v1.json'
    model_path = 'models/transformer_best.pt'
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print("Loading data...")
    with open(data_path, 'r') as f:
        matches = json.load(f)
    if isinstance(matches, dict) and 'matches' in matches:
        matches = matches['matches']
        
    # Split
    train_matches = []
    val_matches = []
    test_matches = []
    for m in matches:
        date_str = m['date']
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            if dt.year < 2024:
                train_matches.append(m)
            elif dt.year == 2024:
                val_matches.append(m)
            else:
                test_matches.append(m)
        except:
            train_matches.append(m)
            
    print(f"Split: Train={len(train_matches)}, Val={len(val_matches)}, Test={len(test_matches)}")
    
    test_ds = MatchSequenceDataset(test_matches)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = MatchPredictor(
        input_dim=9,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Evaluating on {device}...")
    
    ll, auc, acc = evaluate(test_loader, model, device)
    print(f"\nTest Results (2025+):")
    print(f"LogLoss: {ll:.4f}")
    print(f"AUC:     {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
