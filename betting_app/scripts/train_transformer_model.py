import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import os
from tqdm import tqdm
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
        
        # seq_a, seq_b are lists of lists (features)
        # JSON keys from extract_transformer_data.py: t1_seq, t2_seq, y
        seq_a = torch.tensor(match['t1_seq'], dtype=torch.float)
        seq_b = torch.tensor(match['t2_seq'], dtype=torch.float)
        
        # y: 1 for Team 1 win, 0 for Team 2 win
        label = torch.tensor([float(match['y'])], dtype=torch.float)
        
        # Padding masks (True for padded elements)
        # Our sequences are already fixed length from extraction, but let's be safe
        mask_a = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask_b = torch.zeros(self.max_seq_len, dtype=torch.bool)
        
        return seq_a, seq_b, mask_a, mask_b, label

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cpu'
):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq_a, seq_b, mask_a, mask_b, labels in train_loader:
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            mask_a, mask_b = mask_a.to(device), mask_b.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(seq_a, seq_b, mask_a, mask_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for seq_a, seq_b, mask_a, mask_b, labels in val_loader:
                seq_a, seq_b = seq_a.to(device), seq_b.to(device)
                mask_a, mask_b = mask_a.to(device), mask_b.to(device)
                labels = labels.to(device)
                
                logits = model(seq_a, seq_b, mask_a, mask_b)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Metrics
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        val_logloss = log_loss(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LL: {val_logloss:.4f} | Val AUC: {val_auc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/transformer_best.pt')
            print("Saved best model.")

def main():
    data_path = 'data/transformer_team_sequences_v2.json'
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    print("Loading data...")
    with open(data_path, 'r') as f:
        matches = json.load(f)
    
    # If data was saved as a dict with 'matches' key (older version or different script)
    if isinstance(matches, dict) and 'matches' in matches:
        matches = matches['matches']
        
    print(f"Loaded {len(matches)} matches.")
    
    # Chronological split
    # matches are already sorted by date in extraction
    # Let's use 2024 as validation, 2025+ as test, rest as train
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
            train_matches.append(m) # Fallback
            
    print(f"Split: Train={len(train_matches)}, Val={len(val_matches)}, Test={len(test_matches)}")
    
    train_ds = MatchSequenceDataset(train_matches)
    val_ds = MatchSequenceDataset(val_matches)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    model = MatchPredictor(
        input_dim=51,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    train_model(train_loader, val_loader, model, epochs=30, device=device)

if __name__ == "__main__":
    main()
