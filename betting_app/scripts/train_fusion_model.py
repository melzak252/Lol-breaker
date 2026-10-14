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

from betting_app.models.transformer.fusion_model import FusionModel

class FusionDataset(Dataset):
    def __init__(self, matches: List[Dict[str, Any]], max_seq_len: int = 15):
        self.matches = matches
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        
        # seq_a, seq_b: (seq_len, 16)
        seq_a = torch.tensor(match['t1_seq'], dtype=torch.float)
        seq_b = torch.tensor(match['t2_seq'], dtype=torch.float)
        
        # static_feats: (20,)
        static_feats = np.array(match['static_feats'], dtype=np.float32)
        
        # Normalization of static features
        # 0-5: probabilities (already 0-1)
        # 6-9: Elo/Glicko ratings (around 1500)
        static_feats[6:10] = (static_feats[6:10] - 1500.0) / 500.0
        # 10-11: Glicko RD (around 350)
        static_feats[10:12] = static_feats[10:12] / 350.0
        # 12-13: TrueSkill Sigma (around 8.33)
        static_feats[12:14] = static_feats[12:14] / 8.33
        # 14-15: OpenSkill Sigma (around 3.5)
        static_feats[14:16] = static_feats[14:16] / 3.5
        # 16-19: Other Sigmas (around 8.33)
        static_feats[16:20] = static_feats[16:20] / 8.33
        
        static_feats = torch.from_numpy(static_feats)
        
        # y: 1 for Team 1 win, 0 for Team 2 win
        label = torch.tensor([float(match['y'])], dtype=torch.float)
        
        # Padding masks (True for padded elements)
        # Padding is at the beginning: [[0]*16, ..., actual_data]
        mask_a = (seq_a.abs().sum(dim=-1) == 0)
        mask_b = (seq_b.abs().sum(dim=-1) == 0)
        
        # Ensure at least one element is not masked to avoid NaN in Transformer
        if mask_a.all(): mask_a[-1] = False
        if mask_b.all(): mask_b[-1] = False
        
        return seq_a, seq_b, static_feats, mask_a, mask_b, label

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cpu',
    model_name: str = 'fusion_best.pt'
):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq_a, seq_b, static, mask_a, mask_b, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            seq_a, seq_b = seq_a.to(device), seq_b.to(device)
            static = static.to(device)
            mask_a, mask_b = mask_a.to(device), mask_b.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(seq_a, seq_b, static, mask_a, mask_b)
            
            if torch.isnan(logits).any():
                print(f"\nNaN detected in logits at Epoch {epoch+1}")
                # Check weights
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN detected in parameter: {name}")
                return # Stop training
                
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for seq_a, seq_b, static, mask_a, mask_b, labels in val_loader:
                seq_a, seq_b = seq_a.to(device), seq_b.to(device)
                static = static.to(device)
                mask_a, mask_b = mask_a.to(device), mask_b.to(device)
                labels = labels.to(device)
                
                logits = model(seq_a, seq_b, static, mask_a, mask_b)
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
        
        if np.isnan(all_preds).any():
            print(f"Epoch {epoch+1} | NaN detected in validation predictions!")
            val_logloss = 1.0
            val_auc = 0.5
        else:
            val_logloss = log_loss(all_labels, all_preds)
            val_auc = roc_auc_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LL: {val_logloss:.4f} | Val AUC: {val_auc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{model_name}')
            print(f"Saved best model to models/{model_name}")

def main():
    data_path = 'data/fusion_dataset_v1.json'
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
        
    print("Loading data...")
    start_time = datetime.now()
    with open(data_path, 'r') as f:
        matches = json.load(f)
    print(f"Loaded {len(matches)} games in {datetime.now() - start_time}")
    
    # Chronological split
    print("Splitting data...")
    train_matches = []
    val_matches = []
    test_matches = []
    
    for m in tqdm(matches, desc="Splitting"):
        date_str = m['date']
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.year < 2024:
                train_matches.append(m)
            elif dt.year == 2024:
                val_matches.append(m)
            else:
                test_matches.append(m)
        except:
            train_matches.append(m)
            
    print(f"Split: Train={len(train_matches)}, Val={len(val_matches)}, Test={len(test_matches)}")
    
    train_ds = FusionDataset(train_matches)
    val_ds = FusionDataset(val_matches)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = FusionModel(
        seq_input_dim=16,
        static_input_dim=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    train_model(train_loader, val_loader, model, epochs=30, device=device, model_name='fusion_best.pt')

if __name__ == "__main__":
    main()
