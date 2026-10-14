#!/usr/bin/env python3
"""
Extract embeddings from trained transformer model for fusion with baseline features.
"""
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from betting_app.models.transformer.team_transformer import MatchPredictor

def extract_embeddings(
    data_path: str = "data/transformer_team_sequences_v2.json",
    model_path: str = "models/transformer_best.pt",
    output_path: str = "data/transformer_embeddings_v2.json",
    input_dim: int = 51,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Extract embeddings from trained transformer for all matches.
    
    Args:
        data_path: Path to transformer sequences JSON
        model_path: Path to trained transformer model
        output_path: Path to save embeddings JSON
        input_dim: Number of input features per game
        d_model: Transformer dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        batch_size: Batch size for inference
        device: Device to use (cuda/cpu)
    """
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        matches = json.load(f)
    
    print(f"Loaded {len(matches)} matches")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = MatchPredictor(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Extracting embeddings on {device}...")
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(matches), batch_size), desc="Extracting"):
        batch = matches[i:i+batch_size]
        
        # Prepare batch tensors
        seq_a_list = []
        seq_b_list = []
        match_ids = []
        
        for match in batch:
            seq_a = torch.tensor(match['t1_seq'], dtype=torch.float)
            seq_b = torch.tensor(match['t2_seq'], dtype=torch.float)
            seq_a_list.append(seq_a)
            seq_b_list.append(seq_b)
            match_ids.append(match['match_id'])
        
        seq_a_batch = torch.stack(seq_a_list).to(device)
        seq_b_batch = torch.stack(seq_b_list).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            emb_batch = model.get_embedding(seq_a_batch, seq_b_batch)
        
        # Convert to numpy and store
        emb_np = emb_batch.cpu().numpy()
        for j, match_id in enumerate(match_ids):
            embeddings.append({
                'match_id': match_id,
                'embedding': emb_np[j].tolist()  # 192-dim vector
            })
    
    # Save embeddings
    print(f"Saving embeddings to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(embeddings, f)
    
    print(f"Extracted {len(embeddings)} embeddings (dim={len(embeddings[0]['embedding'])})")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    extract_embeddings()
