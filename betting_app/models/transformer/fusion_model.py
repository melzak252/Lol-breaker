import torch
import torch.nn as nn
from .team_transformer import TeamEncoder

class FusionModel(nn.Module):
    """
    Combines Transformer-based team form embeddings with static player rankings.
    """
    def __init__(self, 
                 seq_input_dim=16, 
                 static_input_dim=20, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2, 
                 dim_feedforward=128, 
                 dropout=0.1):
        super(FusionModel, self).__init__()
        
        # Branch 1: Team Form (Transformer)
        self.team_encoder = TeamEncoder(
            input_dim=seq_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Branch 2: Static Rankings (MLP)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        # Fusion Layer
        # We combine: emb_a, emb_b, (emb_a - emb_b), and static_emb
        combined_dim = d_model * 3 + d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Probability of Team 1 winning the game
        )
        
    def forward(self, seq_a, seq_b, static_feats, mask_a=None, mask_b=None):
        # seq_a/b: (batch, seq_len, seq_input_dim)
        # static_feats: (batch, static_input_dim)
        
        emb_a = self.team_encoder(seq_a, mask_a)
        emb_b = self.team_encoder(seq_b, mask_b)
        static_emb = self.static_encoder(static_feats)
        
        # Combine
        diff = emb_a - emb_b
        combined = torch.cat([emb_a, emb_b, diff, static_emb], dim=-1)
        
        logits = self.classifier(combined)
        return logits
