import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TeamEncoder(nn.Module):
    """
    Encodes a sequence of historical games for a single team into a fixed-size embedding.
    """
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TeamEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # We'll use the mean of the sequence or the last element as the embedding.
        # For now, let's use a learnable [CLS] token or just mean pooling.
        # Mean pooling is often more stable for short sequences.
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x) # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Mean pooling across the sequence dimension
        # mask is True for padded elements
        if mask is not None:
            # mask shape: (batch_size, seq_len)
            # Expand mask to (batch_size, seq_len, 1)
            expanded_mask = mask.unsqueeze(-1).float()
            # Zero out padded elements
            x = x * (1.0 - expanded_mask)
            # Sum and divide by actual sequence length
            sum_x = x.sum(dim=1)
            count_x = (1.0 - expanded_mask).sum(dim=1).clamp(min=1.0)
            embedding = sum_x / count_x
        else:
            embedding = x.mean(dim=1)
            
        return embedding

class MatchPredictor(nn.Module):
    """
    Siamese Transformer model to predict match outcome between two teams.
    """
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(MatchPredictor, self).__init__()
        self.team_encoder = TeamEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # MLP to predict winner from team embeddings
        # We concatenate [emb_a, emb_b, emb_a - emb_b]
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Probability of Team A winning
        )
        
    def forward(self, seq_a, seq_b, mask_a=None, mask_b=None):
        emb_a = self.team_encoder(seq_a, mask_a)
        emb_b = self.team_encoder(seq_b, mask_b)
        
        # Combine embeddings
        diff = emb_a - emb_b
        combined = torch.cat([emb_a, emb_b, diff], dim=-1)
        
        logits = self.classifier(combined)
        return logits # Return logits, use sigmoid in loss or for inference
    
    def get_embedding(self, seq_a, seq_b, mask_a=None, mask_b=None):
        """
        Extract combined embedding without final classification.
        Returns: tensor of shape (batch_size, d_model * 3) = (batch_size, 192)
        """
        emb_a = self.team_encoder(seq_a, mask_a)
        emb_b = self.team_encoder(seq_b, mask_b)
        
        # Combine embeddings
        diff = emb_a - emb_b
        combined = torch.cat([emb_a, emb_b, diff], dim=-1)
        
        return combined # Return embedding for fusion with other features
