"""Dataset Loader for A1 Training and Canonical Benchmark Evaluation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from scripts.a1.global_meta_aggregator import compute_14d_global_meta_matrix


class A1Dataset(Dataset):
    """PyTorch Dataset for A1 competitive match outcomes."""

    def __init__(
        self,
        context: np.ndarray,
        player_ratings: np.ndarray,
        player_history: np.ndarray,
        player_mask: np.ndarray,
        player_champions: np.ndarray,
        meta_champions: np.ndarray,
        best_of: np.ndarray,
        labels: np.ndarray | None = None,
    ):
        self.context = torch.as_tensor(context, dtype=torch.float32)
        self.player_ratings = torch.as_tensor(player_ratings, dtype=torch.float32)
        self.player_history = torch.as_tensor(player_history, dtype=torch.float32)
        self.player_mask = torch.as_tensor(player_mask, dtype=torch.uint8)
        self.player_champions = torch.as_tensor(player_champions, dtype=torch.long)
        self.meta_champions = torch.as_tensor(meta_champions, dtype=torch.float32)
        self.best_of = torch.as_tensor(best_of, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.context)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "context": self.context[idx],
            "player_ratings": self.player_ratings[idx],
            "player_history": self.player_history[idx],
            "player_mask": self.player_mask[idx],
            "player_champions": self.player_champions[idx],
            "meta_champions": self.meta_champions[idx],
            "best_of": self.best_of[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item
