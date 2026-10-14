"""Dynamic Champion Encoder Package.

Generates rich, patch-adaptive, inductive champion embeddings from recent match history
without relying on static categorical ID lookup tables.
"""

from __future__ import annotations

from src.models.champion_encoder.model import (
    DynamicChampionEncoder,
    ChampionEncoderConfig,
    MultiTaskPretrainingHead,
)

__all__ = ["DynamicChampionEncoder", "ChampionEncoderConfig", "MultiTaskPretrainingHead"]
