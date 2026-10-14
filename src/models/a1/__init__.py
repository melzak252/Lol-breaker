"""A1 Architecture Package: Hierarchical Global-Meta & Team-Attention Transformer.

Integrates:
1. 14-day rolling tier-weighted Global Meta Champion Attention.
2. 5-node Role-Conditioned Set Transformer for intra-team duo and cross-lane synergy.
3. Anti-symmetric bilateral competitive head.
"""

from __future__ import annotations

from src.models.a1.architecture import A1Model, A1Config
from src.models.a1.predictor import A1Predictor

__all__ = ["A1Model", "A1Config", "A1Predictor"]
