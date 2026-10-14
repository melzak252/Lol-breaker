"""ML models for LoL match prediction, residual learning, and series simulation."""

from __future__ import annotations

from betting_app.ml.models.market_residual import (
    EdgeSignal,
    MarketResidualModel,
    ResidualEdgeDetector,
    compute_differential_features,
)

__all__ = [
    "EdgeSignal",
    "MarketResidualModel",
    "ResidualEdgeDetector",
    "compute_differential_features",
]

# Gracefully re-export markov_series if present
try:
    from betting_app.ml.models.markov_series import *  # noqa: F401, F403
    from betting_app.ml.models import markov_series

    for _name in getattr(markov_series, "__all__", []):
        if _name not in __all__:
            __all__.append(_name)
except ImportError:
    pass
