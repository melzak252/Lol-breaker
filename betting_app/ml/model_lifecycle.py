"""Central lifecycle policy for model families excluded from public surfaces."""

from __future__ import annotations

# This generic weekly tabular retrain was never promoted to the application
# prediction contract. Preserve historical rows for audit, but never expose or
# select them in public API responses.
RETIRED_PUBLIC_MODEL_NAME = "Operational-Retrained-Tabular"
RETIRED_PUBLIC_MODEL_NAMES = frozenset({RETIRED_PUBLIC_MODEL_NAME})
