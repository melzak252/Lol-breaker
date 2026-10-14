"""Validated persistence contract for Venn-Abers prediction intervals.

Only intervals emitted by a fitted Venn-Abers calibrator may pass the financial
risk gate.  Point forecasts and heuristic margins are deliberately excluded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConformalBounds:
    """Side-aligned Venn-Abers interval for one binary match prediction."""

    lower_a: float
    upper_a: float
    lower_b: float
    upper_b: float


def _probability(value: Any) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def conformal_bounds_from_diagnostics(value: Any) -> ConformalBounds | None:
    """Return validated bounds persisted by registry inference, else ``None``.

    The persisted contract is ``diagnostics_json.conformal`` with an explicit
    ``method`` and side-A bounds. Side-B bounds are derived by complement so
    storage and all consumers retain binary side symmetry.
    """

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    conformal = value.get("conformal")
    if not isinstance(conformal, dict) or conformal.get("method") != "venn_abers":
        return None

    lower_a = _probability(conformal.get("p_lower_a"))
    upper_a = _probability(conformal.get("p_upper_a"))
    if lower_a is None or upper_a is None or lower_a > upper_a:
        return None

    return ConformalBounds(
        lower_a=lower_a,
        upper_a=upper_a,
        lower_b=1.0 - upper_a,
        upper_b=1.0 - lower_a,
    )


def conformal_bounds_for_side(value: Any, side: str) -> tuple[float, float] | None:
    """Return ``(p_lower, p_upper)`` for canonical side ``a`` or ``b``."""

    bounds = conformal_bounds_from_diagnostics(value)
    if bounds is None:
        return None
    if side == "a":
        return bounds.lower_a, bounds.upper_a
    if side == "b":
        return bounds.lower_b, bounds.upper_b
    raise ValueError("side must be 'a' or 'b'")
