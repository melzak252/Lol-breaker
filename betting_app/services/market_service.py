"""Market-analysis helpers: expected value, Kelly criterion, arbitrage detection.

All functions are stateless and operate on raw numeric inputs so they are safe
to call from both the Streamlit UI and the FastAPI backend.
"""

from __future__ import annotations

import json
from typing import Any


def expected_value(probability: float, odds: float, tax_rate: float = 0.12) -> float:
    """Return EV for a single binary bet after Polish withholding tax."""
    return probability * odds * (1.0 - tax_rate) - 1.0


def kelly_fraction(
    probability: float, odds: float, tax_rate: float = 0.12
) -> float:
    """Full Kelly fraction for a binary bet after tax on winnings.

    The effective decimal return is *odds × (1 - tax_rate)*.
    Negative Kelly is clipped to 0 (no-bet according to Kelly).
    """
    p = float(probability)
    eff = float(odds) * (1.0 - tax_rate)
    net = eff - 1.0
    if p <= 0.0 or eff <= 1.0 or net <= 0.0:
        return 0.0
    return max(0.0, (p * eff - 1.0) / net)


# ── Arbitrage ──────────────────────────────────────────────────────────────


def enrich_arbitrage(record: dict[str, Any], *, tax_rate: float = 0.12) -> None:
    """Mutate *record* with arbitrage flags/margins (mutates in place).

    Keys added: ``arb_no_tax``, ``arb_after_tax``, ``arb_margin_no_tax``,
    ``arb_margin_after_tax``.
    """
    odds_a = none_or_float(record.get("best_odds_a"))
    odds_b = none_or_float(record.get("best_odds_b"))
    if not odds_a or not odds_b or odds_a <= 1 or odds_b <= 1:
        record["arb_no_tax"] = False
        record["arb_after_tax"] = False
        record["arb_margin_no_tax"] = None
        record["arb_margin_after_tax"] = None
        return
    inv = 1.0 / odds_a + 1.0 / odds_b
    inv_tax = 1.0 / (odds_a * (1.0 - tax_rate)) + 1.0 / (odds_b * (1.0 - tax_rate))
    record["arb_no_tax"] = inv < 1.0
    record["arb_after_tax"] = inv_tax < 1.0
    record["arb_margin_no_tax"] = 1.0 - inv
    record["arb_margin_after_tax"] = 1.0 - inv_tax


# ── Data helpers ────────────────────────────────────────────────────────────


def safe_json_get(raw: Any, path: list[str]) -> Any:
    """Safely traverse a JSON value (string or dict) given a *path* of keys."""
    current = raw
    if isinstance(current, str):
        try:
            current = json.loads(current)
        except Exception:  # noqa: BLE001
            return None
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def none_or_float(value: Any) -> float | None:
    """Coerce *value* to float or return None."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_odds(value: Any) -> str:
    """Format a decimal-odds number or return '—'."""
    number = none_or_float(value)
    return "—" if number is None else f"{number:.2f}"


def format_pct(value: Any) -> str:
    """Format a probability (0–1) as a percentage string or return '—'."""
    number = none_or_float(value)
    return "—" if number is None else f"{number * 100:.1f}%"
