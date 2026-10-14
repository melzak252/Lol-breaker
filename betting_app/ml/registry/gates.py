"""Promotion gate helpers for candidate model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromotionGateConfig:
    min_bets: int = 50
    min_comparison_observations: int = 50
    min_roi: float | None = None
    max_drawdown: float | None = None
    require_model_logloss_better_than_market: bool = True
    require_model_brier_better_than_market: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_market_baseline_gate(metrics: dict[str, Any], config: PromotionGateConfig | None = None) -> PromotionDecision:
    """Evaluate whether a model is good enough to be promoted/shadowed.

    This is intentionally conservative and compares to the bookmaker market when
    available. A production-vs-candidate gate can be layered on top later.
    """
    cfg = config or PromotionGateConfig()
    failures: list[str] = []

    if int(metrics.get("bets") or 0) < cfg.min_bets:
        failures.append(f"not enough bets: {metrics.get('bets')} < {cfg.min_bets}")
    if int(metrics.get("comparison_observations") or 0) < cfg.min_comparison_observations:
        failures.append(
            f"not enough comparison observations: "
            f"{metrics.get('comparison_observations')} < {cfg.min_comparison_observations}"
        )
    if cfg.min_roi is not None and float(metrics.get("roi") or 0.0) < cfg.min_roi:
        failures.append(f"roi below threshold: {metrics.get('roi')} < {cfg.min_roi}")
    if cfg.max_drawdown is not None and float(metrics.get("max_drawdown") or 0.0) > cfg.max_drawdown:
        failures.append(f"drawdown above threshold: {metrics.get('max_drawdown')} > {cfg.max_drawdown}")

    model_log_loss = metrics.get("model_log_loss")
    market_log_loss = metrics.get("market_log_loss")
    if cfg.require_model_logloss_better_than_market and model_log_loss is not None and market_log_loss is not None:
        if float(model_log_loss) > float(market_log_loss):
            failures.append(f"model logloss worse than market: {model_log_loss} > {market_log_loss}")

    model_brier = metrics.get("model_brier")
    market_brier = metrics.get("market_brier")
    if cfg.require_model_brier_better_than_market and model_brier is not None and market_brier is not None:
        if float(model_brier) > float(market_brier):
            failures.append(f"model brier worse than market: {model_brier} > {market_brier}")

    return PromotionDecision(passed=not failures, reasons=failures)
