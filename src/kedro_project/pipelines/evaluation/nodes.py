"""Kedro nodes wrapping the historical evaluation pipeline.

Each node delegates to the existing plain-Python pipeline in
``betting_app.ml.pipelines.evaluation``, keeping all business logic
in one place while gaining Kedro's configuration, logging, and
reproducibility benefits.
"""

from __future__ import annotations

from typing import Any

from betting_app.ml.config import BacktestConfig, StakingConfig
from betting_app.ml.pipelines.evaluation import (
    EvaluationPipelineConfig,
    EvaluationPipelineResult,
    run_evaluation_pipeline,
)
from betting_app.ml.registry.gates import PromotionGateConfig


def run_historical_evaluation(params: dict[str, Any]) -> dict[str, Any]:
    """Run the historical evaluation pipeline with Kedro-provided parameters.

    Parameters
    ----------
    params : dict
        Parameters from ``conf/base/parameters.yml`` under the
        ``evaluation`` key.

    Returns
    -------
    dict
        Summary metrics of the evaluation run.
    """
    backtest_params = params.get("backtest", {})
    staking_params = backtest_params.get("staking", {})
    gate_params = params.get("promotion_gate", {})

    config = EvaluationPipelineConfig(
        model_name=params["model_name"],
        model_version=params["model_version"],
        days_back=params.get("days_back", 365),
        include_stale=params.get("include_stale", False),
        latest_per_match=params.get("latest_per_match", True),
        backtest=BacktestConfig(
            bankroll_start=backtest_params.get("bankroll_start", 1_000.0),
            min_ev=backtest_params.get("min_ev", 0.0),
            staking=StakingConfig(
                strategy=staking_params.get("strategy", "fractional_kelly"),
                fixed_stake=staking_params.get("fixed_stake", 10.0),
                kelly_fraction=staking_params.get("kelly_fraction", 0.25),
            ),
            max_bets_per_match=backtest_params.get("max_bets_per_match", 1),
        ),
        promotion_gate=PromotionGateConfig(
            min_bets=gate_params.get("min_bets", 50),
            min_comparison_observations=gate_params.get("min_comparison_observations", 50),
            min_roi=gate_params.get("min_roi"),
        ),
        register_candidate=params.get("register_candidate", False),
        run_type=params.get("run_type", "strict_market_benchmark"),
    )
    result: EvaluationPipelineResult = run_evaluation_pipeline(config)
    return {
        **result.metrics,
        "evaluation_run_id": result.evaluation_run_id,
        "promotion_gate_passed": result.decision.passed,
        "promotion_gate_reasons": result.decision.reasons,
    }