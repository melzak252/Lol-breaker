"""Historical model evaluation pipeline.

This pipeline evaluates existing predictions from `canonical_predictions`
against historical match results and collected bookmaker odds. It is the first
production-oriented building block for regular model testing and future model
promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from betting_app.core.db import get_session
from betting_app.ml.backtesting.comparison import compare_predictions_to_market
from betting_app.ml.backtesting.loaders import load_finished_match_labels, load_odds_quotes, load_predictions
from betting_app.ml.backtesting.temporal import select_temporally_eligible_predictions
from betting_app.ml.config import BacktestConfig
from betting_app.ml.registry.gates import PromotionDecision, PromotionGateConfig, evaluate_market_baseline_gate
from betting_app.ml.registry.repository import (
    EvaluationRunRecord,
    ModelVersionRecord,
    record_evaluation_run,
    register_model_version,
)


@dataclass(frozen=True)
class EvaluationPipelineConfig:
    model_name: str
    model_version: str
    days_back: int | None = 365
    include_stale: bool = False
    latest_per_match: bool = True
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    promotion_gate: PromotionGateConfig = field(default_factory=PromotionGateConfig)
    register_candidate: bool = True
    run_type: str = "strict_market_benchmark"


@dataclass(frozen=True)
class EvaluationPipelineResult:
    metrics: dict[str, Any]
    decision: PromotionDecision
    evaluation_run_id: str


def run_evaluation_pipeline(
    config: EvaluationPipelineConfig,
    session: Session | None = None,
) -> EvaluationPipelineResult:
    """Run strict point-in-time model-versus-market quality evaluation."""
    own_session = session is None
    sess = session or get_session()
    try:
        labels = load_finished_match_labels(days_back=config.days_back, session=sess)
        label_ids = {label.canonical_match_id for label in labels}
        loaded_predictions = load_predictions(
            model_name=config.model_name,
            model_version=config.model_version,
            only_active=not config.include_stale,
            latest_per_match=(
                False if config.backtest.strict_temporal_eligibility else config.latest_per_match
            ),
            session=sess,
        )
        predictions = [
            prediction
            for prediction in loaded_predictions
            if prediction.canonical_match_id in label_ids
        ]
        temporal_selection = select_temporally_eligible_predictions(
            predictions, labels
        ) if config.backtest.strict_temporal_eligibility else None
        eligible_predictions = (
            temporal_selection.predictions if temporal_selection is not None else predictions
        )
        odds = load_odds_quotes(canonical_match_ids=label_ids, session=sess)

        comparison = compare_predictions_to_market(
            eligible_predictions, labels, odds, config.backtest
        )

        eligible_match_ids = {
            prediction.canonical_match_id for prediction in eligible_predictions
        }
        cohort_starts = sorted(
            label.start_time
            for label in labels
            if label.canonical_match_id in eligible_match_ids
            and label.start_time is not None
        )
        metrics: dict[str, Any] = {
            "predictions_loaded": len(predictions),
            "labels_loaded": len(labels),
            "odds_quotes_loaded": len(odds),
            "predictions_temporally_eligible": len(eligible_predictions),
            "prediction_temporal_exclusions": (
                temporal_selection.exclusions if temporal_selection is not None else {}
            ),
            "cohort_start_at": cohort_starts[0].isoformat() if cohort_starts else None,
            "cohort_end_at": cohort_starts[-1].isoformat() if cohort_starts else None,
            "financial_execution_evaluated": False,
            "comparison_observations": comparison.observations,
            "model_log_loss": round(comparison.model_log_loss, 6),
            "market_log_loss": round(comparison.market_log_loss, 6),
            "model_brier": round(comparison.model_brier, 6),
            "market_brier": round(comparison.market_brier, 6),
            "model_accuracy": round(comparison.model_accuracy, 6),
            "market_accuracy": round(comparison.market_accuracy, 6),
            "model_auc": round(comparison.model_auc, 6),
            "market_auc": round(comparison.market_auc, 6),
            "model_ece": round(comparison.model_ece, 6),
            "market_ece": round(comparison.market_ece, 6),
            "comparison_eligible_predictions": comparison.eligible_predictions,
            "comparison_prediction_exclusions": comparison.prediction_exclusions,
            "comparison_quote_exclusions": comparison.quote_exclusions,
        }

        decision = evaluate_market_baseline_gate(metrics, config.promotion_gate)
        if config.register_candidate:
            register_model_version(
                ModelVersionRecord(
                    model_name=config.model_name,
                    model_version=config.model_version,
                    status="shadow" if decision.passed else "candidate",
                    metrics=metrics,
                    notes="Auto-registered by strict model-versus-market benchmark",
                ),
                session=sess,
            )

        run = record_evaluation_run(
            EvaluationRunRecord(
                model_name=config.model_name,
                model_version=config.model_version,
                run_type=config.run_type,
                status="completed",
                config=asdict(config),
                metrics={
                    **metrics,
                    "promotion_gate_passed": decision.passed,
                    "promotion_gate_reasons": decision.reasons,
                },
                notes="Strict point-in-time model-versus-no-vig-market benchmark",
            ),
            session=sess,
        )
        return EvaluationPipelineResult(metrics=metrics, decision=decision, evaluation_run_id=run.id)
    finally:
        if own_session:
            sess.close()
