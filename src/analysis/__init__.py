"""Analysis utilities for metrics, EDA, and market diagnostics."""

from src.analysis.financial_benchmark import (
    FinancialBenchmarkReport,
    FinancialBenchmarkSummary,
    FinancialSlice,
    calculate_quarter_kelly,
    run_financial_benchmark,
)
from src.analysis.model_benchmark import (
    BaselineComparisonResult,
    FailureModeSlice,
    ModelBenchmarkReport,
    ModelBenchmarkSummary,
    PromotionGateResult,
    compute_baseline_comparison,
    compute_benchmark_summary,
    compute_failure_mode_slices,
    evaluate_promotion_gate,
    run_model_benchmark,
)
from src.analysis.odds_bracket_benchmark import (
    DEFAULT_ODDS_BRACKETS,
    BracketCalibrationStats,
    OddsBenchmarkReport,
    OddsBracketTier,
    evaluate_odds_bracket_benchmark,
)

__all__ = [
    "BaselineComparisonResult",
    "BracketCalibrationStats",
    "DEFAULT_ODDS_BRACKETS",
    "FailureModeSlice",
    "ModelBenchmarkReport",
    "ModelBenchmarkSummary",
    "OddsBenchmarkReport",
    "OddsBracketTier",
    "PromotionGateResult",
    "FinancialBenchmarkReport",
    "FinancialBenchmarkSummary",
    "FinancialSlice",
    "calculate_quarter_kelly",
    "run_financial_benchmark",
    "compute_baseline_comparison",
    "compute_benchmark_summary",
    "compute_failure_mode_slices",
    "evaluate_odds_bracket_benchmark",
    "evaluate_promotion_gate",
    "run_model_benchmark",
]
