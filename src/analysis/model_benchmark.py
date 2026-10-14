"""Unified model evaluation and promotion benchmark for professional LoL.

This module implements the standard benchmark protocol codified in AGENTS.md.
Every candidate model must be evaluated against this benchmark on a locked
walk-forward cohort before promotion or operational adoption.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.metrics import (
    DEFAULT_PROBABILITY_EPSILON,
    brier_decomposition,
    calculate_ece,
    clip_probabilities,
)


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    """Core probabilistic, discrimination, and calibration metrics."""

    sample_size: int
    log_loss: float
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    auc: float
    accuracy: float
    ece: float
    mce: float
    calibration_slope: float
    calibration_intercept: float


@dataclass(frozen=True)
class BaselineComparisonResult:
    """Paired statistical comparison against a locked comparative baseline."""

    baseline_name: str
    delta_log_loss: float
    ci_lower_log_loss: float
    ci_upper_log_loss: float
    p_value_log_loss: float
    is_statistically_significant: bool
    delta_brier: float
    delta_accuracy: float
    delta_auc: float


@dataclass(frozen=True)
class FailureModeSlice:
    """Granular metric record for risk cohorts (where the model errs)."""

    dimension: str
    bucket: str
    sample_size: int
    log_loss: float
    brier_score: float
    accuracy: float
    mean_predicted_prob: float
    observed_win_rate: float
    calibration_gap: float  # observed - predicted


@dataclass(frozen=True)
class PromotionGateResult:
    """Formal pass/fail evaluation against AGENTS.md promotion criteria."""

    passed: bool
    checklist: dict[str, bool]
    reasons: list[str]


@dataclass(frozen=True)
class ModelBenchmarkReport:
    """Comprehensive benchmark deliverable."""

    model_name: str
    model_version: str
    cohort_start: str | None
    cohort_end: str | None
    metrics: ModelBenchmarkSummary
    comparison: BaselineComparisonResult | None = None
    slices: dict[str, list[FailureModeSlice]] = field(default_factory=dict)
    promotion_gate: PromotionGateResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_markdown(self) -> str:
        """Render a clean Markdown summary report."""
        lines: list[str] = [
            f"# Model Benchmark Report: {self.model_name} ({self.model_version})",
            "",
            f"- **Cohort Period:** `{self.cohort_start or 'N/A'}` to `{self.cohort_end or 'N/A'}`",
            f"- **Sample Size:** {self.metrics.sample_size:,} matches",
            "",
            "## 1. Core Performance & Calibration",
            "",
            "| Metric | Value | Reference Target |",
            "| :--- | :---: | :---: |",
            f"| **LogLoss** | **{self.metrics.log_loss:.5f}** | Primary loss (min) |",
            f"| **Brier Score** | **{self.metrics.brier_score:.5f}** | Quadratic error (min) |",
            f"| -- Brier Reliability | {self.metrics.brier_reliability:.5f} | Calibration component (< 0.010) |",
            f"| -- Brier Resolution | {self.metrics.brier_resolution:.5f} | Discrimination component (max) |",
            f"| **ROC-AUC** | **{self.metrics.auc:.4f}** | Ranking quality (max) |",
            f"| **Accuracy (P >= 0.50)** | **{self.metrics.accuracy * 100:.2f}%** | Threshold 0.50 |",
            f"| **ECE (10 bins)** | **{self.metrics.ece:.4f}** | Calibration error (<= 0.030) |",
            f"| **MCE (Max Bin Error)** | {self.metrics.mce:.4f} | Worst-bin deviation |",
            f"| **Calibration Slope** | **{self.metrics.calibration_slope:.3f}** | Target 1.0 (range [0.85, 1.15]) |",
            f"| **Calibration Intercept** | {self.metrics.calibration_intercept:.3f} | Target 0.0 (bias) |",
            "",
        ]

        if self.comparison is not None:
            lines.extend(
                [
                    f"## 2. Paired Comparison vs {self.comparison.baseline_name}",
                    "",
                    f"- **Delta LogLoss:** `{self.comparison.delta_log_loss:+.6f}` (negative is better)",
                    f"- **95% Monthly-block Bootstrap CI:** `[{self.comparison.ci_lower_log_loss:+.6f}, {self.comparison.ci_upper_log_loss:+.6f}]`",
                    f"- **p-value (one-sided):** `{self.comparison.p_value_log_loss:.4f}`",
                    f"- **Statistically Superior:** `{'YES (CI entirely < 0)' if self.comparison.is_statistically_significant else 'NO (CI contains >= 0)'}`",
                    f"- **Delta Brier:** `{self.comparison.delta_brier:+.6f}`",
                    f"- **Delta AUC:** `{self.comparison.delta_auc:+.4f}`",
                    f"- **Delta Accuracy:** `{self.comparison.delta_accuracy * 100:+.2f} p.p.`",
                    "",
                ]
            )

        if self.slices:
            lines.extend(["## 3. Diagnostic Slices ('Gdzie model się myli')", ""])
            for dim, slice_items in self.slices.items():
                lines.extend(
                    [
                        f"### Slice: {dim}",
                        "",
                        "| Bucket | N | LogLoss | Brier | Acc | Pred P | Obs Win | Calib Gap |",
                        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    ]
                )
                for s in slice_items:
                    lines.append(
                        f"| {s.bucket} | {s.sample_size:,} | {s.log_loss:.4f} | {s.brier_score:.4f} | "
                        f"{s.accuracy * 100:.1f}% | {s.mean_predicted_prob:.3f} | {s.observed_win_rate:.3f} | "
                        f"{s.calibration_gap:+.3f} |"
                    )
                lines.append("")

        if self.promotion_gate is not None:
            status_str = "**PASSED (Ready for promotion)**" if self.promotion_gate.passed else "**FAILED (Promotion blocked)**"
            lines.extend(
                [
                    "## 4. Promotion Gate Evaluation",
                    "",
                    f"**Status:** {status_str}",
                    "",
                    "| Gate Criterion | Status |",
                    "| :--- | :---: |",
                ]
            )
            for criterion, ok in self.promotion_gate.checklist.items():
                lines.append(f"| {criterion} | {'PASS' if ok else 'FAIL'} |")
            lines.append("")
            if self.promotion_gate.reasons:
                lines.append("**Evaluation Notes & Gate Diagnostics:**")
                for r in self.promotion_gate.reasons:
                    lines.append(f"- {r}")
                lines.append("")

        return "\n".join(lines)


def calculate_calibration_parameters(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    probability_epsilon: float | None = 1e-6,
) -> tuple[float, float]:
    """Fit unregularized logistic calibration regression logit(y) = alpha + beta * logit(p)."""
    p_clipped = (np.asarray(probabilities, dtype=float) if probability_epsilon is None
                 else clip_probabilities(probabilities, epsilon=probability_epsilon))
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    
    # Use very low L2 penalty (C=1e9) to obtain unregularized maximum likelihood estimates
    clf = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    try:
        clf.fit(logits, y_true)
        slope = float(clf.coef_[0][0])
        intercept = float(clf.intercept_[0])
    except Exception:
        slope = float("nan")
        intercept = float("nan")
    return slope, intercept


def calculate_mce(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Maximum Calibration Error across populated confidence bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (probabilities > lower) & (probabilities <= upper)
        if not np.any(in_bin):
            continue
        err = abs(float(np.mean(y_true[in_bin])) - float(np.mean(probabilities[in_bin])))
        if err > max_error:
            max_error = err
    return max_error


def compute_benchmark_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    probability_epsilon: float | None = DEFAULT_PROBABILITY_EPSILON,
) -> ModelBenchmarkSummary:
    """Compute complete benchmark metric summary for a single prediction stream."""
    y = np.asarray(y_true, dtype=int)
    p = (np.asarray(probabilities, dtype=float) if probability_epsilon is None
         else clip_probabilities(probabilities, epsilon=probability_epsilon))
    if probability_epsilon is None and (not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1))):
        raise ValueError("Unclipped benchmark requires finite probabilities strictly inside(0,1)")
    n = len(y)

    ll = float(-np.mean(np.where(y, np.log(p), np.log1p(-p)))) if probability_epsilon is None else float(log_loss(y, p))
    bs = float(brier_score_loss(y, p))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    acc = float(accuracy_score(y, p >= 0.5))

    brier_decomp = brier_decomposition(pd.Series(y), pd.Series(p), n_bins=10)
    ece = calculate_ece(y, p, n_bins=10)
    mce = calculate_mce(y, p, n_bins=10)
    slope, intercept = calculate_calibration_parameters(
        y, p, probability_epsilon=None if probability_epsilon is None else 1e-6)

    return ModelBenchmarkSummary(
        sample_size=n,
        log_loss=ll,
        brier_score=bs,
        brier_reliability=brier_decomp.get("brier_reliability", 0.0),
        brier_resolution=brier_decomp.get("brier_resolution", 0.0),
        brier_uncertainty=brier_decomp.get("brier_uncertainty", 0.0),
        auc=auc,
        accuracy=acc,
        ece=ece,
        mce=mce,
        calibration_slope=slope,
        calibration_intercept=intercept,
    )


def compute_baseline_comparison(
    data: pd.DataFrame,
    candidate_prob_col: str,
    baseline_prob_col: str,
    target_col: str = "y_true",
    date_col: str = "date",
    baseline_name: str = "Frozen Baseline",
    n_bootstraps: int = 5000,
) -> BaselineComparisonResult:
    """Compute paired per-match delta metrics with monthly-block bootstrap CI."""
    sub = data[[candidate_prob_col, baseline_prob_col, target_col, date_col]].dropna().copy()
    y = sub[target_col].astype(int).to_numpy()
    p_cand = clip_probabilities(sub[candidate_prob_col].to_numpy(dtype=float))
    p_base = clip_probabilities(sub[baseline_prob_col].to_numpy(dtype=float))

    cand_loss = -(y * np.log(p_cand) + (1 - y) * np.log(1.0 - p_cand))
    base_loss = -(y * np.log(p_base) + (1 - y) * np.log(1.0 - p_base))
    sub["delta_logloss"] = cand_loss - base_loss

    observed, ci_low, ci_high, boot_samples = monthly_block_bootstrap_delta(
        sub,
        delta_column="delta_logloss",
        date_column=date_col,
        n_bootstraps=n_bootstraps,
    )

    p_val = float((boot_samples >= 0.0).mean())
    is_sig = bool(ci_high < 0.0)

    cand_brier = float(brier_score_loss(y, p_cand))
    base_brier = float(brier_score_loss(y, p_base))
    cand_acc = float(accuracy_score(y, p_cand >= 0.5))
    base_acc = float(accuracy_score(y, p_base >= 0.5))
    cand_auc = float(roc_auc_score(y, p_cand))
    base_auc = float(roc_auc_score(y, p_base))

    return BaselineComparisonResult(
        baseline_name=baseline_name,
        delta_log_loss=observed,
        ci_lower_log_loss=ci_low,
        ci_upper_log_loss=ci_high,
        p_value_log_loss=p_val,
        is_statistically_significant=is_sig,
        delta_brier=cand_brier - base_brier,
        delta_accuracy=cand_acc - base_acc,
        delta_auc=cand_auc - base_auc,
    )


def compute_failure_mode_slices(
    data: pd.DataFrame,
    prob_col: str,
    target_col: str = "y_true",
    bon_col: str | None = None,
    tier_col: str | None = None,
    odds_col: str | None = None,
    disagreement_col: str | None = None,
) -> dict[str, list[FailureModeSlice]]:
    """Compute segmented metric slices to detect specific failure modes."""
    slices: dict[str, list[FailureModeSlice]] = {}

    sub = data.dropna(subset=[prob_col, target_col]).copy()
    y = sub[target_col].astype(int).to_numpy()
    p = clip_probabilities(sub[prob_col].to_numpy(dtype=float))
    sub["_p_eval"] = p

    # 1. Probability / Odds Confidence Bins
    p_bins = [
        ("Heavy Underdog (P < 0.25)", p < 0.25),
        ("Moderate Underdog (0.25 <= P < 0.40)", (p >= 0.25) & (p < 0.40)),
        ("Coin-Flip / Close (0.40 <= P <= 0.60)", (p >= 0.40) & (p <= 0.60)),
        ("Moderate Favorite (0.60 < P <= 0.75)", (p > 0.60) & (p <= 0.75)),
        ("Heavy Favorite (P > 0.75)", p > 0.75),
    ]
    prob_slices: list[FailureModeSlice] = []
    for label, mask in p_bins:
        if not np.any(mask):
            continue
        y_b = y[mask]
        p_b = p[mask]
        obs_win = float(np.mean(y_b))
        pred_p = float(np.mean(p_b))
        prob_slices.append(
            FailureModeSlice(
                dimension="Confidence Level",
                bucket=label,
                sample_size=int(len(y_b)),
                log_loss=float(log_loss(y_b, p_b, labels=[0, 1])),
                brier_score=float(brier_score_loss(y_b, p_b)),
                accuracy=float(accuracy_score(y_b, p_b >= 0.5)),
                mean_predicted_prob=pred_p,
                observed_win_rate=obs_win,
                calibration_gap=obs_win - pred_p,
            )
        )
    slices["Confidence / Expected Prob"] = prob_slices

    # 2. Underdog Odds Quarantine Zone [3.50, 5.00] if odds are provided
    if odds_col is not None and odds_col in sub.columns:
        odds = sub[odds_col].to_numpy(dtype=float)
        odds_bins = [
            ("Heavy Favorite (Odds < 1.33)", (odds > 1.0) & (odds < 1.33)),
            ("Moderate Favorite (1.33 <= Odds < 1.67)", (odds >= 1.33) & (odds < 1.67)),
            ("Toss-Up (1.67 <= Odds <= 2.20)", (odds >= 1.67) & (odds <= 2.20)),
            ("Moderate Underdog (2.20 < Odds <= 3.50)", (odds > 2.20) & (odds <= 3.50)),
            ("Quarantine Underdog (3.50 < Odds <= 5.00)", (odds > 3.50) & (odds <= 5.00)),
            ("Extreme Underdog (Odds > 5.00)", odds > 5.00),
        ]
        odds_slices: list[FailureModeSlice] = []
        for label, mask in odds_bins:
            if not np.any(mask):
                continue
            y_b = y[mask]
            p_b = p[mask]
            obs_win = float(np.mean(y_b))
            pred_p = float(np.mean(p_b))
            odds_slices.append(
                FailureModeSlice(
                    dimension="Market Odds Tier",
                    bucket=label,
                    sample_size=int(len(y_b)),
                    log_loss=float(log_loss(y_b, p_b, labels=[0, 1])),
                    brier_score=float(brier_score_loss(y_b, p_b)),
                    accuracy=float(accuracy_score(y_b, p_b >= 0.5)),
                    mean_predicted_prob=pred_p,
                    observed_win_rate=obs_win,
                    calibration_gap=obs_win - pred_p,
                )
            )
        slices["Market Odds Tier"] = odds_slices

    # 3. Series Format (Bo1, Bo3, Bo5) if available
    if bon_col is not None and bon_col in sub.columns:
        bon_slices: list[FailureModeSlice] = []
        for bon_val, group in sub.groupby(bon_col):
            y_b = group[target_col].astype(int).to_numpy()
            p_b = group["_p_eval"].to_numpy(dtype=float)
            obs_win = float(np.mean(y_b))
            pred_p = float(np.mean(p_b))
            bon_slices.append(
                FailureModeSlice(
                    dimension="Series Format (BoN)",
                    bucket=f"Bo{bon_val}",
                    sample_size=int(len(y_b)),
                    log_loss=float(log_loss(y_b, p_b, labels=[0, 1])),
                    brier_score=float(brier_score_loss(y_b, p_b)),
                    accuracy=float(accuracy_score(y_b, p_b >= 0.5)),
                    mean_predicted_prob=pred_p,
                    observed_win_rate=obs_win,
                    calibration_gap=obs_win - pred_p,
                )
            )
        slices["Series Format (BoN)"] = bon_slices

    # 4. Competition Tier if available
    if tier_col is not None and tier_col in sub.columns:
        tier_slices: list[FailureModeSlice] = []
        for tier_val, group in sub.groupby(tier_col):
            y_b = group[target_col].astype(int).to_numpy()
            p_b = group["_p_eval"].to_numpy(dtype=float)
            obs_win = float(np.mean(y_b))
            pred_p = float(np.mean(p_b))
            tier_slices.append(
                FailureModeSlice(
                    dimension="Competition Tier",
                    bucket=str(tier_val),
                    sample_size=int(len(y_b)),
                    log_loss=float(log_loss(y_b, p_b, labels=[0, 1])),
                    brier_score=float(brier_score_loss(y_b, p_b)),
                    accuracy=float(accuracy_score(y_b, p_b >= 0.5)),
                    mean_predicted_prob=pred_p,
                    observed_win_rate=obs_win,
                    calibration_gap=obs_win - pred_p,
                )
            )
        slices["Competition Tier"] = tier_slices

    # 5. Signal Disagreement if available (|P_player - P_team|)
    if disagreement_col is not None and disagreement_col in sub.columns:
        disag = sub[disagreement_col].to_numpy(dtype=float)
        disag_bins = [
            ("High Agreement (Delta <= 0.05)", disag <= 0.05),
            ("Moderate Agreement (0.05 < Delta <= 0.12)", (disag > 0.05) & (disag <= 0.12)),
            ("High Disagreement (Delta > 0.12)", disag > 0.12),
            ("Severe Disagreement (Delta > 0.20)", disag > 0.20),
        ]
        disag_slices: list[FailureModeSlice] = []
        for label, mask in disag_bins:
            if not np.any(mask):
                continue
            y_b = y[mask]
            p_b = p[mask]
            obs_win = float(np.mean(y_b))
            pred_p = float(np.mean(p_b))
            disag_slices.append(
                FailureModeSlice(
                    dimension="Player-Team Disagreement",
                    bucket=label,
                    sample_size=int(len(y_b)),
                    log_loss=float(log_loss(y_b, p_b, labels=[0, 1])),
                    brier_score=float(brier_score_loss(y_b, p_b)),
                    accuracy=float(accuracy_score(y_b, p_b >= 0.5)),
                    mean_predicted_prob=pred_p,
                    observed_win_rate=obs_win,
                    calibration_gap=obs_win - pred_p,
                )
            )
        slices["Player-Team Disagreement"] = disag_slices

    return slices


def evaluate_promotion_gate(
    metrics: ModelBenchmarkSummary,
    comparison: BaselineComparisonResult | None = None,
    slices: dict[str, list[FailureModeSlice]] | None = None,
    baseline_ece: float = 0.030,
) -> PromotionGateResult:
    """Evaluate formal promotion checklist according to AGENTS.md."""
    checklist: dict[str, bool] = {}
    reasons: list[str] = []

    # 1. Calibration Slope
    slope_ok = 0.85 <= metrics.calibration_slope <= 1.15
    checklist["Calibration slope within [0.85, 1.15]"] = slope_ok
    if not slope_ok:
        reasons.append(
            f"Calibration slope {metrics.calibration_slope:.3f} outside acceptable range [0.85, 1.15]"
        )

    # 2. Expected Calibration Error
    ece_ok = metrics.ece <= max(0.035, baseline_ece + 0.002)
    checklist["ECE within threshold (<= baseline ECE)"] = ece_ok
    if not ece_ok:
        reasons.append(f"ECE {metrics.ece:.4f} exceeds baseline reference {baseline_ece:.4f}")

    # 3. Statistical superiority if comparison provided
    if comparison is not None:
        stat_ok = comparison.is_statistically_significant and (comparison.delta_log_loss < 0.0)
        checklist["Statistically superior (95% Bootstrap CI upper bound < 0)"] = stat_ok
        if not stat_ok:
            reasons.append(
                f"95% bootstrap CI [{comparison.ci_lower_log_loss:+.6f}, {comparison.ci_upper_log_loss:+.6f}] does not prove significant superiority"
            )
    else:
        checklist["Statistically superior (95% Bootstrap CI upper bound < 0)"] = True

    # 4. Underdog quarantine safety (check slices if available)
    underdog_safe = True
    if slices and "Market Odds Tier" in slices:
        for s in slices["Market Odds Tier"]:
            if "Quarantine Underdog" in s.bucket and s.mean_predicted_prob > 0.45:
                underdog_safe = False
                reasons.append(
                    f"Quarantine underdog tier ({s.bucket}) has inflated predicted win prob {s.mean_predicted_prob:.3f}"
                )
    checklist["Underdog quarantine safety (no false favorites on odds > 3.50)"] = underdog_safe

    passed = all(checklist.values())
    return PromotionGateResult(passed=passed, checklist=checklist, reasons=reasons)


def run_model_benchmark(
    data: pd.DataFrame,
    candidate_prob_col: str,
    baseline_prob_col: str | None = None,
    target_col: str = "y_true",
    date_col: str = "date",
    model_name: str = "Candidate Model",
    model_version: str = "v1.0",
    baseline_name: str = "Frozen Baseline",
    bon_col: str | None = None,
    tier_col: str | None = None,
    odds_col: str | None = None,
    disagreement_col: str | None = None,
    n_bootstraps: int = 5000,
) -> ModelBenchmarkReport:
    """Run full benchmark pipeline on an evaluated dataset."""
    sub = data.dropna(subset=[candidate_prob_col, target_col]).copy()
    y = sub[target_col].astype(int).to_numpy()
    p_cand = sub[candidate_prob_col].to_numpy(dtype=float)

    metrics = compute_benchmark_summary(y, p_cand)

    comparison: BaselineComparisonResult | None = None
    if baseline_prob_col is not None and baseline_prob_col in sub.columns:
        comparison = compute_baseline_comparison(
            sub,
            candidate_prob_col=candidate_prob_col,
            baseline_prob_col=baseline_prob_col,
            target_col=target_col,
            date_col=date_col,
            baseline_name=baseline_name,
            n_bootstraps=n_bootstraps,
        )

    slices = compute_failure_mode_slices(
        sub,
        prob_col=candidate_prob_col,
        target_col=target_col,
        bon_col=bon_col,
        tier_col=tier_col,
        odds_col=odds_col,
        disagreement_col=disagreement_col,
    )

    baseline_ece = 0.030
    if baseline_prob_col is not None and baseline_prob_col in sub.columns:
        p_base = sub[baseline_prob_col].to_numpy(dtype=float)
        baseline_ece = calculate_ece(y, clip_probabilities(p_base), n_bins=10)

    promotion_gate = evaluate_promotion_gate(
        metrics=metrics,
        comparison=comparison,
        slices=slices,
        baseline_ece=baseline_ece,
    )

    cohort_dates = sorted(sub[date_col].dropna().astype(str).unique()) if date_col in sub.columns else []
    cohort_start = cohort_dates[0] if cohort_dates else None
    cohort_end = cohort_dates[-1] if cohort_dates else None

    return ModelBenchmarkReport(
        model_name=model_name,
        model_version=model_version,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        metrics=metrics,
        comparison=comparison,
        slices=slices,
        promotion_gate=promotion_gate,
    )
