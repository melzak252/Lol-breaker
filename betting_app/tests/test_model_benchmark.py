"""Tests for the unified model evaluation and promotion benchmark suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.model_benchmark import (
    compute_baseline_comparison,
    compute_benchmark_summary,
    compute_failure_mode_slices,
    evaluate_promotion_gate,
    run_model_benchmark,
)


def test_compute_benchmark_summary_deterministic() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    # Well-aligned probabilities
    y_prob = np.array([0.8, 0.2, 0.9, 0.1, 0.7, 0.85, 0.15, 0.3])

    summary = compute_benchmark_summary(y_true, y_prob)

    assert summary.sample_size == 8
    assert summary.log_loss < 0.40
    assert summary.brier_score < 0.10
    assert summary.auc == 1.0  # Perfectly separated ranks
    assert summary.accuracy == 1.0  # 100% accuracy at threshold 0.5
    assert summary.ece < 0.25
    assert summary.calibration_slope > 0.0


def test_failure_mode_slices_and_odds_quarantine() -> None:
    data = pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0, 0, 1],
            "prob": [0.85, 0.15, 0.70, 0.45, 0.20, 0.35],
            "odds": [1.25, 6.00, 1.50, 2.00, 4.20, 2.80],
            "bon": [3, 1, 3, 3, 1, 5],
            "tier": ["Tier-1", "Tier-2", "Tier-1", "Tier-1", "Tier-2", "Tier-1"],
        }
    )

    slices = compute_failure_mode_slices(
        data,
        prob_col="prob",
        target_col="y_true",
        bon_col="bon",
        tier_col="tier",
        odds_col="odds",
    )

    assert "Confidence / Expected Prob" in slices
    assert "Market Odds Tier" in slices
    assert "Series Format (BoN)" in slices
    assert "Competition Tier" in slices

    # Check quarantine underdog slice
    quarantine_items = [s for s in slices["Market Odds Tier"] if "Quarantine" in s.bucket]
    assert len(quarantine_items) == 1
    assert quarantine_items[0].sample_size == 1
    assert quarantine_items[0].observed_win_rate == 0.0


def test_promotion_gate_evaluation() -> None:
    # Create realistic calibrated probabilities
    np.random.seed(42)
    n = 1000
    logits = np.random.normal(0.0, 1.2, size=n)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.uniform(0, 1, size=n) < p).astype(int)

    summary = compute_benchmark_summary(y, p)
    assert 0.85 <= summary.calibration_slope <= 1.15
    assert summary.ece <= 0.050

    # Baseline with slightly worse noise
    p_base = np.clip(p + np.random.normal(0.0, 0.2, size=n), 0.05, 0.95)
    df = pd.DataFrame(
        {
            "y_true": y,
            "candidate": p,
            "baseline": p_base,
            "date": [f"2024-{(i%12)+1:02d}-01" for i in range(n)],
        }
    )
    comp = compute_baseline_comparison(
        df,
        candidate_prob_col="candidate",
        baseline_prob_col="baseline",
        n_bootstraps=200,
    )

    gate_result = evaluate_promotion_gate(summary, comparison=comp, baseline_ece=0.040)
    assert gate_result.passed is True
    assert gate_result.checklist["Calibration slope within [0.85, 1.15]"] is True
    assert gate_result.checklist["Statistically superior (95% Bootstrap CI upper bound < 0)"] is True


def test_run_model_benchmark_end_to_end() -> None:
    np.random.seed(42)
    n = 200
    y = np.random.binomial(1, 0.5, size=n)
    p = np.clip(y * 0.4 + np.random.uniform(0.1, 0.5, size=n), 0.05, 0.95)
    dates = [f"2024-{(i%12)+1:02d}-15" for i in range(n)]

    df = pd.DataFrame({
        "y_true": y,
        "prob_cand": p,
        "prob_base": np.full(n, 0.5),
        "date": dates,
        "bon": np.random.choice([1, 3], size=n),
    })

    report = run_model_benchmark(
        data=df,
        candidate_prob_col="prob_cand",
        baseline_prob_col="prob_base",
        bon_col="bon",
        n_bootstraps=100,
    )

    md = report.format_markdown()
    assert "# Model Benchmark Report" in md
    assert "## 1. Core Performance & Calibration" in md
    assert "## 2. Paired Comparison vs Frozen Baseline" in md
    assert "## 3. Diagnostic Slices ('Gdzie model się myli')" in md
    assert "## 4. Promotion Gate Evaluation" in md
