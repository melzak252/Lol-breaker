"""Automated End-to-End Test Suite for Kedro 8-Layer Simulation Pipeline.

Verifies:
1. Idempotent canonical research bundle consolidation & doctor status
2. Model benchmark proper scores across 2026 assessment & 11,550 walk-forward
3. Bayesian shrinkage grid optimality
4. Betting simulation, Polish production replay, and Optimal Synergy ledger
5. Econometric audit: Net CLV significance, exact 3-part identity, and time-to-profit
6. CLI execution and reporting outputs
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.simulation_pipeline import (
    PROJECT_ROOT,
    consolidate_canonical_bundle,
    run_stage1_verify_inputs,
    run_stage2_model_benchmark,
    run_stage3_betting_simulation,
    run_stage4_econometric_audit,
    run_stage5_generate_report,
)


def test_consolidate_canonical_bundle_idempotent() -> None:
    """Consolidate external research files into data/05_model_input/canonical_benchmark."""
    res = consolidate_canonical_bundle(PROJECT_ROOT)
    assert res["status"] == "consolidated"
    assert res["pointer_value"] == "data/05_model_input/canonical_benchmark"
    assert res["files_verified"] >= 380

    target_dir = Path(res["target_directory"])
    assert (target_dir / "rating-foundation-20260915/data/08_reporting/paired.parquet").is_file()
    assert (target_dir / "rating-foundation-20260915/STATUS.json").is_file()
    assert (target_dir / "a0-phase-walkforward-20260914/RAPORT.md").is_file()
    assert (target_dir / "auxiliary-player-20260912/data/03_primary/targets.npy").is_file()
    assert (target_dir / "unified-ratings-20260911/data/04_feature/roster-optional-v1/metadata.jsonl").is_file()

    # Second run must be completely idempotent (0 files copied)
    res_second = consolidate_canonical_bundle(PROJECT_ROOT)
    assert res_second["files_copied"] == 0
    assert res_second["files_verified"] == res["files_verified"]


def test_stage1_verify_inputs() -> None:
    """Stage 1: Verify data catalog, SHA-256 hashes, and canonical benchmark doctor suite."""
    manifest = run_stage1_verify_inputs()
    assert manifest["status"] == "VERIFIED"
    assert manifest["doctor_suite"]["status"] == "READY"
    assert manifest["doctor_suite"]["suite"] == "causal-a0-rating-controls-v1"

    # Pinned catalog datasets must all be verified
    datasets = manifest["datasets"]
    assert "canonical_benchmark_paired" in datasets
    assert "c0_2026_assessment_joined" in datasets
    assert "c0_2026_market_t1_joined" in datasets
    assert "production_2h_trajectory" in datasets
    assert "optimal_synergy_ledger" in datasets
    assert "prop_distribution_bundle" in datasets
    assert all(d["verified"] for d in datasets.values())


def test_stage2_model_benchmark_invariants() -> None:
    """Stage 2: Verify exact benchmark invariants across 2026 assessment & 11,550 walk-forward."""
    res = run_stage2_model_benchmark()

    # 1. 2026 Canonical Assessment (3,103 series)
    cohort_a = res["cohort_a_2026"]
    c0 = cohort_a["Causal-C0 (W32/E12)"]
    a0 = cohort_a["Annual Causal A0"]
    glicko = cohort_a["Calibrated Glicko-2"]
    elo = cohort_a["Calibrated Elo"]

    assert c0["sample_size"] == 3103
    assert c0["log_loss"] == pytest.approx(0.535449, abs=1e-5)
    assert c0["brier_score"] == pytest.approx(0.181034, abs=1e-5)
    assert c0["accuracy"] == pytest.approx(0.7203, abs=1e-3)
    assert c0["tails_over_threshold"] == 12
    assert c0["display_name"] == "PlayerHistory-Attention-Ensemble"
    assert c0["short_code"] == "C0"
    assert res["models_by_semantic_name"]["PlayerHistory-Attention-Ensemble"]["log_loss"] == c0["log_loss"]

    assert a0["sample_size"] == 3103
    assert a0["log_loss"] == pytest.approx(0.536757, abs=1e-5)
    assert a0["brier_score"] == pytest.approx(0.181573, abs=1e-5)
    assert a0["display_name"] == "WalkForward-Reference-Ensemble"
    assert a0["short_code"] == "A0"
    assert glicko["log_loss"] == pytest.approx(0.570519, abs=1e-5)
    assert elo["log_loss"] == pytest.approx(0.580481, abs=1e-5)

    # 2. Full 11,550 Walk-Forward Cohort
    cohort_b_full = res["cohort_b_full_11550"]
    a0_full = cohort_b_full["Annual Causal A0"]
    assert a0_full["sample_size"] == 11550
    assert a0_full["log_loss"] == pytest.approx(0.551246, abs=1e-5)
    assert a0_full["brier_score"] == pytest.approx(0.187103, abs=1e-5)
    assert a0_full["tails_over_threshold"] == 33

    # 3. Common 9,907 Support Cohort with EXP-039
    cohort_b_common = res["cohort_b_common_9907"]
    exp039 = cohort_b_common["EXP-039 Research Derivative"]
    a0_common = cohort_b_common["Annual Causal A0"]
    assert exp039["sample_size"] == 9907
    assert exp039["log_loss"] == pytest.approx(0.567337, abs=1e-5)
    assert exp039["tails_over_threshold"] == 16
    assert a0_common["tails_over_threshold"] == 30
    assert exp039["display_name"] == "ElasticNet-TeamForm-Baseline"
    assert exp039["short_code"] == "EXP-039"

    # 4. Bayesian Shrinkage Grid
    grid_csv = PROJECT_ROOT / "data/08_reporting/simulation_pipeline/stage2_shrinkage_grid.csv"
    assert grid_csv.is_file()


def test_stage3_betting_simulation_invariants() -> None:
    """Stage 3: Verify production Polish betting simulation and Optimal Synergy strategy ledger."""
    res = run_stage3_betting_simulation()

    opt = res["optimal_synergy_summary"]
    assert opt["n_bets"] == 50
    assert opt["n_wins"] == 29
    assert opt["win_rate_pct"] == 58.0
    assert opt["mean_raw_clv_pct"] == pytest.approx(19.13, abs=0.1)
    assert opt["mean_net_clv_pct"] == pytest.approx(4.83, abs=0.1)
    assert opt["mean_model_ev_pct"] == pytest.approx(13.41, abs=0.1)
    assert opt["realized_yield_pct"] == pytest.approx(23.91, abs=0.2)
    assert opt["bankroll_roi_pct"] == pytest.approx(17.15, abs=0.2)
    assert opt["max_drawdown_pct"] <= 10.0

    # Monthly breakdown must cover 2026-05 through 2026-09
    months = [m["month"] for m in res["monthly_breakdown"]]
    assert months == ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
    assert sum(m["bets"] for m in res["monthly_breakdown"]) == 50

    # Strategy comparison file must exist and have realistic win rates (0% to 100%)
    strat_csv = PROJECT_ROOT / "data/08_reporting/simulation_pipeline/stage3_strategy_comparison.csv"
    assert strat_csv.is_file()
    strat_rows = res["strategy_comparison"]
    assert len(strat_rows) == 4
    for row in strat_rows:
        assert 0.0 <= row["win_rate_pct"] <= 100.0

    # Strict RFC 8259 JSON validity check: no bare NaN or Infinity tokens allowed
    stage3_json_path = PROJECT_ROOT / "data/08_reporting/simulation_pipeline/stage3_betting_simulation.json"
    with stage3_json_path.open("r", encoding="utf-8") as f:
        parsed = json.load(
            f,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid non-standard JSON constant: {c}")),
        )
    assert isinstance(parsed, dict)

def test_stage4_econometric_audit_invariants() -> None:
    """Stage 4: Verify exact three-part identity, hypothesis testing, variance ratio, and props."""
    res = run_stage4_econometric_audit()

    decomp = res["decomposition"]
    # Exact three-part identity: Realized Yield = Claimed EV - Curse B + Noise eta
    assert decomp["max_identity_discrepancy"] < 1e-9
    assert decomp["pure_c0_claimed_ev_pct"] == pytest.approx(22.05, abs=0.1)
    assert decomp["shrunk_hybrid_claimed_ev_pct"] == pytest.approx(13.41, abs=0.1)
    assert decomp["curse_removed_by_shrinkage_pp"] == pytest.approx(8.64, abs=0.1)
    assert decomp["residual_optimizers_curse_pp"] == pytest.approx(8.57, abs=0.1)
    assert decomp["match_outcome_noise_pp"] == pytest.approx(19.42, abs=0.1)
    assert decomp["true_long_term_edge_net_clv_pct"] == pytest.approx(4.83, abs=0.05)
    assert decomp["realized_yield_stake_weighted_pct"] == pytest.approx(23.91, abs=0.2)

    # Hypothesis testing
    hyp = res["hypothesis_testing"]
    net_clv_stats = hyp["net_clv"]
    assert net_clv_stats["point_estimate_pct"] == pytest.approx(4.83, abs=0.05)
    assert net_clv_stats["t_statistic"] == pytest.approx(3.00, abs=0.05)
    assert net_clv_stats["p_value"] < 0.005
    assert hyp["variance_ratio"] == pytest.approx(104.5, abs=0.5)

    # Time-to-profit
    t2p = res["time_to_profit"]
    assert t2p["n_min_12pct_tax"] == 1396
    assert t2p["n_min_0pct_tax"] == 478

    # Noise decay table
    noise_csv = PROJECT_ROOT / "data/08_reporting/simulation_pipeline/stage4_noise_decay_table.csv"
    assert noise_csv.is_file()

    # Prop bundle verification
    prop_ver = res["prop_distribution_verification"]
    assert prop_ver["bundle_verified"] is True
    assert prop_ver["metrics"]["kills_ridge1"]["nll"] == pytest.approx(3.590488, abs=1e-5)
    assert prop_ver["metrics"]["duration_tree15"]["nll"] == pytest.approx(3.076194, abs=1e-5)
    assert prop_ver["metrics"]["series_length_neural"]["nll"] == pytest.approx(0.714580, abs=1e-5)


def test_stage5_generate_report_and_cli() -> None:
    """Stage 5: Verify report generation, DVC summary_metrics.json, and CLI execution."""
    res = run_stage5_generate_report()
    assert res["status"] == "REPORT_GENERATED"

    report_path = Path(res["report_path"])
    metrics_path = Path(res["metrics_path"])
    assert report_path.is_file()
    assert metrics_path.is_file()

    # Verify DVC metrics schema
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["stage1_doctor_status"] == "READY"
    assert metrics["stage2_c0_2026_log_loss"] == pytest.approx(0.535449, abs=1e-5)
    assert metrics["stage3_production_bets_count"] == 50
    assert metrics["stage3_production_net_clv_pct"] == pytest.approx(4.83, abs=0.05)
    assert metrics["stage4_net_clv_t_stat"] == pytest.approx(3.00, abs=0.05)
    assert metrics["stage4_n_min_12pct_tax"] == 1396

    # Test CLI invocation with --stage all
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/run_simulation_pipeline.py"), "--stage", "all"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0
    assert "Pipeline finished successfully" in proc.stdout
