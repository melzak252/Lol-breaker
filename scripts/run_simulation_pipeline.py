#!/usr/bin/env python3
"""Unified CLI Runner for LoL-Breaker 5-Stage Simulation Pipeline.

Supports granular stage execution (--stage {stage_name}) and end-to-end runs (--stage all).
Enforces Kedro 8-layer architecture and integrates with DVC DAG workflows.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.simulation_pipeline import (
    consolidate_canonical_bundle,
    run_stage1_verify_inputs,
    run_stage2_model_benchmark,
    run_stage3_betting_simulation,
    run_stage4_econometric_audit,
    run_stage5_generate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Kedro 8-layer model & betting simulation pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "consolidate",
            "verify_inputs",
            "model_benchmark",
            "betting_simulation",
            "econometric_audit",
            "generate_report",
            "all",
        ],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "conf/base/catalog.yml",
        help="Path to Kedro-style data catalog (default: conf/base/catalog.yml).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "conf/base/parameters.yml",
        help="Path to simulation parameters (default: conf/base/parameters.yml).",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=None,
        help="Optional path to external research archive if not yet consolidated.",
    )

    args = parser.parse_args()
    catalog_path = args.catalog.resolve()
    params_path = args.params.resolve()

    start_time = time.perf_counter()
    print("=" * 78)
    print(f"LoL-Breaker Simulation Pipeline Runner | Stage: {args.stage.upper()}")
    print(f"Catalog:    {catalog_path.relative_to(PROJECT_ROOT) if catalog_path.is_relative_to(PROJECT_ROOT) else catalog_path}")
    print(f"Parameters: {params_path.relative_to(PROJECT_ROOT) if params_path.is_relative_to(PROJECT_ROOT) else params_path}")
    print("=" * 78)

    try:
        if args.stage == "consolidate":
            res = consolidate_canonical_bundle(PROJECT_ROOT, external_root=args.external_root)
            print("\n[Stage: Consolidate Canonical Bundle]")
            print(f"Target Directory: {res['target_directory']}")
            print(f"Pointer File:     {res['pointer_file']} -> {res['pointer_value']}")
            print(f"Files Copied:     {res['files_copied']}")
            print(f"Files Verified:   {res['files_verified']}")
            print(f"Total Size:       {res['total_megabytes']} MB")

        elif args.stage == "verify_inputs":
            res = run_stage1_verify_inputs(catalog_path, params_path)
            print("\n[Stage 1: Verify Inputs]")
            print(f"Status:        {res['status']}")
            print(f"Doctor Suite:  {res['doctor_suite']['status']}")
            print(f"Datasets:      {len(res['datasets'])} verified")

        elif args.stage == "model_benchmark":
            res = run_stage2_model_benchmark(catalog_path, params_path)
            c0 = res["cohort_a_2026"]["Causal-C0 (W32/E12)"]
            a0 = res["cohort_a_2026"]["Annual Causal A0"]
            pin24 = res["pinnacle_horizon_comparison"]["24"]
            pin12 = res["pinnacle_horizon_comparison"]["12"]
            pol2p = res["polish_production_comparison"]["Polish2plus"]
            pol_refit = res["polish_production_comparison"]["monthly_refit_production"]
            print("\n[Stage 2: Model Benchmark & Multi-Horizon Bookmaker Comparison]")
            print(f"2026 C0 (W32/E12):  LogLoss = {c0['log_loss']:.6f}, Brier = {c0['brier_score']:.6f}")
            print(f"2026 Annual A0:     LogLoss = {a0['log_loss']:.6f}, Brier = {a0['brier_score']:.6f}")
            print(f"Full 11,550 A0:     LogLoss = {res['cohort_b_full_11550']['Annual Causal A0']['log_loss']:.6f}")
            print(f"Pinnacle T-24h:     Pure C0 {pin24['c0_log_loss']:.6f} vs Pinnacle {pin24['pinnacle_log_loss']:.6f} (C0 edge: {pin24['delta_ll_c0_vs_pinnacle']:+.6f} LL, N={pin24['n_matches']})")
            print(f"Pinnacle T-12h:     Pure C0 {pin12['c0_log_loss']:.6f} vs Pinnacle {pin12['pinnacle_log_loss']:.6f} (C0 edge: {pin12['delta_ll_c0_vs_pinnacle']:+.6f} LL, N={pin12['n_matches']})")
            print(f"Polish 2+ Panel:    Hybrid-50 {pol2p['hybrid_log_loss']:.6f} vs Market {pol2p['market_log_loss']:.6f} (Hybrid edge: {pol2p['delta_ll_hybrid_vs_market']:+.6f} LL, N={pol2p['n_matches']})")
            print(f"Polish Live Refit:  Hybrid-50 {pol_refit['hybrid_log_loss']:.6f} vs Market {pol_refit['market_log_loss']:.6f} (Hybrid edge: {pol_refit['delta_ll_hybrid_vs_market']:+.6f} LL, N={pol_refit['n_matches']})")
            print("Shrinkage Grid:     data/08_reporting/simulation_pipeline/stage2_shrinkage_grid.csv generated")
            print("Horizon Comparison: data/08_reporting/simulation_pipeline/stage2_bookmaker_horizon_comparison.csv generated")

        elif args.stage == "betting_simulation":
            res = run_stage3_betting_simulation(catalog_path, params_path)
            opt = res["optimal_synergy_summary"]
            print("\n[Stage 3: Betting Simulation]")
            print(f"Optimal Synergy Bets: {opt['n_bets']} (Wins: {opt['n_wins']}, Win Rate: {opt['win_rate_pct']}%)")
            print(f"Financial Yield:      +{opt['realized_yield_pct']}% (Turnover: {opt['total_turnover_pln']} PLN)")
            print(f"Bankroll ROI:         +{opt['bankroll_roi_pct']}% (Profit: +{opt['total_net_profit_pln']} PLN)")
            print(f"CLV Performance:      Raw CLV = +{opt['mean_raw_clv_pct']}%, Net CLV = +{opt['mean_net_clv_pct']}%")
            print(f"Max Drawdown:         {opt['max_drawdown_pct']}%")

        elif args.stage == "econometric_audit":
            res = run_stage4_econometric_audit(catalog_path, params_path)
            decomp = res["decomposition"]
            hyp = res["hypothesis_testing"]
            t2p = res["time_to_profit"]
            print("\n[Stage 4: Econometric Audit]")
            print(f"Identity Check:       Discrepancy = {decomp['max_identity_discrepancy']:.2e}")
            print(f"Pure C0 Claimed EV:   +{decomp['pure_c0_claimed_ev_pct']}%")
            print(f"Shrunk Hybrid EV:     +{decomp['shrunk_hybrid_claimed_ev_pct']}%")
            print(f"Curse Removed:        {decomp['curse_removed_by_shrinkage_pp']} pp")
            print(f"True Net CLV (Edge):  +{decomp['true_long_term_edge_net_clv_pct']}% (t = {hyp['net_clv']['t_statistic']}, p = {hyp['net_clv']['p_value']})")
            print(f"Match Outcome Noise:  +{decomp['match_outcome_noise_pp']} pp")
            print(f"Variance Ratio:       {hyp['variance_ratio']}x (Match outcome vs Net CLV)")
            print(f"Time-to-Profit N_min: {t2p['n_min_12pct_tax']} bets (12% tax) | {t2p['n_min_0pct_tax']} bets (0% tax)")

        elif args.stage == "generate_report":
            res = run_stage5_generate_report(catalog_path, params_path)
            print("\n[Stage 5: Generate Report & Metrics]")
            print(f"Status:       {res['status']}")
            print(f"Report:       {res['report_path']}")
            print(f"Metrics:      {res['metrics_path']}")

        elif args.stage == "all":
            # 0. Ensure consolidation
            consolidate_canonical_bundle(PROJECT_ROOT, external_root=args.external_root)

            # 1. Stage 1
            s1 = run_stage1_verify_inputs(catalog_path, params_path)
            print("\n[1/5] Stage 1 Verified: Doctor Suite is READY, all catalog datasets intact.")

            # 2. Stage 2
            s2 = run_stage2_model_benchmark(catalog_path, params_path)
            c0 = s2["cohort_a_2026"]["Causal-C0 (W32/E12)"]
            pin24 = s2["pinnacle_horizon_comparison"]["24"]
            pin12 = s2["pinnacle_horizon_comparison"]["12"]
            pol2p = s2["polish_production_comparison"]["Polish2plus"]
            print(f"[2/5] Stage 2 Benchmarked: 2026 C0 LogLoss = {c0['log_loss']:.6f}, Pinnacle T-24h C0 edge = {pin24['delta_ll_c0_vs_pinnacle']:+.6f} LL, T-12h C0 edge = {pin12['delta_ll_c0_vs_pinnacle']:+.6f} LL, Polish 2+ Hybrid edge = {pol2p['delta_ll_hybrid_vs_market']:+.6f} LL")

            # 3. Stage 3
            s3 = run_stage3_betting_simulation(catalog_path, params_path)
            opt = s3["optimal_synergy_summary"]
            print(f"[3/5] Stage 3 Simulated: {opt['n_bets']} bets, WinRate = {opt['win_rate_pct']}%, Yield = +{opt['realized_yield_pct']}%, Net CLV = +{opt['mean_net_clv_pct']}%")

            # 4. Stage 4
            s4 = run_stage4_econometric_audit(catalog_path, params_path)
            decomp = s4["decomposition"]
            hyp = s4["hypothesis_testing"]
            print(f"[4/5] Stage 4 Audited: True Net CLV = +{decomp['true_long_term_edge_net_clv_pct']}% (t={hyp['net_clv']['t_statistic']}, p={hyp['net_clv']['p_value']}), N_min = {s4['time_to_profit']['n_min_12pct_tax']} bets")

            # 5. Stage 5
            s5 = run_stage5_generate_report(catalog_path, params_path)
            print(f"[5/5] Stage 5 Complete: Report written to {s5['report_path']}")

        elapsed = time.perf_counter() - start_time
        print("\n" + "=" * 78)
        print(f"Pipeline finished successfully in {elapsed:.2f} seconds.")
        print("=" * 78)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline stage '{args.stage}' failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
