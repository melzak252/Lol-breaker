#!/usr/bin/env python3
"""CLI runner for the standard walk-forward model benchmark.

Evaluates a candidate prediction column against ground truth and an optional
comparative baseline according to the AGENTS.md promotion benchmark.

Example:
    .venv/bin/python scripts/run_model_benchmark.py \
      --data data/golgg_y_predicts.csv \
      --candidate-col player_elo \
      --target-col y_true \
      --date-col date \
      --min-date 2024-01-01 \
      --model-name "Player-Elo" \
      --model-version "v1.0"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.model_benchmark import run_model_benchmark

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified model evaluation and promotion benchmark."
    )
    parser.add_argument("--data", help="Path to CSV or Parquet evaluation dataset.")
    parser.add_argument("--candidate-col", help="Candidate model prediction column.")
    parser.add_argument("--baseline-col", default=None, help="Comparative baseline prediction column.")
    parser.add_argument("--target-col", default="y_true", help="Ground truth binary label column (default: y_true).")
    parser.add_argument("--date-col", default="date", help="Match date column for temporal blocks (default: date).")
    parser.add_argument("--min-date", default="2024-01-01", help="Start date for the evaluation cohort (default: 2024-01-01).")
    parser.add_argument("--max-date", default=None, help="End date for the evaluation cohort.")
    parser.add_argument("--model-name", default="Candidate Model", help="Human-readable model name.")
    parser.add_argument("--model-version", default="v1.0", help="Model version string.")
    parser.add_argument("--baseline-name", default="Frozen Baseline", help="Baseline name.")
    parser.add_argument("--bon-col", default=None, help="Series format column (Bo1/Bo3/Bo5).")
    parser.add_argument("--tier-col", default=None, help="League/tournament tier column.")
    parser.add_argument("--odds-col", default=None, help="Bookmaker odds column.")
    parser.add_argument("--disagreement-col", default=None, help="Internal disagreement column.")
    parser.add_argument("--bootstraps", type=int, default=5000, help="Number of monthly bootstrap resamples (default: 5000).")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON benchmark output.")
    parser.add_argument("--suite", type=Path, nargs="?", const=PROJECT_ROOT / "conf/base/research_benchmark.json", help="Versioned research manifest; canonical multi-model mode.")
    parser.add_argument("--research-root", type=Path, help="Directory containing the research artifacts (or ENSEMBLE_RESEARCH_ROOT).")
    parser.add_argument("--output-dir", type=Path, help="New output directory for a suite run.")
    parser.add_argument("--doctor", action="store_true", help="Check suite inputs/hashes without scoring.")
    parser.add_argument("--candidate-data", type=Path, help="Optional complete CSV/Parquet candidate predictions for the locked cohort.")
    parser.add_argument("--thesis-audit", action="store_true", help="Run the corrected monthly thesis replay.")
    parser.add_argument("--thesis-features", type=Path)
    parser.add_argument("--thesis-market", type=Path)
    parser.add_argument("--thesis-identities", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--production-market", type=Path, help="Read-only production sports snapshot directory.")
    parser.add_argument("--market-identities", type=Path, help="Verified fixture and quote identity directory.")
    parser.add_argument("--monthly-a0-predictions", type=Path, help="Optional sealed monthly A0 sensitivity, distinct from annual A0.")
    parser.add_argument("--reliability-study", type=Path, help="Frozen multi-origin model reliability evaluation protocol; no fitting or promotion.")
    parser.add_argument("--reliability-integration", type=Path, help="Frozen conditional native12/rating integration protocol.")
    parser.add_argument("--reliability-tradeoff", type=Path, help="Frozen native6/native12 reliability-cost comparison protocol.")
    parser.add_argument("--c0-meta", type=Path, help="Frozen C0 role-aware thirty-day meta protocol; score saved corrections without fitting.")
    parser.add_argument("--champion-history", help="Completed frozen champion-history run ID; score saved models without fitting.")
    args = parser.parse_args()
    if args.champion_history:
        if any((args.c0_meta, args.reliability_tradeoff, args.reliability_study,
                args.reliability_integration, args.suite, args.data, args.production_market,
                args.thesis_audit, args.candidate_data, args.doctor, args.market_identities,
                args.monthly_a0_predictions)):
            parser.error("--champion-history cannot be combined with another evaluation mode")
        if not args.output_dir:
            parser.error("--champion-history requires a new --output-dir")
        if args.bootstraps != 5000:
            parser.error("--champion-history uses the resampling settings in its frozen protocol")
        return args
    if args.c0_meta:
        if any((args.reliability_tradeoff, args.reliability_study, args.reliability_integration,
                args.suite, args.data, args.production_market, args.thesis_audit,
                args.candidate_data, args.doctor)):
            parser.error("--c0-meta cannot be combined with another evaluation mode")
        if not args.output_dir:
            parser.error("--c0-meta requires a new --output-dir")
        if args.bootstraps != 5000:
            parser.error("--c0-meta uses the resampling settings in its frozen protocol")
    if args.reliability_tradeoff:
        if any((args.reliability_study, args.reliability_integration, args.suite, args.data,
                args.production_market, args.thesis_audit, args.candidate_data, args.doctor)):
            parser.error("--reliability-tradeoff cannot be combined with another evaluation mode")
        if not args.output_dir:
            parser.error("--reliability-tradeoff requires a new --output-dir")
        if args.bootstraps != 5000:
            parser.error("--reliability-tradeoff uses the resampling settings in its frozen protocol")
    if args.reliability_integration:
        if any((args.reliability_study, args.suite, args.data, args.production_market, args.thesis_audit,
                args.candidate_data, args.doctor)):
            parser.error("--reliability-integration cannot be combined with another evaluation mode")
        if not args.output_dir:
            parser.error("--reliability-integration requires a new --output-dir")
        if args.bootstraps != 5000:
            parser.error("--reliability-integration uses the resampling settings in its frozen protocol")
    elif args.reliability_study:
        if any((args.suite, args.data, args.production_market, args.thesis_audit, args.candidate_data, args.doctor)):
            parser.error("--reliability-study cannot be combined with another evaluation mode")
        if args.bootstraps != 5000:
            parser.error("--reliability-study uses the resampling settings in its frozen protocol")
    if args.production_market:
        if any((args.suite, args.data, args.thesis_audit, args.candidate_data, args.doctor,
                args.reliability_study, args.reliability_integration)):
            parser.error("--production-market cannot be combined with suite, legacy, thesis, candidate, doctor or reliability modes")
        if not args.market_identities:
            parser.error("--production-market requires --market-identities")
    elif args.market_identities or args.monthly_a0_predictions:
        parser.error("market identities and monthly sensitivity require --production-market")
    if args.thesis_audit:
        if args.suite or args.data or not all((args.thesis_features,args.thesis_market,args.thesis_identities,args.output_dir)):
            parser.error("--thesis-audit requires features, market, identities and a new output directory; do not combine modes")
    elif args.suite or args.production_market or args.reliability_study or args.reliability_integration or args.reliability_tradeoff or args.c0_meta:
        import os
        args.research_root = args.research_root or os.environ.get("ENSEMBLE_RESEARCH_ROOT")
        local_root = PROJECT_ROOT / "data/research_root.txt"
        if not args.research_root and local_root.is_file():
            args.research_root = local_root.read_text().strip()
        if not args.research_root:
            parser.error("research evaluation requires --research-root, ENSEMBLE_RESEARCH_ROOT or data/research_root.txt")
        if not args.doctor and not args.output_dir:
            parser.error("research evaluation requires a new --output-dir unless --doctor")
        if args.data:
            parser.error("--suite cannot be combined with legacy --data")
    elif not args.data or not args.candidate_col:
        parser.error("use --suite or both --data and --candidate-col")
    return args


def main() -> None:
    args = parse_args()
    if args.champion_history:
        import json
        from src.analysis.champion_history_benchmark import benchmark_saved

        result = benchmark_saved(args.champion_history, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.c0_meta:
        import json
        from src.analysis.c0_meta_evaluation import evaluate_meta

        result = evaluate_meta(args.c0_meta, args.output_dir, args.research_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.reliability_tradeoff:
        import json
        from src.analysis.reliability_tradeoff import evaluate_tradeoff

        result = evaluate_tradeoff(args.reliability_tradeoff, args.output_dir, args.research_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.reliability_study:
        import json
        from src.analysis.model_reliability import evaluate_study

        result = evaluate_study(args.reliability_study, args.output_dir, args.research_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.reliability_integration:
        import json
        from src.analysis.model_reliability_integration import evaluate_integration

        result = evaluate_integration(args.reliability_integration, args.output_dir, args.research_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.production_market:
        import json
        from src.analysis.production_market_benchmark import run_production_market_benchmark

        manifest = json.loads((PROJECT_ROOT / "conf/base/research_benchmark.json").read_text())
        forecasts = Path(args.research_root) / manifest["input"]["path"]
        result = run_production_market_benchmark(
            args.production_market, args.market_identities, forecasts, args.output_dir,
            monthly_path=args.monthly_a0_predictions,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.thesis_audit:
        import json
        from src.analysis.thesis_audit import run_thesis_audit
        result=run_thesis_audit(args.thesis_features,args.thesis_market,args.thesis_identities,args.output_dir,args.workers)
        print(json.dumps(result,indent=2))
        return
    if args.suite:
        import json
        from src.analysis.research_benchmark import run_suite
        result = run_suite(args.suite, args.research_root, args.output_dir, args.doctor,
                           args.candidate_data, args.candidate_col or "p", args.bootstraps)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.doctor and result["status"] != "READY":
            raise SystemExit(2)
        return
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Error: Data file not found at {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if args.min_date and args.date_col in df.columns:
        df = df[df[args.date_col].astype(str) >= args.min_date]
    if args.max_date and args.date_col in df.columns:
        df = df[df[args.date_col].astype(str) <= args.max_date]

    print(f"Loaded {len(df):,} evaluation rows from {data_path} (filtered to >= {args.min_date})")

    report = run_model_benchmark(
        data=df,
        candidate_prob_col=args.candidate_col,
        baseline_prob_col=args.baseline_col,
        target_col=args.target_col,
        date_col=args.date_col,
        model_name=args.model_name,
        model_version=args.model_version,
        baseline_name=args.baseline_name,
        bon_col=args.bon_col,
        tier_col=args.tier_col,
        odds_col=args.odds_col,
        disagreement_col=args.disagreement_col,
        n_bootstraps=args.bootstraps,
    )

    markdown = report.format_markdown()
    print("\n" + markdown)

    if args.output_json:
        import json
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nSaved benchmark JSON to {out_path}")


if __name__ == "__main__":
    main()
