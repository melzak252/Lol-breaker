#!/usr/bin/env python3
"""CLI runner for the legacy Odds Bracket Calibration Benchmark.

NOTE: For unified odds bracket calibration and zero-gap EV benchmarks, prefer:
    `scripts/run_unified_betting_evaluation.py`
which guarantees identical numbers across backend and CLI.

Evaluates calibration and betting metrics partitioned by bookmaker odds brackets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.odds_bracket_benchmark import evaluate_odds_bracket_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Odds Bracket Calibration & Betting Benchmark"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="reports/exp039_db_market_backtest_v2/exp039_market_common.csv",
        help="Path to market CSV/parquet dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Application Matches (2026)",
        help="Name of the evaluated dataset",
    )
    parser.add_argument(
        "--odds-col",
        type=str,
        default=None,
        help="Column for decimal odds. If None, auto-detected or reconstructed from raw prob",
    )
    parser.add_argument(
        "--model-col",
        type=str,
        default="exp039_prob_team_a",
        help="Column for model predicted probability",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="y_team_a",
        help="Column for ground truth outcome (1=win, 0=loss)",
    )
    parser.add_argument(
        "--market-col",
        type=str,
        default="market_close_p_a_novig",
        help="Column for market consensus no-vig probability",
    )
    parser.add_argument(
        "--tax-rate",
        type=float,
        default=0.12,
        help="Turnover tax rate (default: 0.12 for Polish bookmakers)",
    )
    parser.add_argument(
        "--min-net-ev",
        type=float,
        default=None,
        help="Filter bets with Net EV >= this threshold after tax (e.g. 0.05 for 5%% EV)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional path to write the markdown report",
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        default=True,
        help="Expand matches into both Team A and Team B betting lines (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_parquet(data_path)

    # If two-sided expansion is requested and we have the standard exp039_market_common structure
    if args.two_sided and "market_close_p_a_raw" in df.columns and "exp039_prob_team_a" in df.columns:
        rows = []
        for _, r in df.iterrows():
            # Side A
            p_raw_a = float(r["market_close_p_a_raw"])
            odds_a = 1.0 / p_raw_a if p_raw_a > 0 else np.nan
            p_novig_a = float(r.get(args.market_col, 1.0 / odds_a))
            p_model_a = float(r[args.model_col])
            y_a = int(r[args.target_col])
            rows.append({
                "odds": odds_a,
                "p_novig": p_novig_a,
                "p_model": p_model_a,
                "y_true": y_a,
            })

            # Side B
            margin = float(r.get("market_close_avg_margin", 0.06))
            p_raw_b = (1.0 + margin) - p_raw_a
            odds_b = 1.0 / p_raw_b if p_raw_b > 0 else np.nan
            p_novig_b = 1.0 - p_novig_a
            p_model_b = 1.0 - p_model_a
            y_b = 1 - y_a
            rows.append({
                "odds": odds_b,
                "p_novig": p_novig_b,
                "p_model": p_model_b,
                "y_true": y_b,
            })
        bench_df = pd.DataFrame(rows)
        odds_col = "odds"
        model_col = "p_model"
        target_col = "y_true"
        market_col = "p_novig"
    else:
        bench_df = df
        odds_col = args.odds_col or "odds"
        model_col = args.model_col
        target_col = args.target_col
        market_col = args.market_col

    report = evaluate_odds_bracket_benchmark(
        bench_df,
        odds_col=odds_col,
        model_prob_col=model_col,
        target_col=target_col,
        market_prob_col=market_col,
        tax_rate=args.tax_rate,
        min_net_ev=args.min_net_ev,
        dataset_name=args.dataset_name,
    )
    md_output = report.to_markdown()
    print(md_output)

    if args.output_md:
        out_path = Path(args.output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_output, encoding="utf-8")
        print(f"\n[Saved markdown report to {out_path}]")


if __name__ == "__main__":
    main()
