#!/usr/bin/env python3
"""CLI runner for the legacy standalone financial and betting benchmark.

NOTE: For new production and research evaluations, prefer the unified Single
Source of Truth runner:
    `scripts/run_unified_betting_evaluation.py`
which enforces consistent qualification, calibrated EV, and non-conflicting metrics.

Evaluates executable betting performance under the 12% Polish turnover tax.
Example:
    .venv/bin/python scripts/run_financial_benchmark.py \
      --data reports/exp039_db_market_backtest_v2/exp039_market_common.csv \
      --candidate-col exp039_prob_team_a \
      --target-col y_team_a \
      --odds-a-col best_open_t1 \
      --odds-b-col best_open_t2 \
      --date-col date \
      --min-date 2024-01-01 \
      --min-ev 0.03 \
      --tax-rate 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.financial_benchmark import run_financial_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone financial and betting benchmark (with 12% turnover tax)."
    )
    parser.add_argument("--data", required=True, help="Path to CSV or Parquet evaluation dataset.")
    parser.add_argument("--candidate-col", required=True, help="Candidate model prediction column (prob Team A).")
    parser.add_argument("--target-col", default="y_true", help="Ground truth binary outcome (1 = Team A won, default: y_true).")
    parser.add_argument("--odds-a-col", default=None, help="Decimal odds column for Team A.")
    parser.add_argument("--odds-b-col", default=None, help="Decimal odds column for Team B.")
    parser.add_argument("--odds-col", default=None, help="Single decimal odds column.")
    parser.add_argument("--close-odds-a-col", default=None, help="Closing decimal odds column for Team A (for CLV analysis).")
    parser.add_argument("--close-odds-b-col", default=None, help="Closing decimal odds column for Team B (for CLV analysis).")
    parser.add_argument("--close-odds-col", default=None, help="Single closing decimal odds column (for CLV analysis).")
    parser.add_argument("--time-horizon", default="opening (>12h)", help="Betting entry time horizon (default: 'opening (>12h)').")
    parser.add_argument("--date-col", default="date", help="Match date column (default: date).")
    parser.add_argument("--min-date", default="2024-01-01", help="Start date for evaluation cohort (default: 2024-01-01).")
    parser.add_argument("--max-date", default=None, help="End date for evaluation cohort.")
    parser.add_argument("--tax-rate", type=float, default=0.12, help="Polish turnover tax fraction (default: 0.12).")
    parser.add_argument("--min-ev", type=float, default=0.05, help="Minimum net EV threshold to place a bet (default: 0.05 = +5%%).")
    parser.add_argument("--max-odds", type=float, default=5.0, help="Maximum odds threshold for safety/quarantine (default: 5.0).")
    parser.add_argument("--min-odds", type=float, default=1.05, help="Minimum odds threshold (default: 1.05).")
    parser.add_argument("--bankroll", type=float, default=10000.0, help="Initial starting bankroll in PLN (default: 10000.0).")
    parser.add_argument(
        "--staking",
        default="flat_100",
        choices=["flat_100", "quarter_kelly", "fixed_pct_1"],
        help="Staking policy: flat_100 (100 PLN flat), quarter_kelly (25%% Kelly, cap 2%%), or fixed_pct_1 (1%% bankroll).",
    )
    parser.add_argument("--model-name", default="Candidate Model", help="Human-readable model name.")
    parser.add_argument("--model-version", default="v1.0", help="Model version string.")
    parser.add_argument("--output-json", default=None, help="Optional path to save benchmark results as JSON.")
    parser.add_argument("--output-md", default=None, help="Optional path to save benchmark Markdown report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Error: Data file not found at {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Filter date range
    if args.date_col in df.columns:
        if args.min_date:
            df = df[df[args.date_col].astype(str) >= str(args.min_date)]
        if args.max_date:
            df = df[df[args.date_col].astype(str) <= str(args.max_date)]

    report = run_financial_benchmark(
        data=df,
        candidate_prob_col=args.candidate_col,
        target_col=args.target_col,
        odds_a_col=args.odds_a_col,
        odds_b_col=args.odds_b_col,
        odds_col=args.odds_col,
        close_odds_a_col=args.close_odds_a_col,
        close_odds_b_col=args.close_odds_b_col,
        close_odds_col=args.close_odds_col,
        time_horizon=args.time_horizon,
        date_col=args.date_col,
        tax_rate=args.tax_rate,
        min_ev=args.min_ev,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        initial_bankroll=args.bankroll,
        staking_strategy=args.staking,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    md_output = report.format_markdown()
    print(md_output)

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nSaved financial benchmark JSON to {out_json_path}")

    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_output)
        print(f"\nSaved financial benchmark Markdown to {out_md_path}")


if __name__ == "__main__":
    main()
