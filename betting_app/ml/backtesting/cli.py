"""CLI for historical model-vs-bookmaker backtests.

Example:
    python -m betting_app.ml.backtesting.cli \
      --model-name Sym-Cal LR-ElasticNet-W20-Binomial \
      --model-version exp-039 \
      --days-back 365 \
      --min-ev 0.02 \
      --staking fractional_kelly
"""

from __future__ import annotations

import argparse
import json

from betting_app.core.db import get_session
from betting_app.ml.backtesting.comparison import compare_predictions_to_market
from betting_app.ml.backtesting.engine import run_backtest
from betting_app.ml.backtesting.loaders import load_finished_match_labels, load_odds_quotes, load_predictions
from betting_app.ml.config import BacktestConfig, StakingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest historical model predictions against collected bookmaker odds")
    parser.add_argument("--model-name", default=None, help="Filter canonical_predictions.model_name")
    parser.add_argument("--model-version", default=None, help="Filter canonical_predictions.model_version")
    parser.add_argument("--include-stale", action="store_true", help="Include stale predictions, not only active ones")
    parser.add_argument("--all-predictions-per-match", action="store_true", help="Do not deduplicate to latest prediction per match/model/version")
    parser.add_argument("--days-back", type=int, default=None, help="Limit finished matches by start_time_normalized")
    parser.add_argument("--bankroll", type=float, default=1_000.0)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--tax-rate", type=float, default=0.12)
    parser.add_argument("--staking", choices=["fixed", "percent", "fractional_kelly"], default="fractional_kelly")
    parser.add_argument("--fixed-stake", type=float, default=10.0)
    parser.add_argument("--bankroll-fraction", type=float, default=0.01)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-bets-per-match", type=int, default=1)
    parser.add_argument("--odds-policy", choices=["latest_pre_match", "all_pre_match"], default="latest_pre_match")
    parser.add_argument("--min-minutes-before-start", type=int, default=0)
    parser.add_argument("--require-prediction-before-odds", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    parser.add_argument("--show-bets", type=int, default=0, help="Print first N simulated bets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BacktestConfig(
        bankroll_start=args.bankroll,
        min_ev=args.min_ev,
        tax_rate=args.tax_rate,
        staking=StakingConfig(
            strategy=args.staking,
            fixed_stake=args.fixed_stake,
            bankroll_fraction=args.bankroll_fraction,
            kelly_fraction=args.kelly_fraction,
        ),
        max_bets_per_match=args.max_bets_per_match,
        odds_policy=args.odds_policy,
        min_minutes_before_start=args.min_minutes_before_start,
        require_prediction_before_odds=args.require_prediction_before_odds,
    )

    with get_session() as session:
        labels = load_finished_match_labels(days_back=args.days_back, session=session)
        label_ids = {label.canonical_match_id for label in labels}
        predictions = [
            prediction
            for prediction in load_predictions(
                model_name=args.model_name,
                model_version=args.model_version,
                only_active=not args.include_stale,
                latest_per_match=not args.all_predictions_per_match,
                session=session,
            )
            if prediction.canonical_match_id in label_ids
        ]
        odds = load_odds_quotes(canonical_match_ids=label_ids, session=session)

    result = run_backtest(predictions, labels, odds, config)
    comparison = compare_predictions_to_market(predictions, labels, odds, config)
    summary = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "predictions_loaded": len(predictions),
        "labels_loaded": len(labels),
        "odds_quotes_loaded": len(odds),
        "matches_seen": result.matches_seen,
        "matches_bet": result.matches_bet,
        "bets": len(result.bets),
        "bankroll_start": round(result.bankroll_start, 2),
        "bankroll_end": round(result.bankroll_end, 2),
        "total_staked": round(result.total_staked, 2),
        "total_profit": round(result.total_profit, 2),
        "roi": round(result.roi, 6),
        "hit_rate": round(result.hit_rate, 6),
        "max_drawdown": round(result.max_drawdown, 2),
        "comparison_observations": comparison.observations,
        "model_log_loss": round(comparison.model_log_loss, 6),
        "market_log_loss": round(comparison.market_log_loss, 6),
        "model_brier": round(comparison.model_brier, 6),
        "market_brier": round(comparison.market_brier, 6),
        "model_accuracy": round(comparison.model_accuracy, 6),
        "market_accuracy": round(comparison.market_accuracy, 6),
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("Historical model-vs-bookmaker backtest")
        for key, value in summary.items():
            print(f"{key}: {value}")

    for bet in result.bets[: max(0, args.show_bets)]:
        print(
            f"bet match={bet.canonical_match_id} side={bet.side} book={bet.bookmaker_name} "
            f"odds={bet.odds:.3f} prob={bet.model_prob:.3f} ev={bet.ev:.3f} "
            f"stake={bet.stake:.2f} result={bet.result} profit={bet.profit:.2f}"
        )


if __name__ == "__main__":
    main()
