"""Read-only event-time ROI benchmark for one immutable model version.

Timestamped predictions are strict by default. Retrospective prediction files
without decision-time timestamps require an explicit non-executable proxy flag.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from betting_app.core.db import get_session
from betting_app.ml.backtesting.comparison import compare_predictions_to_market
from betting_app.ml.backtesting.engine import run_backtest
from betting_app.ml.backtesting.loaders import (
    load_finished_match_labels,
    load_odds_quotes,
    load_predictions,
)
from betting_app.ml.backtesting.odds_selection import select_quotes_for_match
from betting_app.ml.backtesting.types import (
    BacktestBet,
    HistoricalPrediction,
    MatchLabel,
    OddsQuote,
)
from betting_app.ml.config import BacktestConfig, StakingConfig

load_dotenv()

DEFAULT_OUTPUT_ROOT = Path("reports/model_roi_benchmarks")
DEFAULT_PROBABILITY_COLUMN = "prob_a"
ODDS_POLICIES = {
    "open": "open_pre_match",
    "mid": "mid_pre_match",
    "close": "latest_pre_match",
}
TAX_SCENARIOS = {
    "poland_tax_12": 0.12,
    "no_tax": 0.0,
}


@dataclass(frozen=True)
class BenchmarkInputs:
    predictions: list[HistoricalPrediction]
    labels: list[MatchLabel]
    odds: list[OddsQuote]
    cohort_by_id: dict[int, dict[str, Any]]
    source: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one immutable model version against historical bookmaker odds"
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional prediction CSV. Omit to read canonical_predictions. "
            "CSV mode remains read-only and still loads labels/odds from the database."
        ),
    )
    parser.add_argument("--probability-column", default=DEFAULT_PROBABILITY_COLUMN)
    parser.add_argument("--probability-b-column", default=None)
    parser.add_argument("--predicted-at-column", default="predicted_at")
    parser.add_argument("--data-cutoff-column", default="data_cutoff_at")
    parser.add_argument("--outcome-column", default=None)
    parser.add_argument(
        "--eligibility-column",
        action="append",
        default=[],
        help="Boolean CSV eligibility column; repeat to require several filters.",
    )
    parser.add_argument(
        "--allow-retrospective-proxy",
        action="store_true",
        help=(
            "Allow CSV predictions without predicted_at/data_cutoff_at. "
            "The report is then non-executable and cannot support promotion."
        ),
    )
    parser.add_argument("--days-back", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initial-bankroll", type=float, default=1_000.0)
    parser.add_argument("--fixed-stake", type=float, default=10.0)
    parser.add_argument(
        "--staking",
        choices=("fixed", "fractional_kelly"),
        default="fixed",
    )
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-stake-fraction", type=float, default=0.01)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument(
        "--timings",
        nargs="+",
        choices=sorted(ODDS_POLICIES),
        default=list(ODDS_POLICIES),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_902)
    return parser


def _true_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _parse_optional_datetime(value: Any) -> datetime | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _ensure_unique_predictions(
    predictions: list[HistoricalPrediction],
) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for prediction in predictions:
        if prediction.canonical_match_id in seen:
            duplicates.add(prediction.canonical_match_id)
        seen.add(prediction.canonical_match_id)
    if duplicates:
        raise ValueError(
            "Benchmark requires one immutable prediction per canonical match; "
            f"duplicates: {sorted(duplicates)[:10]}"
        )


def load_csv_predictions(
    path: Path,
    *,
    model_name: str,
    model_version: str,
    probability_column: str,
    probability_b_column: str | None,
    predicted_at_column: str,
    data_cutoff_column: str,
    outcome_column: str | None,
    eligibility_columns: list[str],
    allow_retrospective_proxy: bool,
) -> tuple[
    list[HistoricalPrediction],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    cohort = pd.read_csv(path)
    rows_read = len(cohort)
    required = {"canonical_match_id", probability_column, *eligibility_columns}
    if probability_b_column:
        required.add(probability_b_column)
    if outcome_column:
        required.add(outcome_column)
    if not allow_retrospective_proxy:
        required.update({predicted_at_column, data_cutoff_column})
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Missing required prediction CSV columns: {missing}")

    exclusions: dict[str, int] = {}
    for column in eligibility_columns:
        before = len(cohort)
        cohort = cohort[_true_mask(cohort[column])].copy()
        exclusions[column] = before - len(cohort)
    if cohort.empty:
        raise ValueError("No prediction rows remain after eligibility filters")

    match_ids = pd.to_numeric(
        cohort["canonical_match_id"], errors="raise"
    )
    if ((match_ids % 1) != 0).any():
        raise ValueError("canonical_match_id values must be integers")
    cohort["canonical_match_id"] = match_ids.astype(int)
    probabilities_a = pd.to_numeric(
        cohort[probability_column], errors="raise"
    ).astype(float)
    probabilities_b = (
        pd.to_numeric(cohort[probability_b_column], errors="raise").astype(float)
        if probability_b_column
        else 1.0 - probabilities_a
    )
    invalid_probability = (
        ~probabilities_a.between(0.0, 1.0)
        | ~probabilities_b.between(0.0, 1.0)
        | ((probabilities_a + probabilities_b - 1.0).abs() > 1e-6)
    )
    if invalid_probability.any():
        raise ValueError("Prediction CSV contains invalid probability pairs")

    cohort[probability_column] = probabilities_a
    if probability_b_column:
        cohort[probability_b_column] = probabilities_b
    if cohort["canonical_match_id"].duplicated().any():
        duplicates = cohort.loc[
            cohort["canonical_match_id"].duplicated(keep=False),
            "canonical_match_id",
        ].tolist()
        raise ValueError(
            "Benchmark requires one immutable prediction per canonical match; "
            f"duplicates: {duplicates[:10]}"
        )

    predictions: list[HistoricalPrediction] = []
    for row in cohort.to_dict("records"):
        predicted_at = _parse_optional_datetime(row.get(predicted_at_column))
        data_cutoff_at = _parse_optional_datetime(row.get(data_cutoff_column))
        if not allow_retrospective_proxy and (
            predicted_at is None or data_cutoff_at is None
        ):
            raise ValueError(
                "Strict benchmark rows require valid predicted_at and "
                "data_cutoff_at timestamps"
            )
        probability_a = float(row[probability_column])
        probability_b = (
            float(row[probability_b_column])
            if probability_b_column
            else 1.0 - probability_a
        )
        predictions.append(
            HistoricalPrediction(
                canonical_match_id=int(row["canonical_match_id"]),
                model_name=model_name,
                model_version=model_version,
                prob_a=probability_a,
                prob_b=probability_b,
                predicted_at=predicted_at,
                data_cutoff_at=data_cutoff_at,
                diagnostics={
                    "scope": (
                        "retrospective_proxy"
                        if allow_retrospective_proxy
                        else "timestamped_prediction_csv"
                    )
                },
            )
        )

    records = {
        int(row["canonical_match_id"]): row
        for row in cohort.to_dict("records")
    }
    source = {
        "kind": "csv",
        "path": str(path),
        "rows_read": rows_read,
        "rows_after_eligibility": len(cohort),
        "eligibility_exclusions": exclusions,
        "probability_column": probability_column,
        "probability_b_column": probability_b_column,
        "predicted_at_column": predicted_at_column,
        "data_cutoff_column": data_cutoff_column,
        "outcome_column": outcome_column,
    }
    return predictions, records, source


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _validate_temporal_prediction_contract(
    predictions: list[HistoricalPrediction],
    labels: list[MatchLabel],
    *,
    allow_retrospective_proxy: bool,
) -> None:
    if allow_retrospective_proxy:
        return
    labels_by_id = {label.canonical_match_id: label for label in labels}
    missing_timestamps: list[int] = []
    invalid_order: list[int] = []
    for prediction in predictions:
        predicted_at = _as_utc(prediction.predicted_at)
        data_cutoff_at = _as_utc(prediction.data_cutoff_at)
        if predicted_at is None or data_cutoff_at is None:
            missing_timestamps.append(prediction.canonical_match_id)
            continue
        label = labels_by_id[prediction.canonical_match_id]
        start_time = _as_utc(label.start_time)
        if data_cutoff_at > predicted_at or (
            start_time is not None and predicted_at >= start_time
        ):
            invalid_order.append(prediction.canonical_match_id)
    if missing_timestamps or invalid_order:
        raise ValueError(
            "Prediction temporal contract failed: "
            f"missing timestamps={missing_timestamps[:10]}, "
            f"invalid order={invalid_order[:10]}"
        )

def load_benchmark_inputs(args: argparse.Namespace) -> BenchmarkInputs:
    if args.allow_retrospective_proxy and args.input is None:
        raise ValueError("--allow-retrospective-proxy is valid only with --input")

    with get_session() as session:
        all_labels = load_finished_match_labels(
            days_back=args.days_back,
            session=session,
        )
        label_ids = {label.canonical_match_id for label in all_labels}
        if args.input is not None:
            predictions, csv_records, source = load_csv_predictions(
                args.input,
                model_name=args.model_name,
                model_version=args.model_version,
                probability_column=args.probability_column,
                probability_b_column=args.probability_b_column,
                predicted_at_column=args.predicted_at_column,
                data_cutoff_column=args.data_cutoff_column,
                outcome_column=args.outcome_column,
                eligibility_columns=args.eligibility_column,
                allow_retrospective_proxy=args.allow_retrospective_proxy,
            )
            prediction_ids = {
                prediction.canonical_match_id for prediction in predictions
            }
            missing_labels = sorted(prediction_ids - label_ids)
            if missing_labels:
                raise ValueError(
                    "Prediction CSV rows lack finished canonical labels: "
                    f"{missing_labels[:10]}"
                )
            labels = [
                label
                for label in all_labels
                if label.canonical_match_id in prediction_ids
            ]
        else:
            loaded_predictions = load_predictions(
                model_name=args.model_name,
                model_version=args.model_version,
                only_active=False,
                latest_per_match=False,
                session=session,
            )
            predictions = [
                prediction
                for prediction in loaded_predictions
                if prediction.canonical_match_id in label_ids
            ]
            csv_records = {}
            source = {
                "kind": "canonical_predictions",
                "rows_read": len(loaded_predictions),
                "rows_for_finished_cohort": len(predictions),
            }
            prediction_ids = {
                prediction.canonical_match_id for prediction in predictions
            }
            labels = [
                label
                for label in all_labels
                if label.canonical_match_id in prediction_ids
            ]

        if not predictions:
            raise ValueError(
                "No finished predictions found for the requested model version"
            )
        _ensure_unique_predictions(predictions)
        _validate_temporal_prediction_contract(
            predictions,
            labels,
            allow_retrospective_proxy=args.allow_retrospective_proxy,
        )

        if args.input is not None and args.outcome_column:
            expected_winners: dict[int, str] = {}
            for match_id, row in csv_records.items():
                raw_outcome = pd.to_numeric(
                    row[args.outcome_column],
                    errors="raise",
                )
                if raw_outcome not in {0, 1}:
                    raise ValueError(
                        f"{args.outcome_column} must contain only 0/1 labels"
                    )
                expected_winners[match_id] = (
                    "a" if int(raw_outcome) == 1 else "b"
                )
            conflicts = [
                label.canonical_match_id
                for label in labels
                if expected_winners[label.canonical_match_id]
                != label.winner_side
            ]
            if conflicts:
                raise ValueError(
                    "Current canonical outcomes disagree with prediction CSV: "
                    f"{conflicts[:10]}"
                )

        odds = load_odds_quotes(
            canonical_match_ids=prediction_ids,
            session=session,
        )

    cohort_by_id: dict[int, dict[str, Any]] = {}
    for label in labels:
        row = dict(csv_records.get(label.canonical_match_id, {}))
        row["canonical_match_id"] = label.canonical_match_id
        row["start_time_normalized"] = (
            label.start_time.isoformat() if label.start_time else None
        )
        row["date"] = (
            label.start_time.date().isoformat() if label.start_time else None
        )
        row["league"] = row.get("league") or label.league
        cohort_by_id[label.canonical_match_id] = row

    source["retrospective_proxy"] = args.allow_retrospective_proxy
    source["predictions_loaded"] = len(predictions)
    source["missing_predicted_at"] = sum(
        prediction.predicted_at is None for prediction in predictions
    )
    source["missing_data_cutoff_at"] = sum(
        prediction.data_cutoff_at is None for prediction in predictions
    )
    return BenchmarkInputs(
        predictions=predictions,
        labels=labels,
        odds=odds,
        cohort_by_id=cohort_by_id,
        source=source,
    )


def build_closing_quote_index(
    labels: list[MatchLabel],
    odds: list[OddsQuote],
) -> dict[tuple[int, int], OddsQuote]:
    odds_by_match: dict[int, list[OddsQuote]] = defaultdict(list)
    for quote in odds:
        odds_by_match[quote.canonical_match_id].append(quote)
    config = BacktestConfig(odds_policy="latest_pre_match")
    output: dict[tuple[int, int], OddsQuote] = {}
    for label in labels:
        for quote in select_quotes_for_match(
            odds_by_match.get(label.canonical_match_id, []), label, config
        ):
            output[(label.canonical_match_id, quote.bookmaker_id)] = quote
    return output


def _longest_losing_streak(bets: list[BacktestBet]) -> int:
    longest = 0
    current = 0
    for bet in sorted(
        bets,
        key=lambda item: (
            getattr(item, "settled_at", None) or item.placed_at,
            item.canonical_match_id,
        ),
    ):
        if bet.result == "lost":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _bootstrap_yield(
    ledger: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if ledger.empty or resamples <= 0:
        return {
            "calendar_week_blocks": 0,
            "yield_ci95": [None, None],
            "probability_positive_yield": None,
        }
    weeks = (
        pd.to_datetime(ledger["placed_at"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("W-SUN")
        .astype(str)
    )
    grouped = ledger.assign(_week=weeks).groupby("_week", sort=True)
    profit = grouped["profit"].sum().to_numpy(dtype=float)
    staked = grouped["stake"].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(profit), size=(resamples, len(profit)))
    sampled_profit = profit[draws].sum(axis=1)
    sampled_staked = staked[draws].sum(axis=1)
    sampled_yield = np.divide(
        sampled_profit,
        sampled_staked,
        out=np.zeros_like(sampled_profit),
        where=sampled_staked > 0,
    )
    return {
        "calendar_week_blocks": int(len(profit)),
        "yield_ci95": [
            float(value) for value in np.quantile(sampled_yield, [0.025, 0.975])
        ],
        "probability_positive_yield": float((sampled_yield > 0).mean()),
    }


def result_ledger(
    bets: list[BacktestBet],
    *,
    cohort_by_id: dict[int, dict[str, Any]],
    closing_quotes: dict[tuple[int, int], OddsQuote],
    scenario: str,
    timing: str,
    tax_rate: float,
    staking_strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bet in bets:
        match = cohort_by_id[bet.canonical_match_id]
        close_quote = closing_quotes.get((bet.canonical_match_id, bet.bookmaker_id))
        close_odds = None
        if close_quote is not None:
            close_odds = close_quote.odds_a if bet.side == "a" else close_quote.odds_b
        start_at = pd.to_datetime(
            match.get("start_time_normalized"), utc=True, errors="coerce"
        )
        bet_settled_at = getattr(bet, "settled_at", None)
        settled_at = bet_settled_at if bet_settled_at is not None else (
            start_at.to_pydatetime() if pd.notna(start_at) else bet.placed_at
        )
        hours_before_start = (
            (start_at.to_pydatetime() - bet.placed_at).total_seconds() / 3600
            if pd.notna(start_at) and bet.placed_at is not None
            else None
        )
        settlement_delay_hours = (
            (settled_at - start_at.to_pydatetime()).total_seconds() / 3600
            if pd.notna(start_at) and settled_at is not None
            else None
        )
        expected_profit = getattr(bet, "expected_profit", None)
        if expected_profit is None:
            expected_profit = bet.stake * bet.ev
        available_before = getattr(bet, "available_bankroll_before", None)
        if available_before is None:
            available_before = bet.bankroll_before
        reserved_after = getattr(bet, "reserved_stake_after_placement", None)
        if reserved_after is None:
            reserved_after = bet.stake
        rows.append(
            {
                "scenario": scenario,
                "timing": timing,
                "tax_rate": tax_rate,
                "staking_strategy": staking_strategy,
                "canonical_match_id": bet.canonical_match_id,
                "golgg_match_id": match.get("golgg_match_id"),
                "date": match.get("date"),
                "league": match.get("league"),
                "best_of": match.get("best_of"),
                "team_a_name": match.get("team_a_name"),
                "team_b_name": match.get("team_b_name"),
                "side": bet.side,
                "bookmaker_id": bet.bookmaker_id,
                "bookmaker_name": bet.bookmaker_name,
                "odds_snapshot_id": bet.odds_snapshot_id,
                "placed_at": bet.placed_at.isoformat(),
                "start_at": start_at.isoformat() if pd.notna(start_at) else None,
                "settled_at": settled_at.isoformat() if settled_at is not None else None,
                "hours_before_start": hours_before_start,
                "settlement_delay_hours": settlement_delay_hours,
                "entry_odds": bet.odds,
                "close_odds_same_book": close_odds,
                "clv_odds_pct": (
                    bet.odds / close_odds - 1.0
                    if close_odds is not None and close_odds > 0
                    else None
                ),
                "model_probability": bet.model_prob,
                "market_probability_novig": bet.market_prob,
                "probability_edge": bet.model_prob - bet.market_prob,
                "breakeven_probability": 1.0 / (bet.odds * (1.0 - tax_rate)),
                "expected_value": bet.ev,
                "stake": bet.stake,
                "tax_paid": bet.stake * tax_rate,
                "expected_profit": expected_profit,
                "profit": bet.profit,
                "return_on_stake": bet.profit / bet.stake,
                "result": bet.result,
                "won": bet.result == "won",
                "bankroll_before_placement": bet.bankroll_before,
                "available_before_placement": available_before,
                "reserved_after_placement": reserved_after,
                "bankroll_after_settlement": bet.bankroll_after,
            }
        )
    return pd.DataFrame(rows)


def summarize_result(
    result,
    ledger: pd.DataFrame,
    *,
    scenario: str,
    timing: str,
    tax_rate: float,
    min_ev: float,
    initial_bankroll: float,
    staking: StakingConfig,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    positive_profit = ledger.loc[ledger["profit"] > 0, "profit"].sum() if not ledger.empty else 0.0
    negative_profit = -ledger.loc[ledger["profit"] < 0, "profit"].sum() if not ledger.empty else 0.0
    clv = ledger["clv_odds_pct"].dropna() if not ledger.empty else pd.Series(dtype=float)
    bootstrap = _bootstrap_yield(
        ledger,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "scenario": scenario,
        "timing": timing,
        "odds_policy": ODDS_POLICIES[timing],
        "tax_rate": tax_rate,
        "ev_threshold": min_ev,
        "initial_bankroll": initial_bankroll,
        "staking_strategy": staking.strategy,
        "fixed_stake": staking.fixed_stake if staking.strategy == "fixed" else None,
        "kelly_fraction": (
            staking.kelly_fraction
            if staking.strategy == "fractional_kelly"
            else None
        ),
        "max_stake_fraction": getattr(
            staking, "max_bankroll_fraction", getattr(staking, "bankroll_fraction", None)
        ),
        "matches_seen": result.matches_seen,
        "matches_temporally_ineligible": result.matches_temporally_ineligible,
        "qualifying_bets": len(result.bets),
        "matches_bet": result.matches_bet,
        "bet_coverage": (
            result.matches_bet
            / (result.matches_seen - result.matches_temporally_ineligible)
            if result.matches_seen > result.matches_temporally_ineligible
            else 0.0
        ),
        "money_bet": result.total_staked,
        "turnover_multiple": result.turnover,
        "expected_profit": result.expected_profit,
        "expected_roi": result.expected_yield,
        "realized_profit": result.total_profit,
        "true_roi_on_initial_bankroll": result.bankroll_return,
        "yield_on_money_bet": result.roi,
        "ending_money": result.bankroll_end,
        "maximum_drawdown_money": result.max_drawdown,
        "maximum_drawdown_fraction": result.max_drawdown_fraction,
        "hit_rate": result.hit_rate,
        "expected_hit_rate": (
            float(np.average(ledger["model_probability"], weights=ledger["stake"]))
            if not ledger.empty
            else 0.0
        ),
        "average_breakeven_hit_rate": (
            float(np.average(ledger["breakeven_probability"], weights=ledger["stake"]))
            if not ledger.empty
            else 0.0
        ),
        "average_odds": float(ledger["entry_odds"].mean()) if not ledger.empty else None,
        "average_ev": float(ledger["expected_value"].mean()) if not ledger.empty else None,
        "median_ev": float(ledger["expected_value"].median()) if not ledger.empty else None,
        "average_probability_edge": (
            float(ledger["probability_edge"].mean()) if not ledger.empty else None
        ),
        "tax_paid": float(ledger["tax_paid"].sum()) if not ledger.empty else 0.0,
        "profit_factor": float(positive_profit / negative_profit) if negative_profit > 0 else None,
        "longest_losing_streak": _longest_losing_streak(result.bets),
        "max_open_stake": result.max_open_stake,
        "max_open_stake_fraction": result.max_open_stake / initial_bankroll,
        "max_open_bets": result.max_open_bets,
        "bets_skipped_insufficient_funds": result.bets_skipped_insufficient_funds,
        "average_clv_odds_pct": float(clv.mean()) if len(clv) else None,
        "median_clv_odds_pct": float(clv.median()) if len(clv) else None,
        "positive_clv_rate": float((clv > 0).mean()) if len(clv) else None,
        **bootstrap,
    }


def _build_staking_config(args: argparse.Namespace) -> StakingConfig:
    if args.staking == "fixed":
        return StakingConfig(
            strategy="fixed",
            fixed_stake=args.fixed_stake,
            max_stake=args.fixed_stake,
            max_bankroll_fraction=args.max_stake_fraction,
        )
    return StakingConfig(
        strategy="fractional_kelly",
        kelly_fraction=args.kelly_fraction,
        min_stake=0.0,
        max_stake=None,
        max_bankroll_fraction=args.max_stake_fraction,
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()
    return slug or "model"


def _resolve_output_dir(
    args: argparse.Namespace,
    generated_at: datetime,
) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_id = generated_at.strftime("%Y%m%dT%H%M%SZ")
    return (
        DEFAULT_OUTPUT_ROOT
        / _slug(args.model_name)
        / _slug(args.model_version)
        / run_id
    )


def _comparison_payload(
    predictions: list[HistoricalPrediction],
    labels: list[MatchLabel],
    odds: list[OddsQuote],
    *,
    timing: str,
    require_prediction_before_odds: bool,
) -> dict[str, Any]:
    comparison = compare_predictions_to_market(
        predictions,
        labels,
        odds,
        BacktestConfig(
            odds_policy=ODDS_POLICIES[timing],
            require_prediction_before_odds=require_prediction_before_odds,
        ),
    )
    return {
        key: (
            None
            if isinstance(value, float) and not np.isfinite(value)
            else value
        )
        for key, value in asdict(comparison).items()
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.initial_bankroll <= 0:
        raise ValueError("--initial-bankroll must be positive")
    if args.fixed_stake <= 0:
        raise ValueError("--fixed-stake must be positive")
    if args.min_ev < 0:
        raise ValueError("--min-ev must be non-negative")
    if not 0 < args.kelly_fraction <= 1:
        raise ValueError("--kelly-fraction must be in (0, 1]")
    if not 0 < args.max_stake_fraction <= 1:
        raise ValueError("--max-stake-fraction must be in (0, 1]")
    if args.days_back is not None and args.days_back <= 0:
        raise ValueError("--days-back must be positive")

    generated_at = datetime.now(UTC)
    output_dir = _resolve_output_dir(args, generated_at)
    staking = _build_staking_config(args)
    inputs = load_benchmark_inputs(args)
    predictions = inputs.predictions
    labels = inputs.labels
    odds = inputs.odds
    closing_quotes = build_closing_quote_index(labels, odds)
    output_dir.mkdir(parents=True, exist_ok=True)
    require_prediction_before_odds = not args.allow_retrospective_proxy

    probability_benchmarks = {
        timing: _comparison_payload(
            predictions,
            labels,
            odds,
            timing=timing,
            require_prediction_before_odds=require_prediction_before_odds,
        )
        for timing in args.timings
    }

    scenario_summaries: list[dict[str, Any]] = []
    all_ledgers: list[pd.DataFrame] = []
    for timing_index, timing in enumerate(args.timings):
        for tax_index, (scenario, tax_rate) in enumerate(TAX_SCENARIOS.items()):
            config = BacktestConfig(
                bankroll_start=args.initial_bankroll,
                min_ev=args.min_ev,
                tax_rate=tax_rate,
                staking=staking,
                max_bets_per_match=1,
                odds_policy=ODDS_POLICIES[timing],
                min_minutes_before_start=0,
                require_prediction_before_odds=require_prediction_before_odds,
            )
            result = run_backtest(predictions, labels, odds, config)
            ledger = result_ledger(
                result.bets,
                cohort_by_id=inputs.cohort_by_id,
                closing_quotes=closing_quotes,
                scenario=scenario,
                timing=timing,
                tax_rate=tax_rate,
                staking_strategy=staking.strategy,
            )
            scenario_seed = args.seed + timing_index * 10 + tax_index
            summary = summarize_result(
                result,
                ledger,
                scenario=scenario,
                timing=timing,
                tax_rate=tax_rate,
                min_ev=args.min_ev,
                initial_bankroll=args.initial_bankroll,
                staking=staking,
                bootstrap_resamples=args.bootstrap_resamples,
                seed=scenario_seed,
            )
            scenario_summaries.append(summary)
            all_ledgers.append(ledger)
            ledger.to_csv(
                output_dir / f"ledger_{timing}_{scenario}.csv",
                index=False,
            )

    scenario_table = pd.DataFrame(scenario_summaries)
    scenario_table.to_csv(output_dir / "roi_scenarios.csv", index=False)
    combined_ledger = (
        pd.concat(all_ledgers, ignore_index=True)
        if all_ledgers
        else pd.DataFrame()
    )
    combined_ledger.to_csv(
        output_dir / "roi_ledger_all_scenarios.csv",
        index=False,
    )

    start_times = [
        label.start_time for label in labels if label.start_time is not None
    ]
    result_recorded_at = [
        getattr(label, "result_available_at", None)
        for label in labels
        if getattr(label, "result_available_at", None) is not None
    ]
    if args.allow_retrospective_proxy:
        scope = (
            "retrospective prediction proxy benchmark; "
            "not executable live performance and not promotion evidence"
        )
        prediction_quote_order = (
            "not enforced because --allow-retrospective-proxy was explicit"
        )
        source_limitation = (
            "Prediction timestamps are absent or retrospective; this run cannot "
            "support candidate, shadow, or production promotion."
        )
    else:
        scope = (
            "strict timestamped historical benchmark; "
            "not proof of future profitability"
        )
        prediction_quote_order = (
            "data_cutoff_at <= predicted_at <= quote_at < match start"
        )
        source_limitation = (
            "Stored timestamps establish decision order but do not independently "
            "prove every upstream feature timestamp."
        )

    report = {
        "generated_at": generated_at.isoformat(),
        "output_dir": str(output_dir),
        "scope": scope,
        "source": inputs.source,
        "predictions_loaded": len(predictions),
        "labels_loaded": len(labels),
        "odds_quotes_loaded": len(odds),
        "model_name": args.model_name,
        "model_version": args.model_version,
        "data_dates": {
            "minimum": min(start_times).date().isoformat() if start_times else None,
            "maximum": max(start_times).date().isoformat() if start_times else None,
        },
        "result_availability_dates": {
            "minimum": (
                min(result_recorded_at).isoformat()
                if result_recorded_at
                else None
            ),
            "maximum": (
                max(result_recorded_at).isoformat()
                if result_recorded_at
                else None
            ),
        },
        "selection": (
            "one highest model-implied EV side/bookmaker per match; "
            "eligibility is recalculated separately for each tax rate"
        ),
        "configuration": {
            "initial_bankroll": args.initial_bankroll,
            "staking_strategy": staking.strategy,
            "fixed_stake": (
                staking.fixed_stake if staking.strategy == "fixed" else None
            ),
            "kelly_fraction": (
                staking.kelly_fraction
                if staking.strategy == "fractional_kelly"
                else None
            ),
            "max_stake_fraction": getattr(
                staking, "max_bankroll_fraction", getattr(staking, "bankroll_fraction", None)
            ),
            "max_bets_per_match": 1,
            "timings": args.timings,
            "tax_scenarios": TAX_SCENARIOS,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "require_prediction_before_odds": (
                require_prediction_before_odds
            ),
        },
        "metric_definitions": {
            "expected_roi": (
                "stake-weighted model EV: sum(stake * EV) / sum(stake)"
            ),
            "true_roi_on_initial_bankroll": (
                "realized profit / initial bankroll"
            ),
            "yield_on_money_bet": "realized profit / total money bet",
            "maximum_drawdown": (
                "largest peak-to-trough decline in total bankroll "
                "after settlement events"
            ),
            "tax": (
                "Polish turnover convention: effective decimal return is "
                "odds * (1 - tax_rate)"
            ),
            "fractional_kelly": (
                "available bankroll * kelly_fraction * full Kelly fraction, "
                "capped by max_stake_fraction of available bankroll"
            ),
        },
        "temporal_ledger": {
            "prediction_quote_order": prediction_quote_order,
            "placement": "selected stored quote timestamp",
            "capital": "stake reserved until the result became available",
            "settlement": (
                "canonical_matches.result_recorded_at; rows at or before "
                "kickoff are ineligible"
            ),
        },
        "probability_benchmarks": probability_benchmarks,
        "limitations": [
            source_limitation,
            (
                "Closing prices are a diagnostic upper-information benchmark, "
                "not an executable entry claim."
            ),
            (
                "Result availability may lag the actual match end, so capital "
                "reservation is conservative."
            ),
            (
                "Expected ROI is model-implied and is not a guaranteed future "
                "return."
            ),
            (
                "Kelly sizing assumes accurate probabilities; estimation and "
                "calibration errors compound into stake-size errors."
            ),
            (
                "Yield intervals are calendar-week block bootstraps conditional "
                "on the realized stake path."
            ),
        ],
        "scenarios": scenario_summaries,
    }
    with (output_dir / "roi_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
