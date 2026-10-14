"""Deterministic historical backtest engine for model-vs-bookmaker analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from betting_app.core.ev import expected_value, fair_market_probabilities
from betting_app.ml.config import BacktestConfig
from betting_app.ml.metrics import max_drawdown, roi
from betting_app.ml.backtesting.odds_selection import select_quotes_for_match
from betting_app.ml.backtesting.settlement import settle_profit
from betting_app.ml.backtesting.staking import stake_for_bet
from betting_app.ml.backtesting.types import BacktestBet, BacktestResult, HistoricalPrediction, MatchLabel, OddsQuote


def run_backtest(
    predictions: Iterable[HistoricalPrediction],
    labels: Iterable[MatchLabel],
    odds_quotes: Iterable[OddsQuote],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Evaluate historical predictions against collected bookmaker odds.

    The engine places at most ``config.max_bets_per_match`` bets. With the
    default value ``1``, it chooses the single highest-EV side/bookmaker quote
    available under the odds timing policy.
    """

    config = config or BacktestConfig()
    labels_by_match = {label.canonical_match_id: label for label in labels}
    quotes_by_match: dict[int, list[OddsQuote]] = defaultdict(list)
    for quote in odds_quotes:
        quotes_by_match[quote.canonical_match_id].append(quote)

    bankroll = config.bankroll_start
    curve = [bankroll]
    bets: list[BacktestBet] = []
    matches_seen = 0

    for prediction in sorted(predictions, key=lambda p: (p.predicted_at is None, p.predicted_at, p.canonical_match_id)):
        label = labels_by_match.get(prediction.canonical_match_id)
        if label is None:
            continue
        matches_seen += 1

        quotes_for_match = quotes_by_match.get(prediction.canonical_match_id, [])
        if config.require_prediction_before_odds and prediction.predicted_at is not None:
            quotes_for_match = [q for q in quotes_for_match if q.scraped_at >= prediction.predicted_at]
        eligible_quotes = select_quotes_for_match(quotes_for_match, label, config)
        candidates: list[tuple[float, OddsQuote, str, float, float, float]] = []
        for quote in eligible_quotes:
            market_a, market_b = fair_market_probabilities(quote.odds_a, quote.odds_b)
            ev_a = expected_value(prediction.prob_a, quote.odds_a, config.tax_rate)
            ev_b = expected_value(prediction.prob_b, quote.odds_b, config.tax_rate)
            if ev_a > config.min_ev:
                candidates.append((ev_a, quote, "a", prediction.prob_a, quote.odds_a, market_a))
            if ev_b > config.min_ev:
                candidates.append((ev_b, quote, "b", prediction.prob_b, quote.odds_b, market_b))

        candidates.sort(key=lambda c: c[0], reverse=True)
        for ev, quote, side, model_prob, odds, market_prob in candidates[: max(0, config.max_bets_per_match)]:
            stake = stake_for_bet(bankroll, model_prob, odds, config.staking, config.tax_rate)
            if stake <= 0:
                continue
            bankroll_before = bankroll
            result, profit = settle_profit(stake, odds, side, label.winner_side, config.tax_rate)  # type: ignore[arg-type]
            bankroll += profit
            curve.append(bankroll)
            bets.append(
                BacktestBet(
                    canonical_match_id=prediction.canonical_match_id,
                    side=side,  # type: ignore[arg-type]
                    bookmaker_id=quote.bookmaker_id,
                    bookmaker_name=quote.bookmaker_name,
                    odds_snapshot_id=quote.odds_snapshot_id,
                    placed_at=quote.scraped_at,
                    odds=odds,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    ev=ev,
                    stake=stake,
                    profit=profit,
                    result=result,  # type: ignore[arg-type]
                    bankroll_before=bankroll_before,
                    bankroll_after=bankroll,
                    prediction_id=prediction.prediction_id,
                    settled_at=label.start_time if label is not None else None,
                    expected_profit=stake * ev,
                    available_bankroll_before=bankroll_before,
                    reserved_stake_after_placement=stake,
                )
            )

    total_staked = sum(b.stake for b in bets)
    total_profit = sum(b.profit for b in bets)
    expected_profit = sum(b.expected_profit or (b.stake * b.ev) for b in bets)
    wins = sum(1 for b in bets if b.result == "won")
    turnover = total_staked / config.bankroll_start if config.bankroll_start > 0 else 0.0
    expected_yield = expected_profit / total_staked if total_staked > 0 else 0.0
    bankroll_return = (bankroll - config.bankroll_start) / config.bankroll_start if config.bankroll_start > 0 else 0.0
    peak = config.bankroll_start
    dd_fraction = 0.0
    for val in curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd_fraction = max(dd_fraction, (peak - val) / peak)

    return BacktestResult(
        bets=bets,
        bankroll_start=config.bankroll_start,
        bankroll_end=bankroll,
        total_staked=total_staked,
        total_profit=total_profit,
        roi=roi(total_profit, total_staked),
        hit_rate=wins / len(bets) if bets else 0.0,
        max_drawdown=max_drawdown(curve),
        matches_seen=matches_seen,
        matches_bet=len({b.canonical_match_id for b in bets}),
        matches_temporally_ineligible=0,
        turnover=turnover,
        expected_profit=expected_profit,
        expected_yield=expected_yield,
        bankroll_return=bankroll_return,
        max_drawdown_fraction=dd_fraction,
        max_open_stake=max((b.stake for b in bets), default=0.0),
        max_open_bets=1 if bets else 0,
        bets_skipped_insufficient_funds=0,
    )
