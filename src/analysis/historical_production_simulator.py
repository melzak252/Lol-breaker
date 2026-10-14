"""Realistic chronological betting operations simulator and wallet engine.

Replays betting operations on production 2-hour decision trajectories:
- Tracks free capital, reserved capital, and open exposure.
- Enforces strict no-overdraft constraint.
- Deducts 12% Polish turnover tax on every winning settlement: payout = stake * odds * 0.88.
- Processes timeline events chronologically: decisions at T-48h, T-24h, T-12h, T-6h, T-2h, T-1h, T-30m,
  and settlements at match start + 3 hours.
- Evaluates policies: Unconstrained Raw C0, Shrunk Hybrid, and the Comfort Policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.analysis.betting_guards_a_to_d import (
    apply_guard_b_dynamic_disagreement,
    calculate_clv,
    devig_shin,
    dynamic_logit_disagreement,
)


@dataclass
class OpenBet:
    bet_id: str
    match_id: Any
    side: str
    stake: float
    odds: float
    placed_at: datetime
    settle_at: datetime
    ev_claimed: float
    p_model: float
    p_market: float
    clv: float


class OperationalWallet:
    """Cash and capital management wallet enforcing strict financial invariants."""

    def __init__(self, initial_capital: float = 10000.0) -> None:
        if initial_capital <= 0.0 or not math.isfinite(initial_capital):
            raise ValueError("Initial capital must be positive and finite.")
        self.initial_capital: float = float(initial_capital)
        self.free_capital: float = float(initial_capital)
        self.reserved_capital: float = 0.0
        self.open_bets: dict[str, OpenBet] = {}
        self.settled_bets: list[dict[str, Any]] = []
        self.equity_history: list[dict[str, Any]] = [{
            "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
            "equity": self.initial_capital,
            "free_capital": self.free_capital,
            "reserved_capital": 0.0,
            "open_bets_count": 0,
        }]

    @property
    def total_equity(self) -> float:
        return self.free_capital + self.reserved_capital

    def reserve(
        self,
        bet_id: str,
        amount: float,
        match_id: Any,
        side: str,
        odds: float,
        placed_at: datetime,
        settle_at: datetime,
        ev_claimed: float = 0.0,
        p_model: float = 0.5,
        p_market: float = 0.5,
        clv: float = 0.0,
    ) -> None:
        """Lock stake amount from free capital to reserve."""
        if amount <= 0.0 or not math.isfinite(amount):
            raise ValueError("Stake amount must be positive and finite.")
        if amount > self.free_capital + 1e-9:
            raise ValueError(
                f"Insufficient free capital ({self.free_capital:.2f}) for stake ({amount:.2f})"
            )
        if bet_id in self.open_bets:
            raise ValueError(f"Bet {bet_id} is already open.")

        self.free_capital -= amount
        self.reserved_capital += amount
        self.open_bets[bet_id] = OpenBet(
            bet_id=bet_id,
            match_id=match_id,
            side=side,
            stake=amount,
            odds=odds,
            placed_at=placed_at,
            settle_at=settle_at,
            ev_claimed=ev_claimed,
            p_model=p_model,
            p_market=p_market,
            clv=clv,
        )
        self.equity_history.append({
            "timestamp": placed_at,
            "equity": self.total_equity,
            "free_capital": self.free_capital,
            "reserved_capital": self.reserved_capital,
            "open_bets_count": len(self.open_bets),
        })

    def settle(
        self,
        bet_id: str,
        won: bool,
        settled_at: datetime,
        tax_rate: float = 0.12,
    ) -> dict[str, Any]:
        """Settle an open bet, credit net payout, and release reserved capital."""
        if bet_id not in self.open_bets:
            raise KeyError(f"Bet {bet_id} is not open.")

        bet = self.open_bets.pop(bet_id)
        stake = bet.stake
        self.reserved_capital -= stake

        if won:
            payout = stake * bet.odds * (1.0 - tax_rate)
            profit = payout - stake
        else:
            payout = 0.0
            profit = -stake

        self.free_capital += payout
        record = {
            "bet_id": bet.bet_id,
            "match_id": bet.match_id,
            "side": bet.side,
            "stake": stake,
            "odds": bet.odds,
            "ev_claimed": bet.ev_claimed,
            "p_model": bet.p_model,
            "p_market": bet.p_market,
            "clv": bet.clv,
            "placed_at": bet.placed_at,
            "settled_at": settled_at,
            "won": won,
            "payout": payout,
            "profit": profit,
            "free_capital": self.free_capital,
            "total_equity": self.total_equity,
        }
        self.settled_bets.append(record)
        self.equity_history.append({
            "timestamp": settled_at,
            "equity": self.total_equity,
            "free_capital": self.free_capital,
            "reserved_capital": self.reserved_capital,
            "open_bets_count": len(self.open_bets),
        })
        return record


class HistoricalProductionSimulator:
    """Replays discrete multi-horizon betting operations sequentially."""

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        policy_kind: str = "comfort_guard",
        alpha: float = 0.35,
        guard_b_logit: float | None = 0.85,
        min_ev: float = 0.05,
        max_ev: float | None = 0.25,
        tier1_only: bool = True,
        stake_mode: str = "flat",
        flat_stake: float = 100.0,
        max_stake_cap: float = 250.0,
        tax_rate: float = 0.12,
    ) -> None:
        self.initial_bankroll = float(initial_bankroll)
        self.policy_kind = policy_kind
        self.alpha = float(alpha)
        self.guard_b_logit = guard_b_logit
        self.min_ev = float(min_ev)
        self.max_ev = max_ev
        self.tier1_only = tier1_only
        self.stake_mode = stake_mode
        self.flat_stake = float(flat_stake)
        self.max_stake_cap = float(max_stake_cap)
        self.tax_rate = float(tax_rate)

    def run(self, trajectory_df: pd.DataFrame) -> dict[str, Any]:
        """Execute full chronological simulation across quote events and settlements."""
        wallet = OperationalWallet(initial_capital=self.initial_bankroll)
        df = trajectory_df.copy()

        # Ensure datetime columns are aware UTC
        df["decision_dt"] = pd.to_datetime(df["scraped_at"], utc=True)
        df["start_dt"] = pd.to_datetime(df["start_time_normalized"], utc=True)
        df["settle_dt"] = df["start_dt"] + pd.Timedelta(hours=3)

        # Build chronological event queue:
        # Event 1: Decision opportunities (grouped by decision_dt)
        # Event 2: Match settlements (grouped by settle_dt)
        matches_bet_ids: set[Any] = set()

        # Sort all decision rows chronologically
        decision_rows = df.sort_values(by="decision_dt")

        # Track settlements by timestamp
        settlement_events = []
        unique_matches = df.drop_duplicates(subset=["canonical_match_id"])
        for _, m in unique_matches.iterrows():
            if pd.notna(m.get("winner_side")) and m["winner_side"] in ("team_a", "team_b", "a", "b"):
                settlement_events.append({
                    "type": "settle",
                    "timestamp": m["settle_dt"],
                    "match_id": m["canonical_match_id"],
                    "winner_side": "a" if m["winner_side"] in ("team_a", "a") else "b",
                })
        settle_df = pd.DataFrame(settlement_events).sort_values(by="timestamp")

        # Merge decision opportunities and settlements into one chronological timeline
        events = []
        for _, r in decision_rows.iterrows():
            events.append({"type": "quote", "timestamp": r["decision_dt"], "row": r})
        for _, s in settle_df.iterrows():
            events.append({"type": "settle", "timestamp": s["timestamp"], "settle": s})

        events.sort(key=lambda x: x["timestamp"])

        for event in events:
            if event["type"] == "settle":
                s_info = event["settle"]
                mid = s_info["match_id"]
                # Settle all open bets for this match
                open_to_settle = [
                    bid for bid, ob in wallet.open_bets.items()
                    if ob.match_id == mid
                ]
                for bid in open_to_settle:
                    ob = wallet.open_bets[bid]
                    won = (ob.side == s_info["winner_side"])
                    wallet.settle(bid, won=won, settled_at=event["timestamp"], tax_rate=self.tax_rate)

            elif event["type"] == "quote":
                r = event["row"]
                mid = r["canonical_match_id"]
                # If we already have an open bet on this match, skip
                if mid in matches_bet_ids:
                    continue

                # Check Tier-1 filter if active
                if self.tier1_only:
                    league = str(r.get("league") or "")
                    is_t1 = any(k in league for k in ["LCK", "LPL", "LEC", "LCS", "Worlds", "MSI"]) and "Challengers" not in league and "Academy" not in league
                    if not is_t1:
                        continue

                # Must have start time in future
                if event["timestamp"] >= r["start_dt"]:
                    continue

                oa, ob = float(r["odds_a"]), float(r["odds_b"])
                psa = float(r["p_shin_a"])
                psb = 1.0 - psa

                # Obtain model probability
                if pd.isna(r.get("p_model_a")):
                    continue
                p_c0 = float(r["p_model_a"])

                # Determine effective probability based on policy
                if self.policy_kind in ("unconstrained", "comfort_guard"):
                    p_eff_a = p_c0
                elif self.policy_kind == "shrunk_hybrid":
                    # Bayesian logit shrinkage with market
                    lm = logit(max(1e-6, min(1.0 - 1e-6, p_c0)))
                    lk = logit(max(1e-6, min(1.0 - 1e-6, psa)))
                    p_eff_a = expit(self.alpha * lm + (1.0 - self.alpha) * lk)
                else:
                    p_eff_a = p_c0

                p_eff_b = 1.0 - p_eff_a

                # Calculate net EV under Polish tax
                eva = p_eff_a * (oa * (1.0 - self.tax_rate)) - 1.0
                evb = p_eff_b * (ob * (1.0 - self.tax_rate)) - 1.0

                # Select best side
                if eva >= evb and eva > self.min_ev:
                    side = "a"
                    p_pick, pmkt_pick, odds_pick, ev_pick = p_eff_a, psa, oa, eva
                elif evb > self.min_ev:
                    side = "b"
                    p_pick, pmkt_pick, odds_pick, ev_pick = p_eff_b, psb, ob, evb
                else:
                    continue

                # Check max EV ceiling (Optimizer's curse cap)
                if self.max_ev is not None and ev_pick > self.max_ev:
                    continue

                # Check Guard B (dynamic logit gap)
                gap = dynamic_logit_disagreement(p_pick, pmkt_pick)
                if self.guard_b_logit is not None and gap > self.guard_b_logit:
                    continue

                # Sizing
                if self.stake_mode == "flat":
                    stake = self.flat_stake
                elif self.stake_mode == "quarter_kelly":
                    b_odds = odds_pick * (1.0 - self.tax_rate) - 1.0
                    f_star = max(0.0, (b_odds * p_pick - (1.0 - p_pick)) / b_odds) if b_odds > 0 else 0.0
                    stake = min(self.max_stake_cap, wallet.free_capital * (0.25 * f_star))
                else:
                    stake = self.flat_stake

                stake = min(stake, self.max_stake_cap)
                if stake <= 0.0 or stake > wallet.free_capital:
                    continue

                bet_id = f"bet_{mid}_{r.get('snapshot_id', len(wallet.open_bets))}"
                clv = calculate_clv(odds_pick, 1.0 / pmkt_pick, tax_rate=self.tax_rate)
                try:
                    wallet.reserve(
                        bet_id=bet_id,
                        amount=stake,
                        match_id=mid,
                        side=side,
                        odds=odds_pick,
                        placed_at=event["timestamp"],
                        settle_at=r["settle_dt"],
                        ev_claimed=ev_pick,
                        p_model=p_pick,
                        p_market=pmkt_pick,
                        clv=clv,
                    )
                    matches_bet_ids.add(mid)
                except ValueError:
                    continue

        return self._summarize(wallet)

    def _summarize(self, wallet: OperationalWallet) -> dict[str, Any]:
        settled = pd.DataFrame(wallet.settled_bets)
        if settled.empty:
            return {
                "bets": 0, "win_rate": 0.0, "total_turnover": 0.0,
                "net_profit": 0.0, "roi_pct": 0.0, "final_equity": wallet.total_equity,
                "max_drawdown_units": 0.0, "max_drawdown_pct": 0.0,
                "profit_factor": 0.0, "sharpe_ratio": 0.0, "ev_gap": 0.0,
            }

        n_bets = len(settled)
        win_rate = settled["won"].mean() * 100.0
        total_turnover = settled["stake"].sum()
        net_profit = settled["profit"].sum()
        roi_pct = (net_profit / total_turnover) * 100.0 if total_turnover > 0 else 0.0

        claimed_ev = settled["ev_claimed"].mean() * 100.0
        ev_gap = claimed_ev - roi_pct

        # Drawdown analysis on equity curve
        eq_df = pd.DataFrame(wallet.equity_history)
        equities = eq_df["equity"].values
        running_max = np.maximum.accumulate(equities)
        dd_units = running_max - equities
        dd_pct = (dd_units / running_max) * 100.0
        max_dd_units = float(np.max(dd_units))
        max_dd_pct = float(np.max(dd_pct))

        # Profit factor
        gross_wins = settled[settled["profit"] > 0]["profit"].sum()
        gross_losses = np.abs(settled[settled["profit"] < 0]["profit"].sum())
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.nan

        # Sharpe ratio (per-bet normalized)
        returns = settled["profit"] / settled["stake"]
        sharpe = (returns.mean() / returns.std()) * np.sqrt(n_bets) if returns.std() > 0 else 0.0

        return {
            "bets": n_bets,
            "win_rate": round(win_rate, 2),
            "total_turnover": round(total_turnover, 2),
            "net_profit": round(net_profit, 2),
            "roi_pct": round(roi_pct, 2),
            "final_equity": round(wallet.total_equity, 2),
            "max_drawdown_units": round(max_dd_units, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "ev_gap": round(ev_gap, 2),
            "settled_df": settled,
            "equity_df": eq_df,
        }
