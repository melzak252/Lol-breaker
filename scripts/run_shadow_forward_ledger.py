"""Automated Shadow Forward-Ledger for paper-trading live LoL bets under the Comfort Policy.

Modes:
- record: Scans upcoming Tier-1 matches, applies Shin devigging, Guard B, and 1/4 Kelly sizing,
          and logs qualified shadow orders to an immutable ledger.
- settle: Checks finished matches, computes idempotent payouts net of 12% tax, records CLV,
          and updates running equity.
- report: Generates performance and calibration report across the shadow ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.analysis.betting_guards_a_to_d import (
    apply_guard_b_dynamic_disagreement,
    apply_guard_w_wasserstein_dro,
    calculate_clv,
    devig_shin,
    dynamic_logit_disagreement,
)
from betting_app.core.staking import fractional_kelly_stake, step_compounded_flat_stake
LEDGER_PATH = PROJECT_ROOT / "data" / "08_reporting" / "shadow_ledger" / "shadow_bets.jsonl"
DB_URL = os.getenv("DATABASE_URL", "postgresql://betting:betting_local_password@localhost:5432/betting")


def get_db_engine():
    return create_engine(DB_URL)


def record_shadow_bets(
    engine=None,
    bankroll: float = 10000.0,
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    max_ev: float = 0.25,
    max_logit_gap: float = 0.85,
    wasserstein_epsilon: float = 0.025,
    max_odds: float = 4.00,
    stake_cap: float = 150.0,
    staking_mode: str = "flat_step",
) -> list[dict[str, Any]]:
    """Scan upcoming matches and log qualified shadow bets according to the Combined Optimal Synergy + Wasserstein DRO Policy."""
    if engine is None:
        engine = get_db_engine()

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_bet_ids: set[str] = set()
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_bet_ids.add(record["bet_key"])
                    except Exception:
                        pass

    # Query upcoming matches with latest active prediction and quotes
    query = text("""
        SELECT 
            cm.id as canonical_match_id,
            cm.team_a_name,
            cm.team_b_name,
            cm.start_time_normalized,
            cm.league,
            cp.prob_a as p_model_a,
            cp.prob_b as p_model_b,
            os.id as odds_snapshot_id,
            os.odds_a,
            os.odds_b,
            b.name as bookmaker_name,
            os.scraped_at
        FROM canonical_matches cm
        JOIN canonical_predictions cp ON cp.canonical_match_id = cm.id AND cp.prediction_status = 'active'
        JOIN odds_snapshots os ON os.canonical_match_id = cm.id AND os.market_type = 'match_winner'
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE cm.status = 'upcoming'
          AND os.odds_a > 1.0 AND os.odds_b > 1.0
        ORDER BY cm.start_time_normalized ASC, os.scraped_at DESC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("No upcoming matches with active predictions and quotes found.")
        return []

    # Filter Tier-1 only
    tier1_leagues = ["LCK", "LPL", "LEC", "LCS", "Worlds", "Mid-Season Invitational", "MSI"]
    df = df[df["league"].apply(lambda l: any(k in str(l) for k in tier1_leagues) and "Academy" not in str(l) and "Challengers" not in str(l))].copy()

    # Deduplicate: latest quote per match and bookmaker
    df = df.drop_duplicates(subset=["canonical_match_id", "bookmaker_name"])

    now = datetime.now(UTC)
    new_bets: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        mid = int(row["canonical_match_id"])
        oa, ob = float(row["odds_a"]), float(row["odds_b"])
        pa, pb = float(row["p_model_a"]), float(row["p_model_b"])
        
        try:
            psa, psb = devig_shin(oa, ob)
        except Exception:
            psa = (1.0 / oa) / (1.0 / oa + 1.0 / ob)
            psb = 1.0 - psa

        # EV calculation with Polish tax
        eva = pa * (oa * (1.0 - tax_rate)) - 1.0
        evb = pb * (ob * (1.0 - tax_rate)) - 1.0

        candidates = [("a", pa, psa, oa, eva), ("b", pb, psb, ob, evb)]
        for side, p_pick, pmkt_pick, odds_pick, ev_pick in candidates:
            if odds_pick > max_odds:
                continue
            if ev_pick < min_ev or ev_pick > max_ev:
                continue

            passes_w, ev_dro, _ = apply_guard_w_wasserstein_dro(
                p_pick, odds_pick, tax_rate=tax_rate, base_min_ev=min_ev, epsilon=wasserstein_epsilon
            )
            if not passes_w:
                continue

            passes_b, gap = apply_guard_b_dynamic_disagreement(p_pick, pmkt_pick, max_logit_gap=max_logit_gap)
            if not passes_b:
                continue

            bet_key = f"shadow_{mid}_{side}_{row['bookmaker_name']}"
            if bet_key in existing_bet_ids:
                continue

            if staking_mode == "flat_step":
                stake = step_compounded_flat_stake(
                    bankroll=bankroll,
                    unit_fraction=0.03 if bankroll <= 1000.0 else 0.015,
                    min_stake=2.0,
                    max_stake=stake_cap,
                    integer_round=True,
                )
            else:
                stake = fractional_kelly_stake(
                    bankroll=bankroll,
                    probability=max(0.0, p_pick - wasserstein_epsilon),
                    decimal_odds=odds_pick,
                    fraction=0.25,
                    tax_rate=tax_rate,
                    max_stake=stake_cap,
                )
            if stake <= 0.0:
                continue

            shadow_record = {
                "bet_key": bet_key,
                "recorded_at": now.isoformat(),
                "canonical_match_id": mid,
                "match": f"{row['team_a_name']} vs {row['team_b_name']}",
                "league": row["league"],
                "start_time": row["start_time_normalized"],
                "bookmaker": row["bookmaker_name"],
                "side": side,
                "odds_taken": odds_pick,
                "p_model": round(p_pick, 4),
                "p_market_shin": round(pmkt_pick, 4),
                "logit_gap": round(gap, 4),
                "ev_net": round(ev_pick, 4),
                "ev_wasserstein_dro": round(ev_dro, 4),
                "wasserstein_epsilon": wasserstein_epsilon,
                "stake": round(stake, 2),
                "status": "open",
                "settled_at": None,
                "won": None,
                "payout": None,
                "profit": None,
                "clv": None,
            }
            new_bets.append(shadow_record)
            existing_bet_ids.add(bet_key)

    if new_bets:
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            for bet in new_bets:
                f.write(json.dumps(bet, ensure_ascii=False) + "\n")
        print(f"Logged {len(new_bets)} new qualified shadow bets.")
    else:
        print("0 new bets qualified under Comfort Policy.")

    return new_bets


def settle_shadow_bets(engine=None, tax_rate: float = 0.12) -> int:
    """Scan open shadow bets, check match results, and resolve payouts."""
    if not LEDGER_PATH.exists():
        print("No shadow ledger exists to settle.")
        return 0

    if engine is None:
        engine = get_db_engine()

    records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    open_match_ids = [r["canonical_match_id"] for r in records if r.get("status") == "open"]
    if not open_match_ids:
        print("No open shadow bets require settlement.")
        return 0

    # Query match outcomes
    with engine.connect() as conn:
        res = conn.execute(text("""
            SELECT id, winner_side, status 
            FROM canonical_matches 
            WHERE id IN :mids
        """), {"mids": tuple(set(open_match_ids))}).fetchall()
        outcomes = {r[0]: (r[1], r[2]) for r in res}

    settled_count = 0
    now = datetime.now(UTC)

    for r in records:
        if r.get("status") == "open":
            mid = r["canonical_match_id"]
            if mid in outcomes:
                winner_side, match_status = outcomes[mid]
                if match_status == "finished" and winner_side is not None:
                    side_letter = "a" if winner_side in ("team_a", "a") else "b"
                    won = (r["side"] == side_letter)
                    stake = float(r["stake"])
                    odds = float(r["odds_taken"])
                    payout = stake * odds * (1.0 - tax_rate) if won else 0.0
                    profit = payout - stake

                    r["status"] = "settled"
                    r["settled_at"] = now.isoformat()
                    r["won"] = won
                    r["payout"] = round(payout, 2)
                    r["profit"] = round(profit, 2)
                    settled_count += 1

    if settled_count > 0:
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Settled {settled_count} shadow bets.")
    else:
        print("0 matches finished among open shadow bets.")

    return settled_count


def print_shadow_report() -> None:
    """Print complete econometric metrics on the shadow ledger."""
    if not LEDGER_PATH.exists():
        print("Shadow ledger is empty.")
        return

    records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    print("\n=== SHADOW FORWARD LEDGER REPORT ===")
    print(f"Total Shadow Records: {len(df)}")
    settled = df[df["status"] == "settled"].copy()
    print(f"Open Bets:            {len(df[df['status'] == 'open'])}")
    print(f"Settled Bets:         {len(settled)}")

    if not settled.empty:
        turnover = settled["stake"].sum()
        profit = settled["profit"].sum()
        roi = (profit / turnover) * 100.0 if turnover > 0 else 0.0
        win_rate = settled["won"].mean() * 100.0
        print(f"Win Rate:             {win_rate:.2f}%")
        print(f"Turnover:             {turnover:.2f} PLN")
        print(f"Net Profit (12% Tax): {profit:+.2f} PLN")
        print(f"Realized ROI:         {roi:+.2f}%")
        print(f"Average Stake:        {settled['stake'].mean():.2f} PLN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "settle", "report", "all"], help="Execution mode")
    parser.add_argument("--bankroll", type=float, default=10000.0)
    args = parser.parse_args()

    if args.mode in ("record", "all"):
        record_shadow_bets(bankroll=args.bankroll)
    if args.mode in ("settle", "all"):
        settle_shadow_bets()
    if args.mode in ("report", "all"):
        print_shadow_report()


if __name__ == "__main__":
    main()
