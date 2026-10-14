"""Financial Betting Benchmark & EV Calibration on 48-Hour Advance Window (N = 441 Matches).

Evaluates historical Shrunk Hybrid A1 at ~38.1h prior to kickoff, not recorded execution:
1. Loads matches with odds available in [36h, 72h] before kickoff.
2. Identifies best available odds at 48h across licensed sportsbooks.
3. Tracks Closing Line Value (CLV) against sharp closing quotes (< 2h prior).
4. Evaluates EV Calibration Bins (Expected EV vs Realized Cash Yield per bracket).
5. Tests both 0% Tax (Betclic Promotion / International) and Polish 12% Turnover Tax.
The financial policy retains its explicitly assumed research haircut, not measured uncertainty.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.analysis.unified_evaluation_engine import UnifiedBettingEngine, EvaluatedBet, SimulationSummary
from scripts.run_unified_betting_evaluation import format_conditional_metric


def print_portfolio_summary(summary: SimulationSummary) -> None:
    """Render existing results without opening a database or replaying the sealed study."""
    print(f"\n--- PORTFOLIO SUMMARY ---")
    print(f"Qualified Bets:       {summary.bets_count}")
    print(f"Average Odds Taken:   {format_conditional_metric(summary.avg_odds, '.2f')}")
    print(f"Total Staked:         {summary.total_staked:.2f} zł")
    print(f"Net Cash Profit:      {summary.total_pnl:+.2f} zł")
    print(f"Ending Bankroll:      {summary.final_bankroll:.2f} zł (from 1,000 zł initial)")
    print(f"Realized ROI / Yield: {format_conditional_metric(summary.roi_pct, '+.2f', '%')}")
    print(f"Expected ROI / EV:    {format_conditional_metric(summary.expected_ev_pct, '+.2f', '%')}")
    print(f"Calibration Gap (PLN):{format_conditional_metric(summary.calibration_gap_pln, '+.2f', ' zł / bet')}")
    print(f"Maximum Drawdown:     {summary.max_drawdown_pct:.2f}%")
    if summary.median_clv_pct is not None:
        print(f"Median Realized CLV:  {summary.median_clv_pct:+.2f}% (vs market close)")

    # These bins contain eligible bets only; their conditional metrics are defined.
    print(f"\n" + "-" * 110)
    print(f"{'EV Calibration Bracket':<24} | {'Bets':<5} | {'Win Rate':<8} | {'Avg Odds':<8} | {'Expected EV':<12} | {'Realized Yield':<14} | {'Gap (Yield - EV)':<16}")
    print("-" * 110)
    for b in summary.bins:
        if b.bets_count > 0:
            print(f"{b.label:<24} | {b.bets_count:<5} | {b.win_rate_pct:6.1f}%  | {b.avg_odds:<8.2f} | {b.expected_ev_pct:+10.2f}%  | {b.realized_ev_pct:+12.2f}%  | {b.calibration_gap_pct:+12.2f} pp")
    print("-" * 110)

    print(f"\n" + "-" * 110)
    print(f"{'Odds Bracket':<24} | {'Bets':<5} | {'Win Rate':<8} | {'Avg Odds':<8} | {'Expected EV':<12} | {'Realized Yield':<14} | {'Net PnL':<12}")
    print("-" * 110)
    for b in summary.by_odds_bracket:
        if b.bets_count > 0:
            print(f"{b.label:<24} | {b.bets_count:<5} | {b.win_rate_pct:6.1f}%  | {b.avg_odds:<8.2f} | {b.expected_ev_pct:+10.2f}%  | {b.realized_ev_pct:+12.2f}%  | {b.total_pnl:+10.2f} zł")
    print("-" * 110 + "\n")


def main():
    research_root = Path((project_root / "data/research_root.txt").read_text().strip())
    print("=== Loading 48-Hour Advance Window Cohort from Database ===")
    paired_path = research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    paired_df = pd.read_parquet(paired_path)
    paired_df["golgg_match_id"] = paired_df["golgg_match_id"].astype(str)

    a1_path = project_root / "data/07_model_output/upgrades/a0_tier_calibrated_p0_p2.parquet"
    a1_df = pd.read_parquet(a1_path)
    a1_df["golgg_match_id"] = a1_df["golgg_match_id"].astype(str)
    paired_df = paired_df.merge(a1_df[["golgg_match_id", "p_tier_cal"]], on="golgg_match_id", how="left")

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://betting:13cd8fbc0b149a22b7748262e25006fa6d502baaa8383de12efa20b00d5589dc@192.168.1.17:5432/betting",
    )
    engine = create_engine(db_url)

    with engine.connect() as conn:
        matches = pd.read_sql(
            text("""
            SELECT cm.id as canonical_match_id, gmm.golgg_match_id, cm.team_a_name, cm.team_b_name,
                   cm.winner_side, cm.start_time_normalized, cm.league, cm.best_of
            FROM canonical_matches cm
            JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
            WHERE cm.status IN ('finished', 'completed') AND cm.winner_side IN ('team_a', 'team_b')
            ORDER BY cm.start_time_normalized ASC
            """),
            conn,
        )

        odds = pd.read_sql(
            text("""
            SELECT os.canonical_match_id, b.name as bookmaker, os.odds_a, os.odds_b, os.scraped_at
            FROM odds_snapshots os
            JOIN bookmakers b ON b.id = os.bookmaker_id
            WHERE os.market_type = 'match_winner' AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a > 1.0 AND os.odds_b > 1.0
            ORDER BY os.scraped_at ASC
            """),
            conn,
        )

    matches["golgg_match_id"] = matches["golgg_match_id"].astype(str)
    cohort = matches.merge(paired_df, on="golgg_match_id", how="inner")

    # Side alignment check: team_a == team1 or team_a == team2
    cohort["is_team_a_team1"] = (
        ((cohort["winner_side"] == "team_a") & (cohort["y"] == 1)) |
        ((cohort["winner_side"] == "team_b") & (cohort["y"] == 0))
    )

    df_odds = odds.merge(
        cohort[["canonical_match_id", "start_time_normalized", "is_team_a_team1"]],
        on="canonical_match_id",
        how="inner",
    )
    df_odds["start_dt"] = pd.to_datetime(df_odds["start_time_normalized"], format="mixed", utc=True)
    df_odds["scraped_dt"] = pd.to_datetime(df_odds["scraped_at"], format="mixed", utc=True)
    df_odds = df_odds[df_odds["scraped_dt"] < df_odds["start_dt"]].copy()
    df_odds["hours_before"] = (df_odds["start_dt"] - df_odds["scraped_dt"]).dt.total_seconds() / 3600.0

    # Multiplicative devigging
    inv_a = 1.0 / df_odds["odds_a"]
    inv_b = 1.0 / df_odds["odds_b"]
    tot = inv_a + inv_b
    prop_p_a = inv_a / tot

    df_odds["p_mkt_team1"] = np.where(df_odds["is_team_a_team1"], prop_p_a, 1.0 - prop_p_a)
    df_odds["odds_team1"] = np.where(df_odds["is_team_a_team1"], df_odds["odds_a"], df_odds["odds_b"])
    df_odds["odds_team2"] = np.where(df_odds["is_team_a_team1"], df_odds["odds_b"], df_odds["odds_a"])

    # 48-Hour Horizon: Snapshots in [36h, 72h] before kickoff
    df_48h = df_odds[(df_odds["hours_before"] >= 36.0) & (df_odds["hours_before"] <= 72.0)].copy()
    print(f"Total snapshots in 48h window ([36h, 72h]): {len(df_48h)}")
    print(f"Matches with 48h odds: {df_48h['canonical_match_id'].nunique()}")

    h48_per_book = df_48h.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
    h48_cons = h48_per_book.groupby("canonical_match_id").agg(
        h48_mkt_p=("p_mkt_team1", "mean"),
        h48_hours_before=("hours_before", "mean"),
        best_odds_team1=("odds_team1", "max"),
        best_odds_team2=("odds_team2", "max"),
        best_book_team1=("bookmaker", "first"),
        best_book_team2=("bookmaker", "last"),
    ).reset_index()

    # Closing odds for CLV (< 2h before kickoff)
    df_close = df_odds[df_odds["hours_before"] <= 2.0].copy()
    if len(df_close) > 0:
        close_per_book = df_close.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
        close_cons = close_per_book.groupby("canonical_match_id").agg(
            close_odds_team1=("odds_team1", "mean"),
            close_odds_team2=("odds_team2", "mean"),
            close_mkt_p=("p_mkt_team1", "mean"),
        ).reset_index()
    else:
        close_cons = pd.DataFrame(columns=["canonical_match_id", "close_odds_team1", "close_odds_team2", "close_mkt_p"])

    # Master 48h evaluation cohort
    eval_48h = cohort.merge(h48_cons, on="canonical_match_id", how="inner").merge(close_cons, on="canonical_match_id", how="left")
    print(f"Master 48-Hour Advance Cohort: N = {len(eval_48h)} matches (Mean {eval_48h['h48_hours_before'].mean():.1f}h prior to start).")

    # Shrunk Hybrid at 48h: 50/50 logit blend between A1 and 48h Market Consensus
    p_a1 = eval_48h["p_tier_cal"].values
    p_mkt_48h = eval_48h["h48_mkt_p"].values

    zm = np.log(np.clip(p_a1, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p_a1, 1e-6, 1.0 - 1e-6)))
    zk = np.log(np.clip(p_mkt_48h, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p_mkt_48h, 1e-6, 1.0 - 1e-6)))
    zh = 0.50 * zm + 0.50 * zk
    p_hyb_48h = 1.0 / (1.0 + np.exp(-zh))
    p_hyb_48h = np.clip(p_hyb_48h, 1e-6, 1.0 - 1e-6)
    eval_48h["p_hybrid_48h"] = p_hyb_48h

    all_simulation_results = {}

    for tax in [0.0, 0.12]:
        tax_str = "0% Tax (Betclic / International)" if tax == 0.0 else "Polish 12% Turnover Tax"
        print("\n" + "=" * 110)
        print(f"48-HOUR ADVANCE WINDOW BETTING BENCHMARK ({tax_str})")
        print(f"1/4 Kelly Staking, 1,000 PLN Initial Capital, Max Fraction = 2.5%, Min EV = +3%")
        print("=" * 110)

        b_engine = UnifiedBettingEngine(
            tax_rate=tax,
            min_ev_net=0.03,
            max_ev_longshot=0.25,
            longshot_odds_threshold=3.50,
            max_bo1_market_gap=0.12,
            bo1_max_odds=3.00,
        )

        evaluated_bets: list[EvaluatedBet] = []

        for idx, row in eval_48h.iterrows():
            best_of_val = int(row.get("best_of") or row.get("best_of_x") or row.get("best_of_y") or 1)
            league_val = str(row.get("league") or row.get("league_x") or row.get("league_y") or "")
            p_mod1 = float(row["p_hybrid_48h"])
            p_mod2 = 1.0 - p_mod1
            p_standalone_1 = float(row["p_tier_cal"])
            p_standalone_2 = 1.0 - p_standalone_1
            y_won1 = bool(row["y"] == 1)

            # Check Team 1 Quote at 48h
            odds1 = float(row["best_odds_team1"])
            close_odds1 = float(row["close_odds_team1"]) if pd.notna(row["close_odds_team1"]) else None
            if odds1 <= 3.50:
                elig1, bet1 = b_engine.qualify_quote(
                    match_id=str(row["canonical_match_id"]),
                    side="team1",
                    odds=odds1,
                    odds_close=close_odds1,
                    prob_model=p_mod1,
                    prob_market_novig=float(row["h48_mkt_p"]),
                    won=y_won1,
                    league=league_val,
                    best_of=best_of_val,
                    bookmaker=str(row["best_book_team1"]),
                    prob_standalone_model=p_standalone_1,
                    uncertainty_basis="assumed_research_haircut",
                )
                if elig1:
                    evaluated_bets.append(bet1)

            # Check Team 2 Quote at 48h
            odds2 = float(row["best_odds_team2"])
            close_odds2 = float(row["close_odds_team2"]) if pd.notna(row["close_odds_team2"]) else None
            if odds2 <= 3.50:
                elig2, bet2 = b_engine.qualify_quote(
                    match_id=str(row["canonical_match_id"]),
                    side="team2",
                    odds=odds2,
                    odds_close=close_odds2,
                    prob_model=p_mod2,
                    prob_market_novig=1.0 - float(row["h48_mkt_p"]),
                    won=not y_won1,
                    league=league_val,
                    best_of=best_of_val,
                    bookmaker=str(row["best_book_team2"]),
                    prob_standalone_model=p_standalone_2,
                    uncertainty_basis="assumed_research_haircut",
                )
                if elig2:
                    evaluated_bets.append(bet2)

        # Run 1/4 Kelly simulation
        summary = b_engine.run_simulation(
            evaluated_bets,
            strategy="quarter_kelly",
            initial_bankroll=1000.0,
            cap_fraction=0.025,
        )

        print_portfolio_summary(summary)
        all_simulation_results[f"tax_{int(tax*100)}"] = summary.to_dict()

    out_file = project_root / "data/08_reporting/48h_betting_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_simulation_results, f, indent=2)
    print(f"Saved complete 48h betting benchmark results to {out_file}")


if __name__ == "__main__":
    main()
