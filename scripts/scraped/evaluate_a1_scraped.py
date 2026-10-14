"""Evaluate Consolidated A1 on Scraped Production Matches (850 Finished Fixtures).

Queries the real PostgreSQL production database on 192.168.1.17:
1. canonical_matches (850 finished matches with verified winner_side)
2. odds_snapshots (3,763 quotes across Polish bookmakers)
3. Computes Bookmaker Market Open, Base Causal A0, Consolidated A1, and Shrunk Hybrid A1
4. Runs historical research simulation with an explicitly assumed haircut (not measured uncertainty)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.shin_devig import shin_implied_probabilities
from src.analysis.unified_evaluation_engine import UnifiedBettingEngine, EvaluatedBet, SimulationSummary
from scripts.run_unified_betting_evaluation import format_conditional_metric


def print_portfolio_summary(summary: SimulationSummary) -> None:
    """Render a portfolio without acquiring data or replaying the research study."""
    win_rate = format_conditional_metric(
        summary.win_rate_pct, ".1f", f"% ({summary.wins_count}/{summary.bets_count})"
    )
    print(f"Qualified Bets:     {summary.bets_count}")
    print(f"Win Rate:           {win_rate}")
    print(f"Total Staked:       {summary.total_staked:.2f} zł")
    print(f"Net Cash Profit:    {summary.total_pnl:+.2f} zł")
    print(f"Ending Bankroll:    {summary.final_bankroll:.2f} zł")
    print(f"Realized ROI:       {format_conditional_metric(summary.roi_pct, '+.2f', '%')}")
    print(f"Max Drawdown:       {summary.max_drawdown_pct:.2f}%")


def main():
    import os
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://betting:13cd8fbc0b149a22b7748262e25006fa6d502baaa8383de12efa20b00d5589dc@timescaledb:5432/betting")
    engine = create_engine(db_url)

    print("=== Loading 850 Scraped Production Matches from Database ===")
    with engine.connect() as conn:
        # 1. Fetch finished canonical matches
        matches_df = pd.read_sql(
            text("""
            SELECT id as canonical_match_id, team_a_name, team_b_name, winner_side,
                   start_time_normalized, league, best_of
            FROM canonical_matches
            WHERE status IN ('finished', 'completed') AND winner_side IN ('team_a', 'team_b')
            ORDER BY start_time_normalized ASC
            """),
            conn,
        )

        # 2. Fetch existing predictions
        preds_df = pd.read_sql(
            text("""
            SELECT canonical_match_id, model_name, model_version, prob_a, prob_b, diagnostics_json
            FROM canonical_predictions
            WHERE prediction_status = 'active'
            """),
            conn,
        )

        # 3. Fetch pre-match odds snapshots
        odds_df = pd.read_sql(
            text("""
            SELECT os.canonical_match_id, b.name as bookmaker, os.odds_a, os.odds_b, os.scraped_at
            FROM odds_snapshots os
            JOIN bookmakers b ON b.id = os.bookmaker_id
            WHERE os.market_type = 'match_winner' AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a > 1.0 AND os.odds_b > 1.0
            """),
            conn,
        )

    print(f"Loaded {len(matches_df)} finished matches, {len(preds_df)} predictions, {len(odds_df)} odds quotes.")

    # Deduplicate predictions to one per match
    preds_clean = preds_df.drop_duplicates(subset=["canonical_match_id"]).copy()

    # Pre-match odds filtering
    df = preds_clean.merge(matches_df, on="canonical_match_id").merge(odds_df, on="canonical_match_id")
    df = df[df["scraped_at"] < df["start_time_normalized"]].copy()
    print(f"Pre-match odds snapshots available for {df['canonical_match_id'].nunique()} matches.")

    # Compute Shin fair market odds
    shin_probs_a = []
    shin_probs_b = []
    for _, r in df.iterrows():
        try:
            pa, pb = shin_implied_probabilities(float(r["odds_a"]), float(r["odds_b"]))
        except Exception:
            inv_a = 1.0 / r["odds_a"]
            inv_b = 1.0 / r["odds_b"]
            tot = inv_a + inv_b
            pa, pb = inv_a / tot, inv_b / tot
        shin_probs_a.append(pa)
        shin_probs_b.append(pb)

    df["shin_market_a"] = shin_probs_a
    df["shin_market_b"] = shin_probs_b

    # Opening odds consensus per match
    first_odds = df.sort_values("scraped_at").groupby(["canonical_match_id", "bookmaker"]).first().reset_index()
    cons = first_odds.groupby("canonical_match_id").agg(
        market_open_a=("shin_market_a", "median"),
        market_open_b=("shin_market_b", "median"),
        best_odds_a=("odds_a", "max"),
        best_odds_b=("odds_b", "max"),
        best_book_a=("bookmaker", "first"),
        best_book_b=("bookmaker", "last"),
    ).reset_index()

    cohort = preds_clean.merge(matches_df, on="canonical_match_id").merge(cons, on="canonical_match_id")
    print(f"Final evaluated single-cohort size: N = {len(cohort)} matches.")

    y_true = np.where(cohort["winner_side"] == "team_a", 1.0, 0.0)
    p_market = np.clip(cohort["market_open_a"].values, 1e-6, 1.0 - 1e-6)
    p_base_a0 = np.clip(cohort["prob_a"].values, 1e-6, 1.0 - 1e-6)

    # 4. Compute Consolidated A1 (Tier Parity Scaling on Major/Dev)
    from src.models.competition_tiers import classify_competition
    leagues = cohort["league"].values
    tiers = [classify_competition(l).tier.value for l in leagues]
    is_major_dev = np.isin(tiers, ["major", "development"])

    z_base = np.log(p_base_a0 / (1.0 - p_base_a0))
    z_a1 = z_base.copy()
    z_a1[is_major_dev] *= 0.94  # Tier parity scaling
    p_a1 = 1.0 / (1.0 + np.exp(-z_a1))
    p_a1 = np.clip(p_a1, 1e-6, 1.0 - 1e-6)

    # 5. Compute Shrunk Hybrid A1 (50/50 logit blend with Shin market)
    z_market = np.log(p_market / (1.0 - p_market))
    z_hybrid = 0.50 * z_a1 + 0.50 * z_market
    p_hybrid = 1.0 / (1.0 + np.exp(-z_hybrid))
    p_hybrid = np.clip(p_hybrid, 1e-6, 1.0 - 1e-6)

    def compute_metrics(p, y):
        ll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        brier = np.mean((p - y) ** 2)
        acc = np.mean((p >= 0.5) == y)
        blowouts = np.sum(-np.log(np.where(y == 1, p, 1.0 - p)) >= 2.5)
        return ll, brier, acc, blowouts

    m_mkt = compute_metrics(p_market, y_true)
    m_a0 = compute_metrics(p_base_a0, y_true)
    m_a1 = compute_metrics(p_a1, y_true)
    m_hyb = compute_metrics(p_hybrid, y_true)

    print("\n=========================================================================")
    print("ACCURACY BENCHMARK ON 850 SCRAPED PRODUCTION MATCHES")
    print("=========================================================================")
    print(f"{'Model Configuration':<32} | {'LogLoss':<8} | {'Brier':<8} | {'Accuracy':<8} | {'Blowouts (>=2.5)':<16}")
    print("-" * 80)
    print(f"{'Bookmaker OPEN (Shin Devig)':<32} | {m_mkt[0]:.4f}   | {m_mkt[1]:.4f}   | {m_mkt[2]*100:.1f}%    | {m_mkt[3]:<16}")
    print(f"{'Base Causal A0':<32} | {m_a0[0]:.4f}   | {m_a0[1]:.4f}   | {m_a0[2]*100:.1f}%    | {m_a0[3]:<16}")
    print(f"{'Consolidated A1 (Tier Parity)':<32} | {m_a1[0]:.4f}   | {m_a1[1]:.4f}   | {m_a1[2]*100:.1f}%    | {m_a1[3]:<16}")
    print(f"{'Shrunk Hybrid A1 (alpha=0.50)':<32} | {m_hyb[0]:.4f}   | {m_hyb[1]:.4f}   | {m_hyb[2]*100:.1f}%    | {m_hyb[3]:<16}")
    print("=========================================================================\n")
    print("=== Running Financial Betting Simulations (1/4 Kelly, 1000 PLN Bankroll) ===")

    for gated in [False, True]:
        gated_label = "With Production Risk Gating (Rules A+B+C, Odds <= 3.50)" if gated else "Unconstrained Raw Bets"
        print(f"\n#########################################################################")
        print(f"FINANCIAL SIMULATION: {gated_label}")
        print(f"#########################################################################")

        for tax in [0.0, 0.12]:
            tax_str = "0% Tax (Betclic / International)" if tax == 0.0 else "Polish 12% Turnover Tax"
            b_engine = UnifiedBettingEngine(
                tax_rate=tax,
                min_ev_net=0.03,
                max_ev_longshot=0.25 if gated else 1.0,
                longshot_odds_threshold=3.50,
                max_bo1_market_gap=0.12 if gated else 1.0,
                bo1_max_odds=3.00 if gated else None,
            )

            evaluated_bets = []
            for _, row in cohort.iterrows():
                best_of_val = int(row["best_of"]) if pd.notna(row["best_of"]) else 1
                league_val = str(row["league"]) if pd.notna(row["league"]) else None
                p_mod = p_hybrid[_]
                y_won = bool(row["winner_side"] == "team_a")

                # Check Side A
                odds_a_val = float(row["best_odds_a"])
                if not gated or odds_a_val <= 3.50:
                    elig_a, bet_a = b_engine.qualify_quote(
                        match_id=str(row["canonical_match_id"]),
                        side="team_a",
                        odds=odds_a_val,
                        prob_model=float(p_mod),
                        prob_market_novig=float(row["market_open_a"]),
                        won=y_won,
                        league=league_val,
                        best_of=best_of_val,
                        bookmaker=str(row["best_book_a"]),
                        uncertainty_basis="assumed_research_haircut",
                    )
                    if elig_a:
                        evaluated_bets.append(bet_a)

                # Check Side B
                odds_b_val = float(row["best_odds_b"])
                if not gated or odds_b_val <= 3.50:
                    p_mod_b = 1.0 - p_mod
                    elig_b, bet_b = b_engine.qualify_quote(
                        match_id=str(row["canonical_match_id"]),
                        side="team_b",
                        odds=odds_b_val,
                        prob_model=float(p_mod_b),
                        prob_market_novig=float(row["market_open_b"]),
                        won=not y_won,
                        league=league_val,
                        best_of=best_of_val,
                        bookmaker=str(row["best_book_b"]),
                        uncertainty_basis="assumed_research_haircut",
                    )
                    if elig_b:
                        evaluated_bets.append(bet_b)

            sim_summary = b_engine.run_simulation(evaluated_bets, strategy="quarter_kelly", initial_bankroll=1000.0, cap_fraction=0.025)

            print(f"\n--- Strategy: 1/4 Kelly | {tax_str} ---")
            print_portfolio_summary(sim_summary)
    results_out = {
        "n_matches": len(cohort),
        "metrics": {
            "market_open": {"log_loss": m_mkt[0], "brier": m_mkt[1], "accuracy": m_mkt[2], "blowouts": int(m_mkt[3])},
            "base_a0": {"log_loss": m_a0[0], "brier": m_a0[1], "accuracy": m_a0[2], "blowouts": int(m_a0[3])},
            "consolidated_a1": {"log_loss": m_a1[0], "brier": m_a1[1], "accuracy": m_a1[2], "blowouts": int(m_a1[3])},
            "shrunk_hybrid_a1": {"log_loss": m_hyb[0], "brier": m_hyb[1], "accuracy": m_hyb[2], "blowouts": int(m_hyb[3])},
        }
    }

    out_file = project_root / "data/08_reporting/a1_scraped_production_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results_out, indent=2))
    print(f"\nSaved scraped production evaluation to {out_file}")


if __name__ == "__main__":
    main()
