#!/usr/bin/env python3
"""Run betting diagnostics.

``production-calibration`` evaluates unblended A0 on frozen production quotes.
``input-repair`` replays frozen A0 with explicit roster and quote provenance.
The legacy invocation retains application qualification and portfolio simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.analysis.unified_evaluation_engine import (
    EvaluatedBet,
    UnifiedBettingEngine,
)

def format_conditional_metric(value: float | None, spec: str, suffix: str = "") -> str:
    """Keep populated CLI formatting while marking undefined no-bet metrics."""
    if value is None:
        return "N/A (no eligible bets)"
    return f"{value:{spec}}{suffix}"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified EV Calibration and Betting Backtest Runner."
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to evaluation dataset (Parquet or CSV). Default uses research best_available_odds.",
    )
    parser.add_argument(
        "--tax-rate",
        type=float,
        default=0.12,
        help="Turnover tax rate (default: 0.12 for Poland, 0.0 for Betclic/international).",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.03,
        help="Minimum net expected value threshold (default: 0.03 = +3%%).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Bayesian shrinkage weight toward consensus market odds (default: 0.25, 0.0 = pure model).",
    )
    parser.add_argument(
        "--max-odds",
        type=float,
        default=3.50,
        help="Hard odds ceiling to eliminate lottery swings and high-kurtosis variance (default: 3.50).",
    )
    parser.add_argument(
        "--strategy",
        choices=["flat", "quarter_kelly", "fixed_fraction"],
        default="quarter_kelly",
        help="Staking strategy to simulate (default: quarter_kelly).",
    )
    parser.add_argument(
        "--flat-stake",
        type=float,
        default=100.0,
        help="Flat stake in PLN (default: 100.0).",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll in PLN (default: 1000.0).",
    )
    parser.add_argument(
        "--max-coinflip-market-gap",
        type=float,
        default=0.08,
        help="Maximum model-market disagreement in coinflip corridor [1.80, 2.50] (default: 0.08).",
    )
    parser.add_argument(
        "--bo1-max-odds",
        type=float,
        default=3.00,
        help="Hard odds ceiling on Best-of-1 series to prevent high variance (default: 3.00).",
    )
    parser.add_argument(
        "--no-dynamic-hurdle",
        action="store_true",
        help="Disable low-odds dynamic hurdle penalty.",
    )
    return parser.parse_args()


def run_production_calibration(argv: list[str]) -> None:
    """Score a dated production-only cohort without fitting or simulating capital."""
    from src.analysis import bootstrap, production_market_benchmark
    from src.analysis.research_benchmark import sha

    parser = argparse.ArgumentParser(
        prog="run_unified_betting_evaluation.py production-calibration",
        description="Retrospective base-A0 EV/probability calibration on production quotes.",
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fixture-coverage", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "conf/base/research_benchmark.json"
    )
    parser.add_argument("--end-date", required=True, help="Inclusive UTC date, YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Defaults to six calendar months before end-date.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tax-rate", type=float, default=0.12)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument("--flat-stake", type=float, default=100.0)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    end_day = pd.Timestamp(args.end_date).tz_localize("UTC")
    start = (
        pd.Timestamp(args.start_date).tz_localize("UTC")
        if args.start_date
        else end_day - pd.DateOffset(months=6)
    )
    end = end_day + pd.Timedelta(days=1)
    if start >= end or start != start.normalize() or end_day != end_day.normalize():
        parser.error("Supply an ordered range of whole UTC calendar dates")

    root = args.research_root or os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    root_file = PROJECT_ROOT / "data/research_root.txt"
    if root is None and root_file.is_file():
        root = root_file.read_text().strip()
    if not root:
        parser.error("Supply --research-root, ENSEMBLE_RESEARCH_ROOT or data/research_root.txt")
    manifest = json.loads(args.manifest.read_text())
    forecast_path = Path(root).expanduser() / manifest["input"]["path"]
    forecast_hash = sha(forecast_path)
    if forecast_hash != manifest["input"]["sha256"]:
        raise ValueError("Canonical A0 forecast hash does not match the unchanged manifest")
    input_receipt = args.panel.parent / "input_receipt.json"
    receipt = json.loads(input_receipt.read_text())
    panel_hash = sha(args.panel)
    if panel_hash != receipt["panel_sha256"]:
        raise ValueError("Production panel hash does not match its input receipt")
    for relative_path, expected_hash in receipt["source_sha256"].items():
        if sha(PROJECT_ROOT / relative_path) != expected_hash:
            raise ValueError(f"Production panel source hash mismatch: {relative_path}")

    panel = pd.read_parquet(args.panel)
    coverage = pd.read_parquet(args.fixture_coverage)
    if coverage["canonical_match_id"].isna().any() or coverage["canonical_match_id"].duplicated().any():
        raise ValueError("Fixture coverage must have unique nonmissing canonical identities")
    # Production exports contain both ISO 'T' and SQL-style timestamp strings.
    coverage_starts = production_market_benchmark._aware_series(coverage["start_time_normalized"])
    if coverage_starts.isna().any():
        raise ValueError("Fixture coverage contains missing or timezone-naive start timestamps")
    panel_starts = production_market_benchmark._aware_series(panel["start_dt"])
    if panel_starts.isna().any():
        raise ValueError("Production panel contains missing or timezone-naive start timestamps")
    coverage_in_window = coverage_starts.between(start, end, inclusive="left")
    coverage = coverage.loc[coverage_in_window].copy()
    panel = panel.loc[panel_starts.between(start, end, inclusive="left")].copy()
    if panel.empty:
        raise ValueError("No production A0/odds rows within the requested date window")

    forecasts = pd.read_parquet(
        forecast_path, columns=["golgg_match_id", "date", "best_of", "y", "p_full"]
    )
    forecasts["golgg_match_id"] = forecasts["golgg_match_id"].astype(str)
    panel["golgg_match_id"] = panel["golgg_match_id"].astype(str)
    aligned = panel.merge(
        forecasts, on="golgg_match_id", how="left", validate="many_to_one",
        suffixes=("", "_canonical"), indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Production panel includes a match outside canonical A0 coverage")
    if not np.allclose(aligned["p_a0"], aligned["p_full"], rtol=0, atol=1e-12):
        raise ValueError("Production panel probabilities are not canonical base A0")
    for column in ("y", "best_of"):
        if not aligned[column].eq(aligned[f"{column}_canonical"]).all():
            raise ValueError(f"Production panel conflicts with canonical A0 {column}")
    if not pd.to_datetime(aligned["date"], utc=True).eq(
        pd.to_datetime(aligned["date_canonical"], utc=True)
    ).all():
        raise ValueError("Production panel dates conflict with canonical A0")
    eligible = coverage.loc[coverage["exclusion_reason"].fillna("").eq("")]
    lineage = panel[["canonical_match_id", "golgg_match_id"]].merge(
        eligible[["canonical_match_id", "golgg_match_id"]],
        on="canonical_match_id", how="left", validate="many_to_one",
        suffixes=("", "_coverage"),
    )
    if not lineage["golgg_match_id"].eq(lineage["golgg_match_id_coverage"]).all():
        raise ValueError("Panel identities conflict with the eligible production fixture ledger")

    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha(path)
        for path in (
            Path(__file__).resolve(),
            Path(production_market_benchmark.__file__).resolve(),
            Path(bootstrap.__file__).resolve(),
            PROJECT_ROOT / "betting_app/core/ev.py",
        )
    }
    protocol = {
        "status": "FROZEN_RETROSPECTIVE_PROTOCOL",
        "model": "Causal A0 annual, p_full; no blending, uncertainty adjustment or retraining",
        "population": "Production-recorded fixtures and quotes only",
        "window": {"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat()},
        "parameters": {
            "tax_rate": args.tax_rate, "min_ev_strictly_greater_than": args.min_ev,
            "flat_stake": args.flat_stake, "bootstrap_draws": args.draws, "seed": args.seed,
        },
        "selection": "Highest full-precision raw net EV; ties choose A; no additional gates",
        "ev_formula": "p * decimal_odds * (1 - tax_rate) - 1",
        "yield_formula": "sum(stake * (won * decimal_odds * (1 - tax_rate) - 1)) / sum(stake)",
        "bootstrap": "Paired calendar-month blocks from each entire pre-selection cohort; undefined replicas invalidate the CI",
        "interval_tolerances": {"yield_error": [-0.05, 0.05], "win_rate_error": [-0.03, 0.03]},
        "probability_bands": "Deciles and fixed near-40% band [0.35, 0.45)",
        "primary_cohort": "all, OPEN; other horizons and subsets are separate sensitivities",
        "inputs": {
            "panel": {"path": str(args.panel), "sha256": panel_hash},
            "fixture_coverage": {"path": str(args.fixture_coverage), "sha256": sha(args.fixture_coverage)},
            "manifest": {"path": str(args.manifest), "sha256": sha(args.manifest)},
            "canonical_forecasts": {
                "path_relative_to_research_root": manifest["input"]["path"], "sha256": forecast_hash,
            },
            "panel_input_receipt": {"path": str(input_receipt), "sha256": sha(input_receipt)},
        },
        "code_sha256": source_hashes,
        "limitations": [
            "Historical development, not a new independent holdout or production-readiness certificate",
            "Production odds paired with retrospective research A0, not logged operational A0 decisions",
            "Strict prior-day history does not establish roster publication or forecast creation time",
            "CLOSE is a retrospective price reference, not a claim of an executable decision",
            "Flat-return accounting is not Kelly, bankroll availability or execution simulation",
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    summaries, ledger, profiles = production_market_benchmark.evaluate_a0_calibration(
        panel, tax_rate=args.tax_rate, min_ev=args.min_ev, flat_stake=args.flat_stake,
        draws=args.draws, seed=args.seed,
    )
    coverage_summary = {
        "source_fixtures": int(len(coverage_in_window)),
        "fixtures_in_window": int(len(coverage)),
        "fixtures_outside_window": int((~coverage_in_window).sum()),
        "eligible_a0_fixtures": int(len(eligible)),
        "fixture_exclusions": coverage["exclusion_reason"].fillna("ELIGIBLE").replace("", "ELIGIBLE").value_counts().to_dict(),
        "source_start_range": [
            coverage_starts.loc[coverage_in_window].min().isoformat(),
            coverage_starts.loc[coverage_in_window].max().isoformat(),
        ],
        "evaluated_start_range": [panel["start_dt"].min().isoformat(), panel["start_dt"].max().isoformat()],
        "per_horizon": {
            horizon: {
                "matches": int(group["canonical_match_id"].nunique()),
                "eligible_without_horizon_quote": int(len(eligible) - group["canonical_match_id"].nunique()),
                "months": sorted(pd.to_datetime(group["date"]).dt.strftime("%Y-%m").unique().tolist()),
            }
            for horizon in production_market_benchmark.HORIZONS
            for group in [panel.loc[panel["horizon"].eq(horizon)]]
        },
    }
    result = {
        "status": "RETROSPECTIVE_PRODUCTION_ODDS_CALIBRATION",
        "coverage": coverage_summary,
        "results": summaries,
        "not_run": ["training", "new forecast generation", "independent holdout", "Kelly/execution", "deployment"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    coverage.to_csv(args.output_dir / "fixture_coverage.csv", index=False)
    ledger.to_csv(args.output_dir / "selected_bets.csv", index=False, float_format="%.17g")
    profiles.to_csv(args.output_dir / "calibration_profiles.csv", index=False, float_format="%.17g")
    flat_rows = []
    for row in summaries:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for metric in ("yield_gap", "win_rate_gap"):
            flat.update({f"{metric}_{key}": value for key, value in row[f"{metric}_interval"].items()})
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(args.output_dir / "summary.csv", index=False, float_format="%.17g")
    headline = pd.DataFrame([row for row in summaries if row["cohort"] == "all"])
    print(headline[["horizon", "n_matches", "bets", "mean_ev", "yield", "mean_probability", "win_rate"]].to_string(index=False))
    print(f"Retrospective results and provenance: {args.output_dir}")


def main() -> None:
    if sys.argv[1:2] == ["c0-market-study"]:
        run_c0_market_study(sys.argv[2:])
        return
    if sys.argv[1:2] == ["production-calibration"]:
        run_production_calibration(sys.argv[2:])
        return
    if sys.argv[1:2] == ["approach-comparison"]:
        from src.analysis.ev_approach_runner import approach_comparison_main

        approach_comparison_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["input-repair"]:
        from src.analysis.ev_input_repair import input_repair_main

        input_repair_main(sys.argv[2:])
        return
    args = parse_args()

    data_path = args.data
    if data_path is None:
        root_file = PROJECT_ROOT / "data/research_root.txt"
        if root_file.is_file():
            research_root = Path(root_file.read_text().strip())
            candidate = research_root / "individual-bookmakers-20260911/best_available_odds.parquet"
            if candidate.is_file():
                data_path = candidate
        if data_path is None:
            raise FileNotFoundError("Could not resolve default best_available_odds.parquet. Specify via --data.")

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)

    engine = UnifiedBettingEngine(
        tax_rate=args.tax_rate,
        min_ev_net=args.min_ev,
        dynamic_hurdle=not args.no_dynamic_hurdle,
        max_ev_longshot=0.20,
        longshot_odds_threshold=3.50,
        bo1_max_odds=args.bo1_max_odds,
        max_coinflip_market_gap=args.max_coinflip_market_gap,
        coinflip_min_odds=1.80,
        coinflip_max_odds=2.50,
        uncertainty_required=True,
    )

    qualified_bets: list[EvaluatedBet] = []

    # Detect format: historical best_available_odds vs database export
    if "best_a_open" in df.columns:
        from scipy.special import expit, logit

        for _, r in df.iterrows():
            mid = r.get("golgg_match_id", r.get("match_id"))
            oa = float(r["best_a_open"])
            ob = float(r["best_b_open"])
            ca = float(r["best_a_close"]) if "best_a_close" in r and pd.notna(r["best_a_close"]) else oa
            cb = float(r["best_b_close"]) if "best_b_close" in r and pd.notna(r["best_b_close"]) else ob
            y = int(r["y"])
            bo = int(r["best_of"]) if "best_of" in r and pd.notna(r["best_of"]) else None

            pa_raw = float(r["p_full"])
            inv_a, inv_b = 1.0 / oa, 1.0 / ob
            mkt_a, mkt_b = inv_a / (inv_a + inv_b), inv_b / (inv_a + inv_b)

            if args.alpha > 0.0:
                z_mod_a, z_mkt_a = logit(np.clip(pa_raw, 0.001, 0.999)), logit(np.clip(mkt_a, 0.001, 0.999))
                pa = float(expit((1.0 - args.alpha) * z_mod_a + args.alpha * z_mkt_a))
            else:
                pa = pa_raw
            pb = 1.0 - pa

            if args.max_odds is None or oa <= args.max_odds:
                el_a, bet_a = engine.qualify_quote(
                    match_id=mid, side="team_a", odds=oa, odds_close=ca, prob_model=pa,
                    won=y == 1, bookmaker=str(r.get("best_a_book_open", "unknown")), prob_market_novig=mkt_a,
                    best_of=bo, flat_stake=args.flat_stake,
                    # Retain the legacy research haircut, not measured model uncertainty.
                    uncertainty_basis="assumed_research_haircut",
                )
                if el_a:
                    qualified_bets.append(bet_a)

            if args.max_odds is None or ob <= args.max_odds:
                el_b, bet_b = engine.qualify_quote(
                    match_id=mid, side="team_b", odds=ob, odds_close=cb, prob_model=pb,
                    won=y == 0, bookmaker=str(r.get("best_b_book_open", "unknown")), prob_market_novig=mkt_b,
                    best_of=bo, flat_stake=args.flat_stake,
                    uncertainty_basis="assumed_research_haircut",
                )
                if el_b:
                    qualified_bets.append(bet_b)
    else:
        raise ValueError("Unsupported data format; requires best_available_odds format.")

    # Group by match and side
    best_bets: dict[tuple[Any, str], EvaluatedBet] = {}
    for b in qualified_bets:
        key = (b.match_id, b.side)
        if key not in best_bets or b.odds > best_bets[key].odds:
            best_bets[key] = b

    bets_to_sim = list(best_bets.values())
    print(f"Total matches evaluated: {len(df):,}")
    print(f"Qualified bets passing canonical safety gates: {len(bets_to_sim):,}")

    custom_ev_bins = [
        ("0% - 3% EV", 0.00, 0.03),
        ("3% - 6% EV", 0.03, 0.06),
        ("6% - 10% EV", 0.06, 0.10),
        ("> 10% EV", 0.10, 99.00),
    ]

    summary = engine.run_simulation(
        qualified_bets=bets_to_sim,
        strategy=args.strategy,
        initial_bankroll=args.initial_bankroll,
        flat_stake=args.flat_stake,
        ev_bins=custom_ev_bins,
    )

    print("\n" + "=" * 115)
    print(f"WYNIKI Z KANONICZNEGO SILNIKA (STRATEGIA: {summary.strategy.upper()}, PODATEK: {args.tax_rate*100:.1f}%)")
    print("=" * 115)
    print(f"Liczba zakładów:                  {summary.bets_count}")
    win_rate = format_conditional_metric(
        summary.win_rate_pct, ".2f", f"% ({summary.wins_count}/{summary.bets_count})"
    )
    print(f"Skuteczność (Win Rate):           {win_rate}")
    print(f"Średni kurs:                      {format_conditional_metric(summary.avg_odds, '.2f')}")
    print(f"Surowe (niekalibrowane) EV / bet: {format_conditional_metric(summary.raw_ev_pct, '+.2f', '%')}")
    print(f"Oczekiwane EV / bet (%):          {format_conditional_metric(summary.expected_ev_pct, '+.2f', '%')}")
    print(f"Oczekiwany zysk na 1 bet:         {format_conditional_metric(summary.expected_profit_per_bet, '+.2f', ' zł')}")
    print(f"Rzeczywiste EV / bet (ROI %):     {format_conditional_metric(summary.realized_ev_pct, '+.2f', '%')}")
    print(f"Rzeczywisty zysk na 1 bet:        {format_conditional_metric(summary.realized_profit_per_bet, '+.2f', ' zł')}")
    print(f"Błąd kalibracji na 1 bet:         {format_conditional_metric(summary.calibration_gap_pln, '+.2f', ' zł')}")
    print(f"Całkowity obrót:                  {summary.total_staked:,.2f} zł")
    print(f"Zysk całkowity:                   {summary.total_pnl:+,.2f} zł")
    print(f"Maksymalny drawdown:              {summary.max_drawdown_pct:.2f}% ({summary.max_drawdown_pln:,.2f} zł)")
    if summary.median_clv_pct is not None:
        print(f"Mediana CLV:                      {summary.median_clv_pct:+.2f}%")

    print("\n" + "-" * 115)
    print("KALIBRACJA WG KOSZYKÓW EV (OCZEKIWANY VS RZECZYWISTY ZYSK NA 1 BET)")
    print("-" * 115)
    rows_ev = []
    for b in summary.bins:
        rows_ev.append({
            "Koszyk EV": b.label,
            "Bety": b.bets_count,
            "Win Rate": format_conditional_metric(b.win_rate_pct, ".1f", "%"),
            "Śr. Kurs": format_conditional_metric(b.avg_odds, ".2f"),
            "Oczekiwane EV": format_conditional_metric(b.expected_ev_pct, "+.2f", "%"),
            "Oczekiwany zysk/bet": format_conditional_metric(b.expected_profit_per_bet, "+.2f", " zł"),
            "Rzeczywiste EV": format_conditional_metric(b.realized_ev_pct, "+.2f", "%"),
            "Rzeczywisty zysk/bet": format_conditional_metric(b.realized_profit_per_bet, "+.2f", " zł"),
            "Błąd (zł/bet)": format_conditional_metric(b.calibration_gap_pln, "+.2f", " zł"),
            "Realizacja": format_conditional_metric(b.ev_realization_rate_pct, ".1f", "%"),
        })
    if rows_ev:
        print(pd.DataFrame(rows_ev).to_string(index=False))

    print("\n" + "-" * 115)
    print("ROZBICIE WG PRZEDZIAŁÓW KURSOWYCH")
    print("-" * 115)
    rows_odds = []
    for ob in summary.by_odds_bracket:
        rows_odds.append({
            "Przedział kursowy": ob.label,
            "Bety": ob.bets_count,
            "Win Rate": format_conditional_metric(ob.win_rate_pct, ".1f", "%"),
            "Śr. Kurs": format_conditional_metric(ob.avg_odds, ".2f"),
            "Oczekiwane EV": format_conditional_metric(ob.expected_ev_pct, "+.2f", "%"),
            "Oczekiwany zysk/bet": format_conditional_metric(ob.expected_profit_per_bet, "+.2f", " zł"),
            "Rzeczywiste EV": format_conditional_metric(ob.realized_ev_pct, "+.2f", "%"),
            "Rzeczywisty zysk/bet": format_conditional_metric(ob.realized_profit_per_bet, "+.2f", " zł"),
            "Błąd (zł/bet)": format_conditional_metric(ob.calibration_gap_pln, "+.2f", " zł"),
            "Mediana CLV": f"{ob.median_clv_pct:+.2f}%" if ob.median_clv_pct is not None else "N/A",
        })
    if rows_odds:
        print(pd.DataFrame(rows_odds).to_string(index=False))


def _validate_c0_previous_stage(path, *, expected_stage, protocol, synthetic):
    """Load only a complete, clock-valid seal with its exact declared head lineage."""
    import hashlib
    from datetime import datetime, timezone
    from src.analysis.ev_approach_data import C0_PHASES, c0_clock, c0_digest
    from src.models.ev_approach_models import C0MarketHead, C0_RECIPES
    prior = json.loads(Path(path).read_text())
    phase = {"train": "TRAIN", "select": "CAL"}[expected_stage]
    count = 12 if expected_stage == "train" else 7
    fit_count = 12 if expected_stage == "train" else 0
    if (prior["stage"] != expected_stage or prior["phase"] != phase
            or prior["protocol_sha256"] != c0_digest(protocol)
            or prior["code_hashes"] != protocol["code_hashes"]
            or prior["evidence_class"] != ("synthetic" if synthetic else "development")
            or prior["status"] != ("SYNTHETIC_SOFTWARE_ONLY" if synthetic else "COMPLETE")
            or prior["model_count"] != count or prior["attempt_count"] != fit_count
            or prior["failed_fit_count"] != 0
            or prior["synthetic_fit_count"] != (fit_count if synthetic else 0)
            or prior["real_fit_count"] != (0 if synthetic else fit_count)):
        raise ValueError("Prior stage is not a complete matching seal")
    _, _, lock, deadline = C0_PHASES[phase]
    clock, actual, completed = (c0_clock(prior[key]) for key in
                               ("stage_clock", "actual_execution_at", "completed_at"))
    if (clock < c0_clock(lock) or (deadline and clock >= c0_clock(deadline))
            or not actual <= completed <= c0_clock(datetime.now(timezone.utc))
            or (not synthetic and (not actual <= clock <= completed
                                    or (deadline and completed >= c0_clock(deadline))))):
        raise ValueError("Prior stage clock/deadline invalid")
    families = ("S-cal", "M-cal", "M-shape", "H")
    head_names = ({f"candidates/{family}-{value}.json": (family, value)
                   for family in families for value in protocol["lambdas"]}
                  if expected_stage == "train" else
                  {f"selected/{recipe}.json": (recipe, None) for recipe in C0_RECIPES})
    required = {"admission.json", "membership.json", *head_names}
    if expected_stage == "train":
        required.update(f"candidate_predictions/{family}-{value}.json"
                        for family in families for value in protocol["lambdas"])
    else:
        required.add("probabilities.json")
        if len(prior["previous_stage_sha256"]) != 64:
            raise ValueError("Selected stage lacks its TRAIN receipt seal")
    if set(prior["artifacts"]) != required:
        raise ValueError("Prior stage artifact set incomplete or undeclared")
    root = Path(path).parent
    for relative, sha in prior["artifacts"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != sha:
            raise ValueError("Previous artifact seal changed")
    membership = json.loads((root / "membership.json").read_text())
    if membership["membership_sha256"] != prior["membership_sha256"]:
        raise ValueError("Prior opportunity membership changed")
    bindings = prior["training_bindings"]
    if (bindings["protocol_sha256"] != c0_digest(protocol) or bindings["code_hashes"] != protocol["code_hashes"]
            or (expected_stage == "train" and (
                bindings["membership_sha256"] != prior["membership_sha256"]
                or bindings["common_membership_sha256"] != c0_digest(membership["common_row_ids"])
                or bindings["common_count"] != len(membership["common_row_ids"])))):
        raise ValueError("Prior TRAIN membership lineage invalid")
    for relative, (recipe, regularization) in head_names.items():
        head = C0MarketHead.from_dict(json.loads((root / relative).read_text()))
        if (head.recipe_id != recipe or head.bindings != bindings
                or (regularization is not None and head.regularization != regularization)
                or (expected_stage == "select" and recipe in families
                    and head.regularization not in protocol["lambdas"])):
            raise ValueError("Prior head family/lambda/TRAIN lineage mismatch")
    return prior


def run_c0_market_study(argv):
    """Immutable, gated native study. Synthetic stages never authorize real fits."""
    from datetime import datetime, timezone
    import hashlib
    from src.analysis.ev_approach_data import (
        C0_PHASES, c0_canonical_bytes, c0_digest, c0_clock, validate_c0_protocol,
        prepare_c0_market_study_rows, c0_label_available_at,
    )
    from src.models.ev_approach_models import (
        C0_RECIPES, C0MarketHead, fit_c0_market_head, c0_features, select_c0_market_head,
    )
    from src.analysis.ev_approach_evaluation import evaluate_c0_market_heads
    parser = argparse.ArgumentParser(prog="c0-market-study")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--no-fit", action="store_true")
    group.add_argument("--synthetic-smoke", action="store_true")
    group.add_argument("--stage", choices=("train","select","assess"))
    parser.add_argument("--run-id")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-stage-receipt", type=Path)
    args = parser.parse_args(argv)
    actual = datetime.now(timezone.utc).isoformat()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.freeze:
        if not args.run_id:
            parser.error("--freeze requires --run-id")
    elif not args.protocol:
        parser.error("--protocol is required")
    if args.stage and not args.input_dir:
        parser.error("Real stage requires --input-dir")
    if args.stage in {"select","assess"} and not args.previous_stage_receipt:
        parser.error("Stage requires --previous-stage-receipt")
    def write_json(path, value):
        data = c0_canonical_bytes(value) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".c0-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return hashlib.sha256(data).hexdigest()
    code_paths = ("src/models/ev_approach_models.py","src/analysis/ev_approach_data.py",
                  "src/analysis/ev_approach_evaluation.py","scripts/run_unified_betting_evaluation.py",
                  "betting_app/services/c0_shadow_readiness.py","betting_app/services/forecast_evidence.py",
                  "betting_app/services/c0_inference.py","betting_app/services/c0_worker.py",
                  "betting_app/core/models/native_c0.py")
    code_hashes = {p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in code_paths}
    if args.freeze:
        activation_path = PROJECT_ROOT/"data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json"
        serving_path = PROJECT_ROOT/"conf/base/c0_serving.json"
        serving = json.loads(serving_path.read_text())
        plan06_path = PROJECT_ROOT/"data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/protocol.json"
        plan06 = json.loads(plan06_path.read_text())
        source_contracts_path = plan06_path.with_name("source_contracts.json")
        source_contracts = json.loads(source_contracts_path.read_text())
        completion_path = PROJECT_ROOT/serving["completion_report"]
        inventory = {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (serving_path,completion_path)}
        native_inventory = {"serving":serving,"completion":json.loads(completion_path.read_text()),
                            "file_hashes":inventory,"mode":"full"}
        protocol = {"schema":"c0_market_study_v1","protocol_id":"native-c0-market-study-v1",
            "run_id":args.run_id,"native_identity":"Causal-C0:c0-native-2026-w32-e12-v1","mode":"full",
            "calendar":{k:list(v) for k,v in C0_PHASES.items()},"epsilon":1e-6,"lambdas":[.01,.1,1.],
            "recipes":list(C0_RECIPES),"objective":"mean stable Bernoulli NLL + lambda/2 squared centered penalty",
            "activation_deadline":"2026-10-31T00:00:00Z","primary_origin":"T-1h","max_quote_age_seconds":300,
            "max_completion_seconds":60,"tax":.12,"minimum_net_ev_exclusive":.05,
            "native_inventory_hashes":inventory,"native_serving_inventory":serving,
            "native_completion_inventory":json.loads(completion_path.read_text()),"code_hashes":code_hashes,
            "native_inventory":native_inventory,"native_inventory_sha256":c0_digest(native_inventory),
            "plan06_sha256":hashlib.sha256(plan06_path.read_bytes()).hexdigest(),
            "plan06_contract":plan06,"population":plan06["population"],"horizon":plan06["horizon"],
            "activation_matrix_sha256":hashlib.sha256(activation_path.read_bytes()).hexdigest(),
            "activation_matrix":json.loads(activation_path.read_text()),"frozen_at":actual,
            "trusted_evidence_refs":[],"source_contracts":source_contracts["sources"],
            "source_contracts_sha256":hashlib.sha256(source_contracts_path.read_bytes()).hexdigest()}
        # Only a separately reviewed local activation record may supply trust
        # roots. A CLI caller cannot turn arbitrary evidence into trusted pins.
        activation = protocol["activation_matrix"].get("native_c0_activation")
        if activation is not None:
            from betting_app.services.c0_shadow_readiness import evaluate_c0_readiness
            if not isinstance(activation,dict) or activation.get("schema") != "native_c0_activation_record_v1":
                raise ValueError("Reviewed native activation record schema invalid")
            approved_contracts = activation["source_contracts"]
            original_ids = {s["source_id"] for s in source_contracts["sources"]}
            if {s["source_id"] for s in approved_contracts} != original_ids or len(approved_contracts) != 8:
                raise ValueError("Activation changed original source universe")
            protocol.update(source_contracts=approved_contracts,
                original_source_contracts=source_contracts,
                trusted_evidence_refs=activation["trusted_evidence_refs"],
                evidence_root=activation["evidence_root"],
                authentication_receipts=activation["authentication_receipts"],
                captures=activation["captures"])
            authority_path = Path(activation["authorization_path"])
            if not authority_path.is_absolute():
                authority_path = PROJECT_ROOT/authority_path
            authorization = json.loads(authority_path.read_text())
            readiness = evaluate_c0_readiness(protocol,now=datetime.now(timezone.utc),authorization=authorization)
            if readiness["status"] != "READY_AUTHORIZED":
                raise ValueError("Reviewed activation evidence does not pass: "+", ".join(readiness["reasons"]))
            protocol["activation_authorization_sha256"] = c0_digest(authorization)
            protocol["activation_evidence_sha256"] = c0_digest(activation)
            protocol["activation_readiness"] = readiness
        write_json(args.output_dir/"protocol.json",protocol)
        print(json.dumps({"status":"FROZEN","protocol_sha256":c0_digest(protocol),"real_fit_count":0}))
        return
    protocol = json.loads(args.protocol.read_text())
    validate_c0_protocol(protocol)
    if args.run_id is not None and args.run_id != protocol.get("run_id"):
        raise ValueError("Run ID conflicts with frozen protocol")
    if protocol["code_hashes"] != code_hashes:
        raise ValueError("Frozen implementation hashes changed")
    if args.no_fit:
        from betting_app.services.c0_shadow_readiness import evaluate_c0_readiness
        authorization = None
        activation = protocol["activation_matrix"].get("native_c0_activation")
        if activation and protocol.get("activation_authorization_sha256"):
            path = Path(activation["authorization_path"])
            if not path.is_absolute():
                path = PROJECT_ROOT/path
            authorization = json.loads(path.read_text())
            if c0_digest(authorization) != protocol["activation_authorization_sha256"]:
                raise ValueError("Retained activation authority changed")
        readiness = evaluate_c0_readiness(protocol,now=datetime.now(timezone.utc),authorization=authorization)
        blockers = list(readiness["reasons"]) + ["phase_bundle_not_supplied","phase_support_not_verified"]
        if c0_clock(actual) < c0_clock(C0_PHASES["TRAIN"][2]):
            blockers.append("future_phase_locks_absent")
        receipt = {"schema":"c0_stage_receipt_v1","stage":"no-fit",
            "status":"G0_BLOCKED" if readiness["status"] != "READY_AUTHORIZED" else "NO_FIT",
            "evidence_class":"blocked","real_fit_count":0,"synthetic_fit_count":0,"model_count":0,
            "protocol_sha256":c0_digest(protocol),"actual_execution_at":actual,"readiness":readiness,
            "blockers":sorted(set(blockers))}
        write_json(args.output_dir/"receipt.json",receipt)
        print(json.dumps(receipt))
        return
    families = ("S-cal","M-cal","M-shape","H")
    def support(frame, *, rank_required=True):
        common = frame.loc[frame["native_admitted"] & frame["p_C0"].notna() & frame["q"].notna() & frame["y"].notna()].copy()
        if common.empty or common["y"].nunique() != 2:
            raise ValueError("INSUFFICIENT_FIT_SUPPORT")
        if rank_required:
            for recipe in families:
                x = c0_features(common,recipe)
                if np.linalg.matrix_rank(x) != x.shape[1]:
                    raise ValueError("INSUFFICIENT_FIT_SUPPORT")
        common.attrs = frame.attrs.copy()
        return common
    def stage(stage_name, frame, admission, destination, now, previous=None, synthetic=False):
        phase = {"train":"TRAIN","select":"CAL","assess":"DEVELOPMENT"}[stage_name]
        _, _, lock, deadline = C0_PHASES[phase]
        receipt = {"schema":"c0_stage_receipt_v1","stage":stage_name,"phase":phase,
            "protocol_sha256":c0_digest(protocol),"code_hashes":code_hashes,
            "membership_sha256":admission["membership_sha256"],"admission":admission,
            "actual_execution_at":actual,"stage_clock":now.isoformat(),
            "evidence_class":"synthetic" if synthetic else "development",
            "real_fit_count":0,"synthetic_fit_count":0,"attempt_count":0,"failed_fit_count":0,"model_count":0,
            "status":"SYNTHETIC_SOFTWARE_ONLY" if synthetic else "COMPLETE","artifacts":{}}
        clock_failure = ("PHASE_LABEL_LOCK_NOT_REACHED" if now < c0_clock(lock) else
                         "MISSED_ARTIFACT_DEADLINE" if deadline and now >= c0_clock(deadline) else None)
        if clock_failure:
            receipt.update(status=clock_failure,evidence_class="synthetic" if synthetic else "blocked")
            write_json(destination/"receipt.json",receipt)
            return receipt
        if not synthetic and admission["status"] != "PASS":
            receipt.update(status="G0_BLOCKED",evidence_class="blocked")
            write_json(destination/"receipt.json",receipt)
            return receipt
        try:
            common = support(frame, rank_required=stage_name == "train") if stage_name != "assess" else frame.loc[
                frame["native_admitted"] & frame["p_C0"].notna() & frame["q"].notna() & frame["y"].notna()].copy()
        except ValueError as error:
            receipt.update(status="INSUFFICIENT_FIT_SUPPORT",failure=str(error))
            write_json(destination/"receipt.json",receipt)
            return receipt
        common.attrs["c0_bindings"] = {**frame.attrs.get("c0_bindings",{}),
            "common_membership_sha256":c0_digest(common["row_id"].tolist()),"common_count":len(common)}
        receipt["artifacts"]["admission.json"] = write_json(destination/"admission.json",admission)
        receipt["artifacts"]["membership.json"] = write_json(destination/"membership.json",{
            "row_ids":frame["row_id"].tolist(),"common_row_ids":common["row_id"].tolist(),
            "membership_sha256":admission["membership_sha256"]})
        if previous:
            previous_path = Path(previous)
            expected = "train" if stage_name == "select" else "select"
            prior = _validate_c0_previous_stage(previous_path, expected_stage=expected,
                                               protocol=protocol, synthetic=synthetic)
            receipt["training_bindings"] = prior["training_bindings"]
            receipt["previous_stage_sha256"] = hashlib.sha256(previous_path.read_bytes()).hexdigest()
        if stage_name == "train":
            receipt["training_bindings"] = dict(common.attrs["c0_bindings"])
            if not synthetic:
                write_json(args.protocol.parent/"training_consumed.json",
                           {"protocol_sha256":c0_digest(protocol),"actual_execution_at":actual,
                            "destination":str(destination)})
            for family in families:
                for regularization in protocol["lambdas"]:
                    if not synthetic and c0_clock(datetime.now(timezone.utc)) >= c0_clock(deadline):
                        receipt.update(status="MISSED_ARTIFACT_DEADLINE",completed_at=datetime.now(timezone.utc).isoformat())
                        write_json(destination/"receipt.json",receipt)
                        return receipt
                    receipt["attempt_count"] += 1
                    count = "synthetic_fit_count" if synthetic else "real_fit_count"
                    receipt[count] += 1
                    try:
                        head = fit_c0_market_head(family,common,regularization=regularization)
                    except (ValueError,RuntimeError) as error:
                        receipt.update(status="FIT_FAILED",failed_fit_count=1,failure=str(error))
                        write_json(destination/"receipt.json",receipt)
                        return receipt
                    name = f"candidates/{family}-{regularization}.json"
                    receipt["artifacts"][name] = write_json(destination/name,head.to_dict())
                    receipt["model_count"] += 1
                    details = head.predict_details(common)
                    rows = [{"row_id":row_id,"raw_logit":float(details["logit"][i]),
                        "raw_probability":float(details["raw_probability"][i]),
                        "probability":float(details["probability"][i]),
                        "clipped":bool(details["clipped"][i]),"input_clipped":bool(details["input_clipped"][i])}
                        for i,row_id in enumerate(common["row_id"])]
                    name = f"candidate_predictions/{family}-{regularization}.json"
                    receipt["artifacts"][name] = write_json(destination/name,rows)
        else:
            heads = {}
            if stage_name == "select":
                for family in families:
                    candidates = []
                    for regularization in protocol["lambdas"]:
                        head = C0MarketHead.from_dict(json.loads((previous_path.parent/f"candidates/{family}-{regularization}.json").read_text()))
                        candidates.append(head)
                    chosen, choice = select_c0_market_head(candidates, common)
                    heads[family] = chosen
                    receipt.setdefault("cal_choices",{})[family] = choice
                common.attrs["c0_bindings"] = dict(prior["training_bindings"])
                for recipe in ("S","M","Fixed50"):
                    heads[recipe] = fit_c0_market_head(recipe,common,regularization=0)
                for recipe,head in heads.items():
                    name = f"selected/{recipe}.json"
                    receipt["artifacts"][name] = write_json(destination/name,head.to_dict())
                    receipt["model_count"] += 1
            else:
                # One assessment consumes the selected seal, even across output paths.
                if not synthetic:
                    write_json(args.protocol.parent/"assessment_consumed.json",
                               {"protocol_sha256":c0_digest(protocol),"actual_execution_at":actual,
                                "destination":str(destination)})
                consume = previous_path.parent/"assessment_consumed.json"
                write_json(consume,{"actual_execution_at":actual,"destination":str(destination)})
                heads = {r:C0MarketHead.from_dict(json.loads((previous_path.parent/f"selected/{r}.json").read_text())) for r in C0_RECIPES}
                report = evaluate_c0_market_heads(frame,heads)
                receipt["decision"] = "SYNTHETIC_SOFTWARE_ONLY" if synthetic else report["decision"]
                report["evidence_class"] = receipt["evidence_class"]
                if synthetic:
                    report["decision"] = "SYNTHETIC_SOFTWARE_ONLY"
                receipt["artifacts"]["assessment.json"] = write_json(destination/"assessment.json",report)
                receipt["model_count"] = len(heads)
            predictions = []
            for recipe,head in heads.items():
                for _,row in frame.iterrows():
                    try:
                        if not row.get("primary_decision",True):
                            raise ValueError("non_primary_event")
                        if recipe in {"S","S-cal","Fixed50","H"} and not row["native_admitted"]:
                            raise ValueError("native_not_admitted")
                        details = head.predict_details(pd.DataFrame([row]))
                        p = float(details["probability"][0])
                        predictions.append({"row_id":row["row_id"],"recipe":recipe,"probability":p,
                            "probability_b":1-p,"raw_probability":float(details["raw_probability"][0]),
                            "raw_logit":float(details["logit"][0]),"clipped":bool(details["clipped"][0]),
                            "input_clipped":bool(details["input_clipped"][0]),"absence":None})
                    except (ValueError,KeyError,TypeError) as error:
                        predictions.append({"row_id":row["row_id"],"recipe":recipe,"probability":None,"absence":str(error)})
            receipt["artifacts"]["probabilities.json"] = write_json(destination/"probabilities.json",predictions)
        receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
        if not synthetic and deadline and c0_clock(receipt["completed_at"]) >= c0_clock(deadline):
            receipt["status"] = "MISSED_ARTIFACT_DEADLINE"
        write_json(destination/"receipt.json",receipt)
        return receipt
    if args.synthetic_smoke:
        rng = np.random.default_rng(20260929)
        from scipy.special import expit
        previous = None
        receipts = []
        for phase, stage_name, clock in (("TRAIN","train","2027-11-16T00:00:00Z"),
                                        ("CAL","select","2028-06-16T00:00:00Z"),
                                        ("DEVELOPMENT","assess","2029-07-16T00:00:00Z")):
            z = rng.normal(0,2,200)
            m = rng.normal(0,2,200)
            p, q = expit(z), expit(m)
            dates = pd.date_range(C0_PHASES[phase][0],C0_PHASES[phase][1],periods=201)[:-1]
            frame = pd.DataFrame({"row_id":[f"{phase}-{i}" for i in range(200)],
                "p_C0":p,"q":q,"y":rng.binomial(1,expit(.2*z+.8*m)),
                "decision_at":dates,"month":[d.strftime("%Y-%m") for d in dates],
                "native_admitted":True,"financial_eligible":True})
            frame["quotes"] = [[{"bookmaker":"SyntheticA","odds_snapshot_id":i*2,
                "odds_a":1/(q[i]*1.04),"odds_b":1/((1-q[i])*1.04)},
                {"bookmaker":"SyntheticB","odds_snapshot_id":i*2+1,
                 "odds_a":1/(q[i]*1.05),"odds_b":1/((1-q[i])*1.05)}] for i in range(200)]
            # Irrelevant missing native input must not disable market controls.
            frame.loc[0,"p_C0"] = np.nan
            frame.loc[1,"q"] = np.nan
            frame.at[1,"quotes"] = []
            frame.attrs["c0_bindings"] = {"protocol_sha256":c0_digest(protocol),
                "membership_sha256":c0_digest(frame["row_id"].tolist()),"code_hashes":code_hashes,
                "evidence_class":"synthetic","source_class":"local_fixture"}
            admission = {"status":"SYNTHETIC_SOFTWARE_ONLY","membership_sha256":c0_digest(frame["row_id"].tolist()),
                         "phase":phase,"opportunity_count":200,"evidence_class":"synthetic"}
            destination = args.output_dir/stage_name
            receipts.append(stage(stage_name,frame,admission,destination,c0_clock(clock),previous,True))
            previous = destination/"receipt.json"
        # The same support and reducers exercise failure/undefined cases without
        # adding fits or granting synthetic authority.
        cases = {}
        try:
            support(frame.assign(q=.5))
        except ValueError as error:
            cases["rank_deficiency"] = str(error)
        try:
            validate_c0_protocol({**protocol,"mode":"no_history"})
        except ValueError as error:
            cases["wrong_mode"] = str(error)
        try:
            c0_label_available_at({"available_at":"2029-07-16T00:00:00Z"},
                {"response_received_at":"2029-07-16T00:00:00Z","recorded_at":"2029-07-16T00:00:00Z"},
                phase="DEVELOPMENT")
        except ValueError as error:
            cases["late_release"] = str(error)
        selected_heads = {r:C0MarketHead.from_dict(json.loads((args.output_dir/"select"/f"selected/{r}.json").read_text()))
                          for r in C0_RECIPES}
        no_quotes = frame.copy()
        no_quotes["quotes"] = [[] for _ in range(len(frame))]
        cases["zero_selection"] = evaluate_c0_market_heads(no_quotes,selected_heads)["metrics"]["H"]["ROI"]
        cases["one_contributing_month"] = evaluate_c0_market_heads(frame.assign(month="2028-07"),selected_heads)["contrasts"]
        write_json(args.output_dir/"synthetic_cases.json",cases)
        result = {"status":"SYNTHETIC_SOFTWARE_ONLY","real_fit_count":0,
                  "synthetic_fit_count":sum(r["synthetic_fit_count"] for r in receipts),
                  "recipe_count":7,"model_count":7,"real_model_count":0,
                  "candidate_model_count":12,"row_count":600,"actual_execution_at":actual,
                  "stage_receipts":receipts,"failure_cases":cases}
        write_json(args.output_dir/"receipt.json",result)
        print(json.dumps({"status":result["status"],"real_fit_count":0,"synthetic_fit_count":result["synthetic_fit_count"]}))
    else:
        phase = {"train":"TRAIN","select":"CAL","assess":"DEVELOPMENT"}[args.stage]
        panel, admission = prepare_c0_market_study_rows(args.input_dir,protocol,phase=phase)
        receipt = stage(args.stage,panel,admission,args.output_dir,c0_clock(datetime.now(timezone.utc)),
                        args.previous_stage_receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
