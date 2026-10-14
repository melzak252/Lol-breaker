"""Script for collecting, ingesting, and evaluating LoL proposition odds (IDEA-018).

Usage:
  python -m betting_app.scripts.collect_prop_odds --demo
  python -m betting_app.scripts.collect_prop_odds --match-id 101 --evaluate-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, UTC
from sqlalchemy import text

from betting_app.core.db import get_session
from betting_app.models.bookmaker import Bookmaker
from betting_app.models.match import CanonicalMatch
from betting_app.scrapers.props_parser import ParsedMatchProps, ParsedPropLine
from betting_app.services.prop_odds_service import (
    evaluate_match_props,
    get_latest_prop_odds,
    get_prop_odds_timeline,
    save_prop_snapshots,
)
from betting_app.services.prop_scraper_service import (
    PropScraperService,
    SCRAPERS_REGISTRY,
    run_all_prop_scrapers,
)
from betting_app.api.routers.props import get_cached_pace_tracker


def run_demo_ingestion(canonical_match_id: int | None = None) -> list[int]:
    """Ingest realistic proposition market odds for active upcoming matches to test the pipeline safely."""
    sess = get_session()
    try:
        # Check or retrieve upcoming matches
        query = text("""
            SELECT id, team_a_name, team_b_name, league, start_time_normalized
            FROM canonical_matches
            WHERE status = 'upcoming'
            ORDER BY id DESC
            LIMIT 5
        """)
        matches = sess.execute(query).fetchall()

        if not matches:
            print("No upcoming matches found in canonical_matches. Using default match ID.")
            target_matches = [(canonical_match_id or 1, "Karmine Corp", "G2 Esports", "LEC", "2026-09-05T18:00:00Z")]
        else:
            target_matches = [
                (row.id, row.team_a_name, row.team_b_name, row.league, row.start_time_normalized)
                for row in matches
            ]

        if canonical_match_id:
            target_matches = [m for m in target_matches if m[0] == canonical_match_id] or target_matches[:1]

        processed_ids = []
        for m_id, team_a, team_b, league, start_time in target_matches:
            print(f"\n--- Ingesting Demo Prop Odds for [{m_id}] {team_a} vs {team_b} ({league}) ---")

            # Bookmaker 1: Fortuna / efortuna
            p_fortuna = ParsedMatchProps(
                bookmaker="efortuna",
                raw_team_a=team_a,
                raw_team_b=team_b,
                map_number=1,
                lines=[
                    ParsedPropLine(market_type="total_kills", line=26.5, odds_over=1.85, odds_under=1.85, raw_market_name="Liczba zabójstw na mapie 1 (26.5)"),
                    ParsedPropLine(market_type="total_kills", line=28.5, odds_over=2.25, odds_under=1.55, raw_market_name="Liczba zabójstw na mapie 1 (28.5)"),
                    ParsedPropLine(market_type="team_kills", team=team_a, line=13.5, odds_over=1.82, odds_under=1.88, raw_market_name=f"{team_a} liczba zabójstw (13.5)"),
                    ParsedPropLine(market_type="team_kills", team=team_b, line=13.5, odds_over=1.80, odds_under=1.90, raw_market_name=f"{team_b} liczba zabójstw (13.5)"),
                    ParsedPropLine(market_type="handicap_kills", line=-2.5, team=team_a, odds_cover_a=1.95, odds_cover_b=1.75, raw_market_name=f"Handicap zabójstw {team_a} (-2.5)"),
                ],
            )
            inserted_f = save_prop_snapshots(p_fortuna, canonical_match_id=m_id, league=league, match_start_time=start_time, db_session=sess)

            # Bookmaker 2: STS
            p_sts = ParsedMatchProps(
                bookmaker="sts",
                raw_team_a=team_a,
                raw_team_b=team_b,
                map_number=1,
                lines=[
                    ParsedPropLine(market_type="total_kills", line=26.5, odds_over=1.88, odds_under=1.82, raw_market_name="Total Kills Map 1 (26.5)"),
                    ParsedPropLine(market_type="total_kills", line=24.5, odds_over=1.52, odds_under=2.35, raw_market_name="Total Kills Map 1 (24.5)"),
                    ParsedPropLine(market_type="team_kills", team=team_a, line=13.5, odds_over=1.85, odds_under=1.85, raw_market_name=f"{team_a} Kills Map 1 (13.5)"),
                    ParsedPropLine(market_type="handicap_kills", line=-2.5, team=team_a, odds_cover_a=1.90, odds_cover_b=1.80, raw_market_name=f"Kills Handicap {team_a} (-2.5)"),
                ],
            )
            inserted_s = save_prop_snapshots(p_sts, canonical_match_id=m_id, league=league, match_start_time=start_time, db_session=sess)

            # Bookmaker 3: Betclic
            p_betclic = ParsedMatchProps(
                bookmaker="betclic",
                raw_team_a=team_a,
                raw_team_b=team_b,
                map_number=1,
                lines=[
                    ParsedPropLine(market_type="total_kills", line=26.5, odds_over=1.82, odds_under=1.88, raw_market_name="Zabójstwa Łącznie Mapa 1"),
                    ParsedPropLine(market_type="handicap_kills", line=-2.5, team=team_a, odds_cover_a=2.00, odds_cover_b=1.70, raw_market_name="Handicap Zabójstw Mapa 1"),
                ],
            )
            inserted_b = save_prop_snapshots(p_betclic, canonical_match_id=m_id, league=league, match_start_time=start_time, db_session=sess)

            total_inserted = len(inserted_f) + len(inserted_s) + len(inserted_b)
            print(f"  Successfully inserted {total_inserted} prop snapshots across efortuna, sts, betclic.")
            processed_ids.append(m_id)

        sess.commit()
        return processed_ids
    finally:
        sess.close()


def print_match_analysis(canonical_match_id: int, tax_rate: float = 0.12) -> None:
    """Evaluate and display model comparison for match proposition lines."""
    sess = get_session()
    try:
        tracker = get_cached_pace_tracker()
        analysis = evaluate_match_props(canonical_match_id, tracker=tracker, tax_rate=tax_rate, db_session=sess)

        exp = analysis["model_expectations"]
        print("=" * 80)
        print(f"PROPOSITION ODDS & STATISTICAL MODEL ANALYSIS: Match [{canonical_match_id}]")
        print(f"Teams: {analysis['team_a']} vs {analysis['team_b']} ({analysis['league']}) | Map {analysis['map_number']}")
        print(f"Model Expectations: Total Kills: {exp['expected_total_kills']:.1f} | {analysis['team_a']}: {exp['mu_team_a']:.1f} | {analysis['team_b']}: {exp['mu_team_b']:.1f}")
        print(f"Expected Spread: {exp['expected_spread_a_minus_b']:+.1f} kills | League Avg: {exp['league_avg_kills']:.1f} ({exp['league_pace_category']})")
        print("=" * 80)

        lines = analysis["evaluated_lines"]
        print(f"\nActive Bookmaker Prop Lines ({len(lines)} detected):")
        print(f"{'Bookmaker':<10} {'Market':<15} {'Line':<6} {'Target':<14} {'Over/A':<7} {'Under/B':<7} {'Model P(O)':<11} {'Net EV(O)':<10} {'Net EV(U)':<10}")
        print("-" * 95)
        for l in lines:
            tgt = l["target_team"] or "-"
            m_po = f"{l['model_prob_over']*100:.1f}%" if l["model_prob_over"] else "-"
            ev_o = f"{l['ev_over_net']*100:+.1f}%" if l["ev_over_net"] is not None else "-"
            ev_u = f"{l['ev_under_net']*100:+.1f}%" if l["ev_under_net"] is not None else "-"
            print(f"{l['bookmaker']:<10} {l['market_type']:<15} {l['line']:<6.1f} {tgt:<14} {l['odds_over'] or '-':<7} {l['odds_under'] or '-':<7} {m_po:<11} {ev_o:<10} {ev_u:<10}")

        signals = analysis["signals"]
        print("\nVALUE SIGNALS (Positive Net EV after 12% Polish Betting Tax):")
        if not signals:
            print("  No positive EV signals detected under current bookmaker pricing.")
        else:
            for i, sig in enumerate(signals, 1):
                print(f"  [{i}] {sig['bookmaker'].upper()} | {sig['market_type']} | Pick: {sig['selection']} @ {sig['odds']:.2f}")
                print(f"      Model Prob: {sig['model_prob']*100:.1f}% (Fair Odds: {sig['fair_odds']:.2f}) -> NET EV: {sig['net_ev_tax12']:+.1f}%")
        print("=" * 80)
    finally:
        sess.close()


def main():
    parser = argparse.ArgumentParser(description="Collect, ingest, and analyze proposition odds across bookmakers.")
    parser.add_argument("--match-id", type=int, help="Specific canonical match ID")
    parser.add_argument("--demo", action="store_true", help="Ingest realistic demo prop odds for upcoming matches")
    parser.add_argument("--scrape", choices=list(SCRAPERS_REGISTRY.keys()), help="Scrape props from a specific bookmaker (sts, efortuna, betclic, superbet)")
    parser.add_argument("--scrape-all", action="store_true", help="Scrape props from all supported Polish bookmakers")
    parser.add_argument("--max-matches", type=int, default=10, help="Maximum matches to scrape per bookmaker (default: 10)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing prop odds without new ingestion")
    parser.add_argument("--tax-rate", type=float, default=0.12, help="Betting tax rate (default: 0.12)")
    args = parser.parse_args()

    if args.scrape_all:
        print("=== Running Prop Odds Scraping Across All Supported Polish Bookmakers ===")
        results = asyncio.run(run_all_prop_scrapers(max_matches_per_bookmaker=args.max_matches, headless=args.headless))
        print("Scraping results:", results)
    elif args.scrape:
        print(f"=== Running Prop Odds Scraping for {args.scrape.upper()} ===")
        results = asyncio.run(run_all_prop_scrapers(bookmakers=[args.scrape], max_matches_per_bookmaker=args.max_matches, headless=args.headless))
        print("Scraping results:", results)
    elif args.demo:
        match_ids = run_demo_ingestion(canonical_match_id=args.match_id)
        for m_id in match_ids:
            print_match_analysis(m_id, tax_rate=args.tax_rate)
    elif args.match_id:
        print_match_analysis(args.match_id, tax_rate=args.tax_rate)
    else:
        sess = get_session()
        try:
            m = sess.execute(text("SELECT id FROM canonical_matches WHERE status = 'upcoming' LIMIT 1")).fetchone()
            if m:
                print_match_analysis(m.id, tax_rate=args.tax_rate)
            else:
                print("No upcoming matches found. Run with --demo to test proposition ingestion.")
        finally:
            sess.close()


if __name__ == "__main__":
    main()
