"""Run the end-to-end daily automation workflow for upcoming predictions.

Default mode refreshes odds, rematches canonical events, rebuilds features and
predictions.  GOL.GG/rating rebuilds are optional because they are heavier and
usually needed only after new finished matches are imported.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_BOOKMAKERS = ("sts", "betclic", "superbet", "efortuna", "betfan", "totalbet", "lebull")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bookmaker", action="append", choices=DEFAULT_BOOKMAKERS, help="Bookmaker to scrape; repeatable.")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--refresh-golgg", action="store_true", help="Fetch missing finished GOL.GG matches first.")
    parser.add_argument("--reimport-golgg", action="store_true", help="Import GOL.GG JSON into SQLite.")
    parser.add_argument(
        "--rebuild-ratings",
        action="store_true",
        help="Fully rebuild the regional ratings-v2 snapshot from SQLite GOL.GG.",
    )
    parser.add_argument("--rebuild-w20", action="store_true", help="Rebuild w20-latest rolling features.")
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument(
        "--operational-hybrid",
        action="store_true",
        help="Use ratings-v2 operational+market hybrid probabilities for EV signals.",
    )
    args = parser.parse_args()

    if args.refresh_golgg:
        run_module("betting_app.scripts.refresh_golgg_results")
    if args.reimport_golgg:
        run_module("betting_app.scripts.import_golgg_to_db")
    if args.rebuild_ratings:
        run_module(
            "betting_app.scripts.rebuild_regional_ratings",
            "--ratings-version",
            "ratings-v2",
            "--source",
            "manual-regional-ratings-v2",
        )
    if args.rebuild_w20:
        run_module("betting_app.scripts.rebuild_w20_features", "--feature-version", "w20-latest", "--window-size", "20")

    if not args.skip_scrape:
        for bookmaker in args.bookmaker or DEFAULT_BOOKMAKERS:
            command = ["betting_app.scripts.scrape_odds", "--bookmaker", bookmaker]
            if bookmaker in {"betclic", "superbet", "efortuna", "betfan"}:
                command.append("--headless")
            run_module(*command)

    run_module("betting_app.scripts.rematch_canonical_matches", "--rebuild")
    pipeline_args = ["--min-ev", str(args.min_ev)]
    if args.operational_hybrid:
        pipeline_args.append("--operational-hybrid")
    run_module("betting_app.scripts.run_upcoming_prediction_pipeline", *pipeline_args)
    run_module("betting_app.scripts.list_upcoming_model_predictions", "--positive-only")


def run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
