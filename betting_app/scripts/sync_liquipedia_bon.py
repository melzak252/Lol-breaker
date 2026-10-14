"""Script to sync upcoming matches, Best-of formats, and team rosters from Liquipedia."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from betting_app.services.liquipedia_service import sync_liquipedia_best_of, sync_liquipedia_team_rosters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize Best-of series format and team rosters from Liquipedia.")
    parser.add_argument("--limit", type=int, default=50, help="Number of match ticker entries to fetch (default: 50)")
    parser.add_argument("--sync-rosters", action="store_true", help="Also sync active rosters for upcoming teams")
    parser.add_argument("--team", action="append", help="Specific team name(s) to sync rosters for")
    args = parser.parse_args()

    bon_result = sync_liquipedia_best_of(limit=args.limit)
    print("BoN sync result:", json.dumps(bon_result, indent=2))

    if args.sync_rosters or args.team:
        roster_result = sync_liquipedia_team_rosters(team_names=args.team, wait_for_spacing=True)
        print("Roster sync result:", json.dumps(roster_result, indent=2))


if __name__ == "__main__":
    main()
