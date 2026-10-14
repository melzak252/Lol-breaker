"""CLI script to verify and synchronize team rosters against LoL Fandom and Liquipedia."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import sys
import time

from betting_app.services.liquipedia_service import LiquipediaClient
from betting_app.services.roster_verification_service import LiquipediaRosterAdapter, RosterVerificationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("roster_verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and sync active team rosters from LoL Fandom (Leaguepedia) and Liquipedia."
    )
    parser.add_argument(
        "--team",
        action="append",
        help="Specific team name(s) to verify (can be repeated or comma-separated)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "fandom", "liquipedia"],
        default="auto",
        help="Data source to use (default: auto -> fandom with liquipedia fallback)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check and print discrepancies without saving to the database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite even if current stored roster timestamp is recent",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum upcoming teams to check (default: 50)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()

    teams: list[str] = []
    if args.team:
        for t in args.team:
            for sub in t.split(","):
                clean = sub.strip()
                if clean and clean not in teams:
                    teams.append(clean)

    service = RosterVerificationService(
        liquipedia_client=LiquipediaRosterAdapter(
            client=LiquipediaClient(wait_for_spacing=True),
        ),
    )

    logger.info(
        "Starting team roster verification (source=%s, dry_run=%s, force=%s, specified_teams=%d)",
        args.source,
        args.dry_run,
        args.force,
        len(teams),
    )

    try:
        summary = service.verify_and_sync_rosters(
            team_names=teams or None,
            source=args.source,
            force=args.force,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    except Exception as e:
        logger.error("Roster verification failed: %s", e, exc_info=True)
        return 1

    duration = time.time() - start_time
    summary["duration_seconds"] = round(duration, 2)
    summary["executed_at"] = datetime.now(UTC).isoformat()

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("\n" + "=" * 70)
    print(f" TEAM ROSTER VERIFICATION SUMMARY ({summary['executed_at']})")
    print("=" * 70)
    print(f" Source:      {summary['source']}")
    print(f" Mode:        {'DRY RUN (no DB changes)' if summary['dry_run'] else 'APPLY CHANGES'}")
    print(f" Total teams: {summary['total_teams']}")
    print(f" Up to date:  {summary['up_to_date_count']}")
    print(f" Updated:     {summary['updated_count']}")
    print(f" Failed:      {summary['failed_count']}")
    print(f" Duration:    {summary['duration_seconds']}s")
    print("-" * 70)

    if summary["updated"]:
        print("\nROSTER UPDATES / DISCREPANCIES:")
        for u in summary["updated"]:
            print(f"  • {u['team_name']} [{u['source']}]:")
            for ch in u.get("changes", []):
                print(f"      - {ch['role']}: {ch['old_player']} -> {ch['new_player']} ({ch['change_type']})")
            if not u.get("changes"):
                players_str = ", ".join(f"{p['role']}:{p['player_id']}" for p in u.get("players", []))
                print(f"      Initial 5-man roster populated: [{players_str}]")

    if summary["failed"]:
        print("\nFAILED TEAMS (could not resolve 5-man starting lineup):")
        for f in summary["failed"]:
            print(f"  • {f['team_name']}: {f.get('reason', 'unknown')}")

    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
