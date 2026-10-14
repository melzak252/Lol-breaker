#!/usr/bin/env python3
"""Report local tournament recipe coverage; never fetch rules or access a DB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tournament_catalog import DEFAULT_BUNDLE, audit_catalog, instantiate_profile, list_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, help="Save complete JSON report, including all phase/source/readiness rows")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of summary")
    parser.add_argument("--list-profiles", action="store_true", help="Print profile IDs, slots, scope and assumptions as JSON")
    parser.add_argument("--profile", help="Instantiate one profile as engine JSON; requires --teams")
    parser.add_argument("--teams", nargs="+", help="Ordered exact team IDs; quote names containing spaces")
    args = parser.parse_args()
    if args.teams and not args.profile:
        parser.error("--teams requires --profile")
    if args.profile and not args.teams:
        parser.error("--profile requires --teams")
    if args.profile and args.list_profiles:
        parser.error("--profile and --list-profiles are mutually exclusive")
    try:
        if args.profile:
            result = instantiate_profile(args.profile, args.teams)
        elif args.list_profiles:
            result = list_profiles()
        else:
            result = audit_catalog(args.bundle)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json or args.profile or args.list_profiles:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f"Offline tournament format audit: {result['phase_count']} phases, {result['profile_count']} profiles")
    print("Family       phases configured unconfigured historical-ready historical-unresolved")
    for family, row in result["families"].items():
        print(f"{family:12} {row['phase_count']:6} {row['configured_phase_count']:10} "
              f"{row['unconfigured_phase_count']:12} {row['historically_ready_count']:16} "
              f"{row['historically_unresolved_count']:21}")
    print(result["coverage_note"])
    if result["catalogued_phases_missing_from_bundle"]:
        print(f"Missing catalogued phase IDs: {result['catalogued_phases_missing_from_bundle']}")
    if args.output:
        print(f"Complete phase/source/readiness report: {args.output}")


if __name__ == "__main__":
    main()
