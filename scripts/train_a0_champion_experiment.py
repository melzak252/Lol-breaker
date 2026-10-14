#!/usr/bin/env python3
"""Unified CLI runner for champion-pool augmented experiment (IDEA-031)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.a0_champion_runner import freeze_run, run_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-run", metavar="RUN_ID", help="Freeze protocol for experiment run ID")
    group.add_argument("--protocol", metavar="PATH", type=Path, help="Path to frozen protocol.json")
    parser.add_argument("--smoke", action="store_true", help="Run reduced diagnostic smoke validation")
    parser.add_argument("--research-root", type=Path, help="Explicit research root path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze_run:
        if args.smoke:
            raise SystemExit("--smoke requires --protocol")
        proto_path = freeze_run(args.freeze_run, research_root=args.research_root)
        print(f"Frozen protocol saved: {proto_path}")
    else:
        results = run_protocol(args.protocol, smoke=args.smoke, research_root=args.research_root)
        decision = results.get("decision", {})
        status = decision.get("status", "UNKNOWN")
        print(json.dumps({"status": status, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
