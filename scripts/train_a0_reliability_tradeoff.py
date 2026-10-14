#!/usr/bin/env python3
"""Freeze or run the candidate-only native6 reliability-duration study."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.a0_reliability_tradeoff import freeze_protocol, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--freeze-run", help="Freeze the authorized reliability-tradeoff study before fitting.")
    source.add_argument("--protocol", type=Path, help="Run the frozen nine-fit study.")
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--smoke", action="store_true",
                         help="Run a separate one-origin, one-epoch, reduced-row real-data diagnostic only.")
    args = parser.parse_args()
    if args.freeze_run:
        if args.smoke:
            parser.error("--smoke requires --protocol")
        freeze_protocol(args.freeze_run, args.research_root)
    else:
        run_experiment(args.protocol, research_root=args.research_root, smoke=args.smoke)


if __name__ == "__main__":
    main()
