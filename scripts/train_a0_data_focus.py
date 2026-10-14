#!/usr/bin/env python3
"""Run the frozen single-split A0 data-population/history-masking experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.a0_data_focus import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--research-root', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='Separate unscored artifacts; one epoch on small TRAIN/CAL subsets.')
    mode.add_argument('--evaluate-only', action='store_true', help='Rescore the completed fixed run with shared benchmark metrics.')
    args = parser.parse_args()
    if args.evaluate_only:
        from src.analysis.a0_data_focus import evaluate_experiment
        evaluate_experiment(args.protocol, research_root=args.research_root)
    else:
        run_experiment(args.protocol, research_root=args.research_root, smoke=args.smoke)


if __name__ == '__main__':
    main()
