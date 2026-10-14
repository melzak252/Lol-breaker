#!/usr/bin/env python3
"""Run the separately versioned, odds-free historical prop-distribution study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.prop_distribution_runner import freeze_run, predict_saved_row, run_experiment, run_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'smoke', 'run', 'predict'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--match-id', help='Retained historical ID for saved-row inference; not a live roster adapter.')
    args = parser.parse_args()
    if args.command == 'predict':
        if not args.match_id:
            parser.error('predict requires --match-id')
        result = predict_saved_row(args.run_id, args.match_id)
    else:
        result = {'freeze': freeze_run, 'smoke': run_smoke, 'run': run_experiment}[args.command](args.run_id)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
