#!/usr/bin/env python3
"""Execute the separately bound champion-history research recipe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.champion_history_runner import (
    RUN_ID, freeze_run, predict_saved, prepare_run, run_experiment, run_smoke,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'freeze', 'smoke', 'run', 'predict'))
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--research-root')
    parser.add_argument('--match-id', nargs='+')
    parser.add_argument('--variant', choices=('A', 'B', 'C', 'D'), default='D')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare_run(args.run_id, args.research_root)
    elif args.action == 'freeze':
        result = freeze_run(args.run_id)
    elif args.action == 'smoke':
        result = run_smoke(args.run_id)
    elif args.action == 'run':
        result = run_experiment(args.run_id)
    else:
        if not args.match_id:
            parser.error('predict requires --match-id')
        result = predict_saved(args.run_id, args.match_id, args.variant)
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


if __name__ == '__main__':
    main()
