#!/usr/bin/env python3
"""Execute the separate frozen Stage2 draft-information gate and saved-row inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.draft_prop_runner import freeze_run, predict_saved_row, prepare_run, run_smoke, run_stage2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'freeze', 'smoke', 'run', 'predict'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--match-id', help='Retained historical target; actual-composition oracle is not a live pre-match forecast.')
    parser.add_argument('--variant', choices=['matched_no_draft', 'actual_composition_oracle'], default='actual_composition_oracle')
    args = parser.parse_args()
    if args.command == 'predict':
        if not args.match_id:
            parser.error('predict requires --match-id')
        result = predict_saved_row(args.run_id, args.match_id, args.variant)
    else:
        result = {'prepare': prepare_run, 'freeze': freeze_run, 'smoke': run_smoke, 'run': run_stage2}[args.command](args.run_id)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
