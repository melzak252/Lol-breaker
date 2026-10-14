#!/usr/bin/env python3
"""Freeze, prepare, fit or replay the bounded frozen-C0 thirty-day meta study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.c0_meta_study import freeze_protocol, prepare_features, predict_saved, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', metavar='RUN_ID')
    mode.add_argument('--prepare', type=Path, metavar='PROTOCOL')
    mode.add_argument('--run', type=Path, metavar='PROTOCOL')
    mode.add_argument('--smoke', type=Path, metavar='PROTOCOL', help='Separate unscored one-seed/one-epoch real-data exercise.')
    mode.add_argument('--predict', type=Path, metavar='BUNDLE', help='Replay saved correction on prepared, origin-eligible feature rows.')
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--output', type=Path, help='New JSON receipt for saved-artifact prediction.')
    args = parser.parse_args()
    if args.predict:
        if args.features is None or args.indices is None:
            parser.error('--predict requires --features and --indices')
        result = predict_saved(args.predict, args.features, args.indices)
        payload = {key: value.tolist() for key, value in result.items()}
        payload['indices'] = args.indices
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(payload, stream, sort_keys=True, allow_nan=False)
                stream.write('\n')
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
    elif args.freeze:
        freeze_protocol(args.freeze, args.research_root)
    elif args.prepare:
        # Only this fresh local process activates the archived native namespace.
        os.environ['BETTING_APP_C0_WORKER'] = '1'
        prepare_features(args.prepare, args.research_root)
    else:
        run_experiment(args.run or args.smoke, args.research_root, smoke=args.smoke is not None)


if __name__ == '__main__':
    main()
