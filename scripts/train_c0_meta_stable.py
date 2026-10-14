#!/usr/bin/env python3
"""Freeze, prepare, fit or replay the fixed180-player/30-meta C0 follow-up."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.c0_meta_stable_study import (
    freeze_protocol, prepare_features, predict_saved, run_experiment, evaluate_support_profiles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', metavar='RUN_ID')
    mode.add_argument('--prepare', type=Path, metavar='PROTOCOL')
    mode.add_argument('--run', type=Path, metavar='PROTOCOL')
    mode.add_argument('--smoke', type=Path, metavar='PROTOCOL', help='Separate unscored one-seed/one-epoch real-data exercise.')
    mode.add_argument('--predict', type=Path, metavar='BUNDLE', help='Replay prepared origin-eligible saved rows.')
    mode.add_argument('--support-profiles', type=Path, metavar='PROTOCOL', help='Supplement a completed central benchmark; no new decision.')
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--benchmark-dir', type=Path)
    parser.add_argument('--output', type=Path, help='New JSON receipt; no overwrite.')
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
        prepare_features(args.prepare, args.research_root)
    elif args.support_profiles:
        if args.benchmark_dir is None or args.output is None:
            parser.error('--support-profiles requires --benchmark-dir and --output')
        evaluate_support_profiles(args.support_profiles,args.benchmark_dir,args.output,args.research_root)
    else:
        run_experiment(args.run or args.smoke,args.research_root,smoke=args.smoke is not None)


if __name__ == '__main__':
    main()
