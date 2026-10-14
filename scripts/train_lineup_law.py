#!/usr/bin/env python3
"""Bounded historical R1 pipeline; no DB, source acquisition or live forecasts.

Usage: train_lineup_law.py --run-id RUN [--repo-root ROOT]
       [--research-root ARCHIVE] [--evidence-root AUTHORIZATION] freeze|prepare|run
       train_lineup_law.py --run-id RUN predict --match-id ID [--law conditional]
       train_lineup_law.py --run-id NEW recover-evaluation
       --parent-run-id FAILED --authorization EXACT_RECOVERY_AUTHORIZATION_JSON

Freeze requires original frozen local pinned inputs, not publication-authenticated
data. Prepare closes unsupported runs with zero fits. Run consumes each of two
TRAIN slots before invocation and never restarts. Predict reads only saved
candidate state and exact native captures; it does not require actual target
labels and never returns an invented scalar.
Recovery reuses two accepted parent fits and complete saved scores byte-for-byte
in a fresh namespace, with zero new fits and no retry of parent native targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.analysis.lineup_law_reliability import freeze_run, prepare_run, run_study, predict_run, recover_evaluation, _safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze')
    commands.add_parser('prepare')
    commands.add_parser('run')
    recovery = commands.add_parser('recover-evaluation')
    recovery.add_argument('--parent-run-id', required=True)
    recovery.add_argument('--authorization', required=True, type=Path)
    predict = commands.add_parser('predict')
    predict.add_argument('--match-id', action='append', required=True)
    predict.add_argument('--law', choices=('conditional', 'global'), default='conditional')
    args = parser.parse_args(argv)
    roots = {'repo_root': args.repo_root, 'research_root': args.research_root}
    if args.command == 'freeze':
        result = freeze_run(args.run_id, evidence_root=args.evidence_root, **roots)
    elif args.command == 'prepare':
        result = prepare_run(args.run_id, **roots)
    elif args.command == 'run':
        result = run_study(args.run_id, **roots)
    elif args.command == 'recover-evaluation':
        result = recover_evaluation(args.run_id, parent_run_id=args.parent_run_id,
                                    authorization_path=args.authorization, **roots)
    else:
        result = predict_run(args.run_id, args.match_id, law=args.law, **roots)
    print(json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
