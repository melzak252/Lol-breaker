#!/usr/bin/env python3
"""Freeze, prepare, run, or inspect the isolated historical P1 duration study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.duration_scale_reliability import freeze_run, prepare_run, run_experiment, predict_saved_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'prepare', 'run', 'predict'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--match-id')
    args = parser.parse_args()
    research = args.research_root
    if research is None:
        research = Path((args.repo_root / 'data/research_root.txt').read_text().strip())
    if args.command == 'predict' and args.match_id is None: parser.error('predict requires --match-id')
    if args.command != 'predict' and args.match_id is not None: parser.error('--match-id is only valid for predict')
    functions = {'freeze':freeze_run, 'prepare':prepare_run, 'run':run_experiment, 'predict':predict_saved_row}
    keywords = {'repo_root':args.repo_root, 'research_root':research}
    if args.command == 'predict': keywords['match_id'] = args.match_id
    result = functions[args.command](args.run_id, **keywords)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
