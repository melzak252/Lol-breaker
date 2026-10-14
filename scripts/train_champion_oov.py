#!/usr/bin/env python3
"""Run the isolated W1 historical OOV study, never a serving-model cutover.

Example: python scripts/train_champion_oov.py inventory --run-id RUN
Then freeze and run the same fresh RUN only if its Stage0 receipt passes support.
Every fit attempt is journaled before invocation; an existing journal is final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.champion_oov_reliability import inventory_run, freeze_run, run_experiment, predict_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "freeze", "run", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--repo-root", type=Path, default=ROOT)
        command.add_argument("--research-root", type=Path)
        command.add_argument("--evidence-root", type=Path, help="Explicit authorization/contracts root; defaults to run-derived reporting root")
        if name == "predict":
            command.add_argument("--match-id", action="extend", nargs="+", required=True, help="One or more immutable canonical IDs; supplied order retained")
            command.add_argument("--arm", choices=("H", "C", "HC"), default="HC")
    args = parser.parse_args()
    options = {"repo_root": args.repo_root, "research_root": args.research_root, "evidence_root": args.evidence_root}
    if args.command == "predict":
        result = predict_saved(args.run_id, args.match_id, arm=args.arm, **options)
    else:
        result = {"inventory": inventory_run, "freeze": freeze_run, "run": run_experiment}[args.command](args.run_id, **options)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
