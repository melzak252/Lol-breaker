#!/usr/bin/env python3
"""Explicit bound completion of saved research scoring; no fitting or auto-retry."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.champion_history_runner import RUN_ID, score_saved_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    args = parser.parse_args()
    print(json.dumps(score_saved_run(args.run_id), indent=2))
