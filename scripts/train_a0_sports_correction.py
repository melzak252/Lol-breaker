#!/usr/bin/env python3
"""Run the frozen anchored-sports correction; benchmark with run_model_benchmark.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.a0_correction_experiment import evaluate, predict, prepare, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "predict"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    if args.stage == "predict":
        if args.artifact is None or args.inputs is None:
            parser.error("predict requires --artifact and --inputs")
        result = predict(args.artifact, args.inputs)
        print(json.dumps(result, allow_nan=False))
        return
    if args.protocol is None:
        parser.error("training/evaluation requires --protocol")
    is_c0 = json.loads(args.protocol.read_text()).get("schema") == "c0_raw_specialist_connector_v1"
    if args.stage == "prepare":
        result = prepare(args.protocol, args.research_root)
        fields = ("status", "rows", "primary_rows", "warmup_rows", "joined_sha256") if is_c0 else (
            "status", "rows", "reversed_specialist_rows", "joined_sha256")
        summary = {key: result[key] for key in fields}
    elif args.stage == "train":
        result = train(args.protocol, args.research_root)
        fields = ("status", "rows", "warmup_rows", "head_fits", "base_fits") if is_c0 else (
            "status", "rows", "warmup_rows", "correction_fits", "base_fits")
        summary = {key: result[key] for key in fields}
    else:
        result = evaluate(args.protocol)
        summary = {"decision": result["decision"]} if is_c0 else result["decision"]
    print(json.dumps({"stage": args.stage, **summary}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
