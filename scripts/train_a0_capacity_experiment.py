"""Freeze and execute the native A0 capacity experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.a0_capacity_runner import freeze_run, run_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-run", metavar="RUN_ID")
    group.add_argument("--protocol", metavar="PATH")
    parser.add_argument("--smoke", action="store_true", help="run isolated reduced diagnostic training")
    parser.add_argument("--distill", action="store_true", help="run conditional training-only distillation")
    parser.add_argument("--research-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze_run:
        if args.smoke or args.distill:
            raise SystemExit("--smoke and --distill require --protocol")
        path = freeze_run(args.freeze_run, args.research_root)
        print(path)
    else:
        result = run_protocol(args.protocol, smoke=args.smoke, distill=args.distill,
                             research_root=args.research_root)
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
