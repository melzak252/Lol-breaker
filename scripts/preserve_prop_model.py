#!/usr/bin/env python3
"""Preserve or verify the standalone prop study; no training or serving changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.prop_model_preservation import create_bundle, replay_restored, restore_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create', help='Archive all five study stages and the inference source closure.')
    create.add_argument('--run-id', required=True)
    create.add_argument('--output', required=True, type=Path)
    verify = commands.add_parser('verify', help='Restore temporarily and compare complete marginal distributions.')
    verify.add_argument('--bundle', required=True, type=Path)
    verify.add_argument('--sha256', required=True, help='Trusted full-bundle digest from the creation receipt.')
    verify.add_argument('--receipt', type=Path, help='New receipt path; never overwrite an existing receipt.')
    restore = commands.add_parser('restore', help='Validate and extract into a new directory; never overwrite.')
    restore.add_argument('--bundle', required=True, type=Path)
    restore.add_argument('--sha256', required=True)
    restore.add_argument('--destination', required=True, type=Path)
    replay = commands.add_parser('replay', help='Replay recovery examples using the extracted sources/environment.')
    replay.add_argument('--root', required=True, type=Path)
    replay.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'create':
        result = create_bundle(ROOT, args.run_id, args.output)
    elif args.command == 'verify':
        if args.receipt and args.receipt.exists():
            raise FileExistsError(args.receipt)
        result = verify_bundle(args.bundle, args.sha256, ROOT)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write('\n')
    elif args.command == 'restore':
        manifest = restore_bundle(args.bundle, args.sha256, args.destination)
        result = {'run_id': manifest['run_id'], 'destination': str(args.destination), 'members': len(manifest['members']) + 1}
    else:
        result = replay_restored(args.root, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
