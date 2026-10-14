"""CLI for registry-backed shadow inference."""

from __future__ import annotations

import argparse
import json

from betting_app.ml.inference.registry_inference import run_registry_shadow_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run shadow inference for registered ML models")
    parser.add_argument("--status", action="append", default=None, help="Registry status to include; can be repeated. Default: shadow")
    parser.add_argument("--model-name")
    parser.add_argument("--model-version")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_registry_shadow_inference(
        statuses=tuple(args.status or ["shadow"]),
        model_name=args.model_name,
        model_version=args.model_version,
        limit=args.limit,
    )
    payload = {
        "models_seen": result.models_seen,
        "models_loaded": result.models_loaded,
        "feature_rows_seen": result.feature_rows_seen,
        "predictions_written": result.predictions_written,
        "model_versions": result.model_versions,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
