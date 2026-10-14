"""CLI entrypoint for the strict model-versus-market benchmark.

Example:

    .venv/bin/python -m betting_app.ml.pipelines.evaluate_existing_model \
      --model-name Sym-Cal-LR-ElasticNet-W20-Binomial \
      --model-version exp-039 \
      --json
"""

from __future__ import annotations

import argparse
import json

from betting_app.ml.pipelines.evaluation import EvaluationPipelineConfig, run_evaluation_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict point-in-time model-versus-market benchmark"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="Include predictions marked stale; strict timestamp eligibility still applies.",
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_evaluation_pipeline(
        EvaluationPipelineConfig(
            model_name=args.model_name,
            model_version=args.model_version,
            days_back=args.days_back,
            include_stale=args.include_stale,
            register_candidate=not args.no_register,
        )
    )
    payload = {
        **result.metrics,
        "evaluation_run_id": result.evaluation_run_id,
        "promotion_gate_passed": result.decision.passed,
        "promotion_gate_reasons": result.decision.reasons,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Strict model-versus-market benchmark")
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
