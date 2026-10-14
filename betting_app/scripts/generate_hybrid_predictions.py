"""Generate hybrid upcoming predictions from model and market consensus."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db
from betting_app.services.upcoming_inference_service import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_HYBRID_MODEL_NAME,
    DEFAULT_HYBRID_TEMPERATURE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    generate_hybrid_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_ALPHA)
    parser.add_argument("--temperature", type=float, default=DEFAULT_HYBRID_TEMPERATURE)
    parser.add_argument("--hybrid-model-name", default=DEFAULT_HYBRID_MODEL_NAME)
    parser.add_argument("--hybrid-model-version")
    args = parser.parse_args()

    init_db()
    rows = generate_hybrid_predictions(
        base_model_name=args.base_model_name,
        base_model_version=args.base_model_version,
        alpha=args.alpha,
        temperature=args.temperature,
        hybrid_model_name=args.hybrid_model_name,
        hybrid_model_version=args.hybrid_model_version,
    )
    print(f"Generated hybrid predictions: {len(rows)} | alpha={args.alpha:.2f} T={args.temperature:.2f}")
    for row in rows[:20]:
        print(f"#{row['canonical_match_id']} pA={row['prob_a']:.3f} pB={row['prob_b']:.3f}")


if __name__ == "__main__":
    main()
