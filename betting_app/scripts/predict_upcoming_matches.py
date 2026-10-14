"""Generate automatic probabilities for upcoming canonical matches."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db
from betting_app.services.upcoming_inference_service import (
    DEFAULT_FEATURE_VERSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    DEFAULT_RATINGS_VERSION,
    predict_all_upcoming,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    parser.add_argument("--ratings-version", default=DEFAULT_RATINGS_VERSION)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--include-partial", action="store_true", help="Also predict rows with missing ratings/W20.")
    args = parser.parse_args()

    init_db()
    predictions = predict_all_upcoming(
        feature_version=args.feature_version,
        ratings_version=args.ratings_version,
        model_name=args.model_name,
        model_version=args.model_version,
        include_partial=args.include_partial,
    )
    print(f"Generated predictions: {len(predictions)}")
    for row in predictions[:20]:
        print(f"#{row['canonical_match_id']} {row['match']} | pA={row['prob_a']:.3f} pB={row['prob_b']:.3f}")


if __name__ == "__main__":
    main()
