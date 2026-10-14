"""Build feature vectors for upcoming canonical matches."""

from __future__ import annotations

import argparse
from collections import Counter

from betting_app.core.db import init_db
from betting_app.services.upcoming_inference_service import (
    DEFAULT_FEATURE_VERSION,
    DEFAULT_RATINGS_VERSION,
    DEFAULT_W20_VERSION,
    build_all_upcoming_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    parser.add_argument("--ratings-version", default=DEFAULT_RATINGS_VERSION)
    parser.add_argument("--w20-version", default=DEFAULT_W20_VERSION)
    parser.add_argument("--min-mapping-confidence", type=float, default=0.72)
    parser.add_argument("--include-past", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    init_db()
    results = build_all_upcoming_features(
        feature_version=args.feature_version,
        ratings_version=args.ratings_version,
        w20_version=args.w20_version,
        min_mapping_confidence=args.min_mapping_confidence,
        include_past=args.include_past,
        limit=args.limit,
    )
    counts = Counter(row["status"] for row in results)
    print(f"Built upcoming features: {len(results)} matches | {dict(counts)}")
    for row in results[:15]:
        missing = "; ".join(row["missing"][:4])
        suffix = f" | missing: {missing}" if missing else ""
        print(f"#{row['canonical_match_id']} {row['status']}{suffix}")


if __name__ == "__main__":
    main()
