"""Generate model EV signals from latest predictions and bookmaker odds."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db
from betting_app.core.models import get_active_hybrid
from betting_app.services.upcoming_inference_service import (
    DEFAULT_HYBRID_MODEL_NAME,
    generate_model_ev_signals,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    hybrid = get_active_hybrid()
    parser.add_argument("--model-name", default=DEFAULT_HYBRID_MODEL_NAME)
    parser.add_argument("--model-version", default=hybrid.hybrid_model_version)
    parser.add_argument("--tax-rate", type=float, default=0.12)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--bankroll", type=float, default=100.0)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    init_db()
    signals = generate_model_ev_signals(
        model_name=args.model_name,
        model_version=args.model_version,
        tax_rate=args.tax_rate,
        min_ev=args.min_ev,
        bankroll=args.bankroll,
    )
    print(f"Generated EV signals: {len(signals)} | tax={args.tax_rate:.2%} | min_ev={args.min_ev:.2%}")
    for row in signals[: args.limit]:
        team_side = "Team A" if row["side"] == "a" else "Team B"
        print(
            f"#{row['canonical_match_id']} {row['match']} | {row['bookmaker']} | {team_side} "
            f"odds={row['odds']:.2f} p={row['model_prob']:.3f} EV={row['ev']:.2%} stake={row['stake_suggestion']:.2f}"
        )
        if row.get("offer_url"):
            print(f"  {row['offer_url']}")


if __name__ == "__main__":
    main()
