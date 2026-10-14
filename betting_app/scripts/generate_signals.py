"""Generate EV+ signals from latest odds and predictions."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db
from betting_app.services.betting_service import generate_signals


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-ev", type=float, default=None)
    parser.add_argument("--tax-rate", type=float, default=None)
    args = parser.parse_args()
    init_db()
    created = generate_signals(min_ev=args.min_ev, tax_rate=args.tax_rate)
    print(f"Created signals: {created}")


if __name__ == "__main__":
    main()
