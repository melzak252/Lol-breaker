"""Initialize the betting app SQLite database."""

from __future__ import annotations

from betting_app.core.db import init_db
from betting_app.services.betting_service import initialize_bankroll
from betting_app.services.mapping_service import sync_golgg_teams


def main() -> None:
    """Create database and seed basic reference data."""

    path = init_db()
    initialize_bankroll(100.0)
    teams = sync_golgg_teams()
    print(f"Initialized betting DB: {path}")
    print(f"Synced GOL.GG team candidates: {teams}")


if __name__ == "__main__":
    main()
