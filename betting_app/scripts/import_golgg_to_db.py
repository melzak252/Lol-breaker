"""Import local GOL.GG JSON cache into betting_app SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path

from betting_app.core.db import init_db, query_df
from betting_app.services.golgg_import_service import DEFAULT_GOLGG_MATCHES_PATH, import_golgg_matches


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Import data/golgg_matches.json into relational SQLite tables")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_GOLGG_MATCHES_PATH)
    parser.add_argument("--limit", type=int, help="Import only first N matches for testing")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    init_db()
    stats = import_golgg_matches(args.input_path, limit=args.limit, batch_size=args.batch_size)
    print(
        "Imported GOL.GG: "
        f"matches={stats['matches']}, games={stats['games']}, players={stats['players']}, team_upserts={stats['teams']}"
    )
    print(current_counts().to_string(index=False))


def current_counts():
    """Return current relational GOL.GG table counts."""

    return query_df(
        """
        SELECT 'golgg_matches' AS table_name, COUNT(*) AS rows FROM golgg_matches
        UNION ALL SELECT 'golgg_games', COUNT(*) FROM golgg_games
        UNION ALL SELECT 'golgg_game_players', COUNT(*) FROM golgg_game_players
        UNION ALL SELECT 'golgg_teams', COUNT(*) FROM golgg_teams
        """
    )


if __name__ == "__main__":
    main()
