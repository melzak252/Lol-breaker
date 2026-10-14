"""Prepare and inspect SQLite tables required for upcoming-match model inference."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db, query_df
from betting_app.services.upcoming_inference_service import register_operational_model


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-default-model", action="store_true", help="Register the configured operational C0 model.")
    args = parser.parse_args()

    db_path = init_db()
    print(f"DB ready: {db_path}")
    if args.register_default_model:
        model_id = register_operational_model()
        print(f"Registered operational model artifact #{model_id}")
    print_counts()


def print_counts() -> None:
    """Print operational readiness counts."""

    counts = query_df(
        """
        SELECT 'golgg_matches' AS table_name, COUNT(*) AS rows FROM golgg_matches
        UNION ALL SELECT 'golgg_games', COUNT(*) FROM golgg_games
        UNION ALL SELECT 'golgg_game_players', COUNT(*) FROM golgg_game_players
        UNION ALL SELECT 'golgg_teams', COUNT(*) FROM golgg_teams
        UNION ALL SELECT 'canonical_matches', COUNT(*) FROM canonical_matches
        UNION ALL SELECT 'odds_snapshots', COUNT(*) FROM odds_snapshots
        UNION ALL SELECT 'rating_runs', COUNT(*) FROM rating_runs
        UNION ALL SELECT 'entity_ratings', COUNT(*) FROM entity_ratings
        UNION ALL SELECT 'team_rolling_features', COUNT(*) FROM team_rolling_features
        UNION ALL SELECT 'upcoming_match_features', COUNT(*) FROM upcoming_match_features
        UNION ALL SELECT 'model_artifacts', COUNT(*) FROM model_artifacts
        UNION ALL SELECT 'canonical_predictions', COUNT(*) FROM canonical_predictions
        UNION ALL SELECT 'model_ev_signals', COUNT(*) FROM model_ev_signals
        """
    )
    print(counts.to_string(index=False))
    latest = query_df(
        """
        SELECT cm.id, cm.team_a_name || ' vs ' || cm.team_b_name AS match,
               cm.start_time_normalized, COUNT(DISTINCT b.name) AS bookmakers
        FROM canonical_matches cm
        LEFT JOIN odds_snapshots os ON os.canonical_match_id = cm.id
        LEFT JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE cm.status = 'upcoming'
        GROUP BY cm.id
        ORDER BY cm.start_time_normalized ASC
        LIMIT 10
        """
    )
    if not latest.empty:
        print("\nUpcoming canonical matches sample:")
        print(latest.to_string(index=False))


if __name__ == "__main__":
    main()
