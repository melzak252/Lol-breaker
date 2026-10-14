"""List latest upcoming model predictions with best odds and EV."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db, query_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()

    init_db()
    where = "WHERE 1=1"
    if args.positive_only:
        where += " AND best_ev > 0"
    frame = query_df(
        f"""
        WITH latest_pred AS (
            SELECT p.*
            FROM canonical_predictions p
            JOIN (
                SELECT canonical_match_id, model_name, model_version, MAX(predicted_at) AS predicted_at
                FROM canonical_predictions
                WHERE prediction_status = 'active'
                GROUP BY canonical_match_id, model_name, model_version
            ) lp ON lp.canonical_match_id = p.canonical_match_id
                 AND lp.model_name = p.model_name
                 AND lp.model_version = p.model_version
                 AND lp.predicted_at = p.predicted_at
        ), best_signals AS (
            SELECT canonical_match_id,
                   MAX(ev) AS best_ev,
                   COUNT(*) AS positive_signal_count
            FROM model_ev_signals
            WHERE status = 'new'
            GROUP BY canonical_match_id
        ), book_counts AS (
            SELECT canonical_match_id, COUNT(DISTINCT bookmaker_id) AS bookmaker_count
            FROM odds_snapshots
            GROUP BY canonical_match_id
        )
        SELECT cm.id AS canonical_match_id,
               cm.team_a_name || ' vs ' || cm.team_b_name AS match,
               cm.start_time_normalized,
               cm.league,
               COALESCE(bc.bookmaker_count, 0) AS bookmakers,
               lp.model_name,
               lp.model_version,
               ROUND(lp.prob_a, 4) AS prob_a,
               ROUND(lp.prob_b, 4) AS prob_b,
               ROUND(bs.best_ev, 4) AS best_ev,
               COALESCE(bs.positive_signal_count, 0) AS positive_signal_count
        FROM latest_pred lp
        JOIN canonical_matches cm ON cm.id = lp.canonical_match_id
        LEFT JOIN best_signals bs ON bs.canonical_match_id = cm.id
        LEFT JOIN book_counts bc ON bc.canonical_match_id = cm.id
        {where}
        ORDER BY cm.start_time_normalized ASC, best_ev DESC
        LIMIT ?
        """,
        (args.limit,),
    )
    if frame.empty:
        print("No predictions found.")
    else:
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
