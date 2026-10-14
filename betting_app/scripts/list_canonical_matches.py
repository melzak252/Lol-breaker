"""List cross-bookmaker canonical matches and available odds."""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db, query_df
from betting_app.services.canonical_match_service import align_snapshot_odds, canonical_match_overview


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="List canonical cross-bookmaker LoL matches")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--details", action="store_true", help="Also print latest bookmaker odds for each match")
    args = parser.parse_args()

    init_db()
    overview = canonical_match_overview(limit=args.limit)
    if overview.empty:
        print("No canonical matches yet. Run scrapers or rematch_canonical_matches first.")
        return
    print(overview.to_string(index=False))
    if args.details:
        for match_id in overview["canonical_match_id"].tolist():
            print(f"\n# canonical_match_id={match_id}")
            details = bookmaker_odds_for_match(int(match_id))
            print(details.to_string(index=False) if not details.empty else "No odds")


def bookmaker_odds_for_match(canonical_match_id: int):
    """Return latest two-sided odds per bookmaker for one canonical match."""

    rows = query_df(
        """
        WITH ranked AS (
            SELECT
                cm.normalized_team_a,
                cm.normalized_team_b,
                cm.league, cm.start_time_normalized,
                b.name AS bookmaker,
                os.raw_team_a,
                os.raw_team_b,
                os.odds_a,
                os.odds_b,
                os.offer_url,
                os.scraped_at,
                ROW_NUMBER() OVER (PARTITION BY b.name ORDER BY os.scraped_at DESC, os.id DESC) AS rn
            FROM odds_snapshots os
            JOIN canonical_matches cm ON cm.id = os.canonical_match_id
            JOIN bookmakers b ON b.id = os.bookmaker_id
            WHERE os.canonical_match_id = ?
              AND os.market_type = 'match_winner'
        )
        SELECT normalized_team_a, normalized_team_b, league, start_time_normalized, bookmaker, raw_team_a, raw_team_b, odds_a, odds_b, offer_url, scraped_at
        FROM ranked
        WHERE rn = 1
        ORDER BY bookmaker
        """,
        (canonical_match_id,),
    )
    if rows.empty:
        return rows
    aligned_rows = []
    for row in rows.to_dict("records"):
        aligned = align_snapshot_odds(
            str(row["normalized_team_a"]),
            str(row["normalized_team_b"]),
            str(row["raw_team_a"]),
            str(row["raw_team_b"]),
            row["odds_a"],
            row["odds_b"],
            league=row.get("league"), source_system="bookmaker",
            match_date=row.get("start_time_normalized"),
        )
        row["canonical_odds_a"] = aligned[0] if aligned else None
        row["canonical_odds_b"] = aligned[1] if aligned else None
        aligned_rows.append(row)
    return rows.__class__(aligned_rows)


if __name__ == "__main__":
    main()
