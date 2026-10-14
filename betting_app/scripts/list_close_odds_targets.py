"""List bookmaker event links for manual close-odds checks.

Use this before match start to open the stored event page and refresh closing
odds, e.g.:

    python -m betting_app.scripts.list_close_odds_targets --bookmaker betclic
"""

from __future__ import annotations

import argparse

from betting_app.core.db import init_db, query_df


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="List stored bookmaker offer URLs")
    parser.add_argument(
        "--bookmaker",
        choices=["sts", "betclic", "superbet", "efortuna", "fortuna", "betfan", "totalbet", "lebull"],
        help="Filter bookmaker",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows")
    args = parser.parse_args()

    init_db()
    rows = close_odds_targets(bookmaker=args.bookmaker, limit=args.limit)
    if rows.empty:
        print("No close-odds targets with offer_url found yet.")
        return
    for row in rows.to_dict("records"):
        print(
            f"[{row['bookmaker']}] {row['match_start_time'] or '?'} | "
            f"{row['raw_team_a']} vs {row['raw_team_b']} | {row['league'] or '?'}\n"
            f"  {row['offer_url']}"
        )


def close_odds_targets(bookmaker: str | None = None, limit: int = 50):
    """Return latest known matches with per-event bookmaker links."""

    params: list[object] = []
    where = "WHERE COALESCE(os.offer_url, um.offer_url) IS NOT NULL"
    if bookmaker:
        where += " AND b.name = ?"
        params.append(bookmaker)
    params.append(limit)
    return query_df(
        f"""
        SELECT
            b.name AS bookmaker,
            os.raw_team_a,
            os.raw_team_b,
            os.match_start_time,
            os.raw_league AS league,
            COALESCE(os.offer_url, um.offer_url) AS offer_url,
            MAX(os.scraped_at) AS last_scraped_at
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        LEFT JOIN upcoming_matches um ON um.id = os.match_id
        {where}
        GROUP BY b.name, os.raw_team_a, os.raw_team_b, os.match_start_time, os.raw_league, COALESCE(os.offer_url, um.offer_url)
        ORDER BY last_scraped_at DESC, os.match_start_time ASC
        LIMIT ?
        """,
        tuple(params),
    )


if __name__ == "__main__":
    main()
