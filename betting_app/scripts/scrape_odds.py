"""Run an odds scraper and store snapshots in SQLite."""

from __future__ import annotations

import argparse
import asyncio

from betting_app.core.db import init_db
from betting_app.scrapers.betfan_nodriver import BETFAN_ESPORT_URL, BetfanNoDriverScraper
from betting_app.scrapers.betclic_nodriver import BETCLIC_LOL_URL, BetclicNoDriverScraper
from betting_app.scrapers.dry_run import DryRunScraper
from betting_app.scrapers.efortuna_nodriver import EFORTUNA_LOL_URL, EFortunaNoDriverScraper
from betting_app.scrapers.generic_nodriver import GenericNoDriverScraper
from betting_app.scrapers.lebull_api import LEBULL_ESPORT_URL, LebullApiScraper
from betting_app.scrapers.sts_nodriver import STS_LOL_URL, STSNoDriverScraper
from betting_app.scrapers.superbet_nodriver import SUPERBET_LOL_URL, SuperbetNoDriverScraper
from betting_app.scrapers.totalbet_api import TOTALBET_ESPORT_URL, TotalbetApiScraper
from betting_app.services.odds_service import finish_scrape_run, insert_snapshot, start_scrape_run


BOOKMAKER_URLS = {
    "sts": STS_LOL_URL,
    "betclic": BETCLIC_LOL_URL,
    "superbet": SUPERBET_LOL_URL,
    "efortuna": EFORTUNA_LOL_URL,
    "betfan": BETFAN_ESPORT_URL,
    "totalbet": TOTALBET_ESPORT_URL,
    "lebull": LEBULL_ESPORT_URL,
}


def build_scraper(bookmaker: str, url: str | None, headless: bool | None):
    """Build scraper instance for CLI arguments."""

    if bookmaker == "dry-run":
        return DryRunScraper()
    if bookmaker == "sts":
        return STSNoDriverScraper(start_url=url or BOOKMAKER_URLS["sts"], headless=headless)
    if bookmaker == "betclic":
        return BetclicNoDriverScraper(start_url=url or BOOKMAKER_URLS["betclic"], headless=headless)
    if bookmaker == "superbet":
        return SuperbetNoDriverScraper(start_url=url or BOOKMAKER_URLS["superbet"], headless=headless)
    if bookmaker == "efortuna":
        return EFortunaNoDriverScraper(start_url=url or BOOKMAKER_URLS["efortuna"], headless=headless)
    if bookmaker == "betfan":
        return BetfanNoDriverScraper(start_url=url or BOOKMAKER_URLS["betfan"], headless=headless)
    if bookmaker == "totalbet":
        return TotalbetApiScraper(start_url=url or BOOKMAKER_URLS["totalbet"], headless=headless)
    if bookmaker == "lebull":
        return LebullApiScraper(start_url=url or BOOKMAKER_URLS["lebull"], headless=headless)
    if bookmaker in BOOKMAKER_URLS or url:
        return GenericNoDriverScraper(bookmaker=bookmaker, start_url=url or BOOKMAKER_URLS[bookmaker], headless=headless)
    raise ValueError(f"Unknown bookmaker: {bookmaker}")


async def run(bookmaker: str, url: str | None, headless: bool | None) -> int:
    """Run selected scraper and store snapshots."""

    init_db()
    scraper = build_scraper(bookmaker, url, headless)
    scrape_run_id = start_scrape_run(
        bookmaker=getattr(scraper, "bookmaker", bookmaker),
        scraper_name=type(scraper).__name__,
        scraper_version=getattr(scraper, "SCRAPER_VERSION", None) or getattr(scraper, "scraper_version", None),
        source_url=getattr(scraper, "start_url", url),
    )
    inserted = 0
    try:
        snapshots = await scraper.scrape_upcoming_matches()
        for snapshot in snapshots:
            payload = snapshot.to_dict()
            payload["scrape_run_id"] = scrape_run_id
            row_id = insert_snapshot(payload)
            if row_id:
                inserted += 1
        finish_scrape_run(
            scrape_run_id,
            status="success",
            items_seen=len(snapshots),
            items_inserted=inserted,
        )
        return inserted
    except Exception as exc:
        finish_scrape_run(scrape_run_id, status="failed", items_inserted=inserted, error=f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Scrape bookmaker odds into local SQLite DB")
    parser.add_argument(
        "--bookmaker",
        default="dry-run",
        help="dry-run, sts, betclic, superbet, efortuna, betfan, totalbet, lebull, or custom name",
    )
    parser.add_argument("--url", default=None, help="Override bookmaker URL")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    inserted = asyncio.run(run(args.bookmaker, args.url, args.headless))
    print(f"Inserted odds snapshots: {inserted}")


if __name__ == "__main__":
    main()
