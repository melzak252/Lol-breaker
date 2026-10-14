"""Dry-run scraper used for testing the storage pipeline without a bookmaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from betting_app.scrapers.base import RawOddsSnapshot


class DryRunScraper:
    """Return deterministic sample LoL odds snapshots."""

    bookmaker = "manual"

    async def scrape_upcoming_matches(self) -> list[RawOddsSnapshot]:
        """Return sample upcoming match winner markets."""

        now = datetime.now(UTC).replace(microsecond=0)
        return [
            RawOddsSnapshot(
                bookmaker=self.bookmaker,
                raw_team_a="T1",
                raw_team_b="Gen.G",
                odds_a=2.05,
                odds_b=1.75,
                scraped_at=now.isoformat(),
                raw_league="LCK",
                match_start_time=(now + timedelta(days=1)).isoformat(),
                source_url="dry-run://sample/t1-geng",
                is_live=False,
                scraper_name="dry_run",
                scraper_version="0.1",
            ),
            RawOddsSnapshot(
                bookmaker=self.bookmaker,
                raw_team_a="Bilibili Gaming",
                raw_team_b="Top Esports",
                odds_a=1.82,
                odds_b=1.95,
                scraped_at=now.isoformat(),
                raw_league="LPL",
                match_start_time=(now + timedelta(days=2)).isoformat(),
                source_url="dry-run://sample/blg-tes",
                is_live=False,
                scraper_name="dry_run",
                scraper_version="0.1",
            ),
        ]
