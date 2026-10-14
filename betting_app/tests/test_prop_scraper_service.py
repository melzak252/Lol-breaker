"""Tests for PropScraperService orchestrator."""

import pytest
from unittest.mock import AsyncMock, patch

from betting_app.scrapers.props_parser import ParsedMatchProps, ParsedPropLine
from betting_app.services.prop_scraper_service import (
    PropScraperService,
    SCRAPERS_REGISTRY,
    run_all_prop_scrapers,
)


def test_registry_contains_all_bookmakers():
    assert "sts" in SCRAPERS_REGISTRY
    assert "efortuna" in SCRAPERS_REGISTRY
    assert "betclic" in SCRAPERS_REGISTRY
    assert "superbet" in SCRAPERS_REGISTRY


@pytest.mark.asyncio
async def test_scrape_unknown_bookmaker_raises():
    service = PropScraperService()
    with pytest.raises(ValueError, match="Unknown or unsupported bookmaker"):
        await service.scrape_bookmaker("non_existent_bookmaker")


@pytest.mark.asyncio
async def test_run_all_prop_scrapers_mocked():
    sample_props = ParsedMatchProps(
        bookmaker="sts",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(
                market_type="total_kills",
                line=26.5,
                odds_over=1.85,
                odds_under=1.95,
                novig_prob_over=0.5132,
                novig_prob_under=0.4868,
            )
        ],
    )

    with patch.object(PropScraperService, "scrape_bookmaker", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = [sample_props]
        with patch.object(PropScraperService, "process_and_persist_props") as mock_process:
            mock_process.return_value = {
                "total_matches_processed": 1,
                "matched_to_canonical": 1,
                "total_lines_persisted": 1,
                "predictions_generated": 1,
            }

            res = await run_all_prop_scrapers(bookmakers=["sts"])
            assert res["results"]["sts"]["status"] == "success"
            assert res["results"]["sts"]["scraped_matches"] == 1
            assert res["persistence"]["total_lines_persisted"] == 1
