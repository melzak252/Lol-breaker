"""Tests for match view proposition odds scraping across Polish sportsbooks."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from betting_app.scrapers.sts_props_scraper import STSPropsScraper
from betting_app.scrapers.betclic_props_scraper import BetclicPropsScraper
from betting_app.scrapers.efortuna_props_scraper import EFortunaPropsScraper
from betting_app.scrapers.superbet_props_scraper import SuperbetPropsScraper
from betting_app.services.prop_scraper_service import PropScraperService
from betting_app.scrapers.props_parser import ParsedMatchProps, ParsedPropLine

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from betting_app.models.base import Base

@pytest.fixture
def sqlite_db():
    """Create an isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_sts_extract_fixture_offer_urls():
    """Verify STSPropsScraper constructs direct match detail URLs from overview SSR data."""
    scraper = STSPropsScraper()
    overview_payload = {
        "T": {
            "t1": {
                "FX": {
                    "f3035211": {"H": {"N": "3V"}, "A": {"N": "Fuego"}},
                    "f3279937": {"H": "Lyon", "A": "Estral"},
                }
            }
        }
    }
    urls = scraper.extract_fixture_offer_urls(overview_payload)
    assert len(urls) == 2
    assert urls[0]["fixture_id"] == "f3035211"
    assert urls[0]["home"] == "3V"
    assert urls[0]["away"] == "Fuego"
    assert "https://www.sts.pl/kursy/" in urls[0]["url"]
    assert "f3035211" in urls[0]["url"]

    assert urls[1]["fixture_id"] == "f3279937"
    assert urls[1]["home"] == "Lyon"
    assert urls[1]["away"] == "Estral"
    assert "f3279937" in urls[1]["url"]


@pytest.mark.asyncio
async def test_sts_scrape_upcoming_props_with_match_urls():
    """Verify STSPropsScraper navigates to specific match views and parses proposition lines."""
    scraper = STSPropsScraper()
    match_urls = ["https://www.sts.pl/kursy/fnatic-g2/f12345"]

    mock_detail_payload = {
        "H": "Fnatic",
        "A": "G2 Esports",
        "a": {"o1": {}},
        "O": {
            "o1": {
                "m": {
                    "m1": {
                        "n": "1. mapa - suma zabójstw (26.5)",
                        "l": {
                            "l1": {
                                "o": {
                                    "1": {"n": "powyżej 26.5", "O": 1.85},
                                    "2": {"n": "poniżej 26.5", "O": 1.95},
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    mock_tab = MagicMock()
    mock_tab.evaluate = AsyncMock(return_value=mock_detail_payload)

    mock_client = MagicMock()
    mock_client.open = AsyncMock(return_value=mock_tab)

    with patch("betting_app.scrapers.sts_props_scraper.NoDriverClient") as mock_nd:
        mock_nd.return_value.__aenter__.return_value = mock_client
        results = await scraper.scrape_upcoming_props(max_matches=5, match_urls=match_urls)

    assert len(results) == 1
    mp = results[0]
    assert mp.bookmaker == "sts"
    assert mp.raw_team_a == "Fnatic"
    assert mp.raw_team_b == "G2 Esports"
    assert mp.source_url == match_urls[0]
    
    kills_lines = mp.get_total_kills_lines()
    assert len(kills_lines) == 1
    assert kills_lines[0].line == 26.5
    assert kills_lines[0].odds_over == 1.85
    assert kills_lines[0].odds_under == 1.95


@pytest.mark.asyncio
async def test_betclic_scrape_upcoming_props_with_match_urls():
    """Verify BetclicPropsScraper navigates to match views and parses market cards."""
    scraper = BetclicPropsScraper()
    match_urls = ["https://www.betclic.pl/league-of-legends-slol/fnatic-g2-esports-m12345"]

    mock_markets_payload = [
        {
            "name": "Liczba zabójstw na 1. mapie (28.5)",
            "selections": [
                {"name": "Powyżej 28.5", "odds": "1.82"},
                {"name": "Poniżej 28.5", "odds": "1.90"},
            ]
        },
        {
            "name": "Handicap zabójstw na 1. mapie (-3.5)",
            "selections": [
                {"name": "Fnatic (-3.5)", "odds": "1.95"},
                {"name": "G2 Esports (+3.5)", "odds": "1.78"},
            ]
        }
    ]

    mock_info = {"team_a": "Fnatic", "team_b": "G2 Esports"}

    mock_tab = MagicMock()
    mock_client = MagicMock()
    mock_client.open = AsyncMock(return_value=mock_tab)

    with patch("betting_app.scrapers.betclic_props_scraper.NoDriverClient") as mock_nd:
        mock_nd.return_value.__aenter__.return_value = mock_client
        with patch.object(scraper, "_extract_match_info", AsyncMock(return_value=mock_info)):
            with patch.object(scraper, "_extract_event_links_and_markets", AsyncMock(return_value=mock_markets_payload)):
                results = await scraper.scrape_upcoming_props(max_matches=5, match_urls=match_urls)

    assert len(results) == 1
    mp = results[0]
    assert mp.bookmaker == "betclic"
    assert mp.raw_team_a == "Fnatic"
    assert mp.raw_team_b == "G2 Esports"
    assert mp.source_url == match_urls[0]

    tk = mp.get_total_kills_lines()
    assert len(tk) == 1
    assert tk[0].line == 28.5
    assert tk[0].odds_over == 1.82
    assert tk[0].odds_under == 1.90

    hk = mp.get_handicap_lines()
    assert len(hk) == 1
    assert hk[0].line == 3.5


@pytest.mark.asyncio
async def test_superbet_scrape_upcoming_props_with_match_urls():
    """Verify SuperbetPropsScraper navigates to match views and parses market items."""
    scraper = SuperbetPropsScraper()
    match_urls = ["https://superbet.pl/zaklady-bukmacherskie/league-of-legends/mecz-12345"]

    mock_markets_payload = [
        {
            "name": "Liczba zabójstw na 1. mapie (27.5)",
            "choices": [
                {"name": "Powyżej 27.5", "price": "1.85"},
                {"name": "Poniżej 27.5", "price": "1.85"},
            ]
        }
    ]

    mock_tab = MagicMock()
    mock_client = MagicMock()
    mock_client.open = AsyncMock(return_value=mock_tab)

    with patch("betting_app.scrapers.superbet_props_scraper.NoDriverClient") as mock_nd:
        mock_nd.return_value.__aenter__.return_value = mock_client
        with patch.object(scraper, "_extract_markets", AsyncMock(return_value=mock_markets_payload)):
            results = await scraper.scrape_upcoming_props(max_matches=5, match_urls=match_urls)

    assert len(results) == 1
    mp = results[0]
    assert mp.bookmaker == "superbet"
    assert mp.source_url == match_urls[0]
    tk = mp.get_total_kills_lines()
    assert len(tk) == 1
    assert tk[0].line == 27.5


@pytest.mark.asyncio
async def test_prop_scraper_service_persists_match_view_props(sqlite_db):
    """Verify PropScraperService matches offer_url directly to canonical_match_id and stores snapshots."""
    from betting_app.models.match import CanonicalMatch
    from betting_app.models.bookmaker import Bookmaker
    from betting_app.models.odds import BookmakerEvent, PropOddsSnapshot

    # Create dummy canonical match and bookmaker event
    cm = CanonicalMatch(
        id=999,
        canonical_key="t1-vs-geng-2026-09-30",
        team_a_name="T1",
        team_b_name="Gen.G",
        normalized_team_a="t1",
        normalized_team_b="geng",
        status="upcoming",
        league="LCK",
        start_time_normalized="2026-09-30T10:00:00Z",
    )
    sqlite_db.add(cm)
    bm = Bookmaker(id=2, name="sts", base_url="https://www.sts.pl")
    sqlite_db.add(bm)
    sqlite_db.flush()

    be_url = "https://www.sts.pl/kursy/t1-geng/f9999"
    be = BookmakerEvent(
        bookmaker_id=bm.id,
        bookmaker_event_id="f9999",
        canonical_match_id=cm.id,
        raw_team_a="T1",
        raw_team_b="Gen.G",
        offer_url=be_url,
    )
    sqlite_db.add(be)
    sqlite_db.commit()

    service = PropScraperService()
    scraped_props = [
        ParsedMatchProps(
            bookmaker="sts",
            raw_team_a="T1",
            raw_team_b="Gen.G",
            map_number=1,
            lines=[
                ParsedPropLine(market_type="total_kills", line=25.5, odds_over=1.90, odds_under=1.85),
                ParsedPropLine(market_type="handicap_kills", line=-4.5, odds_cover_a=1.95, odds_cover_b=1.80),
            ],
            source_url=be_url,
        )
    ]

    summary = service.process_and_persist_props(scraped_props, db_session=sqlite_db, evaluate_models=False)
    assert summary["matched_to_canonical"] == 1
    assert summary["total_lines_persisted"] == 2

    # Query prop_odds_snapshots
    rows = sqlite_db.query(PropOddsSnapshot).filter_by(canonical_match_id=cm.id).all()
    assert len(rows) == 2
    types = {r.market_type for r in rows}
    assert "total_kills" in types
    assert "handicap_kills" in types


@pytest.mark.asyncio
async def test_expanded_props_and_outcomes_payload_persistence(sqlite_db):
    """Verify persistence of multi-outcome markets (correct_score) and expanded objectives."""
    import json
    from betting_app.models.match import CanonicalMatch
    from betting_app.models.bookmaker import Bookmaker
    from betting_app.models.odds import BookmakerEvent, PropOddsSnapshot

    cm = CanonicalMatch(
        id=1001,
        canonical_key="g2-vs-fnc-2026-10-01",
        team_a_name="G2 Esports",
        team_b_name="Fnatic",
        normalized_team_a="g2 esports",
        normalized_team_b="fnatic",
        status="upcoming",
        league="LEC",
        start_time_normalized="2026-10-01T17:00:00Z",
    )
    sqlite_db.add(cm)
    bm = sqlite_db.query(Bookmaker).filter_by(name="sts").first()
    if not bm:
        bm = Bookmaker(id=2, name="sts", base_url="https://www.sts.pl")
        sqlite_db.add(bm)
    sqlite_db.flush()

    be_url = "https://www.sts.pl/kursy/g2-fnatic/f8888"
    be = BookmakerEvent(
        bookmaker_id=bm.id,
        bookmaker_event_id="f8888",
        canonical_match_id=cm.id,
        raw_team_a="G2 Esports",
        raw_team_b="Fnatic",
        offer_url=be_url,
    )
    sqlite_db.add(be)
    sqlite_db.commit()

    score_distribution = {"2:0": 2.45, "2:1": 3.20, "1:2": 4.10, "0:2": 6.50}
    scraped_props = [
        ParsedMatchProps(
            bookmaker="sts",
            raw_team_a="G2 Esports",
            raw_team_b="Fnatic",
            map_number=0,
            lines=[
                ParsedPropLine(
                    market_type="correct_score",
                    line=0.0,
                    raw_market_name="Dokładny wynik meczu",
                    outcomes_payload=score_distribution,
                ),
                ParsedPropLine(
                    market_type="total_maps",
                    line=2.5,
                    odds_over=1.95,
                    odds_under=1.75,
                    raw_market_name="Liczba map (2.5)",
                ),
            ],
            source_url=be_url,
        ),
        ParsedMatchProps(
            bookmaker="sts",
            raw_team_a="G2 Esports",
            raw_team_b="Fnatic",
            map_number=1,
            lines=[
                ParsedPropLine(
                    market_type="first_dragon",
                    line=0.0,
                    odds_cover_a=1.80,
                    odds_cover_b=1.90,
                    raw_market_name="1. mapa - 1. smok",
                ),
                ParsedPropLine(
                    market_type="first_baron",
                    line=0.0,
                    odds_cover_a=1.75,
                    odds_cover_b=1.95,
                    raw_market_name="1. mapa - 1. baron",
                ),
            ],
            source_url=be_url,
        ),
    ]

    service = PropScraperService()
    summary = service.process_and_persist_props(scraped_props, db_session=sqlite_db, evaluate_models=False)
    assert summary["matched_to_canonical"] == 2
    assert summary["total_lines_persisted"] == 4

    # Query stored rows in prop_odds_snapshots
    rows = sqlite_db.query(PropOddsSnapshot).filter_by(canonical_match_id=cm.id).all()
    assert len(rows) == 4
    types = {r.market_type for r in rows}
    assert "correct_score" in types
    assert "total_maps" in types
    assert "first_dragon" in types
    assert "first_baron" in types

    # Check outcomes_payload on correct_score row
    cs_row = next(r for r in rows if r.market_type == "correct_score")
    assert cs_row.outcomes_payload is not None
    parsed_payload = json.loads(cs_row.outcomes_payload)
    assert parsed_payload["2:0"] == 2.45
    assert parsed_payload["0:2"] == 6.50
    assert cs_row.source_url == be_url
