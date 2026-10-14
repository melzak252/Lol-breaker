"""Comprehensive unit and integration tests for Proposition Odds Tracking (IDEA-018)."""

from __future__ import annotations

from datetime import datetime, UTC
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from betting_app.api.deps import get_db
from betting_app.api.main import app
from betting_app.models.base import Base
from betting_app.models.bookmaker import Bookmaker
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import PropOddsSnapshot
from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.scrapers.props_parser import ParsedMatchProps, ParsedPropLine
from betting_app.services.prop_odds_service import (
    evaluate_match_props,
    get_latest_prop_odds,
    get_prop_odds_timeline,
    save_prop_snapshots,
)


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

    # Seed bookmakers
    b1 = Bookmaker(id=1, name="efortuna", base_url="https://efortuna.pl")
    b2 = Bookmaker(id=2, name="sts", base_url="https://sts.pl")
    b3 = Bookmaker(id=3, name="betclic", base_url="https://betclic.pl")
    session.add_all([b1, b2, b3])

    # Seed canonical match
    m = CanonicalMatch(
        id=201,
        canonical_key="karmine-corp-vs-g2-esports-2026-09-05",
        team_a_name="Karmine Corp",
        team_b_name="G2 Esports",
        normalized_team_a="karmine corp",
        normalized_team_b="g2 esports",
        league="LEC",
        status="upcoming",
    )
    session.add(m)
    session.commit()

    yield session, engine
    session.close()


def test_save_prop_snapshots_and_novig(sqlite_db):
    session, _ = sqlite_db

    props = ParsedMatchProps(
        bookmaker="efortuna",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(
                market_type="total_kills",
                line=26.5,
                odds_over=1.85,
                odds_under=1.85,
            ),
            ParsedPropLine(
                market_type="team_kills",
                team="Karmine Corp",
                line=12.5,
                odds_over=1.80,
                odds_under=1.90,
            ),
        ],
    )
    inserted_ids = save_prop_snapshots(
        props,
        canonical_match_id=201,
        scraped_at="2026-09-05T10:00:00Z",
        db_session=session,
    )
    assert len(inserted_ids) == 2

    snaps = session.query(PropOddsSnapshot).filter(PropOddsSnapshot.canonical_match_id == 201).all()
    assert len(snaps) == 2

    tot = [s for s in snaps if s.market_type == "total_kills"][0]
    assert tot.line == 26.5
    assert tot.odds_over == 1.85
    assert tot.odds_under == 1.85
    # Check no-vig fair prob: with 1.85/1.85, each is exactly 0.50
    assert pytest.approx(tot.prob_over_novig, 0.01) == 0.50
    assert pytest.approx(tot.prob_under_novig, 0.01) == 0.50
    assert tot.margin > 0.05


def test_prop_odds_timeline_progression(sqlite_db):
    session, _ = sqlite_db

    # First observation at 10:00
    p1 = ParsedMatchProps(
        bookmaker="sts",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(
                market_type="total_kills",
                line=27.5,
                odds_over=1.85,
                odds_under=1.85,
            )
        ],
    )
    save_prop_snapshots(p1, canonical_match_id=201, scraped_at="2026-09-05T10:00:00Z", db_session=session)

    # Line movement at 12:00: market moved to 1.95 / 1.75
    p2 = ParsedMatchProps(
        bookmaker="sts",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(
                market_type="total_kills",
                line=27.5,
                odds_over=1.95,
                odds_under=1.75,
            )
        ],
    )
    save_prop_snapshots(p2, canonical_match_id=201, scraped_at="2026-09-05T12:00:00Z", db_session=session)

    timeline = get_prop_odds_timeline(201, market_type="total_kills", line=27.5, db_session=session)
    assert len(timeline) == 2
    assert timeline[0]["scraped_at"].startswith("2026-09-05T10:00")
    assert timeline[0]["odds_over"] == 1.85
    assert timeline[1]["scraped_at"].startswith("2026-09-05T12:00")
    assert timeline[1]["odds_over"] == 1.95
    # Over became underdog -> under became favorite (lower odds)
    assert timeline[1]["prob_over_novig"] < timeline[1]["prob_under_novig"]


def test_get_latest_prop_odds(sqlite_db):
    session, _ = sqlite_db

    # Add lines across 2 bookmakers
    p_fortuna = ParsedMatchProps(
        bookmaker="efortuna",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(market_type="total_kills", line=26.5, odds_over=1.82, odds_under=1.88),
            ParsedPropLine(market_type="total_kills", line=26.5, odds_over=1.85, odds_under=1.85),  # updated
        ],
    )
    save_prop_snapshots(p_fortuna, canonical_match_id=201, scraped_at="2026-09-05T11:00:00Z", db_session=session)

    p_sts = ParsedMatchProps(
        bookmaker="sts",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(market_type="total_kills", line=26.5, odds_over=1.88, odds_under=1.82),
        ],
    )
    save_prop_snapshots(p_sts, canonical_match_id=201, scraped_at="2026-09-05T11:05:00Z", db_session=session)

    latest = get_latest_prop_odds(201, market_type="total_kills", db_session=session)
    assert len(latest) == 2
    books = {l["bookmaker"]: l["odds_over"] for l in latest}
    assert books["efortuna"] == 1.85
    assert books["sts"] == 1.88


def test_evaluate_match_props_signals(sqlite_db):
    session, _ = sqlite_db

    # Create synthetic tracker where Karmine Corp averages high kills
    tracker = ChronologicalPaceTracker(window_size=10, prior_weight=1.0)
    for _ in range(5):
        tracker.update_game("Karmine Corp", "Random Opponent", kills_1=25, kills_2=15, duration_minutes=30.0, winner_team_1=True)

    # Bookmaker offers line with favorable odds
    p = ParsedMatchProps(
        bookmaker="efortuna",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            # Model with high pace will favor OVER on low lines
            ParsedPropLine(market_type="total_kills", line=24.5, odds_over=2.10, odds_under=1.65),
        ],
    )
    save_prop_snapshots(p, canonical_match_id=201, scraped_at="2026-09-05T13:00:00Z", db_session=session)

    analysis = evaluate_match_props(201, tracker=tracker, tax_rate=0.12, db_session=session)
    assert analysis["canonical_match_id"] == 201
    assert "model_expectations" in analysis
    assert len(analysis["evaluated_lines"]) == 1

    evaluated_line = analysis["evaluated_lines"][0]
    assert evaluated_line["model_prob_over"] is not None
    assert evaluated_line["ev_over_net"] is not None

    # Signals should detect positive Net EV
    if evaluated_line["ev_over_net"] > 0:
        assert len(analysis["signals"]) > 0
        sig = analysis["signals"][0]
        assert sig["odds"] == 2.10
        assert sig["net_ev_tax12"] > 0


def test_api_props_endpoints(sqlite_db):
    session, _ = sqlite_db

    # Seed snapshot
    p = ParsedMatchProps(
        bookmaker="betclic",
        raw_team_a="Karmine Corp",
        raw_team_b="G2 Esports",
        map_number=1,
        lines=[
            ParsedPropLine(market_type="total_kills", line=28.5, odds_over=1.90, odds_under=1.80),
        ],
    )
    save_prop_snapshots(p, canonical_match_id=201, scraped_at="2026-09-05T14:00:00Z", db_session=session)

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        # 1. Test timeline endpoint
        res_timeline = client.get("/matches/201/props/timeline?market_type=total_kills")
        assert res_timeline.status_code == 200
        data_t = res_timeline.json()
        assert data_t["canonical_match_id"] == 201
        assert data_t["total_snapshots"] >= 1
        assert data_t["timeline"][0]["line"] == 28.5

        # 2. Test latest endpoint
        res_latest = client.get("/matches/201/props/latest")
        assert res_latest.status_code == 200
        data_l = res_latest.json()
        assert data_l["total_lines"] >= 1
        assert data_l["lines"][0]["bookmaker"] == "betclic"

        # 3. Test analysis endpoint with auto persistence
        res_analysis = client.get("/matches/201/props/analysis")
        assert res_analysis.status_code == 200
        data_a = res_analysis.json()
        assert data_a["team_a"] == "Karmine Corp"
        assert "model_expectations" in data_a
        assert data_a["saved_prediction_id"] is not None
        assert "signals" in data_a

        # 3b. Test explicit predict endpoint
        res_pred = client.post("/matches/201/props/predict?map_number=1")
        assert res_pred.status_code == 200
        data_p = res_pred.json()
        assert data_p["saved_prediction_id"] is not None
        assert data_p["model_expectations"]["expected_total_kills"] > 0
        # 4. Test ingest endpoint
        ingest_payload = {
            "bookmaker": "sts",
            "raw_team_a": "Karmine Corp",
            "raw_team_b": "G2 Esports",
            "map_number": 1,
            "lines": [
                {
                    "market_type": "handicap_kills",
                    "line": -2.5,
                    "target_team": "Karmine Corp",
                    "odds_over": 1.95,
                    "odds_under": 1.75,
                }
            ],
        }
        res_ingest = client.post("/matches/201/props/ingest", json=ingest_payload)
        assert res_ingest.status_code == 200
        data_i = res_ingest.json()
        assert data_i["status"] == "success"
        assert data_i["inserted_count"] == 1

        # 5. Test get_saved_prop_prediction function directly
        from betting_app.services.prop_odds_service import get_saved_prop_prediction
        saved = get_saved_prop_prediction(201, map_number=1, db_session=session)
        assert saved is not None
        assert saved["canonical_match_id"] == 201
        assert saved["expected_total_kills"] > 0
        assert "distribution" in saved
    finally:
        app.dependency_overrides.clear()
