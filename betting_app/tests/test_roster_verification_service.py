"""Tests for LoL Fandom / Liquipedia roster verification service and API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from betting_app.services.roster_verification_service import (
    FandomRosterClient,
    RosterVerificationService,
    VerifiedRosterResult,
    normalize_role,
)
from betting_app.api.main import app
from betting_app.api.deps import get_db


# --- Mock data fixtures ---

MOCK_FANDOM_PLAYERS_CARGO = [
    {"ID": "Doran", "Name": "Choi Hyeon-jun", "Team": "T1", "Role": "Top", "IsSubstitute": 0},
    {"ID": "Oner", "Name": "Mun Hyeon-jun", "Team": "T1", "Role": "Jungle", "IsSubstitute": 0},
    {"ID": "Faker", "Name": "Lee Sang-hyeok", "Team": "T1", "Role": "Mid", "IsSubstitute": 0},
    {"ID": "Peyz", "Name": "Kim Su-hwan", "Team": "T1", "Role": "Bot", "IsSubstitute": 0},
    {"ID": "Keria", "Name": "Ryu Min-seok", "Team": "T1", "Role": "Support", "IsSubstitute": 0},
    {"ID": "Tom", "Name": "Im Jae-hyeon", "Team": "T1", "Role": "Coach", "IsSubstitute": 0},
]

MOCK_FANDOM_SCOREBOARD_PLAYERS = [
    {"Name": "Doran", "Role": "Top", "DateTime_UTC": "2026-08-30 12:00:00"},
    {"Name": "Oner", "Role": "Jungle", "DateTime_UTC": "2026-08-30 12:00:00"},
    {"Name": "Faker", "Role": "Mid", "DateTime_UTC": "2026-08-30 12:00:00"},
    {"Name": "Peyz", "Role": "Bot", "DateTime_UTC": "2026-08-30 12:00:00"},
    {"Name": "Keria", "Role": "Support", "DateTime_UTC": "2026-08-30 12:00:00"},
]


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session with team_current_roster_players schema."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE team_current_roster_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name VARCHAR(200) NOT NULL,
                normalized_team_name VARCHAR(200) NOT NULL,
                source_match_id VARCHAR(100),
                source_game_id VARCHAR(100),
                source_match_date VARCHAR(50),
                source VARCHAR(50) DEFAULT 'auto',
                team_id VARCHAR(100),
                player_id VARCHAR(100) NOT NULL,
                player_name VARCHAR(200) NOT NULL,
                role VARCHAR(30) NOT NULL,
                updated_at DATETIME,
                UNIQUE (normalized_team_name, role)
            )
            """)
        )
        conn.execute(
            text("""
            CREATE TABLE canonical_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_a_name VARCHAR(200) NOT NULL,
                team_b_name VARCHAR(200) NOT NULL,
                start_time_normalized VARCHAR(50),
                status VARCHAR(50) DEFAULT 'upcoming'
            )
            """)
        )

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# --- Unit Tests: FandomRosterClient ---


def test_fandom_role_normalization():
    assert normalize_role("Top") == "TOP"
    assert normalize_role("Top Laner") == "TOP"
    assert normalize_role("Jungle") == "JUNGLE"
    assert normalize_role("Jungler") == "JUNGLE"
    assert normalize_role("Mid") == "MID"
    assert normalize_role("Middle") == "MID"
    assert normalize_role("Bot") == "ADC"
    assert normalize_role("AD Carry") == "ADC"
    assert normalize_role("Support") == "SUPPORT"
    assert normalize_role("Coach") is None
    assert normalize_role("Streamer") is None


def test_fandom_fetch_active_roster_success():
    client = FandomRosterClient()

    with patch.object(client, "query_cargo") as mock_cargo:
        def cargo_side_effect(table, *args, **kwargs):
            if table == "Teams":
                return [{"OverviewPage": "T1", "Name": "T1", "Short": "T1"}]
            if table == "Players":
                return MOCK_FANDOM_PLAYERS_CARGO
            if table == "ScoreboardPlayers":
                return MOCK_FANDOM_SCOREBOARD_PLAYERS
            return []

        mock_cargo.side_effect = cargo_side_effect

        result = client.fetch_active_roster("T1")

        assert result is not None
        assert result.team_name == "T1"
        assert result.source == "fandom"
        assert len(result.players) == 5
        roles = {p["role"] for p in result.players}
        assert roles == {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}

        top = next(p for p in result.players if p["role"] == "TOP")
        assert top["player_id"] == "Doran"
        assert top["player_name"] == "Doran"


def test_fandom_fetch_active_roster_handles_missing_team():
    client = FandomRosterClient()

    with patch.object(client, "query_cargo", return_value=[]):
        result = client.fetch_active_roster("NonExistentTeamXYZ")
        assert result is None


# --- Unit Tests: Roster Diffing & Comparison ---


def test_roster_comparison_identical():
    stored = [
        {"role": "TOP", "player_id": "Doran", "player_name": "Choi Hyeon-jun"},
        {"role": "JUNGLE", "player_id": "Oner", "player_name": "Mun Hyeon-jun"},
        {"role": "MID", "player_id": "Faker", "player_name": "Lee Sang-hyeok"},
        {"role": "ADC", "player_id": "Peyz", "player_name": "Kim Su-hwan"},
        {"role": "SUPPORT", "player_id": "Keria", "player_name": "Ryu Min-seok"},
    ]
    new = [
        {"role": "TOP", "player_id": "doran", "player_name": "Choi Hyeon-jun"},
        {"role": "JUNGLE", "player_id": "oner", "player_name": "Mun Hyeon-jun"},
        {"role": "MID", "player_id": "faker", "player_name": "Lee Sang-hyeok"},
        {"role": "ADC", "player_id": "peyz", "player_name": "Kim Su-hwan"},
        {"role": "SUPPORT", "player_id": "keria", "player_name": "Ryu Min-seok"},
    ]
    is_diff, changes = RosterVerificationService.compare_rosters(stored, new)
    assert not is_diff
    assert len(changes) == 0


def test_roster_comparison_substitution():
    stored = [
        {"role": "TOP", "player_id": "Zeus", "player_name": "Choi Woo-je"},
        {"role": "JUNGLE", "player_id": "Oner", "player_name": "Mun Hyeon-jun"},
        {"role": "MID", "player_id": "Faker", "player_name": "Lee Sang-hyeok"},
        {"role": "ADC", "player_id": "Gumayusi", "player_name": "Lee Min-hyeong"},
        {"role": "SUPPORT", "player_id": "Keria", "player_name": "Ryu Min-seok"},
    ]
    new = [
        {"role": "TOP", "player_id": "Doran", "player_name": "Choi Hyeon-jun"},
        {"role": "JUNGLE", "player_id": "Oner", "player_name": "Mun Hyeon-jun"},
        {"role": "MID", "player_id": "Faker", "player_name": "Lee Sang-hyeok"},
        {"role": "ADC", "player_id": "Peyz", "player_name": "Kim Su-hwan"},
        {"role": "SUPPORT", "player_id": "Keria", "player_name": "Ryu Min-seok"},
    ]
    is_diff, changes = RosterVerificationService.compare_rosters(stored, new)
    assert is_diff
    assert len(changes) == 2
    changed_roles = {c["role"] for c in changes}
    assert changed_roles == {"TOP", "ADC"}


# --- Integration Tests: Database Persistence & Service ---


def test_verify_team_roster_persists_to_db(sqlite_session):
    mock_fandom = MagicMock()
    mock_fandom.fetch_active_roster.return_value = VerifiedRosterResult(
        team_name="T1",
        source="fandom",
        players=[
            {"player_id": "Doran", "player_name": "Choi Hyeon-jun", "role": "TOP"},
            {"player_id": "Oner", "player_name": "Mun Hyeon-jun", "role": "JUNGLE"},
            {"player_id": "Faker", "player_name": "Lee Sang-hyeok", "role": "MID"},
            {"player_id": "Peyz", "player_name": "Kim Su-hwan", "role": "ADC"},
            {"player_id": "Keria", "player_name": "Ryu Min-seok", "role": "SUPPORT"},
        ],
        source_match_date="2026-08-30T12:00:00+00:00",
        notes="Fandom test",
    )

    service = RosterVerificationService(fandom_client=mock_fandom)

    # 1. Initial verification -> should update DB
    status, changes = service.verify_team_roster(sqlite_session, "T1", source="fandom")
    assert status == "updated"
    assert len(changes) == 5

    stored = service.get_stored_roster(sqlite_session, "T1")
    assert len(stored) == 5
    assert {p["role"] for p in stored} == {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}
    assert stored[0]["source"] == "fandom"

    # 2. Second verification -> should be up-to-date
    status2, changes2 = service.verify_team_roster(sqlite_session, "T1", source="fandom")
    assert status2 == "up_to_date"
    assert len(changes2) == 0


def test_verify_team_roster_handles_failure(sqlite_session):
    mock_fandom = MagicMock()
    mock_fandom.fetch_active_roster.return_value = None

    service = RosterVerificationService(fandom_client=mock_fandom)
    status, changes = service.verify_team_roster(sqlite_session, "Unknown Team", source="fandom")

    assert status == "failed"
    assert len(changes) == 0


# --- API Endpoint Tests ---


def test_api_rosters_status_endpoint(client):
    from betting_app.core.db import get_session
    with get_session() as s:
        s.execute(
            text("""
            INSERT INTO canonical_matches (canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b, start_time_normalized, status)
            VALUES ('t1_geng_20260906', 'T1', 'Gen.G', 't1', 'geng', '2026-09-06 12:00:00', 'upcoming')
            """)
        )
        s.commit()

    response = client.get("/matches/rosters/status")
    assert response.status_code == 200
    data = response.json()
    assert data["upcoming_matches_count"] >= 1
    assert "T1" in data["teams_missing"] or "T1" in [t["team_name"] for t in data["teams_verified"]]


def test_api_rosters_verify_endpoint_dry_run(client):
    with patch("betting_app.api.routers.matches.RosterVerificationService") as MockServiceCls:
        instance = MockServiceCls.return_value
        instance.verify_and_sync_rosters.return_value = {
            "total_teams": 1,
            "updated_count": 1,
            "up_to_date_count": 0,
            "failed_count": 0,
            "updated": [{"team_name": "T1", "changes": []}],
            "up_to_date": [],
            "failed": [],
        }

        response = client.post(
            "/matches/rosters/verify",
            json={"team_names": ["T1"], "dry_run": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_teams"] == 1
        assert data["updated_count"] == 1

def test_roster_collection_deduplicates_explicit_targets(monkeypatch):
    from types import SimpleNamespace

    fetched = []
    def fetch(team):
        fetched.append(team)
        return None

    service = RosterVerificationService(fandom_client=SimpleNamespace(fetch_active_roster=fetch))
    monkeypatch.setattr(service, "get_stored_roster", lambda *_args: [])
    summary = service.verify_and_sync_rosters(
        team_names=["T1", "T1", " T1 ", "", "G2"],
        session=SimpleNamespace(), source="fandom", dry_run=True,
        delay_between_requests=0,
    )
    assert fetched == ["T1", "G2"]
    assert summary["total_teams"] == 2
    assert summary["failed_count"] == 2
