"""Tests for /api/matches endpoints."""

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import text

from betting_app.core.db import get_session


def _bookmaker_id(session, name: str) -> int:
    bm = session.execute(text("SELECT id FROM bookmakers WHERE name=:name"), {"name": name}).fetchone()
    assert bm is not None
    return int(bm[0])


def _seed_single_match() -> int:
    """Insert one canonical match + one odds_snapshot using SQLAlchemy, return canonical ID."""
    session = get_session()
    bm_id = _bookmaker_id(session, "manual")

    session.execute(
        text("""
        INSERT INTO canonical_matches (id, canonical_key, team_a_name, team_b_name,
                                       normalized_team_a, normalized_team_b,
                                       start_time_normalized, league, status, match_confidence)
        VALUES (1, 'test-key-1', 'Fnatic', 'G2 Esports',
                'fnatic', 'g2', '2026-12-31T12:00:00+00:00', 'LEC', 'upcoming', 1.0)
        ON CONFLICT (id) DO NOTHING
        """),
    )
    session.execute(
        text("""
        INSERT INTO odds_snapshots (bookmaker_id, market_type, is_live, canonical_match_id,
                                    raw_team_a, raw_team_b, odds_a, odds_b, scraped_at, source_url)
        VALUES (:bm, 'match_winner', 0, 1, 'Fnatic', 'G2 Esports', 2.0, 1.8, NOW(), 'https://x.pl/')
        """),
        {"bm": bm_id},
    )
    session.execute(
        text("""
        INSERT INTO upcoming_matches (bookmaker_id, bookmaker_match_key, canonical_match_id,
                                      raw_team_a, raw_team_b, normalized_team_a, normalized_team_b,
                                      last_seen_at)
        VALUES (:bm, 'test-bm-key-1', 1, 'Fnatic', 'G2 Esports', 'fnatic', 'g2', NOW())
        ON CONFLICT (bookmaker_match_key) DO UPDATE SET last_seen_at = EXCLUDED.last_seen_at
        """),
        {"bm": bm_id},
    )
    session.commit()
    session.close()
    return 1


def _seed_match_with_stale_best_odds() -> int:
    """Seed one match where a stale bookmaker has better odds than a fresh one."""
    match_id = _seed_single_match()
    session = get_session()
    sts_id = _bookmaker_id(session, "sts")
    session.execute(
        text("""
        INSERT INTO odds_snapshots (bookmaker_id, market_type, is_live, canonical_match_id,
                                    raw_team_a, raw_team_b, odds_a, odds_b, scraped_at, source_url)
        VALUES (:bm, 'match_winner', 0, 1, 'Fnatic', 'G2 Esports', 9.0, 9.0, NOW() - INTERVAL '25 hours', 'https://stale.pl/')
        """),
        {"bm": sts_id},
    )
    session.commit()
    session.close()
    return match_id


class TestMatchList:
    def test_empty_returns_zero(self, client: TestClient):
        resp = client.get("/matches")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["matches"] == []

    def test_single_seeded(self, client: TestClient):
        _seed_single_match()
        resp = client.get("/matches?min_books=1&days_ahead=365")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        item = data["matches"][0]
        assert item["match"] == "Fnatic vs G2 Esports"
        assert item["bookmaker_count"] == 1
        assert item["best_odds_a"] == 2.0
        assert item["best_odds_b"] == 1.8
        assert item["quote_status"] == "actionable"
        assert item["diagnostic_best_odds_a"] == 2.0
        assert item["diagnostic_best_odds_b"] == 1.8
        for field in ("model_prob_a", "model_prob_b", "hybrid_prob_a", "hybrid_prob_b",
                      "hybrid_ev_a", "hybrid_ev_b", "recommended_side"):
            assert item[field] is None

    def test_min_books_filter(self, client: TestClient):
        _seed_single_match()
        resp = client.get("/matches?min_books=2&days_ahead=365")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_ignores_odds_older_than_24h(self, client: TestClient):
        _seed_match_with_stale_best_odds()
        resp = client.get("/matches?min_books=1&days_ahead=365")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        item = data["matches"][0]
        assert item["bookmaker_count"] == 1
        assert item["best_odds_a"] == 2.0
        assert item["best_odds_b"] == 1.8
        assert item["best_bookmaker_a"] == "manual"
        assert item["best_bookmaker_b"] == "manual"

    def test_can_override_max_odds_age_for_diagnostics(self, client: TestClient):
        _seed_match_with_stale_best_odds()
        resp = client.get("/matches?min_books=1&days_ahead=365&max_odds_age_hours=48")
        assert resp.status_code == 200
        item = resp.json()["matches"][0]
        assert item["bookmaker_count"] == 2
        assert item["diagnostic_best_odds_a"] == 9.0
        assert item["diagnostic_best_odds_b"] == 9.0
        assert item["best_odds_a"] == 2.0
        assert item["best_odds_b"] == 1.8
        assert item["best_bookmaker_a"] == item["best_bookmaker_b"] == "manual"
        assert item["hybrid_prob_a"] is None
        assert item["hybrid_ev_a"] is None
        assert item["recommended_side"] is None

    def test_alignment_uses_canonical_names_not_normalized_cache(self, client: TestClient):
        _seed_single_match()
        with get_session() as session:
            session.execute(text("""
                UPDATE canonical_matches
                SET normalized_team_a='team-a', normalized_team_b='team-b' WHERE id=1
            """))
            session.commit()
        response = client.get("/matches?min_books=1&days_ahead=365")
        assert response.status_code == 200
        assert response.json()["total"] == 1
        item = response.json()["matches"][0]
        assert item["best_odds_a"] == 2.0
        assert item["best_odds_b"] == 1.8

    def test_quote_only_match_remains_visible_without_actionable_forecast(self, client: TestClient):
        _seed_single_match()
        with get_session() as session:
            session.execute(text("UPDATE odds_snapshots SET scraped_at=NOW() - INTERVAL '1 hour'"))
            session.commit()
        response = client.get("/matches?min_books=1&days_ahead=365")
        assert response.status_code == 200
        assert response.json()["total"] == 1
        item = response.json()["matches"][0]
        assert item["quote_status"] == "diagnostic"
        assert item["diagnostic_best_odds_a"] == 2.0
        assert item["diagnostic_best_odds_b"] == 1.8
        for field in ("best_odds_a", "best_odds_b", "model_prob_a", "model_prob_b",
                      "hybrid_prob_a", "hybrid_prob_b", "hybrid_ev_a", "hybrid_ev_b",
                      "recommended_side", "recommended_stake_a", "recommended_stake_b"):
            assert item[field] is None
        assert not item["arb_after_tax"]

    @pytest.mark.parametrize("clock", ["NULL", "NOW() + INTERVAL '1 hour'", "NOW() - INTERVAL '49 hours'"])
    def test_diagnostic_age_override_does_not_admit_unknown_future_or_older_quotes(
        self, client: TestClient, clock: str,
    ):
        _seed_single_match()
        with get_session() as session:
            session.execute(text(f"UPDATE odds_snapshots SET scraped_at={clock}"))
            session.commit()
        board = client.get("/matches?min_books=1&days_ahead=365&max_odds_age_hours=48")
        detail = client.get("/matches/1?max_odds_age_hours=48")
        assert board.status_code == detail.status_code == 200
        assert board.json()["matches"] == []
        assert detail.json()["odds"] == []

    @pytest.mark.parametrize("withdrawal", ["unknown_clock", "live", "missing_price"])
    def test_latest_unavailable_observation_does_not_revive_previous_quote(
        self, client: TestClient, withdrawal: str,
    ):
        _seed_single_match()
        with get_session() as session:
            session.execute(text("""
                INSERT INTO odds_snapshots (bookmaker_id, market_type, is_live, canonical_match_id,
                                            raw_team_a, raw_team_b, odds_a, odds_b, scraped_at, source_url)
                VALUES (:bm, 'match_winner', :live, 1, 'Fnatic', 'G2 Esports',
                        :odds_a, 1.8, :clock, 'https://x.pl/')
            """), {
                "bm": _bookmaker_id(session, "manual"),
                "live": 1 if withdrawal == "live" else 0,
                "odds_a": None if withdrawal == "missing_price" else 2.0,
                "clock": None if withdrawal == "unknown_clock" else session.execute(text("SELECT NOW()")).scalar_one(),
            })
            session.commit()
        board = client.get("/matches?min_books=1&days_ahead=365")
        detail = client.get("/matches/1")
        assert board.status_code == detail.status_code == 200
        assert board.json()["matches"] == []
        rows = detail.json()["odds"]
        if withdrawal == "unknown_clock":
            assert rows == []
        else:
            assert len(rows) == 1
            assert rows[0]["quote_status"] == "unavailable"
            assert rows[0]["ev_a"] is None and rows[0]["ev_b"] is None


class TestMatchDetail:
    def test_not_found(self, client: TestClient):
        resp = client.get("/matches/999")
        assert resp.status_code == 404

    def test_found(self, client: TestClient):
        _seed_single_match()
        resp = client.get("/matches/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["canonical_match_id"] == 1
        assert data["team_a_name"] == "Fnatic"
        assert len(data["odds"]) == 1
        assert data["odds"][0]["bookmaker"] == "manual"

    def test_detail_ignores_odds_older_than_24h(self, client: TestClient):
        _seed_match_with_stale_best_odds()
        resp = client.get("/matches/1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["odds"]) == 1
        assert data["odds"][0]["bookmaker"] == "manual"
        assert data["odds"][0]["canonical_odds_a"] == 2.0

    def test_detail_age_override_is_diagnostic_not_actionable(self, client: TestClient):
        _seed_match_with_stale_best_odds()
        response = client.get("/matches/1?max_odds_age_hours=48")
        assert response.status_code == 200
        rows = {row["bookmaker"]: row for row in response.json()["odds"]}
        assert set(rows) == {"manual", "sts"}
        assert rows["manual"]["quote_status"] == "actionable"
        stale = rows["sts"]
        assert stale["quote_status"] == "diagnostic"
        assert stale["canonical_odds_a"] == stale["canonical_odds_b"] == 9.0
        assert stale["ev_a"] is None and stale["ev_b"] is None
        assert stale["kelly_a"] is None and stale["kelly_b"] is None


class TestOddsHistory:
    def test_history_not_found(self, client: TestClient):
        resp = client.get("/matches/999/odds-history")
        assert resp.status_code == 404

    def test_history_returns_rows(self, client: TestClient):
        _seed_single_match()
        resp = client.get("/matches/1/odds-history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
