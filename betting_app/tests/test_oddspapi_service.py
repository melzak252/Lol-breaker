"""Unit and integration tests for OddsPapi service, budget guard, and market comparison."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from betting_app.core.db import get_session
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import OddsSnapshot
from betting_app.models.oddspapi import OddspapiFixtureMapping, OddspapiRequestLog
from betting_app.models.prediction import CanonicalPrediction
from betting_app.services.odds_service import get_or_create_bookmaker
from betting_app.services.prematch_quote_service import load_prematch_quotes
from betting_app.services.oddspapi_service import (
    OddsPapiBudgetExhaustedError,
    OddsPapiBudgetGuard,
    OddsPapiClient,
    compare_match_market,
    fetch_pinnacle_horizon_odds,
    sync_oddspapi_fixtures,
)
from src.analysis.quote_provenance import extract_winner_quote
from betting_app.services.forecast_evidence import load_source_capture

class MockOddsPapiClient(OddsPapiClient):
    """Mock client returning predetermined payloads without external network calls."""

    def __init__(self, responses: dict[str, Any] | None = None, budget_guard: OddsPapiBudgetGuard | None = None) -> None:
        super().__init__(api_key="test-key", budget_guard=budget_guard or OddsPapiBudgetGuard())
        self.responses = responses or {}
        self.requested_paths: list[tuple[str, dict[str, Any] | None]] = []

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        session: Session | None = None,
        fixture_id: str | None = None,
    ) -> Any:
        sess = session or get_session()
        allowed, reason, _ = self.budget_guard.check_budget(sess)
        if not allowed:
            raise OddsPapiBudgetExhaustedError(reason)

        self.requested_paths.append((path, params))
        self.budget_guard.record_request(sess, path, fixture_id, 200, 15)

        if path in self.responses:
            return self.responses[path]
        if fixture_id and f"fixture:{fixture_id}" in self.responses:
            return self.responses[f"fixture:{fixture_id}"]
        return self.responses.get("default", {})


def test_budget_guard_daily_and_monthly_caps(client: TestClient) -> None:
    session = get_session()
    try:
        guard = OddsPapiBudgetGuard(monthly_limit=250, daily_limit=8)

        # Initially allowed
        allowed, reason, stats = guard.check_budget(session)
        assert allowed is True
        assert stats["monthly_used"] == 0
        assert stats["daily_used"] == 0
        assert stats["daily_remaining"] == 8

        # Record 7 requests
        for i in range(7):
            guard.record_request(session, "/v4/odds", f"fix_{i}", 200, 20)

        allowed, _, stats = guard.check_budget(session)
        assert allowed is True
        assert stats["daily_used"] == 7
        assert stats["daily_remaining"] == 1

        # 8th request reaches daily cap
        guard.record_request(session, "/v4/odds", "fix_7", 200, 20)
        allowed, reason, stats = guard.check_budget(session)
        assert allowed is False
        assert "daily cap reached" in reason
        assert stats["daily_used"] == 8
        assert stats["daily_remaining"] == 0

        # Now test monthly limit
        small_monthly_guard = OddsPapiBudgetGuard(monthly_limit=10, daily_limit=50)
        # We currently have 8 requests in the DB
        allowed, _, stats = small_monthly_guard.check_budget(session)
        assert allowed is True
        assert stats["monthly_remaining"] == 2

        guard.record_request(session, "/v4/odds", "fix_8", 200, 20)
        guard.record_request(session, "/v4/odds", "fix_9", 200, 20)

        allowed, reason, stats = small_monthly_guard.check_budget(session)
        assert allowed is False
        assert "monthly quota reached" in reason
    finally:
        session.close()


def test_extract_winner_quote_live_and_historical() -> None:
    live_payload = {
        "fixtureId": "id123",
        "bookmakerOdds": {
            "pinnacle": {
                "markets": {
                    "181": {
                        "outcomes": {
                            "181": {"players": {"0": {"price": 1.75, "active": True, "createdAt": "2026-06-01T12:00:00Z"}}},
                            "182": {"players": {"0": {"price": 2.15, "active": True, "createdAt": "2026-06-01T12:00:00Z"}}},
                        }
                    }
                }
            }
        },
    }

    live_quote = extract_winner_quote(live_payload, "pinnacle")
    assert live_quote is not None
    assert live_quote["odds_1"] == 1.75
    assert live_quote["odds_2"] == 2.15
    assert live_quote["quote_at_1"] == "2026-06-01T12:00:00Z"
    assert live_quote["participant_binding_status"] == "unverified_provider_outcome_convention"

    historical_payload = {
        "fixtureId": "id123",
        "bookmakers": {
            "pinnacle": {
                "markets": {
                    "181": {
                        "outcomes": {
                            "181": {
                                "players": {
                                    "0": [
                                        {"price": 1.90, "active": True, "createdAt": "2026-06-01T10:00:00Z"},
                                        {"price": 1.80, "active": True, "createdAt": "2026-06-01T11:00:00Z"},
                                        {"price": 1.70, "active": True, "createdAt": "2026-06-01T14:00:00Z"},
                                    ]
                                }
                            },
                            "182": {
                                "players": {
                                    "0": [
                                        {"price": 1.95, "active": True, "createdAt": "2026-06-01T10:00:00Z"},
                                        {"price": 2.05, "active": True, "createdAt": "2026-06-01T11:00:00Z"},
                                        {"price": 2.20, "active": True, "createdAt": "2026-06-01T14:00:00Z"},
                                    ]
                                }
                            },
                        }
                    }
                }
            }
        },
    }

    cutoff_quote = extract_winner_quote(
        historical_payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert cutoff_quote is not None
    assert cutoff_quote["odds_1"] == 1.80
    assert cutoff_quote["odds_2"] == 2.05

    latest_quote = extract_winner_quote(historical_payload, "pinnacle")
    assert latest_quote is not None
    assert latest_quote["odds_1"] == 1.70
    assert latest_quote["odds_2"] == 2.20


def test_sync_oddspapi_fixtures(client: TestClient) -> None:
    session = get_session()
    try:
        # Seed canonical match
        now = datetime.now(UTC)
        start_str = (now + timedelta(days=2)).replace(microsecond=0).isoformat()

        session.execute(
            text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b, start_time_normalized,
                league, status, match_confidence
            ) VALUES (
                101, 'lck-t1-geng', 'T1', 'Gen.G',
                't1', 'geng', :start_time,
                'LCK', 'upcoming', 1.0
            )
            """),
            {"start_time": start_str},
        )
        session.commit()

        # Mock OddsPapi /fixtures response
        mock_fixtures = [
            {
                "fixtureId": "fix_t1_geng_101",
                "participant1Name": "T1 Esports",
                "participant2Name": "Gen.G Esports",
                "startTime": start_str,
                "tournamentName": "LCK Champions",
                "hasOdds": True,
            },
            {
                "fixtureId": "fix_other_match",
                "participant1Name": "Fnatic",
                "participant2Name": "G2 Esports",
                "startTime": start_str,
                "tournamentName": "LEC",
                "hasOdds": True,
            },
        ]

        mock_client = MockOddsPapiClient(responses={"/fixtures": mock_fixtures})
        result = sync_oddspapi_fixtures(session=session, client=mock_client)

        assert result["status"] == "success"
        assert result["fixtures_synced"] == 2
        assert result["mapped_to_canonical"] == 1

        # Verify mapping in DB
        mapping = session.scalar(
            text("SELECT fixture_id, canonical_match_id, provider_team_1_is_a FROM oddspapi_fixture_mappings WHERE fixture_id='fix_t1_geng_101'")
        )
        assert mapping is not None

        mapping_row = session.execute(
            text("SELECT canonical_match_id, provider_team_1_is_a FROM oddspapi_fixture_mappings WHERE fixture_id='fix_t1_geng_101'")
        ).fetchone()
        assert mapping_row[0] == 101
        assert mapping_row[1] == 1  # T1 was team A
    finally:
        session.close()


def test_fetch_pinnacle_horizon_odds_quarantines_unverified_binding_without_old_price_fallback(client: TestClient) -> None:
    session = get_session()
    try:
        now = datetime.now(UTC)
        match_start = now + timedelta(hours=6)
        match_start_str = match_start.replace(microsecond=0).isoformat()

        session.execute(
            text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b, start_time_normalized,
                league, status, match_confidence
            ) VALUES (
                202, 'lec-fnc-g2', 'Fnatic', 'G2 Esports',
                'fnatic', 'g2 esports', :start_time,
                'LEC', 'upcoming', 1.0
            )
            """),
            {"start_time": match_start_str},
        )

        mapping = OddspapiFixtureMapping(
            fixture_id="fix_fnc_g2_202",
            canonical_match_id=202,
            sport_id=18,
            league="LEC",
            provider_team_1="G2 Esports",
            provider_team_2="Fnatic",
            provider_team_1_is_a=0,
            start_time=match_start,
            has_odds=1,
        )
        session.add(mapping)
        session.commit()

        pinnacle_id = get_or_create_bookmaker("pinnacle")
        session.add(OddsSnapshot(
            canonical_match_id=202,
            bookmaker_id=pinnacle_id,
            market_type="match_winner",
            raw_team_a="Fnatic",
            raw_team_b="G2 Esports",
            odds_a=2.60,
            odds_b=1.55,
            is_live=0,
            scraped_at=now - timedelta(hours=2, minutes=1),
        ))
        session.commit()
        prior_context = load_prematch_quotes(
            [202], decision_at=now, max_age_seconds=3 * 3600, session=session,
        )[202]
        assert len(prior_context.quotes) == 1
        assert (prior_context.quotes[0].odds_a, prior_context.quotes[0].odds_b) == (2.60, 1.55)

        odds_payload = {
            "fixtureId": "fix_fnc_g2_202",
            "isLive": False,
            "bookmakerOdds": {
                "pinnacle": {
                    "markets": {
                        "181": {
                            "outcomes": {
                                "181": {"players": {"0": {"price": 1.50, "active": True, "createdAt": now.isoformat()}}},
                                "182": {"players": {"0": {"price": 2.70, "active": True, "createdAt": now.isoformat()}}},
                            }
                        }
                    }
                }
            },
        }

        mock_client = MockOddsPapiClient(responses={"fixture:fix_fnc_g2_202": odds_payload})
        result = fetch_pinnacle_horizon_odds(
            session=session,
            target_horizon_hours=6.0,
            tolerance_hours=1.0,
            max_requests=2,
            client=mock_client,
        )

        assert result["status"] == "success"
        assert result["candidates"] == 1
        assert result["fetched"] == 1
        assert result["saved"] == 1
        assert result["admitted"] == 0
        assert result["quarantined"] == 1

        snapshot = session.scalars(
            select(OddsSnapshot).where(
                OddsSnapshot.canonical_match_id == 202,
                OddsSnapshot.bookmaker_id == pinnacle_id,
            ).order_by(OddsSnapshot.scraped_at.desc())
        ).first()
        assert snapshot is not None
        assert snapshot.odds_a is None and snapshot.odds_b is None
        assert (snapshot.raw_team_a, snapshot.raw_team_b) == ("G2 Esports", "Fnatic")
        assert snapshot.is_live == 0
        persisted = json.loads(snapshot.raw_payload)
        observed_at = datetime.fromisoformat(persisted["response_observed_at"])
        assert snapshot.scraped_at == observed_at
        assert observed_at > now
        assert persisted["provider_payload"] == odds_payload
        assert persisted["participant_bindings"] == []
        provenance = persisted["quote_provenance"]
        assert provenance["quote_at_1"] == now.isoformat()
        assert provenance["quote_at_2"] == now.isoformat()
        assert provenance["quote_admission_status"] == "unverified_provider_outcome_convention"
        assert provenance["participant_binding_status"] == "unverified_provider_outcome_convention"
        assert provenance["participant_binding_evidence"] == []
        assert provenance["participant_1_id"] is None and provenance["participant_2_id"] is None
        assert provenance["participant_1_name"] is None and provenance["participant_2_name"] is None
        assert provenance["outcome_fixture_participant_indexes"] == {}
        assert provenance["execution_eligibility"] == "not_certified"
        assert provenance["state_1"]["status"] == provenance["state_2"]["status"] == "active"
        assert (provenance["odds_1"], provenance["odds_2"]) == (1.50, 2.70)

        context = load_prematch_quotes(
            [202], decision_at=observed_at, max_age_seconds=3 * 3600, session=session,
        )[202]
        assert context.quotes == ()
        assert context.prob_a is None
        assert len(context.rejections) == 1
        assert context.rejections[0].odds_snapshot_id == snapshot.id
        assert context.rejections[0].reason == "quote_price_invalid"
    finally:
        session.close()


def test_compare_match_market_and_api_endpoint(client: TestClient) -> None:
    session = get_session()
    try:
        now = datetime.now(UTC)
        match_start = now + timedelta(hours=6)
        match_start_str = match_start.replace(microsecond=0).isoformat()

        session.execute(
            text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b, start_time_normalized,
                league, status, match_confidence
            ) VALUES (
                303, 'lck-t1-dk', 'T1', 'Dplus KIA',
                't1', 'dplus kia', :start_time,
                'LCK', 'upcoming', 1.0
            )
            """),
            {"start_time": match_start_str},
        )

        pinnacle_id = get_or_create_bookmaker("pinnacle")
        sts_id = get_or_create_bookmaker("sts")

        quote_time = (now - timedelta(minutes=10)).replace(microsecond=0).isoformat()

        # Pinnacle: 1.80 vs 2.10
        session.execute(
            text("""
            INSERT INTO odds_snapshots (
                bookmaker_id, canonical_match_id, market_type, raw_team_a, raw_team_b,
                odds_a, odds_b, is_live, scraped_at, source_url
            ) VALUES (
                :bm_pin, 303, 'match_winner', 'T1', 'Dplus KIA',
                1.80, 2.10, 0, :scraped_at, 'oddspapi://test'
            )
            """),
            {"bm_pin": pinnacle_id, "scraped_at": quote_time},
        )

        # STS: 1.75 vs 2.15
        session.execute(
            text("""
            INSERT INTO odds_snapshots (
                bookmaker_id, canonical_match_id, market_type, raw_team_a, raw_team_b,
                odds_a, odds_b, is_live, scraped_at, source_url
            ) VALUES (
                :bm_sts, 303, 'match_winner', 'T1', 'Dplus KIA',
                1.75, 2.15, 0, :scraped_at, 'sts://test'
            )
            """),
            {"bm_sts": sts_id, "scraped_at": quote_time},
        )

        # Model prediction: prob_a = 0.58
        session.execute(
            text("""
            INSERT INTO canonical_predictions (
                id, canonical_match_id, model_name, model_version, features_version,
                prob_a, prob_b, predicted_at
            ) VALUES (
                999, 303, 'Sym-Cal LR-ElasticNet-W20-Binomial', 'exp-039', 'w20_v1',
                0.58, 0.42, :pred_at
            )
            """),
            {"pred_at": quote_time},
        )
        session.commit()

        # Test service function
        comparison = compare_match_market(canonical_match_id=303, session=session, horizon_hours=6.0)
        assert comparison is not None
        assert comparison["team_a"] == "T1"
        assert comparison["pinnacle_novig_prob_a"] is not None
        # 1/1.8 / (1/1.8 + 1/2.1) = 0.5555 / (0.5555 + 0.4761) = ~0.5385
        assert abs(comparison["pinnacle_novig_prob_a"] - 0.5385) < 0.01

        books = {b["bookmaker"]: b for b in comparison["bookmakers"]}
        assert "sts" in books
        assert "pinnacle" in books
        assert books["pinnacle"]["delta_to_pinnacle"] == 0.0
        # STS EV team A with 12% tax: 0.58 * (1.75 * 0.88) - 1.0 = 0.58 * 1.54 - 1.0 = -0.1068
        assert books["sts"]["ev_team_a_tax12"] is not None
        assert "ev_conformal_low_a" in books["sts"]
        assert "is_conformal_value_a" in books["sts"]
        assert isinstance(books["sts"]["is_conformal_value_a"], bool)
        # Test API endpoint
        response = client.get("/matches/303/market-comparison?horizon_hours=6.0")
        assert response.status_code == 200
        data = response.json()
        assert data["canonical_match_id"] == 303
        assert data["team_a"] == "T1"
        assert len(data["bookmakers"]) == 2
    finally:
        session.close()


def test_fetch_pinnacle_horizon_odds_persists_admitted_reverse_binding_as_canonical_sides(client: TestClient) -> None:
    session = get_session()
    try:
        source_quote_at = datetime.now(UTC).replace(microsecond=0)
        match_start = source_quote_at + timedelta(hours=6)
        session.execute(
            text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b, start_time_normalized,
                league, status, match_confidence
            ) VALUES (
                203, 'lec-fnc-g2-binding', 'Fnatic', 'G2 Esports',
                'fnatic', 'g2 esports', :start_time,
                'LEC', 'upcoming', 1.0
            )
            """),
            {"start_time": match_start.isoformat()},
        )
        session.add(
            OddspapiFixtureMapping(
                fixture_id="fix_fnc_g2_binding_203",
                canonical_match_id=203,
                sport_id=18,
                league="LEC",
                provider_team_1="G2 Esports",
                provider_team_2="Fnatic",
                provider_team_1_is_a=0,
                start_time=match_start,
                has_odds=1,
            )
        )
        session.execute(
            text("""
            INSERT INTO canonical_matches (
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b, start_time_normalized,
                league, status, match_confidence
            ) VALUES (
                204, 'lec-t1-dk-binding', 'T1', 'Dplus KIA',
                't1', 'dplus kia', :start_time,
                'LEC', 'upcoming', 1.0
            )
            """),
            {"start_time": match_start.isoformat()},
        )
        session.add(
            OddspapiFixtureMapping(
                fixture_id="fix_t1_dk_binding_204",
                canonical_match_id=204,
                sport_id=18,
                league="LEC",
                provider_team_1="T1",
                provider_team_2="Dplus KIA",
                provider_team_1_is_a=1,
                start_time=match_start,
                has_odds=1,
            )
        )
        session.commit()

        odds_payload = {
            "fixtureId": "fix_fnc_g2_binding_203",
            "isLive": False,
            "bookmakerOdds": {
                "pinnacle": {
                    "markets": {
                        "181": {
                            "outcomes": {
                                "181": {
                                    "players": {
                                        "0": {
                                            "price": 1.50,
                                            "active": True,
                                            "createdAt": source_quote_at.isoformat(),
                                        }
                                    }
                                },
                                "182": {
                                    "players": {
                                        "0": {
                                            "price": 2.70,
                                            "active": True,
                                            "createdAt": source_quote_at.isoformat(),
                                        }
                                    }
                                },
                            }
                        }
                    }
                }
            },
        }
        second_odds_payload = {
            "fixtureId": "fix_t1_dk_binding_204",
            "isLive": False,
            "bookmakerOdds": {
                "pinnacle": {
                    "markets": {
                        "181": {
                            "outcomes": {
                                "181": {
                                    "players": {
                                        "0": {
                                            "price": 1.60,
                                            "active": True,
                                            "createdAt": source_quote_at.isoformat(),
                                        }
                                    }
                                },
                                "182": {
                                    "players": {
                                        "0": {
                                            "price": 2.30,
                                            "active": True,
                                            "createdAt": source_quote_at.isoformat(),
                                        }
                                    }
                                },
                            }
                        }
                    }
                }
            },
        }
        evidence_observed_at = source_quote_at - timedelta(seconds=1)
        binding_rows = [
            {
                "fixture_id": "fix_fnc_g2_binding_203",
                "bookmaker": "pinnacle",
                "market_id": "181",
                "outcome_id": "181",
                "participant_id": "provider-fnatic-id",
                "participant_name": "Fnatic",
                "bookmaker_fixture_id": "pinnacle-fnatic-g2",
                "bookmaker_market_id": "pinnacle-match-winner",
                "bookmaker_outcome_id": "pinnacle-fnatic",
                "bookmaker_participant_id": "pinnacle-team-fnatic",
                "published_at": (evidence_observed_at - timedelta(minutes=1)).isoformat(),
                "observed_at": evidence_observed_at.isoformat(),
                "evidence_uri": "https://evidence.example/pinnacle/fnatic-g2",
            },
            {
                "fixture_id": "fix_fnc_g2_binding_203",
                "bookmaker": "pinnacle",
                "market_id": "181",
                "outcome_id": "182",
                "participant_id": "provider-g2-id",
                "participant_name": "G2 Esports",
                "bookmaker_fixture_id": "pinnacle-fnatic-g2",
                "bookmaker_market_id": "pinnacle-match-winner",
                "bookmaker_outcome_id": "pinnacle-g2",
                "bookmaker_participant_id": "pinnacle-team-g2",
                "published_at": (evidence_observed_at - timedelta(minutes=1)).isoformat(),
                "observed_at": evidence_observed_at.isoformat(),
                "evidence_uri": "https://evidence.example/pinnacle/fnatic-g2",
            },
        ]
        second_binding_rows = [
            {
                "fixture_id": "fix_t1_dk_binding_204",
                "bookmaker": "pinnacle",
                "market_id": "181",
                "outcome_id": "181",
                "participant_id": "provider-t1-id",
                "participant_name": "T1",
                "bookmaker_fixture_id": "pinnacle-t1-dk",
                "bookmaker_market_id": "pinnacle-match-winner",
                "bookmaker_outcome_id": "pinnacle-t1",
                "bookmaker_participant_id": "pinnacle-team-t1",
                "published_at": (evidence_observed_at - timedelta(minutes=1)).isoformat(),
                "observed_at": evidence_observed_at.isoformat(),
                "evidence_uri": "https://evidence.example/pinnacle/t1-dk",
            },
            {
                "fixture_id": "fix_t1_dk_binding_204",
                "bookmaker": "pinnacle",
                "market_id": "181",
                "outcome_id": "182",
                "participant_id": "provider-dk-id",
                "participant_name": "Dplus KIA",
                "bookmaker_fixture_id": "pinnacle-t1-dk",
                "bookmaker_market_id": "pinnacle-match-winner",
                "bookmaker_outcome_id": "pinnacle-dk",
                "bookmaker_participant_id": "pinnacle-team-dk",
                "published_at": (evidence_observed_at - timedelta(minutes=1)).isoformat(),
                "observed_at": evidence_observed_at.isoformat(),
                "evidence_uri": "https://evidence.example/pinnacle/t1-dk",
            },
        ]
        mock_client = MockOddsPapiClient(
            responses={
                "fixture:fix_fnc_g2_binding_203": odds_payload,
                "fixture:fix_t1_dk_binding_204": second_odds_payload,
            }
        )

        result = fetch_pinnacle_horizon_odds(
            session=session,
            target_horizon_hours=6.0,
            tolerance_hours=1.0,
            max_requests=2,
            client=mock_client,
            participant_bindings=tuple(binding_rows + second_binding_rows),
        )

        assert result["status"] == "success"
        assert result["fetched"] == 2
        assert result["saved"] == 2
        assert result["admitted"] == 2
        assert result["quarantined"] == 0

        pinnacle_id = get_or_create_bookmaker("pinnacle")
        snapshot = session.execute(
            text("""
            SELECT odds_a, odds_b, scraped_at, raw_payload FROM odds_snapshots
            WHERE canonical_match_id=203 AND bookmaker_id=:pinnacle_id
            """),
            {"pinnacle_id": pinnacle_id},
        ).fetchone()
        assert snapshot is not None
        assert (float(snapshot[0]), float(snapshot[1])) == (1.5, 2.7)

        stored_at = snapshot[2]
        if stored_at.tzinfo is None:
            stored_at = stored_at.replace(tzinfo=UTC)
        persisted = json.loads(snapshot[3])
        observed_at = datetime.fromisoformat(persisted["response_observed_at"])
        assert stored_at == observed_at
        assert stored_at > source_quote_at
        assert persisted["provider_payload"] == odds_payload
        assert persisted["quote_provenance"]["state_1"]["status"] == "active"
        assert persisted["quote_provenance"]["participant_binding_evidence"] == binding_rows
        assert persisted["quote_provenance"]["quote_admission_status"] == "admitted_with_bookmaker_evidence"
        assert persisted["quote_provenance"]["execution_eligibility"] == "not_certified"
        assert persisted["quote_provenance"]["quote_at_1"] == source_quote_at.isoformat()
        assert persisted["quote_provenance"]["outcome_fixture_participant_indexes"] == {"181": 2, "182": 1}
        assert persisted["participant_bindings"] == binding_rows
        second_snapshot = session.execute(
            text("""
            SELECT odds_a, odds_b, raw_payload FROM odds_snapshots
            WHERE canonical_match_id=204 AND bookmaker_id=:pinnacle_id
            """),
            {"pinnacle_id": pinnacle_id},
        ).fetchone()
        assert second_snapshot is not None
        assert (float(second_snapshot[0]), float(second_snapshot[1])) == (1.6, 2.3)
        second_persisted = json.loads(second_snapshot[2])
        assert second_persisted["provider_payload"] == second_odds_payload
        assert second_persisted["participant_bindings"] == second_binding_rows
        assert second_persisted["quote_provenance"]["participant_binding_evidence"] == second_binding_rows

        contexts = load_prematch_quotes(
            [203, 204], decision_at=datetime.now(UTC), session=session,
        )
        assert len(contexts[203].quotes) == len(contexts[204].quotes) == 1
        # Outcome 181 is explicitly bound to Fnatic, despite provider order G2/Fnatic.
        assert (contexts[203].quotes[0].odds_a, contexts[203].quotes[0].odds_b) == (1.5, 2.7)
        assert (contexts[204].quotes[0].odds_a, contexts[204].quotes[0].odds_b) == (1.6, 2.3)
    finally:
        session.close()


def test_odds_client_capture_retains_raw_fixture_response_without_network(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    body = b'{ \"fixtures\" : [{\"fixtureId\":\"fixture-1\"}] }'
    requested: list[Request] = []

    class Guard:
        def check_budget(self, _session):
            return True, "", {}

        def record_request(self, *_args, **_kwargs):
            return None

    class Response:
        status = 200
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Encoding": "identity",
            "X-Request-ID": "fixture-response",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return body

    def fake_urlopen(request, timeout):
        assert timeout > 0
        requested.append(request)
        return Response()

    monkeypatch.setattr("betting_app.services.oddspapi_service.urlopen", fake_urlopen)
    session = get_session()
    try:
        api = OddsPapiClient(api_key="fixture-api-secret", budget_guard=Guard())
        payload, capture = api.get_captured(
            "/fixtures",
            params={"league": "lol"},
            session=session,
            capture_id="local-client-attempt-1",
            source_id="oddspapi-fixture-feed",
            evidence_uri="https://api.oddspapi.io/v4/fixtures?apiKey=fixture-api-secret",
            body_store=tmp_path / "raw",
            source_class="local_fixture",
        )
        retained = load_source_capture(
            session, capture.capture_id, body_store=tmp_path / "raw"
        )
        assert payload == {"fixtures": [{"fixtureId": "fixture-1"}]}
        assert retained["body_bytes"] == body
        assert retained["sha256"] == hashlib.sha256(body).hexdigest()
        assert retained["response_received_at"] is not None
        assert retained["retrieved_at"] == retained["response_received_at"]
        assert retained["source_authentication_status"] == "local_fixture"
        assert len(requested) == 1 and "fixture-api-secret" in requested[0].full_url
        assert "fixture-api-secret" not in json.dumps(retained["request_metadata"])
        assert "fixture-api-secret" not in retained["evidence_uri"]
    finally:
        session.close()


def test_odds_client_capture_records_http_error_bytes_and_status_without_network(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    body = b'{\"error\":\"denied\"}'

    class Guard:
        def check_budget(self, _session):
            return True, "", {}

        def record_request(self, *_args, **_kwargs):
            return None

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 403, "denied", {"Content-Type": "application/json"}, io.BytesIO(body))

    monkeypatch.setattr("betting_app.services.oddspapi_service.urlopen", fake_urlopen)
    session = get_session()
    try:
        api = OddsPapiClient(api_key="fixture-api-secret", budget_guard=Guard())
        payload, capture = api.get_captured(
            "/fixtures",
            params={},
            session=session,
            capture_id="local-client-attempt-403",
            source_id="oddspapi-fixture-feed",
            evidence_uri="fixture://source/fixtures",
            body_store=tmp_path / "raw",
            source_class="local_fixture",
        )
        retained = load_source_capture(
            session, capture.capture_id, body_store=tmp_path / "raw"
        )
        assert payload is None
        assert capture.http_status == 403 and capture.attempt_status == "failed"
        assert capture.failure_reason == "http_error"
        assert retained["body_bytes"] == body
        assert retained["sha256"] == hashlib.sha256(body).hexdigest()
    finally:
        session.close()
