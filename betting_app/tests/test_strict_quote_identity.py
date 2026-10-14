"""Consumer-visible strict identity and observed-state ingestion regressions."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from betting_app.core.db import query_df, transaction
from betting_app.scrapers.base import RawOddsSnapshot
from betting_app.scrapers.lebull_api import LebullApiScraper
from betting_app.scrapers.totalbet_api import TotalbetApiScraper
from betting_app.services.canonical_match_service import align_snapshot_odds
from betting_app.services.mapping_service import upsert_alias
from betting_app.services.odds_service import insert_odds_snapshot
from betting_app.services.team_alias_service import clear_alias_schema_cache


@pytest.fixture
def isolated_db(client):
    clear_alias_schema_cache()
    yield
    clear_alias_schema_cache()


def test_reversed_quote_is_aligned_exactly_once(isolated_db):
    assert align_snapshot_odds("Alpha", "Beta", "Beta", "Alpha", 2.4, 1.6) == (1.6, 2.4)


@pytest.mark.parametrize("raw_a,raw_b", [("Unknown", "Beta"), ("Alpha", "Alpha"), ("Alpha Academy", "Beta"), ("", "Beta")])
def test_incomplete_or_wrong_squad_identity_is_unavailable(isolated_db, raw_a, raw_b):
    assert align_snapshot_odds("Alpha", "Beta", raw_a, raw_b, 2.4, 1.6) is None


@pytest.mark.parametrize("price", [None, True, False, "bad", float("nan"), float("inf"), 1, 0, 10 ** 400])
def test_invalid_prices_are_unavailable_without_conversion_crash(isolated_db, price):
    assert align_snapshot_odds("Alpha", "Beta", "Alpha", "Beta", price, 2.0) is None


def test_scoped_alias_ambiguity_and_blocking_cannot_fall_back(isolated_db):
    for target in ("Alpha", "Another Alpha"):
        upsert_alias("AAA", target, source=f"manual-{target}", source_system="bookmaker", league_pattern="LCK")
    assert align_snapshot_odds("Alpha", "Beta", "AAA", "Beta", 2, 3, league="LCK", source_system="bookmaker") is None
    blocked_id = upsert_alias("Alpha", "Alpha", source="manual-block", source_system="bookmaker", league_pattern="LCK")
    with transaction() as connection:
        connection.execute("UPDATE team_aliases SET is_blocked = 1 WHERE id = ?", (blocked_id,))
    assert align_snapshot_odds("Alpha", "Beta", "Alpha", "Beta", 2, 3, league="LCK", source_system="bookmaker") is None


def test_contextual_koi_is_exact_not_main_academy_collapse(isolated_db):
    assert align_snapshot_odds("Movistar KOI Fenix", "Beta", "Movistar KOI", "Beta", 2, 3, league="LES", source_system="bookmaker") == (2, 3)
    assert align_snapshot_odds("Movistar KOI", "Beta", "KOI Academy", "Beta", 2, 3, league="LEC", source_system="bookmaker") is None


def test_missing_context_cannot_authorize_short_form_expansion(isolated_db):
    assert align_snapshot_odds("Dplus Kia", "Beta", "DK", "Beta", 2, 3) is None
    assert align_snapshot_odds("Dplus Kia", "Beta", "DK", "Beta", 2, 3, league="LCK", source_system="bookmaker") is None
    upsert_alias("DK", "Dplus Kia", source="manual-approved", source_system="bookmaker", league_pattern="LCK")
    assert align_snapshot_odds("Dplus Kia", "Beta", "DK", "Beta", 2, 3, league="LCK", source_system="bookmaker") == (2, 3)


def totalbet_event():
    return {"uuid": "observed-event", "type": "prematch", "status": "active", "path": [{"name": "League of Legends"}, {"name": "LCK"}], "participants": [{"name": "Beta"}, {"name": "Alpha"}], "start_at": (datetime.now(UTC) + timedelta(days=1)).isoformat(), "markets": [{"name": "Zwycięzca meczu", "outcomes": [{"sort": 1, "odds": 2.4}, {"sort": 2, "odds": 1.6}]}]}


def test_parser_to_insertion_retains_suspended_latest_and_original_orientation(isolated_db):
    event = totalbet_event()
    event["start_at"] = "2026-10-07T12:00:00+00:00"
    good = TotalbetApiScraper().parse_event(event)
    good_dict = good.to_dict()
    good_dict["scraped_at"] = "2026-10-06T10:00:00+00:00"
    first_id = insert_odds_snapshot(good_dict)
    first = query_df("SELECT * FROM odds_snapshots WHERE id = ?", (first_id,)).iloc[0]
    assert (first.raw_team_a, first.raw_team_b, first.odds_a, first.odds_b) == ("Beta", "Alpha", 2.4, 1.6)
    event["status"] = "suspended"
    event["markets"][0]["outcomes"] = [{"sort": 1, "odds": "bad"}]
    withdrawn = TotalbetApiScraper().parse_event(event)
    assert withdrawn is not None and withdrawn.odds_a is None and withdrawn.odds_b is None
    payload = withdrawn.to_dict()
    payload["scraped_at"] = "2026-10-06T10:01:00+00:00"
    newest_id = insert_odds_snapshot(payload)
    assert newest_id != first_id
    from betting_app.services.prematch_quote_service import load_prematch_quotes
    context = load_prematch_quotes(
        [int(first.canonical_match_id)],
        decision_at=datetime(2026, 10, 6, 10, 1, tzinfo=UTC),
    )[int(first.canonical_match_id)]
    assert context.quotes == () and context.prob_a is None
    payload["scraped_at"] = good_dict["scraped_at"]
    assert insert_odds_snapshot(payload) == first_id
    rows = query_df("SELECT * FROM odds_snapshots WHERE id = ?", (first_id,))
    assert len(rows) == 1 and rows.iloc[0].odds_a is None
    assert json.loads(rows.iloc[0].raw_payload)["event"]["status"] == "suspended"


@pytest.mark.parametrize("live", [None, "false", "true", 2, 0.0])
def test_unsupported_live_does_not_become_prematch(isolated_db, live):
    payload = RawOddsSnapshot("totalbet", "Alpha", "Beta", 2.41, 1.6).to_dict()
    payload["is_live"] = live
    row_id = insert_odds_snapshot(payload)
    assert query_df("SELECT is_live FROM odds_snapshots WHERE id = ?", (row_id,)).iloc[0].is_live is None


@pytest.mark.parametrize("live,stored", [(False, 0), (True, 1), (0, 0), (1, 1)])
def test_explicit_live_states_round_trip(isolated_db, live, stored):
    payload = RawOddsSnapshot("totalbet", "Alpha", "Beta", 2.41, 1.6).to_dict()
    payload["is_live"] = live
    row_id = insert_odds_snapshot(payload)
    assert query_df("SELECT is_live FROM odds_snapshots WHERE id = ?", (row_id,)).iloc[0].is_live == stored


def test_omitted_live_is_unknown_through_actual_insertion(isolated_db):
    payload = RawOddsSnapshot("totalbet", "Alpha", "Beta", 2.41, 1.6).to_dict()
    payload.pop("is_live")
    row_id = insert_odds_snapshot(payload)
    assert query_df("SELECT is_live FROM odds_snapshots WHERE id = ?", (row_id,)).iloc[0].is_live is None


def test_lebull_live_incomplete_state_survives_parser():
    event = {"eventId": 42, "countryName": "League of Legends", "teamA": "Alpha", "teamB": "Beta", "isLive": True, "stakeTypes": [{"stakeTypeName": "Winner", "stakes": [{"stakeCode": 1, "isActive": False, "betFactor": 2.4}]}]}
    snapshot = LebullApiScraper().parse_event_detail(event)
    assert snapshot is not None and snapshot.is_live is True
    assert snapshot.odds_a is None and snapshot.odds_b is None
    assert snapshot.raw_payload["event"] == event


def test_absent_market_is_not_invented_withdrawal():
    event = totalbet_event()
    event["markets"] = []
    assert TotalbetApiScraper().parse_event(event) is None


def test_map_market_is_not_invented_match_winner_withdrawal():
    event = totalbet_event()
    event["markets"][0]["name"] = "Zwycięzca mapy 1"
    assert TotalbetApiScraper().parse_event(event) is None


@pytest.mark.parametrize("live", [None, "false", "true", 2, 0.0])
def test_totalbet_unsupported_explicit_live_is_unknown(live):
    event = totalbet_event()
    event["isLive"] = live
    snapshot = TotalbetApiScraper().parse_event(event)
    assert snapshot is not None and snapshot.is_live is None


def test_parser_retains_actual_observation_not_provider_creation():
    event = totalbet_event()
    event["createdAt"] = "2000-01-01T00:00:00+00:00"
    observed = datetime(2026, 10, 6, 10, 1, 2, 345678, tzinfo=UTC)
    snapshot = TotalbetApiScraper().parse_event(event, observed_at=observed)
    assert snapshot.scraped_at == observed.isoformat()


def test_retrospective_market_comparison_aligns_and_never_revives_invalid_state(isolated_db):
    from betting_app.services.oddspapi_service import compare_match_market

    with transaction() as connection:
        match = connection.execute(
            "INSERT INTO canonical_matches(canonical_key, team_a_name, team_b_name, normalized_team_a, normalized_team_b, start_time_normalized, league, status, match_confidence, best_of) "
            "VALUES ('retrospective-alpha-beta','Alpha','Beta','alpha','beta','2026-10-08T12:00:00+00:00','LCK','upcoming',1,3) RETURNING id"
        ).fetchone()
    payload = RawOddsSnapshot(
        "totalbet", "Beta", "Alpha", 2.4, 1.6,
        scraped_at="2026-10-08T05:00:00+00:00", raw_league="LCK",
        match_start_time="2026-10-08T12:00:00+00:00", is_live=False,
    ).to_dict()
    insert_odds_snapshot(payload)
    comparison = compare_match_market(match["id"])
    assert comparison["bookmakers"][0]["odds_a"] == 1.6
    assert comparison["bookmakers"][0]["odds_b"] == 2.4
    payload.update(scraped_at="2026-10-08T05:01:00+00:00", odds_a=None)
    insert_odds_snapshot(payload)
    comparison = compare_match_market(match["id"])
    assert comparison["bookmakers"][0]["odds_a"] is None
    assert comparison["bookmakers"][0]["ev_team_a_tax12"] is None


def test_oddspapi_identifiable_withdrawal_retains_observation_and_payload():
    from betting_app.models.oddspapi import OddspapiFixtureMapping
    from betting_app.services.oddspapi_service import build_oddspapi_observed_snapshot
    from src.analysis.quote_provenance import extract_winner_quote

    payload = {"fixtureId": "event", "isLive": True, "bookmakerOdds": {"pinnacle": {"markets": {"181": {"outcomes": {}}}}}}
    now = datetime.now(UTC)
    extracted = extract_winner_quote(payload, "pinnacle", cutoff_at=now)
    snapshot = build_oddspapi_observed_snapshot(
        payload, extracted, mapping=OddspapiFixtureMapping(fixture_id="event", provider_team_1="Beta", provider_team_2="Alpha"),
        canonical_match_id=1, bookmaker_id=10, observed_at=now, participant_bindings=[],
    )
    assert snapshot.scraped_at == now and snapshot.is_live == 1
    assert snapshot.odds_a is None and snapshot.odds_b is None
    assert (snapshot.raw_team_a, snapshot.raw_team_b) == ("Beta", "Alpha")
    assert json.loads(snapshot.raw_payload)["provider_payload"] == payload


@pytest.mark.parametrize("live", [None, "false", "true", 2, 0.0])
def test_oddspapi_unsupported_live_is_unknown(live):
    from betting_app.services.oddspapi_service import _observed_live_state
    assert _observed_live_state({"isLive": live}, {"explicit_scope_status": "rejected_explicit_live_scope"}) is None
