from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.analysis.quote_provenance import extract_winner_quote


def _payload(
    side_1: list[dict[str, Any]],
    side_2: list[dict[str, Any]],
    *,
    other_markets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "fixtureId": "fixture-1",
        "bookmakers": {
            "pinnacle": {
                "markets": {
                    **(other_markets or {}),
                    "181": {
                        "outcomes": {
                            "181": {"players": {"0": side_1}},
                            "182": {"players": {"0": side_2}},
                        }
                    },
                }
            }
        },
    }


def _state(at: str, price: float | None, *, active: bool = True, **extra: Any) -> dict[str, Any]:
    return {"createdAt": at, "price": price, "active": active, **extra}


def _binding_rows(*, observed_at: str = "2026-06-01T11:30:00Z") -> list[dict[str, Any]]:
    common = {
        "fixture_id": "fixture-1",
        "bookmaker": "pinnacle",
        "market_id": "181",
        "bookmaker_fixture_id": "pinnacle-fixture-1",
        "bookmaker_market_id": "pinnacle-winner",
        "published_at": "2026-06-01T11:00:00Z",
        "observed_at": observed_at,
        "evidence_uri": "https://evidence.example/bindings/fixture-1",
    }
    return [
        {
            **common,
            "outcome_id": "181",
            "participant_id": "team-b-id",
            "participant_name": "Team B",
            "bookmaker_outcome_id": "pinnacle-away",
            "bookmaker_participant_id": "pinnacle-team-b",
        },
        {
            **common,
            "outcome_id": "182",
            "participant_id": "team-a-id",
            "participant_name": "Team A",
            "bookmaker_outcome_id": "pinnacle-home",
            "bookmaker_participant_id": "pinnacle-team-a",
        },
    ]


def test_latest_inactive_state_invalidates_previous_active_price() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8), _state("2026-06-01T11:00:00Z", None, active=False)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    quote = extract_winner_quote(payload, "pinnacle")

    assert quote is not None
    assert quote["odds_1"] is None
    assert quote["state_1"]["status"] == "inactive"
    assert quote["odds_2"] == 2.1
    assert quote["quote_at"] is None
    assert quote["quote_at_1"] == "2026-06-01T11:00:00Z"
    assert quote["state_1"]["observations"][-1]["raw_state"]["active"] is False


def test_latest_invalid_active_price_supersedes_previous_valid_price() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8), _state("2026-06-01T11:00:00Z", 1.0)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    quote = extract_winner_quote(payload, "pinnacle")

    assert quote is not None
    assert quote["odds_1"] is None
    assert quote["state_1"]["status"] == "invalid_price"
    assert quote["state_1"]["latest_states"][0]["raw_state"]["price"] == 1.0


def test_provider_outcome_convention_does_not_flip_or_certify_binding() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    forward_quote = extract_winner_quote(
        payload,
        "pinnacle",
        fixture={"participant1Name": "Canonical A", "participant2Name": "Canonical B"},
    )
    reverse_quote = extract_winner_quote(
        payload,
        "pinnacle",
        fixture={"participant1Name": "Canonical B", "participant2Name": "Canonical A"},
    )

    assert forward_quote is not None
    assert reverse_quote is not None
    assert (forward_quote["odds_1"], forward_quote["odds_2"]) == (1.8, 2.1)
    assert (reverse_quote["odds_1"], reverse_quote["odds_2"]) == (1.8, 2.1)
    assert forward_quote["participant_binding_status"] == "unverified_provider_outcome_convention"
    assert reverse_quote["participant_binding_status"] == "unverified_provider_outcome_convention"
    assert reverse_quote["execution_eligibility"] == "not_certified"
    assert reverse_quote["participant_1_id"] is None
    assert reverse_quote["participant_1_name"] is None


def test_source_state_time_is_not_reported_as_observation_time() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )

    assert quote is not None
    assert quote["state_1"]["source_created_at"] == "2026-06-01T10:00:00+00:00"
    assert quote["state_1"]["observed_at"] is None
    assert quote["state_1"]["observations"][0]["created_at"] == "2026-06-01T10:00:00Z"
    assert quote["state_1"]["observations"][0]["observed_at"] is None
    assert quote["quote_at_1"] == "2026-06-01T10:00:00Z"


def test_future_state_is_not_reused_at_an_earlier_cutoff() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8), _state("2026-06-01T12:00:00Z", None, active=False)],
        [_state("2026-06-01T10:30:00Z", 2.1)],
    )

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 11, 0, tzinfo=UTC),
    )

    assert quote is not None
    assert quote["odds_1"] == 1.8
    assert quote["quote_at_1"] == "2026-06-01T10:00:00Z"
    assert any(item["status"] == "after_cutoff" for item in quote["state_1"]["observations"])


def test_missing_or_naive_timestamp_makes_side_unavailable() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8), {"price": 1.7, "active": True}],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    quote = extract_winner_quote(payload, "pinnacle")

    assert quote is not None
    assert quote["odds_1"] is None
    assert quote["state_1"]["status"] == "unorderable_state_present"
    assert quote["quote_at"] is None

    naive_payload = _payload(
        [_state("2026-06-01T11:00:00", 1.7)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    naive_quote = extract_winner_quote(naive_payload, "pinnacle")
    assert naive_quote is not None
    assert naive_quote["odds_1"] is None
    assert naive_quote["state_1"]["status"] == "unorderable_state_present"


def test_equal_time_conflicting_updates_are_not_arbitrarily_selected() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8), _state("2026-06-01T10:00:00+00:00", 1.9)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    quote = extract_winner_quote(payload, "pinnacle")

    assert quote is not None
    assert quote["odds_1"] is None
    assert quote["state_1"]["status"] == "equal_time_conflict"
    assert len(quote["state_1"]["latest_states"]) == 2


def test_timestamps_compare_as_instants_across_full_winner_history_only() -> None:
    payload = _payload(
        [
            _state("2026-06-01T11:00:00+02:00", 1.8),
            _state("2026-06-01T10:00:00Z", 1.7),
        ],
        [_state("2026-06-01T09:30:00Z", 2.1)],
        other_markets={
            "183": {
                "outcomes": {
                    "181": {"players": {"0": [_state("2026-06-01T12:00:00Z", 99.0)]}},
                    "182": {"players": {"0": [_state("2026-06-01T12:00:00Z", 99.0)]}},
                }
            }
        },
    )
    numeric_quote = extract_winner_quote(payload, "pinnacle")
    assert numeric_quote is not None
    assert numeric_quote["odds_1"] == 1.7
    assert numeric_quote["quote_at_1"] == "2026-06-01T10:00:00Z"

    payload["bookmakers"]["pinnacle"]["markets"]["181"]["outcomes"]["181"]["players"]["4"] = [
        _state("2026-06-01T10:30:00Z", 1.65)
    ]

    quote = extract_winner_quote(payload, "pinnacle")

    assert quote is not None
    assert quote["market_id"] == "181"
    assert quote["odds_1"] == 1.65
    assert quote["quote_at_1"] == "2026-06-01T10:30:00Z"
    assert quote["odds_2"] == 2.1


def test_explicit_participant_mismatch_is_preserved_for_inspection() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8, participantId="book-team-x", participantName="Team X")],
        [_state("2026-06-01T10:00:00Z", 2.1, participantId="book-team-y", participantName="Team Y")],
    )

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        fixture={"participant1Id": "fixture-team-a", "participant1Name": "Team A",
                 "participant2Id": "fixture-team-b", "participant2Name": "Team B"},
    )

    assert quote is not None
    assert quote["participant_1_id"] == "book-team-x"
    assert quote["participant_1_name"] == "Team X"
    assert quote["participant_2_id"] == "book-team-y"
    assert quote["participant_2_name"] == "Team Y"
    assert quote["participant_binding_status"] == "source_participants_mismatch_fixture"
    assert quote["fixture_participant_1_id"] == "fixture-team-a"
    assert quote["fixture_participant_1_name"] == "Team A"
    assert quote["fixture_participant_2_id"] == "fixture-team-b"
    assert quote["fixture_participant_2_name"] == "Team B"
    assert "no_independent_bookmaker_participant_binding_evidence" in quote["provenance_limitations"]




def test_explicit_reverse_binding_stays_source_ordered_and_unverified() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8, participantId="book-b", participantName="Team B")],
        [_state("2026-06-01T10:00:00Z", 2.1, participantId="book-a", participantName="Team A")],
    )

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        fixture={
            "participant1Id": "fixture-a",
            "participant1Name": "Team A",
            "participant2Id": "fixture-b",
            "participant2Name": "Team B",
        },
    )

    assert quote is not None
    assert quote["odds_1"] == 1.8
    assert quote["odds_2"] == 2.1
    assert quote["participant_1_id"] == "book-b"
    assert quote["participant_1_name"] == "Team B"
    assert quote["participant_2_id"] == "book-a"
    assert quote["participant_2_name"] == "Team A"
    assert quote["participant_binding_status"] == "source_participants_match_fixture_reverse_unverified"
    assert quote["execution_eligibility"] == "not_certified"


def _fixture_context() -> dict[str, str]:
    return {
        "fixtureId": "fixture-1",
        "participant1Id": "team-a-id",
        "participant1Name": "Team A",
        "participant2Id": "team-b-id",
        "participant2Name": "Team B",
    }


def test_external_binding_admits_reverse_outcome_order_without_execution_certification() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows()),
    )

    assert quote is not None
    assert quote["quote_admission_status"] == "admitted_with_bookmaker_evidence"
    assert quote["execution_eligibility"] == "not_certified"
    assert quote["outcome_fixture_participant_indexes"] == {"181": 2, "182": 1}
    assert quote["participant_binding_evidence"] == _binding_rows()
    assert (quote["odds_1"], quote["odds_2"]) == (1.8, 2.1)


def test_future_or_wrong_scope_binding_evidence_is_rejected() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    cutoff = datetime(2026, 6, 1, 12, tzinfo=UTC)

    future_quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=cutoff,
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows(observed_at="2026-06-01T12:01:00Z")),
    )
    assert future_quote is not None
    assert future_quote["quote_admission_status"] == "rejected_binding_evidence"
    assert future_quote["execution_eligibility"] == "not_certified"

    wrong_scope = _binding_rows()
    wrong_scope[0]["bookmaker"] = "unibet"
    wrong_scope_quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=cutoff,
        fixture=_fixture_context(),
        participant_bindings=tuple(wrong_scope),
    )
    assert wrong_scope_quote is not None
    assert wrong_scope_quote["quote_admission_status"] == "rejected_binding_evidence"


def test_source_fixture_or_explicit_payload_identity_mismatch_blocks_admission() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    payload["fixtureId"] = "different-source-fixture"
    cutoff = datetime(2026, 6, 1, 12, tzinfo=UTC)

    mismatched_fixture = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=cutoff,
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows()),
    )
    assert mismatched_fixture is not None
    assert mismatched_fixture["quote_admission_status"] == "source_fixture_mismatch"

    payload["fixtureId"] = "fixture-1"
    payload["bookmakers"]["pinnacle"]["markets"]["181"]["outcomes"]["181"]["players"]["0"][0].update(
        {"participantId": "unexpected-id", "participantName": "Unexpected Team"}
    )
    mismatched_identity = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=cutoff,
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows()),
    )
    assert mismatched_identity is not None
    assert mismatched_identity["quote_admission_status"] == "rejected_binding_evidence"


def test_split_source_identity_fields_cannot_hide_conflicting_id() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8, participantId="unexpected-provider-id")],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    payload["bookmakers"]["pinnacle"]["markets"]["181"]["outcomes"]["181"]["participantName"] = "Team B"

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows()),
    )

    assert quote is not None
    assert quote["participant_1_id"] == "unexpected-provider-id"
    assert quote["participant_1_name"] == "Team B"
    assert quote["quote_admission_status"] == "rejected_binding_evidence"
    assert "binding_conflicts_with_payload_participant_id" in quote["binding_validation_reasons"]


def test_explicit_live_or_map_only_market_scope_is_rejected() -> None:
    live_payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    live_payload["bookmakers"]["pinnacle"]["markets"]["181"]["isLive"] = True
    live_quote = extract_winner_quote(
        live_payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows()),
    )
    assert live_quote is not None
    assert live_quote["quote_admission_status"] == "rejected_explicit_live_scope"
    assert (live_quote["odds_1"], live_quote["odds_2"]) == (None, None)

    map_payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    map_payload["bookmakers"]["pinnacle"]["markets"]["181"]["marketName"] = "Map 1 Winner"
    map_quote = extract_winner_quote(
        map_payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
        fixture=_fixture_context(),
        participant_bindings=tuple(_binding_rows()),
    )
    assert map_quote is not None
    assert map_quote["quote_admission_status"] == "rejected_map_only_scope"
    assert (map_quote["odds_1"], map_quote["odds_2"]) == (None, None)


def test_unrelated_map_market_and_future_live_update_do_not_relabel_prematch_quote() -> None:
    payload = _payload(
        [
            _state("2026-06-01T10:00:00Z", 1.8),
            _state("2026-06-01T12:00:00Z", 2.5, isLive=True),
        ],
        [_state("2026-06-01T10:30:00Z", 2.1)],
        other_markets={
            "183": {
                "marketName": "Map 1 Winner",
                "outcomes": {},
            }
        },
    )

    quote = extract_winner_quote(
        payload,
        "pinnacle",
        cutoff_at=datetime(2026, 6, 1, 11, tzinfo=UTC),
    )

    assert quote is not None
    assert (quote["odds_1"], quote["odds_2"]) == (1.8, 2.1)
    assert quote["explicit_scope_status"] is None


def test_binding_identifiers_reject_coercible_non_ids_and_missing_placeholders() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    for invalid_id in (True, {}, float("nan"), "NaN"):
        rows = _binding_rows()
        rows[0]["participant_id"] = invalid_id
        quote = extract_winner_quote(
            payload,
            "pinnacle",
            cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
            fixture=_fixture_context(),
            participant_bindings=tuple(rows),
        )
        assert quote is not None
        assert quote["quote_admission_status"] == "rejected_binding_evidence"


def test_participant_and_book_names_are_not_market_scope_metadata() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    bookmaker = payload["bookmakers"]["pinnacle"]
    bookmaker["name"] = "Live Betting"
    market = bookmaker["markets"]["181"]
    market["outcomes"]["181"]["name"] = "Game 1 Esports"
    quote = extract_winner_quote(payload, "pinnacle")
    assert quote is not None
    assert (quote["odds_1"], quote["odds_2"]) == (1.8, 2.1)
    market["name"] = "Map 1 winner"
    quote = extract_winner_quote(payload, "pinnacle")
    assert quote is not None
    assert (quote["odds_1"], quote["odds_2"]) == (None, None)


def test_earlier_known_actual_start_excludes_later_source_states() -> None:
    payload = _payload(
        [_state("2026-06-01T11:30:00Z", 1.8)],
        [_state("2026-06-01T11:30:00Z", 2.1)],
    )
    fixture = _fixture_context()
    fixture.update(startTime="2026-06-01T13:00:00Z", trueStartTime="2026-06-01T11:00:00Z")
    quote = extract_winner_quote(payload, "pinnacle", cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC), fixture=fixture)
    assert quote is not None
    assert (quote["odds_1"], quote["odds_2"]) == (None, None)


def test_old_prematch_price_cannot_be_admitted_at_match_start() -> None:
    payload = _payload(
        [_state("2026-06-01T10:00:00Z", 1.8)],
        [_state("2026-06-01T10:00:00Z", 2.1)],
    )
    fixture = _fixture_context()
    fixture["startTime"] = "2026-06-01T12:00:00Z"
    quote = extract_winner_quote(
        payload, "pinnacle", cutoff_at=datetime(2026, 6, 1, 12, tzinfo=UTC),
        fixture=fixture, participant_bindings=_binding_rows(),
    )
    assert quote is not None
    assert quote["quote_admission_status"] == "rejected_binding_evidence"
    assert "cutoff_not_before_fixture_start" in quote["binding_validation_reasons"]
