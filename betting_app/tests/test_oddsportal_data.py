"""Offline behavioral contracts for retained OddsPortal data, not source certification."""
import base64
import copy
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import pytest

from src.utils.oddsportal_data import (
    decrypt_server_response,
    event_id_from_url,
    parse_archive_row,
    parse_detailed_odds,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/oddsportal/match_event.json").read_text())
BOOKIES = {"165": "STS", "502": "LV BET"}
EVENT_ID = "SbYznU0K"
MARKET = "E-3-2-0-0-0"
PASSWORD = "offline-password"
SALT = "offline-salt"


def encrypted_bytes(plaintext, *, padded=False):
    iv = bytes(range(16))
    key = hashlib.pbkdf2_hmac("sha256", PASSWORD.encode(), SALT.encode(), 1000, dklen=32)
    ciphertext = AES.new(key, AES.MODE_CBC, iv).encrypt(plaintext if padded else pad(plaintext, 16))
    return base64.b64encode(base64.b64encode(ciphertext) + b":" + iv.hex().encode()).decode()


def archive_row():
    return {
        "encodeEventId": EVENT_ID,
        "url": "/pl/esports/league-of-legends/cup/alpha-beta-SbYznU0K/",
        "date-start-base": 1765699200,
        "tournament-name": "Example Cup",
        "home-name": "Alpha",
        "away-name": "Beta",
        "homeResult": "2",
        "awayResult": "1",
        "home-winner": "win",
        "away-winner": "lose",
    }


def test_retained_pairs_and_source_clocks_preserve_provider_side_order():
    result = parse_detailed_odds(FIXTURE, EVENT_ID, BOOKIES)
    assert result["openingsOdds"]["165"] == [1.55, 2.29]
    assert result["closingOdds"]["165"] == [1.4, 2.75]
    assert result["odds1_sts_open"] == 1.55
    assert result["odds2_sts_close"] == 2.75
    assert result["odds1_lv_bet_close"] == 1.38
    assert result["event_id"] == EVENT_ID
    assert result["market_key"] == MARKET
    assert result["source_outcome_ids"] == ["98lq7x2s5a4x0xphnup", "98lq7x2s5a4x0xphnur"]
    assert result["source_open_change_at"]["165"] == [
        datetime.fromtimestamp(1765628964, timezone.utc).isoformat(),
        datetime.fromtimestamp(1765628964, timezone.utc).isoformat(),
    ]
    assert result["source_last_change_at"]["165"] == [
        datetime.fromtimestamp(1765698443, timezone.utc).isoformat(),
        datetime.fromtimestamp(1765698443, timezone.utc).isoformat(),
    ]
    assert result["source_bookmaker_active"]["165"] is True
    assert result["quote_timing_semantics"] == "provider_open_and_last_not_certified_close"


def test_exact_market_not_dictionary_order_selects_prices():
    payload = copy.deepcopy(FIXTURE)
    wrong = copy.deepcopy(payload["d"]["oddsdata"]["back"][MARKET])
    wrong["odds"]["165"] = [9.0, 1.01]
    payload["d"]["oddsdata"]["back"] = {"E-3-2-0-1-0": wrong, MARKET: payload["d"]["oddsdata"]["back"][MARKET]}
    assert parse_detailed_odds(payload, EVENT_ID, BOOKIES)["closingOdds"]["165"] == [1.4, 2.75]


@pytest.mark.parametrize("field,value", [("bt", 2), ("sc", 1), ("encodeventId", "Other123")])
def test_wrong_event_or_top_level_market_identity_is_rejected(field, value):
    payload = copy.deepcopy(FIXTURE)
    payload["d"][field] = value
    with pytest.raises(ValueError):
        parse_detailed_odds(payload, EVENT_ID, BOOKIES)


@pytest.mark.parametrize("field,value", [
    ("bettingTypeId", 2), ("scopeId", 1), ("handicapTypeId", 1),
    ("handicapValue", 1), ("mixedParameterId", 1), ("mixedParameterName", "Map 1"),
    ("isBack", False), ("outcomeId", ["same", "same"]),
    ("outcomeId", ["a", "b", "draw"]), ("outcomeId", ["a", None]),
])
def test_unsupported_market_or_ambiguous_outcomes_are_rejected(field, value):
    payload = copy.deepcopy(FIXTURE)
    payload["d"]["oddsdata"]["back"][MARKET][field] = value
    with pytest.raises(ValueError):
        parse_detailed_odds(payload, EVENT_ID, BOOKIES)


def test_missing_exact_market_cannot_fall_back_to_another_market():
    payload = copy.deepcopy(FIXTURE)
    market = payload["d"]["oddsdata"]["back"].pop(MARKET)
    payload["d"]["oddsdata"]["back"]["E-3-2-0-1-0"] = market
    with pytest.raises(ValueError):
        parse_detailed_odds(payload, EVENT_ID, BOOKIES)


@pytest.mark.parametrize("pair", [[1.0, 2], [0, 2], [True, 2], [None, 2], [float("nan"), 2], [float("inf"), 2], [2, "bad"], [2], [2, 3, 4], {"home": 2, "away": 3}])
@pytest.mark.parametrize("field", ["openingOdd", "odds"])
def test_invalid_or_ambiguous_prices_raise_instead_of_making_up_a_pair(field, pair):
    payload = copy.deepcopy(FIXTURE)
    payload["d"]["oddsdata"]["back"][MARKET][field]["165"] = pair
    with pytest.raises(ValueError):
        parse_detailed_odds(payload, EVENT_ID, BOOKIES)


def test_missing_and_per_side_clocks_remain_missing_without_capture_substitution():
    payload = copy.deepcopy(FIXTURE)
    market = payload["d"]["oddsdata"]["back"][MARKET]
    del market["openingChangeTime"]
    market["changeTime"]["165"] = [None, 1765698443]
    market["act"]["165"] = "unknown"
    result = parse_detailed_odds(payload, EVENT_ID, BOOKIES)
    assert result["source_open_change_at"]["165"] == [None, None]
    assert result["source_last_change_at"]["165"][0] is None
    assert result["source_last_change_at"]["165"][1].endswith("+00:00")
    assert result["odds1_sts_open_at"] is None
    assert result["odds2_sts_open_at"] is None
    assert result["odds1_sts_close_at"] is None
    assert result["odds2_sts_close_at"] == result["source_last_change_at"]["165"][1]
    assert result["source_bookmaker_active"]["165"] is None
    assert result["closingOdds"]["165"] == [1.4, 2.75]


@pytest.mark.parametrize("compressed", [False, True])
def test_encryption_round_trip_preserves_plaintext(compressed):
    plaintext = json.dumps(FIXTURE).encode()
    wire = encrypted_bytes(gzip.compress(plaintext) if compressed else plaintext)
    assert decrypt_server_response(wire, PASSWORD, SALT) == plaintext.decode()


def test_damaged_pkcs7_padding_is_rejected_even_when_last_byte_looks_valid():
    damaged = b"{}" + b"\x0d" + b"\x0e" * 13
    with pytest.raises(ValueError):
        decrypt_server_response(encrypted_bytes(damaged, padded=True), PASSWORD, SALT)


@pytest.mark.parametrize("wire", ["not-base64", "", base64.b64encode(b"%%%%:" + b"00" * 16).decode(), base64.b64encode(b"YQ==:" + b"00" * 16).decode(), base64.b64encode(b"YQ==:00").decode()])
def test_malformed_wire_is_not_empty_json_success(wire):
    with pytest.raises(ValueError):
        decrypt_server_response(wire, PASSWORD, SALT)


@pytest.mark.parametrize("plaintext", [b"\x1f\x8btruncated", b"\xff"])
def test_invalid_gzip_or_utf8_is_rejected(plaintext):
    with pytest.raises(ValueError):
        decrypt_server_response(encrypted_bytes(plaintext), PASSWORD, SALT)


@pytest.mark.parametrize("url", [
    "https://www.oddsportal.com/pl/esports/cup/alpha-beta-SbYznU0K/",
    "/pl/esports/cup/alpha-beta-SbYznU0K/",
    "https://oddsportal.com/pl/esports/cup/alpha-beta/#SbYznU0K",
    "/pl/esports/cup/alpha-beta-SbYznU0K/#SbYznU0K",
])
def test_supported_event_url_forms_extract_exact_case_sensitive_identity(url):
    assert event_id_from_url(url) == EVENT_ID


@pytest.mark.parametrize("url", [
    "https://example.org/alpha-beta-SbYznU0K/",
    "//example.org/alpha-beta-SbYznU0K/",
    "https://www.oddsportal.com.example.org/alpha-beta-SbYznU0K/",
    "https://user@www.oddsportal.com/alpha-beta-SbYznU0K/",
    "/alpha-beta-SbYznU0K/#Other123",
    "/alpha-beta/", "/alpha-beta-short/", "",
])
def test_off_host_missing_or_conflicting_url_identity_is_rejected(url):
    with pytest.raises(ValueError):
        event_id_from_url(url)


def test_h2h_route_uses_event_fragment_not_participant_ids_in_path():
    url = "https://www.oddsportal.com/pl/esports/h2h/cloud9-league-of-legends-2aEvjI9r/sentinels-league-of-legends-2mq5T67M/#ne8lT6bs"
    assert event_id_from_url(url) == "ne8lT6bs"
    with pytest.raises(ValueError):
        event_id_from_url(url.split("#")[0])


def test_archive_preserves_identity_names_results_and_utc_start():
    result = parse_archive_row(archive_row())
    assert result["event_id"] == result["encodeEventId"] == EVENT_ID
    assert result["date"] == "2025-12-14T08:00:00+00:00"
    assert result["date-start-base"] == 1765699200
    assert (result["home_team"], result["away_team"]) == ("Alpha", "Beta")
    assert (result["homeResult"], result["awayResult"]) == ("2", "1")
    assert result["winner"] == "home"
    assert result["tournament"] == "Example Cup"
    assert result["url"] == "https://www.oddsportal.com/pl/esports/league-of-legends/cup/alpha-beta-SbYznU0K/"


@pytest.mark.parametrize("home_flag,away_flag,scores,expected", [
    (None, None, (None, None), None), (None, None, ("0", "0"), None),
    ("", "", ("-", "-"), None), ("draw", "draw", ("1", "1"), "draw"),
    ("lose", "win", ("1", "2"), "away"),
])
def test_unknown_and_unplayed_results_are_not_invented_draws(home_flag, away_flag, scores, expected):
    row = archive_row()
    row.update({"home-winner": home_flag, "away-winner": away_flag, "homeResult": scores[0], "awayResult": scores[1]})
    assert parse_archive_row(row)["winner"] == expected


@pytest.mark.parametrize("patch", [
    {"home-name": None}, {"away-name": ""}, {"away-name": " alpha "},
    {"encodeEventId": "Other123"}, {"encodeEventId": None},
    {"url": "https://example.org/alpha-beta-SbYznU0K/"},
    {"home-winner": "win", "away-winner": "win"},
])
def test_archive_missing_equal_or_conflicting_identity_is_rejected(patch):
    row = archive_row()
    row.update(patch)
    with pytest.raises(ValueError):
        parse_archive_row(row)


def test_side_specific_provider_epochs_produce_distinct_utc_datetime_columns():
    payload = copy.deepcopy(FIXTURE)
    market = payload["d"]["oddsdata"]["back"][MARKET]
    market["openingChangeTime"]["165"] = [0, 60]
    market["changeTime"]["165"] = [120, 180]
    result = parse_detailed_odds(payload, EVENT_ID, BOOKIES)
    assert result["odds1_sts_open_at"] == "1970-01-01T00:00:00+00:00"
    assert result["odds2_sts_open_at"] == "1970-01-01T00:01:00+00:00"
    assert result["odds1_sts_close_at"] == "1970-01-01T00:02:00+00:00"
    assert result["odds2_sts_close_at"] == "1970-01-01T00:03:00+00:00"
    assert result["source_open_change_at"]["165"] == [result["odds1_sts_open_at"], result["odds2_sts_open_at"]]
    assert result["source_last_change_at"]["165"] == [result["odds1_sts_close_at"], result["odds2_sts_close_at"]]


@pytest.mark.parametrize("epoch", ["1765698443", True, {}, [], float("nan"), float("inf"), 10**100])
@pytest.mark.parametrize("field", ["openingChangeTime", "changeTime"])
def test_wrong_type_nonfinite_or_unrepresentable_source_epochs_fail_explicitly(field, epoch):
    payload = copy.deepcopy(FIXTURE)
    payload["d"]["oddsdata"]["back"][MARKET][field]["165"] = [epoch, None]
    with pytest.raises(ValueError):
        parse_detailed_odds(payload, EVENT_ID, BOOKIES)


def test_no_source_clocks_does_not_turn_provider_time_base_into_quote_datetimes():
    payload = copy.deepcopy(FIXTURE)
    market = payload["d"]["oddsdata"]["back"][MARKET]
    del market["openingChangeTime"]
    del market["changeTime"]
    result = parse_detailed_odds(payload, EVENT_ID, BOOKIES)
    for suffix in ("open", "close"):
        assert result[f"odds1_sts_{suffix}_at"] is None
        assert result[f"odds2_sts_{suffix}_at"] is None
    assert result["source_open_change_at"]["165"] == [None, None]
    assert result["source_last_change_at"]["165"] == [None, None]


def test_provider_inactive_flag_is_retained_separately_from_reported_prices():
    payload = copy.deepcopy(FIXTURE)
    market = payload["d"]["oddsdata"]["back"][MARKET]
    market["act"]["165"] = False
    del market["openingOdd"]["165"]
    result = parse_detailed_odds(payload, EVENT_ID, BOOKIES)
    assert result["source_bookmaker_active"]["165"] is False
    assert "165" not in result["openingsOdds"]
    assert result["closingOdds"]["165"] == [1.4, 2.75]


def test_archive_unknown_start_is_null_and_wrong_clock_types_are_rejected():
    row = archive_row()
    row["date-start-base"] = None
    assert parse_archive_row(row)["date"] is None
    row["date-start-base"] = "2025-12-14 08:00"
    with pytest.raises(ValueError):
        parse_archive_row(row)


@pytest.mark.parametrize("reverse_keys", [False, True])
def test_observed_explicit_zero_one_indices_preserve_prices_clocks_and_outcomes(reverse_keys):
    payload = json.loads((Path(__file__).parent / "fixtures/oddsportal/current_indexed_market.json").read_text())
    market = payload["d"]["oddsdata"]["back"][MARKET]
    if reverse_keys:
        market["outcomeId"] = dict(reversed(list(market["outcomeId"].items())))
        for field in ("openingOdd", "odds", "openingChangeTime", "changeTime"):
            market[field] = {bid: dict(reversed(list(pair.items()))) for bid, pair in market[field].items()}
    result = parse_detailed_odds(payload, "UqgCXjSb", BOOKIES)
    assert result["source_outcome_ids"] == ["b12t3x2s5a4x0xu0iqf", "b12t3x2s5a4x0xu0iqd"]
    assert result["openingsOdds"]["165"] == [1.72, 2.0]
    assert result["closingOdds"]["165"] == [1.6, 2.2]
    assert result["odds1_sts_open_at"] == datetime.fromtimestamp(1791085973, timezone.utc).isoformat()
    assert result["odds2_sts_open_at"] == result["odds1_sts_open_at"]
