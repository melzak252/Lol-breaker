"""Offline contracts from retained current provider responses; not authentication."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.utils.oddsportal_current import parse_current_archive, parse_event_shell, parse_price_history

FIXTURES = Path(__file__).parent / "fixtures/oddsportal"
SHELL = json.loads((FIXTURES / "current_event_shell.json").read_text())
HISTORY = json.loads((FIXTURES / "current_price_history.json").read_text())
EVENT_ID = "ne8lT6bs"
OUTCOMES = ["9o8cbx2s5a4x0xqrp4b", "9o8cbx2s5a4x0xqrp4d"]
ARCHIVE = json.loads((FIXTURES / "current_archive_props.json").read_text())
ARCHIVE_URL = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"


def test_current_shell_keeps_provider_side_order_not_h2h_path_order():
    result = parse_event_shell(SHELL, EVENT_ID)
    assert (result["home_team"], result["away_team"]) == ("Sentinels", "Cloud9")
    assert result["event_id"] == EVENT_ID
    assert result["date"] == datetime.fromtimestamp(1776030300, timezone.utc).isoformat()
    assert result["results"] == {"home": "1", "away": "2"}
    assert result["winner"] == "away"
    assert result["sport_id"] == 36
    assert result["version_id"] == 1
    assert result["isFinished"] is True
    assert result["isPostponed"] is False
    assert result["source_event_data"] == SHELL["d"]["eventData"]
    assert result["source_event_body"] == SHELL["d"]["eventBody"]
    assert result["request_bases"]["requestBasePreMatch"] == "/match-event/"
    assert result["request_bases"]["requestBaseOddsHistory"] == "/match-event-history/"


@pytest.mark.parametrize("field,value", [
    ("id", "Other123"), ("id", None), ("sportId", 35), ("sportId", "36"),
    ("sportId", True), ("versionId", 0), ("versionId", -1),
    ("versionId", 1.0), ("versionId", True), ("versionId", None),
    ("home", ""), ("home", None), ("away", " sentinels "),
])
def test_shell_invalid_identity_and_version_fail_explicitly(field, value):
    payload = copy.deepcopy(SHELL)
    payload["d"]["eventData"][field] = value
    with pytest.raises(ValueError):
        parse_event_shell(payload, EVENT_ID)


@pytest.mark.parametrize("parser,args,payload", [
    (parse_event_shell, (EVENT_ID,), SHELL),
    (parse_price_history, (OUTCOMES, "165"), HISTORY),
])
@pytest.mark.parametrize("status", [0, 2, None, True, "1"])
def test_non_success_source_status_is_never_empty_success(parser, args, payload, status):
    payload = copy.deepcopy(payload)
    payload["s"] = status
    with pytest.raises(ValueError):
        parser(payload, *args)


@pytest.mark.parametrize("flag", [False, None, "true", 1])
def test_only_explicit_finished_state_can_derive_winner(flag):
    payload = copy.deepcopy(SHELL)
    payload["d"]["eventData"]["isFinished"] = flag
    assert parse_event_shell(payload, EVENT_ID)["winner"] is None


@pytest.mark.parametrize("scores,winner", [
    (("2", "1"), "home"), (("1", "2"), "away"), (("2", "2"), None),
    ((None, "2"), None), (("-", "2"), None), ((True, "2"), None),
    (("nan", "2"), None), (("inf", "2"), None), (("-1", "2"), None),
])
def test_finished_winner_requires_valid_nonnegative_unequal_scores(scores, winner):
    payload = copy.deepcopy(SHELL)
    payload["d"]["eventBody"].update(homeResult=scores[0], awayResult=scores[1])
    assert parse_event_shell(payload, EVENT_ID)["winner"] == winner


def test_missing_start_and_missing_scores_remain_unknown():
    payload = copy.deepcopy(SHELL)
    body = payload["d"]["eventBody"]
    for field in ("startDate", "homeResult", "awayResult"):
        del body[field]
    result = parse_event_shell(payload, EVENT_ID)
    assert result["date"] is None
    assert result["winner"] is None
    assert result["results"] == {"home": None, "away": None}


@pytest.mark.parametrize("epoch", [True, "1776030300", {}, float("nan"), float("inf"), 10**100])
def test_shell_bad_start_clock_is_not_capture_time(epoch):
    payload = copy.deepcopy(SHELL)
    payload["d"]["eventBody"]["startDate"] = epoch
    with pytest.raises(ValueError):
        parse_event_shell(payload, EVENT_ID)


@pytest.mark.parametrize("url", [
    "/esports/h2h/alpha-12345678/beta-87654321/#Other123",
    "https://example.org/event/#ne8lT6bs", "/event/#bad",
    "/esports/h2h/alpha-12345678/beta-87654321/", None,
])
def test_shell_present_prematch_locator_must_identify_requested_event(url):
    payload = copy.deepcopy(SHELL)
    payload["d"]["eventBody"]["tabs"]["eventDetail"]["prematch"]["url"] = url
    with pytest.raises(ValueError):
        parse_event_shell(payload, EVENT_ID)


def test_shell_absent_prematch_locator_does_not_invent_one():
    payload = copy.deepcopy(SHELL)
    del payload["d"]["eventBody"]["tabs"]
    assert parse_event_shell(payload, EVENT_ID)["event_id"] == EVENT_ID


def test_current_history_keeps_initial_rows_without_inventing_terminal_prices():
    result = parse_price_history(HISTORY, OUTCOMES, "165")
    assert len(result) == 2
    assert [row["odds"] for row in result] == [4.3, 1.18]
    assert [row["outcome_id"] for row in result] == OUTCOMES
    assert [row["outcome_index"] for row in result] == [0, 1]
    for row in result:
        assert row["bookmaker_id"] == "165"
        assert row["volume"] == 0
        assert row["source_epoch"] == 1775510590
        assert row["source_change_at"] == datetime.fromtimestamp(1775510590, timezone.utc).isoformat()
        assert row["source_row_index"] == 0
        assert row["source_row"] == HISTORY["d"]["history"]["back"][row["outcome_id"]]["165"][0]
        assert row["quote_timing_semantics"] == "provider_history_change_not_certified_complete"


def test_history_source_row_order_is_not_sorted_or_relabelled_as_open_close():
    payload = copy.deepcopy(HISTORY)
    rows = [["2.50", 12, 200], ["4.30", 0, 100], ["2.60", 1, 200]]
    payload["d"]["history"]["back"][OUTCOMES[0]]["165"] = rows
    result = parse_price_history(payload, OUTCOMES, "165")
    assert [row["source_row"] for row in result[:3]] == rows
    assert [row["source_row_index"] for row in result[:3]] == [0, 1, 2]
    assert [row["source_epoch"] for row in result[:3]] == [200, 100, 200]
    reversed_result = parse_price_history(payload, OUTCOMES[::-1], "165")
    assert reversed_result[0]["outcome_id"] == OUTCOMES[1]
    assert reversed_result[0]["outcome_index"] == 0


@pytest.mark.parametrize("patch", ["foreign_outcome", "foreign_provider", "missing_outcome", "missing_provider", "missing_back", "active_lay"])
def test_history_unknown_or_missing_identity_is_explicit_error(patch):
    payload = copy.deepcopy(HISTORY)
    back = payload["d"]["history"]["back"]
    if patch == "foreign_outcome":
        back["foreign"] = {"165": [["2.0", 0, 100]]}
    elif patch == "foreign_provider":
        back[OUTCOMES[0]]["502"] = [["2.0", 0, 100]]
    elif patch == "missing_outcome":
        del back[OUTCOMES[0]]
    elif patch == "missing_provider":
        del back[OUTCOMES[0]]["165"]
    elif patch == "missing_back":
        del payload["d"]["history"]["back"]
    else:
        payload["d"]["history"]["lay"] = {OUTCOMES[0]: {"165": [["2.0", 0, 100]]}}
    with pytest.raises(ValueError):
        parse_price_history(payload, OUTCOMES, "165")


@pytest.mark.parametrize("row", [
    ["1.0", 0, 100], ["0", 0, 100], ["nan", 0, 100], ["inf", 0, 100],
    [2.0, 0, 100], [True, 0, 100], [None, 0, 100],
    ["2.0", True, 100], ["2.0", -1, 100], ["2.0", None, 100],
    ["2.0", float("nan"), 100], ["2.0", "0", 100],
    ["2.0", 0, True], ["2.0", 0, "100"], ["2.0", 0, None],
    ["2.0", 0, float("inf")], ["2.0", 0, 10**100],
    ["2.0", 0], ["2.0", 0, 100, "extra"], {"odds": "2.0"},
])
def test_history_malformed_rows_do_not_disappear_into_partial_success(row):
    payload = copy.deepcopy(HISTORY)
    payload["d"]["history"]["back"][OUTCOMES[1]]["165"] = [row]
    with pytest.raises(ValueError):
        parse_price_history(payload, OUTCOMES, "165")


@pytest.mark.parametrize("outcomes,bookmaker", [
    ([OUTCOMES[0]], "165"), ([OUTCOMES[0], OUTCOMES[0]], "165"),
    ([OUTCOMES[0], None], "165"), ("ab", "165"), (OUTCOMES, 165),
    (OUTCOMES, ""), (OUTCOMES, " 165 "),
])
def test_history_request_identity_arguments_are_strict(outcomes, bookmaker):
    with pytest.raises(ValueError):
        parse_price_history(HISTORY, outcomes, bookmaker)


@pytest.mark.parametrize("parser,args", [(parse_event_shell, (EVENT_ID,)), (parse_price_history, (OUTCOMES, "165"))])
@pytest.mark.parametrize("payload", [None, [], {"s": 1}, {"s": 1, "d": []}])
def test_malformed_envelope_is_never_success(parser, args, payload):
    with pytest.raises(ValueError):
        parser(payload, *args)


def archive_markup(props, *, split=False, record_id="different-record"):
    record = record_id + ":" + json.dumps(["$", "$Labc", None, props]) + "\n"
    chunks = [record[:len(record) // 2], record[len(record) // 2:]] if split else [record]
    return "<html>" + "".join(
        "<script>self.__next_f.push(" + json.dumps([1, chunk]) + ")</script>"
        for chunk in chunks
    ) + "</html>"


@pytest.mark.parametrize("split", [False, True])
def test_archive_flight_reads_actual_props_without_record_id_or_chunk_assumptions(split):
    result = parse_current_archive(archive_markup(ARCHIVE, split=split), ARCHIVE_URL)
    assert result["rows"] == ARCHIVE["initial"]["rows"]
    assert len(result["rows"]) == 2
    assert result["seasons"] == ARCHIVE["initial"]["seasonTabs"]
    assert len(result["seasons"]) == 12
    assert result["pagination"] == ARCHIVE["initial"]["pagination"]
    assert result["active_season"] == ARCHIVE["initial"]["activeSeason"]
    assert result["source_initial"] == ARCHIVE["initial"]
    assert result["rows"][0]["odds"] == ARCHIVE["initial"]["rows"][0]["odds"]


@pytest.mark.parametrize("html", [
    "<html>Page not found</html>", "<script>self.__next_f.push([0])</script>",
    '<script>self.__next_f.push([1,"bad:unparseable\\n"])</script>',
    '<script>self.__next_f.push([1,"truncated"',
    "<script>self.__next_f.push(notJSON())</script>",
])
def test_missing_or_malformed_flight_archive_is_explicit_failure(html):
    with pytest.raises(ValueError):
        parse_current_archive(html, ARCHIVE_URL)


def test_two_archive_props_are_ambiguous_not_first_match_success():
    html = archive_markup(ARCHIVE) + archive_markup(ARCHIVE, record_id="other")
    with pytest.raises(ValueError):
        parse_current_archive(html, ARCHIVE_URL)


@pytest.mark.parametrize("field,value", [
    ("pageCount", 0), ("pageCount", True), ("pageCount", "2"),
    ("activePage", 0), ("activePage", 3), ("activePage", True),
])
def test_archive_pagination_requires_strict_positive_in_range_page(field, value):
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["pagination"][field] = value
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


@pytest.mark.parametrize("field,value", [
    ("rows", {}), ("pagination", None), ("activeSeason", []), ("seasonTabs", {}),
])
def test_archive_outer_shape_cannot_be_empty_success(field, value):
    props = copy.deepcopy(ARCHIVE)
    props["initial"][field] = value
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


@pytest.mark.parametrize("patch", [
    {"sport-id": 1}, {"sport-id": "36"}, {"sport-url-name": "football"},
    {"country-name": "Counter Strike"}, {"tournament-url": "another-tournament"},
])
def test_archive_foreign_row_context_rejects_whole_page(patch):
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["rows"][0].update(patch)
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


@pytest.mark.parametrize("part,url", [
    ("sport", "/football/"), ("country", "/esports/counter-strike/"),
    ("tournament", "/esports/league-of-legends/another-tournament/"),
])
def test_archive_foreign_breadcrumb_context_is_rejected(part, url):
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["rows"][0]["breadcrumbs"][part]["url"] = url
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


@pytest.mark.parametrize("url", [
    "https://example.org/pl/esports/league-of-legends/league-of-legends-lcs/results/",
    "https://user@www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/",
    "https://www.oddsportal.com/pl/football/england/premier-league/results/",
    "https://www.oddsportal.com/pl/esports/league-of-legends-lcs/results/",
])
def test_expected_archive_url_must_be_exact_provider_lol_league_locator(url):
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(ARCHIVE), url)


def test_archive_different_active_season_locator_is_not_the_requested_page():
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["activeSeason"]["url"] = "/esports/league-of-legends/another-league/results/"
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


def test_archive_keeps_invalid_individual_rows_for_collector_failure_accounting():
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["rows"][0]["encodeEventId"] = None
    props["initial"]["rows"].append(None)
    result = parse_current_archive(archive_markup(props), ARCHIVE_URL)
    assert result["rows"] == props["initial"]["rows"]


def test_empty_page_is_distinct_from_missing_archive():
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["rows"] = []
    result = parse_current_archive(archive_markup(props), ARCHIVE_URL)
    assert result["rows"] == []
    assert result["pagination"]["pageCount"] == 2
    assert result["archive_status"] == "empty_page"


@pytest.mark.parametrize("season", [2015, 2016])
def test_observed_empty_season_retains_explicit_source_absence(season):
    props = json.loads((FIXTURES / f"current_archive_empty_{season}.json").read_text())
    url = "https://www.oddsportal.com" + props["initial"]["activeSeason"]["url"]
    result = parse_current_archive(archive_markup(props), url)
    assert result["archive_status"] == "no_source_rows"
    assert result["rows"] == []
    assert result["pagination"] is None
    assert result["source_initial"] == props["initial"]
    assert result["source_initial"]["pagination"] == []
    assert result["active_season"]["name"] == season


@pytest.mark.parametrize("pagination", [None, {}, "", [1]])
def test_empty_rows_do_not_authorize_unknown_or_malformed_pagination(pagination):
    props = copy.deepcopy(ARCHIVE)
    props["initial"].update(rows=[], pagination=pagination)
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


def test_positive_rows_do_not_authorize_empty_list_pagination():
    props = copy.deepcopy(ARCHIVE)
    props["initial"]["pagination"] = []
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), ARCHIVE_URL)


@pytest.mark.parametrize("patch", ["foreign_sport", "missing_sport", "foreign_season", "missing_season_url"])
def test_observed_absence_requires_explicit_matching_source_context(patch):
    props = json.loads((FIXTURES / "current_archive_empty_2016.json").read_text())
    url = "https://www.oddsportal.com" + props["initial"]["activeSeason"]["url"]
    if patch == "foreign_sport":
        props["initial"]["sport"] = "football"
    elif patch == "missing_sport":
        del props["initial"]["sport"]
    elif patch == "foreign_season":
        props["initial"]["activeSeason"]["url"] = "/esports/league-of-legends/other/results/"
    else:
        del props["initial"]["activeSeason"]["url"]
    with pytest.raises(ValueError):
        parse_current_archive(archive_markup(props), url)
