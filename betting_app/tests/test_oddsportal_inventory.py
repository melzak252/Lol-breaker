"""Offline exhaustive inventory regressions; no detailed-odds routes are allowed."""
import asyncio
import base64
import csv
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from src.utils import oddsportal_collection as c
from src.utils.oddsportal_inventory import collect_inventory

FIXTURES = Path(__file__).parent / "fixtures/oddsportal"
LCS = c.GAME_ROOT + "league-of-legends-lcs/results/"
LEC = c.GAME_ROOT + "league-of-legends-lec/results/"
OLD = c.GAME_ROOT + "league-of-legends-lcs-2024/results/"
RENAMED = c.GAME_ROOT + "league-of-legends-lta-north-2025/results/"


def archive(url, *, event_id="AbCd1234", pages=1, seasons=None, home=None, bad_row=False):
    props = json.loads((FIXTURES / "current_archive_props.json").read_text())
    initial = props["initial"]
    path = httpx.URL(url).path
    slug = path.rstrip("/").split("/")[-2]
    row = initial["rows"][0]
    old_id = row["encodeEventId"]
    row["encodeEventId"] = event_id
    row["url"] = row["url"].replace(old_id, event_id)
    if home is not None:
        row["home-name"] = home
    row["tournament-url"] = slug
    row["breadcrumbs"]["tournament"]["url"] = path.rsplit("results/", 1)[0]
    initial["activeSeason"]["url"] = path
    initial["seasonTabs"] = seasons if seasons is not None else [{"href": path, "tabName": 2026}]
    initial["rows"] = [row] + ([{"encodeEventId": "bad"}] if bad_row else [])
    initial["pagination"].update(pageCount=pages, activePage=1)
    return props


def html(props):
    flight = "9d:" + json.dumps(["$", "$Larchive", None, props]) + "\n"
    return "<html><script>self.__next_f.push(" + json.dumps([1, flight]) + ")</script></html>"


def encrypted(payload):
    iv = bytes(range(16))
    key = hashlib.pbkdf2_hmac("sha256", c.PASSWORD.encode(), c.SALT.encode(), 1000, dklen=32)
    body = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(json.dumps(payload).encode(), 16))
    return base64.b64encode(base64.b64encode(body) + b":" + iv.hex().encode()).decode()


def execute(tmp_path, responses, **kwargs):
    calls = []
    def handler(request):
        url = str(request.url)
        calls.append(url)
        assert url in responses, f"Unexpected request (including odds): {url}"
        value = responses[url]
        if isinstance(value, httpx.Response):
            return value
        if isinstance(value, int):
            return httpx.Response(value, text="missing or denied")
        return httpx.Response(200, text=value)
    run = c.CollectionRun.create("inventory-test", tmp_path / "data", mode="archive", delay_seconds=0)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collect_inventory(client, run, **kwargs)
    manifest = asyncio.run(scenario())
    records = json.loads(run.json_path.read_text())
    ledger = json.loads(Path(manifest["inventory_coverage_ledger_path"]).read_text())
    return manifest, records, ledger, calls, run


@pytest.mark.parametrize("old_body", [404, "<html>404 Offside - page not found</html>"])
def test_failed_earlier_season_does_not_drop_renamed_season_or_later_league(tmp_path, old_body):
    tabs = [{"href": httpx.URL(OLD).path}, {"href": httpx.URL(RENAMED).path}, {"href": httpx.URL(LCS).path}]
    responses = {
        c.TOURNAMENTS_URL: '<a href="/pl/esports//league-of-legends-lcs/results/">LCS</a><a href="/pl/esports//league-of-legends-lec/results/">LEC</a>',
        LCS: html(archive(LCS, seasons=tabs)), OLD: old_body,
        RENAMED: html(archive(RENAMED, event_id="BcDe2345")),
        LEC: html(archive(LEC, event_id="CdEf3456")),
    }
    manifest, rows, ledger, calls, _ = execute(tmp_path, responses)
    assert {row["event_id"] for row in rows} == {"AbCd1234", "BcDe2345", "CdEf3456"}
    assert calls.count(LCS) == 1
    assert OLD in calls and RENAMED in calls and LEC in calls
    assert manifest["inventory_complete"] is False
    assert any(season["url"] == OLD and season["status"] == "failed" for season in ledger["seasons"])
    assert manifest["failures"][0]["capture"]["body_path"]


@pytest.mark.parametrize("localized", [False, True])
def test_all_advertised_pages_are_requested_once_and_duplicate_events_keep_occurrences(tmp_path, localized):
    props = archive(LCS, pages=3)
    initial = props["initial"]
    if localized:
        initial["pagination"]["url"] = c.BASE_URL + "/pl" + httpx.URL(initial["pagination"]["url"]).path
        initial["activeSeason"]["ajaxUrl"] = "/pl" + initial["activeSeason"]["ajaxUrl"]
    responses = {LCS: html(props)}
    for page in (2, 3):
        url = c.BASE_URL + "/proxy" + httpx.URL(initial["pagination"]["url"]).path.rstrip("/") + f"/page/{page}/"
        responses[url] = encrypted({"s": 1, "d": {
            "total": 3, "onePage": 1, "page": page, "rows": initial["rows"],
            "pagination": {**initial["pagination"], "activePage": page},
        }})
    manifest, rows, ledger, calls, run = execute(tmp_path, responses, archive_url=LCS)
    assert len(rows) == 1 and len(rows[0]["source_occurrences"]) == 3
    assert len(calls) == 3 and calls.count(LCS) == 1
    assert [page["page"] for page in ledger["seasons"][0]["pages"]] == [1, 2, 3]
    assert all(page["status"] == "completed" and page["row_count"] == 1 for page in ledger["seasons"][0]["pages"])
    assert manifest["inventory_complete"] is True
    progress = [json.loads(line) for line in Path(manifest["inventory_progress_path"]).read_text().splitlines()]
    assert len(progress) == 3
    sidecar = json.loads((run.derived_dir / "inventory-metadata.json").read_text())
    assert sidecar["inventory_complete"] is True
    assert sidecar["inventory_sha256"] == hashlib.sha256(run.json_path.read_bytes()).hexdigest()


def test_exact_id_legacy_union_preserves_differences_without_certifying_old_metadata_or_odds(tmp_path):
    path = tmp_path / "historical.csv"
    fields = ["url", "home_team", "away_team", "date", "avg_odds_home", "odds1_sts_open", "odds1_sts_open_at", "event_id"]
    originals = [
        dict(zip(fields, [c.BASE_URL + "/old-match-AbCd1234/", "Old home", "Old away", "2020-01-02 03:04:05", "2.0", "2.5", "2020-01-01T00:00:00Z", "AbCd1234"])),
        dict(zip(fields, [c.BASE_URL + "/other-match-ZzYy9876/", "Only legacy", "Other", "2019-01-01", "4.0", "4.5", "2020-01-01T00:00:00Z", "ZzYy9876"])),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([originals[0], originals[0], originals[1]])
    manifest, rows, _, calls, _ = execute(tmp_path, {LCS: html(archive(LCS))}, archive_url=LCS, known_events_csv=path)
    by_id = {row["event_id"]: row for row in rows}
    assert set(by_id) == {"AbCd1234", "ZzYy9876"}
    combined = by_id["AbCd1234"]
    assert combined["home_team"] != "Old home" and combined["date"].endswith("+00:00")
    assert len(combined["source_occurrences"]) == 3 and combined["metadata_conflicts"]
    assert combined["legacy_row"] == originals[0]
    assert by_id["ZzYy9876"]["date"] is None
    assert by_id["ZzYy9876"]["legacy_identity_certified"] is False
    for row in rows:
        assert row["avg_odds_home"] is None and row["odds1_sts_open"] is None and row["odds1_sts_open_at"] is None
        assert row["odds_status"] == "not_requested_inventory"
    snapshot = Path(manifest["known_events"]["snapshot_path"])
    assert snapshot.read_bytes() == path.read_bytes()
    assert manifest["known_events"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [LCS]


def test_access_denial_stops_requests_and_names_unattempted_coverage(tmp_path):
    tabs = [{"href": httpx.URL(LCS).path}, {"href": httpx.URL(OLD).path}, {"href": httpx.URL(RENAMED).path}]
    responses = {LCS: html(archive(LCS, seasons=tabs)), OLD: 403, RENAMED: html(archive(RENAMED))}
    manifest, rows, ledger, calls, _ = execute(tmp_path, responses, archive_url=LCS)
    assert calls == [LCS, OLD] and len(rows) == 1
    assert manifest["stop_reason"] == "access_denied" and manifest["inventory_complete"] is False
    pending = next(season for season in ledger["seasons"] if season["url"] == RENAMED)
    assert pending["attempted"] is False and pending["status"] == "not_attempted"


def test_real_empty_shell_is_retained_as_named_source_absence(tmp_path):
    props = json.loads((FIXTURES / "current_archive_empty_2015.json").read_text())
    url = c.BASE_URL + props["initial"]["activeSeason"]["url"]
    props["initial"]["seasonTabs"] = [{"href": props["initial"]["activeSeason"]["url"]}]
    manifest, rows, ledger, calls, _ = execute(tmp_path, {url: html(props)}, archive_url=url)
    assert rows == [] and calls == [url]
    assert manifest["source_absences"][0]["reason"] == "archive_season_no_source_rows"
    assert manifest["source_absences"][0]["observed_pagination"] == []
    assert ledger["seasons"][0]["status"] == "source_absent"
    assert ledger["seasons"][0]["capture"]["body_path"]


def test_invalid_individual_rows_are_named_and_valid_neighbors_survive(tmp_path):
    manifest, rows, ledger, _, _ = execute(tmp_path, {LCS: html(archive(LCS, bad_row=True))}, archive_url=LCS)
    assert len(rows) == 1 and manifest["inventory_complete"] is False
    failure = next(item for item in manifest["failures"] if item["stage"] == "archive_row")
    assert failure["source_row"] == {"encodeEventId": "bad"} and failure["row_index"] == 1
    assert failure["archive_capture"]["body_path"]
    assert ledger["seasons"][0]["pages"][0]["invalid_rows"] == 1


def test_explicit_limit_marks_inventory_incomplete_and_keeps_advertised_unattempted_pages(tmp_path):
    manifest, rows, ledger, calls, _ = execute(tmp_path, {LCS: html(archive(LCS, pages=3))}, archive_url=LCS, max_matches=1)
    assert len(rows) == 1 and calls == [LCS] and manifest["inventory_complete"] is False
    assert manifest["max_matches"] == 1
    assert [page["attempted"] for page in ledger["seasons"][0]["pages"]] == [True, False, False]


def test_bad_proxy_page_does_not_prevent_remaining_advertised_pages(tmp_path):
    props = archive(LCS, pages=3)
    initial = props["initial"]
    prefix = c.BASE_URL + "/proxy" + httpx.URL(initial["pagination"]["url"]).path.rstrip("/")
    page3_row = archive(LCS, event_id="BcDe2345")["initial"]["rows"]
    responses = {
        LCS: html(props), prefix + "/page/2/": encrypted({"s": 1, "d": {"rows": []}}),
        prefix + "/page/3/": encrypted({"s": 1, "d": {"page": 3, "total": 3, "onePage": 1, "rows": page3_row, "pagination": {**initial["pagination"], "activePage": 3}}}),
    }
    manifest, rows, ledger, calls, _ = execute(tmp_path, responses, archive_url=LCS)
    assert {row["event_id"] for row in rows} == {"AbCd1234", "BcDe2345"}
    assert prefix + "/page/3/" in calls and manifest["inventory_complete"] is False
    assert [page["status"] for page in ledger["seasons"][0]["pages"]] == ["completed", "failed", "completed"]


def test_failed_initial_league_does_not_abort_later_catalogue_leagues(tmp_path):
    responses = {
        c.TOURNAMENTS_URL: '<a href="/pl/esports//league-of-legends-lcs/results/">LCS</a><a href="/pl/esports//league-of-legends-lec/results/">LEC</a>',
        LCS: 404, LEC: html(archive(LEC, event_id="CdEf3456")),
    }
    manifest, rows, ledger, calls, _ = execute(tmp_path, responses)
    assert [row["event_id"] for row in rows] == ["CdEf3456"]
    assert calls == [c.TOURNAMENTS_URL, LCS, LEC]
    assert manifest["inventory_complete"] is False
    assert ledger["leagues"][1]["completed"] is True


def test_html_proxy_protocol_change_stops_remaining_pages(tmp_path):
    props = archive(LCS, pages=3)
    prefix = c.BASE_URL + "/proxy" + httpx.URL(props["initial"]["pagination"]["url"]).path.rstrip("/")
    manifest, rows, ledger, calls, _ = execute(
        tmp_path, {LCS: html(props), prefix + "/page/2/": "<html>unexpected archive protocol</html>"},
        archive_url=LCS,
    )
    assert len(rows) == 1 and calls == [LCS, prefix + "/page/2/"]
    assert manifest["stop_reason"] == "provider_protocol_changed"
    assert manifest["inventory_complete"] is False
    assert ledger["seasons"][0]["pages"][2]["attempted"] is False


def test_challenge_on_archive_stops_remaining_leagues(tmp_path):
    responses = {
        c.TOURNAMENTS_URL: '<a href="/pl/esports//league-of-legends-lcs/results/">LCS</a><a href="/pl/esports//league-of-legends-lec/results/">LEC</a>',
        LCS: "<html><title>Just a moment</title></html>",
    }
    manifest, rows, ledger, calls, _ = execute(tmp_path, responses)
    assert rows == [] and calls == [c.TOURNAMENTS_URL, LCS]
    assert manifest["stop_reason"] == "access_denied" and manifest["inventory_complete"] is False
    assert ledger["leagues"][1]["attempted"] is False


def test_season_advertisements_discovered_later_are_exhausted_transitively(tmp_path):
    responses = {
        LCS: html(archive(LCS, seasons=[{"href": httpx.URL(OLD).path}])),
        OLD: html(archive(OLD, event_id="BcDe2345", seasons=[{"href": httpx.URL(RENAMED).path}])),
        RENAMED: html(archive(RENAMED, event_id="CdEf3456")),
    }
    manifest, rows, _, calls, _ = execute(tmp_path, responses, archive_url=LCS)
    assert calls == [LCS, OLD, RENAMED]
    assert len(rows) == 3 and manifest["inventory_complete"] is True


def test_advertised_season_locale_redirect_keeps_capture_chain_and_actual_events(tmp_path):
    unlocalized = c.BASE_URL + httpx.URL(OLD).path.removeprefix("/pl")
    localized = OLD
    responses = {
        unlocalized: httpx.Response(301, headers={"location": localized}),
        localized: html(archive(OLD)),
    }
    manifest, rows, ledger, calls, run = execute(tmp_path, responses, archive_url=unlocalized)
    assert [row["event_id"] for row in rows] == ["AbCd1234"]
    assert manifest["inventory_complete"] is True
    assert calls == [unlocalized, localized]
    assert [capture["http_status"] for capture in run.captures] == [301, 200]
    assert ledger["seasons"][0]["capture"]["url"] == localized


def test_archive_locale_redirect_cannot_fetch_football_or_other_hosts(tmp_path):
    for target in ("/pl/football/results/", "https://invalid.example/pl" + httpx.URL(OLD).path):
        directory = tmp_path / ("football" if "football" in target else "external")
        directory.mkdir()
        manifest, rows, _, calls, _ = execute(directory, {
            OLD: httpx.Response(301, headers={"location": target}),
        }, archive_url=OLD)
        assert not rows and manifest["failure_count"] == 1
        assert calls == [OLD]


def test_mismatched_current_alias_does_not_hide_explicit_historical_season_links(tmp_path):
    tabs = [{"href": httpx.URL(OLD).path}, {"href": httpx.URL(RENAMED).path}]
    responses = {
        LCS: html(archive(RENAMED, event_id="ZzYy9876", seasons=tabs)),
        OLD: html(archive(OLD, event_id="BcDe2345")),
        RENAMED: html(archive(RENAMED, event_id="CdEf3456")),
    }
    manifest, rows, ledger, calls, _ = execute(tmp_path, responses, archive_url=LCS)
    assert {row["event_id"] for row in rows} == {"BcDe2345", "CdEf3456"}
    assert calls == [LCS, OLD, RENAMED]
    assert ledger["seasons"][0]["status"] == "failed"
    assert manifest["inventory_complete"] is False
