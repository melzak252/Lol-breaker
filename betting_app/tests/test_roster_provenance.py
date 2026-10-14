"""Temporal team-lineage and announced-roster evidence contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from src.analysis.roster_provenance import load_roster_evidence, resolve_roster_evidence

UTC = timezone.utc
CUTOFF = datetime(2026, 9, 29, 10, tzinfo=UTC)
START = datetime(2026, 9, 29, 12, tzinfo=UTC)


def stamp(day: int, hour: int = 8) -> str:
    return datetime(2026, 9, day, hour, tzinfo=UTC).isoformat()


def link(source: str, target: str = "200", **updates):
    value = {
        "source_team_id": source,
        "target_team_id": target,
        "valid_from": stamp(1),
        "valid_to": None,
        "published_at": stamp(2),
        "observed_at": stamp(3),
        "evidence_uri": "https://example.invalid/lineage/1",
    }
    value.update(updates)
    return value


def announcement(match: str = "m1", team: str = "200", roster=None, **updates):
    value = {
        "match_id": match,
        "team_id": team,
        "roster": roster or ["p1", "p2", "p3", "p4", "p5"],
        "published_at": stamp(29, 7),
        "observed_at": stamp(29, 8),
        "evidence_uri": "https://example.invalid/announcement/1",
        "status": "announced",
    }
    value.update(updates)
    return value


def resolve(*, team="200", links=(), events=(), match="m1", cutoff=CUTOFF, start=START):
    return resolve_roster_evidence(
        team_id=team,
        match_id=match,
        cutoff_at=cutoff,
        start_at=start,
        identity_links=links,
        announcements=events,
    )


def test_temporally_available_valid_lineage_exposes_only_explicit_predecessor():
    evidence = link("100")
    result = resolve(links=[evidence])
    assert result["history_team_ids"] == ["200", "100"]
    assert result["identity_evidence"] == [evidence]
    assert result["announcement"] is None
    assert result["status"] == "resolved"


@pytest.mark.parametrize("clock_field", ["published_at", "observed_at"])
def test_lineage_unavailable_when_either_evidence_clock_is_after_cutoff(clock_field):
    evidence = link("100", **{clock_field: "2026-09-29T10:00:01+00:00"})
    result = resolve(links=[evidence])
    assert result["history_team_ids"] == ["200"]
    assert result["identity_evidence"] == []


def test_lineage_validity_scope_and_team_ids_are_exact():
    expired = link("old", valid_to=stamp(29))
    academy = link("academy", target="201")
    transfer = link("new-team", target="300")
    result = resolve(links=[expired, academy, transfer])
    assert result["history_team_ids"] == ["200"]
    assert result["identity_evidence"] == []


def test_duplicate_lineage_is_stable_and_conflicting_or_cyclic_lineage_is_ambiguous():
    evidence = link("100")
    result = resolve(links=[evidence, dict(evidence)])
    assert result["history_team_ids"] == ["200", "100"]
    assert result["identity_evidence"] == [evidence]
    conflicting = [link("100"), link("101")]
    assert resolve(links=conflicting)["status"] == "ambiguous"
    reversed_link = link("200", target="100")
    assert resolve(links=[reversed_link])["status"] == "ambiguous"
    cyclic = [link("100"), link("200", target="100")]
    assert resolve(links=cyclic)["status"] == "ambiguous"


def test_announcement_requires_both_clocks_and_exact_match_team():
    available = announcement()
    assert resolve(events=[available])["announcement"]["roster"] == available["roster"]
    future_publication = announcement(published_at="2026-09-29T10:01:00+00:00")
    future_observation = announcement(observed_at="2026-09-29T10:01:00+00:00")
    wrong_match = announcement(match_id="other")
    wrong_team = announcement(team="academy")
    assert resolve(events=[future_publication])["announcement"] is None
    assert resolve(events=[future_observation])["announcement"] is None
    assert resolve(events=[wrong_match, wrong_team])["announcement"] is None


def test_same_day_announcement_is_valid_if_clocks_precede_cutoff_and_start():
    same_day = announcement(published_at=stamp(29, 7), observed_at=stamp(29, 8))
    assert resolve(events=[same_day])["announcement"] == same_day


def test_announcement_after_start_or_without_timezone_is_rejected():
    after_start = announcement(
        published_at="2026-09-29T12:01:00+00:00",
        observed_at="2026-09-29T12:02:00+00:00",
    )
    with pytest.raises(ValueError):
        resolve(events=[after_start], cutoff=datetime(2026, 9, 29, 12, tzinfo=UTC))
    no_zone = announcement(published_at="2026-09-29T07:00:00")
    with pytest.raises(ValueError):
        resolve(events=[no_zone])


def test_duplicate_or_incomplete_announced_players_are_ambiguous_and_not_ready():
    duplicate = announcement(roster=["p1", "p2", "p3", "p4", "p4"])
    incomplete = announcement(roster=["p1", None, "p3", "p4", "p5"])
    duplicate_result = resolve(events=[duplicate])
    incomplete_result = resolve(events=[incomplete])
    assert duplicate_result["status"] == "ambiguous"
    assert duplicate_result["ambiguity_evidence"] == [duplicate]
    assert incomplete_result["status"] == "ambiguous"
    assert incomplete_result["ambiguity_evidence"] == [incomplete]


def test_newer_retraction_supersedes_older_announcement_without_resurrection():
    old = announcement()
    retracted = announcement(
        published_at="2026-09-29T09:00:00+00:00",
        observed_at="2026-09-29T09:30:00+00:00",
        status="retracted",
    )
    result = resolve(events=[old, retracted])
    assert result["announcement"] is None
    assert result["status"] == "ambiguous"
    assert result["missing_reason"]



def test_newer_retraction_without_roster_supersedes_older_announcement():
    old = announcement()
    retracted = announcement(
        published_at="2026-09-29T09:00:00+00:00",
        observed_at="2026-09-29T09:30:00+00:00",
        status="retracted",
    )
    del retracted["roster"]
    result = resolve(events=[old, retracted])
    assert result["announcement"] is None
    assert result["status"] == "ambiguous"
    assert result["missing_reason"] == "latest_announcement_retracted"
    assert result["ambiguity_evidence"] == [retracted]

def test_future_roster_retraction_cannot_change_earlier_resolution():
    initial = announcement()
    later = announcement(
        published_at="2026-09-29T10:30:00+00:00",
        observed_at="2026-09-29T10:45:00+00:00",
        status="retracted",
    )
    before_change = resolve(events=[initial])
    with_future_event = resolve(events=[initial, later])
    assert with_future_event == before_change


def test_equal_clock_conflicting_lineups_are_ambiguous_not_input_ordered():
    first = announcement(roster=["a", "b", "c", "d", "e"])
    second = announcement(roster=["f", "g", "h", "i", "j"])
    result = resolve(events=[second, first])
    assert result["status"] == "ambiguous"
    assert result["announcement"] is None
    assert result["ambiguity_evidence"] == [second, first] or set(
        item["roster"][0] for item in result["ambiguity_evidence"]
    ) == {"a", "f"}


def test_no_evidence_falls_back_to_original_id_only():
    result = resolve()
    assert result == {
        "history_team_ids": ["200"],
        "announcement": None,
        "identity_evidence": [],
        "status": "resolved",
        "missing_reason": None,
        "ambiguity_evidence": [],
    }


def test_loader_hashes_local_evidence_and_none_means_empty(tmp_path):
    assert load_roster_evidence(None) == {
        "identity_links": [], "announcements": [], "input_hashes": {}
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"identity_links": [link("100")], "announcements": []}))
    loaded = load_roster_evidence(path)
    assert loaded["identity_links"] == [link("100")]
    path.write_text(json.dumps({"identity_links": [], "announcements": []}))
    changed = load_roster_evidence(path)
    assert changed["input_hashes"] != loaded["input_hashes"]


@pytest.mark.parametrize("missing_id", [float("nan"), "nan", {"id": "p5"}])
def test_missing_or_structured_player_identity_is_not_an_announced_five(missing_id):
    event = announcement(roster=["p1", "p2", "p3", "p4", missing_id])
    result = resolve(events=[event])
    assert result["announcement"] is None
    assert result["status"] == "ambiguous"


def test_one_predecessor_cannot_identify_two_simultaneous_successors():
    links = [link("100", "200"), link("100", "300")]
    result = resolve(links=links)
    assert result["history_team_ids"] == ["200"]
    assert result["status"] == "ambiguous"
