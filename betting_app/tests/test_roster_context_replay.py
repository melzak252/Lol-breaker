"""Behavioral guards against cross-competition lineup carry-forward."""
from __future__ import annotations

import pandas as pd

from src.analysis.ev_asof_replay import build_asof_targets


def event(team, roster_prefix, day, competition, match):
    return {"team_id": team, "roster": [f"{roster_prefix}{i}" for i in range(5)],
            "effective_release_day": day, "original_map_day": day,
            "source_match_id": match, "source_series_order": (0, int(match)),
            "map_index": 1, "tournament_name": competition,
            "competition_context": competition, "competition_family": "LCK"}


def target(events, competition="Main League 2026", announcements=()):
    opportunities = pd.DataFrame([{"row_id": "origin", "golgg_match_id": "99",
                                   "decision_at": "2026-08-05T08:00:00Z",
                                   "start_at": "2026-08-05T09:00:00Z"}])
    metadata = pd.DataFrame([{"golgg_match_id": "99", "team1_id": "A", "team2_id": "B",
                              "date": "2026-08-05", "best_of": 3,
                              "tournament_name": competition, "competition_context": competition,
                              "competition_family": "LCK"}])
    return build_asof_targets(opportunities, metadata, events,
                              roster_evidence={"identity_links": [], "announcements": list(announcements)}).iloc[0]


def test_newer_same_family_different_competition_cannot_replace_target_context_lineup():
    events = [event("A", "main-a-", "2026-08-01", "LCK Spring 2026", "10"),
              event("B", "main-b-", "2026-08-01", "LCK Spring 2026", "10"),
              event("A", "cup-a-", "2026-08-03", "LCK Cup 2026", "20")]
    result = target(events, competition="LCK Spring 2026")
    assert result.roster_a == [f"main-a-{i}" for i in range(5)]
    assert result.roster_b == [f"main-b-{i}" for i in range(5)]
    assert result.roster_status == "prior_observed_scenario"


def test_no_matching_context_is_unresolved_instead_of_borrowing_another_five():
    result = target([event("A", "cup-a-", "2026-08-03", "Cup 2026", "20"),
                     event("B", "cup-b-", "2026-08-03", "Cup 2026", "20")])
    assert result.roster_status == "unresolved"
    assert result.roster_a == [None] * 5
    assert result.roster_b == [None] * 5


def test_missing_target_context_does_not_enable_unscoped_historical_fallback():
    result = target([event("A", "a-", "2026-08-03", "Cup 2026", "20"),
                     event("B", "b-", "2026-08-03", "Cup 2026", "20")], competition=None)
    assert result.roster_status == "unresolved"
 
def test_latest_incomplete_same_context_event_does_not_resurrect_older_roster():
    newest = event("A", "new-a-", "2026-08-03", "Main League 2026", "20")
    newest["roster"] = ["new-a-0", "new-a-1", None, "new-a-3", "new-a-4"]
    result = target([
        event("A", "old-a-", "2026-08-01", "Main League 2026", "10"),
        event("B", "main-b-", "2026-08-01", "Main League 2026", "10"),
        newest,
    ])
    assert result.roster_status == "unresolved"
    assert result.roster_a == ["new-a-0", "new-a-1", None, "new-a-3", "new-a-4"]


def test_valid_announcement_remains_available_without_target_competition_context():
    announcements = [
        {"match_id": "99", "team_id": "A", "roster": [f"a{i}" for i in range(5)],
         "published_at": "2026-08-05T07:00:00Z", "observed_at": "2026-08-05T07:05:00Z",
         "evidence_uri": "local://announcement/A"},
        {"match_id": "99", "team_id": "B", "roster": [f"b{i}" for i in range(5)],
         "published_at": "2026-08-05T07:00:00Z", "observed_at": "2026-08-05T07:05:00Z",
         "evidence_uri": "local://announcement/B"},
    ]
    result = target([event("A", "cup-a-", "2026-08-03", "Cup 2026", "20")],
                    competition=None, announcements=announcements)
    assert result.roster_status == "announced"
    assert result.roster_a == [f"a{i}" for i in range(5)]
    assert result.roster_b == [f"b{i}" for i in range(5)]
