from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.ev_asof_replay import (
    _feature_values,
    _replay_targets,
    build_asof_targets,
    build_roster_events,
    select_prior_roster,
)


def test_select_prior_roster_requires_release_strictly_before_cutoff_day() -> None:
    events = [
        {"team_id": "T", "competition_context": "League", "effective_release_day": "2026-04-01", "map_index": 1, "roster": [f"old-{i}" for i in range(5)]},
        {"team_id": "T", "competition_context": "League", "effective_release_day": "2026-04-02", "map_index": 1, "roster": [f"same-day-{i}" for i in range(5)]},
    ]

    selected = select_prior_roster(events, "T", "League", pd.Timestamp("2026-04-02").date())

    assert selected == [f"old-{i}" for i in range(5)]


def test_select_prior_roster_uses_stable_latest_released_map_order() -> None:
    events = [
        {"team_id": "T", "competition_context": "League", "effective_release_day": "2026-04-01", "source_series_order": 1, "map_index": 2, "roster": [f"later-map-{i}" for i in range(5)]},
        {"team_id": "T", "competition_context": "League", "effective_release_day": "2026-04-01", "source_series_order": 1, "map_index": 1, "roster": [f"earlier-map-{i}" for i in range(5)]},
        {"team_id": "T", "competition_context": "League", "effective_release_day": "2026-04-01", "source_series_order": 1, "map_index": 3, "roster": [f"latest-map-{i}" for i in range(5)]},
    ]

    assert select_prior_roster(events, "T", "League", pd.Timestamp("2026-04-02").date()) == [f"latest-map-{i}" for i in range(5)]


def test_build_asof_targets_ignores_target_future_label_draft_and_roster_mutations() -> None:
    opportunities = pd.DataFrame(
        [{"row_id": "r1", "golgg_match_id": "m1", "horizon": "48h", "decision_at": "2026-04-02T00:00:00Z", "date": "2026-04-04"}]
    )
    metadata = pd.DataFrame(
        [{"golgg_match_id": "m1", "team1_id": "A", "team2_id": "B", "best_of": 3, "y": 1, "tournament_name": "League", "roster_a": ["future-a"], "roster_b": ["future-b"], "draft": ["future-pick"]}]
    )
    events = [
        {"team_id": "A", "competition_context": "League", "effective_release_day": "2026-04-01", "map_index": 1, "roster": ["a1", "a2", "a3", "a4", "a5"]},
        {"team_id": "B", "competition_context": "League", "effective_release_day": "2026-04-01", "map_index": 1, "roster": ["b1", "b2", "b3", "b4", "b5"]},
    ]
    before = build_asof_targets(opportunities, metadata, events)
    changed_metadata = metadata.copy(deep=True)
    changed_metadata.loc[0, "y"] = 0
    changed_metadata.at[0, "roster_a"] = ["changed", "future", "lineup", "must", "not-leak"]
    changed_metadata.at[0, "draft"] = ["changed-pick"]
    changed_events = [*events, {"team_id": "A", "competition_context": "League", "effective_release_day": "2026-04-04", "map_index": 1, "roster": [f"after-{i}" for i in range(5)]}]
    after = build_asof_targets(opportunities, changed_metadata, changed_events)
    before_raw, before_hist, *_ = _run_targets(before, events)
    after_raw, after_hist, *_ = _run_targets(after, changed_events)

    assert before.loc[0, "roster_a"] == after.loc[0, "roster_a"] == ["a1", "a2", "a3", "a4", "a5"]
    assert before.loc[0, "roster_b"] == after.loc[0, "roster_b"] == ["b1", "b2", "b3", "b4", "b5"]
    assert before.loc[0, "draft_status"] == after.loc[0, "draft_status"] == "unknown"
    assert before.loc[0, "y"] != after.loc[0, "y"]
    for key in before_raw:
        assert np.array_equal(before_raw[key], after_raw[key])
    for key in before_hist:
        assert np.array_equal(before_hist[key], after_hist[key])



def test_select_prior_roster_keeps_transfer_and_substitute_events_team_scoped() -> None:
    events = [
        {"team_id": "A", "competition_context": "League", "effective_release_day": "2026-04-01", "map_index": 1, "roster": ["p1", "p2", "p3", "p4", "p5"]},
        {"team_id": "A", "competition_context": "League", "effective_release_day": "2026-04-03", "map_index": 1, "roster": ["p1", "p2", "p3", "p4", "sub"]},
        {"team_id": "B", "competition_context": "League", "effective_release_day": "2026-04-03", "map_index": 1, "roster": ["p5", "q2", "q3", "q4", "q5"]},
    ]
    cutoff = pd.Timestamp("2026-04-04").date()

    assert select_prior_roster(events, "A", "League", cutoff) == ["p1", "p2", "p3", "p4", "sub"]
    assert select_prior_roster(events, "B", "League", cutoff) == ["p5", "q2", "q3", "q4", "q5"]


def test_build_asof_targets_uses_cutoff_year_and_explicit_unknown_roster_slots() -> None:
    opportunities = pd.DataFrame(
        [
            {"row_id": "r25", "golgg_match_id": "m25", "horizon": "OPEN", "decision_at": "2025-12-31T23:00:00Z", "date": "2026-01-01"},
            {"row_id": "r26", "golgg_match_id": "m26", "horizon": "OPEN", "decision_at": "2026-01-01T00:00:00Z", "date": "2026-01-02"},
        ]
    )
    metadata = pd.DataFrame(
        [
            {"golgg_match_id": "m25", "team1_id": "A", "team2_id": "B", "best_of": 3, "y": 1, "tournament_name": "League"},
            {"golgg_match_id": "m26", "team1_id": "C", "team2_id": "D", "best_of": 5, "y": 0, "tournament_name": "League"},
        ]
    )
    targets = build_asof_targets(opportunities, metadata, [])

    assert targets["model_year"].tolist() == [2025, 2026]
    assert targets.loc[0, "roster_a"] == [None] * 5
    assert targets.loc[0, "roster_b"] == [None] * 5
    assert targets.loc[1, "roster_a"] == [None] * 5
    assert targets.loc[1, "roster_b"] == [None] * 5
    assert targets["roster_status"].tolist() == ["unresolved", "unresolved"]


def test_build_asof_targets_marks_transferred_player_overlap_unavailable() -> None:
    opportunities = pd.DataFrame(
        [{"row_id": "r1", "golgg_match_id": "m1", "decision_at": "2026-04-04T00:00:00Z"}]
    )
    metadata = pd.DataFrame(
        [{"golgg_match_id": "m1", "team1_id": "A", "team2_id": "B", "best_of": 3, "y": 1, "tournament_name": "League"}]
    )
    events = [
        {"team_id": "A", "competition_context": "League", "effective_release_day": "2026-04-01", "map_index": 1, "roster": ["shared", "a2", "a3", "a4", "a5"]},
        {"team_id": "B", "competition_context": "League", "effective_release_day": "2026-04-02", "map_index": 1, "roster": ["shared", "b2", "b3", "b4", "b5"]},
    ]

    target = build_asof_targets(opportunities, metadata, events).iloc[0]

    assert target["roster_status"] == "unresolved"
    assert target["overlapping_prior_player_ids"] == ["shared"]


def test_build_roster_events_applies_optional_release_overrides_with_source_day_floor() -> None:
    def series(match_id: str) -> dict[str, object]:
        games = []
        for day in ("2026-04-01", "2026-04-03"):
            games.append({
                "date": day,
                "t1_id": "A",
                "t2_id": "B",
                "t1_players": {f"role-{slot}": {"player_id": f"{match_id}-a{slot}"} for slot in range(5)},
                "t2_players": {f"role-{slot}": {"player_id": f"{match_id}-b{slot}"} for slot in range(5)},
            })
        return {"match_id": match_id, "tournament_name": "League", "games": games}

    matches = [series("later"), series("floor"), series("fallback")]
    events = build_roster_events(matches, {"later": "2026-04-05", "floor": "2026-04-02"})
    by_series = {
        match_id: {event["effective_release_day"] for event in events if event["source_match_id"] == match_id}
        for match_id in ("later", "floor", "fallback")
    }
    reasons = {
        match_id: {event["release_source"] for event in events if event["source_match_id"] == match_id}
        for match_id in ("later", "floor", "fallback")
    }

    assert by_series == {
        "later": {"2026-04-05"},
        "floor": {"2026-04-03"},
        "fallback": {"2026-04-03"},
    }
    assert {event["competition_context"] for event in events} == {"League"}
    assert reasons == {
        "later": {"corrected_release_map_override"},
        "floor": {"corrected_release_map_override"},
        "fallback": {"latest_source_game_day_fallback"},
    }


def _player_code(player_id: str) -> float:
    return float(sum((position + 1) * ord(character) for position, character in enumerate(player_id)))


class _ReplayState:
    def __init__(self) -> None:
        self.raw_query_rosters: list[tuple[list[str], list[str]]] = []
        self.pair_calls = 0
        gl = SimpleNamespace(player_ratings={})
        base = SimpleNamespace(last_day=None, last_order=None, manager=SimpleNamespace(systems={"gl": gl}))
        self.base = SimpleNamespace(base=base, pair=self.pair)
        self.histories: dict[str, object] = {}
        self.vocabulary: dict[str, int] = {}

    def pair(self, team_a, roster_a, team_b, roster_b, cutoff, best_of):
        self.pair_calls += 1
        self.raw_query_rosters.append((list(roster_a), list(roster_b)))
        raw_rosters = (sorted(roster_a), sorted(roster_b))
        players = np.zeros((2, 5, 21), dtype=np.float32)
        for side, roster in enumerate(raw_rosters):
            for slot, player_id in enumerate(roster):
                players[side, slot, :] = _player_code(player_id)
        raw = {
            "players": players,
            "core": np.zeros((2, 5), dtype=np.float32),
            "team": np.zeros((2, 11), dtype=np.float32),
            "w20": np.zeros((2, 10), dtype=np.float32),
            "gates": np.zeros((2, 2), dtype=np.float32),
        }
        proof = [
            {"team_id": team_a, "roster": raw_rosters[0]},
            {"team_id": team_b, "roster": raw_rosters[1]},
        ]
        return raw, proof

    def player(self, player_id, cutoff):
        history = np.zeros((16, 33), dtype=np.float32)
        history[0, 0] = _player_code(player_id)
        mask = np.zeros(16, dtype=np.uint8)
        mask[0] = 1
        return {
            "history": history,
            "mask": mask,
            "champions": np.zeros(16, dtype=np.uint16),
        }

    def consume(self, item, metadata, extra) -> None:
        self.base.base.last_day = item.games[-1].day
        self.base.base.last_order = item.order


class _HistoryItem:
    def __init__(self, identifier: str, released_day: str) -> None:
        self.identifier = identifier
        self.order = identifier
        self.games = [SimpleNamespace(day=pd.Timestamp(released_day).date())]


def _roster_event(team_id: str, source_match_id: str, roster: list[str], day: str = "2026-04-01"):
    return {
        "team_id": team_id,
        "source_match_id": source_match_id,
        "effective_release_day": day,
        "original_map_day": day,
        "source_series_order": 1,
        "map_index": 1,
        "competition_context": "League",
        "roster": roster,
    }


def _run_targets(targets, events):
    replay_events = []
    for index, source in enumerate(events):
        event = dict(source)
        event.setdefault("source_match_id", f"test-history-{index}")
        replay_events.append(event)
    by_match: dict[str, str] = {}
    for event in replay_events:
        by_match[event["source_match_id"]] = event["effective_release_day"]
    series = [_HistoryItem(identifier, day) for identifier, day in by_match.items()]
    state = _ReplayState()
    raw, hist, rows, receipt = _replay_targets(
        targets,
        series,
        {item.identifier: {} for item in series},
        {item.identifier: [] for item in series},
        replay_events,
        {"TemporalState": lambda: state, "champion_vocabulary": {"": 0}},
    )
    return raw, hist, pd.DataFrame(rows), receipt, state


def _opportunities_and_metadata(team_pairs, match_ids=None):
    match_ids = match_ids or [f"m{index}" for index in range(len(team_pairs))]
    opportunities = pd.DataFrame([
        {
            "row_id": f"r{index}",
            "golgg_match_id": match_id,
            "decision_at": "2026-04-04T00:00:00Z",
            "start_at": "2026-04-05T00:00:00Z",
        }
        for index, match_id in enumerate(match_ids)
    ])
    metadata = pd.DataFrame([
        {
            "golgg_match_id": match_id,
            "team1_id": team_a,
            "team2_id": team_b,
            "best_of": 3,
            "y": index % 2,
            "tournament_name": "League",
        }
        for index, (match_id, (team_a, team_b)) in enumerate(zip(match_ids, team_pairs))
    ])
    return opportunities, metadata


def _announcement(match_id: str, team_id: str, roster: list[str], published_at="2026-04-03T10:00:00Z"):
    return {
        "match_id": match_id,
        "team_id": team_id,
        "roster": roster,
        "published_at": published_at,
        "observed_at": "2026-04-03T10:05:00Z",
        "evidence_uri": f"local://rosters/{match_id}-{team_id}",
    }


def test_released_target_is_retained_without_roster_lookup_or_snapshot(monkeypatch) -> None:
    from src.analysis import ev_asof_replay as replay

    opportunities, metadata = _opportunities_and_metadata([("A", "B")], ["T"])
    events = [
        _roster_event("A", "T", [f"a{i}" for i in range(5)], "2026-04-03"),
        _roster_event("B", "T", [f"b{i}" for i in range(5)], "2026-04-03"),
    ]

    def forbidden_lookup(*args, **kwargs):
        pytest.fail("A released target must be withheld before roster resolution or lookup")

    monkeypatch.setattr(replay, "_resolve_roster_side", forbidden_lookup)
    monkeypatch.setattr(replay, "_select_prior_event_for_teams", forbidden_lookup)
    targets = build_asof_targets(opportunities, metadata, events)
    raw, hist, replayed, receipt, state = _run_targets(targets, events)

    assert replayed["row_id"].tolist() == ["r0"]
    assert not bool(replayed.loc[0, "forecast_eligible"])
    assert replayed.loc[0, "forecast_absence"] == "target_already_released_before_cutoff"
    assert replayed.loc[0, "missing_reason"] == "target_already_released_before_cutoff"
    assert replayed.loc[0, "target_native_effective_release_day"] == "2026-04-03"
    assert replayed.loc[0, "roster_a"] == [None] * 5
    assert replayed.loc[0, "roster_b"] == [None] * 5
    assert replayed.loc[0, "roster_source_a"]["match_id"] is None
    assert replayed.loc[0, "roster_source_b"]["match_id"] is None
    assert not bool(replayed.loc[0, "feature_available"])
    assert receipt["feature_unavailable_rows"] == 1
    assert state.pair_calls == 0
    assert all(np.isnan(array).all() for array in raw.values())
    assert np.isnan(hist["history"]).all()


@pytest.mark.parametrize("release_field", [
    "effective_label_release_day", "label_release_day", "result_release_day",
])
def test_explicit_target_label_release_withholds_before_roster_lookup(monkeypatch, release_field) -> None:
    from src.analysis import ev_asof_replay as replay

    opportunities, metadata = _opportunities_and_metadata([("A", "B")], ["T"])
    metadata[release_field] = "2026-04-03"
    events = [
        _roster_event("A", "prior-a", [f"a{i}" for i in range(5)]),
        _roster_event("B", "prior-b", [f"b{i}" for i in range(5)]),
    ]

    def forbidden_lookup(*args, **kwargs):
        pytest.fail("Explicit target-label release must precede roster lookup")

    monkeypatch.setattr(replay, "_select_prior_event_for_teams", forbidden_lookup)
    targets = build_asof_targets(opportunities, metadata, events)
    raw, hist, replayed, _, state = _run_targets(targets, events)

    assert replayed.loc[0, "forecast_absence"] == "target_already_released_before_cutoff"
    assert replayed.loc[0, "target_effective_label_release_day"] == "2026-04-03"
    assert replayed.loc[0, "target_native_effective_release_day"] is None
    assert replayed.loc[0, "missing_reason"] == "target_already_released_before_cutoff"
    assert state.pair_calls == 0
    assert all(np.isnan(array).all() for array in raw.values())
    assert np.isnan(hist["history"]).all()


@pytest.mark.parametrize("label_day", [None, "2026-04-04", "2026-04-08"])
@pytest.mark.parametrize("outcome", [0, 1])
def test_independent_native_release_inventory_blocks_even_unreleased_label(label_day, outcome) -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")], ["T"])
    metadata["effective_label_release_day"] = label_day
    metadata["y"] = outcome
    events = [
        _roster_event("A", "prior-a", [f"a{i}" for i in range(5)]),
        _roster_event("B", "prior-b", [f"b{i}" for i in range(5)]),
    ]
    targets = build_asof_targets(
        opportunities, metadata, events,
        target_native_release_days={"T": pd.Timestamp("2026-04-03").date()},
    )

    assert not bool(targets.loc[0, "forecast_eligible"])
    assert targets.loc[0, "forecast_absence"] == "target_already_released_before_cutoff"
    assert targets.loc[0, "target_native_effective_release_day"] == "2026-04-03"
    assert targets.loc[0, "roster_a"] == [None] * 5
    assert targets.loc[0, "roster_b"] == [None] * 5


@pytest.mark.parametrize("target_release", ["2026-04-04", "2026-04-06"])
def test_same_day_and_later_target_release_remain_forecast_eligible_without_self_history(target_release) -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")], ["T"])
    opportunities["decision_at"] = "2026-04-03T20:00:00-04:00"
    metadata["effective_label_release_day"] = target_release
    prior_a, prior_b = [f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)]
    events = [
        _roster_event("A", "prior", prior_a),
        _roster_event("B", "prior", prior_b),
        _roster_event("A", "T", [f"self-a{i}" for i in range(5)], target_release),
        _roster_event("B", "T", [f"self-b{i}" for i in range(5)], target_release),
    ]
    targets = build_asof_targets(opportunities, metadata, events)
    raw, hist, replayed, _, state = _run_targets(targets, events)

    assert bool(replayed.loc[0, "forecast_eligible"])
    assert replayed.loc[0, "forecast_absence"] is None
    assert bool(replayed.loc[0, "feature_available"])
    assert replayed.loc[0, "roster_source_a"]["match_id"] == "prior"
    assert replayed.loc[0, "roster_source_b"]["match_id"] == "prior"
    assert replayed.loc[0, "feature_source_max_day"] == "2026-04-01"
    assert state.raw_query_rosters == [(prior_a, prior_b)]
    assert state.base.base.last_order == "prior"
    assert all(np.isfinite(array).all() for array in raw.values())
    assert np.isfinite(hist["history"]).all()


@pytest.mark.parametrize("release_clock", ["native", "label"])
def test_direct_replay_bypass_still_withholds_released_target_and_preserves_absence_precedence(release_clock) -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")], ["T"])
    prior_events = [
        _roster_event("A", "prior", [f"a{i}" for i in range(5)]),
        _roster_event("B", "prior", [f"b{i}" for i in range(5)]),
    ]
    targets = build_asof_targets(opportunities, metadata, prior_events)
    targets["forecast_eligible"] = True
    targets["forecast_absence"] = None
    targets["roster_missing_reason"] = "an_unrelated_roster_error"
    events = list(prior_events)
    if release_clock == "native":
        events.extend([
            _roster_event("A", "T", [f"self-a{i}" for i in range(5)], "2026-04-03"),
            _roster_event("B", "T", [f"self-b{i}" for i in range(5)], "2026-04-03"),
        ])
    else:
        targets["result_release_day"] = "2026-04-03"
    raw, hist, replayed, _, state = _run_targets(targets, events)

    assert not bool(replayed.loc[0, "feature_available"])
    assert not bool(replayed.loc[0, "forecast_eligible"])
    assert replayed.loc[0, "forecast_absence"] == "target_already_released_before_cutoff"
    assert replayed.loc[0, "missing_reason"] == "target_already_released_before_cutoff"
    assert state.pair_calls == 0
    assert all(np.isnan(array).all() for array in raw.values())
    assert np.isnan(hist["history"]).all()


def test_released_target_history_remains_available_to_later_other_target() -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B"), ("A", "B")], ["T", "OTHER"])
    opportunities.loc[1, "decision_at"] = "2026-04-06T00:00:00Z"
    opportunities.loc[1, "start_at"] = "2026-04-07T00:00:00Z"
    roster_a, roster_b = [f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)]
    events = [
        _roster_event("A", "T", roster_a, "2026-04-03"),
        _roster_event("B", "T", roster_b, "2026-04-03"),
    ]
    targets = build_asof_targets(opportunities, metadata, events)
    raw, hist, replayed, receipt, state = _run_targets(targets, events)

    assert replayed["golgg_match_id"].tolist() == ["T", "OTHER"]
    assert replayed["feature_available"].tolist() == [False, True]
    assert replayed.loc[0, "missing_reason"] == "target_already_released_before_cutoff"
    assert replayed.loc[1, "roster_source_a"]["match_id"] == "T"
    assert replayed.loc[1, "roster_source_b"]["match_id"] == "T"
    assert replayed.loc[1, "feature_source_max_day"] == "2026-04-03"
    assert state.raw_query_rosters == [(roster_a, roster_b)]
    assert receipt["feature_available_rows"] == 1
    assert all(np.isnan(array[0]).all() and np.isfinite(array[1]).all() for array in raw.values())
    assert np.isnan(hist["history"][0]).all()
    assert np.isfinite(hist["history"][1]).all()


def test_unknown_target_release_and_minimal_roster_events_remain_forecast_eligible() -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")], ["T"])
    metadata["date"] = "2026-04-01"
    metadata["result_day"] = "2026-04-01"
    events = [
        _roster_event("A", "prior-a", [f"a{i}" for i in range(5)]),
        _roster_event("B", "prior-b", [f"b{i}" for i in range(5)]),
    ]
    for event in events:
        event.pop("source_match_id")
    targets = build_asof_targets(opportunities, metadata, iter(events))
    _, _, replayed, _, state = _run_targets(targets, events)

    assert bool(replayed.loc[0, "forecast_eligible"])
    assert replayed.loc[0, "forecast_absence"] is None
    assert replayed.loc[0, "target_native_effective_release_day"] is None
    assert replayed.loc[0, "target_effective_label_release_day"] is None
    assert bool(replayed.loc[0, "feature_available"])
    assert state.pair_calls == 1


def test_feature_values_align_arbitrary_serialization_and_lexical_string_id_order() -> None:
    target = {
        "team1_id": "team-10",
        "team2_id": "team-2",
        "cutoff_day": "2026-04-04",
        "best_of": 3,
    }
    roster_a = ["2", "10", "4", "3", "5"]
    roster_b = ["b5", "b1", "b4", "b2", "b3"]
    original_a, original_b = list(roster_a), list(roster_b)
    state = _ReplayState()

    raw, hist, proof = _feature_values(state, target, (roster_a, roster_b))
    permuted_raw, permuted_hist, permuted_proof = _feature_values(
        state, target, (["5", "3", "10", "2", "4"], ["b3", "b2", "b1", "b5", "b4"])
    )

    assert proof["query_roster_a"] == ["10", "2", "3", "4", "5"]
    assert proof["query_roster_b"] == ["b1", "b2", "b3", "b4", "b5"]
    assert state.raw_query_rosters[-2:] == [
        (proof["query_roster_a"], proof["query_roster_b"]),
        (proof["query_roster_a"], proof["query_roster_b"]),
    ]
    assert proof["query_roster_a"] == permuted_proof["query_roster_a"]
    for key in raw:
        assert np.array_equal(raw[key], permuted_raw[key])
    for key in hist:
        assert np.array_equal(hist[key], permuted_hist[key])
    assert raw["players"][0, :, 0].tolist() == [
        _player_code(player_id) for player_id in proof["query_roster_a"]
    ]
    assert hist["history"][0, :, 0, 0].tolist() == raw["players"][0, :, 0].tolist()
    assert roster_a == original_a
    assert roster_b == original_b


def test_feature_values_joint_side_reversal_swaps_tensors_without_mutating_rosters() -> None:
    team_a = ["a5", "a2", "a4", "a1", "a3"]
    team_b = ["b5", "b2", "b4", "b1", "b3"]
    saved_a, saved_b = list(team_a), list(team_b)
    forward_state = _ReplayState()
    reverse_state = _ReplayState()
    forward_raw, forward_hist, _ = _feature_values(
        forward_state,
        {"team1_id": "A", "team2_id": "B", "cutoff_day": "2026-04-04", "best_of": 3},
        (team_a, team_b),
    )
    reverse_raw, reverse_hist, _ = _feature_values(
        reverse_state,
        {"team1_id": "B", "team2_id": "A", "cutoff_day": "2026-04-04", "best_of": 3},
        (team_b, team_a),
    )

    for key in forward_raw:
        assert np.array_equal(forward_raw[key][::-1], reverse_raw[key])
    for key in forward_hist:
        assert np.array_equal(forward_hist[key][::-1], reverse_hist[key])
    assert team_a == saved_a
    assert team_b == saved_b


def test_unresolved_one_or_both_lineups_and_overlapping_players_stay_unavailable() -> None:
    pairs = [("A", "B"), ("C", "D"), ("E", "F")]
    opportunities, metadata = _opportunities_and_metadata(pairs)
    events = [
        _roster_event("A", "hist-a", ["a1", "a2", "a3", "a4", "a5"]),
        _roster_event("E", "hist-e", ["shared", "e2", "e3", "e4", "e5"]),
        _roster_event("F", "hist-f", ["shared", "f2", "f3", "f4", "f5"]),
    ]
    targets = build_asof_targets(opportunities, metadata, events)

    raw, hist, replayed, receipt, state = _run_targets(targets, events)

    assert replayed["feature_available"].tolist() == [False, False, False]
    assert replayed["missing_reason"].notna().all()
    assert replayed["roster_status"].tolist() == ["unresolved", "unresolved", "unresolved"]
    assert replayed.loc[0, "roster_a"] == ["a1", "a2", "a3", "a4", "a5"]
    assert replayed.loc[0, "roster_b"] == [None] * 5
    assert replayed.loc[1, "roster_a"] == [None] * 5
    assert replayed.loc[1, "roster_b"] == [None] * 5
    assert replayed.loc[2, "overlapping_prior_player_ids"] == ["shared"]
    assert state.pair_calls == 0
    assert receipt["feature_unavailable_rows"] == 3
    assert all(np.isnan(array).all() for array in raw.values())
    assert all(np.isnan(array).all() for key, array in hist.items() if key == "history")


def test_complete_announced_lineups_on_new_organizations_are_forecastable() -> None:
    opportunities, metadata = _opportunities_and_metadata([("new-org-a", "new-org-b")])
    roster_a = ["2", "10", "4", "3", "5"]
    roster_b = ["b5", "b1", "b4", "b2", "b3"]
    evidence = {
        "identity_links": [],
        "announcements": [
            _announcement("m0", "new-org-a", roster_a),
            _announcement("m0", "new-org-b", roster_b),
        ],
        "input_hashes": {},
    }
    targets = build_asof_targets(opportunities, metadata, [], roster_evidence=evidence)

    raw, hist, replayed, receipt, state = _run_targets(targets, [])

    assert replayed.loc[0, "roster_status"] == "announced"
    assert replayed.loc[0, "roster_a"] == roster_a
    assert replayed.loc[0, "roster_b"] == roster_b
    assert replayed.loc[0, "snapshot_query_roster_a"] == ["10", "2", "3", "4", "5"]
    assert replayed.loc[0, "snapshot_query_roster_b"] == ["b1", "b2", "b3", "b4", "b5"]
    assert bool(replayed.loc[0, "feature_available"])
    assert state.pair_calls == 1
    assert receipt["feature_available_rows"] == 1
    assert all(np.isfinite(array).all() for array in raw.values())
    assert all(np.isfinite(array).all() for array in hist.values())


def test_valid_announcements_override_prior_observed_lineups() -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")])
    prior_events = [
        _roster_event("A", "prior-a", ["old-a1", "old-a2", "old-a3", "old-a4", "old-a5"]),
        _roster_event("B", "prior-b", ["old-b1", "old-b2", "old-b3", "old-b4", "old-b5"]),
    ]
    announced_a = ["new-a1", "new-a2", "new-a3", "new-a4", "new-a5"]
    announced_b = ["new-b1", "new-b2", "new-b3", "new-b4", "new-b5"]
    evidence = {
        "identity_links": [],
        "announcements": [
            _announcement("m0", "A", announced_a),
            _announcement("m0", "B", announced_b),
        ],
        "input_hashes": {},
    }

    target = build_asof_targets(
        opportunities, metadata, prior_events, roster_evidence=evidence
    ).iloc[0]

    assert target["roster_status"] == "announced"
    assert target["roster_a"] == announced_a
    assert target["roster_b"] == announced_b
    assert target["roster_source_a"]["source_type"] == "announced"
    assert target["roster_source_b"]["source_type"] == "announced"
    assert target["roster_announcement_a"]["published_at"] == "2026-04-03T10:00:00Z"
    assert target["roster_announcement_a"]["observed_at"] == "2026-04-03T10:05:00Z"
    assert target["roster_source_a"]["evidence_uri"] == "local://rosters/m0-A"


def test_local_roster_evidence_path_is_hashed_on_target_rows(tmp_path) -> None:
    opportunities, metadata = _opportunities_and_metadata([("new-a", "new-b")])
    evidence = {
        "identity_links": [],
        "announcements": [
            _announcement("m0", "new-a", ["a1", "a2", "a3", "a4", "a5"]),
            _announcement("m0", "new-b", ["b1", "b2", "b3", "b4", "b5"]),
        ],
    }
    path = tmp_path / "roster-evidence.json"
    payload = json.dumps(evidence).encode()
    path.write_bytes(payload)

    target = build_asof_targets(
        opportunities, metadata, [], roster_evidence=path
    ).iloc[0]

    assert target["roster_status"] == "announced"
    assert target["roster_evidence_input_hashes"] == {
        str(path.resolve()): hashlib.sha256(payload).hexdigest()
    }


def test_announced_rosters_are_not_admitted_at_or_after_start() -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")])
    opportunities.loc[0, "start_at"] = "2026-04-04T00:00:00Z"
    evidence = {
        "identity_links": [],
        "announcements": [
            _announcement("m0", "A", ["a1", "a2", "a3", "a4", "a5"]),
            _announcement("m0", "B", ["b1", "b2", "b3", "b4", "b5"]),
        ],
        "input_hashes": {},
    }

    target = build_asof_targets(
        opportunities, metadata, [], roster_evidence=evidence
    ).iloc[0]

    assert target["roster_status"] == "unresolved"
    assert target["roster_announcement_a"] is None
    assert target["roster_announcement_b"] is None


def test_duplicate_player_ids_are_rejected_instead_of_deduplicated() -> None:
    state = _ReplayState()
    target = {
        "team1_id": "A",
        "team2_id": "B",
        "cutoff_day": "2026-04-04",
        "best_of": 3,
    }

    with pytest.raises(ValueError):
        _feature_values(
            state,
            target,
            (["a1", "a1", "a3", "a4", "a5"], ["b1", "b2", "b3", "b4", "b5"]),
        )


def test_explicit_temporal_identity_links_extend_only_strictly_prior_history() -> None:
    opportunities, metadata = _opportunities_and_metadata([("current-id", "other")])
    prior = ["p1", "p2", "p3", "p4", "p5"]
    events = [
        _roster_event("former-id", "prior-match", prior, "2026-04-01"),
        _roster_event("former-id", "same-day-match", ["s1", "s2", "s3", "s4", "s5"], "2026-04-04"),
        _roster_event("other", "other-match", ["o1", "o2", "o3", "o4", "o5"], "2026-04-01"),
    ]
    link = {
        "source_team_id": "former-id",
        "target_team_id": "current-id",
        "valid_from": "2026-03-01T00:00:00Z",
        "valid_to": "2026-05-01T00:00:00Z",
        "published_at": "2026-04-02T10:00:00Z",
        "observed_at": "2026-04-02T10:05:00Z",
        "evidence_uri": "local://identity/current-id",
    }
    targets = build_asof_targets(
        opportunities,
        metadata,
        events,
        roster_evidence={"identity_links": [link], "announcements": [], "input_hashes": {}},
    )
    without_link = build_asof_targets(opportunities, metadata, events)

    assert targets.loc[0, "roster_status"] == "prior_observed_scenario"
    assert targets.loc[0, "roster_a"] == prior
    assert targets.loc[0, "roster_source_a"]["match_id"] == "prior-match"
    assert targets.loc[0, "roster_source_a"]["match_id"] != "same-day-match"
    assert without_link.loc[0, "roster_a"] == [None] * 5
    assert without_link.loc[0, "roster_status"] == "unresolved"


def test_future_announcement_does_not_change_earlier_prior_roster_scenario() -> None:
    opportunities, metadata = _opportunities_and_metadata([("A", "B")])
    events = [
        _roster_event("A", "hist-a", ["a1", "a2", "a3", "a4", "a5"]),
        _roster_event("B", "hist-b", ["b1", "b2", "b3", "b4", "b5"]),
    ]
    future_evidence = {
        "identity_links": [],
        "announcements": [
            _announcement("m0", "A", ["x1", "x2", "x3", "x4", "x5"], "2026-04-05T10:00:00Z")
        ],
        "input_hashes": {},
    }
    before = build_asof_targets(opportunities, metadata, events)
    after = build_asof_targets(opportunities, metadata, events, roster_evidence=future_evidence)

    for field in ("roster_a", "roster_b", "roster_status", "roster_source_a", "roster_source_b"):
        assert before.loc[0, field] == after.loc[0, field]
    assert after.loc[0, "roster_status"] == "prior_observed_scenario"


def test_replay_uses_declared_context_when_target_not_in_archived_results(monkeypatch, tmp_path):
    from src.analysis import ev_asof_replay as replay

    opportunities = pd.DataFrame([{
        "row_id": "scheduled-target", "golgg_match_id": "future",
        "decision_at": "2026-04-02T08:00:00Z", "start_at": "2026-04-02T09:00:00Z",
    }])
    metadata = pd.DataFrame([{
        "golgg_match_id": "future", "team1_id": "A", "team2_id": "B",
        "date": "2026-04-02", "best_of": 3, "competition_context": "League",
    }])
    events = [
        _roster_event("A", "past", ["1", "2", "3", "4", "5"]),
        _roster_event("B", "past", ["6", "7", "8", "9", "10"]),
    ]
    backend = {"TemporalState": _ReplayState, "champion_vocabulary": {"": 0}}
    monkeypatch.setattr(replay, "_activate_feature_backend", lambda root: (backend, {}))
    monkeypatch.setattr(replay, "_stream_history", lambda *args: (
        [_HistoryItem("past", "2026-04-01")], {"past": {}}, {"past": []}, events, {}, {}, {},
    ))

    result = replay._build_replay_arrays(opportunities, metadata, tmp_path, tmp_path / "unused.json", {})

    assert result.targets.iloc[0].feature_available
    assert result.targets.iloc[0].roster_a == ["1", "2", "3", "4", "5"]
    assert result.targets.iloc[0].roster_b == ["6", "7", "8", "9", "10"]
    assert result.targets.iloc[0].roster_context_provenance == "target_metadata_exact"
