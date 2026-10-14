"""Unavailable identities and legitimate empty histories are different contracts."""
import numpy as np
import pytest

from src.analysis.a0_prediction_input import snapshot_evidence_status
from src.analysis.ev_asof_replay import AsOfSnapshot


def test_unavailable_roster_cannot_receive_predictable_evidence_status():
    snapshot = AsOfSnapshot({"competition_context": "LCK 2026"}, None, None, False, "ambiguous_roster")
    status = snapshot_evidence_status(snapshot)
    assert status["can_predict"] is False
    assert status["input_status"] == "UNAVAILABLE"
    assert "ambiguous_roster" in status["reasons"]


def test_known_new_players_keep_forecast_with_explicit_limited_history():
    snapshot = AsOfSnapshot(
        {"competition_context": "LCK 2026", "roster_a": [str(i) for i in range(5)],
         "roster_b": [str(i) for i in range(5, 10)]},
        {"gates": np.zeros(2, dtype=np.float32)},
        {"mask": np.zeros((2, 5, 16), dtype=np.float32),
         "history": np.zeros((2, 5, 16, 33), dtype=np.float32)},
        True, None,
    )
    status = snapshot_evidence_status(snapshot)
    assert status["can_predict"] is True
    assert status["input_status"] == "LIMITED_EVIDENCE"
    assert status["history"]["players_without_history"] == 10
    assert status["predictive_reliability"] == "NOT_ASSESSED"


def test_roster_support_distinguishes_announced_from_prior_scenario():
    snapshot = AsOfSnapshot(
        {"roster_status": "prior_observed_scenario", "competition_context": "LCK",
         "roster_source_a": {"source_type": "announced"},
         "roster_source_b": {"source_type": "prior_observed_scenario"},
         "cutoff_day": "2026-04-01", "feature_source_max_day": "2026-03-29"},
        {"gates": np.ones((2, 2), dtype=np.float32)},
        {"mask": np.ones((2, 5, 16), dtype=np.uint8)},
        True, None,
    )
    status = snapshot_evidence_status(snapshot)
    assert status["can_predict"] is True
    assert status["roster"]["announced_sides"] == 1
    assert status["roster"]["prior_observed_sides"] == 1
    assert "roster_is_prior_observed_scenario_not_match_announcement" in status["reasons"]
    assert status["predictive_reliability"] == "NOT_ASSESSED"


def test_valid_history_tokens_do_not_certify_observed_statistics():
    snapshot = AsOfSnapshot(
        {"roster_status": "announced", "competition_context": "LCK",
         "roster_source_a": {"source_type": "announced", "known_slots": 5},
         "roster_source_b": {"source_type": "announced", "known_slots": 5},
         "cutoff_day": "2026-04-01", "feature_source_max_day": "2026-03-31"},
        {"gates": np.ones((2, 2), dtype=np.float32)},
        {"mask": np.ones((2, 5, 16), dtype=np.uint8),
         "history": np.zeros((2, 5, 16, 33), dtype=np.float32)}, True, None,
    )
    status = snapshot_evidence_status(snapshot)
    assert status["input_status"] == "LIMITED_EVIDENCE"
    assert status["history"]["players_with_history"] == 10
    assert status["history"]["history_stat_coverage"] == 0.
    snapshot.hist["history"][..., 12:24] = 1.
    supported = snapshot_evidence_status(snapshot)
    assert supported["input_status"] == "SUPPORTED_INPUT"
    assert supported["history"]["history_stat_coverage"] == 1.
    assert supported["predictive_reliability"] == "NOT_ASSESSED"


def test_same_day_history_is_not_eligible_under_daily_release_contract():
    snapshot = AsOfSnapshot(
        {"cutoff_day": "2026-04-01", "feature_source_max_day": "2026-04-01"},
        {"gates": np.ones((2, 2), dtype=np.float32)},
        {"mask": np.ones((2, 5, 16), dtype=np.uint8),
         "history": np.zeros((2, 5, 16, 33), dtype=np.float32)}, True, None,
    )
    assert snapshot_evidence_status(snapshot)["can_predict"] is False


def test_rating_only_history_without_release_provenance_is_unavailable():
    snapshot = AsOfSnapshot(
        {"cutoff_day": "2026-04-01", "feature_source_max_day": None},
        {"gates": np.ones((2, 2), dtype=np.float32)},
        {"mask": np.zeros((2, 5, 16), dtype=np.uint8),
         "history": np.zeros((2, 5, 16, 33), dtype=np.float32)}, True, None,
    )
    status = snapshot_evidence_status(snapshot)
    assert status["can_predict"] is False
    assert status["input_status"] == "UNAVAILABLE"


def test_prediction_inputs_require_unique_exact_identity_and_actual_cutoff():
    import pandas as pd
    import pytest
    import src.analysis.a0_prediction_input as inputs

    fixtures = pd.DataFrame([{
        "row_id": "r1", "golgg_match_id": "fixture-1", "team1_id": "A",
        "team2_id": "B", "best_of": 3, "decision_at": "2026-01-01T10:00:00Z",
        "start_at": "2026-01-01T11:00:00Z", "odds_a": 1.7,
    }])
    metadata = pd.DataFrame([{
        "golgg_match_id": "fixture-1", "team1_id": "A", "team2_id": "B",
        "best_of": 3, "tournament_name": "League",
    }])
    future = fixtures.copy()
    future.loc[0, "decision_at"] = "2026-01-01T11:00:00Z"
    with pytest.raises(ValueError, match="strictly before"):
        inputs.predict_fixture_inputs(future, metadata, [{"match_id": "old"}], {"old": "2025-12-31"})
    duplicate = pd.concat([fixtures, fixtures], ignore_index=True)
    duplicate.loc[1, "row_id"] = "r2"
    with pytest.raises(ValueError, match="match IDs"):
        inputs.predict_fixture_inputs(duplicate, metadata, [{"match_id": "old"}], {"old": "2025-12-31"})


def test_prediction_input_rejects_future_and_ambiguous_fixture_metadata():
    import pandas as pd
    import pytest
    import src.analysis.a0_prediction_input as inputs

    fixtures = pd.DataFrame([{
        "row_id": "r1", "golgg_match_id": "fixture-1", "team1_id": "A",
        "team2_id": "B", "best_of": 1, "cutoff_at": "2026-01-01T10:00:00Z",
        "start_at": "2026-01-01T12:00:00Z",
    }])
    metadata = pd.DataFrame([{
        "golgg_match_id": "fixture-1", "team1_id": "X", "team2_id": "B",
        "best_of": 1, "tournament_name": "League",
    }])
    with pytest.raises(ValueError, match="conflict"):
        inputs.predict_fixture_inputs(fixtures, metadata, [{"match_id": "old"}], {"old": "2025-12-31"})


@pytest.mark.parametrize("cutoff", [
    "2026-03-31T10:00:00Z",
    "2026-04-01T11:00:00Z",
])
def test_explicit_cutoff_cannot_be_silently_replaced_by_decision(cutoff):
    import pandas as pd
    from src.analysis.a0_prediction_input import predict_fixture_inputs

    fixtures = pd.DataFrame([{
        "row_id": "r1", "golgg_match_id": "fixture-1",
        "team1_id": "A", "team2_id": "B", "best_of": 3,
        "decision_at": "2026-04-01T10:00:00Z", "cutoff_at": cutoff,
        "start_at": "2026-04-01T12:00:00Z",
    }])
    metadata = pd.DataFrame([{
        "golgg_match_id": "fixture-1", "team1_id": "A", "team2_id": "B",
        "best_of": 3, "tournament_name": "League",
    }])
    with pytest.raises(ValueError, match="decision_at.*cutoff_at"):
        predict_fixture_inputs(fixtures, metadata, [{"match_id": "old"}], {})


def test_fixture_context_cannot_be_replaced_by_conflicting_metadata():
    import pandas as pd
    from src.analysis.a0_prediction_input import predict_fixture_inputs

    fixtures = pd.DataFrame([{
        "row_id": "r1", "golgg_match_id": "fixture-1",
        "team1_id": "A", "team2_id": "B", "best_of": 3,
        "decision_at": "2026-04-01T10:00:00Z",
        "start_at": "2026-04-01T12:00:00Z",
        "competition_context": "League",
    }])
    metadata = pd.DataFrame([{
        "golgg_match_id": "fixture-1", "team1_id": "A", "team2_id": "B",
        "best_of": 3, "tournament_name": "Cup",
    }])
    with pytest.raises(ValueError, match="competition context"):
        predict_fixture_inputs(fixtures, metadata, [{"match_id": "old"}], {})


def test_fixture_context_recovers_rosters_when_existing_metadata_lacks_context(monkeypatch):
    import pandas as pd
    import src.analysis.a0_prediction_input as inputs
    from src.analysis.ev_asof_replay import build_asof_targets, build_roster_events

    fixtures = pd.DataFrame([{
        "row_id": "r1", "golgg_match_id": "fixture-1",
        "team1_id": "A", "team2_id": "B", "best_of": 3,
        "decision_at": "2026-04-01T10:00:00Z",
        "cutoff_at": "2026-04-01T12:00:00+02:00",
        "start_at": "2026-04-01T12:00:00Z",
        "competition_context": "League",
    }])
    metadata = pd.DataFrame([{
        "golgg_match_id": "fixture-1", "team1_id": "A",
        "team2_id": "B", "best_of": 3,
    }])
    source = [{
        "match_id": "old", "tournament_name": "League",
        "games": [{
            "date": "2026-03-30", "t1_id": "A", "t2_id": "B",
            "t1_players": {str(i): {"player_id": f"a{i}"} for i in range(5)},
            "t2_players": {str(i): {"player_id": f"b{i}"} for i in range(5)},
        }],
    }]

    def roster_only_replay(opportunities, metadata_rows, raw, releases, **kwargs):
        targets = build_asof_targets(
            opportunities, metadata_rows, build_roster_events(raw, releases), **kwargs
        )
        for target in targets.to_dict("records"):
            yield AsOfSnapshot(target, None, None, False, "native_features_not_built")

    # Exercise actual event admission and roster selection, without loading
    # the external native feature backend into the shared test process.
    monkeypatch.setattr(inputs, "reconstruct_asof_snapshots", roster_only_replay)
    snapshot = inputs.predict_fixture_inputs(fixtures, metadata, source, {})[0]
    assert snapshot.target["roster_a"] == [f"a{i}" for i in range(5)]
    assert snapshot.target["roster_b"] == [f"b{i}" for i in range(5)]
    assert snapshot.target["roster_status"] == "prior_observed_scenario"
    assert snapshot.target["cutoff_day"] == "2026-04-01"
    assert "competition_context" not in metadata


def test_captured_history_requires_exact_bytes_and_uses_receipt_availability_not_backdated_release():
    import hashlib
    import json
    import pytest

    from src.analysis.a0_prediction_input import merge_captured_history

    warm = {"match_id": "warm", "games": [{"date": "2026-04-01"}]}
    captured = {
        "match_id": "new-match",
        "tournament_name": "League 2026",
        "games": [{
            "date": "2026-04-01",
            "t1_id": "A", "t2_id": "B",
            "t1_players": {f"r{i}": {"player_id": f"a{i}"} for i in range(5)},
            "t2_players": {f"r{i}": {"player_id": f"b{i}"} for i in range(5)},
        }],
    }
    body = json.dumps(captured, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    capture = {
        "capture_id": "capture-1",
        "source_id": "golgg-history",
        "sha256": digest,
        "body_bytes": body,
        "observed_at": "2026-04-05T10:15:00+00:00",
        "response_received_at": "2026-04-05T10:15:00+00:00",
        "first_observed_at": "2026-04-05T10:15:00+00:00",
        "source_authentication_status": "unverified",
    }
    history, release_days, lineage = merge_captured_history([warm], {}, [capture])
    assert [match["match_id"] for match in history] == ["warm", "new-match"]
    assert release_days == {"new-match": "2026-04-05"}

    with pytest.raises(ValueError, match="release day may not precede capture availability"):
        merge_captured_history([warm], {"new-match": "2026-04-01"}, [capture])
    tampered = dict(capture, body_bytes=body + b" ")
    with pytest.raises(ValueError, match="capture SHA-256"):
        merge_captured_history([warm], {}, [tampered])
    warm_bytes = json.dumps({**warm, "games": [{"date": "2026-04-02"}]}).encode()
    colliding = dict(capture, body_bytes=warm_bytes, sha256=hashlib.sha256(warm_bytes).hexdigest())
    with pytest.raises(ValueError, match="changes frozen warm-start match"):
        merge_captured_history([warm], {}, [colliding])


def test_captured_history_uses_later_release_and_earliest_identical_receipt():
    import hashlib
    import json
    from src.analysis.a0_prediction_input import merge_captured_history

    record = {"match_id": "new", "games": [{"date": "2026-04-01"}]}
    body = json.dumps(record).encode()
    early = {
        "capture_id": "early", "source_id": "history",
        "body_bytes": body, "sha256": hashlib.sha256(body).hexdigest(),
        "observed_at": "2026-04-05T10:00:00Z", "response_received_at": "2026-04-05T10:00:00Z",
        "first_observed_at": "2026-04-05T10:00:00Z", "published_at": "2026-04-06T09:00:00Z",
        "source_authentication_status": "local_fixture",
    }
    late = {**early, "capture_id": "late", "observed_at": "2026-04-09T10:00:00Z",
            "response_received_at": "2026-04-09T10:00:00Z"}
    forward = merge_captured_history([], {}, [early, late])
    reverse = merge_captured_history([], {}, [late, early])
    assert forward == reverse
    assert forward[1] == {"new": "2026-04-06"}


def test_captured_roster_waits_for_release_and_respects_later_withdrawal():
    import hashlib
    import json
    from src.analysis.a0_prediction_input import _captured_rosters
    from src.analysis.roster_provenance import resolve_roster_evidence

    announcement = {"match_id": "m", "team_id": "a", "published_at": "2026-04-04T09:00:00Z",
                    "roster": ["a1", "a2", "a3", "a4", "a5"], "status": "announced"}
    body = json.dumps({"announcements": [announcement]}).encode()
    capture = {
        "capture_id": "roster", "source_id": "official_roster", "evidence_uri": "fixture://official/roster",
        "body_bytes": body, "sha256": hashlib.sha256(body).hexdigest(),
        "observed_at": "2026-04-05T10:00:00Z", "response_received_at": "2026-04-05T10:00:00Z",
        "first_observed_at": "2026-04-05T10:00:00Z", "published_at": "2026-04-06T09:00:00Z",
        "source_authentication_status": "local_fixture",
    }
    evidence, _ = _captured_rosters([capture])
    def resolve(records, cutoff):
        return resolve_roster_evidence(
            team_id="a", match_id="m", cutoff_at=cutoff, start_at="2026-04-10T12:00:00Z",
            announcements=records["announcements"], identity_links=records["identity_links"],
        )
    assert resolve(evidence, "2026-04-05T12:00:00Z")["announcement"] is None
    assert resolve(evidence, "2026-04-06T12:00:00Z")["announcement"]["roster"] == announcement["roster"]
    withdrawn = json.dumps({"announcements": [{**announcement, "status": "retracted",
                                               "published_at": "2026-04-07T09:00:00Z"}]}).encode()
    withdrawal = {
        **capture, "capture_id": "withdrawal", "body_bytes": withdrawn,
        "sha256": hashlib.sha256(withdrawn).hexdigest(), "published_at": "2026-04-07T09:00:00Z",
        "observed_at": "2026-04-07T10:00:00Z", "response_received_at": "2026-04-07T10:00:00Z",
        "first_observed_at": "2026-04-07T10:00:00Z",
    }
    revised, _ = _captured_rosters([capture, withdrawal])
    assert resolve(revised, "2026-04-08T12:00:00Z")["missing_reason"] == "latest_announcement_retracted"
