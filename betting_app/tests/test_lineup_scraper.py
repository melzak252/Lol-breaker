"""Confirmed lineup ingestion and native PostgreSQL publication regressions."""

from datetime import UTC, datetime, timedelta
import json

import pytest
from sqlalchemy import select

from betting_app.core.db import get_session
from betting_app.core.models import C0_NATIVE
from betting_app.models.match import CanonicalMatch
from betting_app.models.prediction import CanonicalPrediction, ModelArtifact, UpcomingMatchFeature
from betting_app.services import current_roster_service
from betting_app.services import upcoming_inference_service as service
from betting_app.tests.test_a1_persistence_contracts import (
    _seed_current_native_roster_match,
    native_evidence_replay_client,  # noqa: F401 -- dependency of imported serving fixture
    native_serving_environment,  # noqa: F401 -- shared genuine native/owned-PG fixture
)

from betting_app.scrapers.lineup_scraper import (
    ConfirmedLineupScraper,
    check_confirmed_lineup,
    detect_lineup_substitution,
    is_within_lineup_window,
    normalize_player_id,
)
from betting_app.services.upcoming_inference_service import (
    process_confirmed_lineup_update,
)


def _exact_roster(team: str, names: list[str]) -> list[dict[str, str]]:
    roles = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
    return [
        {
            "player_id": f"{team}-{name.lower()}",
            "player_name": name,
            "role": role,
        }
        for name, role in zip(names, roles, strict=True)
    ]


# ============================================================================
# 1. detect_lineup_substitution tests
# ============================================================================

def test_detect_lineup_substitution_no_change():
    """Test no-change scenario: identical starters in same or different order."""
    previous = ["Faker", "Zeus", "Oner", "Gumayusi", "Keria"]
    confirmed_same_order = ["Faker", "Zeus", "Oner", "Gumayusi", "Keria"]
    confirmed_diff_order = ["Keria", "Gumayusi", "Oner", "Zeus", "Faker"]

    has_sub, subs = detect_lineup_substitution(previous, confirmed_same_order)
    assert has_sub is False
    assert subs == []

    has_sub_order, subs_order = detect_lineup_substitution(previous, confirmed_diff_order)
    assert has_sub_order is False
    assert subs_order == []


def test_detect_lineup_substitution_single_sub():
    """Test substitution scenario: 1 sub -> substitution detected, returns substituted player ID."""
    previous = ["Faker", "Zeus", "Oner", "Gumayusi", "Keria"]
    confirmed = ["Faker", "Doran", "Oner", "Gumayusi", "Keria"]

    has_sub, subs = detect_lineup_substitution(previous, confirmed)
    assert has_sub is True
    assert subs == ["Doran"]


def test_detect_lineup_substitution_multiple_subs():
    """Test multiple substitutions: returns all incoming substitute player IDs."""
    previous = ["P1", "P2", "P3", "P4", "P5"]
    confirmed = ["P1", "SubMid", "P3", "P4", "SubSupport"]

    has_sub, subs = detect_lineup_substitution(previous, confirmed)
    assert has_sub is True
    assert subs == ["SubMid", "SubSupport"]


def test_detect_lineup_substitution_dict_rosters():
    """Test detect_lineup_substitution with dictionary player objects."""
    previous = [
        {"role": "TOP", "player_id": "101", "player_name": "Kiin"},
        {"role": "JUNGLE", "player_id": "102", "player_name": "Canyon"},
        {"role": "MID", "player_id": "103", "player_name": "Chovy"},
        {"role": "ADC", "player_id": "104", "player_name": "Peyz"},
        {"role": "SUPPORT", "player_id": "105", "player_name": "Lehends"},
    ]
    confirmed = [
        {"role": "TOP", "player_id": "101", "player_name": "Kiin"},
        {"role": "JUNGLE", "player_id": "102", "player_name": "Canyon"},
        {"role": "MID", "player_id": "103", "player_name": "Chovy"},
        {"role": "ADC", "player_id": "999", "player_name": "Ruler"},
        {"role": "SUPPORT", "player_id": "105", "player_name": "Lehends"},
    ]

    has_sub, subs = detect_lineup_substitution(previous, confirmed)
    assert has_sub is True
    assert subs == ["999"]


def test_detect_lineup_substitution_empty_roster():
    """Test empty roster handling."""
    has_sub, subs = detect_lineup_substitution([], [])
    assert has_sub is False
    assert subs == []


# ============================================================================
# 2. Timing window tests
# ============================================================================

def test_timing_window_valid_range():
    """Test timing window: checks valid within [start - 45m, start] window."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)

    # 45 minutes before start (exact boundary)
    valid_45m, reason_45m = is_within_lineup_window(start_dt, start_dt - timedelta(minutes=45))
    assert valid_45m is True
    assert reason_45m == "valid_window"

    # 30 minutes before start (typical release time)
    valid_30m, reason_30m = is_within_lineup_window(start_dt, start_dt - timedelta(minutes=30))
    assert valid_30m is True
    assert reason_30m == "valid_window"

    # 10 minutes before start
    valid_10m, reason_10m = is_within_lineup_window(start_dt, start_dt - timedelta(minutes=10))
    assert valid_10m is True
    assert reason_10m == "valid_window"

    # At match start (0 minutes)
    valid_0m, reason_0m = is_within_lineup_window(start_dt, start_dt)
    assert valid_0m is True
    assert reason_0m == "valid_window"

    # 5 minutes after start (within 15m post-start grace period)
    valid_post, reason_post = is_within_lineup_window(start_dt, start_dt + timedelta(minutes=5))
    assert valid_post is True
    assert reason_post == "valid_window"


def test_timing_window_too_early_ignored():
    """Test timing window: ignores if > 2h (or > 45m) before match."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)

    # 2 hours before match start (120 minutes) -> MUST be ignored / outside window
    valid_2h, reason_2h = is_within_lineup_window(start_dt, start_dt - timedelta(hours=2))
    assert valid_2h is False
    assert "outside_window_too_early" in reason_2h
    assert "120.0m" in reason_2h

    # 60 minutes before match start -> outside default 45m window
    valid_60m, reason_60m = is_within_lineup_window(start_dt, start_dt - timedelta(minutes=60))
    assert valid_60m is False
    assert "outside_window_too_early" in reason_60m

    # Check via check_confirmed_lineup 2 hours before match
    result = check_confirmed_lineup(
        match_id=1,
        match_start_at=start_dt.isoformat(),
        canonical_match={"team_a_name": "T1", "team_b_name": "Gen.G"},
        current_time=start_dt - timedelta(hours=2),
    )
    assert result["is_confirmed"] is False
    assert result["window_valid"] is False
    assert result["status"] == "outside_window"


def test_timing_window_past_match():
    """Test timing window: rejects if match started long ago (> 15m post start)."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)

    valid_late, reason_late = is_within_lineup_window(start_dt, start_dt + timedelta(minutes=30))
    assert valid_late is False
    assert "outside_window_past_match" in reason_late

    result = check_confirmed_lineup(
        match_id=1,
        match_start_at=start_dt.isoformat(),
        current_time=start_dt + timedelta(minutes=45),
    )
    assert result["is_confirmed"] is False
    assert result["window_valid"] is False
    assert result["status"] == "past_match"


# ============================================================================
# 3. check_confirmed_lineup & ConfirmedLineupScraper tests
# ============================================================================

def test_check_confirmed_lineup_within_window_success():
    """Test check_confirmed_lineup returns confirmed when 5-player rosters are available."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)
    now_dt = start_dt - timedelta(minutes=30)

    canonical_match = {
        "id": 42,
        "team_a_name": "T1",
        "team_b_name": "Gen.G",
        "start_time_normalized": start_dt.isoformat(),
        "confirmed_roster_a": ["Faker", "Zeus", "Oner", "Gumayusi", "Keria"],
        "confirmed_roster_b": ["Chovy", "Kiin", "Canyon", "Peyz", "Lehends"],
        "previous_roster_a": ["Faker", "Zeus", "Oner", "Gumayusi", "Keria"],
        "previous_roster_b": ["Chovy", "Kiin", "Canyon", "Peyz", "Lehends"],
    }

    result = check_confirmed_lineup(
        match_id=42,
        match_start_at=start_dt.isoformat(),
        canonical_match=canonical_match,
        current_time=now_dt,
    )

    assert result["is_confirmed"] is True
    assert result["window_valid"] is True
    assert result["status"] == "confirmed"
    assert result["substitutions_detected"] is False
    assert result["substituted_player_ids"] == []


def test_check_confirmed_lineup_with_substitution():
    """Test check_confirmed_lineup detects substitution within valid window."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)
    now_dt = start_dt - timedelta(minutes=30)

    canonical_match = {
        "id": 100,
        "team_a_name": "T1",
        "team_b_name": "Gen.G",
        "start_time_normalized": start_dt.isoformat(),
        "confirmed_roster_a": ["Faker", "Doran", "Oner", "Gumayusi", "Keria"],  # Doran in for Zeus
        "confirmed_roster_b": ["Chovy", "Kiin", "Canyon", "Peyz", "Lehends"],
        "previous_roster_a": ["Faker", "Zeus", "Oner", "Gumayusi", "Keria"],
        "previous_roster_b": ["Chovy", "Kiin", "Canyon", "Peyz", "Lehends"],
    }

    result = check_confirmed_lineup(
        match_id=100,
        canonical_match=canonical_match,
        current_time=now_dt,
    )

    assert result["is_confirmed"] is True
    assert result["window_valid"] is True
    assert result["substitutions_detected"] is True
    assert result["substituted_player_ids"] == ["Doran"]
    assert result["substitutions"]["team_a"]["substituted_players"] == ["Doran"]
    assert result["substitutions"]["team_b"]["has_sub"] is False


def test_check_confirmed_lineup_incomplete_roster():
    """Test check_confirmed_lineup fails when roster has fewer than 5 players."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)
    now_dt = start_dt - timedelta(minutes=25)

    canonical_match = {
        "id": 101,
        "team_a_name": "T1",
        "team_b_name": "Gen.G",
        "start_time_normalized": start_dt.isoformat(),
        "confirmed_roster_a": ["Faker", "Zeus"],  # Only 2 players
        "confirmed_roster_b": ["Chovy", "Kiin", "Canyon", "Peyz", "Lehends"],
    }

    result = check_confirmed_lineup(
        match_id=101,
        canonical_match=canonical_match,
        current_time=now_dt,
    )

    assert result["is_confirmed"] is False
    assert result["window_valid"] is True
    assert result["status"] == "incomplete_roster"


def test_confirmed_lineup_scraper_class():
    """Test ConfirmedLineupScraper wrapper class."""
    scraper = ConfirmedLineupScraper(window_minutes=45)
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)

    match = {
        "id": 55,
        "start_time_normalized": start_dt.isoformat(),
        "confirmed_roster_a": ["A", "B", "C", "D", "E"],
        "confirmed_roster_b": ["V", "W", "X", "Y", "Z"],
    }

    # Within window
    res_in = scraper.check_match(match, current_time=start_dt - timedelta(minutes=20))
    assert res_in["is_confirmed"] is True

    # 3 hours prior
    res_early = scraper.check_match(match, current_time=start_dt - timedelta(hours=3))
    assert res_early["is_confirmed"] is False
    assert res_early["status"] == "outside_window"


# ============================================================================
# 4. upcoming_inference_service integration tests
# ============================================================================



def _stored_lineup_match(clock):
    match, _ = _seed_current_native_roster_match(clock)
    start = (clock["now"] + timedelta(minutes=30)).isoformat()
    with get_session() as session:
        session.get(CanonicalMatch, match["id"]).start_time_normalized = start
        session.commit()
    match["start_time_normalized"] = start
    for side in ("a", "b"):
        roster = current_roster_service.load_current_roster(match[f"team_{side}_name"])
        match[f"previous_roster_{side}"] = roster["players"]
        match[f"confirmed_roster_{side}"] = roster["players"]
    return match


def test_process_confirmed_lineup_update_no_change(native_serving_environment):
    clock = native_serving_environment
    match = _stored_lineup_match(clock)

    result = process_confirmed_lineup_update(match, current_time=clock["now"])

    assert result["status"] == "confirmed_no_change"
    assert result["is_lineup_confirmed"] is True
    assert result["substitution_detected"] is False
    assert result["re_inference_triggered"] is False
    assert result["substituted_players"] == {"team_a": [], "team_b": []}
    service.validate_native_serving_inputs(result["features"])
    with get_session() as session:
        assert session.scalars(select(CanonicalPrediction)).all() == []
        saved = session.scalar(select(UpcomingMatchFeature))
        features = json.loads(saved.features_json)
        assert features["canonical"]["is_lineup_confirmed"] is True
        assert features["c0_request"]["roster_a"] == [
            player["player_id"] for player in match["confirmed_roster_a"]
        ]


def test_process_confirmed_lineup_update_substitution_forces_reinference(
    native_serving_environment, client,
):
    clock = native_serving_environment
    match = _stored_lineup_match(clock)
    original = process_confirmed_lineup_update(
        match, current_time=clock["now"], force_re_infer=True,
    )
    match["confirmed_roster_a"] = [
        {**player, "player_id": "replacement", "player_name": "Incoming top"}
        if player["role"] == "TOP" else player
        for player in match["confirmed_roster_a"]
    ]
    clock["now"] += timedelta(minutes=1)

    result = process_confirmed_lineup_update(match, current_time=clock["now"])

    assert result["status"] == "re_inferred"
    assert result["substituted_players"] == {"team_a": ["replacement"], "team_b": []}
    assert result["substituted_player_ids"] == ["replacement"]
    assert result["re_inference_triggered"] is True
    assert result["diagnostics"]["lineup_substitution_detected"] is True
    service.validate_native_serving_inputs(result["features"])
    with get_session() as session:
        roster = current_roster_service.load_current_roster("Alpha", session=session)
        top = next(player for player in roster["players"] if player["role"] == "TOP")
        assert (top["player_id"], top["player_name"]) == ("replacement", "Incoming top")
        assert roster["source"] == "confirmed_lineup"
        assert session.get(CanonicalPrediction, original["prediction_id"]).prediction_status == "stale"
        published = session.get(CanonicalPrediction, result["prediction_id"])
        artifact = session.get(ModelArtifact, published.model_artifact_id)
        assert published.prediction_status == "active"
        assert (artifact.model_name, artifact.model_version) == (C0_NATIVE.name, C0_NATIVE.version)
        assert published.prob_a == pytest.approx(result["prob_a"])
        diagnostics = json.loads(published.diagnostics_json)
        assert diagnostics["c0_request"]["roster_a"] == sorted(
            player["player_id"] for player in roster["players"]
        )
        assert diagnostics["requested_decision_at"] == clock["now"].isoformat()
        assert diagnostics["p_low_a"] is None
        assert diagnostics["p_low_b"] is None
        assert diagnostics["epistemic_sigma_z"] is None
    response = client.get(f"/matches/{match['id']}")
    assert response.status_code == 200
    rows = [
        row for row in response.json()["predictions"]
        if row["model_name"] == C0_NATIVE.name
    ]
    assert len(rows) == 1
    assert rows[0]["model_version"] == C0_NATIVE.version
    assert rows[0]["prob_a"] == pytest.approx(result["prob_a"])
    assert rows[0]["raw_diagnostics"] == diagnostics


@pytest.mark.parametrize("failure", ["unbound", "wrong_version", "missing_checkpoint"])
def test_lineup_refresh_refuses_invalid_native_publication_without_staling_forecast(
    native_serving_environment, monkeypatch, failure,
):
    clock = native_serving_environment
    match = _stored_lineup_match(clock)
    original = process_confirmed_lineup_update(
        match, current_time=clock["now"], force_re_infer=True,
    )
    arguments = {"current_time": clock["now"], "force_re_infer": True}
    if failure == "unbound":
        build = service.build_features_for_match

        def unbound_features(*args, **kwargs):
            result = build(*args, **kwargs)
            del result["features"]["c0_request"]
            return result

        monkeypatch.setattr(service, "build_features_for_match", unbound_features)
        error, message = ValueError, "bound canonical request"
    elif failure == "wrong_version":
        arguments["model_version"] = "unregistered-native-version"
        error, message = ValueError, "exact registered model version"
    else:
        def missing_checkpoint(*args, **kwargs):
            raise FileNotFoundError("Unavailable sports checkpoint")

        monkeypatch.setattr(service, "predict_probability_from_features", missing_checkpoint)
        error, message = FileNotFoundError, "Unavailable sports checkpoint"

    with pytest.raises(error, match=message):
        process_confirmed_lineup_update(match, **arguments)
    with get_session() as session:
        predictions = session.scalars(select(CanonicalPrediction)).all()
        assert [(row.id, row.prediction_status) for row in predictions] == [
            (original["prediction_id"], "active"),
        ]


def test_process_confirmed_lineup_update_too_early_ignored():
    """Test process_confirmed_lineup_update ignores match > 2h before start."""
    start_dt = datetime(2026, 9, 16, 18, 0, 0, tzinfo=UTC)
    now_dt = start_dt - timedelta(hours=2, minutes=15)  # 2h15m before match

    match = {
        "id": 203,
        "team_a_name": "T1",
        "team_b_name": "Gen.G",
        "start_time_normalized": start_dt.isoformat(),
        "confirmed_roster_a": ["Faker", "Doran", "Oner", "Gumayusi", "Keria"],
    }

    result = process_confirmed_lineup_update(
        match,
        current_time=now_dt,
    )

    assert result["window_valid"] is False
    assert result["status"] == "outside_window"
    assert result["re_inference_triggered"] is False
    assert result["substitution_detected"] is False


