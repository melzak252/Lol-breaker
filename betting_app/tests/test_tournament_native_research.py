from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from betting_app.core.models.native_c0 import NativeC0Adapter
from src.models.tournament_native import (
    ResearchNativePairProvider,
    native_model_eligibility,
)


@pytest.fixture
def research_event() -> dict:
    return {
        "id": "historical-two-team-final-2024",
        "edition_id": "historical-two-team-2024",
        "year": 2024,
        "cutoff": "2024-05-01T00:00:00Z",
        "event_title": "Historical two-team final",
        "scope": "retrospective_conditional_roster_scenario",
        "teams": ["team-a", "team-b"],
        "rosters": {
            "team-a": ["a1", "a2", "a3", "a4", "a5"],
            "team-b": ["b1", "b2", "b3", "b4", "b5"],
        },
        "roster_basis": "first_event_actual_roster",
        "roster_evidence": {
            "authenticated": False,
            "publication_at": None,
            "source_match_ids": ["historical-opening-series"],
        },
        "spec": {
            "teams": ["team-a", "team-b"],
            "stages": [
                {"id": "final", "format": "single_elimination", "best_of": 5}
            ],
        },
        "labels_path": "separate-observed-labels.json",
        "model_eligibility": {},
        "source_provenance": {"kind": "released_historical_matches"},
    }


@pytest.fixture
def eligible_artifact() -> dict:
    return {
        "model_kind": "c0",
        "model_year": 2024,
        "train_end": "2023-06-30T00:00:00Z",
        "calibration_end": "2023-12-31T00:00:00Z",
        "artifact_paths": {"model": "historical-control/model.pt"},
        "artifact_sha256": {"model": "1" * 64},
        "roster_basis": "first_event_actual_roster",
        "authenticated": False,
    }


def _request(*, decision_at: str = "2024-05-01T00:00:00Z", mode: str = "full") -> dict:
    return {
        "team1_id": "team-a",
        "team2_id": "team-b",
        "roster_a": ["a1", "a2", "a3", "a4", "a5"],
        "roster_b": ["b1", "b2", "b3", "b4", "b5"],
        "best_of": 5,
        "decision_at": decision_at,
        "competition_context": "Historical two-team final",
        "start_at": "2026-05-02T00:00:00Z",
        "mode": mode,
    }


@pytest.mark.parametrize("model_kind", ["c0", "a0"])
def test_prior_fitted_annual_artifact_is_eligible(research_event, eligible_artifact, model_kind):
    eligible_artifact["model_kind"] = model_kind

    result = native_model_eligibility(research_event, model_kind, eligible_artifact)

    assert result["available"] is True
    assert result["reason"] is None


@pytest.mark.parametrize(
    "field, cutoff, label_release, available",
    [
        ("train_end", "2024-05-01T00:00:00Z", "2023-06-30T23:59:59.999999Z", True),
        ("train_end", "2024-05-01T00:00:00Z", "2023-07-01T00:00:00Z", False),
        ("train_end", "2024-05-01T00:00:00Z", "2023-07-01T02:00:00+02:00", False),
        ("calibration_end", "2024-01-01T00:00:00Z", "2023-12-31T23:59:59.999999Z", True),
        ("calibration_end", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", False),
        ("calibration_end", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00.000001Z", False),
        ("calibration_end", "2024-01-01T00:00:00Z", "2024-01-01T02:00:00+02:00", False),
        ("calibration_end", "2024-01-01T00:00:00Z", "2023-12-31T19:59:59.999999-04:00", True),
        ("calibration_end", "2024-05-01T00:00:00Z", "2024-01-01T00:00:00Z", False),
    ],
)
def test_native_annual_labels_obey_strict_train_cal_and_utc_decision_boundaries(
    research_event, eligible_artifact, field, cutoff, label_release, available
):
    research_event["cutoff"] = cutoff
    eligible_artifact[field] = label_release

    result = native_model_eligibility(research_event, "c0", eligible_artifact)

    assert result["available"] is available
    if available:
        assert result["reason"] is None
    else:
        assert isinstance(result["reason"], str) and result["reason"].strip()


@pytest.mark.parametrize("model_year", [2023, 2025, 2026])
def test_wrong_annual_model_is_not_backdated_even_with_prior_labels(
    research_event, eligible_artifact, model_year
):
    eligible_artifact["model_year"] = model_year

    result = native_model_eligibility(research_event, "c0", eligible_artifact)

    assert result["available"] is False
    assert isinstance(result["reason"], str) and result["reason"].strip()


def test_retrospective_actual_roster_remains_eligible_without_inventing_publication(
    research_event, eligible_artifact
):
    before = deepcopy(research_event)

    result = native_model_eligibility(research_event, "c0", eligible_artifact)

    assert result["available"] is True
    assert research_event == before
    assert result.get("authenticated", False) is False
    assert result.get("publication_at") is None


@pytest.mark.parametrize(
    "decision_at, allowed",
    [
        ("2025-12-31T23:59:59.999999Z", False),
        ("2026-01-01T00:00:00Z", True),
        ("2026-01-01T01:00:00+01:00", True),
        ("2026-01-01T00:30:00+01:00", False),
    ],
)
def test_historical_research_does_not_relax_public_serving_utc_minimum(decision_at, allowed):
    adapter = NativeC0Adapter.__new__(NativeC0Adapter)
    request = _request(decision_at=decision_at)

    if allowed:
        validated = adapter._validate_request(request)
        assert validated["decision_at"] >= datetime(2026, 1, 1, tzinfo=timezone.utc)
    else:
        with pytest.raises(ValueError, match="pre-2026-01-01"):
            adapter._validate_request(request)


@pytest.mark.parametrize("field", ["train_end", "calibration_end"])
def test_artifact_with_labels_after_event_cutoff_is_unavailable(
    research_event, eligible_artifact, field
):
    eligible_artifact[field] = "2024-05-02T00:00:00Z"

    result = native_model_eligibility(research_event, "c0", eligible_artifact)

    assert result["available"] is False
    assert isinstance(result["reason"], str) and result["reason"].strip()


@pytest.mark.parametrize(
    "cutoff, model_year",
    [
        ("2024-01-01T01:00:00+01:00", 2024),
        ("2023-12-31T19:00:00-05:00", 2024),
    ],
)
def test_annual_eligibility_uses_utc_year_not_timestamp_local_year(
    research_event, eligible_artifact, cutoff, model_year
):
    research_event["cutoff"] = cutoff
    eligible_artifact["model_year"] = model_year
    result = native_model_eligibility(research_event, "c0", eligible_artifact)

    assert result["available"] is True
    assert result["reason"] is None


def test_research_request_accepts_historical_unseen_rosters_without_organization():
    request = _request(mode="no_organization")
    request["team1_id"] = "unseen-team-a"
    request["team2_id"] = "unseen-team-b"
    request["competition_context"] = None
    request["start_at"] = None

    validated = ResearchNativePairProvider.validate_request(request)

    assert validated["decision_at"] == datetime(2024, 5, 1, tzinfo=timezone.utc)
    assert set(validated["roster_a"]).isdisjoint(validated["roster_b"])
    assert validated["mode"] == "no_organization"


@pytest.mark.parametrize(
    "overrides",
    [
        {"roster_a": ["a1", "a2", "a3", "a4"]},
        {"roster_a": ["a1", "a2", "a3", "a4", "a5", "a6"]},
        {"roster_a": "a1,a2,a3,a4,a5"},
        {"roster_a": ["a1", "a1", "a3", "a4", "a5"]},
        {"roster_a": [1, "1", "a3", "a4", "a5"]},
        {"roster_b": ["a1", "b2", "b3", "b4", "b5"]},
        {"roster_b": [1, "b2", "b3", "b4", "b5"],
         "roster_a": ["1", "a2", "a3", "a4", "a5"]},
        {"team2_id": "team-a"},
        {"team1_id": 1, "team2_id": "1"},
        {"roster_a": ["a1", "a2", "", "a4", "a5"]},
        {"best_of": True},
        {"best_of": 7},
        {"decision_at": "2024-05-01T00:00:00"},
        {"start_at": "2024-05-01T00:00:00Z"},
        {"start_at": "2024-04-30T23:59:59Z"},
    ],
)
def test_research_request_rejects_invalid_series_and_ambiguous_opponent_identity(overrides):
    request = _request()
    request.update(overrides)

    with pytest.raises(ValueError):
        ResearchNativePairProvider.validate_request(request)


@pytest.mark.parametrize("change", ["missing_roster", "extra_field"])
def test_research_request_requires_complete_exact_request_shape(change):
    request = _request()
    if change == "missing_roster":
        del request["roster_a"]
    else:
        request["observed_winner"] = "team-a"

    with pytest.raises(ValueError):
        ResearchNativePairProvider.validate_request(request)


def test_missing_2023_native_artifact_has_named_unavailability_not_proxy_fallback(research_event):
    research_event["year"] = 2023
    research_event["cutoff"] = "2023-05-01T00:00:00Z"

    result = native_model_eligibility(research_event, "c0", None)

    assert result["available"] is False
    assert isinstance(result["reason"], str) and result["reason"].strip()


def test_annual_a0_uses_its_bound_recipe_not_native_c0_train_split(
    research_event, eligible_artifact
):
    eligible_artifact.update({
        "model_kind": "a0",
        "train_end": "2023-09-30T23:59:59Z",
        "train_end_exclusive": "2023-10-01T00:00:00Z",
        "cal_end_exclusive": "2024-01-01T00:00:00Z",
    })

    result = native_model_eligibility(research_event, "a0", eligible_artifact)

    assert result["available"] is True
    assert result["reason"] is None

    eligible_artifact["train_end"] = "2023-10-01T00:00:00Z"
    unavailable = native_model_eligibility(research_event, "a0", eligible_artifact)

    assert unavailable["available"] is False
    assert isinstance(unavailable["reason"], str) and unavailable["reason"].strip()


@pytest.mark.parametrize("artifact_kind", ["a0", "glicko_format", "rating_proxy"])
def test_native_c0_eligibility_never_accepts_other_model_artifacts_as_c0(
    research_event, eligible_artifact, artifact_kind
):
    eligible_artifact["model_kind"] = artifact_kind

    result = native_model_eligibility(research_event, "c0", eligible_artifact)

    assert result["available"] is False
    assert isinstance(result["reason"], str) and result["reason"].strip()


@pytest.fixture
def frozen_2023_control(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile

    import pandas as pd

    from src.models import tournament_native_training as training
    from src.models.a0_native_refit import _contract_digest
    from src.models import tournament_native as provider
    from src.analysis import ev_asof_replay as replay

    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(provider, "PROJECT_ROOT", tmp_path)
    paths = {
        name: tmp_path / f"data/{stage}/tournament_native/snapshot-test/2023"
        for name, stage in (
            ("models", "06_models"),
            ("output", "07_model_output"),
            ("report", "08_reporting"),
        )
    }
    for directory in paths.values():
        directory.mkdir(parents=True)
    with zipfile.ZipFile(paths["models"] / "sources.zip", "x") as archive:
        archive.writestr("primitive.py", "frozen source\n")
    members = pd.DataFrame(
        {
            "golgg_match_id": ["train-1", "cal-1", "validation-1"],
            "date": ["2022-06-01", "2022-08-01", "2023-02-01"],
            "result_day": ["2022-06-02", "2022-08-02", "2023-02-02"],
            "partition": ["train", "cal", "validation"],
        }
    )
    membership_path = paths["report"] / "partitions.parquet"
    members.to_parquet(membership_path, index=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {
        "schema": "tournament_native_2023_control_origin_v1",
        "run_id": "snapshot-test",
        "year": 2023,
        "recipe": {"variant": "control", "seeds": [20260910, 20260911, 20260912], "epochs": 12},
        "source_sha256": {"research:primitive.py": "a" * 64},
        "input_sha256": {"input.parquet": "b" * 64},
        "split_ids_sha256": {
            part: _contract_digest(members.loc[members.partition == part].golgg_match_id.tolist())
            for part in ("train", "cal", "validation")
        },
    }
    protocol = {
        "schema": "tournament_native_2023_control_protocol_v1",
        "status": "FROZEN_BEFORE_FITTING",
        **{key: contract[key] for key in (
            "run_id", "year", "recipe", "source_sha256", "input_sha256", "split_ids_sha256"
        )},
        "source_snapshot_sha256": sha(paths["models"] / "sources.zip"),
        "membership_path": str(membership_path),
        "membership_file_sha256": sha(membership_path),
    }
    protocol_path = paths["models"] / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    contract["protocol_sha256"] = sha(protocol_path)
    (paths["models"] / "execution.json").write_text(json.dumps({"contract": contract}))
    # Invalid checkpoint bytes make an accidental deserialization observable:
    # a correct integrity failure must occur before any native backend activation.
    checkpoint = paths["models"] / "shared/model.joblib"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not a serialized model")
    metadata_path = tmp_path / replay.BANK_RELATIVE / "metadata.parquet"
    metadata_path.parent.mkdir(parents=True)
    members.to_parquet(metadata_path, index=False)
    monkeypatch.setattr(provider, "_BANK_METADATA_SHA256", sha(metadata_path))
    return training, paths, contract, sha


@pytest.mark.parametrize("operation", ["complete", "load", "provider"])
@pytest.mark.parametrize("corruption", ["source_zip", "date", "result_day"])
def test_2023_frozen_snapshot_mutation_cannot_be_blessed_or_loaded(
    frozen_2023_control, operation, corruption
):
    import json
    import zipfile

    import pandas as pd

    training, paths, contract, sha = frozen_2023_control
    if corruption == "source_zip":
        with zipfile.ZipFile(paths["models"] / "sources.zip", "a") as archive:
            archive.writestr("postfit.py", "unfrozen source\n")
        reason = "2023 frozen source snapshot"
    else:
        membership_path = paths["report"] / "partitions.parquet"
        members = pd.read_parquet(membership_path)
        # Keep IDs and valid closed-origin boundaries unchanged.
        members.loc[0, corruption] = "2022-05-31" if corruption == "date" else "2022-06-03"
        members.to_parquet(membership_path, index=False)
        reason = "2023 frozen partition membership"
    if operation in ("load", "provider"):
        # An attacker can inventory the changed bytes, but not replace pre-fit pins.
        artifacts = {
            str(path.relative_to(training.PROJECT_ROOT)): sha(path)
            for directory in paths.values() for path in directory.rglob("*") if path.is_file()
        }
        (paths["report"] / "completed.json").write_text(json.dumps({
            "status": "TRAINING_COMPLETE_NOT_SCORED", "contract": contract, "artifacts": artifacts
        }))
        with pytest.raises(ValueError, match=reason):
            if operation == "load":
                training.load_control(paths["models"], training.PROJECT_ROOT)
            else:
                from src.models.tournament_native import _Worker
                _Worker("c0").identity(2023, training.PROJECT_ROOT, {"2023": str(paths["models"])})
    else:
        with pytest.raises(ValueError, match=reason):
            training._write_control_completion(paths, contract)
        assert not (paths["report"] / "completed.json").exists()


@pytest.mark.parametrize("operation", ["complete", "load"])
@pytest.mark.parametrize("inventory", ["source_sha256", "input_sha256", "split_ids_sha256"])
def test_2023_execution_inventory_must_equal_frozen_protocol(
    frozen_2023_control, operation, inventory
):
    import json

    training, paths, contract, _ = frozen_2023_control
    contract[inventory] = {"replacement": "c" * 64}
    (paths["models"] / "execution.json").write_text(json.dumps({"contract": contract}))
    with pytest.raises(ValueError, match=f"2023 pre-fit protocol mismatch:{inventory}"):
        if operation == "load":
            training.load_control(paths["models"], training.PROJECT_ROOT)
        else:
            training._write_control_completion(paths, contract)
    assert not (paths["report"] / "completed.json").exists()


def test_2023_completion_preserves_frozen_snapshot_pins_and_bytes(frozen_2023_control):
    import json

    training, paths, contract, sha = frozen_2023_control
    protocol_path = paths["models"] / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)

    completion = training._write_control_completion(paths, contract)

    for path, pin in (
        (paths["models"] / "sources.zip", protocol["source_snapshot_sha256"]),
        (paths["report"] / "partitions.parquet", protocol["membership_file_sha256"]),
    ):
        assert sha(path) == pin
        assert completion["artifacts"][str(path.relative_to(training.PROJECT_ROOT))] == pin
    assert protocol_path.read_bytes() == protocol_bytes
    assert json.loads((paths["report"] / "completed.json").read_text()) == completion
