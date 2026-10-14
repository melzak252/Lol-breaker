from __future__ import annotations

import hashlib
import json
import zipfile

import pandas as pd
import numpy as np
import pytest

from src.models import a0_context_reliability_runner as runner


def _toy_identity_frames():
    joined = pd.DataFrame({
        "golgg_match_id": ["m1", "m2"],
        "date": ["2024-01-02", "2024-04-03"],
        "result_day": ["2024-01-03", "2024-04-04"],
        "team1_id": ["a", "c"],
        "team2_id": ["b", "d"],
        "best_of": [3, 5],
        "y": [1, 0],
        "head_label_release_day": ["2024-01-03", "2024-04-04"],
        "label_release_day": ["2024-01-03", "2024-04-04"],
    })
    source = joined.copy()
    return joined, source


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("golgg_match_id", ["m1", "different"]),
        ("team1_id", ["a", "d"]),
        ("team2_id", ["b", "c"]),
        ("date", ["2024-01-02", "2024-04-05"]),
        ("result_day", ["2024-01-03", "2024-04-05"]),
        ("best_of", [3, 3]),
        ("y", [1, 1]),
        ("head_label_release_day", ["2024-01-03", "2024-04-05"]),
        ("label_release_day", ["2024-01-03", "2024-04-05"]),
    ],
)
def test_identity_validation_rejects_fixture_or_clock_conflicts(column, replacement):
    joined, source = _toy_identity_frames()
    source[column] = replacement

    with pytest.raises(ValueError):
        runner.validate_join_identity(joined, source)


def test_identity_validation_rejects_row_reordering_even_when_ids_are_present():
    joined, source = _toy_identity_frames()

    with pytest.raises(ValueError):
        runner.validate_join_identity(joined, source.iloc[::-1].reset_index(drop=True))


def test_origin_membership_uses_event_and_authentic_release_strictly_before_origin():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2023-12-31", "2023-12-30"],
        "label_release_day": ["2024-01-01", "2024-01-01", "2023-12-31"],
    })

    eligible = runner.eligible_history_rows(frame, "2024-01-01")

    assert eligible.tolist() == [False, False, True]


def test_aligned_bundle_uses_native_and_prepared_release_clocks_for_cutoff(tmp_path, monkeypatch):
    native = pd.DataFrame({
        "golgg_match_id": ["m1", "m2"], "date": ["2024-12-10", "2024-12-11"],
        "result_day": ["2024-12-12", "2024-12-13"], "team1_id": ["a", "c"],
        "team2_id": ["b", "d"], "best_of": [3, 5], "y": [1, 0],
        "label_release_day": ["2024-12-20", "2024-12-21"], "p_c0": [0.6, 0.7],
    })
    prepared = pd.DataFrame({
        "golgg_match_id": ["m2", "m1"], "date": ["2024-12-11", "2024-12-10"],
        "result_day": ["2024-12-13", "2024-12-12"], "team1_id": ["c", "a"],
        "team2_id": ["d", "b"], "best_of": [5, 3], "y": [0, 1],
        "head_label_release_day": ["2024-12-31", "2025-01-01"],
    })
    canonical = native[["golgg_match_id", "date", "result_day", "team1_id",
                        "team2_id", "best_of", "y"]].copy()
    canonical["p_a0"] = [0.61, 0.71]
    monkeypatch.setattr(runner, "_source_frame", lambda _root: (native, prepared, canonical))
    monkeypatch.setattr(runner, "_reconstruct_saved_experts", lambda frame, _root: (frame, {}))
    monkeypatch.setattr(runner, "_sha256", lambda _path: "fixed-provenance")
    aligned, _, _ = runner._aligned_bundle(tmp_path)

    assert aligned.label_release_day.tolist() == ["2024-12-20", "2024-12-21"]
    assert aligned.head_label_release_day.tolist() == ["2025-01-01", "2024-12-31"]
    assert runner.eligible_history_rows(aligned, "2025-01-01").tolist() == [False, True]


def test_origin_membership_rejects_release_before_event():
    frame = pd.DataFrame({
        "date": ["2024-01-02"],
        "label_release_day": ["2024-01-01"],
    })

    with pytest.raises(ValueError, match="release precedes its event"):
        runner.eligible_history_rows(frame, "2024-02-01")


def test_conditional_runner_forecasts_by_origin_and_scores_only_closed_past_cv(tmp_path):
    rng = np.random.default_rng(41)
    records = []
    for index, day in enumerate(pd.date_range("2023-01-01", periods=540, freq="D")):
        p = float(0.56 + (index % 34) * 0.01)
        release = day + pd.Timedelta(days=1)
        records.append({
            "golgg_match_id": f"train-{index:04d}", "date": day, "result_day": release,
            "label_release_day": release, "head_label_release_day": release,
            "team1_id": f"a-{index % 5}", "team2_id": f"b-{index % 7}", "best_of": 3,
            "y": int(rng.random() < p), "p_c0": p,
            "x_low_statistics": float(index % 2),
            "x_novel_roster_support": float((index // 2) % 2),
            "x_high_sports_disagreement": float((index // 4) % 2),
        })
    first_origin_ids = []
    first_origin_dates = [
        "2024-10-01", "2024-10-31", "2024-11-01", "2024-11-30",
        "2024-12-01", "2024-12-20", "2024-12-31",
    ]
    for index, value in enumerate(first_origin_dates):
        day = pd.Timestamp(value)
        release = day + pd.Timedelta(days=1)
        match_id = f"q4-{index}"
        first_origin_ids.append(match_id)
        records.append({
            "golgg_match_id": match_id, "date": day, "result_day": release,
            "label_release_day": release, "head_label_release_day": release,
            "team1_id": f"q4-a-{index}", "team2_id": f"q4-b-{index}", "best_of": 3,
            "y": index % 2, "p_c0": 0.62 + index * 0.03,
            "x_low_statistics": float(index % 2),
            "x_novel_roster_support": float((index // 2) % 2),
            "x_high_sports_disagreement": float((index // 4) % 2),
        })
    second_origin_ids = []
    for index, value in enumerate(("2025-01-01", "2025-01-15")):
        day = pd.Timestamp(value)
        release = day + pd.Timedelta(days=1)
        match_id = f"q1-{index}"
        second_origin_ids.append(match_id)
        records.append({
            "golgg_match_id": match_id, "date": day, "result_day": release,
            "label_release_day": release, "head_label_release_day": release,
            "team1_id": f"q1-a-{index}", "team2_id": f"q1-b-{index}", "best_of": 3,
            "y": index % 2, "p_c0": 0.68 + index * 0.04,
            "x_low_statistics": float(index % 2),
            "x_novel_roster_support": float((index // 2) % 2),
            "x_high_sports_disagreement": float((index // 4) % 2),
        })
    frame = pd.DataFrame.from_records(records)

    result = runner._fit_conditional(
        frame, ("2024-10-01", "2025-01-01"), tmp_path / "models", pd.DataFrame())

    first, second = result["predictions"]
    assert first.golgg_match_id.tolist() == first_origin_ids
    assert first.origin.eq("2024-10-01").all()
    assert second.golgg_match_id.tolist() == second_origin_ids
    assert second.origin.eq("2025-01-01").all()
    closed_cv_ids = sorted(first_origin_ids[:-1])
    later_fit = next(
        fit for fit in result["fits"]
        if fit["origin"] == "2025-01-01" and fit["kind"] == "pooled" and fit["penalty"] == 100.0)
    assert later_fit["cv_rows"] == len(closed_cv_ids)
    assert later_fit["cv_ids_sha256"] == runner._digest(closed_cv_ids)
    for column in ("p_pooled", "p_independent"):
        assert np.isfinite(second[column].to_numpy(float)).all()
        assert second[column].between(0, 1, inclusive="neither").all()


def test_freeze_refuses_to_overwrite_existing_run_root(tmp_path, monkeypatch):
    stage_root = tmp_path / "data/05_model_input/a0_context_reliability"
    existing = stage_root / "context_reliability_20261004_01"
    existing.mkdir(parents=True)
    sentinel = existing / "preserve.json"
    sentinel.write_text('{"owner":"existing"}\n')
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileExistsError):
        runner.freeze_run("context_reliability_20261004_01", research_root=tmp_path)

    assert sentinel.read_text() == '{"owner":"existing"}\n'


def test_run_protocol_rejects_changed_frozen_joined_input_before_read_or_fit(tmp_path, monkeypatch):
    input_root = tmp_path / "input/run"
    input_root.mkdir(parents=True)
    report_root = tmp_path / "report/run"
    output_root = tmp_path / "output/run"
    model_root = tmp_path / "models/run"
    for root in (report_root, output_root, model_root):
        root.mkdir(parents=True)
    protocol_path = input_root / "protocol.json"
    protocol_path.write_text('{"frozen":true}\n')
    protocol_sha = runner._sha256(protocol_path)
    original = b"joined input before modification"
    joined_path = input_root / "joined_input.parquet"
    joined_path.write_bytes(b"joined input modified after freeze")
    input_hashes = {"source": "source-hash"}
    protocol = {
        "run_id": "run", "input_sha256": input_hashes,
        "joined_input_sha256": hashlib.sha256(original).hexdigest(),
    }
    (input_root / "input_provenance.json").write_text(json.dumps({
        "protocol_sha256": protocol_sha, "input_sha256": input_hashes,
        "joined_input_sha256": protocol["joined_input_sha256"],
    }))
    paths = {"input": input_root, "report": report_root, "output": output_root, "models": model_root}
    monkeypatch.setattr(runner, "_validate_protocol", lambda _: (protocol, paths))
    monkeypatch.setattr(runner.pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("read before digest check"))
    monkeypatch.setattr(runner, "_fit_globals", lambda *_args, **_kwargs: pytest.fail("fit before digest check"))

    with pytest.raises(ValueError, match="frozen joined input checksum mismatch"):
        runner.run_protocol(protocol_path)


def test_run_protocol_rejects_live_calibration_source_change_before_fit(tmp_path, monkeypatch):

    paths = {
        "input": tmp_path / "data/05_model_input/a0_context_reliability/frozen-run",
        "models": tmp_path / "data/06_models/a0_context_reliability/frozen-run",
        "output": tmp_path / "data/07_model_output/a0_context_reliability/frozen-run",
        "report": tmp_path / "data/08_reporting/a0_context_reliability/frozen-run",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_paths", lambda _run_id: paths)
    source_hashes = {}
    source_bytes = {}
    for relative in runner.SOURCE_CLOSURE:
        if relative == runner.SOURCE_REQUIREMENTS_RELATIVE:
            content = runner.SOURCE_REQUIREMENTS_BYTES
        elif relative == runner.EXECUTION_CONTRACT_RELATIVE:
            content = runner.EXECUTION_CONTRACT_BYTES
        else:
            content = f"frozen source: {relative}\n".encode()
        source_path = tmp_path / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(content)
        key = f"project:{relative}"
        source_hashes[key] = hashlib.sha256(content).hexdigest()
        source_bytes[key] = content

    snapshot = paths["input"] / "source_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w") as archive:
        for key, content in source_bytes.items():
            archive.writestr(key.replace(":", "/", 1), content)
        archive.writestr("input_contract.json", json.dumps(runner.INPUT_CONTRACT))
        archive.writestr("source_activation_requirements.json", runner.SOURCE_REQUIREMENTS_BYTES)
        archive.writestr("execution_contract.json", runner.EXECUTION_CONTRACT_BYTES)
        archive.writestr("source_activation_summary.json", json.dumps(runner.SOURCE_BLOCKED))
    protocol = {
        "schema": "c0_context_reliability_v1", "status": "FROZEN_BEFORE_DATA_SCORING",
        "anchor_identity": runner.INPUT_CONTRACT["anchor"], "inputs": runner.INPUT_CONTRACT,
        "run_id": "frozen-run", "scenario_origin": "2024-07-01", "valid_until": "2026-09-09",
        "classification": "EXPOSED_HISTORICAL_DEVELOPMENT", "source_activation": runner.SOURCE_BLOCKED,
        "source_sha256": source_hashes, "source_snapshot_sha256": runner._sha256(snapshot),
        "runtime": runner._runtime_versions(),
        "source_activation_requirements_sha256": runner.SOURCE_REQUIREMENTS_SHA256,
        "execution_contract": {
            "artifact": runner.EXECUTION_CONTRACT_RELATIVE,
            "sha256": runner.EXECUTION_CONTRACT_SHA256,
        },
    }
    protocol_path = paths["input"] / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    calibration_path = tmp_path / "src/models/a0_context_calibration.py"
    calibration_path.write_bytes(b"modified live calibration source\n")
    monkeypatch.setattr(runner, "_fit_globals", lambda *_args, **_kwargs: pytest.fail("fit before source check"))

    with pytest.raises(ValueError, match="live executed source project:src/models/a0_context_calibration.py checksum mismatch"):
        runner.run_protocol(protocol_path)
