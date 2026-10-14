import numpy as np
import pandas as pd
import pytest

from src.analysis.reliability_tradeoff import (
    apply_confidence_scale,
    calibration_contrast,
    confidence_match_scale,
    select_past_recipes,
)


def test_common_cells_keep_anchor_favorite_when_candidate_reverses_sides():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
        "y": [0, 1, 1, 0],
        "p_a0_full": [.8, .8, .2, .2],
        "control": [.75, .75, .25, .25],
        "candidate": [.4, .6, .6, .4],
    })
    result = calibration_contrast(frame, "candidate", "control", resamples=256)
    assert result["n"] == 4
    assert result["candidate"]["squared_gap"] == pytest.approx(0)
    assert result["reference"]["squared_gap"] == pytest.approx(.25 ** 2)
    assert result["delta_squared"]["point"] == pytest.approx(-.25 ** 2)
    mirrored = frame.copy()
    for column in ("y", "p_a0_full", "control", "candidate"):
        mirrored[column] = 1 - mirrored[column]
    mirror_result = calibration_contrast(mirrored, "candidate", "control", resamples=256)
    for contrast in ("delta_squared", "delta_rms"):
        for key in ("point", "ci_low", "ci_high"):
            assert mirror_result[contrast][key] == pytest.approx(result[contrast][key])


def test_quarter_cells_reveal_cancelling_temporal_gaps():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01"],
        "y": [1, 1, 0, 0],
        "p_a0_full": [.75] * 4,
        "control": [.75] * 4,
        "candidate": [.5] * 4,
    })
    quarterly = calibration_contrast(frame, "candidate", "control", resamples=256)
    pooled = calibration_contrast(frame, "candidate", "control", resamples=256, by_quarter=False)
    assert quarterly["candidate"]["squared_gap"] == pytest.approx(.25)
    assert quarterly["reference"]["squared_gap"] == pytest.approx(.3125)
    assert pooled["candidate"]["squared_gap"] == pytest.approx(0)


def test_bootstrap_zero_mass_cells_do_not_become_missing_aggregate_draws():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-04-01"],
        "y": [0, 1],
        "p_a0_full": [.75, .75],
        "control": [.8, .8],
        "candidate": [.6, .6],
    })
    result = calibration_contrast(frame, "candidate", "control", resamples=256, seed=17)
    # Enumerating the two-cluster resamples gives Jan-only, both, Apr-only.
    assert result["delta_squared"]["point"] == pytest.approx(-.08)
    assert result["delta_squared"]["ci_low"] == pytest.approx(-.28)
    assert result["delta_squared"]["ci_high"] == pytest.approx(.12)
    assert result["delta_rms"]["ci_low"] == pytest.approx(-.2)
    assert result["delta_rms"]["ci_high"] == pytest.approx(.2)


def test_neutral_anchor_rows_are_not_zero_calibration_error_evidence():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"], "y": [0, 1],
        "p_a0_full": [.5, .5], "control": [.8, .2], "candidate": [.6, .4],
    })
    result = calibration_contrast(frame, "candidate", "control", resamples=256)
    assert result["n"] == 0
    assert result["neutral_excluded"] == 2
    assert result["candidate"]["squared_gap"] is None
    assert result["delta_squared"]["point"] is None
    assert result["delta_squared"]["ci_high"] is None


def test_confidence_control_matches_rms_logits_without_orientation_bias():
    reference_z = np.array([1., 2., -3., .5])
    reference = 1 / (1 + np.exp(-reference_z))
    candidate = 1 / (1 + np.exp(-.5 * reference_z))
    scale = confidence_match_scale(candidate, reference)
    assert scale == pytest.approx(.5)
    swap = np.array([True, False, True, False])
    assert confidence_match_scale(np.where(swap, 1-candidate, candidate),
                                  np.where(swap, 1-reference, reference)) == pytest.approx(scale)
    probabilities = np.array([.2, .5, .8])
    transformed = apply_confidence_scale(probabilities, scale)
    np.testing.assert_allclose(transformed, [1/3, .5, 2/3], rtol=0, atol=1e-14)
    np.testing.assert_allclose(apply_confidence_scale(1-probabilities, scale),
                               1-transformed, rtol=0, atol=1e-14)


def test_confidence_control_rejects_unidentified_or_invalid_cal_inputs():
    with pytest.raises(ValueError):
        confidence_match_scale(np.array([.4, .6]), np.array([.5, .5]))
    with pytest.raises(ValueError):
        confidence_match_scale(np.array([np.nan, .6]), np.array([.2, .8]))
    with pytest.raises(ValueError):
        confidence_match_scale(np.array([.4]), np.array([.2, .8]))


def test_selector_uses_released_past_and_not_current_or_future_outcomes():
    frame = pd.DataFrame({
        "golgg_match_id": ["a", "b", "late", "current", "future"],
        "date": ["2024-02-01", "2024-03-01", "2024-12-30", "2025-03-01", "2026-02-01"],
        "label_release_day": ["2024-02-02", "2024-03-02", "2025-01-01", "2025-03-02", "2026-02-02"],
        "y": [1, 0, 0, 1, 0],
        "control": [.55, .45, .1, .6, .4],
        "candidate": [.8, .2, .999, .7, .3],
    })
    result = select_past_recipes(frame, "candidate", "control", rule="ll", config={})
    records = {row["year"]: row for row in result["records"]}
    assert records[2024]["selected_column"] == "control"
    assert records[2025]["selected_column"] == "candidate"
    assert records[2025]["evidence_ids"] == ["a", "b"]
    assert records[2026]["selected_column"] == "control"
    changed = frame.copy()
    changed.loc[2:, "y"] = [1, 0, 1]
    changed.loc[3:, "label_release_day"] = ["2027-01-01", "2028-01-01"]
    perturbed = select_past_recipes(changed, "candidate", "control", rule="ll", config={})
    selected_2025 = next(row for row in perturbed["records"] if row["year"] == 2025)
    assert selected_2025["selected_column"] == records[2025]["selected_column"]
    assert selected_2025["evidence_ids"] == records[2025]["evidence_ids"]


def test_neutral_only_months_stay_in_bootstrap_population():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-04-01"], "y": [0, 1],
        "p_a0_full": [.8, .5], "control": [.8, .2], "candidate": [.6, .4],
    })
    result = calibration_contrast(frame, "candidate", "control", resamples=256, seed=17)
    assert result["n"] == 1
    assert result["blocks"] == 2
    assert result["delta_squared"]["point"] == pytest.approx(-.28)
    assert result["support"]["zero_total_support_draws"] > 0
    assert result["delta_squared"]["ci_low"] is None
    assert result["delta_squared"]["ci_high"] is None


def test_single_month_does_not_give_zero_width_reliability_certainty():
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"], "y": [0, 1],
        "p_a0_full": [.8, .8], "control": [.8, .8], "candidate": [.6, .6],
    })
    result = calibration_contrast(frame, "candidate", "control", resamples=256)
    assert result["blocks"] == 1
    assert result["delta_squared"]["point"] == pytest.approx(-.08)
    assert result["delta_squared"]["ci_high"] is None


def test_fractional_outcomes_are_not_silently_truncated():
    frame = pd.DataFrame({
        "golgg_match_id": ["a", "b"],
        "date": ["2024-01-01", "2024-02-01"],
        "label_release_day": ["2024-01-02", "2024-02-02"],
        "y": [.7, 1], "p_a0_full": [.8, .8],
        "control": [.8, .8], "candidate": [.6, .6],
    })
    with pytest.raises(ValueError):
        calibration_contrast(frame, "candidate", "control", resamples=256)
    with pytest.raises(ValueError):
        select_past_recipes(frame, "candidate", "control", rule="ll", config={})


def test_temporal_guard_is_yearly_quarter_cell_rms_not_worst_quarter():
    rows = []
    for month in range(1, 13):
        day = pd.Timestamp(year=2024, month=month, day=1)
        for position in range(100):
            rows.append({
                "golgg_match_id": f"{month}-{position}",
                "date": day.strftime("%Y-%m-%d"),
                "label_release_day": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "y": int(position % 50 < 35), "best_of": 3 if position < 50 else 5,
                "tournament_name": "LCK Spring 2024", "unknown_players": 1,
                "p_a0_full": .75, "control": .86,
                "candidate": .94 if month <= 3 else .7,
            })
    rows.append({**rows[0], "golgg_match_id": "next-origin", "date": "2025-02-01",
                 "label_release_day": "2025-02-02"})
    frame = pd.DataFrame(rows)
    config = {
        "resamples": 64, "seed": 17,
        "selector_optional_mode_pairs": {
            "no_w20": ("candidate", "control"),
            "no_organization": ("candidate", "control"),
        },
    }
    result = select_past_recipes(frame, "candidate", "control",
                                rule="reliability", config=config, years=(2024, 2025))
    record = next(row for row in result["records"] if row["year"] == 2025)
    # Q1 worsens by 8pp, but annual quarter-cell RMS improves from .16 to .12.
    assert record["gates"]["temporal_stability"]["pass"] is True
    # The separate absolute calibration gate still prevents selection here.
    assert record["selected_column"] == "control"


@pytest.mark.parametrize("changed_file", ["checkpoint", "receipt"])
def test_redirected_control_rejects_coherent_tampering_against_frozen_completion(
    tmp_path, monkeypatch, changed_file
):
    import hashlib
    import json
    import joblib
    from src.models import a0_reliability_tradeoff as study

    monkeypatch.setattr(study, "PROJECT_ROOT", tmp_path)
    model_root = tmp_path / "data/06_models/a0_data_focus/frozen"
    artifacts = {}
    components = ["shared", "control/calibration",
                  *(f"control/seed_{seed}" for seed in study.SEEDS)]
    for component in components:
        checkpoint = model_root / component / "model.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"value": 1}, checkpoint)
        sidecar = checkpoint.with_suffix(".joblib.receipt.json")
        sidecar.write_text(json.dumps({
            "contract": {}, "contract_sha256": study._contract_digest({}),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        }))
        for path in (checkpoint, sidecar):
            artifacts[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    completion = tmp_path / "data/08_reporting/a0_data_focus/frozen/completed.json"
    completion.parent.mkdir(parents=True)
    completion.write_text(json.dumps({"status": "TRAINING_COMPLETE_NOT_SCORED", "artifacts": artifacts}))
    execution = {"reference": {"run_id": "frozen",
                              "completed_sha256": hashlib.sha256(completion.read_bytes()).hexdigest()}}
    verified = study._verify_redirected_control(execution)
    assert verified["artifacts"] == artifacts
    checkpoint = model_root / "shared/model.joblib"
    sidecar = checkpoint.with_suffix(".joblib.receipt.json")
    receipt = json.loads(sidecar.read_text())
    if changed_file == "checkpoint":
        joblib.dump({"value": 2}, checkpoint)
        receipt["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    else:
        receipt["details"] = {"changed_after_completion": True}
    sidecar.write_text(json.dumps(receipt))
    # The legacy per-file loader accepts this internally consistent pair.
    assert study.native._load_checkpoint(checkpoint, {}) == {
        "value": 2 if changed_file == "checkpoint" else 1
    }
    with pytest.raises(ValueError, match="redirected control artifact"):
        study._verify_redirected_control(execution)
