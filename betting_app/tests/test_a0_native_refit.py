"""Behavioral guards for the native cadence/decay experiment's causal objective."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.a0_target_objectives import beta_objective


def test_weighted_native_losses_preserve_mask_hierarchy_and_global_age_scale():
    from src.models.a0_target_objectives import weighted_native_losses

    logits = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(-0.8, dtype=torch.float64, requires_grad=True)
    per_row = beta_objective(logits, [3, 5], [2, 0], [1, 3], eta)
    prediction = torch.zeros((2, 2, 5, 12), requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    target[0, 0, 0, 0], mask[0, 0, 0, 0] = 1.0, True
    target[0, 1, 1, 0], mask[0, 1, 1, 0] = 3.0, True
    target[1, 0, 0, :2], mask[1, 0, 0, :2] = torch.tensor([2.0, 4.0]), True
    weights = torch.tensor([0.25, 1.75], dtype=torch.float64)
    main, auxiliary = weighted_native_losses(per_row, prediction, target, mask, weights)

    expected_main = (per_row[0] * 0.25 + per_row[1] * 1.75) / 2
    torch.testing.assert_close(main, expected_main)
    # Equal observed-player weighting: row losses 1.5 and 2.5, not a pooled-cell mean.
    assert auxiliary.item() == pytest.approx((0.25 * 1.5 + 1.75 * 2.5) / 2)
    actual_gradient = torch.autograd.grad(main, (logits, eta), retain_graph=True)
    expected_gradient = torch.autograd.grad(expected_main, (logits, eta), retain_graph=True)
    for actual, expected in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual, expected)

    pieces = [
        weighted_native_losses(per_row[i:i + 1], prediction[i:i + 1], target[i:i + 1], mask[i:i + 1], weights[i:i + 1])
        for i in range(2)
    ]
    torch.testing.assert_close(main, (pieces[0][0] + pieces[1][0]) / 2)
    torch.testing.assert_close(auxiliary, (pieces[0][1] + pieces[1][1]) / 2)
    assert pieces[0][1].item() == pytest.approx(0.375)
    auxiliary.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[~mask]) == 0


def test_fit_membership_excludes_same_day_and_late_released_results():
    from src.models.a0_native_refit import origin_train_indices

    frame = pd.DataFrame({
        "date": ["2023-12-01", "2023-12-31", "2024-01-01", "2024-02-01"],
        "result_day": ["2023-12-02", "2024-01-01", "2024-01-01", "2024-02-02"],
    })
    np.testing.assert_array_equal(origin_train_indices(frame, "2024-01-01"), [0])
    changed_future = frame.copy()
    changed_future.loc[2:, "result_day"] = "2050-01-01"
    np.testing.assert_array_equal(origin_train_indices(changed_future, "2024-01-01"), [0])


def test_row_decay_uses_event_age_and_normalizes_once_over_train():
    from src.models.a0_native_refit import training_row_weights

    frame = pd.DataFrame({"date": ["2022-01-01", "2023-01-01", "2024-01-01"]})
    indices = np.arange(3)
    weights = training_row_weights(frame, indices, "2024-01-02", half_life_days=365)
    np.testing.assert_allclose(weights, np.array([0.25, 0.5, 1.0]) / (1.75 / 3))
    assert weights.mean() == pytest.approx(1.0)
    assert weights[:1].mean() < 1.0
    np.testing.assert_array_equal(training_row_weights(frame, indices, "2024-01-02", half_life_days=None), np.ones(3))
    with pytest.raises(ValueError, match="before|origin|future"):
        training_row_weights(frame, indices, "2024-01-01", half_life_days=365)


def test_calibration_uses_released_oos_rows_and_rejects_future_fit_origin():
    from src.models.a0_native_refit import calibration_indices

    frame = pd.DataFrame({
        "golgg_match_id": ["a", "b", "c", "d"],
        "date": ["2023-01-01", "2023-12-31", "2022-12-31", "2023-07-01"],
        "result_day": ["2023-01-02", "2024-01-01", "2023-01-01", "2023-07-02"],
    })
    cutoffs = np.array(["2023-01-01", "2023-12-01", "2022-12-01", "2023-07-01"])
    np.testing.assert_array_equal(calibration_indices(frame, cutoffs, "2024-01-01"), [0, 3])
    cutoffs[3] = "2023-08-01"
    with pytest.raises(ValueError, match="future|target|out.of.sample|OOF"):
        calibration_indices(frame, cutoffs, "2024-01-01")


def test_frozen_recipe_rejects_optimizer_settings_it_does_not_execute(tmp_path):
    import json

    from src.models.a0_native_refit import _protocol

    protocol = {
        "schema": "native_neural_cadence_decay_v1",
        "status": "FROZEN_BEFORE_TRAINING_AND_CANDIDATE_SCORING",
        "run_id": "optimizer_contract_probe",
        "model_identity": "Native A0 neural expert only; unchanged annual full Causal A0 is external reference. No non-neural expert or mixture objective changes; this is not an assembled full-A0 replacement.",
        "origins": pd.date_range("2023-01-01", "2026-09-01", freq="MS").strftime("%Y-%m-%d").tolist(),
        "warmup_year": 2023,
        "evaluation_years": [2024, 2025, 2026],
        "expected_canonical_rows": 11550,
        "arms": {
            "A": {"cadence": "annual", "weighting": "uniform"},
            "B": {"cadence": "annual", "weighting": "h365"},
            "C": {"cadence": "monthly", "weighting": "uniform"},
            "D": {"cadence": "monthly", "weighting": "h365"},
        },
        "training": {
            "seeds": [20260910, 20260911, 20260912], "epochs": 12,
            "batch_size": 128, "fresh_initialization": True, "shuffle": False,
            "device": "cpu", "threads_per_process": 2,
            "main_objective": "native stopped-series Beta terminal-score NLL; learned eta included in weighted main loss",
            "auxiliary_coefficient": 0.1, "half_life_days": 365,
            "optimizer": {
                "name": "AdamW", "lr": 0.1, "network_weight_decay": 0.01,
                "eta_weight_decay": 0.0, "gradient_clip": 5,
            },
        },
        "calibration": {
            "cadence": "monthly for all arms", "window_months": 12, "min_rows": 200,
            "objective": "Native joint three-seed stopped-series Beta calibration independently for all3optional-context modes, preserving gate routing",
        },
        "forecast": {"modes": ["full", "no_w20", "no_organization"]},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        _protocol(path)


def test_artifact_publication_hides_interrupted_writes_and_preserves_commits(tmp_path):
    from src.models.a0_native_refit import _artifact_bundle

    artifact = tmp_path / "seed" / "model.joblib"
    with pytest.raises(RuntimeError):
        with _artifact_bundle(artifact) as staged:
            staged.write_bytes(b"partial")
            assert not artifact.parent.exists()
            raise RuntimeError("interrupted writer")
    assert not artifact.parent.exists()

    with _artifact_bundle(artifact) as staged:
        staged.write_bytes(b"complete")
        staged.with_suffix(".joblib.receipt.json").write_text("{}")
        assert not artifact.parent.exists()
    assert artifact.read_bytes() == b"complete"
    assert artifact.with_suffix(".joblib.receipt.json").is_file()

    with pytest.raises(FileExistsError):
        with _artifact_bundle(artifact):
            pytest.fail("committed artifacts must not be overwritten")
    assert artifact.read_bytes() == b"complete"
