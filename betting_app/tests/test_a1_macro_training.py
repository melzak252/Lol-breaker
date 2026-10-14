"""Behavioral contracts for the development-only chronological macro experiment."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.team_macro import train_and_benchmark_macro as macro


def _synthetic_inputs():
    years = [2024] * 4 + [2025] * 4 + [2026] * 4
    dates = [f"{year}-03-{day:02d}" for year in (2024, 2025, 2026) for day in range(1, 5)]
    ids = [f"match-{index:02d}" for index in range(len(years))]
    targets = np.array([0, 1, 0, 1] * 3, dtype=np.int8)
    metadata = pd.DataFrame(
        {
            "golgg_match_id": ids,
            "team1_id": [f"team-a-{index}" for index in range(len(ids))],
            "team2_id": [f"team-b-{index}" for index in range(len(ids))],
            "date": dates,
            "result_day": dates,
            "best_of": [3] * len(ids),
            "y": targets,
            "feature_source_max_day": [f"{year}-02-28" for year in years],
        }
    )
    paired = metadata.copy()
    paired["p_full"] = np.tile([0.35, 0.65, 0.4, 0.6], 3)
    w20 = np.zeros((len(ids), 2, 10), dtype=np.float32)
    w20[:, 0, 1] = np.arange(len(ids), dtype=np.float32) * 2.0
    w20[:, 1, 1] = np.arange(len(ids), dtype=np.float32) * 0.25
    return metadata, paired, w20


def test_future_test_label_mutation_cannot_change_selected_checkpoint(tmp_path):
    metadata, paired, w20 = _synthetic_inputs()
    original_dir = tmp_path / "original"
    macro.run_experiment(metadata, paired, w20, original_dir, {}, seed=31, epochs=4)

    changed_metadata = metadata.copy()
    changed_paired = paired.copy()
    test_rows = pd.to_datetime(metadata["date"]).dt.year == 2026
    changed_metadata.loc[test_rows, "y"] = 1 - changed_metadata.loc[test_rows, "y"]
    changed_paired.loc[test_rows, "y"] = 1 - changed_paired.loc[test_rows, "y"]
    changed_dir = tmp_path / "changed-test-labels"
    macro.run_experiment(changed_metadata, changed_paired, w20, changed_dir, {}, seed=31, epochs=4)

    original_state = torch.load(original_dir / "macro_mlp_state.pt", map_location="cpu", weights_only=True)
    changed_state = torch.load(changed_dir / "macro_mlp_state.pt", map_location="cpu", weights_only=True)
    assert original_state["model_state_dict"].keys() == changed_state["model_state_dict"].keys()
    assert original_state["train_end"] == changed_state["train_end"] == "2024-03-04"
    assert original_state["model_selection_end"] == changed_state["model_selection_end"] == "2025-03-04"
    assert original_state["status"] == "DEVELOPMENT_ONLY"
    assert original_state["calibration_end"] is None
    assert original_state["fit_partition"] == changed_state["fit_partition"] == "train_2024"
    for name in original_state["model_state_dict"]:
        torch.testing.assert_close(
            original_state["model_state_dict"][name],
            changed_state["model_state_dict"][name],
            rtol=0,
            atol=0,
        )

    original_predictions = pd.read_parquet(original_dir / "predictions.parquet")
    changed_predictions = pd.read_parquet(changed_dir / "predictions.parquet")
    np.testing.assert_allclose(original_predictions["p_a0_macro"], changed_predictions["p_a0_macro"], rtol=0, atol=0)
    original_metrics = json.loads((original_dir / "metrics.json").read_text())
    changed_metrics = json.loads((changed_dir / "metrics.json").read_text())
    assert original_metrics["test_2026_partial_canonical_coverage"]["rows_scored"] == 4
    assert changed_metrics["test_2026_partial_canonical_coverage"]["rows_scored"] == 4


def test_label_release_boundaries_exclude_labels_not_available_at_next_origin():
    metadata, paired, w20 = _synthetic_inputs()
    late_rows = pd.DataFrame(
        [
            {
                "golgg_match_id": "late-train-label",
                "team1_id": "late-a",
                "team2_id": "late-b",
                "date": "2024-12-31",
                "result_day": "2025-01-01",
                "best_of": 3,
                "y": 1,
                "feature_source_max_day": "2024-12-30",
                "p_full": 0.7,
            },
            {
                "golgg_match_id": "late-validation-label",
                "team1_id": "late-c",
                "team2_id": "late-d",
                "date": "2025-12-31",
                "result_day": "2026-01-01",
                "best_of": 3,
                "y": 0,
                "feature_source_max_day": "2025-12-30",
                "p_full": 0.4,
            },
        ]
    )
    metadata = pd.concat([metadata, late_rows.drop(columns="p_full")], ignore_index=True)
    paired = pd.concat([paired, late_rows], ignore_index=True)
    w20 = np.concatenate([w20, np.zeros((2, 2, 10), dtype=np.float32)], axis=0)

    cohort = macro.prepare_cohort(metadata, paired, w20)
    ids = set(cohort["frame"]["golgg_match_id"])
    assert "late-train-label" not in ids
    assert "late-validation-label" not in ids
    assert cohort["exclusions"]["train_label_not_released_before_2025_origin"] == 1
    assert cohort["exclusions"]["validation_label_not_released_before_2026_origin"] == 1
    assert cohort["counts"]["included_train_2024"] == 4
    assert cohort["counts"]["included_validation_2025"] == 4


def test_published_predictions_have_honest_cutoffs_and_only_test_rows(tmp_path):
    metadata, paired, w20 = _synthetic_inputs()
    unmatched = paired.iloc[[-1]].copy()
    unmatched.loc[:, "golgg_match_id"] = "canonical-without-feature"
    unmatched.loc[:, "team1_id"] = "unmatched-a"
    unmatched.loc[:, "team2_id"] = "unmatched-b"
    paired = pd.concat([paired, unmatched], ignore_index=True)
    output = tmp_path / "new-run"
    manifest = macro.run_experiment(metadata, paired, w20, output, {}, seed=47, epochs=3)

    predictions = pd.read_parquet(output / "predictions.parquet")
    test_ids = set(metadata.loc[pd.to_datetime(metadata.date).dt.year == 2026, "golgg_match_id"])
    train_ids = set(metadata.loc[pd.to_datetime(metadata.date).dt.year == 2024, "golgg_match_id"])
    assert set(predictions.golgg_match_id) == test_ids
    assert set(predictions.golgg_match_id).isdisjoint(train_ids)
    assert set(predictions.cohort) == {"TEST_2026"}
    assert predictions.train_end.eq("2024-03-04").all()
    assert predictions.model_selection_end.eq("2025-03-04").all()
    assert predictions.calibration_end.isna().all()
    assert manifest["status"] == "DEVELOPMENT_ONLY"
    assert manifest["cutoffs"]["calibration_performed"] is False
    assert manifest["cutoffs"]["train_end"] == "2024-03-04"
    assert manifest["cutoffs"]["model_selection_end"] == "2025-03-04"
    assert manifest["cohort_counts"]["published_oos_predictions"] == 4
    assert manifest["canonical_coverage"]["published_test_rows"] == 4
    assert manifest["canonical_coverage"]["canonical_baseline_rows"] == 5
    assert manifest["canonical_coverage"]["fraction"] == pytest.approx(0.8)
    assert manifest["exclusions"]["missing_feature_metadata"] == 1
    assert "not full 11,550-row canonical coverage" in manifest["claim_scope"]


def test_existing_output_directory_is_preserved_without_training(tmp_path):
    output = tmp_path / "existing-run"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original artifact")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        macro.run_experiment(
            pd.DataFrame(), pd.DataFrame(), np.empty((0, 2, 10)), output, {}
        )
    assert marker.read_text() == "original artifact"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]


def test_saved_checkpoint_reuses_same_test_predictions_without_training(tmp_path):
    metadata, paired, w20 = _synthetic_inputs()
    output = tmp_path / "checkpoint-run"
    macro.run_experiment(metadata, paired, w20, output, {}, seed=53, epochs=3)
    cohort = macro.prepare_cohort(metadata, paired, w20)
    test_rows = cohort["frame"]["partition"].to_numpy() == "test_2026"
    expected = pd.read_parquet(output / "predictions.parquet")["p_a0_macro"].to_numpy()

    restored = macro.load_trained_model(output / "macro_mlp_state.pt")
    actual = macro.predict_probabilities(
        restored,
        cohort["features"][test_rows],
        cohort["frame"].loc[test_rows, "p_full"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_absent_history_dates_do_not_poison_unrelated_canonical_rows():
    metadata, paired, w20 = _synthetic_inputs()
    extra = metadata.iloc[[0]].copy()
    extra["golgg_match_id"] = "unrelated-cold-start"
    extra["feature_source_max_day"] = None
    metadata = pd.concat([metadata, extra], ignore_index=True)
    w20 = np.concatenate([w20, np.zeros((1, 2, 10), dtype=np.float32)])
    cohort = macro.prepare_cohort(metadata, paired, w20)
    assert set(cohort["frame"]["golgg_match_id"]) == set(paired["golgg_match_id"])


def test_unknown_nonzero_macro_history_has_explicit_exclusion():
    metadata, paired, w20 = _synthetic_inputs()
    metadata.loc[11, "feature_source_max_day"] = None
    paired.loc[11, "feature_source_max_day"] = None
    cohort = macro.prepare_cohort(metadata, paired, w20)
    assert "match-11" not in set(cohort["frame"]["golgg_match_id"])
    assert cohort["exclusions"]["unverifiable_feature_cutoff"] == 1
