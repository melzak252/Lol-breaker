"""Regression contracts for experimental Siamese training, not frozen-model quality."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from scripts.train_and_tune_siamese_series import (
    build_model_artifact,
    build_training_features,
    evaluate_ensemble,
    fit_platt_scaling,
    focal_loss_and_grad,
    merge_training_snapshots,
    train_single_member,
    write_json_exclusive,
)
from src.models.siamese_series import SiameseSeriesModel
from src.models.symmetric_series import (
    _PROBABILITY_FIELDS,
    _REQUIRED_BASE_FIELDS,
    build_feature_mapping,
)


def snapshot_frame():
    row = {name: 1.0 for name in _REQUIRED_BASE_FIELDS}
    row.update({name: 0.68 for name in _PROBABILITY_FIELDS})
    row.update(
        golgg_match_id=17,
        best_of=3,
        date="2024-01-02",
        team1_id=11,
        team2_id=12,
        y_true=1,
    )
    row["t1_rolling_towers"] = 7.0
    return pd.DataFrame([row])


def test_training_vector_matches_runtime_with_real_w20_and_series_effects():
    frame = snapshot_frame()
    X, names = build_training_features(frame)
    canonical = build_feature_mapping(frame.iloc[0].to_dict(), best_of=3)
    assert len(names) == 79
    assert list(names) == list(canonical)
    np.testing.assert_array_equal(X[0], list(canonical.values()))
    assert X[0, names.index("w20_towers_diff")] == 6.0
    changed = frame.copy()
    changed["best_of"] = 1
    changed["t1_rolling_towers"] = 1.0
    altered, _ = build_training_features(changed)
    assert altered[0, names.index("w20_towers_diff")] == 0.0
    assert X[0, names.index("team_elo_logit_bo3")] > 0.0
    assert altered[0, names.index("team_elo_logit_bo3")] == 0.0
    assert X[0, names.index("team_elo_series_amplification")] > 0.0


@pytest.mark.parametrize("missing", ["best_of", "t1_rolling_towers", "team_elo_r1"])
def test_missing_raw_inputs_are_not_inferred_or_zero_filled(missing):
    frame = snapshot_frame().drop(columns=[missing])
    frame["BoN"] = 3
    with pytest.raises(ValueError, match=missing):
        build_training_features(frame)


@pytest.mark.parametrize(
    "field,value", [("best_of", 2), ("team_elo", np.nan), ("t1_rolling_gold", np.inf)]
)
def test_invalid_raw_inputs_are_rejected(field, value):
    frame = snapshot_frame()
    frame[field] = value
    with pytest.raises(ValueError):
        build_training_features(frame)


def split_sources():
    full = snapshot_frame()
    rolling_fields = [
        name for name in full if name.startswith(("t1_rolling_", "t2_rolling_"))
    ]
    metadata = ["golgg_match_id", "team1_id", "team2_id", "date"]
    return full.drop(columns=rolling_fields), full[metadata + rolling_fields]


def test_rolling_merge_preserves_rows_and_canonical_values():
    ratings, rolling = split_sources()
    ratings2 = ratings.copy()
    ratings2["golgg_match_id"] = 18
    rolling2 = rolling.copy()
    rolling2["golgg_match_id"] = 18
    rolling2["t1_rolling_towers"] = 9.0
    merged = merge_training_snapshots(
        pd.concat([ratings2, ratings]), pd.concat([rolling, rolling2])
    )
    assert merged["golgg_match_id"].tolist() == [18, 17]
    X, names = build_training_features(merged)
    np.testing.assert_array_equal(X[:, names.index("w20_towers_diff")], [8.0, 6.0])


@pytest.mark.parametrize(
    "problem",
    ["duplicate_ratings", "duplicate_rolling", "swapped", "date", "unmatched"],
)
def test_merge_rejects_ambiguous_or_unmatched_snapshots(problem):
    ratings, rolling = split_sources()
    if problem == "swapped":
        ratings2 = ratings.copy()
        ratings2["golgg_match_id"] = 18
        rolling2 = rolling.copy()
        rolling2["golgg_match_id"] = 18
        ratings = pd.concat([ratings, ratings2], ignore_index=True)
        rolling = pd.concat([rolling, rolling2], ignore_index=True)
    if problem == "duplicate_ratings":
        ratings = pd.concat([ratings, ratings])
    elif problem == "duplicate_rolling":
        rolling = pd.concat([rolling, rolling])
    elif problem == "swapped":
        rolling.loc[1, ["team1_id", "team2_id"]] = [12, 11]
    elif problem == "date":
        rolling["date"] = "2024-01-03"
    else:
        rolling["golgg_match_id"] = 999
    with pytest.raises(ValueError):
        merge_training_snapshots(ratings, rolling)


def test_export_roundtrip_matches_trained_ensemble_and_refuses_overwrite(tmp_path):
    frame = pd.concat([snapshot_frame()] * 4, ignore_index=True)
    frame["team_elo"] = [0.2, 0.4, 0.6, 0.8]
    X, names = build_training_features(frame)
    scale = np.maximum(np.std(X, axis=0), 1.0)
    scaled = X / scale
    y = np.array([0, 1, 0, 1])
    members = [
        train_single_member(scaled, y, scaled, y[:, None], epochs=2, seed=seed)
        for seed in [3, 4]
    ]
    artifact = build_model_artifact(
        members, names, scale, provenance={"purpose": "synthetic regression"}
    )
    path = tmp_path / "experiment.json"
    write_json_exclusive(path, artifact)
    loaded = SiameseSeriesModel.from_mapping(json.loads(path.read_text()))
    for i, row in frame.iterrows():
        row = row.to_dict()
        np.testing.assert_array_equal(loaded._prepare_vector(row, best_of=3), scaled[i])
        logits = np.array(
            [
                m.forward_anti_symmetric(scaled[i : i + 1])[0].item() * m.platt_slope
                for m in members
            ]
        )
        expected = expit(np.mean(logits))
        assert loaded.predict(row, best_of=3) == pytest.approx(expected)
        _, sigma, low, low_b = loaded.predict_with_uncertainty(row, best_of=3)
        assert sigma == pytest.approx(np.std(logits))
        assert low == pytest.approx(expit(np.mean(logits) - 0.75 * np.std(logits)))
        assert low_b == pytest.approx(expit(-np.mean(logits) - 0.75 * np.std(logits)))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"replacement": True})
    assert path.read_bytes() == original
    assert evaluate_ensemble(members, scaled, y) == evaluate_ensemble(
        members, scaled, y[:, None]
    )


def test_focal_gradient_is_shape_safe_and_matches_loss_derivative():
    y = np.array([0, 1, 1])
    z = np.array([[-0.7], [0.2], [1.1]])
    loss, gradient = focal_loss_and_grad(y, z, gamma=1.5)
    column_loss, column_gradient = focal_loss_and_grad(y[:, None], z, gamma=1.5)
    assert loss == column_loss
    assert gradient.shape == z.shape
    np.testing.assert_array_equal(gradient, column_gradient)
    step = 1e-6
    for index in range(len(y)):
        delta = np.zeros_like(z)
        delta[index] = step
        high = focal_loss_and_grad(y, z + delta, gamma=1.5)[0]
        low = focal_loss_and_grad(y, z - delta, gamma=1.5)[0]
        assert gradient[index, 0] / len(y) == pytest.approx(
            (high - low) / (2 * step), rel=1e-5
        )


@pytest.mark.parametrize("magnitude", [0.1, 10.0])
def test_calibration_can_fit_beyond_old_grid_and_accepts_columns(magnitude):
    z = np.full(10, magnitude)
    y = np.array([1] * 8 + [0] * 2)
    slope = fit_platt_scaling(z[:, None], y)
    assert slope == pytest.approx(np.log(4.0) / magnitude, rel=1e-4)
    assert fit_platt_scaling(z, y[:, None]) == pytest.approx(slope)


def test_metrics_handle_single_class_without_fabricated_auc():
    frame = snapshot_frame()
    X, _ = build_training_features(frame)
    model = train_single_member(X, np.array([1]), X, np.array([1]), epochs=1)
    result = evaluate_ensemble([model], X, np.array([[1]]))
    assert result["roc_auc"] is None
    assert np.isfinite(result["log_loss"])
