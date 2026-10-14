"""Contracts for distinct, reproducible full-history research derivatives."""
from __future__ import annotations
import sys
import json

import numpy as np
import pandas as pd
import pytest


from scripts import run_full_historical_predictions as runner


def test_exp081_model_version_distinguishes_focal_loss_gamma() -> None:
    assert runner.exp081_model_version(1.0) == "exp081-priorday-canonical79-focal-g1-annual-v1"
    assert runner.exp081_model_version(0.0) == "exp081-priorday-canonical79-bce-g0-annual-v1"


def test_exp081_loss_metadata_distinguishes_bce_from_focal() -> None:
    assert runner.exp081_loss_name(1.0) == "focal_loss_gamma_1"
    assert runner.exp081_loss_name(0.0) == "binary_cross_entropy"



def test_invalid_gamma_does_not_create_output_directory(monkeypatch, tmp_path) -> None:
    output = tmp_path / "invalid-run"
    monkeypatch.setattr(sys, "argv", ["runner", "--exp081-gamma", "-1", "--output", str(output)])
    try:
        runner.main()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid gamma must be rejected")
    assert not output.exists()

def test_exp081_model_version_rejects_invalid_gamma() -> None:
    for gamma in (-0.1, float("inf"), float("nan")):
        try:
            runner.exp081_model_version(gamma)
        except ValueError:
            continue
        raise AssertionError(f"gamma={gamma!r} must be rejected")


def test_reported_bagging_and_regularization_change_fitted_probabilities() -> None:
    import numpy as np
    from scipy.special import expit
    from scripts.train_and_tune_siamese_series import train_single_member

    rng = np.random.RandomState(9)
    features = rng.normal(size=(48, 4))
    labels = (features[:, 0] + 0.3 * features[:, 1] > 0).astype(float)
    common = dict(epochs=3, batch_size=8, seed=81, d_h1=8, d_h2=4)
    weak = train_single_member(features, labels, features, labels,
                               weight_decay=0.0001, bootstrap_fraction=1.0, **common)
    strong = train_single_member(features, labels, features, labels,
                                 weight_decay=0.01, bootstrap_fraction=0.8, **common)
    weak_p = expit(weak.forward_anti_symmetric(features)[0])
    strong_p = expit(strong.forward_anti_symmetric(features)[0])
    assert not np.allclose(weak_p, strong_p)
    assert np.allclose(
        strong_p + expit(strong.forward_anti_symmetric(-features)[0]), 1.0,
        atol=1e-12,
    )


def test_training_defaults_preserve_seeded_historical_recipe() -> None:
    from scripts.train_and_tune_siamese_series import train_single_member

    rng = np.random.RandomState(17)
    x = rng.normal(size=(25, 4))
    y = (x[:, 0] > 0).astype(float)
    args = dict(epochs=2, batch_size=8, seed=81, d_h1=8, d_h2=4)
    old = train_single_member(x, y, x, y, **args)
    explicit = train_single_member(x, y, x, y, weight_decay=1e-4, bootstrap_fraction=1.0, **args)
    repeated = train_single_member(x, y, x, y, weight_decay=.01, bootstrap_fraction=.8, **args)
    same = train_single_member(x, y, x, y, weight_decay=.01, bootstrap_fraction=.8, **args)
    np.testing.assert_array_equal(old.forward_anti_symmetric(x)[0], explicit.forward_anti_symmetric(x)[0])
    np.testing.assert_array_equal(repeated.forward_anti_symmetric(x)[0], same.forward_anti_symmetric(x)[0])
    assert old.platt_slope == explicit.platt_slope
    assert repeated.platt_slope == same.platt_slope


@pytest.mark.parametrize("kwargs", [
    {"weight_decay": -1}, {"weight_decay": float("nan")}, {"weight_decay": float("inf")},
    {"bootstrap_fraction": 0}, {"bootstrap_fraction": 1.1}, {"bootstrap_fraction": float("nan")},
])
def test_training_rejects_invalid_regularization_or_bagging(kwargs) -> None:
    from scripts.train_and_tune_siamese_series import train_single_member

    with pytest.raises(ValueError):
        train_single_member(np.ones((2, 4)), np.array([0, 1]),
                            np.ones((2, 4)), np.array([0, 1]), epochs=1, **kwargs)


def test_rating_control_converts_map_probability_once_and_reverses() -> None:
    frame = pd.DataFrame({"player_gl": [.7, .7, .7], "best_of": [1, 3, 5]})
    probabilities = runner.rating_series_probabilities(frame)
    np.testing.assert_allclose(probabilities, [.7, .784, .83692])
    np.testing.assert_allclose(probabilities + runner.rating_series_probabilities(frame, reverse=True), 1)
    frame.loc[0, "best_of"] = 2
    with pytest.raises(ValueError, match="Bo1/3/5"):
        runner.rating_series_probabilities(frame)


def test_reuse_rejects_unaudited_old_rows_before_creating_output(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text("[]")
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "audit.json").write_text(json.dumps({
        "status": "complete", "feature_contract": runner.FEATURE_CONTRACT,
        "source": {"sha256": runner.sha256(source)},
        "temporal_violations": 0, "source_transitions_rejected_or_skipped": 0,
        "availability_certified": False,
    }))
    output = tmp_path / "new"
    output.mkdir()
    with pytest.raises(ValueError, match="provenance"):
        runner.reuse_replay(source, previous, output)
    assert not (output / "replay").exists()


def test_matched_strong_artifacts_roundtrip_member_and_ensemble_calibration(tmp_path) -> None:
    from scripts.train_and_tune_siamese_series import build_training_features
    from src.models.siamese_series import SiameseSeriesModel
    from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS

    rows = []
    for year in (2018, 2019, 2020):
        for bo in (1, 3, 5):
            for index in range(4):
                row = {name: 1.0 for name in _REQUIRED_BASE_FIELDS}
                row.update({name: .25 + .15 * index for name in _PROBABILITY_FIELDS})
                row.update(best_of=bo, date=f"{year}-06-{index+1:02d}",
                           y_true=index % 2, golgg_match_id=f"{year}-{bo}-{index}")
                rows.append(row)
    frame = pd.DataFrame(rows)
    x, names = build_training_features(frame)
    masks = tuple(frame.date.str.startswith(str(year)).to_numpy() for year in (2018, 2019, 2020))
    scales = x[masks[0]].std(axis=0)
    scales[scales == 0] = 1
    fold_dir = tmp_path / "models/2020"
    fold_dir.mkdir(parents=True)
    (tmp_path / "replay").mkdir()
    (tmp_path / "replay/snapshots.csv").write_text(frame.to_csv(index=False))
    (tmp_path / "replay/audit.json").write_text("{}")
    (tmp_path / "feature_columns.json").write_text(json.dumps(names))
    (fold_dir / "recipe.json").write_text("{}")
    fold = {"year": 2020, "origin": "2020-01-01", "symmetry_max_error": {},
            "fit": {"date_max": "2018-06-04", "result_day_max": "2018-06-04"},
            "calibration": {"date_max": "2019-06-04", "result_day_max": "2019-06-04", "row_ids_sha256": "test-cal"}}
    protocol = {"source": {"sha256": "synthetic"}, "code_sha256": {}, "limitations": ["synthetic"],
        "models": runner.BENCHMARK_MODELS,
        "config": {"linear79": {"C": .1}, "exp081_strong": {
            "epochs": 2, "batch_size": 8, "lr": .003, "gamma": 1,
            "d_h1": 8, "d_h2": 4, "weight_decay": .01, "bootstrap_fraction": .8, "seeds": [17, 34]}}}
    result = frame.loc[masks[2]].copy()
    runner.fit_benchmark_controls(frame, masks, tuple(x[m] / scales for m in masks),
                                  names, scales, result, fold, fold_dir, protocol)
    for suffix in ("strong", "strong_ensemble"):
        artifact = json.loads((fold_dir / f"exp081_{suffix}.json").read_text())
        model = SiameseSeriesModel.from_mapping(artifact)
        for i, row in result.iterrows():
            probability = model.predict(row.to_dict(), best_of=int(row.best_of))
            assert probability == pytest.approx(result.loc[i, f"p_exp081_{suffix}"])
        assert artifact["eligible_from"] == "2020-01-01T00:00:00+00:00"
        assert artifact["feature_contract_sha256"] == runner.json_digest(runner.FEATURE_CONTRACT)
        assert artifact["ensemble_calibration"]["calibration_end"] < artifact["eligible_from"]
    assert max(fold["symmetry_max_error"].values()) < 1e-8
