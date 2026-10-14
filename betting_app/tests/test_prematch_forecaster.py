"""Observable contracts for the from-scratch pre-draft forecasting pipeline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.prematch_features import ROLE_ORDER, build_features
from src.models.prematch_neural import NeuralModel
from src.models.prematch_tabular import TabularModel
from src.models.prematch_predictor import partition_indices


def _roster(team: str) -> tuple[str, ...]:
    return tuple(f"{team}-{role}" for role in ROLE_ORDER)


def _series(identifier: int, day: str, a: str = "A", b: str = "B", wins: tuple[int, ...] = (1,), best_of: int = 1) -> dict:
    maps = []
    for number, won in enumerate(wins):
        players = {}
        for side, team in (("t1", a), ("t2", b)):
            players[side + "_players"] = {
                role: {"player_id": pid, "player_name": pid, "champion_id": "99", "stats": {"csm": 7.0, "gpm": 400.0, "dpm": 500.0, "gd@15": 0.0}}
                for role, pid in zip(ROLE_ORDER, _roster(team))
            }
        maps.append({"game_id": str(identifier + number), "match_id": str(identifier), "date": day, "t1_id": a, "t2_id": b, "t1_name": a, "t2_name": b, "t1_win": bool(won), "t2_win": not bool(won), "draw": False, "game_duration": 1800, "source_best_of": best_of, **players})
    score_a, score_b = sum(wins), len(wins) - sum(wins)
    return {"match_id": str(identifier), "date": day, "tournament_name": "Test League", "t1_id": a, "t2_id": b, "t1_name": a, "t2_name": b, "sname_t1": a, "sname_t2": b, "best_of": best_of, "t1_score": score_a, "t2_score": score_b, "t1_win": score_a > score_b, "t2_win": score_b > score_a, "draw": score_a == score_b, "games": maps}


def _build(tmp_path: Path, name: str, records: list[dict], releases: dict | None = None):
    source = tmp_path / f"{name}.json"
    source.write_text(json.dumps(records))
    result = build_features(source, releases or {}, tmp_path / name)
    with np.load(result["arrays_path"]) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    return pd.read_parquet(result["metadata_path"]), arrays, joblib.load(result["state_path"])


def _row(metadata, arrays, identifier):
    index = metadata.index[metadata.match_id.astype(str).eq(str(identifier))].item()
    return {key: values[index] for key, values in arrays.items()}


def test_delayed_series_and_future_measurements_cannot_change_earlier_forecast(tmp_path):
    records = [_series(1, "2025-01-01"), _series(10, "2025-01-02"), _series(20, "2025-01-05")]
    metadata, arrays, _ = _build(tmp_path, "original", records, {"1": "2025-01-04"})
    early = _row(metadata, arrays, 10)
    np.testing.assert_allclose(early["baseline_series"], 0.5, atol=1e-12)
    changed = copy.deepcopy(records)
    changed[0] = _series(1, "2025-01-01", wins=(0,))
    changed[2] = _series(20, "2025-01-05", wins=(0,))
    changed[2]["games"][0]["t1_players"]["TOP"]["stats"]["dpm"] = 99999
    changed[2]["games"][0]["t1_players"]["TOP"]["champion_id"] = "1000"
    changed[2]["games"][0]["t1_players"]["TOP"]["player_id"] = "future-substitute"
    changed_metadata, changed_arrays, _ = _build(tmp_path, "changed", changed, {"1": "2025-01-04"})
    for key, value in early.items():
        np.testing.assert_allclose(value, _row(changed_metadata, changed_arrays, 10)[key], equal_nan=True, atol=1e-12)
    assert _row(metadata, arrays, 20)["baseline_series"][1] > _row(changed_metadata, changed_arrays, 20)["baseline_series"][1]
    assert metadata.loc[metadata.match_id.astype(str).eq("10"), "max_history_release_day"].isna().all()


def test_same_release_day_uses_frozen_opponents_not_map_id_order(tmp_path):
    records = [_series(1, "2025-01-01", "A", "B"), _series(10, "2025-01-01", "A", "C"), _series(20, "2025-01-02", "B", "C")]
    first_meta, first_arrays, _ = _build(tmp_path, "first", records)
    reordered = [_series(10, "2025-01-01", "A", "B"), _series(1, "2025-01-01", "A", "C"), records[2]]
    second_meta, second_arrays, _ = _build(tmp_path, "second", reordered)
    first = _row(first_meta, first_arrays, 20)
    second = _row(second_meta, second_arrays, 20)
    np.testing.assert_allclose(first["baseline_series"], 0.5, atol=1e-12)
    for key in first:
        np.testing.assert_allclose(first[key], second[key], equal_nan=True, atol=1e-12)


def test_bo2_history_updates_players_without_creating_binary_targets(tmp_path):
    records = [_series(1, "2025-01-01", wins=(1, 1), best_of=2), _series(10, "2025-01-02")]
    metadata, arrays, _ = _build(tmp_path, "draw_format", records)
    assert metadata.match_id.astype(str).tolist() == ["10"]
    assert arrays["baseline_series"][0, 1] > 0.5


def test_unknown_organization_queries_are_symmetric_pure_and_time_bounded(tmp_path):
    _, _, state = _build(tmp_path, "state", [_series(1, "2025-01-01")])
    before = joblib.hash(state)
    forward = state.features(None, _roster("A"), None, _roster("new"), "2025-01-03", 3)
    reverse = state.features(None, _roster("new"), None, _roster("A"), "2025-01-03", 3)
    np.testing.assert_allclose(forward["baseline_series"] + reverse["baseline_series"], 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(forward["players"], reverse["players"][::-1], equal_nan=True)
    assert joblib.hash(state) == before
    with pytest.raises(ValueError):
        state.features("A", _roster("A"), "B", _roster("B"), "2025-01-01", 1)
    with pytest.raises(ValueError):
        state.features("A", _roster("A"), "B", _roster("B"), "2025-01-03", 2)
    with pytest.raises(ValueError):
        state.features("A", _roster("A"), "B", _roster("A"), "2025-01-03", 1)


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_identical_unseen_rosters_have_exactly_neutral_control_features(tmp_path, best_of):
    _, _, state = _build(tmp_path, "cold", [_series(1, "2025-01-01")])
    before = joblib.hash(state)
    forward = state.features(None, _roster("unseen-A"), None, _roster("unseen-B"), "2025-01-03", best_of)
    reverse = state.features(None, _roster("unseen-B"), None, _roster("unseen-A"), "2025-01-03", best_of)
    # A one-ULP sign around 0.5 can cross a tree split and break query symmetry.
    np.testing.assert_array_equal(forward["baseline_series"], [.5, .5])
    np.testing.assert_array_equal(forward["baseline_series"], 1 - reverse["baseline_series"])
    assert joblib.hash(state) == before


def test_prior_roster_scenario_does_not_peek_at_target_substitute(tmp_path):
    records = [_series(1, "2025-01-01"), _series(10, "2025-01-02")]
    meta, arrays, _ = _build(tmp_path, "known", records)
    replaced = copy.deepcopy(records)
    replaced[1]["games"][0]["t1_players"]["TOP"]["player_id"] = "unannounced-substitute"
    changed_meta, changed_arrays, _ = _build(tmp_path, "substitution", replaced)
    for key in ("prior_players", "prior_teams", "prior_baseline_series"):
        np.testing.assert_allclose(_row(meta, arrays, 10)[key], _row(changed_meta, changed_arrays, 10)[key], equal_nan=True)
    assert not np.allclose(_row(meta, arrays, 10)["players"], _row(changed_meta, changed_arrays, 10)["players"], equal_nan=True)


def _learning_batch():
    rng = np.random.default_rng(13)
    players = rng.normal(size=(120, 2, 5, 4)).astype(np.float32)
    teams = rng.normal(size=(120, 2, 3)).astype(np.float32)
    players[::7, 0, 0, 2] = np.nan
    context = np.eye(3, dtype=np.float32)[np.arange(120) % 3]
    player_margin = players[:, 0, :, 0].mean(axis=1) - players[:, 1, :, 0].mean(axis=1)
    team_margin = teams[:, 0, 0] - teams[:, 1, 0]
    controls = 1 / (1 + np.exp(-np.column_stack((team_margin, player_margin)).astype(float)))
    y = (player_margin > 0).astype(float)
    return {"players": players, "teams": teams, "context": context, "baseline_series": controls}, y


@pytest.mark.parametrize("kind,pool_players", [
    ("logistic", False), ("lightgbm", False), ("neural", False), ("lightgbm", True),
])
def test_fitted_probability_complements_side_swap_and_survives_reload(tmp_path, kind, pool_players):
    batch, y = _learning_batch()
    if kind == "neural":
        model = NeuralModel({"hidden": 8, "epochs": 2, "dropout": .1, "weight_decay": .001}, seed=13)
    else:
        params = {"C": 1.0} if kind == "logistic" else {"num_leaves": 7, "min_child_samples": 10, "n_estimators": 20, "reg_lambda": 5.0}
        params["pool_players"] = pool_players
        model = TabularModel(kind, params, seed=13)
    model.fit(batch, y, np.ones(len(y)))
    swapped = {**batch, "players": batch["players"][:, ::-1], "teams": batch["teams"][:, ::-1], "baseline_series": 1 - batch["baseline_series"]}
    probabilities = model.predict_proba(batch)
    np.testing.assert_allclose(probabilities + model.predict_proba(swapped), 1.0, atol=1e-12, rtol=0)
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    np.testing.assert_allclose(joblib.load(artifact).predict_proba(batch), probabilities, atol=1e-12)
    cold = {key: value[:1].copy() for key, value in batch.items()}
    cold["players"][:] = np.nan
    cold["teams"][:] = np.nan
    cold["baseline_series"][:] = .5
    np.testing.assert_allclose(model.predict_proba(cold), .5, atol=1e-12)


def test_roster_aggregation_preserves_missingness_and_stable_dispersion():
    players = np.full((1, 2, 5, 2), np.nan)
    players[0, 0, :3, 0] = [1e9, 1e9 + 2, 1e9 + 4]
    players[0, 1, 0, 0] = 6
    players[0, 1, :2, 1] = [2, 4]
    teams = np.zeros((1, 2, 1))
    controls = np.full((1, 2), .5)
    fields = TabularModel._side_fields(players, teams, controls, pool_players=True)
    expected = [[[1e9 + 2, np.nan, np.sqrt(8 / 3), np.nan], [6, 3, 0, 1]]]
    np.testing.assert_allclose(fields[:, :, -4:], expected, equal_nan=True)
    permuted = TabularModel._side_fields(players[:, :, ::-1], teams, controls, pool_players=True)
    np.testing.assert_allclose(permuted[:, :, -4:], fields[:, :, -4:], equal_nan=True)


def test_fit_partitions_exclude_labels_released_on_their_origin():
    metadata = pd.DataFrame({
        "date": ["2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31", "2024-01-01", "2024-04-01"],
        "release_day": ["2023-01-01", "2023-07-01", "2023-07-01", "2024-01-01", "2024-01-01", "2024-04-01"],
    })
    train, calibration, forecast = partition_indices(metadata, "2024-01-01")
    np.testing.assert_array_equal(np.flatnonzero(train), [0])
    np.testing.assert_array_equal(np.flatnonzero(calibration), [2])
    np.testing.assert_array_equal(np.flatnonzero(forecast), [4])


def test_raw_only_origin_uses_recent_labels_but_excludes_origin_day_releases(tmp_path):
    from scripts import train_prematch_forecaster as runner

    batch, labels = _learning_batch()
    arrays = {key: np.concatenate([value] * 10 + [value[:4]]) for key, value in batch.items()}
    y = np.concatenate([labels] * 10 + [labels[:4]])
    metadata = pd.DataFrame({
        "match_id": [str(i) for i in range(len(y))],
        "date": ["2023-01-01"] * 600 + ["2023-12-20"] * 600 + ["2024-01-02"] * 4,
        "release_day": ["2023-01-01"] * 600 + ["2023-12-21"] * 599 + ["2024-01-01"] + ["2024-01-02"] * 4,
        "y": y,
    })
    config = {"id": "logistic", "kind": "logistic", "params": {"C": .1}}
    protocol = {
        "calibration_months": 0, "candidate_variants": ["raw"],
        "prediction_dir": str(tmp_path / "predictions"), "model_dir": str(tmp_path / "models"),
        "training_half_life_days": 730, "seed": 13, "probability_epsilon": 1e-6,
    }
    forecasts, members, _ = runner.run_origins(
        protocol, metadata, arrays, ["2024-01-01"], [config], "development"
    )
    train_batch = {key: value[:1199] for key, value in arrays.items()}
    query = {key: value[1200:] for key, value in arrays.items()}
    ages = (pd.Timestamp("2024-01-01") - pd.to_datetime(metadata.date[:1199])).dt.days.to_numpy()
    oracle = TabularModel("logistic", {"C": .1}, seed=13).fit(
        train_batch, y[:1199], np.exp2(-ages / 730)
    )
    np.testing.assert_allclose(forecasts["logistic__raw"], oracle.predict_proba(query), atol=1e-12)
    assert forecasts.match_id.tolist() == ["1200", "1201", "1202", "1203"]
    with pytest.raises(ValueError):
        members["logistic"].predict_proba(query, "temperature")
    reloaded = joblib.load(tmp_path / "models/development/2024-01-01/logistic.joblib")
    np.testing.assert_allclose(reloaded.predict_proba(query, "raw"), forecasts["logistic__raw"], atol=1e-12)


def test_negative_calibration_window_cannot_admit_future_training_labels():
    metadata = pd.DataFrame({"date": ["2024-01-02"], "release_day": ["2024-01-02"]})
    with pytest.raises(ValueError):
        partition_indices(metadata, "2024-01-01", calibration_months=-1)


def test_disabling_calibration_rejects_temperature_recipe(tmp_path):
    from scripts import train_prematch_forecaster as runner

    batch, y = _learning_batch()
    protocol = {"calibration_months": 0, "candidate_variants": ["raw", "temperature"]}
    with pytest.raises(ValueError):
        runner.fit_member(
            {"id": "logistic", "kind": "logistic", "params": {"C": .1}},
            batch, y, np.ones(len(y)), {key: value[:0] for key, value in batch.items()},
            y[:0], protocol, "2024-01-01", "2024-01-01",
        )


def test_feature_reuse_preserves_source_and_rejects_changed_artifact(tmp_path):
    from scripts import train_prematch_forecaster as runner

    source, release = tmp_path / "source.json", tmp_path / "release.json"
    source.write_text(json.dumps([_series(1, "2025-01-01"), _series(2, "2025-01-02")]))
    release.write_text("{}")
    original = {
        "source": str(source), "source_sha256": runner.sha256(source),
        "release_days": str(release), "release_days_sha256": runner.sha256(release),
        "feature_dir": str(tmp_path / "features"), "report_dir": str(tmp_path / "original"),
    }
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps(original))
    runner.prepare(original, original_path)
    before = runner.sha256(tmp_path / "features/arrays.npz")
    reused = {**original, "report_dir": str(tmp_path / "reused"), "feature_source_protocol": str(original_path)}
    reused_path = tmp_path / "reused.json"
    reused_path.write_text(json.dumps(reused))
    runner.prepare(reused, reused_path)
    _, arrays, _ = runner.load_features(reused, reused_path)
    assert len(arrays["players"]) == 2
    assert runner.sha256(tmp_path / "features/arrays.npz") == before
    with (tmp_path / "features/arrays.npz").open("ab") as handle:
        handle.write(b"changed")
    rejected = {**reused, "report_dir": str(tmp_path / "rejected")}
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected))
    with pytest.raises(ValueError):
        runner.prepare(rejected, rejected_path)


def test_map_side_order_does_not_reassign_results_or_player_histories(tmp_path):
    records = [_series(1, "2025-01-01", wins=(1, 1), best_of=3), _series(10, "2025-01-02")]
    first_meta, first_arrays, _ = _build(tmp_path, "map_order", records)
    reversed_map = copy.deepcopy(records)
    game = reversed_map[0]["games"][1]
    for suffix in ("id", "name", "players", "win"):
        game[f"t1_{suffix}"], game[f"t2_{suffix}"] = game[f"t2_{suffix}"], game[f"t1_{suffix}"]
    other_meta, other_arrays, _ = _build(tmp_path, "reversed_map", reversed_map)
    np.testing.assert_array_equal(first_meta.y, other_meta.y)
    for key in first_arrays:
        np.testing.assert_allclose(first_arrays[key], other_arrays[key], equal_nan=True, atol=1e-12, rtol=0)


@pytest.mark.parametrize("query_day", ["2025-01-03", "2035-01-03"])
@pytest.mark.parametrize("best_of", [3, 5])
def test_series_control_integrates_uncertainty_before_projecting(tmp_path, best_of, query_day):
    from scipy.integrate import quad
    from scipy.special import expit
    from scipy.stats import binom, norm
    from src.models.prematch_features import TEAM_FEATURE_NAMES

    _, _, state = _build(tmp_path, "quadrature", [_series(1, "2025-01-01")])
    query = ("A", _roster("A"), "B", _roster("B"), query_day)
    series = state.features(*query, best_of)
    map_probability = state.features(*query, 1)["baseline_series"][0]
    mean_index = TEAM_FEATURE_NAMES.index("slow_rating_mean")
    sd_index = TEAM_FEATURE_NAMES.index("slow_rating_sd")
    mean = float(series["teams"][0, mean_index] - series["teams"][1, mean_index])
    variance = float(np.square(series["teams"][:, sd_index].astype(float)).sum())

    def integrand(z):
        p = expit(mean + np.sqrt(variance) * z)
        return binom.sf(best_of // 2, best_of, p) * norm.pdf(z)

    oracle, _ = quad(integrand, -10, 10, epsabs=1e-11)
    assert series["baseline_series"][0] == pytest.approx(oracle, abs=2e-7)
    plug_in_projection = binom.sf(best_of // 2, best_of, map_probability)
    assert abs(series["baseline_series"][0] - plug_in_projection) > 1e-4


@pytest.mark.parametrize("missing_header", ["t1_id", "t2_id"])
def test_partial_header_identity_survives_reversed_first_map(tmp_path, missing_header):
    record = _series(1, "2025-01-01")
    record.pop(missing_header)
    game = record["games"][0]
    for suffix in ("id", "name", "players", "win"):
        game[f"t1_{suffix}"], game[f"t2_{suffix}"] = game[f"t2_{suffix}"], game[f"t1_{suffix}"]
    metadata, _, state = _build(tmp_path, "partial_header", [record])
    assert metadata.loc[0, ["team_a", "team_b", "y"]].tolist() == ["A", "B", 1]
    future = state.features("A", _roster("A"), "B", _roster("B"), "2025-01-02", 1)
    assert (future["baseline_series"] > .5).all()


def test_final_stage_rejects_changed_shared_calibration_before_fitting(tmp_path, monkeypatch):
    from scripts import train_prematch_forecaster as runner

    protocol = {"report_dir": str(tmp_path), "candidates": [], "final_origins": ["2026-01-01"]}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    prepared = {"feature_artifact_hashes": {}}
    selection = {
        "protocol_sha256": runner.sha256(protocol_path),
        "code_sha256": runner.code_hashes(),
        "feature_artifact_hashes": {},
        "family_best": {},
        "recipe": [],
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    original_hash = runner.sha256
    changed_dependency = "betting_app/ml/calibration/candidate_calibration.py"
    monkeypatch.setattr(runner, "sha256", lambda path: "changed" if str(path) == changed_dependency else original_hash(path))
    monkeypatch.setattr(runner, "load_features", lambda *args: (pd.DataFrame(), {}, prepared))

    def unexpected_fit(*args, **kwargs):
        pytest.fail("A changed calibration implementation reached model fitting")

    monkeypatch.setattr(runner, "run_origins", unexpected_fit)
    with pytest.raises(ValueError):
        runner.finalize(protocol, protocol_path)
