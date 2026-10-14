from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.analysis.a0_uncertainty as uncertainty


def _members(mean: float, spread: float = 0.0) -> list[float]:
    offsets = np.array([-2, -1, 0, 1, 2], dtype=float) * spread / np.sqrt(2)
    values = np.clip(mean + offsets, 0.001, 0.999)
    values += mean - values.mean()
    return values.tolist()


def _fixture(*, banded: bool = False):
    years = (2024, 2025, 2026)
    arms = ("seed_only", "block_bootstrap")
    cal_rows = []
    test_rows = []
    baseline_rows = []
    for year in years:
        cal_values = []
        test_values = []
        if banded:
            favorites = (0.55, 0.65, 0.75, 0.85, 0.95)
            for fav in favorites:
                cal_values.extend([(fav, 0.005), (fav, 0.005), (fav, 0.03), (fav, 0.03)])
                for month in (1, 2):
                    test_values.extend([(fav, 0.005, 1, month), (fav, 0.03, 0, month)])
        else:
            cal_values = [(0.50, 0.0), (0.80, 0.01), (0.50, 0.0), (0.80, 0.01)]
            test_values = [(0.50, 0.0, 1, 1), (0.80, 0.01, 1, 1),
                           (0.50, 0.0, 0, 1), (0.80, 0.01, 0, 1)]
        for arm in arms:
            for index, (probability, spread) in enumerate(cal_values):
                match_id = f"cal-{year}-{index}"
                match_date = pd.Timestamp(year=year - 1, month=11, day=1 + index)
                cal_rows.append(_row(match_id, year, arm, "cal", match_date, index % 2,
                                     probability, spread, _members(probability, spread)))
            for index, (probability, spread, outcome, month) in enumerate(test_values):
                match_id = f"test-{year}-{index}"
                match_date = pd.Timestamp(year=year, month=month, day=index + 1)
                row = _row(match_id, year, arm, "test", match_date, outcome,
                           probability, spread, _members(probability, spread))
                test_rows.append(row)
                if arm == arms[0]:
                    baseline_rows.append({key: row[key] for key in (
                        "golgg_match_id", "date", "result_day", "feature_source_max_day",
                        "team1_id", "team2_id", "best_of", "y")}
                        | {"p_full": 0.5})
    predictions = pd.DataFrame(cal_rows + test_rows)
    baseline = pd.DataFrame(baseline_rows)
    expected = {
        year: {
            "cal": sum(r["model_year"] == year and r["variant"] == arms[0] for r in cal_rows),
            "test": sum(r["model_year"] == year and r["variant"] == arms[0] for r in test_rows),
        }
        for year in years
    }
    return predictions, baseline, expected


def _patch_contract(monkeypatch, expected):
    monkeypatch.setattr(uncertainty, "EXPECTED_PARTITION_ROWS", expected)
    monkeypatch.setattr(uncertainty, "CANONICAL_EXPECTED_ROWS",
                        sum(rows["test"] for rows in expected.values()))
    monkeypatch.setattr(uncertainty, "YEARS", tuple(expected))
    monkeypatch.setattr(uncertainty, "RETENTION_LEVELS", (1.0, 0.9, 0.75, 0.5))
    monkeypatch.setattr(uncertainty, "FAVORITE_PROBABILITY_BANDS",
                        (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


def _row(match_id, year, arm, partition, date, outcome, probability, spread, members):
    day = date.strftime("%Y-%m-%d")
    prior_day = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {
        "golgg_match_id": match_id,
        "date": day,
        "result_day": day,
        "feature_source_max_day": prior_day,
        "team1_id": "team-a",
        "team2_id": "team-b",
        "best_of": 3,
        "y": outcome,
        "model_year": year,
        "variant": arm,
        "partition": partition,
        "train_end": f"{year - 2}-12-31",
        "calibration_end": f"{year - 1}-12-31",
        "p_member_0": members[0],
        "p_member_1": members[1],
        "p_member_2": members[2],
        "p_member_3": members[3],
        "p_member_4": members[4],
        "p": probability,
        "p_no_w20": probability,
        "p_no_organization": probability,
    }


def test_cal_thresholds_are_unchanged_when_test_labels_change(monkeypatch, tmp_path):
    predictions, baseline, expected = _fixture()
    _patch_contract(monkeypatch, expected)
    first = uncertainty.evaluate_uncertainty(predictions, baseline, tmp_path / "first")

    changed_predictions = predictions.copy()
    test_mask = changed_predictions.partition.eq("test")
    changed_predictions.loc[test_mask, "y"] = 1 - changed_predictions.loc[test_mask, "y"]
    changed_baseline = baseline.copy()
    changed_baseline["y"] = 1 - changed_baseline["y"]
    second = uncertainty.evaluate_uncertainty(
        changed_predictions, changed_baseline, tmp_path / "second")

    assert first["thresholds"] == second["thresholds"]


def test_test_coverage_and_side_identity_are_locked(monkeypatch, tmp_path):
    predictions, baseline, expected = _fixture()
    _patch_contract(monkeypatch, expected)
    missing = predictions.drop(predictions.index[predictions.partition.eq("test")][0])
    with pytest.raises(ValueError, match="coverage"):
        uncertainty.evaluate_uncertainty(missing, baseline, tmp_path / "missing")

    wrong_side = predictions.copy()
    candidate_id = wrong_side.loc[wrong_side.partition.eq("test"), "golgg_match_id"].iloc[0]
    wrong_side.loc[wrong_side.golgg_match_id.eq(candidate_id), "team1_id"] = "reversed-or-wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        uncertainty.evaluate_uncertainty(wrong_side, baseline, tmp_path / "identity")


def test_entropy_and_spread_selectors_distinguish_equal_coinflip_members(monkeypatch, tmp_path):
    predictions, baseline, expected = _fixture()
    _patch_contract(monkeypatch, expected)
    uncertainty.evaluate_uncertainty(predictions, baseline, tmp_path / "selectors")
    selected = pd.read_csv(tmp_path / "selectors" / "selections.csv")
    subset = selected[(selected.variant == "seed_only") & (selected.model_year == 2024)
                      & (selected.partition == "test") & (selected.retention == 0.5)]
    spread_ids = set(subset.loc[subset.selector == "p_std", "golgg_match_id"])
    entropy_ids = set(subset.loc[subset.selector == "entropy", "golgg_match_id"])
    assert spread_ids == {"test-2024-0", "test-2024-2"}
    assert entropy_ids == {"test-2024-1", "test-2024-3"}


def test_favorite_band_standardized_excess_loss_finds_controlled_risk_signal(monkeypatch, tmp_path):
    predictions, baseline, expected = _fixture(banded=True)
    _patch_contract(monkeypatch, expected)
    result = uncertainty.evaluate_uncertainty(predictions, baseline, tmp_path / "signal")
    diagnostic = result["adjusted_excess_loss"]["block_bootstrap"]["by_model_year"]["2024"]
    assert diagnostic["status"] == "estimated"
    assert diagnostic["point_estimate"] > 0
    assert diagnostic["ci95"][0] > 0
    assert diagnostic["blocks"] == 2
    assert diagnostic["bootstrap_draws"] >= 5000


def test_cal_policy_profiles_accept_annual_calibration_but_reject_unreleased_labels(monkeypatch, tmp_path):
    predictions, baseline, expected = _fixture()
    _patch_contract(monkeypatch, expected)
    result = uncertainty.evaluate_uncertainty(predictions, baseline, tmp_path / "annual_cal")
    cutoff = next(
        row for row in result["thresholds"]
        if row["variant"] == "block_bootstrap" and row["model_year"] == 2024
        and row["selector"] == "p_std" and row["retention"] == 0.5
    )
    assert cutoff["threshold"] == 0.0
    changed = predictions.copy()
    row = changed.index[(changed.partition == "cal") & (changed.model_year == 2024)][0]
    changed.loc[row, "result_day"] = "2024-01-01"
    with pytest.raises(ValueError):
        uncertainty.evaluate_uncertainty(changed, baseline, tmp_path / "unreleased")


@pytest.mark.parametrize(
    ("column", "late_bound"),
    [("train_end", "2023-01-01"), ("calibration_end", "2024-01-01")],
)
def test_annual_fit_bounds_do_not_advance_with_test_date(monkeypatch, tmp_path, column, late_bound):
    predictions, baseline, expected = _fixture()
    _patch_contract(monkeypatch, expected)
    row = predictions.index[
        predictions.partition.eq("test") & predictions.model_year.eq(2024)
        & predictions.date.eq("2024-01-02")
    ][0]
    predictions.loc[row, column] = late_bound
    with pytest.raises(ValueError):
        uncertainty.evaluate_uncertainty(predictions, baseline, tmp_path / column)


def test_cal_profiles_use_identical_matches_in_both_arms(monkeypatch, tmp_path):
    predictions, baseline, expected = _fixture()
    _patch_contract(monkeypatch, expected)
    row = predictions.index[
        predictions.partition.eq("cal") & predictions.variant.eq("block_bootstrap")
    ][0]
    predictions.loc[row, "golgg_match_id"] = "substituted-cal-match"
    with pytest.raises(ValueError):
        uncertainty.evaluate_uncertainty(predictions, baseline, tmp_path / "different_cal")

