import numpy as np
import pandas as pd
import pytest

from src.analysis.a0_sports_correction import evaluate_correction


def protocol(rows=8, primary_rows=4):
    return {
        "evaluation": {
            "expected_rows": rows,
            "primary_start": "2025-01-01",
            "primary_expected_rows": primary_rows,
            "resamples": 5000,
            "seed": 20261001,
            "primary_blocks": "calendar_month",
            "sensitivity_blocks": "source_tournament_name_within_year_not_verified_editions",
            "guard_ll_margin": 0.005,
            "critical_guards": ["bo3", "bo5", "major", "source_unseen_player"],
            "predictive_success": "Both primary post-warmup paired LL intervals have upper bound below zero; candidate Brier point does not worsen versus either comparator. Full 11550 coverage retained; all-row LL/Brier also improve versus A0.",
            "columns": {"anchor": "p_a0", "control": "p_calibration", "candidate": "p_correction", "specialist": "p_specialist"},
        },
        "profiles": ["year", "best_of", "competition_tier", "competition_family", "experience", "coplay", "confidence", "disagreement", "unknown_players", "new_organization", "w20_min_gate"],
    }


def frame():
    n = 8
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    dates = pd.to_datetime(["2024-02-01", "2024-06-01", "2024-10-01", "2024-12-01", "2025-02-01", "2025-05-01", "2026-02-01", "2026-05-01"])
    out = pd.DataFrame({
        "golgg_match_id": [f"m{i}" for i in range(n)], "date": dates,
        "result_day": dates + pd.Timedelta(days=1),
        "feature_source_max_day": dates - pd.Timedelta(days=1),
        "best_of": [3, 5, 3, 5, 3, 5, 3, 5], "y": y,
        "tournament_name": ["unknown cup"] * n,
        "p_a0": [.70, .30, .70, .30, .70, .30, .70, .30],
        "p_specialist": [.65, .35, .65, .35, .65, .35, .65, .35],
        "p_calibration": [.65, .35, .65, .35, .65, .35, .65, .35],
        "p_correction": [.80, .20, .80, .20, .80, .20, .80, .20],
        "unknown_players": [0, 1] * 4, "new_organization": [0, 1] * 4,
        "w20_min_gate": [0, 1] * 4, "experience": [1, 2] * 4,
        "coplay": [2, 3] * 4, "competition_tier": ["unknown"] * n,
        "competition_family": ["unknown"] * n, "confidence": [.7, .7] * 4,
        "disagreement": [.1, .2] * 4,
    })
    return out


def test_evaluation_retains_declared_cohort_and_returns_same_row_contrasts():
    result = evaluate_correction(frame(), protocol())
    assert result["cohort"]["rows"] == 8
    assert result["cohort"]["primary_rows"] == 4
    contrast = result["contrasts"]["post_warmup"]["candidate_minus_anchor"]["log_loss"]
    assert contrast["n"] == 4
    assert contrast["delta"] < 0
    assert contrast["draws"] == 5000
    assert result["metrics"]["full"]["p_correction"]["sample_size"] == 8


def test_changed_cohort_and_duplicate_identity_are_rejected():
    with pytest.raises(ValueError, match="expected_rows"):
        evaluate_correction(frame().iloc[:-1], protocol())
    duplicate = frame()
    duplicate.loc[1, "golgg_match_id"] = duplicate.loc[0, "golgg_match_id"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_correction(duplicate, protocol())


def test_extreme_probabilities_are_rejected_not_clipped():
    bad = frame()
    bad.loc[0, "p_correction"] = 1.0
    with pytest.raises(ValueError, match="strict"):
        evaluate_correction(bad, protocol())


def test_neutral_anchor_is_excluded_and_fixed_favorite_uses_anchor_direction():
    data = frame().iloc[:4].copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(["2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"])
    data["result_day"] = data.date + pd.Timedelta(days=1)
    data["tournament_name"] = ["unknown"] * 4
    data["p_a0"] = [.5, .8, .2, .8]
    data["p_correction"] = [.99, .1, .9, .1]
    data["y"] = [1, 1, 0, 1]
    result = evaluate_correction(data, protocol(rows=4, primary_rows=4))
    fav = result["favorite_calibration"]["post_warmup"]
    assert fav["summary"]["neutral_excluded"] == 1
    assert fav["summary"]["fixed_a0_favorite"]["p_correction"]["n"] == 3
    # All three non-neutral rows favor the A0 favorite; the correction predicts its opposite.
    fixed = fav["summary"]["fixed_a0_favorite"]["p_correction"]
    assert fixed["mean_probability_assigned_to_a0_favorite"] == pytest.approx(.1)
    assert fixed["gap"] == pytest.approx(-.9)
    assert fav["summary"]["own_favorite"]["p_correction"]["n"] == 4


def test_gate_requires_both_predictive_contrasts_and_noninferiority_guards():
    result = evaluate_correction(frame(), protocol())
    assert result["decision"]["predictive_success"] is True
    assert result["decision"]["eligible"] is False
    assert set(result["guards"]) == {"bo3", "bo5", "major", "source_unseen_player"}
    assert result["limitations"]["reliability_established"] is False
    stronger_control = frame()
    stronger_control["p_calibration"] = np.where(stronger_control.y, .9, .1)
    assert evaluate_correction(stronger_control, protocol())["decision"]["predictive_success"] is False
    assert result["limitations"]["prospective_evidence"] is False


def _c0_case():
    dates = pd.to_datetime(["2024-02-01", "2024-06-01", "2025-02-01", "2025-06-01"] +
                           [f"2026-{month:02d}-01" for month in range(1, 10)])
    y = np.arange(len(dates)) % 2
    primary = dates.year == 2026
    data = pd.DataFrame({
        "golgg_match_id": [f"c0-toy-{i}" for i in range(len(dates))], "date": dates, "y": y,
        "best_of": np.resize([1, 3, 5], len(dates)), "tournament_name": "LCK Spring 2026",
        "unknown_players": 1., "head_origin": dates.to_period("Q").start_time.strftime("%Y-%m-%d"),
    })
    favorites = {
        "p_a0": np.where(primary, .8, .55), "p_c0": np.where(primary, .8, .9),
        "p_specialist": np.full(len(dates), .6),
        "p_connector03": np.where(primary, .82, .55),
        "p_c0_calibration": np.where(primary, .81, .9),
        "p_c0_connector": np.where(primary, .7, .9),
    }
    for name, favorite in favorites.items():
        data[name] = np.where(y, favorite, 1 - favorite)
    return data, {
        "cohort": {"rows": 13, "primary_rows": 9, "primary_start": "2026-01-01",
                   "primary_end_exclusive": "2026-09-09"},
        "evaluation": {"primary_contrasts": ["p_c0", "p_c0_calibration", "p_connector03"]},
    }


def test_c0_full_annual_gain_cannot_override_failed_primary_contrasts():
    from src.analysis.a0_sports_correction import evaluate_c0
    data, config = _c0_case()
    result = evaluate_c0(data, config)
    assert result["metrics"]["full"]["p_c0_connector"]["log_loss"] < result["metrics"]["full"]["p_a0"]["log_loss"]
    assert result["contrasts"]["p_c0"]["log_loss"]["delta"] == pytest.approx(np.log(.8 / .7))
    assert result["gates"]["primary_all_three"] is False
    assert result["decision"] == "NOT_ADOPTED"


def test_c0_keeps_zero_support_month_draws_and_unknown_profile_rows():
    from src.analysis.a0_sports_correction import evaluate_c0
    data, config = _c0_case()
    data["unknown_players"] = 0.
    data.loc[4, "unknown_players"] = 1.
    data.loc[5, "unknown_players"] = np.nan
    result = evaluate_c0(data, config)
    unseen = result["guards"]["source_unseen_player"]
    assert unseen["n"] == 1
    assert unseen["unknown_profile_rows"] == 1
    interval = unseen["comparators"]["p_c0"]["log_loss"]
    assert interval["zero_support_replicates"] > 0
    assert interval["ci_upper"] is None
    assert result["limitations"]["conditional_zero_support_unresolved"] is True
    assert result["decision"] == "NOT_ADOPTED"


def test_c0_missing_control_and_extreme_valid_probability_policy():
    from src.analysis.a0_sports_correction import evaluate_c0
    data, config = _c0_case()
    with pytest.raises(ValueError, match="p_c0_calibration"):
        evaluate_c0(data.drop(columns="p_c0_calibration"), config)
    data.loc[4, ["y", "p_c0_connector"]] = [1, 1e-20]
    result = evaluate_c0(data, config)
    primary = data.loc[data.date.dt.year == 2026]
    expected = -np.where(primary.y, np.log(primary.p_c0_connector), np.log1p(-primary.p_c0_connector)).mean()
    assert result["metrics"]["primary"]["p_c0_connector"]["log_loss"] == pytest.approx(expected, abs=1e-12)


def test_c0_calibration_cells_keep_actual_annual_a0_favorite():
    from src.analysis.a0_sports_correction import evaluate_c0
    data, config = _c0_case()
    data["y"] = 1
    data["p_a0"] = .8
    data["p_c0"] = .2
    data["p_c0_connector"] = .3
    result = evaluate_c0(data, config)["favorite_calibration"]["primary"]
    fixed = result["summary"]["fixed_a0_favorite"]["p_c0_connector"]
    own = result["summary"]["own_favorite"]["p_c0_connector"]
    assert fixed["gap"] == pytest.approx(-.7)
    assert own["gap"] == pytest.approx(.7)
    cells = result["fixed_a0_quarter_confidence"]
    assert sum(cell["n"] for cell in cells["cells"]) == 9
    assert cells["rms_gap"]["p_c0_connector"] == pytest.approx(.7)
