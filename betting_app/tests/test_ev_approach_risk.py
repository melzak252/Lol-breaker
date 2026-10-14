"""Chronological risk diagnostics must not imply per-match betting guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.ev_approach_policies import select_policy_bets
from src.analysis.ev_approach_risk import chronological_risk_predictions


def _opportunities():
    decision_at = pd.date_range("2025-01-01", periods=240, freq="D", tz="UTC")
    return pd.DataFrame({
        "row_id": [f"r{i}" for i in range(len(decision_at))],
        "golgg_match_id": [f"m{i}" for i in range(len(decision_at))],
        "cohort": "synthetic",
        "horizon": "48h",
        "date": decision_at,
        "decision_at": decision_at,
        "start_at": decision_at + pd.Timedelta(hours=48),
        "label_available_at": decision_at + pd.Timedelta(days=3),
        "y": [i % 2 for i in range(len(decision_at))],
        "p_s": [0.35 + (i % 7) * 0.04 for i in range(len(decision_at))],
        "q": [0.38 + (i % 5) * 0.04 for i in range(len(decision_at))],
        "odds_a": 2.0,
        "odds_b": 2.0,
        "bookmaker_a": "book-a",
        "bookmaker_b": "book-b",
        "quote_id_a": [f"qa{i}" for i in range(len(decision_at))],
        "quote_id_b": [f"qb{i}" for i in range(len(decision_at))],
        "quote_at_a": decision_at,
        "quote_at_b": decision_at,
        "best_of": 3,
        "league": "synthetic-league",
        "information_status": "historical_scenario",
        "strict_prior_day": True,
        "is_common": True,
    })


def test_multicalibration_predictions_do_not_change_when_unreleased_future_labels_change():
    frame = _opportunities()
    changed = frame.copy()
    future_ids = frame.loc[frame.index >= 180, "row_id"]
    changed.loc[changed.row_id.isin(future_ids), "y"] = 1 - changed.loc[
        changed.row_id.isin(future_ids), "y"
    ]

    before, _ = chronological_risk_predictions(frame, "multicalibration", alpha=0.1)
    after, _ = chronological_risk_predictions(changed, "multicalibration", alpha=0.1)

    earlier_ids = frame.loc[frame.decision_at < frame.loc[180, "decision_at"], "row_id"]
    left = before.set_index("row_id").loc[earlier_ids]
    right = after.set_index("row_id").loc[earlier_ids]
    assert left["status"].eq("available").any()
    baseline_p = frame.set_index("row_id").loc[earlier_ids, "p_s"]
    available = left["status"].eq("available")
    assert (left.loc[available, "p"].sub(baseline_p.loc[available]).abs() > 1e-6).any()
    np.testing.assert_allclose(left["p"], right["p"])
    for column in ("p_low_a", "p_low_b", "allow_a", "allow_b"):
        if column in left and column in right:
            pd.testing.assert_series_equal(left[column], right[column], check_names=False)


def test_sequential_residual_monitor_waits_for_outcome_release_before_feedback():
    frame = _opportunities().iloc[:3].copy()
    frame["label_available_at"] = frame["decision_at"] + pd.Timedelta(days=10)

    result, receipt = chronological_risk_predictions(frame, "sequential_residual", alpha=0.1)

    assert result["row_id"].tolist() == frame["row_id"].tolist()
    assert result["status"].eq("unavailable").all()
    assert result[["p_low_a", "p_low_b"]].isna().all().all()
    assert not result[["allow_a", "allow_b"]].to_numpy(dtype=bool).any()
    assert receipt["lower_probability_guarantee"] is False


def test_adaptive_conformal_batches_feedback_with_identical_release_times():
    frame = _opportunities().iloc[:3].copy()
    frame["decision_at"] = pd.to_datetime(
        ["2025-01-01", "2025-01-02", "2025-01-05"], utc=True
    )
    frame["label_available_at"] = pd.to_datetime(
        ["2025-01-04", "2025-01-04", "2025-01-20"], utc=True
    )

    result, _ = chronological_risk_predictions(frame, "adaptive_conformal", alpha=0.1)

    forecast = result.set_index("row_id").loc["r2"]
    np.testing.assert_allclose(forecast["adaptive_alpha"], 0.102, atol=1e-12)
    assert forecast["adaptive_updates"] == 1


def test_bayesian_and_robust_kelly_keep_coherent_mean_and_swap_side_bounds():
    frame = _opportunities()
    # Give both orientations the same finite calibration history with released labels.
    frame["label_available_at"] = frame["start_at"] + pd.Timedelta(days=1)
    swapped = frame.copy()
    swapped["p_s"] = 1.0 - frame["p_s"]
    swapped["q"] = 1.0 - frame["q"]
    swapped["y"] = 1 - frame["y"]
    swapped["odds_a"], swapped["odds_b"] = frame["odds_b"], frame["odds_a"]
    swapped["bookmaker_a"], swapped["bookmaker_b"] = frame["bookmaker_b"], frame["bookmaker_a"]
    swapped["quote_id_a"], swapped["quote_id_b"] = frame["quote_id_b"], frame["quote_id_a"]
    swapped["quote_at_a"], swapped["quote_at_b"] = frame["quote_at_b"], frame["quote_at_a"]

    for method in ("bayesian", "robust_kelly"):
        forward, receipt = chronological_risk_predictions(frame, method, alpha=0.1)
        reverse, _ = chronological_risk_predictions(swapped, method, alpha=0.1)
        np.testing.assert_allclose(forward["p"] + reverse["p"], 1.0, atol=1e-12)
        # Conservative bounds are side-specific diagnostics, not a coherent mean pair.
        np.testing.assert_allclose(forward["p_low_a"], reverse["p_low_b"], equal_nan=True)
        np.testing.assert_allclose(forward["p_low_b"], reverse["p_low_a"], equal_nan=True)
        available = forward["status"].eq("available")
        assert available.any()
        assert np.isfinite(forward.loc[available, ["p_low_a", "p_low_b"]].to_numpy()).all()
        assert receipt["lower_probability_guarantee"] is False


def test_exact_neutral_history_does_not_create_a_side_direction():
    frame = _supported_history([1] * 20, 0.5, 0.5)
    frame["q"] = 0.5


    swapped_query = frame.copy()
    current = swapped_query.index[-1]
    for left, right in (
        ("odds_a", "odds_b"),
        ("bookmaker_a", "bookmaker_b"),
        ("quote_id_a", "quote_id_b"),
        ("quote_at_a", "quote_at_b"),
    ):
        swapped_query.loc[current, [left, right]] = frame.loc[current, [right, left]].to_numpy()

    for method in ("bayesian", "robust_kelly"):
        forward, _ = chronological_risk_predictions(frame, method, alpha=0.1)
        reverse, _ = chronological_risk_predictions(swapped_query, method, alpha=0.1)
        a = forward.iloc[-1]
        b = reverse.iloc[-1]
        assert a["status"] == b["status"] == "unavailable"
        assert a["support_count"] == b["support_count"] == 0
        assert a["p"] == b["p"] == 0.5
        assert pd.isna(a["p_low_a"]) and pd.isna(a["p_low_b"])
        assert pd.isna(b["p_low_a"]) and pd.isna(b["p_low_b"])
        assert not bool(a["allow_a"]) and not bool(a["allow_b"])
        assert not bool(b["allow_a"]) and not bool(b["allow_b"])


def _supported_history(history_y, history_p, current_p):
    history_y = list(history_y)
    count = len(history_y)
    frame = _opportunities().iloc[:count + 1].copy()
    decisions = pd.date_range("2025-01-01", periods=count + 1, freq="3D", tz="UTC")
    frame["date"] = decisions
    frame["decision_at"] = decisions
    frame["start_at"] = decisions + pd.Timedelta(hours=48)
    frame["quote_at_a"] = decisions
    frame["quote_at_b"] = decisions
    frame["p_s"] = [*([history_p] * count), current_p]
    frame["y"] = [*history_y, 0]
    frame["label_available_at"] = frame["start_at"] + pd.Timedelta(hours=12)
    return frame



def test_supported_robust_kelly_bounds_bound_its_mean_and_are_selectable():
    frame = _supported_history([1] * 20, 0.6, 0.6)

    forecast, _ = chronological_risk_predictions(frame, "robust_kelly", alpha=0.1)
    row = forecast.iloc[-1]

    assert row["status"] == "available"
    assert row["support_count"] == 20
    assert row["p"] == 1.0
    assert row["p_low_a"] <= row["p"] <= row["p_high_a"]
    assert row["p_low_b"] <= 1.0 - row["p"] <= row["p_high_b"]

    candidate = frame.iloc[[-1]].copy()
    for column in ("p", "p_low_a", "p_low_b", "allow_a", "allow_b"):
        candidate[column] = row[column]
    selected = select_policy_bets(
        candidate, policy="P0", tax_rate=0.12, min_ev=0.03, qualification="raw",
    )
    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "uncertainty_basis"] == "assumed_research_haircut"
    assert selected.loc[0, "uncertainty_reason"] == "provided_research_risk_bound_unverified"


def test_robust_kelly_support_does_not_block_tax_free_sensitivity():
    frame = _supported_history([1] * 20, 0.6, 0.6)
    frame.loc[frame.index[-1], "odds_a"] = 1.5

    forecast, _ = chronological_risk_predictions(frame, "robust_kelly", alpha=0.1)
    row = forecast.iloc[-1]
    assert row["p_low_a"] * 1.5 - 1.0 > 0.03
    assert row["p_low_a"] * 1.5 * (1.0 - 0.12) - 1.0 <= 0.03

    candidate = frame.iloc[[-1]].copy()
    for column in ("p", "p_low_a", "p_low_b", "allow_a", "allow_b"):
        candidate[column] = row[column]
    tax_free = select_policy_bets(
        candidate, policy="P0", tax_rate=0.0, min_ev=0.03, qualification="raw",
    )
    primary_tax = select_policy_bets(
        candidate, policy="P0", tax_rate=0.12, min_ev=0.03, qualification="raw",
    )
    assert tax_free["side"].tolist() == ["A"]
    assert primary_tax.empty


@pytest.mark.parametrize("qualification", ["raw", "application"])
@pytest.mark.parametrize("policy", ["P0", "P5"])
@pytest.mark.parametrize("declared_basis,expected_basis", [
    (None, "assumed_research_haircut"),
    (pd.NA, "assumed_research_haircut"),
    ("unknown_research_bound", "assumed_research_haircut"),
    ("assumed_research_haircut", "assumed_research_haircut"),
    ("measured_bounds", "measured_bounds"),
])
def test_provided_risk_score_preserves_numeric_selection_and_only_declared_provenance(
    qualification, policy, declared_basis, expected_basis,
):
    candidate = _opportunities().iloc[[0]].copy()
    candidate["p"] = 0.60
    candidate["q"] = 0.55
    candidate["odds_a"] = 2.2
    candidate["odds_b"] = 1.4
    candidate["p_low_a"] = 0.56
    candidate["p_low_b"] = 0.30
    candidate["sigma_z"] = 0.20
    candidate["variant"] = "seed_only"
    candidate["model_year"] = 2025
    candidate["uncertainty_basis"] = declared_basis
    candidate["uncertainty_reason"] = "caller_risk_method"

    selected = select_policy_bets(
        candidate, policy=policy, tax_rate=0.12, min_ev=0.03,
        qualification=qualification, cal_thresholds={("seed_only", 2025): 0.20},
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "p"] == 0.60
    assert selected.loc[0, "prob_conservative"] == 0.56
    assert selected.loc[0, "prob_for_ev"] == selected.loc[0, "p_for_ev"] == 0.56
    assert selected.loc[0, "ev"] == pytest.approx(0.56 * 2.2 * 0.88 - 1.0)
    assert selected.loc[0, "uncertainty_basis"] == expected_basis
    assert selected.loc[0, "uncertainty_reason"] == "caller_risk_method"


@pytest.mark.parametrize("policy", ["P0", "P5"])
def test_risk_bound_equal_to_mean_does_not_change_admitted_mean_identity(policy):
    candidate = _opportunities().iloc[[0]].copy()
    candidate["p"] = candidate["p_low_a"] = 0.60
    candidate["odds_a"] = 2.2
    candidate["odds_b"] = 1.4
    candidate["p_low_b"] = 0.40
    candidate["sigma_z"] = 0.20
    candidate["variant"] = "seed_only"
    candidate["model_year"] = 2025
    candidate["uncertainty_basis"] = "measured_bounds"

    selected = select_policy_bets(
        candidate, policy=policy, tax_rate=0.12, min_ev=0.03,
        qualification="raw", cal_thresholds={("seed_only", 2025): 0.20},
    )

    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "prob_conservative"] is None
    assert selected.loc[0, "prob_for_ev"] == 0.60
    assert selected.loc[0, "uncertainty_basis"] == "mean"


def test_measured_risk_bound_does_not_certify_a_combined_research_policy_score():
    candidate = _opportunities().iloc[[0]].copy()
    candidate["p"] = 0.60
    candidate["q"] = 0.55
    candidate["odds_a"] = 2.2
    candidate["odds_b"] = 1.4
    candidate["p_low_a"] = 0.59
    candidate["p_low_b"] = 0.39
    candidate["uncertainty_basis"] = "measured_bounds"
    candidate["uncertainty_reason"] = "caller_measured_bound"

    selected = select_policy_bets(
        candidate, policy="P1", tax_rate=0.12, min_ev=0.03, qualification="application",
    )

    expected = 1.0 / (1.0 + np.exp(-(np.log(0.60 / 0.40) - 0.75 * 0.15)))
    assert selected["side"].tolist() == ["A"]
    assert selected.loc[0, "prob_for_ev"] == pytest.approx(expected)
    assert selected.loc[0, "uncertainty_basis"] == "assumed_research_haircut"
    assert "caller_measured_bound" in selected.loc[0, "uncertainty_reason"]


def test_sequential_residual_reports_supported_boundary_and_alarm():
    frame = _supported_history([1] * 20, 0.01, 0.6)

    result, _ = chronological_risk_predictions(frame, "sequential_residual", alpha=0.1)
    forecast = result.iloc[-1]
    expected_boundary = np.sqrt(
        2.0 * np.log((np.pi ** 2) * (20 ** 2) / (3.0 * 0.1)) / 20
    )

    assert forecast["status"] == "available"
    assert forecast["support_count"] == 20
    assert forecast["residual_mean"] == pytest.approx(0.99)
    assert forecast["residual_boundary"] == pytest.approx(expected_boundary)
    assert forecast["residual_alarm"]


def test_conformal_supported_order_statistic_and_set_membership_change_at_rank():
    one_error = _supported_history([0, *([1] * 19)], 0.8, 0.8)
    two_errors = _supported_history([0, 0, *([1] * 18)], 0.8, 0.8)

    one_result, _ = chronological_risk_predictions(one_error, "conformal", alpha=0.1)
    two_result, _ = chronological_risk_predictions(two_errors, "conformal", alpha=0.1)
    one = one_result.iloc[-1]
    two = two_result.iloc[-1]

    assert one["status"] == two["status"] == "available"
    assert one["conformal_support"] == two["conformal_support"] == 20
    assert one["conformal_quantile"] == pytest.approx(0.2)
    assert bool(one["set_contains_a"])
    assert not bool(one["set_contains_b"])
    assert one["prediction_set_size"] == 1
    assert two["conformal_quantile"] == pytest.approx(0.8)
    assert bool(two["set_contains_a"]) and bool(two["set_contains_b"])
    assert two["prediction_set_size"] == 2


def test_selective_risk_accepts_and_rejects_with_sufficient_released_support():
    perfect = _supported_history([1] * 120, 0.6, 0.6)
    one_error_y = [0, *([1] * 119)]
    one_error = _supported_history(one_error_y, 0.6, 0.6)

    accepted, _ = chronological_risk_predictions(perfect, "selective_risk", alpha=0.1)
    rejected, _ = chronological_risk_predictions(one_error, "selective_risk", alpha=0.1)
    forecast = accepted.iloc[-1]
    abstention = rejected.iloc[-1]
    expected_radius = np.sqrt(np.log(1.0 / 0.1) / (2.0 * 120))

    assert forecast["status"] == "available"
    assert forecast["risk_selected_support"] == 120
    assert forecast["risk_threshold"] == 0.5
    assert forecast["risk_estimate"] == 0.0
    assert forecast["risk_upper_diagnostic"] == pytest.approx(expected_radius)
    assert bool(forecast["allow_a"]) and not bool(forecast["allow_b"])
    assert abstention["status"] == "unavailable"
    assert pd.isna(abstention["risk_threshold"])
    assert not bool(abstention["allow_a"]) and not bool(abstention["allow_b"])
    assert abstention["missing_reason"] == "no_threshold_passed_prior_risk_diagnostic"


def test_empty_calibration_abstains_instead_of_reporting_zero_risk_or_perfect_coverage():
    frame = _opportunities().iloc[:1].copy()
    frame["label_available_at"] = frame["decision_at"] + pd.Timedelta(days=7)

    result, receipt = chronological_risk_predictions(frame, "selective_risk", alpha=0.1)

    assert result.loc[0, "status"] == "unavailable"
    assert not bool(result.loc[0, "allow_a"])
    assert not bool(result.loc[0, "allow_b"])
    assert pd.isna(result.loc[0, "p_low_a"])
    assert pd.isna(result.loc[0, "p_low_b"])
    assert receipt["lower_probability_guarantee"] is False
    assert receipt["available_rows"] == 0


def test_conformal_set_reports_set_membership_without_turning_coverage_into_a_probability_bound():
    frame = _opportunities().iloc[:2].copy()
    frame["label_available_at"] = frame["decision_at"] + pd.Timedelta(days=5)

    result, receipt = chronological_risk_predictions(frame, "conformal", alpha=0.1)

    assert result["status"].eq("unavailable").all()
    assert result[["p_low_a", "p_low_b"]].isna().all().all()
    assert receipt["lower_probability_guarantee"] is False
    assert receipt["available_rows"] == 0

def test_feedback_proof_excludes_future_releases_and_hashes_eligible_neutral_rows():
    import hashlib

    frame = _opportunities().iloc[:3].copy()
    frame["decision_at"] = pd.to_datetime([
        "2025-01-01T00:00:00Z", "2025-01-03T00:00:00Z", "2025-01-05T00:00:00Z",
    ])
    frame["label_available_at"] = pd.to_datetime([
        "2025-01-04T00:00:00Z", "2025-01-06T00:00:00Z", "2025-01-07T00:00:00Z",
    ])
    frame.loc[0, "p_s"] = 0.5

    predictions, _ = chronological_risk_predictions(frame, "sequential_residual")
    first_feedback = predictions.set_index("row_id").loc["r1"]
    later_feedback = predictions.set_index("row_id").loc["r2"]
    expected = hashlib.sha256()
    encoded_id = b"r0"
    expected.update(len(encoded_id).to_bytes(8, "big", signed=False))
    expected.update(encoded_id)

    assert first_feedback["feedback_count"] == 0
    assert first_feedback["feedback_ids_sha256"] == hashlib.sha256(b"").hexdigest()
    assert pd.isna(first_feedback["max_feedback_release_at"])
    assert later_feedback["feedback_count"] == 1
    assert later_feedback["feedback_ids_sha256"] == expected.hexdigest()
    assert later_feedback["max_feedback_release_at"] == pd.Timestamp("2025-01-04T00:00:00Z")
