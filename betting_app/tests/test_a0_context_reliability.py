import numpy as np
import pandas as pd
import pytest

from src.analysis.a0_context_reliability import (
    evaluate_context_predictions,
    evaluate_persistence,
    extract_context_features,
)


def _feature_rows(dates, *, ids=None):
    n = len(dates)
    return pd.DataFrame({
        "golgg_match_id": ids or [f"m{i}" for i in range(n)],
        "date": pd.to_datetime(dates),
        "label_release_day": pd.to_datetime(dates) + pd.Timedelta(days=1),
        "p_c0": np.linspace(.55, .85, n),
        "p_a0": np.linspace(.54, .84, n),
        "p_neural": np.linspace(.60, .80, n),
        "p_rating": np.linspace(.50, .70, n),
        "p_mlp": np.linspace(.55, .75, n),
        "y": np.arange(n) % 2,
        "history_stat_coverage": np.resize([.89, .90, np.nan], n),
        "unknown_players": np.resize([1, 0, np.nan], n),
        "history_token_fraction": np.resize([.2, .5, np.nan], n),
        "tournament_name": ["Event A"] * n,
        "competition_tier": ["tier1"] * n,
        "best_of": [3] * n,
        "team1_id": [f"a{i}" for i in range(n)],
        "team2_id": [f"b{i}" for i in range(n)],
    })


def test_extract_context_features_uses_covariate_only_2024_disagreement_cutoff():
    train_dates = pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01"])
    frame = _feature_rows([*train_dates, "2025-03-01", "2026-02-01"])
    frame.loc[:3, ["p_neural", "p_rating", "p_mlp"]] = [
        [.50, .45, .40], [.60, .55, .50], [.80, .70, .60], [.90, .85, .80],
    ]
    frame.loc[4, ["p_neural", "p_rating", "p_mlp"]] = [.95, .40, .40]
    extracted, metadata = extract_context_features(frame)
    threshold = metadata["high_sports_disagreement_threshold"]
    assert threshold == pytest.approx(.1)
    assert extracted.loc[0, "x_low_statistics"] == 1
    assert extracted.loc[1, "x_low_statistics"] == 0
    assert np.isnan(extracted.loc[2, "x_low_statistics"])
    assert extracted.loc[0, "x_novel_roster_support"] == 1
    assert extracted.loc[1, "x_novel_roster_support"] == 0
    assert np.isnan(extracted.loc[2, "x_novel_roster_support"])
    assert extracted.loc[4, "x_high_sports_disagreement"] == 1
    changed = frame.copy()
    changed.loc[4:, "y"] = 1 - changed.loc[4:, "y"]
    changed.loc[4:, ["p_neural", "p_rating", "p_mlp"]] = [.01, .99, .5]
    _, changed_metadata = extract_context_features(changed)
    assert changed_metadata["high_sports_disagreement_threshold"] == threshold


def test_context_flags_are_invariant_to_side_complement_at_disagreement_ties():
    frame = _feature_rows(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    frame["p_neural"] = [.7, .6, .5, .4]
    frame["p_rating"] = [.5, .4, .3, .3]
    frame["p_mlp"] = [.5, .4, .3, .3]
    frame["history_stat_coverage"] = [.5, .95, .5, .95]
    frame["unknown_players"] = [1, 0, 1, 0]
    original, metadata = extract_context_features(frame)
    assert metadata["disagreement_tie_atol"] == pytest.approx(1e-12)
    tied = original.loc[:2, "x_high_sports_disagreement"]
    assert tied.eq(0).all()

    mirrored = frame.copy()
    for column in ("p_c0", "p_a0", "p_neural", "p_rating", "p_mlp"):
        mirrored[column] = 1 - mirrored[column]
    mirrored["y"] = 1 - mirrored["y"]
    mirrored_output, _ = extract_context_features(mirrored)
    for column in ("x_low_statistics", "x_novel_roster_support", "x_high_sports_disagreement"):
        np.testing.assert_array_equal(original[column], mirrored_output[column])



def test_extract_context_features_rejects_infinite_expert_probability_instead_of_marking_unknown():
    frame = _feature_rows(["2024-08-01"])
    frame.loc[0, "p_rating"] = np.inf
    with pytest.raises(ValueError, match="infinite probability"):
        extract_context_features(frame)
def test_extract_context_features_preserves_unknown_as_not_a_risk_state():
    frame = _feature_rows(["2024-08-01"] * 3)
    frame.loc[2, "p_neural"] = np.nan
    extracted, _ = extract_context_features(frame)
    assert np.isnan(extracted.loc[2, "x_low_statistics"])
    assert np.isnan(extracted.loc[2, "x_novel_roster_support"])
    assert np.isnan(extracted.loc[2, "x_high_sports_disagreement"])


def test_evaluate_persistence_is_invariant_to_2026_outcomes():
    frame = _feature_rows(["2024-08-01", "2025-03-01", "2026-02-01"])
    frame, _ = extract_context_features(frame)
    global_predictions = pd.DataFrame({
        "golgg_match_id": frame.loc[:1, "golgg_match_id"],
        "date": frame.loc[:1, "date"],
        "origin": pd.to_datetime(["2024-07-01", "2025-01-01"]),
        "q_global": [.6, .6],
    })
    original = evaluate_persistence(frame, global_predictions)
    changed = frame.copy()
    changed.loc[2, "y"] = 1 - changed.loc[2, "y"]
    perturbed = evaluate_persistence(changed, global_predictions)
    assert perturbed["gate"] == original["gate"]
    assert perturbed["contexts"] == original["contexts"]


def test_evaluate_persistence_accepts_quarter_start_forecast_date_and_rejects_later_origin():
    frame = _feature_rows(["2024-07-01", "2025-01-01"])
    frame, _ = extract_context_features(frame)
    forecasts = pd.DataFrame({
        "golgg_match_id": frame.golgg_match_id,
        "date": frame.date,
        "origin": frame.date,
        "q_global": [.6, .6],
    })
    accepted = evaluate_persistence(frame, forecasts)
    assert accepted["missingness"]["detection"]["global_forecast_missing"] == 0
    later = forecasts.copy()
    later.loc[0, "origin"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="cannot follow"):
        evaluate_persistence(frame, later)

def test_evaluate_persistence_fails_closed_when_each_context_lacks_support():
    frame = _feature_rows(["2024-08-01", "2025-03-01"])
    frame, _ = extract_context_features(frame)
    global_predictions = pd.DataFrame({
        "golgg_match_id": frame.golgg_match_id,
        "date": frame.date,
        "origin": pd.to_datetime(["2024-07-01", "2025-01-01"]),
        "q_global": [.6, .6],
    })
    result = evaluate_persistence(frame, global_predictions)
    assert result["gate"]["pass"] is False
    assert result["gate"]["status"] == "INSUFFICIENT_SUPPORT"
    assert result["gate"]["passing_contexts"] == []
    assert all(result["contexts"][name][period]["confidence_level"]
               == pytest.approx(0.9916666666666667)
               for name in ("low_statistics", "novel_roster_support", "high_sports_disagreement")
               for period in ("detection", "replication"))


def test_evaluate_persistence_requires_same_nonzero_sign_in_detection_and_replication():
    dates = pd.date_range("2024-07-01", periods=12, freq="MS").append(
        pd.date_range("2025-01-01", periods=12, freq="MS"))
    n = len(dates)
    frame = _feature_rows(dates)
    frame["history_stat_coverage"] = np.tile([.5, .95], n // 2)
    frame["unknown_players"] = np.tile([1, 0], n // 2)
    frame["p_c0"] = .8
    frame["y"] = np.tile([0, 1], n // 2)
    frame, _ = extract_context_features(frame, disagreement_threshold=.5)
    frame["y"] = np.where(
        pd.to_datetime(frame.date).dt.year.eq(2024),
        np.tile([0, 1], n // 2),
        np.tile([1, 0], n // 2),
    )
    gp = pd.DataFrame({
        "golgg_match_id": frame.golgg_match_id,
        "date": frame.date,
        "origin": pd.to_datetime(frame.date).dt.to_period("Q").dt.start_time - pd.DateOffset(months=3),
        "q_global": .6,
    })
    result = evaluate_persistence(frame, gp, config={
        "minimum_group_rows_per_year": 2,
        "minimum_group_months_per_year": 2,
        "minimum_risk_and_reference_rows_per_bin": 1,
        "draws": 200,
    })
    assert result["gate"]["pass"] is False
    assert result["gate"]["passing_contexts"] == []
    claim = result["contexts"]["low_statistics"]
    assert claim["detection"]["estimate"] * claim["replication"]["estimate"] < 0
    assert claim["gate_pass"] is False


def test_evaluate_persistence_keeps_full_month_universe_and_fails_zero_bootstrap_denominators():
    dates = pd.to_datetime(["2024-07-02", "2024-08-02", "2025-01-02", "2025-02-02"])
    frame = _feature_rows(dates)
    frame["history_stat_coverage"] = [.5, .95, .5, .95]
    frame["p_c0"] = [.8, .8, .8, .8]
    frame["y"] = [0, 1, 0, 1]
    frame, _ = extract_context_features(frame, disagreement_threshold=.5)
    gp = pd.DataFrame({
        "golgg_match_id": frame.golgg_match_id, "date": frame.date,
        "origin": pd.to_datetime(["2024-07-01", "2024-07-01", "2025-01-01", "2025-01-01"]),
        "q_global": [.7] * 4,
    })
    result = evaluate_persistence(frame, gp, config={
        "minimum_group_rows_per_year": 1,
        "minimum_group_months_per_year": 1,
        "minimum_risk_and_reference_rows_per_bin": 1,
        "draws": 500,
    })
    claim = result["contexts"]["low_statistics"]
    assert claim["detection"]["months"] == 6
    assert claim["replication"]["months"] == 12
    assert claim["detection"]["status"] == "UNDEFINED_REPLICATES"
    assert claim["gate_pass"] is False


def test_evaluate_context_predictions_scores_neutrals_but_excludes_them_from_residual_diagnostics():
    frame = _feature_rows(["2025-08-01", "2026-01-01", "2026-02-01", "2026-03-01"])
    frame["p_c0"] = [.55, .5, .8, .7]
    frame["p_a0"] = [.55, .5, .75, .65]
    frame["p_global"] = [.55, .62, .78, .72]
    frame["p_independent"] = [.55, .61, .79, .71]
    frame["p_context"] = [.55, .5, .75, .70]
    frame["y"] = [1, 1, 0, 1]
    frame["best_of"] = [1, 3, 5, 3]
    frame["competition_tier"] = ["regular"] * 4
    frame.loc[1, "competition_tier"] = "Major"
    frame.loc[1, "tournament_name"] = "Unknown Tournament"
    frame["source_unseen"] = [False, False, True, False]
    result = evaluate_context_predictions(frame, config={
        "draws": 200,
        "expected_cohort_rows": 4,
        "expected_primary_rows": 3,
        "expected_warmup_rows": 1,
    })
    scores = result["scores"]["cohort"]["context"]
    assert scores["n"] == 4
    expected_brier = np.mean((frame.p_context.to_numpy() - frame.y.to_numpy()) ** 2)
    expected_ll = np.mean(-frame.y * np.log(frame.p_context) -
                          (1 - frame.y) * np.log1p(-frame.p_context))
    assert scores["brier"] == pytest.approx(expected_brier)
    assert scores["log_loss"] == pytest.approx(expected_ll)
    primary_scores = result["scores"]["primary_2026"]["context"]
    assert primary_scores["n"] == 3
    primary = frame.iloc[1:]
    assert primary_scores["brier"] == pytest.approx(
        np.mean((primary.p_context.to_numpy() - primary.y.to_numpy()) ** 2)
    )
    primary_ll = np.mean(-primary.y * np.log(primary.p_context) -
                         (1 - primary.y) * np.log1p(-primary.p_context))
    assert primary_scores["log_loss"] == pytest.approx(primary_ll)
    assert result["neutral_rows"] == 1
    assert result["residual_diagnostics"]["rows"] == 2
    assert result["noninferiority_guards"]["major_vs_c0"]["n"] == 1
    assert result["noninferiority_guards"]["major_vs_annual_a0"]["n"] == 1
