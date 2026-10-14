import json

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_adaptive_series import evaluate


def _predictions():
    return pd.DataFrame(
        {
            "golgg_match_id": range(8),
            "date": pd.date_range("2024-01-01", periods=8, freq="MS"),
            "y_true": [1, 0, 1, 0, 0, 1, 0, 1],
            "best_of": [1, 3, 5, 3, 1, 3, 5, 3],
            "tournament": ["test"] * 8,
            "competition_tier": ["major", "international", "minor", "academy"] * 2,
            "roster_min_prior_series": [12, 3, np.nan, 10] * 2,
            "player_consensus": [0.5, 0.6, 0.8, np.nan] * 2,
            "team_consensus": [0.5, 0.5, 0.6, 0.5] * 2,
            "market": [0.25, 0.75, np.nan, 0.5, 0.25, 0.75, np.nan, 0.5],
            "odds_a": [4.0, 1.3, np.nan, 1.0, 4.0, 1.3, np.nan, 1.9],
            "odds_b": [1.3, 4.0, np.nan, 2.0, 1.3, 4.0, np.nan, 1.9],
            "p__outcome": [0.6, 0.3, 0.7, 0.4, 0.4, 0.6, 0.2, 0.6],
            "p__stored_exp081_diagnostic": [
                0.65,
                0.25,
                0.75,
                0.35,
                0.35,
                0.65,
                0.25,
                0.55,
            ],
            "p__candidate": [0.4, 0.6, 0.7, 0.4, 0.4, 0.6, 0.2, 0.6],
            "p__another_candidate": [0.55, 0.35, 0.6, 0.45, 0.45, 0.55, 0.3, 0.65],
        }
    )


def test_every_model_retains_missing_odds_and_receives_complete_diagnostics(tmp_path):
    frame = _predictions()
    summary = evaluate(frame, tmp_path / "evaluation")
    models = set(frame.filter(like="p__").columns)
    assert set(summary["aggregate"]) == models
    assert all(row["n"] == len(frame) for row in summary["aggregate"].values())
    metrics = pd.read_csv(summary["paths"]["metrics"])
    for model in models:
        rows = metrics[metrics.model == model]
        assert rows.loc[rows.slice == "odds_missing_or_invalid", "n"].item() == 3
        assert rows.loc[rows.slice == "tier1", "n"].item() == 4
        assert rows.loc[rows.slice == "bo5", "n"].item() == 2
        assert rows.loc[rows.slice == "roster_prior_missing", "n"].item() == 2
        assert summary["market_diagnostics"][model]["n"] == 6
        curve = summary["market_diagnostics"][model]["hybrid_curve"]
        assert curve[0]["log_loss"] == pytest.approx(
            summary["market_diagnostics"][model]["market"]["log_loss"]
        )
        assert curve[-1]["log_loss"] == pytest.approx(
            summary["market_diagnostics"][model]["model"]["log_loss"]
        )
    reliability = pd.read_csv(summary["paths"]["reliability"])
    assert set(reliability.model) == models
    for _, rows in reliability.groupby(["model", "orientation"]):
        assert rows.n.sum() == len(frame)
        assert (rows.n == 0).any()
    for model in models - {"p__outcome"}:
        for slice_name, expected_n in (("all", 8), ("tier1", 4)):
            paired = summary["paired"][model][slice_name]
            assert paired["n"] == expected_n
            assert paired["log_loss"]["resamples"] >= 5000
            assert paired["brier"]["resamples"] >= 5000
    assert summary["promotion"] is False
    json.loads(
        (tmp_path / "evaluation" / "evaluation.json").read_text(),
        parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"),
    )


def test_underdog_probabilities_and_outcomes_follow_the_same_selected_side(tmp_path):
    summary = evaluate(_predictions(), tmp_path / "evaluation")
    metrics = pd.read_csv(summary["paths"]["metrics"])
    row = metrics[
        (metrics.model == "p__candidate")
        & (metrics.orientation == "market_underdog")
        & (metrics.slice == "odds_3.50_5.00")
    ].iloc[0]
    assert row["n"] == 4
    assert row.selected_side_a_n == row.selected_side_b_n == 2
    assert row.mean_p == pytest.approx(0.4)
    assert row.observed_rate == pytest.approx(0.5)
    diagnostic = summary["underdog_diagnostics"]["p__candidate"]["selected_35_45"]
    assert diagnostic["n"] == 4
    assert diagnostic["calibration_gap"] == pytest.approx(-0.1)
    assert diagnostic["enough_exact_odds"] is False
    assert summary["underdog_diagnostics"]["p__outcome"]["p_gt_05"]["n"] == 2


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_candidate_probabilities_reject_the_entire_report(tmp_path, invalid):
    frame = _predictions()
    frame.loc[0, "p__another_candidate"] = invalid
    with pytest.raises(ValueError):
        evaluate(frame, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


def test_evaluation_never_reuses_an_existing_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "metrics.csv"
    sentinel.write_bytes(b"prior experiment\n")
    with pytest.raises(FileExistsError):
        evaluate(_predictions(), output)
    assert sentinel.read_bytes() == b"prior experiment\n"
    assert not (output / "evaluation.json").exists()


def test_missing_diagnostic_cohorts_remain_explicit_and_finite(tmp_path):
    frame = _predictions()
    frame[["market", "odds_a", "odds_b", "roster_min_prior_series"]] = np.nan
    frame["competition_tier"] = "minor"
    summary = evaluate(frame, tmp_path / "evaluation")
    for model in summary["prediction_columns"]:
        diagnostic = summary["market_diagnostics"][model]
        assert diagnostic["n"] == 0
        assert diagnostic["log_loss_delta"] is None
        assert diagnostic["pearson"] is None
        assert diagnostic["spearman"] is None
        assert diagnostic["mad"] is None
        assert summary["underdog_diagnostics"][model]["selected_35_45"]["n"] == 0
    for paired in summary["paired"].values():
        assert paired["tier1"]["n"] == 0
        assert paired["tier1"]["log_loss"]["ci95_high"] is None
    json.dumps(summary, allow_nan=False)


def test_qualification_enforces_absolute_calibration_and_tier1_limits():
    from scripts.evaluate_adaptive_series import _qualification

    scores = {
        "n": 10000,
        "calibration_slope": 0.85,
        "ece10": 0.03,
        "brier_reliability_binned": 0.009,
    }
    aggregate = {
        "p__outcome": {**scores, "calibration_slope": 0.5, "ece10": 0.1},
        "p__candidate": scores,
    }
    comparison = {
        "log_loss": {"mean": 0.002, "ci95_high": 0.003},
        "brier": {"mean": 0.0, "ci95_high": 0.001},
    }
    underdogs = {name: {"p_gt_05": {}, "price_350_500_gate": {}} for name in aggregate}
    paired = {"p__candidate": {"all": comparison, "tier1": comparison}}
    result = _qualification(aggregate, paired, underdogs)["p__candidate"]
    assert result["calibration_slope"]["within_required_range"] is True
    assert result["ece10"]["absolute_limit_pass"] is True
    assert result["tier1_degradation"]["log_loss"]["promotion_limit_pass"] is True
    aggregate["p__candidate"] = {**scores, "calibration_slope": 0.84, "ece10": 0.031}
    failed = _qualification(aggregate, paired, underdogs)["p__candidate"]
    assert failed["calibration_slope"]["within_required_range"] is False
    assert failed["ece10"]["absolute_limit_pass"] is False


def test_underdog_gate_includes_price_five_and_requires_market_confirmation(tmp_path):
    frame = _predictions()
    frame.loc[0, ["odds_a", "p__candidate", "market"]] = [5.0, 0.6, 0.3]
    frame.loc[1, ["odds_b", "p__candidate", "market"]] = [5.0, 0.4, 0.4]
    summary = evaluate(frame, tmp_path / "evaluation")
    gate = summary["underdog_diagnostics"]["p__candidate"]["price_350_500_gate"]
    assert gate["p_gt_05_n"] == 2
    assert gate["unconfirmed_n"] == 1
    assert gate["pass_on_observed_prices"] is False
    diagnostic = summary["underdog_diagnostics"]["p__candidate"][
        "p_gt_05_price_350_500"
    ]
    assert diagnostic["n"] == 2
    assert diagnostic["mean_p"] == pytest.approx(0.6)
