import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.a0_correction_market import evaluate_market
from src.models.a0_correction_experiment import sha256


def test_numeric_metadata_and_reversed_prices_preserve_fixed_choice_payoffs(tmp_path):
    dates = pd.Series(["2026-01-02", "2026-01-03", "2026-02-02", "2026-02-03"])
    day = pd.to_datetime(dates)
    history = (day - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    frame = pd.DataFrame({
        "golgg_match_id": ["1", "2", "3", "4"], "team1_id": "10", "team2_id": "20",
        "date": dates, "best_of": 3, "y": [1, 0, 1, 0],
        "feature_source_max_day": history, "head_origin": "2026-01-01",
        "p_a0": .8, "p_calibration": .75, "p_correction": .7,
    })
    quotes = frame.drop(columns=["p_calibration", "p_correction", "head_origin"]).copy()
    quotes["fixture_id"] = frame.golgg_match_id
    quotes["best_of"] = quotes.best_of.astype(float)
    quotes["y"] = quotes.y.astype(float)
    quotes["provider_team1_is_a"] = [True, False, True, False]
    quotes["odds_1"] = [2., 3., 2., 3.]
    quotes["odds_2"] = [3., 2., 3., 2.]
    quotes["p_book"] = .6
    quotes["bookmaker"] = "pinnacle"
    quotes["horizon"] = "T-1h"
    quotes["strict_prior_day"] = True
    quotes["max_side_age_hours"] = 1.
    quotes["evaluation_cutoff_at"] = dates + "T14:00:00Z"
    quotes["start_cutoff_at"] = dates + "T15:00:00Z"
    for key in ("quote_at", "outcome_1_quote_at", "outcome_2_quote_at"):
        quotes[key] = dates + "T13:00:00Z"
    quotes["valid_until_exclusive_at"] = dates + "T15:00:00Z"
    quotes["tournament_name"] = "scenario"
    source, forecasts = tmp_path / "quotes.parquet", tmp_path / "forecasts.parquet"
    quotes.to_parquet(source, index=False)
    frame.to_parquet(forecasts, index=False)
    config = {
        "source": str(source), "source_sha256": sha256(source), "model_columns": ["p_a0", "p_calibration", "p_correction"],
        "candidate_column": "p_correction", "comparison_columns": ["p_a0", "p_calibration", "p_market"],
        "bookmaker": "pinnacle", "horizon": "T-1h", "maximum_side_age_hours": 24.,
        "tax": .12, "bands": ["all_EV_gt_5pct", "EV_5_to_10pct"],
        "resamples": 5000, "seed": 20261001, "tolerance": .02, "limits": ["synthetic diagnostic"],
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config))
    result = evaluate_market(protocol, forecasts, tmp_path / "output")
    fixed = {r["model"]: r for r in result["selected_diagnostics"]
             if r["policy"] == "fixed_a0" and r["band"] == "all_EV_gt_5pct"}
    assert fixed["p_a0"]["mean_realized_unit_net_return"] == pytest.approx(-.12)
    assert fixed["p_correction"]["mean_realized_unit_net_return"] == pytest.approx(-.12)
    assert fixed["p_a0"]["overstatement"]["estimate"] == pytest.approx(.528)
    assert fixed["p_correction"]["overstatement"]["estimate"] == pytest.approx(.352)
    saved = pd.read_parquet(tmp_path / "output" / "selected.parquet")
    assert set(saved.side) == {"a"}
    np.testing.assert_array_equal(saved.odds, np.full(len(saved), 2.))


def test_c0_candidate_contrasts_do_not_replace_fixed_a0_selection(tmp_path):
    dates = pd.Series(["2026-01-02", "2026-01-03", "2026-02-02", "2026-02-03"])
    frame = pd.DataFrame({
        "golgg_match_id": ["1", "2", "3", "4"], "team1_id": "10", "team2_id": "20",
        "date": dates, "best_of": 3, "y": [1, 0, 1, 0],
        "feature_source_max_day": (pd.to_datetime(dates) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
        "head_origin": "2026-01-01", "p_a0": [.65, .65, .601, .5],
        "p_c0": [.3, .3, .9, .7], "p_c0_calibration": [.4, .4, .8, .6],
        "p_connector03": [.61, .61, .55, .51], "p_c0_connector": [.2, .2, .8, .7],
    })
    columns = ["p_a0", "p_c0", "p_c0_calibration", "p_connector03", "p_c0_connector"]
    quotes = frame[["golgg_match_id", "team1_id", "team2_id", "date", "best_of", "y", "p_a0"]].copy()
    quotes["fixture_id"] = frame.golgg_match_id
    quotes["provider_team1_is_a"] = [True, False, True, False]
    quotes["odds_1"] = [2., 3., 2., 3.]
    quotes["odds_2"] = [3., 2., 3., 2.]
    quotes["p_book"] = .6
    quotes["bookmaker"] = "pinnacle"
    quotes["horizon"] = "T-1h"
    quotes["strict_prior_day"] = True
    quotes["max_side_age_hours"] = 1.
    quotes["evaluation_cutoff_at"] = dates + "T14:00:00Z"
    quotes["start_cutoff_at"] = dates + "T15:00:00Z"
    for key in ("quote_at", "outcome_1_quote_at", "outcome_2_quote_at"):
        quotes[key] = dates + "T13:00:00Z"
    quotes["valid_until_exclusive_at"] = dates + "T15:00:00Z"
    quotes["tournament_name"] = "scenario"
    source, forecasts = tmp_path / "quotes.parquet", tmp_path / "forecasts.parquet"
    quotes.to_parquet(source, index=False)
    frame.to_parquet(forecasts, index=False)
    config = {
        "source": str(source), "source_sha256": sha256(source), "model_columns": columns,
        "candidate_column": "p_c0_connector",
        "comparison_columns": ["p_a0", "p_c0", "p_c0_calibration", "p_connector03", "p_market"],
        "bookmaker": "pinnacle", "horizon": "T-1h", "maximum_side_age_hours": 24.,
        "tax": .12, "bands": ["all_EV_gt_5pct", "EV_5_to_10pct"],
        "resamples": 5000, "seed": 20261002, "tolerance": .02, "limits": ["synthetic diagnostic"],
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config))
    result = evaluate_market(protocol, forecasts, tmp_path / "output")
    selected = pd.read_parquet(tmp_path / "output" / "selected.parquet")
    candidate = selected.loc[(selected.model == "p_c0_connector") & (selected.band == "all_EV_gt_5pct")]
    fixed = candidate.loc[candidate.policy == "fixed_a0"].sort_values("golgg_match_id")
    own = candidate.loc[candidate.policy == "own"].sort_values("golgg_match_id")
    assert fixed.side.tolist() == ["a", "a", "a", "b"]
    assert own.side.tolist() == ["b", "b", "a", "a"]
    np.testing.assert_allclose(fixed.odds, [2., 2., 2., 3.])
    assert fixed.realized.mean() == pytest.approx(.54)
    assert own.realized.mean() == pytest.approx(.1)
    assert fixed.error.mean() == pytest.approx(-.814)
    near = selected.loc[(selected.policy == "fixed_a0") & (selected.model == "p_c0_connector") &
                        (selected.band == "EV_5_to_10pct")]
    assert near.golgg_match_id.tolist() == ["3"]
    y = frame.y.to_numpy()
    loss = lambda p: -np.where(y, np.log(p), np.log1p(-p))
    expected = (loss(frame.p_c0_connector) - loss(frame.p_c0)).mean()
    assert result["paired_log_loss"]["p_c0"]["delta"] == pytest.approx(expected)
