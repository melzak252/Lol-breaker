"""Coverage in diagnostic reports distinguishes forecasts from stage eligibility."""
import json

import numpy as np
import pandas as pd

from src.analysis.a0_stage_diagnostics import MAP_STAGE, _fixed_selection_summary


def test_raw_map_report_does_not_label_series_exclusions_as_missing_forecasts():
    universe = pd.DataFrame({
        "origin_id": ["bo1", "bo3", "missing_bo1", "missing_bo3"],
        "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "edition_proxy": ["one", "two", "three", "four"],
    })
    rows = universe.assign(
        stage=MAP_STAGE,
        stage_semantics="raw map probability; eligible only on BO1",
        native_available=[True, True, False, False],
        stage_available=[True, False, False, False],
        p_selected=[0.6, np.nan, np.nan, np.nan],
        expected_ev_net_per_unit=[0.056, np.nan, np.nan, np.nan],
        realized_ev_net_per_unit=[0.76, np.nan, np.nan, np.nan],
        ev_error_expected_minus_realized=[-0.704, np.nan, np.nan, np.nan],
        won=[True, False, False, True],
        missing_reason=[None, None, "no_prior_context", "no_prior_context"],
    )

    report = _fixed_selection_summary(universe, rows, draws=128, seed=20260930).iloc[0]

    assert json.loads(report.missing_reasons_json) == {"no_prior_context": 2}
    assert report.n_inventory == 4
    assert report.n_scored == 1
    assert report.n_missing_forecast == 2
    assert report.n_stage_ineligible_available_forecast == 1
    assert report.n_missing_or_ineligible == 3
