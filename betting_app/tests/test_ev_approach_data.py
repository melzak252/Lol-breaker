"""Contracts for decision-time opportunity provenance and released OOF rows."""

import pandas as pd
import pytest

from src.analysis import ev_approach_data as data


UTC = "2026-06-01T12:00:00Z"


def opportunity(**overrides):
    row = dict(
        row_id="m1:48h",
        golgg_match_id="m1",
        cohort="common",
        horizon="48h",
        date="2026-06-03",
        decision_at=pd.Timestamp(UTC),
        start_at=pd.Timestamp("2026-06-03T12:00:00Z"),
        label_available_at=pd.Timestamp("2026-06-04T12:00:00Z"),
        y=1,
        p_s=0.61,
        q=0.57,
        odds_a=1.9,
        odds_b=2.1,
        bookmaker_a="book-a",
        bookmaker_b="book-b",
        quote_id_a="qa1",
        quote_id_b="qb1",
        quote_at_a=pd.Timestamp("2026-06-01T11:58:00Z"),
        quote_at_b=pd.Timestamp("2026-06-01T11:59:00Z"),
        best_of=3,
        league="L",
        information_status="recorded",
        strict_prior_day=True,
        is_common=True,
        feature_available_at=pd.Timestamp("2026-06-01T11:00:00Z"),
    )
    row.update(overrides)
    return row


def prediction(**overrides):
    row = dict(
        row_id="m1:48h",
        golgg_match_id="m1",
        cohort="common",
        horizon="48h",
        decision_at=pd.Timestamp("2026-05-28T10:00:00Z"),
        label_available_at=pd.Timestamp("2026-05-30T12:00:00Z"),
        p_s=0.61,
        sigma_z=0.12,
        prediction_at=pd.Timestamp("2026-05-28T10:00:00Z"),
        fit_origin=pd.Timestamp("2026-05-27T00:00:00Z"),
        is_oof=True,
        recipe_status="COMPLETE-RECIPE",
        prediction_role="calibration",
    )
    row.update(overrides)
    return row


def test_validate_opportunities_rejects_duplicate_row_identity_within_horizon():
    frame = pd.DataFrame([opportunity(), opportunity()])

    with pytest.raises(ValueError, match="duplicate|unique|row_id"):
        data.validate_opportunities(frame)


def test_validate_opportunities_rejects_repeated_fixture_in_same_cohort_horizon():
    frame = pd.DataFrame([
        opportunity(row_id="first"),
        opportunity(row_id="second"),
    ])

    with pytest.raises(ValueError, match="duplicate|fixture|match|cohort|horizon"):
        data.validate_opportunities(frame)


def test_validate_opportunities_rejects_side_label_or_probability_corruption():
    valid = data.validate_opportunities(pd.DataFrame([opportunity()]))
    assert valid.iloc[0].y == 1

    for corrupted in (
        opportunity(y=2),
        opportunity(p_s=1.01),
        opportunity(q=-0.01),
    ):
        with pytest.raises(ValueError, match=r"side|outcome|probab|\by\b|\bp_s\b|\bq\b"):
            data.validate_opportunities(pd.DataFrame([corrupted]))


def test_validate_opportunities_rejects_quote_or_source_observed_after_decision():
    for override in (
        {"quote_at_b": pd.Timestamp("2026-06-01T12:00:01Z")},
        {"feature_available_at": pd.Timestamp("2026-06-01T12:00:01Z")},
    ):
        with pytest.raises(ValueError, match="quote|source|decision|future|after"):
            data.validate_opportunities(pd.DataFrame([opportunity(**override)]))


def test_validate_opportunities_does_not_promote_scenario_time_to_recorded_quote():
    frame = pd.DataFrame([opportunity(
        information_status="historical_scenario",
        quote_at_a=pd.NaT,
        quote_at_b=pd.NaT,
        scenario_quote_at_a=pd.Timestamp("2026-06-01T11:58:00Z"),
        scenario_quote_at_b=pd.Timestamp("2026-06-01T11:59:00Z"),
        timestamp_assumption="daily_prior_release",
    )])

    validated = data.validate_opportunities(frame)

    assert pd.isna(validated.iloc[0].quote_at_a)
    assert pd.isna(validated.iloc[0].quote_at_b)
    assert validated.iloc[0].information_status == "historical_scenario"


@pytest.mark.parametrize("column", ["decision_at", "start_at", "label_available_at", "quote_at_a", "quote_at_b", "feature_available_at"])
def test_validate_opportunities_requires_timezone_aware_known_timestamps(column):
    row = opportunity()
    row[column] = pd.Timestamp("2026-06-01 11:00:00")

    with pytest.raises(ValueError, match="timezone|aware|timestamp|time"):
        data.validate_opportunities(pd.DataFrame([row]))


def test_calibration_rows_before_keeps_only_released_strict_past_complete_recipe_oof():
    origin = pd.Timestamp("2026-06-01T12:00:00Z")
    frame = pd.DataFrame([
        prediction(row_id="released-before-origin", golgg_match_id="m2",
                   label_available_at=pd.Timestamp("2026-05-27T00:00:01Z")),
        prediction(row_id="eligible"),
        prediction(row_id="label-at-origin", label_available_at=origin),
        prediction(row_id="label-near-origin", golgg_match_id="m3",
                   label_available_at=pd.Timestamp("2026-05-31T23:59:59Z")),
        prediction(row_id="label-after-origin", label_available_at=pd.Timestamp("2026-06-01T12:00:01Z")),
        prediction(row_id="prediction-at-origin", prediction_at=origin),
        prediction(row_id="fit-at-origin", fit_origin=origin),
        prediction(row_id="not-oof", is_oof=False),
        prediction(row_id="partial-recipe", recipe_status="PARTIAL-RECIPE"),
        prediction(row_id="wrong-recipe", recipe_status="OTHER-COMPLETE-RECIPE"),
    ])

    selected = data.calibration_rows_before(frame, origin, require_oof=True)

    assert selected.row_id.tolist() == ["released-before-origin", "eligible", "label-near-origin"]


def test_calibration_rows_before_never_uses_future_scenario_timestamps_as_evidence():
    origin = pd.Timestamp("2026-06-01T12:00:00Z")
    frame = pd.DataFrame([prediction(
        row_id="scenario-only",
        is_oof=False,
        recipe_status="IN-SAMPLE-CAL",
        prediction_at=pd.NaT,
        fit_origin=pd.NaT,
        label_available_at=pd.NaT,
        information_status="historical_scenario",
        scenario_prediction_at=pd.Timestamp("2026-06-01T10:00:00Z"),
        scenario_label_available_at=pd.Timestamp("2026-05-31T00:00:00Z"),
    )])

    selected = data.calibration_rows_before(frame, origin, require_oof=True)

    assert selected.empty


def test_historical_scenario_oof_uses_explicit_assumptions_without_backdating_actual_times():
    origin = pd.Timestamp("2026-06-01T12:00:00Z")
    frame = pd.DataFrame([prediction(
        row_id="scenario-oof",
        information_status="historical_scenario",
        decision_at=pd.Timestamp("2026-05-28T10:00:00Z"),
        prediction_at=pd.NaT,
        fit_origin=pd.Timestamp("2026-05-27T00:00:00Z"),
        label_available_at=pd.NaT,
        scenario_prediction_at=pd.Timestamp("2026-05-28T10:00:00Z"),
        scenario_label_available_at=pd.Timestamp("2026-05-30T12:00:00Z"),
        scenario_timestamp_assumption="daily_prior_release",
    )])

    selected = data.calibration_rows_before(frame, origin, require_oof=True)

    assert selected.row_id.tolist() == ["scenario-oof"]
    assert pd.isna(selected.iloc[0].prediction_at)
    assert selected.iloc[0].information_status == "historical_scenario"


def test_in_sample_native_cal_is_excluded_by_default_and_opt_in_is_explicit():
    origin = pd.Timestamp("2026-06-01T12:00:00Z")
    frame = pd.DataFrame([
        prediction(row_id="native-cal-for-calibration", is_oof=False,
                   recipe_status="IN-SAMPLE-CAL", prediction_role="calibration"),
        prediction(row_id="oof"),
        prediction(row_id="native-cal", is_oof=False, recipe_status="IN-SAMPLE-CAL",
                   prediction_role="sigma_reference"),
    ])

    default_rows = data.calibration_rows_before(frame, origin)
    opt_in_rows = data.calibration_rows_before(frame, origin, require_oof=False)
    assert "native-cal-for-calibration" not in set(opt_in_rows.row_id)
    assert default_rows.row_id.tolist() == ["oof"]
    assert set(opt_in_rows.row_id) == {"oof", "native-cal"}
    assert opt_in_rows.set_index("row_id").loc["native-cal", "prediction_role"] == "sigma_reference"


def test_archival_source_row_keeps_distinct_books_but_rejects_conflicting_quotes():
    canonical = pd.DataFrame([{"golgg_match_id": "1", "team1_id": "A", "team2_id": "B"}])
    common = {
        "golgg_match_id": "1", "source_row": 42, "timing": "open",
        "team1_id": "A", "team2_id": "B",
    }
    alpha = {**common, "bookmaker": "alpha", "odds_a": 2.0, "odds_b": 2.0, "market": 0.5}
    beta = {**common, "bookmaker": "beta", "odds_a": 3.0, "odds_b": 1.5, "market": 1.0 / 3.0}
    quotes = pd.DataFrame([alpha, beta, alpha])

    collapsed, _ = data._collapse_archive_quotes(quotes, canonical)

    assert collapsed.set_index("bookmaker")["q_book"].to_dict() == pytest.approx({
        "alpha": 0.5, "beta": 1.0 / 3.0,
    })
    assert collapsed["q_book"].mean() == pytest.approx(5.0 / 12.0)

    conflicting = pd.concat([quotes, pd.DataFrame([{**beta, "odds_a": 4.0}])], ignore_index=True)
    with pytest.raises(ValueError):
        data._collapse_archive_quotes(conflicting, canonical)


def test_companion_quote_cutoff_must_agree_with_panel_horizon():
    panel = pd.DataFrame([{
        "canonical_match_id": "1", "golgg_match_id": "2", "horizon": "24h",
        "cutoff": UTC, "start_dt": "2026-06-02T12:00:00Z",
    }])
    quotes = pd.DataFrame([{
        "canonical_match_id": "1", "horizon": "24h", "bookmaker": "alpha",
        "quote_id_string": "42", "scraped_at": "2026-06-01T11:59:00Z",
        "cutoff": UTC, "odds_team1": 2.5, "odds_team2": 1.5,
        "p_market_quote": 0.375,
    }])

    selected, _ = data._production_quote_ledger(quotes, panel)

    assert selected["p_market_quote"].mean() == pytest.approx(0.375)
    quotes["cutoff"] = "2026-06-01T13:00:00Z"
    with pytest.raises(ValueError):
        data._production_quote_ledger(quotes, panel)
