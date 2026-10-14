"""Behavioral contracts for chronological approach replay and paired comparisons."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.ev_approach_evaluation import (
    chronological_head_predictions,
    paired_monthly_comparison,
)
from src.analysis.ev_approach_policies import select_policy_bets, simulate_portfolio
from src.models.ev_approach_models import APPROACHES


def _replay_frame():
    # Month-boundary observations across a year transition make both decision
    # time and outcome-release time independently observable.
    dates = pd.date_range("2023-10-15", "2024-03-15", freq="15D", tz="UTC")
    n = len(dates)
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(n)],
            "golgg_match_id": [f"m{i}" for i in range(n)],
            "cohort": "synthetic",
            "horizon": "OPEN",
            "date": dates,
            "decision_at": dates + pd.Timedelta(hours=1),
            "start_at": dates + pd.Timedelta(days=1),
            "label_available_at": dates + pd.Timedelta(days=2),
            "prediction_at": dates,
            "fit_origin": dates - pd.Timedelta(days=30),
            "is_oof": True,
            "recipe_status": "COMPLETE-RECIPE",
            "y": np.arange(n) % 2,
            "p_s": np.linspace(0.3, 0.7, n),
            "q": np.linspace(0.45, 0.55, n),
            "odds_a": 2.0,
            "odds_b": 2.0,
            "bookmaker_a": "book-a",
            "bookmaker_b": "book-b",
            "quote_id_a": [f"qa{i}" for i in range(n)],
            "quote_id_b": [f"qb{i}" for i in range(n)],
            "quote_at_a": dates,
            "quote_at_b": dates,
            "best_of": 3,
            "league": "test",
            "information_status": "historical_scenario",
            "scenario_timestamp_assumption": "synthetic-test-times",
            "strict_prior_day": True,
            "is_common": True,
            "model_year": dates.year,
        }
    )


def test_chronological_head_predictions_respect_month_year_and_label_release_origins():
    frame = _replay_frame()
    predictions, receipts = chronological_head_predictions(
        frame, methods=list(APPROACHES), lambdas=(0.1,), min_train=2
    )

    assert set(predictions.row_id) == set(frame.row_id)
    assert set(predictions.method) == set(APPROACHES)
    assert predictions.p.dropna().between(0, 1, inclusive="neither").all()
    folds = {(pd.Timestamp(receipt["fit_origin"]), receipt["method"]): receipt for receipt in receipts}
    assert any(origin == pd.Timestamp("2024-01-01", tz="UTC") for origin, _ in folds)
    assert any(origin == pd.Timestamp("2024-02-01", tz="UTC") for origin, _ in folds)

    # Actual fitted partitions and hyperparameter-selection rows are distinct:
    # posterior CAL stays disjoint from TRAIN, although selection may be refit.
    indexed = frame.set_index("row_id")
    for (origin, _), receipt in folds.items():
        assert set(receipt["train_ids"]).isdisjoint(receipt["calibration_ids"])
        for membership_key in ("train_ids", "calibration_ids", "inner_validation_ids"):
            member_ids = receipt[membership_key]
            if member_ids:
                members = indexed.loc[member_ids]
                assert (members.decision_at < origin).all()
                assert (members.label_available_at < origin).all()
        train = indexed.loc[receipt["train_ids"]]
        cal = indexed.loc[receipt["calibration_ids"]]
        if not train.empty:
            assert pd.Timestamp(receipt["max_train_label_available_at"]) == train.label_available_at.max()
        if not cal.empty:
            assert pd.Timestamp(receipt["max_calibration_label_available_at"]) == cal.label_available_at.max()

    # Each prediction belongs to the month of its decision, including the
    # December/January transition; it is not assigned to a neighboring year.
    jan_ids = set(frame.loc[frame.decision_at.dt.year.eq(2024), "row_id"])
    jan_predictions = predictions[predictions.row_id.isin(jan_ids)]
    assert pd.to_datetime(jan_predictions.fit_origin, utc=True).dt.year.eq(2024).all()


def test_chronological_head_predictions_record_warmup_and_are_future_label_and_close_invariant():
    frame = _replay_frame()
    warmup, receipts = chronological_head_predictions(
        frame, methods=list(APPROACHES), lambdas=(0.1,), min_train=100
    )
    assert set(zip(warmup.cohort, warmup.horizon, warmup.row_id)) == set(
        zip(frame.cohort, frame.horizon, frame.row_id)
    )
    assert warmup.p.isna().all()
    assert warmup.loc[warmup.method.eq("sports_cal"), "missing_reason"].eq("WARMUP").all()
    early = pd.Timestamp("2024-01-01", tz="UTC")
    early_receipts = [receipt for receipt in receipts if pd.Timestamp(receipt["fit_origin"]) == early]
    assert early_receipts
    assert all(receipt["warmup_required"] == 100 for receipt in early_receipts)
    assert all(len(receipt["train_ids"]) < 100 for receipt in early_receipts)
    assert all(receipt["warmup_covered"] is False for receipt in early_receipts)

    # With enough prior rows, January forecasts are populated; mutating
    # post-origin labels and later CLOSE information cannot alter them.
    baseline, _ = chronological_head_predictions(
        frame, methods=list(APPROACHES), lambdas=(0.1,), min_train=2
    )
    changed = frame.copy()
    future = changed.decision_at.ge(early)
    changed.loc[future, "y"] = 1 - changed.loc[future, "y"]
    changed["close_q"] = np.where(future, 0.99, 0.01)
    changed["close_odds_a"] = np.where(future, 20.0, 1.01)
    changed["close_odds_b"] = np.where(future, 1.01, 20.0)
    changed["close_available_at"] = pd.Timestamp("2025-01-01", tz="UTC")
    altered, _ = chronological_head_predictions(
        changed, methods=list(APPROACHES), lambdas=(0.1,), min_train=2
    )
    left = baseline[pd.to_datetime(baseline.fit_origin, utc=True) == early].sort_values(["method", "row_id"])
    right = altered[pd.to_datetime(altered.fit_origin, utc=True) == early].sort_values(["method", "row_id"])
    assert left.row_id.tolist() == right.row_id.tolist()
    assert left.p.notna().any()
    np.testing.assert_allclose(left.p, right.p, equal_nan=True)


def test_chronological_replay_requires_native_oof_only_for_sports_inputs():
    frame = _replay_frame()
    frame.loc[frame.row_id.eq("r0"), "is_oof"] = False
    frame.loc[frame.row_id.eq("r1"), "recipe_status"] = "IN-SAMPLE-CAL"
    frame.loc[frame.row_id.eq("r2"), "prediction_at"] = pd.Timestamp("2024-01-02", tz="UTC")
    frame.loc[frame.row_id.eq("r4"), "p_s"] = np.nan

    predictions, receipts = chronological_head_predictions(
        frame, methods=["sports_cal", "market_cal"], lambdas=(0.1,), min_train=2
    )
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    by_method = {
        receipt["method"]: receipt for receipt in receipts
        if pd.Timestamp(receipt["fit_origin"]) == origin
    }
    sports = by_method["sports_cal"]
    market = by_method["market_cal"]

    assert sports["outer_train_candidate_ids"] == ["r3", "r5"]
    assert sports["train_ids"] == []
    at_origin = predictions.loc[pd.to_datetime(predictions.fit_origin, utc=True).eq(origin)]
    assert at_origin.loc[at_origin.method.eq("sports_cal"), "p"].isna().all()
    assert at_origin.loc[at_origin.method.eq("market_cal"), "p"].notna().all()
    assert sports["native_oof_required"] is True
    assert sports["native_oof_excluded_train_ids"] == ["r0", "r1", "r2"]
    assert sports["native_oof_exclusion_reason_counts"] == {
        "NOT_OOF": 1,
        "INCOMPLETE_RECIPE": 1,
        "PREDICTION_TIME_NOT_STRICTLY_BEFORE_ORIGIN": 1,
    }
    assert sports["inner_train_native_oof_excluded_ids"]
    assert "r2" in sports["inner_validation_native_oof_excluded_ids"]
    assert sports["inner_train_native_oof_exclusion_reason_counts"] == {
        "NOT_OOF": 1,
        "INCOMPLETE_RECIPE": 1,
        "PREDICTION_TIME_NOT_STRICTLY_BEFORE_ORIGIN": 1,
    }
    assert sports["inner_validation_native_oof_exclusion_reason_counts"] == {
        "NOT_OOF": 1,
        "INCOMPLETE_RECIPE": 1,
        "PREDICTION_TIME_NOT_STRICTLY_BEFORE_ORIGIN": 1,
    }
    assert sports["source_recipe_provenance"] == "COMPLETE-RECIPE-OOF"
    assert market["train_ids"] == ["r0", "r1", "r2", "r3", "r4", "r5"]
    assert market["native_oof_required"] is False
    assert market["native_oof_excluded_train_ids"] == []
    assert market["source_recipe_provenance"] == "NOT_REQUIRED_MARKET_ONLY"


def _comparison_inputs(probability_a=0.6, probability_b=0.4):
    opportunities = pd.DataFrame(
        {
            "row_id": ["a1", "a2", "b1"],
            "date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-02"], utc=True),
        }
    )
    # February deliberately has opportunities but neither policy bets there.
    opportunities = pd.concat(
        [opportunities, pd.DataFrame({"row_id": ["c1"], "date": pd.to_datetime(["2024-02-10"], utc=True)})],
        ignore_index=True,
    )
    rows = []
    for policy, probability, selected in (
        ("A", probability_a, [("a1", 2.0, 100.0), ("b1", 3.0, 100.0)]),
        ("B", probability_b, [("a2", 2.5, 100.0), ("b1", 3.0, 100.0)]),
    ):
        for row_id, odds, stake in selected:
            rows.append(
                {
                    "policy": policy,
                    "row_id": row_id,
                    "date": opportunities.set_index("row_id").loc[row_id, "date"],
                    "side": "A",
                    "odds": odds,
                    "stake": stake,
                    "p": probability,
                    "won": row_id == "b1",
                    "pnl": stake * (odds - 1.0) if row_id == "b1" else -stake,
                    "tax_rate": 0.0,
                    "ev_error": 999.0,
                }
            )
    return opportunities, pd.DataFrame(rows)


def test_paired_monthly_comparison_uses_full_shared_opportunity_month_universe():
    opportunities, bets = _comparison_inputs()
    result = paired_monthly_comparison(
        opportunities,
        bets[bets.policy == "A"],
        bets[bets.policy == "B"],
        draws=200,
        seed=20260928,
    )

    assert result["months"] == ["2024-01", "2024-02", "2024-03"]
    assert result["paired_months"] == result["months"]
    assert result["policy_a"]["bets"] == 2
    assert result["policy_b"]["bets"] == 2
    assert result["policy_a"]["monthly"]["2024-02"]["stake"] == 0
    assert result["policy_b"]["monthly"]["2024-02"]["stake"] == 0
    assert "p_value" not in result
    assert "sign_fraction" not in result


def test_paired_monthly_comparison_marks_empty_replicates_and_ci_unavailable():
    opportunities, bets = _comparison_inputs()
    # All observed stake sits in one month; resampling the full three-month
    # opportunity universe yields zero-stake replicas and therefore no CI.
    one_month = bets[bets.date.dt.month.eq(1)].copy()
    result = paired_monthly_comparison(
        opportunities,
        one_month[one_month.policy == "A"],
        one_month[one_month.policy == "B"],
        draws=200,
        seed=7,
    )

    assert result["bootstrap"]["draws"] == 200
    assert result["bootstrap"]["empty_replicates"] > 0
    assert result["bootstrap"]["ci_lower"] is None
    assert result["bootstrap"]["ci_upper"] is None
    assert result["bootstrap"]["status"] == "UNDEFINED_REPLICATES"


def test_fixed_selection_prices_and_stakes_keep_pnl_when_probabilities_change():
    opportunities, bets = _comparison_inputs(probability_a=0.7, probability_b=0.3)
    changed_probabilities = bets.copy()
    changed_probabilities["p"] = np.where(changed_probabilities.policy.eq("A"), 0.2, 0.9)
    first_a = paired_monthly_comparison(
        opportunities, bets[bets.policy == "A"], bets[bets.policy == "B"], draws=200, seed=19
    )
    second_a = paired_monthly_comparison(
        opportunities,
        changed_probabilities[changed_probabilities.policy == "A"],
        changed_probabilities[changed_probabilities.policy == "B"],
        draws=200,
        seed=19,
    )

    assert first_a["policy_a"]["pnl"] == second_a["policy_a"]["pnl"]
    assert first_a["policy_a"]["stake"] == second_a["policy_a"]["stake"]

    assert first_a["policy_a"]["ev_error"] != second_a["policy_a"]["ev_error"]


def test_paired_monthly_comparison_recomputes_taxed_ev_error_from_primary_fields():
    date = pd.Timestamp("2024-01-05", tz="UTC")
    opportunities = pd.DataFrame({"row_id": ["taxed"], "date": [date]})
    bets = pd.DataFrame([{
        "row_id": "taxed",
        "date": date,
        "odds": 2.5,
        "stake": 100.0,
        "won": True,
        "p": 0.99,
        "p_for_ev": 0.6,
        "tax_rate": 0.12,
        "unit_return": 2.5 * (1.0 - 0.12) - 1.0,
        "pnl": 17.25,
        "ev_error": -999.0,
    }])

    result = paired_monthly_comparison(
        opportunities, bets, bets.iloc[0:0], draws=10, seed=5
    )

    assert result["policy_a"]["ev_error"] == pytest.approx(0.88)
    assert result["policy_a"]["pnl"] == 17.25


def test_paired_monthly_comparison_does_not_label_probability_error_as_ev_error():
    date = pd.Timestamp("2024-01-05", tz="UTC")
    opportunities = pd.DataFrame({"row_id": ["no-ev-inputs"], "date": [date]})
    bets = pd.DataFrame([{
        "row_id": "no-ev-inputs",
        "date": date,
        "stake": 100.0,
        "won": True,
        "p": 0.8,
        "pnl": 100.0,
        "ev_error": 123.0,
    }])

    result = paired_monthly_comparison(
        opportunities, bets, bets.iloc[0:0], draws=10, seed=5
    )

    assert result["policy_a"]["ev_error"] is None


@pytest.mark.parametrize("research_policy,spread", [("P1", 0.04), ("P3", 0.0)])
def test_comparison_uses_research_score_without_changing_fixed_stakes_or_certifying_it(
    research_policy, spread,
):
    opportunities = pd.DataFrame([{
        "row_id": "research-score",
        "date": "2024-01-05",
        "decision_at": "2024-01-05T10:00:00Z",
        "label_available_at": "2024-01-06T12:00:00Z",
        "p": 0.60,
        "p_std": spread,
        "odds_a": 2.2,
        "odds_b": 1.4,
        "y": 1,
    }])
    ledgers = []
    for policy in ("P0", research_policy):
        selected = select_policy_bets(
            opportunities, policy=policy, tax_rate=0.12, min_ev=0.05,
            qualification="raw",
        )
        ledger, _ = simulate_portfolio(
            selected, strategy="flat", initial_bankroll=100.0,
            flat_stake=10.0, tax_rate=0.12,
        )
        ledgers.append(ledger)

    mean_ledger, research_ledger = ledgers
    result = paired_monthly_comparison(
        opportunities, mean_ledger, research_ledger, draws=10, seed=5,
    )
    research_score = (
        1.0 / (1.0 + np.exp(-(np.log(0.60 / 0.40) - 0.75 * 0.15)))
        if research_policy == "P1" else 0.60
    )

    assert mean_ledger.loc[0, "prob_conservative"] is None
    assert mean_ledger.loc[0, "uncertainty_basis"] == "mean"
    assert research_ledger.loc[0, "prob_conservative"] == pytest.approx(research_score)
    assert research_ledger.loc[0, "uncertainty_basis"] == "assumed_research_haircut"
    assert result["policy_a"]["stake"] == result["policy_b"]["stake"] == 10.0
    assert result["policy_a"]["pnl"] == result["policy_b"]["pnl"] == 9.36
    assert result["policy_a"]["ev_error"] == pytest.approx((1.0 - 0.60) * 2.2 * 0.88)
    assert result["policy_b"]["ev_error"] == pytest.approx((1.0 - research_score) * 2.2 * 0.88)
